"""
Behavioral Cloning (Imitation Learning) — TORCS Hot Lap

Trains a PolicyNetwork on continuous-dimensional spaces from the human demonstrations (HDF5).

Features:
  - Multi-file support: accepts either a single .h5 or a directory of lap_[0-9]*.h5 (complete laps only)
  - CPU and GPU hardware support via pytorch for faster training on GPU systems
  - Dataset split into train and validation sets (80/20) with Early Stopping to avoid overfitting
  - The Cosine LR scheduler dynamically modifies the learning rate during training for better model convergence.
    Thanks to this feature the learning rate varies following the cosine trend; had we used other scheduler types
    the model might have converged more slowly or not at all by the end of training. This way the learning rate is
    very high at the start and gradually decreases to the minimum value set, i.e. 1e-6.

  - Bojarski-style data augmentation to try to mitigate the covariate shift between training and inference; it is set to perturb in the following ways:
      - Lateral perturbation: adds a random lateral perturbation between -0.4 and +0.4 (40% of the trackpos), simulating the car's position on track.
      - Angular perturbation: adds a random angular perturbation between -0.08 and +0.08 radians (~4.5°), simulating the car's angle relative to the track.
      This teaches the network to recover from out-of-distribution states, improving its generalization ability.
      The perturbation values were chosen empirically (changing them to more appropriate values would probably improve the BC performance),
      but the weakness of the BC was corrected by the RL with TD3+BC.

NOTE: The HDF5 data is already normalized by data_collection.flatten_state():
  - track[19]: /200 (via gym_torcs.make_observaton)
  - speedX/Y/Z: /50 (via gym_torcs.make_observaton, default_speed=50)
  - wheelSpinVel[4]: /100 (via data_collection.flatten_state)
  - rpm: /10000 (via data_collection.flatten_state)
  - distFromStart: Collected but removed from the dataset because we did not want the network to learn to correlate the action with the distance from the finish line (causing a potential train-test mismatch).


Mapping of the actions produced by the neural network and then sent to TORCS:

  Torcs Index  | Action              | Range        | Network activation function
  ----------------------------------------------------------------------------------
  [0]          | steering            | [-1, 1]      | Tanh output (perfect because its codomain is [-1, 1], like the recorded actions)
  [1]          | acceleration        | [0, 1]       | Sigmoid output (perfect because its codomain is [0, 1], like the recorded actions)
  [2]          | brake               | [0, 1]       | Sigmoid output (perfect because its codomain is [0, 1], like the recorded actions)

NOTE: the gear is not predicted by the network; it is present in the dataset as data but is ignored.
Gear shifting is delegated to the gearing.py script which selects the appropriate gear based on engine revs and speed.
"""

import os
import sys
import glob
import argparse
import numpy as np
import h5py #library used to read h5 files
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from datetime import datetime

# Iterative Motors: BC network shared by the package.
_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
from iterative_motors.models.networks import PolicyNetwork
from iterative_motors.data.hdf5_dataset import TorcsHDF5Dataset, load_dataset
from iterative_motors.bc.augmentation import AugmentConfig, augment_batch

BATCH_SIZE = 256
LR = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"




# ──────────────────────────────────────────────────────────────────────
#  Trainer
# ──────────────────────────────────────────────────────────────────────

class BehaviorCloningTrainer:
    """Trains the PolicyNetwork

    Functionality:
        - Continuous loss weighted for steering, throttle and brake:
            - A weighted mean squared error (MSE) is used for each action.
            - The weights are assigned to give more importance to critical and less frequent actions:
                - Steering: base weight 1.0 (with a 3x boost in corners to improve the trajectory). A corner is detected if the steering rotation exceeds STEER_CURVE_THRESHOLD, now set to 0.10 (10% of the maximum rotation).
                    This hyperparameter will probably need to be increased to improve the BC and adapted to the circuit.
                - Throttle: base weight 1.0.
                - Brake: base weight 5.0 (with a dynamic 25x boost when the driver brakes, to force the network to learn the braking points).

      - 80/20 validation split with configurable Early Stopping
            - The dataset is split into two parts: 80% for training and 20% for validation
            - Early stopping prevents overfitting by stopping training when the validation loss stops improving (avoids wasting compute resources)

      - Cosine Annealing LR scheduler
        - The Learning Rate (LR) is gradually reduced during training following a cosine curve

      - Bojarski-style data augmentation with structured Gaussian noise on the states
        - Gaussian noise is added to the states to increase the model's robustness

      - Saving the model weights based on the best validation loss
        - During the various training epochs, the neural network is saved only if its validation-loss score is better than the previous ones.
    """

    # Revised loss weights (Iterative Motors): the old brake (base 5 × boost 25 = up to
    # 125× the steering) made the loss almost a single braking regressor, worsening the
    # steering precision. Reduced to a ~24× peak (base 3 × boost 8); more emphasis in corners.
    STEER_CURVE_THRESHOLD = 0.07  # steering threshold for a corner (in the [-1,1] range)
    STEER_BOOST_FACTOR = 4.0      # error multiplier for the steering in corners
    BRAKE_ACTIVE_THRESHOLD = 0.05  # threshold above which we consider the human to be braking
    BRAKE_BOOST_FACTOR = 8.0       # error multiplier for the brake when active

    # The global defaults are overridden by the CLI arguments passed from main().

    def __init__(self, model: nn.Module, dataset: Dataset,
                 batch_size: int = BATCH_SIZE, lr: float = LR, device: str = DEVICE,
                 state_mean=None, state_std=None, aug_cfg: AugmentConfig = None):

        ## Device check and moving the model to GPU if available
        self.device = torch.device(device)
        self.model = model.to(self.device)
        # Bojarski-style data augmentation configuration (default = AugmentConfig()).
        self.aug_cfg = aug_cfg or AugmentConfig()
        print(f"  Modello spostato su: {self.device}")


        # Preparation of the normalization parameters (mean and std); normalization will be applied after augmentation and before the forward pass to the network
        # The parameters will be saved in state_norm.npz and reused if already present
        # This approach was validated by Fujimoto & Gu 2021, who showed that normalizing each feature
        # using the global statistics (mean and standard deviation) computed in advance over the entire dataset
        # stabilizes offline-RL training and improves its performance.
        if state_mean is not None:
            self.state_mean = torch.tensor(state_mean, dtype=torch.float32, device=self.device)
            self.state_std = torch.tensor(state_std, dtype=torch.float32, device=self.device)
        else:
            self.state_mean, self.state_std = None, None


        # Adam optimizer (Adaptive Moment Estimation) with learning rate lr and weight decay 1e-5
        # Adam is a standard in neural-network training; it decides what and how much to change the neuron weights during learning.
        # weight_decay is a parameter that penalizes overly large model weights, avoiding overfitting.
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=1e-5
        )

        # Dataset split (with a fixed seed for reproducibility) into training and validation sets (80% training, 20% validation)
        total = len(dataset)
        val_size = max(1, int(total * 0.2))
        train_size = total - val_size
        self.train_dataset, self.val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        # A DataLoader is an iterator that allows scrolling through the dataset
        # shuffle=True makes the data be shuffled at each epoch (prevents the agent from learning the data in order)
        # num_workers=2 makes the data be loaded in parallel (speeds up training)
        # pin_memory if true makes the data be copied into the GPU memory (speeds up training)
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=2, pin_memory=(device != "cpu")
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=batch_size,
            shuffle=False, num_workers=2, pin_memory=(device != "cpu")
        )

        self.best_val_loss = float('inf')

        print(f"Dataset split: {train_size} train / {val_size} val")


    # Function that computes the error made by the network relative to the human driver's actions.
    # It is called combined because it combines several errors (mse for steer, accel, brake) into a single final value
    # pred_continuous: actions predicted by the neural network
    # target_actions: human driver's actions
    def _combined_loss(self, pred_continuous, target_actions):

        targets_cont = target_actions[:, 0:3]
        sq_error = (pred_continuous - targets_cont) ** 2

        # Weights per continuous channel: [steer, accel, brake] (brake base reduced 5 -> 3)
        channel_weights = torch.tensor([1.0, 1.0, 3.0], device=pred_continuous.device)

        # Dynamic brake boost if the human brakes
        brake_target = targets_cont[:, 2]
        is_braking = (brake_target > self.BRAKE_ACTIVE_THRESHOLD).float()
        brake_boost = 1.0 + (self.BRAKE_BOOST_FACTOR - 1.0) * is_braking

        # Steering boost in corners
        steer_target = targets_cont[:, 0].abs()
        is_curve = (steer_target > self.STEER_CURVE_THRESHOLD).float()
        steer_boost = 1.0 + (self.STEER_BOOST_FACTOR - 1.0) * is_curve

        weighted_sq = sq_error * channel_weights.unsqueeze(0)
        weighted_sq[:, 0] = weighted_sq[:, 0] * steer_boost
        weighted_sq[:, 2] = weighted_sq[:, 2] * brake_boost
        return weighted_sq.mean()

    # Function that defines one training epoch; at each epoch the network weights are modified to minimize the error.
    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0

        for states, targets in self.train_loader:
            states = states.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # Temporary reshape to apply the augmentation on each of the 3 frames coherently
            batch_size = states.size(0)
            states = states.view(batch_size, 3, 29)

            # Bojarski-style data augmentation (parameters configurable in AugmentConfig).
            states, targets = augment_batch(states, targets, self.aug_cfg)

            # State standardization and final flattening:
            # - Normalization (z-score) is applied at this stage since the previous augmentation
            #   must operate on the real physical quantities (meters, radians, km/h).
            # - We restore the flat dimensionality (87D) required as input by the neural network.
            if self.state_mean is not None:
                states = (states - self.state_mean) / (self.state_std + 1e-3)
            states = states.view(batch_size, 87)

            self.optimizer.zero_grad() #zeroes the gradients from the previous step
            pred_cont = self.model(states) #passes the batch to the neural network to be processed
            loss = self._combined_loss(pred_cont, targets) #computes the loss
            loss.backward() #computes the gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0) #clips the gradients to avoid gradient explosion if they exceed 1.0
            self.optimizer.step() #updates the neural-network weights

            total_loss += loss.item() #accumulates the loss to compute the final average

        return total_loss / len(self.train_loader) #returns the average loss over the batch

    #This method is called to compute the loss on the validation set.
    #It is called at the end of each epoch to evaluate the model's performance.
    @torch.no_grad()    # no need to compute gradients for the validation
    def validate(self) -> float:
        self.model.eval() #put the model in validation mode
        total_loss = 0.0

        for states, targets in self.val_loader: #iterate over the validation set
            states = states.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # Same normalization as training (no augmentation in validation)
            # Normalization and Forward Pass
            if self.state_mean is not None:
                states = states.view(states.size(0), 3, 29)
                states = (states - self.state_mean) / (self.state_std + 1e-3)
                states = states.view(states.size(0), 87)

            pred_cont = self.model(states) #computes the model's prediction
            loss = self._combined_loss(pred_cont, targets) #computes the loss
            total_loss += loss.item() #accumulates the loss to compute the final average

        return total_loss / len(self.val_loader) #returns the average loss over the batch

    # Method that coordinates the BC model training process
    def train(self, max_epochs: int = 200,
              checkpoint_path: str = "train_set/checkpoints/bc_policy.pth",
              patience: int = 50, log_path: str = None):

        # File logging to keep track of the training
        logf = open(log_path, "a", encoding="utf-8") if log_path else None

        # Logging utility function: prints the given line, adds a newline and saves it to the log file
        def _log(line: str):
            print(line)
            if logf:
                logf.write(line + "\n")
                logf.flush()

        # Training initialization
        try:
            _log(f"\n  Inizio training Behavioral Cloning su {self.device}...")
            _log(f"  Max epochs: {max_epochs} | Early Stopping patience: {patience}\n")

            # Cosine Annealing LR
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max_epochs, eta_min=1e-6
            )

            patience_counter = 0

            # Main training loop
            for epoch in range(max_epochs):
                train_loss = self.train_epoch() # Run the training epoch
                val_loss = self.validate() # Validate the result on the validation set
                lr = self.optimizer.param_groups[0]['lr'] # take the learning rate
                scheduler.step() # update the learning rate

                improved = ""
                if val_loss < self.best_val_loss: # If the validation-set loss is better than the previous best loss
                    self.best_val_loss = val_loss # Update the previous best loss
                    torch.save(self.model.state_dict(), checkpoint_path) # Save the model checkpoint
                    improved = " saved" # Append " saved" to the string to print
                    patience_counter = 0 # Reset the patience counter
                else:
                    patience_counter += 1

                _log(
                    f"  Epoch {epoch+1:03d}/{max_epochs} | "
                    f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                    f"LR: {lr:.2e}{improved}"
                )

                # Early Stopping: if the validation-set loss does not improve for 'patience' epochs, stop training
                if patience_counter >= patience:
                    _log(f"\n  Early Stopping: nessun miglioramento per {patience} epoche.")
                    break

            _log(f"\n  Training completato. Best val loss: {self.best_val_loss:.6f}")
            _log(f"  Miglior checkpoint: {checkpoint_path}")
            _log(f"  Fine: {datetime.now().isoformat()}")
        finally: # closes the log file when training is complete or in case of an error
            if logf:
                logf.close()


# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────

def main():
    # Defines the function that handles the command-line arguments and passes them to the model to regulate its behaviour
    parser = argparse.ArgumentParser(
        description="Behavioral Cloning per l'addestramento dell'agente"
    )

    # argument to change the training-data directory
    parser.add_argument(
        "--dataset", type=str, default="train_set/laps",
        help="Path al dataset HDF5 (file singolo, o directory: il BC carica SOLO i giri interi lap_[0-9]*.h5, i segmenti lap_seg_*.h5 sono esclusi)"
    )

    # argument to change the number of training epochs
    parser.add_argument("--epochs", type=int, default=300, help="Max epoche")

    # argument to change the batch size
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")

    # argument to change the learning rate
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")

    # argument to change the output path
    parser.add_argument(
        "--output", type=str, default="train_set/checkpoints/bc_policy.pth",
        help="Path di output per i pesi del modello (state_norm.npz viene salvato nella stessa cartella)"
    )
    parser.add_argument(
        "--auto_laps", type=str, default=None,
        help="Directory opzionale di giri auto-raccolti dalla TD3 (lap_*.h5) da unire al dataset "
             "umano per l'arricchimento (flywheel dati). La normalizzazione viene ricalcolata sull'unione."
    )

    args = parser.parse_args() #reads the command-line arguments
    auto_dirs = [args.auto_laps] if args.auto_laps else []

    # Check whether a GPU is available, otherwise use the CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'=' * 64}")
    print(f"  BEHAVIORAL CLONING ")
    print(f"  Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Stride Type: static (k=6, 0.24s)")
    print(f"{'=' * 64}\n")

    # Dataset loading (human + any self-recorded laps for enrichment)
    print("  Caricamento dataset...")
    dataset, total_samples = load_dataset(args.dataset, extra_dirs=auto_dirs)

    # Detect the dataset dimensions
    sample_state, _sample_action = dataset[0]
    state_dim = sample_state.shape[0]
    print(f"  Dimensioni: state={state_dim}, action_dim=4 (steer, accel, brake, gear registrata)")

    # Make sure the output directory exists
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    if os.path.isdir(args.dataset):
        _h5s = sorted(glob.glob(os.path.join(args.dataset, "**/lap_[0-9]*.h5"), recursive=True))
    else:
        _h5s = [args.dataset]
    # Also include the self-recorded laps in the normalization computation (consistency with the dataset).
    for _ad in auto_dirs:
        if _ad and os.path.isdir(_ad):
            _h5s.extend(sorted(glob.glob(os.path.join(_ad, "**/lap_*.h5"), recursive=True)))
    _all_states = []
    for _f in _h5s:
        try:
            with h5py.File(_f, 'r') as _h:
                _all_states.append(_h['states'][:])
        except Exception:
            pass
    _all_states = np.concatenate(_all_states, axis=0).astype(np.float32)  # (N, 29)
    state_mean = _all_states.mean(axis=0)
    state_std = _all_states.std(axis=0)
    _norm_path = os.path.join(os.path.dirname(args.output) or ".", "state_norm.npz")
    np.savez(_norm_path, mean=state_mean, std=state_std)
    print(f"  Normalizzazione stati salvata: {_norm_path} (mean/std su {len(_all_states)} stati 29D)")

    # Create the neural network
    model = PolicyNetwork(state_dim=state_dim)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parametri totali: {total_params:,}")

    # Create the neural-network trainer
    trainer = BehaviorCloningTrainer(
        model=model,
        dataset=dataset,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        state_mean=state_mean,
        state_std=state_std
    )

    # Start the training
    trainer.train(max_epochs=args.epochs, checkpoint_path=args.output, patience=100)

    print("\n  Addestramento Behavioral Cloning completato.")
    print(f"  Pesi salvati in: {args.output}\n")

if __name__ == "__main__":
    main()
