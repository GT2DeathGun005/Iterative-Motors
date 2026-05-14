"""
Behavioral Cloning (Imitation Learning) — TORCS Giro Secco

Addestra una PolicyNetwork sulle dimostrazioni umane (HDF5) per il warm start del SAC.

Features:
  - Supporto multi-file: accetta sia un singolo .h5 sia una directory di lap_*.h5 (solo giri completi)
  - Normalizzazione corretta delle azioni per Tanh output [-1, 1]
  - Sanity check preventivi (NaN, Inf, gruppi mancanti)
  - Device CPU/CUDA coerente in tutta la pipeline
  - Early stopping con validation split

Mapping delle azioni:
  [0] steering  [-1, 1]  → diretto (già in range Tanh)
  [1] accel     [0, 1]   → scalato a [-1, 1] con x*2-1
  [2] brake     [0, 1]   → scalato a [-1, 1] con x*2-1
  [3] gear      [0, 6]   → scalato a [-1, 1] con (x/3)-1

Inversione (per inferenza):
  steering = output[0]
  accel    = (output[1] + 1) / 2
  brake    = (output[2] + 1) / 2
  gear     = round((output[3] + 1) * 3)   → clamp [0, 6]
"""

import os
import glob
import argparse
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split


# ──────────────────────────────────────────────────────────────────────
#  Dataset HDF5
# ──────────────────────────────────────────────────────────────────────

class TorcsHDF5Dataset(Dataset):
    """Dataset da un singolo file HDF5 con gruppi 'states' e 'actions'.

    Esegue sanity check all'inizializzazione:
      - Verifica presenza dei gruppi richiesti
      - Verifica assenza di NaN e Inf
      - Clamp del gear a [0, 6] (esclude retromarcia)
    """

    def __init__(self, file_path: str):
        super().__init__()
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File dataset non trovato: {file_path}")

        self.file_path = file_path

        with h5py.File(self.file_path, 'r') as h5f:
            # ── Verifica gruppi ──
            if 'states' not in h5f:
                raise KeyError(f"Gruppo 'states' mancante in {file_path}")
            if 'actions' not in h5f:
                raise KeyError(f"Gruppo 'actions' mancante in {file_path}")

            states_np = h5f['states'][:]
            actions_np = h5f['actions'][:]

            # ── Sanity check numerici ──
            if np.any(np.isnan(states_np)):
                raise ValueError(f"NaN rilevati in 'states' di {file_path}")
            if np.any(np.isinf(states_np)):
                raise ValueError(f"Inf rilevati in 'states' di {file_path}")
            if np.any(np.isnan(actions_np)):
                raise ValueError(f"NaN rilevati in 'actions' di {file_path}")
            if np.any(np.isinf(actions_np)):
                raise ValueError(f"Inf rilevati in 'actions' di {file_path}")

            # ── Clamp gear a [0, 6] (ignora retromarcia -1) ──
            actions_np[:, 3] = np.clip(actions_np[:, 3], 0.0, 6.0)

            self.states = torch.tensor(states_np, dtype=torch.float32)
            self.actions = torch.tensor(actions_np, dtype=torch.float32)

        self.length = self.states.shape[0]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        return self.states[idx], self.actions[idx]


def load_dataset(path: str) -> Dataset:
    """Carica un dataset da un file .h5 o da una directory di file lap_*.h5.

    Se `path` è una directory, concatena tutti i file lap_*.h5 trovati.
    Se `path` è un singolo file, lo carica direttamente.
    """
    if os.path.isdir(path):
        h5_files = sorted(glob.glob(os.path.join(path, "**/lap_*.h5"), recursive=True))
        
        # Filtriamo gli snippet delle curve (sia ideali che diverse).
        # Il BC usa ESCLUSIVAMENTE i giri completi (che contengono già le curve).
        # Gli snippet separati servono solo per la stratificazione del Replay Buffer in sac_rl.py.
        h5_files = [f for f in h5_files if "lap_curve_" not in os.path.basename(f)]
        
        if not h5_files:
            raise FileNotFoundError(
                f"Nessun file lap_*.h5 trovato in {path} o nelle sue sottocartelle"
            )
        print(f"  Trovati {len(h5_files)} file HDF5:")
        datasets = []
        total_samples = 0
        for f in h5_files:
            ds = TorcsHDF5Dataset(f)
            datasets.append(ds)
            total_samples += len(ds)
            print(f"    ✓ {os.path.relpath(f, path)}: {len(ds)} campioni")
        print(f"  Totale: {total_samples} campioni")
        return ConcatDataset(datasets), total_samples
    else:
        ds = TorcsHDF5Dataset(path)
        print(f"  Caricato {os.path.basename(path)}: {len(ds)} campioni")
        return ds, len(ds)


# ──────────────────────────────────────────────────────────────────────
#  Policy Network
# ──────────────────────────────────────────────────────────────────────

class PolicyNetwork(nn.Module):
    """Rete Actor per Behavioral Cloning: stato → azione continua.

    Architettura feed-forward con LayerNorm e output Tanh [-1, 1].
    Struttura del Sequential (per riferimento nel warm start SAC):
      net.0: Linear(state_dim → hidden)
      net.1: LayerNorm(hidden)
      net.2: ReLU
      net.3: Linear(hidden → hidden)
      net.4: LayerNorm(hidden)
      net.5: ReLU
      net.6: Linear(hidden → action_dim)
      net.7: Tanh
    """

    def __init__(self, state_dim: int = 30, action_dim: int = 4,
                 hidden_size: int = 256):
        super(PolicyNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),       # 0
            nn.LayerNorm(hidden_size),                # 1
            nn.ReLU(),                                # 2
            nn.Linear(hidden_size, hidden_size),      # 3
            nn.LayerNorm(hidden_size),                # 4
            nn.ReLU(),                                # 5
            nn.Linear(hidden_size, action_dim),       # 6
            nn.Tanh()                                 # 7
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


# ──────────────────────────────────────────────────────────────────────
#  Normalizzazione azioni (target mapping per Tanh)
# ──────────────────────────────────────────────────────────────────────

def normalize_actions(actions: torch.Tensor) -> torch.Tensor:
    """Normalizza il tensore azioni dal range naturale al range Tanh [-1, 1].

    Input ranges:
      [0] steering: [-1, 1]  → invariato
      [1] accel:    [0, 1]   → [-1, 1]  con x*2-1
      [2] brake:    [0, 1]   → [-1, 1]  con x*2-1
      [3] gear:     [0, 6]   → [-1, 1]  con (x/3)-1

    Gear mapping: 0→-1.0, 1→-0.667, 2→-0.333, 3→0.0, 4→0.333, 5→0.667, 6→1.0
    Tutti i valori sono raggiungibili da Tanh.
    """
    norm = actions.clone()
    norm[:, 1] = actions[:, 1] * 2.0 - 1.0    # accel [0,1] → [-1,1]
    norm[:, 2] = actions[:, 2] * 2.0 - 1.0    # brake [0,1] → [-1,1]
    norm[:, 3] = actions[:, 3] / 3.0 - 1.0    # gear  [0,6] → [-1,1]
    return norm


# ──────────────────────────────────────────────────────────────────────
#  Trainer
# ──────────────────────────────────────────────────────────────────────

class BehaviorCloningTrainer:
    """Addestra la PolicyNetwork con Steering-Weighted MSE loss.

    Features:
      - Loss pesata: lo sterzo in curva (|steer| > 0.1) pesa 5x di più
        per contrastare lo sbilanciamento dei dati (64.5% rettilinei)
      - Validation split 80/20 con seed fisso per riproducibilità
      - Early stopping basato sulla val loss
      - Salvataggio automatico del miglior checkpoint

    Nota sulla validation split:
      La split 80/20 funge da regolarizzazione implicita: il modello si
      ferma quando inizia a memorizzare il rumore nei dati anziché i pattern
      di guida. Senza di essa (training su 100%) il modello raggiunge
      train loss molto basse (0.021) ma overffitta, degradando la performance
      in ambiente reale (reward media: +20 vs +296 con early stopping).
      Con ~71k campioni mescolati da 20 giri, la probabilità di perdere tutti
      i campioni di una curva specifica è trascurabile.
    """

    # Peso extra per lo sterzo in curva. Senza questo, la MSE media converge
    # verso steer≈0 perché il 64.5% dei campioni è in rettilineo, causando
    # sotto-sterzo catastrofico che porta fuori pista alla prima curva.
    STEER_CURVE_WEIGHT = 5.0
    STEER_CURVE_THRESHOLD = 0.1  # |steer_normalized| sopra questa soglia

    def __init__(self, model: nn.Module, dataset: Dataset,
                 batch_size: int = 128, val_split: float = 0.2,
                 lr: float = 3e-4, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        print(f"  Modello spostato su: {self.device}")

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=1e-5
        )

        # ── Validation split ──
        total = len(dataset)
        val_size = int(total * val_split)
        train_size = total - val_size

        self.train_dataset, self.val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=2, pin_memory=(device != "cpu")
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=batch_size,
            shuffle=False, num_workers=2, pin_memory=(device != "cpu")
        )

        self.best_val_loss = float('inf')
        self.patience_counter = 0

        print(f"  Split: {train_size} train / {val_size} val")

    def _weighted_mse(self, predictions, targets_norm, states):
        """MSE con peso extra sullo sterzo in curva e sulla partenza da fermo.

        Per i campioni dove |steer_target| > threshold, il peso dello sterzo
        è STEER_CURVE_WEIGHT (5x). Per tutti gli altri campioni e dimensioni
        il peso è 1.0 (MSE standard). Per la partenza a bassa velocità, applichiamo
        un peso extra di 10x per forzare l'apprendimento di marcia 1 e gas.
        """
        # Errore quadratico per-dimensione [batch, 4]
        sq_error = (predictions - targets_norm) ** 2

        # Peso per-campione sullo sterzo (dim 0)
        steer_target = targets_norm[:, 0].abs()
        is_curve = (steer_target > self.STEER_CURVE_THRESHOLD).float()
        # Peso: 1.0 in rettilineo, STEER_CURVE_WEIGHT in curva
        steer_weight = 1.0 + (self.STEER_CURVE_WEIGHT - 1.0) * is_curve

        # Applica il peso SOLO allo sterzo
        weighted_sq = sq_error.clone()
        weighted_sq[:, 0] = sq_error[:, 0] * steer_weight

        # Peso extra per il cambio (indice 3) per imparare meglio le marce
        weighted_sq[:, 3] = sq_error[:, 3] * 5.0

        # Peso extra per bassa velocità (speedX è all'indice 21 dello stato)
        # speedX < 0.8 corrisponde a < 40 km/h
        speed_x = states[:, 21].abs()
        is_low_speed = (speed_x < 0.8).float()
        # Moltiplicatore 10x per i campioni a bassa velocità
        speed_weight = 1.0 + 9.0 * is_low_speed

        # Applica il moltiplicatore a tutto il campione (tutte e 4 le dimensioni)
        weighted_sq = weighted_sq * speed_weight.unsqueeze(1)

        return weighted_sq.mean()

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0

        for states, targets in self.train_loader:
            states = states.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # Normalizza i target nel range [-1, 1] per il Tanh
            targets_norm = normalize_actions(targets)

            self.optimizer.zero_grad()
            predictions = self.model(states)
            loss = self._weighted_mse(predictions, targets_norm, states)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def validate(self) -> float:
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for states, targets in self.val_loader:
                states = states.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                targets_norm = normalize_actions(targets)
                predictions = self.model(states)
                loss = self._weighted_mse(predictions, targets_norm, states)
                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def train(self, max_epochs: int = 200, patience: int = 15,
              checkpoint_path: str = "train_set/checkpoints/bc_policy.pth"):
        print(f"\n  Inizio training Behavioral Cloning su {self.device}...")
        print(f"  Max epochs: {max_epochs} | Patience: {patience}\n")

        for epoch in range(max_epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()

            improved = ""
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                torch.save(self.model.state_dict(), checkpoint_path)
                improved = " ★ saved"
            else:
                self.patience_counter += 1

            print(
                f"  Epoch {epoch+1:03d}/{max_epochs} | "
                f"Train MSE: {train_loss:.6f} | "
                f"Val MSE: {val_loss:.6f}{improved}"
            )

            if self.patience_counter >= patience:
                print(
                    f"\n  Early stopping all'epoca {epoch+1}. "
                    f"Miglior Val Loss: {self.best_val_loss:.6f}"
                )
                break

        print(f"\n  Training completato. Miglior checkpoint: {checkpoint_path}")


# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Behavioral Cloning per agente TORCS (Giro Secco)"
    )
    parser.add_argument(
        "--dataset", type=str, default="train_set/laps",
        help="Path al dataset HDF5 (file singolo o directory di lap_*.h5)"
    )
    parser.add_argument("--epochs", type=int, default=200, help="Max epoche")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument(
        "--output", type=str, default="train_set/checkpoints/bc_policy.pth",
        help="Path di output per i pesi del modello"
    )
    args = parser.parse_args()

    # ── Device ──
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'=' * 64}")
    print(f"  🧠 BEHAVIORAL CLONING — TORCS Giro Secco")
    print(f"  Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"{'=' * 64}\n")

    # ── Caricamento dataset ──
    print("  Caricamento dataset...")
    dataset, total_samples = load_dataset(args.dataset)

    # ── Rileva dimensioni ──
    # Accedi al primo campione per ottenere le dimensioni
    sample_state, sample_action = dataset[0]
    state_dim = sample_state.shape[0]
    action_dim = sample_action.shape[0]
    print(f"  Dimensioni: state={state_dim}, action={action_dim}")

    # ── Assicurati che la directory di output esista ──
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # ── Modello ──
    model = PolicyNetwork(state_dim=state_dim, action_dim=action_dim)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parametri totali: {total_params:,}")

    # ── Trainer ──
    trainer = BehaviorCloningTrainer(
        model=model,
        dataset=dataset,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device
    )

    trainer.train(max_epochs=args.epochs, checkpoint_path=args.output)

    print("\n  ✅ Addestramento Behavioral Cloning completato.")
    print(f"  Pesi salvati in: {args.output}\n")


if __name__ == "__main__":
    main()
