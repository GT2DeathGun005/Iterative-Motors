"""
Behavioral Cloning (Imitation Learning) — TORCS Giro Secco

Addestra una PolicyNetwork Multi-Head sulle dimostrazioni umane (HDF5).

Features:
  - Supporto multi-file: accetta sia un singolo .h5 sia una directory di lap_*.h5 (solo giri completi)
  - Device CPU/CUDA coerente in tutta la pipeline
  - Validation split (80/20) con Early Stopping per evitare overfitting
  - Cosine LR scheduler per convergenza dolce
  - Bojarski-style data augmentation per anti-covariate shift

NOTA: I dati HDF5 sono GIÀ normalizzati dal data_collection.flatten_state():
  - track[19]: /200 (via gym_torcs.make_observaton)
  - speedX/Y/Z: /50 (via gym_torcs.make_observaton, default_speed=50)
  - wheelSpinVel[4]: /100 (via data_collection.flatten_state)
  - rpm: /10000 (via data_collection.flatten_state)
  - distFromStart: /4000 (via data_collection.flatten_state)
  NON ri-normalizzare in TorcsHDF5Dataset!

Mapping delle azioni (diretto, senza ri-mappatura):
  [0] steering  [-1, 1]  → Tanh output
  [1] accel     [0, 1]   → Sigmoid output
  [2] brake     [0, 1]   → Sigmoid output
  [3] gear      {0..6}   → CrossEntropy (7 classi)
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
import math


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

            # ── NOTA: NESSUNA ri-normalizzazione ──
            # I dati in HDF5 sono GIA' normalizzati da data_collection.flatten_state():
            #   track[1:20] → /200 (via gym_torcs.make_observaton)
            #   speedX/Y/Z[21:24] → /50 (via gym_torcs.make_observaton)
            #   wheelSpinVel[24:28] → /100, rpm[28] → /10000, dist[29] → /4000

            # ── Sanity check numerici ──
            if np.any(np.isnan(states_np)):
                raise ValueError(f"NaN rilevati in 'states' di {file_path}")
            if np.any(np.isinf(states_np)):
                raise ValueError(f"Inf rilevati in 'states' di {file_path}")
            if np.any(np.isnan(actions_np)):
                raise ValueError(f"NaN rilevati in 'actions' di {file_path}")
            if np.any(np.isinf(actions_np)):
                raise ValueError(f"Inf rilevati in 'actions' di {file_path}")

            # ── Clamp gear a [0, 6] (esclude retromarcia -1) ──
            actions_np[:, 3] = np.clip(actions_np[:, 3], 0.0, 6.0)

            self.states = torch.tensor(states_np, dtype=torch.float32)
            self.actions = torch.tensor(actions_np, dtype=torch.float32)

        self.length = self.states.shape[0]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        return self.states[idx], self.actions[idx]


def load_dataset(path: str) -> Dataset:
    """Carica e aggrega l'intero manifold di giri per migliorare la robustezza."""
    if os.path.isdir(path):
        h5_files = sorted(glob.glob(os.path.join(path, "**/lap_*.h5"), recursive=True))
        
        if not h5_files:
            raise FileNotFoundError(
                f"Nessun file lap_*.h5 trovato in {path} o nelle sue sottocartelle"
            )
        print(f"  Trovati {len(h5_files)} file HDF5. Carico l'intero dataset...")
        
        datasets = []
        total_samples = 0
        
        for f in h5_files:
            try:
                ds = TorcsHDF5Dataset(f)
                datasets.append(ds)
                total_samples += len(ds)
            except Exception as e:
                print(f"  [Warning] Impossibile leggere {f}: {e}")
                
        if not datasets:
            raise ValueError("Nessun dataset valido trovato.")
        print(f"  📚 Dataset caricato: {len(datasets)} giri, {total_samples} campioni totali.")
        return ConcatDataset(datasets), total_samples
    else:
        ds = TorcsHDF5Dataset(path)
        print(f"  Caricato {os.path.basename(path)}: {len(ds)} campioni")
        return ds, len(ds)
class PolicyNetwork(nn.Module):
    """Rete Actor per Behavioral Cloning con architettura Multi-Head:
    stato (30D) → testa continua (steer, accel, brake) & testa discreta (gear).

    Il backbone estrae feature condivise. Le due teste separate evitano
    le oscillazioni e i ritardi tipici della regressione sul cambio marcia.
    """

    def __init__(self, state_dim: int = 30, hidden_size: int = 512):
        super(PolicyNetwork, self).__init__()

        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )

        # Testa continua per: steer (1), accel (1), brake (1)
        self.continuous_head = nn.Linear(hidden_size, 3)
        
        # Testa discreta per la marcia (7 classi: 0, 1, 2, 3, 4, 5, 6)
        self.gear_head = nn.Linear(hidden_size, 7)

    def forward(self, state: torch.Tensor):
        features = self.backbone(state)
        
        cont_out = self.continuous_head(features)
        
        # Separiamo e applichiamo le attivazioni corrette
        steer = torch.tanh(cont_out[:, 0:1])          # [-1, 1]
        accel_brake = torch.sigmoid(cont_out[:, 1:3])   # [0, 1]
        
        continuous = torch.cat([steer, accel_brake], dim=1) # 3D: [steer, accel, brake]
        
        gear_logits = self.gear_head(features)          # 7D logits
        
        return continuous, gear_logits


# ──────────────────────────────────────────────────────────────────────
#  Trainer
# ──────────────────────────────────────────────────────────────────────

class BehaviorCloningTrainer:
    """Addestra la PolicyNetwork con Loss combinata MSE + CrossEntropy.

    Features:
      - Loss continua pesata per sterzo, acceleratore e freno (5x freno)
      - Classificazione discreta con CrossEntropy per la marcia (gear)
      - Validation split 80/20 con Early Stopping (patience=30)
      - Cosine Annealing LR scheduler
      - Data augmentation con rumore gaussiano strutturato sugli stati
      - Salvataggio del modello basato sulla migliore validation loss
    """

    # Limiti di soglia
    STEER_CURVE_THRESHOLD = 0.10  # soglia sterzo per curva (nel range [-1,1])

    def __init__(self, model: nn.Module, dataset: Dataset,
                 batch_size: int = 128, lr: float = 3e-4, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        print(f"  Modello spostato su: {self.device}")

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=1e-5
        )

        # ── Validation split 80/20 per Early Stopping ──
        total = len(dataset)
        val_size = max(1, int(total * 0.2))
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

        print(f"  Dataset split: {train_size} train / {val_size} val")

    def _combined_loss(self, pred_continuous, pred_gear_logits, target_actions):
        # target_actions ha dimensione: [batch_size, 4]
        # [0] steer, [1] accel, [2] brake, [3] gear (float)
        
        # 1. Loss Continua (Weighted MSE)
        targets_cont = target_actions[:, 0:3]
        sq_error = (pred_continuous - targets_cont) ** 2
        
        # Pesi per canale continuo: [steer, accel, brake]
        channel_weights = torch.tensor([1.0, 1.0, 5.0], device=pred_continuous.device)
        
        # Boost freno dinamico: se l'umano frena (target > 0.05), aumentiamo il peso del freno di 25x!
        brake_target = targets_cont[:, 2]
        brake_boost = 1.0 + 24.0 * (brake_target > 0.05).float()
        
        # Boost sterzo in curva (3x)
        steer_target = targets_cont[:, 0].abs()
        is_curve = (steer_target > self.STEER_CURVE_THRESHOLD).float()
        steer_boost = 1.0 + 2.0 * is_curve  # 1x rettilineo, 3x curva
        
        weighted_sq = sq_error * channel_weights.unsqueeze(0)
        weighted_sq[:, 0] = weighted_sq[:, 0] * steer_boost
        weighted_sq[:, 2] = weighted_sq[:, 2] * brake_boost
        loss_cont = weighted_sq.mean()
        
        # 2. Loss Discreta (CrossEntropy per il Gear)
        # Il target della marcia deve essere di tipo Long per CrossEntropy
        targets_gear = target_actions[:, 3].long()
        loss_gear = nn.functional.cross_entropy(pred_gear_logits, targets_gear)
        
        # Combinazione bilanciata: la CrossEntropy ha un peso di 2.0 per allinearsi alla scala del MSE
        total_loss = loss_cont + 2.0 * loss_gear
        return total_loss

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0

        for states, targets in self.train_loader:
            states = states.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # ── Data Augmentation: Bojarski-Style Synthetic Recovery (NVIDIA Autopilot) ──
            # Insegna al modello come recuperare quando si sposta lateralmente (covariate shift).
            # Perturbiamo coerentemente trackPos, track sensors e correggiamo il target di sterzo.
            # Riduciamo leggermente anche il target di accel se fuori asse per insegnare a parzializzare.
            
            # Genera offset laterale in trackPos (più leggero: ±0.15)
            delta_pos = torch.randn(states.size(0), device=states.device) * 0.08
            delta_pos = torch.clamp(delta_pos, -0.15, 0.15)
            
            # Estrarre angle (indice 0) e sensori di pista grezzi
            angle = states[:, 0]
            L_0 = states[:, 1] * 200.0   # Sensore -45 gradi
            L_18 = states[:, 19] * 200.0  # Sensore 45 gradi
            
            # Calcolo geometrico dinamico della semi-larghezza della pista
            W_L = L_18 * torch.sin(angle + 0.785398) # 45 gradi = 0.785398 rad
            W_R = L_0 * torch.sin(0.785398 - angle)
            W_half = torch.clamp((W_L + W_R) / 2.0, 4.0, 10.0) # clamping tra 4m e 10m
            
            # Spostamento laterale fisico in metri (scalato del 50% per correzione più leggera)
            dy = delta_pos * W_half * 0.5
            
            # 1. Perturbazione trackPos (indice 20)
            states[:, 20] = states[:, 20] + delta_pos
            
            # 2. Perturbazione geometricamente coerente dei 19 sensori track (indici 1:20)
            alpha = torch.tensor([
                -45.0, -19.0, -12.0, -7.0, -4.0, -2.5, -1.7, -1.0, -0.5, 0.0, 
                0.5, 1.0, 1.7, 2.5, 4.0, 7.0, 12.0, 19.0, 45.0
            ], device=states.device) * 3.14159265 / 180.0
            
            # Angolo assoluto di ciascun raggio rispetto alla linea mediana
            beta = angle.unsqueeze(1) + alpha.unsqueeze(0)
            
            # Perturbazione lineare sui 19 raggi
            dL = - dy.unsqueeze(1) * torch.sin(beta)
            states[:, 1:20] = torch.clamp(states[:, 1:20] + dL / 200.0, 0.0, 1.0)
            
            # 3. Correzione proporzionale target steer (indice 0) (più leggera: 0.12)
            targets[:, 0] = targets[:, 0] - 0.12 * delta_pos
            targets[:, 0] = torch.clamp(targets[:, 0], -1.0, 1.0)
            
            # 4. Correzione parzializzazione throttle (indice 1) (più leggera: 15%)
            targets[:, 1] = targets[:, 1] * (1.0 - 0.15 * delta_pos.abs())
            targets[:, 1] = torch.clamp(targets[:, 1], 0.0, 1.0)

            self.optimizer.zero_grad()
            pred_cont, pred_gear = self.model(states)
            loss = self._combined_loss(pred_cont, pred_gear, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        total_loss = 0.0

        for states, targets in self.val_loader:
            states = states.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            pred_cont, pred_gear = self.model(states)
            loss = self._combined_loss(pred_cont, pred_gear, targets)
            total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def train(self, max_epochs: int = 200,
              checkpoint_path: str = "train_set/checkpoints/bc_policy.pth",
              patience: int = 100):
        print(f"\n  Inizio training Behavioral Cloning su {self.device}...")
        print(f"  Max epochs: {max_epochs} | Early Stopping patience: {patience}\n")

        # Cosine Annealing LR
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max_epochs, eta_min=1e-6
        )

        patience_counter = 0

        for epoch in range(max_epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            lr = self.optimizer.param_groups[0]['lr']
            scheduler.step()

            improved = ""
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), checkpoint_path)
                improved = " ★ saved"
                patience_counter = 0
            else:
                patience_counter += 1

            print(
                f"  Epoch {epoch+1:03d}/{max_epochs} | "
                f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                f"LR: {lr:.2e}{improved}"
            )

            if patience_counter >= patience:
                print(f"\n  ⏹ Early Stopping: nessun miglioramento per {patience} epoche.")
                break

        print(f"\n  Training completato. Best val loss: {self.best_val_loss:.6f}")
        print(f"  Miglior checkpoint: {checkpoint_path}")


# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Behavioral Cloning per agente TORCS (Giro Secco) - Multi-Head"
    )
    parser.add_argument(
        "--dataset", type=str, default="train_set/laps",
        help="Path al dataset HDF5 (file singolo o directory di lap_*.h5)"
    )
    parser.add_argument("--epochs", type=int, default=300, help="Max epoche")
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
    print(f"  🧠 BEHAVIORAL CLONING — TORCS Giro Secco (Multi-Head)")
    print(f"  Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"{'=' * 64}\n")

    # ── Caricamento dataset ──
    print("  Caricamento dataset...")
    dataset, total_samples = load_dataset(args.dataset)

    # ── Rileva dimensioni ──
    sample_state, sample_action = dataset[0]
    state_dim = sample_state.shape[0]
    print(f"  Dimensioni: state={state_dim}, action_dim=4 (steer, accel, brake, gear)")

    # ── Assicurati che la directory di output esista ──
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # ── Modello ──
    model = PolicyNetwork(state_dim=state_dim)
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

    trainer.train(max_epochs=args.epochs, checkpoint_path=args.output, patience=100)

    print("\n  ✅ Addestramento Behavioral Cloning Multi-Head completato.")
    print(f"  Pesi salvati in: {args.output}\n")

if __name__ == "__main__":
    main()
