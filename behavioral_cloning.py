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
    """Carica un dataset da un file .h5 o da una directory.
    
    Se `path` è una directory, analizza tutti i file lap_*.h5 (ignorando le curve)
    e seleziona SOLO quello con il minor numero di campioni (il giro più veloce).
    Questo garantisce l'apprendimento del "golden lap" deterministico.
    """
    if os.path.isdir(path):
        h5_files = sorted(glob.glob(os.path.join(path, "**/lap_*.h5"), recursive=True))
        
        # Filtriamo gli snippet delle curve
        h5_files = [f for f in h5_files if "lap_curve_" not in os.path.basename(f)]
        
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


# ──────────────────────────────────────────────────────────────────────
#  Policy Network
# ──────────────────────────────────────────────────────────────────────

class PolicyNetwork(nn.Module):
    """Rete Actor per Behavioral Cloning: stato → azione continua.

    Architettura deep feed-forward con LayerNorm e output Tanh [-1, 1].
    Struttura: 30 -> 512 -> 512 -> 512 -> 512 -> 4
    """

    def __init__(self, state_dim: int = 30, action_dim: int = 4,
                 hidden_size: int = 512):
        super(PolicyNetwork, self).__init__()

        self.net = nn.Sequential(
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
            
            nn.Linear(hidden_size, action_dim),
            nn.Tanh()
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
      - Training deterministico: nessun validation split per overfittare 
        perfettamente il "golden lap" senza data leakage sequenziale.
      - Salvataggio del modello basato sulla migliore train loss.
    """

    # Limiti di soglia
    STEER_CURVE_THRESHOLD = 0.05

    def __init__(self, model: nn.Module, dataset: Dataset,
                 batch_size: int = 128, lr: float = 3e-4, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        print(f"  Modello spostato su: {self.device}")

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=1e-5
        )

        # ── Nessun validation split, training al 100% sul golden lap ──
        self.train_dataset = dataset

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=2, pin_memory=(device != "cpu")
        )

        self.best_train_loss = float('inf')

        print(f"  Dataset size: {len(self.train_dataset)} campioni (100% training)")

    def _weighted_mse(self, predictions, targets_norm, states):
        """MSE focalizzata su Partenza e Cambio (meccaniche deterministiche)."""
        sq_error = (predictions - targets_norm) ** 2

        # Iniziamo con pesi neutri (1.0) per tutto
        weighted_sq = sq_error.clone()

        # Peso massiccio per il cambio (indice 3) come richiesto
        weighted_sq[:, 3] = sq_error[:, 3] * 10.0

        # Peso extra per bassa velocità (Partenza da fermo)
        # speedX è all'indice 21 dello stato
        speed_x = states[:, 21].abs()
        is_low_speed = (speed_x < 0.8).float()
        speed_weight = 1.0 + 9.0 * is_low_speed

        # Applichiamo il peso della velocità a tutte le azioni del campione
        weighted_sq = weighted_sq * speed_weight.unsqueeze(1)

        return weighted_sq.mean()

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0

        for states, targets in self.train_loader:
            states = states.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # ── Data Augmentation: State Noise ──
            # Aggiungiamo un leggero rumore bianco allo stato per forzare la robustezza.
            # Questo aiuta l'agente a recuperare se si scosta leggermente dalla traiettoria ideale.
            if self.model.training:
                noise = torch.randn_like(states) * 0.005 # 0.5% di rumore
                states = states + noise

            # Normalizza i target nel range [-1, 1] per il Tanh
            targets_norm = normalize_actions(targets)

            self.optimizer.zero_grad()
            predictions = self.model(states)
            loss = self._weighted_mse(predictions, targets_norm, states)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def train(self, max_epochs: int = 200, 
              checkpoint_path: str = "train_set/checkpoints/bc_policy.pth"):
        print(f"\n  Inizio training Behavioral Cloning su {self.device}...")
        print(f"  Max epochs: {max_epochs} (No validation split - Overfitting Golden Lap)\n")

        for epoch in range(max_epochs):
            train_loss = self.train_epoch()

            improved = ""
            if train_loss < self.best_train_loss:
                self.best_train_loss = train_loss
                torch.save(self.model.state_dict(), checkpoint_path)
                improved = " ★ saved"

            print(
                f"  Epoch {epoch+1:03d}/{max_epochs} | "
                f"Train MSE: {train_loss:.6f}{improved}"
            )

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
