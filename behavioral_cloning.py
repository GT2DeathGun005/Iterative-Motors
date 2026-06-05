"""
Behavioral Cloning (Imitation Learning) — TORCS Giro Secco

Addestra una PolicyNetwork Multi-Head sulle dimostrazioni umane (HDF5).

Features:
  - Supporto multi-file: accetta sia un singolo .h5 sia una directory di lap_*.h5 (solo giri completi)
  - Device CPU/CUDA coerente in tutta la pipeline
  - Validation split (80/20) con Early Stopping per evitare overfitting
  - Cosine LR scheduler per convergenza dolce
  - Bojarski-style data augmentation per anti-covariate shift:
      Laterale: perturbazione trackPos ±0.4 (40% della pista)
      Angolare: perturbazione angle ±0.08 rad (~4.5°)
      Insegna alla rete il recupero da stati fuori distribuzione

NOTA: I dati HDF5 sono GIÀ normalizzati dal data_collection.flatten_state():
  - track[19]: /200 (via gym_torcs.make_observaton)
  - speedX/Y/Z: /50 (via gym_torcs.make_observaton, default_speed=50)
  - wheelSpinVel[4]: /100 (via data_collection.flatten_state)
  - rpm: /10000 (via data_collection.flatten_state)
  - distFromStart: RIMOSSA (correlazione ~0 con azioni, causa train-test mismatch)
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
from datetime import datetime


# ──────────────────────────────────────────────────────────────────────
#  Corner Emphasis (Idea #1) — oversampling pesato per zona del tracciato
# ──────────────────────────────────────────────────────────────────────
# I campioni la cui posizione (distFromStart in metri) cade in una zona
# ricevono un peso maggiore nella loss BC, per rinforzare manovre critiche.
# Bersaglio attuale: staccata + tornante stretto ~680-810m, dove l'agente
# arriva troppo veloce e esce di pista (trackPos +1.5).
# NB: la posizione è SOLO un'etichetta per pesare — NON entra nella rete (resta 29D).
# DISATTIVATO di default (lista vuota → tutti i pesi = 1.0).
# Decisione: NON applichiamo un peso artificiale ai campioni in curva. Il ripeso
# della loss ha mostrato di degradare il comportamento closed-loop (la policy
# regrediva, uscendo prima). Il bilanciamento curva/resto-pista va ottenuto in modo
# naturale, con la QUANTITÀ di dati reali raccolti sulla curva (data_collection
# --segment_only), non con un moltiplicatore. L'infrastruttura resta disponibile:
# per riattivarla basta popolare la lista con tuple (start_m, end_m, peso).
CORNER_EMPHASIS_ZONES = []  # es. [(675.0, 720.0, 2.0)] per riattivare
DIST_NORM_DIVISOR = 4012.0  # backup col[29] normalizzato: metri = col * D (track ~3619m)


def _lap_positions(file_path, states_tensor):
    """Posizione (distFromStart, metri) per ogni step del giro.

    Ordine di preferenza (tutto in scala metri raw, coerente con CORNER_EMPHASIS_ZONES):
      1. metadato `dist_from_start` salvato nel giro stesso (nuove raccolte di data_collection);
      2. backup 30D (`dataset_backup/.../<nome>`, colonna 29 normalizzata × DIST_NORM_DIVISOR);
      3. None → nessuna enfasi (peso uniforme di fallback).
    """
    n = states_tensor.shape[0]
    # 1) Metadato diretto nel file del giro (metri raw)
    try:
        with h5py.File(file_path, 'r') as h:
            if 'dist_from_start' in h:
                d = h['dist_from_start'][:].astype(np.float32)
                if d.shape[0] == n:
                    return d
    except Exception:
        pass
    # 2) Backup 30D allineato per numero di step
    base = os.path.basename(file_path)
    for c in glob.glob(os.path.join('dataset_backup', '**', base), recursive=True):
        try:
            with h5py.File(c, 'r') as h:
                bs = h['states'][:]
            if bs.shape[1] >= 30 and bs.shape[0] == n:
                return bs[:, 29].astype(np.float32) * DIST_NORM_DIVISOR
        except Exception:
            pass
    return None


# ──────────────────────────────────────────────────────────────────────
#  Dataset HDF5
# ──────────────────────────────────────────────────────────────────────

class TorcsHDF5Dataset(Dataset):
    """Dataset da un singolo file HDF5 con gruppi 'states' e 'actions'.

    Esegue lo stacking temporale statico di 3 frame con passo costante k=6 (0.24s totali):
      - t-12, t-6, t

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

            # ── Clamp gear a [0, 6] (esclude retromarcia -1) ──
            actions_np[:, 3] = np.clip(actions_np[:, 3], 0.0, 6.0)

            self.states = torch.tensor(states_np, dtype=torch.float32)
            self.actions = torch.tensor(actions_np, dtype=torch.float32)

        self.length = self.states.shape[0]

        # ── Corner Emphasis: peso per campione in base alla posizione sul tracciato ──
        w = np.ones(self.length, dtype=np.float32)
        pos = _lap_positions(file_path, self.states)
        if pos is not None:
            for (a, b, wz) in CORNER_EMPHASIS_ZONES:
                w[(pos >= a) & (pos <= b)] = wz
        self.weight = torch.tensor(w, dtype=torch.float32)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        k = 6
        idx_t6 = max(0, idx - k)
        idx_t12 = max(0, idx - 2 * k)

        stacked = torch.cat([
            self.states[idx_t12],
            self.states[idx_t6],
            self.states[idx]
        ])
        return stacked, self.actions[idx], self.weight[idx]


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
        return ds, len(ds)
class PolicyNetwork(nn.Module):
    """Rete Actor per Behavioral Cloning con architettura Multi-Head:
    stato (29D) → testa continua (steer, accel, brake) & testa discreta (gear).

    Il backbone estrae feature condivise. Le due teste separate evitano
    le oscillazioni e i ritardi tipici della regressione sul cambio marcia.

    NOTA: distFromStart è stata rimossa dal vettore di stato (30D → 29D)
    perché ha correlazione ~0 con le azioni e causa train-test mismatch.
    """

    def __init__(self, state_dim: int = 87, hidden_size: int = 512):
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

    def _combined_loss(self, pred_continuous, pred_gear_logits, target_actions, sample_weight=None):
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
        cont_ps = weighted_sq.mean(dim=1)  # loss continua per-campione [B]

        # 2. Loss Discreta (CrossEntropy per il Gear), per-campione
        # Il target della marcia deve essere di tipo Long per CrossEntropy
        targets_gear = target_actions[:, 3].long()
        gear_ps = nn.functional.cross_entropy(pred_gear_logits, targets_gear, reduction='none')  # [B]

        # Combinazione bilanciata: la CrossEntropy ha un peso di 2.0 per allinearsi alla scala del MSE
        per_sample = cont_ps + 2.0 * gear_ps

        # Corner Emphasis: media pesata per campione (con sample_weight=1 ovunque
        # coincide esattamente con la media semplice → scala/val-loss invariati).
        if sample_weight is not None:
            total_loss = (per_sample * sample_weight).sum() / sample_weight.sum().clamp(min=1e-6)
        else:
            total_loss = per_sample.mean()
        return total_loss

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0

        for states, targets, weights in self.train_loader:
            states = states.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            weights = weights.to(self.device, non_blocking=True)

            # Reshape temporaneo per applicare l'augmentation su ciascuno dei 3 frame in modo coerente
            batch_size = states.size(0)
            states = states.view(batch_size, 3, 29)

            # ── Data Augmentation: Bojarski-Style Synthetic Recovery ──
            # Simula stati fuori distribuzione (covariate shift) perturbando
            # la posizione laterale e l'angolo dell'auto, poi insegnando alla
            # rete la correzione proporzionale per rientrare in traiettoria.

            # --- Perturbazione Laterale (trackPos) ---
            # ±0.4 copre il 40% della larghezza della pista, simulando errori
            # realistici che il BC incontrerebbe a test time (prima era ±0.15,
            # troppo timido per preparare la rete al Corkscrew).
            delta_pos = torch.randn(batch_size, device=states.device) * 0.20
            delta_pos = torch.clamp(delta_pos, -0.40, 0.40)

            # --- Perturbazione Angolare (angle) ---
            # ±0.08 rad (~4.5°) simula l'auto leggermente disallineata rispetto
            # alla pista — un errore composto tipico del covariate shift.
            delta_angle = torch.randn(batch_size, device=states.device) * 0.04
            delta_angle = torch.clamp(delta_angle, -0.08, 0.08)

            # --- Gating dell'augmentation (50%) ---
            # Bojarski AGGIUNGE traiettorie di recupero, non sostituisce i dati
            # puliti: applicando la perturbazione a OGNI campione la rete non
            # vede mai lo stato ideale (delta=0) e perde fedeltà sulla linea
            # ottimale (steer wandering). Azzeriamo le perturbazioni su ~50%
            # del batch: metà impara la guida precisa, metà il recupero OOD.
            aug_mask = (torch.rand(batch_size, device=states.device) < 0.5).float()
            delta_pos = delta_pos * aug_mask
            delta_angle = delta_angle * aug_mask

            for f_idx in range(3):
                frame_states = states[:, f_idx, :]
                
                # Estrarre angle (indice 0) e sensori di pista grezzi
                angle = frame_states[:, 0]
                L_0 = frame_states[:, 1] * 200.0   # Sensore -45 gradi
                L_18 = frame_states[:, 19] * 200.0  # Sensore 45 gradi
                
                # Calcolo geometrico dinamico della semi-larghezza della pista
                W_L = L_18 * torch.sin(angle + 0.785398) # 45 gradi = 0.785398 rad
                W_R = L_0 * torch.sin(0.785398 - angle)
                W_half = torch.clamp((W_L + W_R) / 2.0, 4.0, 10.0) # clamping tra 4m e 10m
                
                # Spostamento laterale fisico in metri (scalato del 50% per correzione più leggera)
                dy = delta_pos * W_half * 0.5
                
                # 1. Perturbazione trackPos (indice 20)
                frame_states[:, 20] = frame_states[:, 20] + delta_pos
                
                # 2. Perturbazione angolare (indice 0)
                # L'angle è già in radianti — aggiungiamo la perturbazione direttamente
                frame_states[:, 0] = frame_states[:, 0] + delta_angle
                
                # 3. Perturbazione geometricamente coerente dei 19 sensori track (indici 1:20)
                alpha = torch.tensor([
                    -45.0, -19.0, -12.0, -7.0, -4.0, -2.5, -1.7, -1.0, -0.5, 0.0, 
                    0.5, 1.0, 1.7, 2.5, 4.0, 7.0, 12.0, 19.0, 45.0
                ], device=states.device) * 3.14159265 / 180.0
                
                # Angolo assoluto di ciascun raggio (usa l'angle perturbato)
                perturbed_angle = frame_states[:, 0]
                beta = perturbed_angle.unsqueeze(1) + alpha.unsqueeze(0)
                
                # Perturbazione lineare sui 19 raggi (combinata: laterale + angolare)
                dL = - dy.unsqueeze(1) * torch.sin(beta)
                frame_states[:, 1:20] = torch.clamp(frame_states[:, 1:20] + dL / 200.0, 0.0, 1.0)
                
            # 4. Correzione proporzionale target steer:
            #    - delta_pos: rientro laterale (0.25 gain — aumentato da 0.16 per match con ±0.4)
            #    - delta_angle: raddrizzamento angolare (1.5 gain per compensazione reattiva)
            targets[:, 0] = targets[:, 0] - 0.25 * delta_pos - 1.5 * delta_angle
            targets[:, 0] = torch.clamp(targets[:, 0], -1.0, 1.0)
            
            # 5. Correzione parzializzazione throttle (indice 1)
            # Quando l'auto è fuori posizione, il throttle deve calare proporzionalmente
            combined_perturbation = delta_pos.abs() + delta_angle.abs() * 5.0
            targets[:, 1] = targets[:, 1] * (1.0 - 0.15 * combined_perturbation)
            targets[:, 1] = torch.clamp(targets[:, 1], 0.0, 1.0)

            # ── Data Augmentation: Speed Perturbation Augmentation ──
            # Se la velocità attuale (speedX, indice 21) del frame più recente (frame 2) è significativa (> 90 km/h)
            # e c'è una curva (sterzo target significativo o sensore frontale decrescente),
            # aumentiamo fittiziamente la velocità in tutti e 3 i frame e aumentiamo il target del freno.
            if torch.rand(1).item() < 0.5:
                # Estraiamo speedX (indice 21) dall'ultimo frame (de-normalizzato)
                speedX_latest = states[:, 2, 21] * 50.0
                steer_target_abs = targets[:, 0].abs()
                sensor_front_latest = states[:, 2, 10]  # track_s9 (indice 10, cioè 0 gradi)

                is_speed_critical = (speedX_latest > 90.0) & ((steer_target_abs > 0.10) | (sensor_front_latest < 0.60))

                if is_speed_critical.any():
                    # Genera un incremento del 10% - 30% per i campioni critici
                    speed_factor = 0.10 + 0.20 * torch.rand(batch_size, device=states.device)
                    speed_factor = speed_factor * is_speed_critical.float()

                    # 1. Aumentiamo speedX in tutti e 3 i frame
                    for f_idx in range(3):
                        states[:, f_idx, 21] = states[:, f_idx, 21] * (1.0 + speed_factor)

                    # 2. Riduciamo l'accelerazione target
                    targets[:, 1] = targets[:, 1] * (1.0 - 0.7 * speed_factor)
                    targets[:, 1] = torch.clamp(targets[:, 1], 0.0, 1.0)

                    # 3. Aumentiamo il freno target (insegniamo a frenare correttivamente)
                    targets[:, 2] = targets[:, 2] + 0.8 * speed_factor
                    targets[:, 2] = torch.clamp(targets[:, 2], 0.0, 1.0)

            # Ri-appiattiamo in 87D prima di passarlo alla rete
            states = states.view(batch_size, 87)

            self.optimizer.zero_grad()
            pred_cont, pred_gear = self.model(states)
            loss = self._combined_loss(pred_cont, pred_gear, targets, sample_weight=weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        total_loss = 0.0

        for states, targets, _weights in self.val_loader:
            states = states.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # Val-loss uniforme (sample_weight=None) per restare comparabile tra run
            pred_cont, pred_gear = self.model(states)
            loss = self._combined_loss(pred_cont, pred_gear, targets)
            total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def train(self, max_epochs: int = 200,
              checkpoint_path: str = "train_set/checkpoints/bc_policy.pth",
              patience: int = 100, log_path: str = None):
        # Log su file (oltre alla console) per tenere traccia del training in session_logs/
        logf = open(log_path, "a", encoding="utf-8") if log_path else None

        def _log(line: str):
            print(line)
            if logf:
                logf.write(line + "\n")
                logf.flush()

        try:
            _log(f"\n  Inizio training Behavioral Cloning su {self.device}...")
            _log(f"  Max epochs: {max_epochs} | Early Stopping patience: {patience}\n")

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

                _log(
                    f"  Epoch {epoch+1:03d}/{max_epochs} | "
                    f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                    f"LR: {lr:.2e}{improved}"
                )

                if patience_counter >= patience:
                    _log(f"\n  ⏹ Early Stopping: nessun miglioramento per {patience} epoche.")
                    break

            _log(f"\n  Training completato. Best val loss: {self.best_val_loss:.6f}")
            _log(f"  Miglior checkpoint: {checkpoint_path}")
            _log(f"  Fine: {datetime.now().isoformat()}")
        finally:
            if logf:
                logf.close()


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
    print(f"  Stride Type: static (k=6, 0.24s)")
    print(f"{'=' * 64}\n")

    # ── Caricamento dataset ──
    print("  Caricamento dataset...")
    dataset, total_samples = load_dataset(args.dataset)

    # ── Rileva dimensioni ──
    sample_state, sample_action, _sample_w = dataset[0]
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

    # ── Log di sessione (timestamp) in train_set/session_logs/ ──
    # Convenzione: --output = train_set/checkpoints/X.pth → log in train_set/session_logs/.
    # Fallback robusto se l'output ha un layout diverso.
    try:
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(args.output))), "session_logs")
        os.makedirs(log_dir, exist_ok=True)
    except OSError:
        log_dir = os.path.dirname(os.path.abspath(args.output)) or "."
        os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"bc_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=== BEHAVIORAL CLONING TRAINING LOG ===\n")
        f.write(f"Avvio:        {datetime.now().isoformat()}\n")
        f.write(f"Dataset:      {args.dataset} | Campioni: {total_samples} | Device: {device}\n")
        f.write(f"Iperparam:    epochs={args.epochs} batch={args.batch_size} lr={args.lr} state_dim={state_dim}\n")
        f.write(f"CornerEmph:   {CORNER_EMPHASIS_ZONES}\n")
        f.write(f"Output:       {args.output}\n")
    print(f"  📝 Log di sessione: {log_path}")

    trainer.train(max_epochs=args.epochs, checkpoint_path=args.output, patience=100, log_path=log_path)

    print("\n  ✅ Addestramento Behavioral Cloning Multi-Head completato.")
    print(f"  Pesi salvati in: {args.output}\n")

if __name__ == "__main__":
    main()
