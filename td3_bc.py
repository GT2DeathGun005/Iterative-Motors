"""
TD3+BC Fine-Tuning — Twin Delayed DDPG con Behavioral Cloning

Architettura Ibrida BC-RL per TORCS (Offline-to-Online), allineata al TD3+BC minimalista
(Fujimoto & Gu, 2021):
  - L'Actor eredita backbone + gear_head dal BC (warm-start). Il TD3 allena TUTTO l'Actor
    (backbone + continuous_head); resta congelata solo la gear_head (marcia discreta).
  - Il Critic (Twin Q-Network) è addestrato da zero.
  - Peso Behavioral Cloning COSTANTE = 1.0 → loss = -λ·Q + (π - a)²  (λ = 2.5 / mean|Q|).
  - Normalizzazione stati mean-0/std-1 (state_norm.npz) applicata prima della rete.
  - Buffer EXPERT separato e permanente (anti-FIFO) + sampling 3-vie 25/15/60.
  - Delayed Policy Update (ogni 2 step), Target Policy Smoothing, update ratio 1:1.

Retro-compatibilità:
  - La 'log_std_head' è mantenuta nell'Actor solo per caricare vecchi checkpoint SAC in
    'test_agent.py' senza crash; è isolata (requires_grad=False) e inutilizzata nel TD3.

Reward (da corsa, minimalista):
  - progress = (speedX/50.0) * cos(angle) * 1.5 ; pos_penalty deadzone oltre |trackPos|>1.0
  - Terminali (schianto, stallo, spin, |trackPos|>1.25 = giro non valido): -10.0
  - Bonus giro VALIDO completato: +50.0
"""

import os
import sys
import argparse
import random
import re
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from datetime import datetime

# Import gym_torcs
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'gym_torcs')))
try:
    from gym_torcs import TorcsEnv
except ImportError:
    print("Warning: gym_torcs non trovato.")

from gearing import compute_gear  # cambio marcia deterministico (anti-hunting)

_PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
_CHECKPOINT_ROOT = os.path.join(_PROJECT_ROOT, 'train_set', 'checkpoints')
_CHECKPOINT_BACKUP_ROOT = os.path.join(_CHECKPOINT_ROOT, 'backups')

def _fsync_file(path):
    """Forza su disco il contenuto del file appena scritto."""
    with open(path, 'rb') as f:
        os.fsync(f.fileno())

def _fsync_dir(path):
    """Forza su disco anche il rename atomico nella directory."""
    dir_fd = os.open(path or '.', os.O_DIRECTORY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)

def _backup_paths(filepath):
    """Restituisce i path dei backup, ordinati in una cartella dedicata ai checkpoint."""
    abs_filepath = os.path.abspath(filepath)
    backup_base = None
    try:
        if os.path.commonpath([abs_filepath, _CHECKPOINT_ROOT]) == _CHECKPOINT_ROOT:
            relative_path = os.path.relpath(abs_filepath, _CHECKPOINT_ROOT)
            if relative_path != 'backups' and not relative_path.startswith('backups' + os.sep):
                backup_base = os.path.join(_CHECKPOINT_BACKUP_ROOT, relative_path)
    except ValueError:
        backup_base = None

    if backup_base is None:
        backup_base = filepath

    return backup_base + ".bak", backup_base + ".prev"

def _rotate_backup(filepath):
    """Mantiene due copie precedenti in backups/: .bak (ultima valida) e .prev (penultima valida)."""
    if not os.path.exists(filepath):
        return
    backup_path, previous_path = _backup_paths(filepath)
    backup_directory = os.path.dirname(backup_path) or '.'
    os.makedirs(backup_directory, exist_ok=True)

    if os.path.exists(backup_path):
        os.replace(backup_path, previous_path)

    temp_backup = backup_path + ".tmp"
    shutil.copy2(filepath, temp_backup)
    _fsync_file(temp_backup)
    os.replace(temp_backup, backup_path)
    _fsync_dir(backup_directory)

def _checkpoint_candidates(filepath):
    """Ordine di recupero: principale, backup ordinati, poi vecchi backup adiacenti legacy."""
    backup_path, previous_path = _backup_paths(filepath)
    candidates = [filepath, backup_path, previous_path, filepath + ".bak", filepath + ".prev"]
    unique_candidates = []
    seen = set()
    for candidate in candidates:
        key = os.path.abspath(candidate)
        if key not in seen:
            unique_candidates.append(candidate)
            seen.add(key)
    return unique_candidates

def safe_save(obj, filepath, keep_backup=True):
    """Salva in modo atomico e conserva backup recenti contro interruzioni nel momento peggiore."""
    directory = os.path.dirname(filepath) or '.'
    os.makedirs(directory, exist_ok=True)
    temp_filepath = filepath + ".tmp"
    torch.save(obj, temp_filepath)
    _fsync_file(temp_filepath)
    if keep_backup:
        _rotate_backup(filepath)
    os.replace(temp_filepath, filepath)
    _fsync_dir(directory)

def safe_write_text(filepath, text, keep_backup=True):
    """Scrive un sidecar testuale in modo atomico, con backup recente."""
    directory = os.path.dirname(filepath) or '.'
    os.makedirs(directory, exist_ok=True)
    temp_filepath = filepath + ".tmp"
    with open(temp_filepath, 'w', encoding='utf-8') as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    if keep_backup:
        _rotate_backup(filepath)
    os.replace(temp_filepath, filepath)
    _fsync_dir(directory)

def safe_read_float(filepath, default):
    """Legge un numero da un sidecar testuale, provando anche backup recenti."""
    for candidate in _checkpoint_candidates(filepath):
        if not os.path.exists(candidate):
            continue
        try:
            with open(candidate, 'r', encoding='utf-8') as f:
                return float(f.read().strip())
        except Exception as e:
            print(f"Impossibile leggere valore numerico da {candidate}: {e}")
    return default

def safe_save_npz(buffer_obj, filepath, keep_backup=True):
    """Salva il buffer in modo atomico e conserva backup recenti del file .npz."""
    if len(buffer_obj.buffer) == 0:
        return
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    # np.savez_compressed appende automaticamente '.npz' se non presente.
    # Per evitarlo, facciamo terminare il file temporaneo con '.tmp.npz'.
    temp_filepath = filepath.replace(".npz", "") + ".tmp.npz"
    buffer_obj.save(temp_filepath)
    if os.path.exists(temp_filepath):
        _fsync_file(temp_filepath)
        if keep_backup:
            _rotate_backup(filepath)
        os.replace(temp_filepath, filepath)
        _fsync_dir(os.path.dirname(filepath) or '.')

# ──────────────────────────────────────────────────────────────────────
#  Normalizzazione stati mean-0 / std-1 (Fujimoto & Gu, 2021)
# ──────────────────────────────────────────────────────────────────────
# Il paper TD3+BC normalizza ogni feature dello stato a media 0 / dev.std 1 sul
# dataset: è uno dei due cambiamenti chiave per la STABILITÀ. Le statistiche
# (mean/std delle 29 feature) sono calcolate da behavioral_cloning.py sul dataset
# e salvate in state_norm.npz; qui vengono caricate e applicate alla 29D PRIMA
# dello stacking, così Actor e Critic vedono sempre stati normalizzati.
_STATE_NORM_PATH = 'train_set/checkpoints/state_norm.npz'

def _load_state_norm():
    if os.path.exists(_STATE_NORM_PATH):
        d = np.load(_STATE_NORM_PATH)
        return d['mean'].astype(np.float32), d['std'].astype(np.float32)
    return None, None

_STATE_MEAN, _STATE_STD = _load_state_norm()

def apply_state_norm(s):
    """(s - mean)/(std + 1e-3) sull'ultima dimensione (29). No-op se mancano le stat."""
    if _STATE_MEAN is None:
        return s
    return ((s - _STATE_MEAN) / (_STATE_STD + 1e-3)).astype(np.float32)

# ──────────────────────────────────────────────────────────────────────
#  Determinismo
# ──────────────────────────────────────────────────────────────────────
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# ──────────────────────────────────────────────────────────────────────
#  Replay Buffer
# ──────────────────────────────────────────────────────────────────────
class ReplayBuffer:
    """Buffer circolare per memorizzare transizioni (s, a, r, s', done).

    Ogni transizione include un flag 'expert' che indica se proviene da
    dati umani (expert=1.0) o da esplorazione RL (expert=0.0).
    Questo flag pilota la BC Penalty nell'Actor: per i campioni expert,
    l'Actor viene penalizzato se si discosta dall'azione registrata.
    """
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        self.expert_masks = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, expert=0.0):
        self.buffer.append((state, action, reward, next_state, done))
        self.expert_masks.append(expert)

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        expert_masks_batch = [self.expert_masks[i] for i in indices]
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done, np.array(expert_masks_batch, dtype=np.float32)

    def save(self, filepath: str):
        if len(self.buffer) == 0: return
        states, actions, rewards, next_states, dones = zip(*self.buffer)
        np.savez_compressed(filepath,
            states=np.array(states, dtype=np.float32),
            actions=np.array(actions, dtype=np.float32),
            rewards=np.array(rewards, dtype=np.float32),
            next_states=np.array(next_states, dtype=np.float32),
            dones=np.array(dones, dtype=np.float32),
            expert_masks=np.array(list(self.expert_masks), dtype=np.float32))

    def load_expert_data(self, h5_dir_or_file: str, max_samples: int = None):
        import glob
        import h5py
        import os

        if os.path.isdir(h5_dir_or_file):
            h5_files = sorted(glob.glob(os.path.join(h5_dir_or_file, "**/lap_*.h5"), recursive=True))
        else:
            h5_files = [h5_dir_or_file]

        loaded = 0
        for f in h5_files:
            if max_samples and loaded >= max_samples: break
            try:
                with h5py.File(f, 'r') as h5f:
                    states_np = h5f['states'][:]   # 29D grezzi (per il reward fisico)
                    actions_np = h5f['actions'][:]

                states_norm = apply_state_norm(states_np)  # normalizzati (per l'input rete)
                length = len(states_np)
                k = 6
                for i in range(length - 1):
                    if max_samples and loaded >= max_samples: break
                    idx_t6 = max(0, i - k)
                    idx_t12 = max(0, i - 2 * k)
                    next_i = i + 1
                    n_idx_t6 = max(0, next_i - k)
                    n_idx_t12 = max(0, next_i - 2 * k)

                    stacked_state = np.concatenate([states_norm[idx_t12], states_norm[idx_t6], states_norm[i]])
                    next_stacked_state = np.concatenate([states_norm[n_idx_t12], states_norm[n_idx_t6], states_norm[next_i]])

                    cont_action = actions_np[i, 0:3].copy()
                    cont_action[1] = (cont_action[1] * 2.0) - 1.0
                    cont_action[2] = (cont_action[2] * 2.0) - 1.0

                    # Valori FISICI dello stato grezzo (non normalizzato) per il reward
                    speedX = states_np[i, 21] * 50.0
                    angle = states_np[i, 0]
                    trackPos = states_np[i, 20]

                    # Reward IDENTICO a gym_torcs.step (progress + deadzone pos_penalty + steer-smooth)
                    progress = (speedX / 50.0) * np.cos(angle)
                    tp = abs(trackPos)
                    pos_penalty = -2.0 * (max(0.0, tp - 1.0) ** 2)
                    steer_change = cont_action[0] - actions_np[i-1, 0] if i > 0 else 0.0
                    reward = (progress * 1.5) + pos_penalty - (0.05 * abs(steer_change))

                    mask = 1.0 # Dati expert non sono terminali

                    self.push(stacked_state, cont_action, reward, next_stacked_state, mask, expert=1.0)
                    loaded += 1
            except Exception as e:
                print(f"Errore caricando {f}: {e}")

        print(f"  [EXPERT INJECTION] Caricati {loaded} campioni esperti nel Replay Buffer.")

    def load(self, filepath: str):
        if not os.path.exists(filepath): return
        with np.load(filepath) as data:
            states = data['states']
            actions = data['actions']
            rewards = data['rewards']
            next_states = data['next_states']
            dones = data['dones']
            expert_masks_data = data['expert_masks'] if 'expert_masks' in data.files else np.zeros(len(states))
            states, actions, rewards, next_states, dones, expert_masks_data = [
                np.asarray(x) for x in (states, actions, rewards, next_states, dones, expert_masks_data)
            ]
        lengths = {len(states), len(actions), len(rewards), len(next_states), len(dones), len(expert_masks_data)}
        if len(lengths) != 1:
            raise ValueError(f"Replay Buffer non coerente in {filepath}: lunghezze diverse {sorted(lengths)}")

        new_buffer = deque(maxlen=self.buffer.maxlen)
        new_expert_masks = deque(maxlen=self.expert_masks.maxlen)
        for i in range(len(states)):
            new_buffer.append((states[i], actions[i], float(rewards[i]), next_states[i], float(dones[i])))
            new_expert_masks.append(float(expert_masks_data[i]))
        self.buffer = new_buffer
        self.expert_masks = new_expert_masks
        print(f"  Replay Buffer caricato: {len(self.buffer)} transizioni")

    def __len__(self):
        return len(self.buffer)

def flatten_state(state_dict: dict) -> np.ndarray:
    def _s(key, default=0.0):
        v = state_dict.get(key, default)
        if isinstance(v, np.ndarray): return float(v.flat[0])
        return float(v) if v is not None else default
    def _a(key, size):
        v = state_dict.get(key, None)
        if v is None: return np.zeros(size, dtype=np.float32)
        return np.array(v, dtype=np.float32).flatten()[:size]
    try:
        s = np.concatenate([
            [_s('angle')], _a('track', 19), [_s('trackPos'), _s('speedX'), _s('speedY'), _s('speedZ')],
            _a('wheelSpinVel', 4) / 100.0, [_s('rpm') / 10000.0]
        ]).astype(np.float32)
        return apply_state_norm(s)  # mean-0/std-1 (stabilità, Fujimoto & Gu 2021)
    except Exception as e:
        # NON ingoiare silenziosamente: uno stato a zero falsa la rete ed è arduo da diagnosticare.
        print(f"flatten_state fallita (stato a zero): {e}")
        return apply_state_norm(np.zeros(29, dtype=np.float32))

# ──────────────────────────────────────────────────────────────────────
#  Architettura TD3
# ──────────────────────────────────────────────────────────────────────
class Actor(nn.Module):
    """Policy deterministica TD3 con architettura Multi-Head.

    Il backbone (4x512 con LayerNorm) estrae feature dallo stato 87D.
    continuous_head: 3 uscite (steer, accel, brake) in [-1,1] via tanh.
    gear_head: 7 logits per la selezione discreta della marcia.
    log_std_head: mantenuta SOLO per compatibilità col caricamento di
                  vecchi checkpoint SAC in test_agent.py. Completamente
                  isolata (requires_grad=False).
    """
    def __init__(self, state_dim=87, hidden_size=512):
        super(Actor, self).__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU(),
        )
        self.continuous_head = nn.Linear(hidden_size, 3)  # steer, accel, brake
        self.gear_head = nn.Linear(hidden_size, 7)        # 7 marce (0-6)

        # Legacy: retro-compatibilità con test_agent.py per vecchi pesi SAC
        self.log_std_head = nn.Linear(hidden_size, 3)
        for param in self.log_std_head.parameters():
            param.requires_grad = False

    def forward(self, state):
        features = self.backbone(state)
        mean = self.continuous_head(features)
        gear_logits = self.gear_head(features)
        action = torch.tanh(mean)
        return action, gear_logits

    def sample(self, state, evaluate=False):
        action, gear_logits = self.forward(state)
        gear_idx = torch.argmax(gear_logits, dim=-1)

        if not evaluate:
            # TD3: Rumore Gaussiano esplorativo
            noise = torch.randn_like(action) * 0.1
            noise = torch.clamp(noise, -0.2, 0.2)
            action = torch.clamp(action + noise, -1.0, 1.0)

        return action, None, gear_idx

    def load_bc_weights(self, bc_path):
        if not os.path.exists(bc_path): return
        bc_state = torch.load(bc_path, map_location='cpu', weights_only=True)
        if 'continuous_head.weight' in bc_state:
            bc_state['continuous_head.weight'][1:3] = bc_state['continuous_head.weight'][1:3] * 0.5
        if 'continuous_head.bias' in bc_state:
            bc_state['continuous_head.bias'][1:3] = bc_state['continuous_head.bias'][1:3] * 0.5
        self.load_state_dict(bc_state, strict=False)
        print(f"Pesi BC caricati con successo da {bc_path} (compensato scaling 0.5 per accel/brake).")

class Critic(nn.Module):
    """Twin Q-Network: due reti Q indipendenti per mitigare l'Overestimation Bias.

    Ogni rete Q riceve la concatenazione di stato (87D) e azione (3D)
    e stima il valore atteso della ricompensa futura scontata (Q-value).
    Durante il training, si usa min(Q1, Q2) per l'update dell'Actor.
    """
    def __init__(self, state_dim=87, action_dim=3, hidden_size=512):
        super(Critic, self).__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
        )

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        return self.q1(xu), self.q2(xu)

# ──────────────────────────────────────────────────────────────────────
#  TD3+BC Agent
# ──────────────────────────────────────────────────────────────────────
class TD3BCAgent:
    def __init__(self, device="cuda"):
        self.device = torch.device(device)
        self.gamma = 0.99   # valore di riferimento TD3+BC (era 0.999: inflazionava i Q di 10× e l'overestimation)
        self.tau = 0.005
        self.policy_freq = 2 # Delayed Policy Update

        self.actor = Actor().to(self.device)
        self.actor_target = Actor().to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Rimuoviamo la vecchia ancora Behavioral Cloning congelata: il target d'imitazione sarà
        # solo l'azione empirica (expert_mask=1.0) e non la predizione OOD.

        self.critic = Critic().to(self.device)
        self.critic_target = Critic().to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Backbone SCONGELATO (TIER 3b): come in TD3+BC originale si allena tutta la rete
        # dell'Actor. È sicuro perché l'ancora Behavioral Cloning è forte e costante
        # (peso = 1.0, variabile bc_weight nel codice, niente decay),
        # e sblocca capacità di apprendimento prima limitata alla sola testa lineare.
        # Resta congelata solo la gear_head (marcia discreta, ereditata dal BC, non soggetta a RL).
        for param in self.actor.gear_head.parameters(): param.requires_grad = False

        # LR di riferimento 3e-4 per Actor e Critic. NON si tunara a tentativi: la
        # normalizzazione λ = α/mean|Q| del TD3+BC normalizza già il learning rate
        # rispetto alla scala di Q (Fujimoto & Gu, 2021). L'iperparametro da tarare è α (=2.5).
        actor_params = [p for p in self.actor.parameters() if p.requires_grad]
        self.actor_optimizer = optim.Adam(actor_params, lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)

    def select_action(self, state, evaluate=False):
        state_t = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            cont_action, _, gear_idx = self.actor.sample(state_t, evaluate=evaluate)
        return cont_action.cpu().numpy()[0], gear_idx.cpu().item()

    def update(self, online_memory, elite_memory, expert_memory, batch_size, global_step):
        # ── Hybrid Sampling a 3 vie: Expert + Online + Elite ──
        # Expert (umano): buffer SEPARATO e permanente — mai soggetto a FIFO, quindi
        #   l'ancora BC è presente in OGNI batch (stabilità del TD3+BC).
        # Online: esperienza dell'agente (FIFO), per generalizzare in stati "sporchi".
        # Elite: record dell'agente (Self-Imitation), se disponibili.
        # Quote: 25% expert + 15% elite + 60% online. Se online/elite sono scarsi
        # (es. inizio training), il resto è riempito dall'expert (sempre pieno).
        b_expert = int(batch_size * 0.25)
        b_elite = min(int(batch_size * 0.15), len(elite_memory.buffer))
        b_online = min(batch_size - b_expert - b_elite, len(online_memory.buffer))
        b_expert = batch_size - b_online - b_elite  # il resto dall'expert (sempre disponibile)

        parts = [expert_memory.sample(b_expert)]
        if b_online > 0: parts.append(online_memory.sample(b_online))
        if b_elite > 0:  parts.append(elite_memory.sample(b_elite))
        state_b      = np.concatenate([p[0] for p in parts], axis=0)
        action_b     = np.concatenate([p[1] for p in parts], axis=0)
        reward_b     = np.concatenate([p[2] for p in parts], axis=0)
        next_state_b = np.concatenate([p[3] for p in parts], axis=0)
        mask_b       = np.concatenate([p[4] for p in parts], axis=0)
        expert_mask_b= np.concatenate([p[5] for p in parts], axis=0)

        # Reward Scaling: con gamma=0.99 i Q sono ~10× più piccoli che con 0.999,
        # quindi alziamo lo scale a 0.02 per tenere i Q in un range sano. Nota: la
        # normalizzazione λ=α/mean|Q| rende comunque l'Actor invariante alla scala;
        # questo scale serve solo a mantenere ragionevole la magnitudo del Critic.
        reward_scale = 0.02
        reward_b = reward_b * reward_scale

        state_b = torch.FloatTensor(state_b).to(self.device)
        next_state_b = torch.FloatTensor(next_state_b).to(self.device)
        action_b = torch.FloatTensor(action_b).to(self.device)
        reward_b = torch.FloatTensor(reward_b).to(self.device).unsqueeze(1)
        mask_b = torch.FloatTensor(mask_b).to(self.device).unsqueeze(1)
        expert_mask_b = torch.FloatTensor(expert_mask_b).to(self.device).unsqueeze(1)

        # ── Aggiornamento del Critic (Bellman equation con Twin Q-Network) ──
        # Il Critic stima il valore Q(s,a) di ogni coppia stato-azione.
        # Usiamo due reti Q indipendenti (Twin) e prendiamo il minimo
        # per mitigare l'Overestimation Bias tipico del Q-learning.
        with torch.no_grad():
            # Target Policy Smoothing (TD3): aggiungiamo rumore clippato
            # all'azione target per regolarizzare il Critic e impedirgli
            # di sovrastimare picchi stretti nella Q-function.
            noise = (torch.randn_like(action_b) * 0.2).clamp(-0.5, 0.5)
            next_action, _ = self.actor_target(next_state_b)
            next_action = (next_action + noise).clamp(-1.0, 1.0)

            q1_next, q2_next = self.critic_target(next_state_b, next_action)
            min_q_next = torch.min(q1_next, q2_next)
            # mask_b=0.0 per crash (Q futuro azzerato), mask_b=1.0 altrimenti
            target_q = reward_b + mask_b * self.gamma * min_q_next

        q1, q2 = self.critic(state_b, action_b)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        # In refinement (Paper 2) l'aggiornamento del Critic è disattivato:
        # l'Actor si raffina verso una value function fissa con vincolo BC ridotto.
        # critic_loss viene comunque calcolata come diagnostica, ma NON viene fatto
        # backward/step sul Critic.
        if not getattr(self, 'refine_mode', False):
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)  # Anti Gradient Explosion
            self.critic_optimizer.step()

        actor_loss_val = 0.0

        # ── Delayed Policy Update (TD3: ogni 2 step del Critic) ──
        # Nei primi 15000 step il Critic si addestra da solo (Warm-Up),
        # proteggendo l'Actor dai gradienti casuali di un Critic immaturo.
        # Dopo il warm-up, l'Actor viene aggiornato ogni 2 step (policy_freq=2)
        # per dare al Critic il tempo di stabilizzare le sue stime.
        if global_step >= 15000 and global_step % self.policy_freq == 0 and not getattr(self, 'actor_frozen', False):
            pi, _ = self.actor(state_b)
            q1_pi, _ = self.critic(state_b, pi)

            # Componente RL: l'Actor massimizza il Q-Value stimato dal Critic
            actor_loss_td3 = -q1_pi.mean()

            # ── BC Penalty (Masking Rigoroso: Solo su sotto-batch Expert) ──
            # Isola i campioni esperti nel batch per evitare la diluizione della loss.
            # La BC penalty è calcolata esclusivamente su questi campioni, garantendo
            # un corretto ancoraggio al comportamento umano indipendentemente dalla quota di campioni online.
            det_steer = pi[:, 0]
            det_accel = (pi[:, 1] + 1.0) / 2.0
            det_brake = (pi[:, 2] + 1.0) / 2.0
            target_steer = action_b[:, 0]
            target_accel = (action_b[:, 1] + 1.0) / 2.0
            target_brake = (action_b[:, 2] + 1.0) / 2.0

            expert_mask_flat = expert_mask_b.squeeze(1)
            expert_indices = torch.where(expert_mask_flat > 0.5)[0]

            if len(expert_indices) > 0:
                steer_loss = F.mse_loss(det_steer[expert_indices], target_steer[expert_indices])
                accel_loss = F.mse_loss(det_accel[expert_indices], target_accel[expert_indices])
                brake_loss = F.mse_loss(det_brake[expert_indices], target_brake[expert_indices])
                # Somma i contributi senza dividere per la somma dei pesi per mantenere
                # la magnitudo corretta contro la costante lambda del paper.
                bc_penalty = (steer_loss * 2.0 + accel_loss + brake_loss * 2.0)
            else:
                bc_penalty = torch.tensor(0.0, device=self.device)

            mutual_exclusion_penalty = (det_accel * det_brake).mean()
            bc_penalty = bc_penalty + (mutual_exclusion_penalty * 0.1)

            # ── Normalizzazione λ del TD3+BC (Fujimoto & Gu, 2021) ──
            # λ = α / mean|Q| con α=2.5: rende il termine RL invariante alla scala di Q
            # (e normalizza implicitamente il learning rate).
            lambda_val = 2.5
            Q_abs_mean = q1_pi.abs().mean().detach().clamp(min=1e-5)
            dynamic_alpha = lambda_val / Q_abs_mean

            # ── Peso Behavioral Cloning COSTANTE = 1.0 (niente decay) ──
            # Questo riproduce ESATTAMENTE la loss del TD3+BC originale:
            #   L = -λ·Q + (π - a)²  (qui bc_penalty è la nostra MSE pesata).
            # Il decay precedente (1.0→0.5) indeboliva la BC nella fase fragile post-warm-up
            # → "troppo RL troppo presto" → collasso (Beeson & Montana 2022, Ablation 1;
            # Fujimoto & Gu 2021, ablation su α). L'eventuale rilassamento del vincolo va fatto
            # in una FASE separata dopo il training stabile, con aggiornamento del Critic
            # disattivato (vedi AUTO-REFINEMENT nel loop di train): lì il peso Behavioral
            # Cloning scende a refine_bc_weight.
            bc_weight = self.refine_bc_weight if getattr(self, 'refine_mode', False) else 1.0

            total_actor_loss = dynamic_alpha * actor_loss_td3 + (bc_weight * bc_penalty)

            self.actor_optimizer.zero_grad()
            total_actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)  # Anti Gradient Explosion
            self.actor_optimizer.step()
            actor_loss_val = total_actor_loss.item()

            # ── Soft Update (Polyak Averaging, τ=0.005) ──
            # Aggiorna lentamente le reti target per stabilizzare il training.
            # Un τ piccolo garantisce che le reti target cambino in modo
            # graduale, evitando oscillazioni violente nei Q-value stimati.
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return critic_loss.item(), actor_loss_val, 0.0

    def save_checkpoint(self, filepath, episode, global_step, memory, elite_memory=None, best_lap_time=float('inf'), best_eval_dist=0.0, best_distance=0.0):
        checkpoint = {
            'checkpoint_version': 2,
            'saved_at': datetime.now().isoformat(),
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'episode': episode,
            'global_step': global_step,
            'best_lap_time': best_lap_time,
            'best_eval_dist': best_eval_dist,
            'best_distance': best_distance,
        }
        buffer_dir = os.path.join(os.path.dirname(filepath), 'buffers')
        os.makedirs(buffer_dir, exist_ok=True)
        base_name = os.path.basename(filepath).replace('.pth', '')

        # I buffer vengono salvati prima: il checkpoint .pth è il commit marker finale.
        # Se il processo viene interrotto a metà, il resume userà il checkpoint completo
        # precedente invece di uno stato neurale più nuovo con buffer ancora vecchi.
        safe_save_npz(memory, os.path.join(buffer_dir, f"{base_name}_buffer.npz"))
        if elite_memory:
            safe_save_npz(elite_memory, os.path.join(buffer_dir, f"{base_name}_elite_buffer.npz"))
        safe_save(checkpoint, filepath)

    def load_checkpoint(self, filepath, memory, elite_memory=None):
        buffer_dir = os.path.join(os.path.dirname(filepath), 'buffers')
        base_name = os.path.basename(filepath).replace('.pth', '')
        buffer_path = os.path.join(buffer_dir, f"{base_name}_buffer.npz")
        elite_buffer_path = os.path.join(buffer_dir, f"{base_name}_elite_buffer.npz")

        for candidate in _checkpoint_candidates(buffer_path):
            if not os.path.exists(candidate):
                continue
            try:
                memory.load(candidate)
                if candidate != buffer_path:
                    print(f"Replay Buffer recuperato dal backup: {candidate}")
                break
            except Exception as e:
                print(f"Impossibile caricare Replay Buffer da {candidate}: {e}")

        if elite_memory:
            for candidate in _checkpoint_candidates(elite_buffer_path):
                if not os.path.exists(candidate):
                    continue
                try:
                    elite_memory.load(candidate)
                    if candidate != elite_buffer_path:
                        print(f"Elite Buffer recuperato dal backup: {candidate}")
                    break
                except Exception as e:
                    print(f"Impossibile caricare Elite Buffer da {candidate}: {e}")

        loaded_ok = False
        for candidate in _checkpoint_candidates(filepath):
            if not os.path.exists(candidate):
                continue
            try:
                checkpoint = torch.load(candidate, map_location=self.device, weights_only=False)
                if isinstance(checkpoint, dict) and 'actor' in checkpoint:
                    required_keys = ['actor', 'critic', 'critic_target', 'actor_optimizer', 'critic_optimizer', 'episode', 'global_step']
                    missing_keys = [k for k in required_keys if k not in checkpoint]
                    if missing_keys:
                        raise KeyError(f"checkpoint incompleto, chiavi mancanti: {missing_keys}")
                    self.actor.load_state_dict(checkpoint['actor'])
                    if 'actor_target' in checkpoint: self.actor_target.load_state_dict(checkpoint['actor_target'])
                    self.critic.load_state_dict(checkpoint['critic'])
                    self.critic_target.load_state_dict(checkpoint['critic_target'])
                    self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
                    self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])

                    best_lap_time = checkpoint.get('best_lap_time', float('inf'))
                    best_eval_dist = checkpoint.get('best_eval_dist', 0.0)
                    best_distance = checkpoint.get('best_distance', 0.0)
                    episode = checkpoint['episode']
                    global_step = checkpoint['global_step']
                    if candidate != filepath:
                        print(f"Checkpoint principale non usato: recupero da backup {candidate}")
                    print(f"Checkpoint caricato: ripresa dall'Episodio {episode}")
                else:
                    # È un file di soli pesi dell'actor (come td3_expl_best_dist.pth).
                    print(f"{candidate} contiene solo pesi dell'Actor. Inizializzazione degli altri componenti.")
                    self.actor.load_state_dict(checkpoint)
                    self.actor_target.load_state_dict(self.actor.state_dict())
                    best_lap_time = float('inf')
                    best_eval_dist = 0.0
                    best_distance = 0.0
                    episode = 0
                    global_step = 0
                loaded_ok = True
                break
            except Exception as e:
                print(f"Checkpoint non utilizzabile da {candidate}: {e}")

        if not loaded_ok:
            if not any(os.path.exists(candidate) for candidate in _checkpoint_candidates(filepath)):
                return 0, 0, float('inf'), 0.0, 0.0
            print("Nessun checkpoint completo valido trovato tra principale e backup recenti.")
            print("Tentativo di recupero minimo delle informazioni dal log...")
            log_file = 'train_set/session_logs/td3_training.log'
            last_ep = 0
            if os.path.exists(log_file):
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            match = re.search(r'\b(?:Episode|Episodio|Ep)\s+(\d+)', line)
                            if match:
                                ep_num = int(match.group(1))
                                last_ep = max(last_ep, ep_num)
                except Exception:
                    pass
            print(f"Ripristinato ultimo episodio: {last_ep}. Il training riprenderà dall'episodio {last_ep + 1}.")
            episode = last_ep + 1
            global_step = episode * 1500
            best_lap_time = float('inf')

            best_eval_dist = 0.0
            det_best_dist_txt = 'train_set/checkpoints/td3_det_best_dist.txt'
            best_eval_dist = safe_read_float(det_best_dist_txt, 0.0)
            best_distance = best_eval_dist

        # NB: nessun fallback hardcoded sui record storici. Valori hardcoded (es. 84.3s/3619m
        # di una run specifica) corrompevano l'Elite Buffer su un resume weights-only:
        # best_distance alto → elite_threshold = best_distance*0.9 si alza subito e il buffer
        # non si riempie più (Self-Imitation spento). Su weights-only i record ripartono puliti.
        return episode, global_step, best_lap_time, best_eval_dist, best_distance

def load_recent_evals_from_log(log_path, max_len=8):
    evals = []
    if os.path.exists(log_path):
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if '[EVAL]' in line:
                        try:
                            match = re.search(r'\b(?:Dist|Distanza)\s+([0-9]+(?:\.[0-9]+)?)m', line)
                            if match:
                                evals.append(float(match.group(1)))
                        except Exception:
                            pass
        except Exception as e:
            print(f"Impossibile leggere lo storico eval dal log: {e}")
    return evals[-max_len:]

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bc_weights', type=str, default='train_set/checkpoints/bc_policy.pth')
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--max_steps', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--rollback', action='store_true', help="Forza il rollback dell'Actor alla migliore policy deterministica e lo congela temporaneamente")
    parser.add_argument('--actor-freeze-episodes', '--actor_freeze_episodes', dest='actor_freeze_episodes',
                        type=int, default=30,
                        help="Numero di episodi di congelamento Actor dopo --rollback (default: 30; aumenta per dare piu' tempo al Critic)")
    parser.add_argument('--no-auto-refine', '--no_auto_refine', dest='no_auto_refine',
                        action='store_true',
                        help="Disattiva solo la refinement automatica da plateau; --refine manuale resta disponibile")
    parser.add_argument('--refine', action='store_true', help="Avvia subito la refinement: aggiornamento del Critic disattivato, loss Critic solo diagnostica, peso Behavioral Cloning ridotto")
    parser.add_argument('--pretrain_critic', action='store_true', help="Esegue il pre-training offline del Critic per 50k passi in caso di emergenza (da usare con --rollback)")
    args = parser.parse_args()
    if args.actor_freeze_episodes < 0:
        parser.error("--actor-freeze-episodes deve essere >= 0")
    actor_freeze_episodes = args.actor_freeze_episodes
    auto_refine_enabled = not args.no_auto_refine

    set_seed(args.seed)

    env = TorcsEnv(vision=False, throttle=True, gear_change=True, early_termination=True)
    # Buffer ONLINE (FIFO) per l'esperienza dell'agente. 1M transizioni (~0.7 GB RAM):
    # con ~700 step/episodio copre ~1400 episodi senza evizione precoce.
    memory = ReplayBuffer(1000000)
    elite_memory = ReplayBuffer(20000)
    # Buffer EXPERT SEPARATO e PERMANENTE (dati umani): capacità > dataset così non viene
    # MAI svuotato dalla FIFO. Risolve la perdita dell'ancora BC e la rende presente in ogni batch.
    expert_memory = ReplayBuffer(400000)

    # ── Inizializzazione Agent ──
    agent = TD3BCAgent()
    # NB: i pesi BC si caricano una sola volta — al fresh-start (blocco sotto). In resume
    # vengono sovrascritti da load_checkpoint, quindi un caricamento qui sarebbe sprecato.
    checkpoint_path = 'train_set/checkpoints/td3_checkpoint.pth'
    start_episode, global_step, best_lap_time, best_eval_dist, best_distance = agent.load_checkpoint(checkpoint_path, memory, elite_memory)

    # I dati esperti vengono SEMPRE (ri)caricati da disco nel buffer permanente — sia al
    # fresh-start sia in resume — così l'ancora umana è garantita per tutto il training.
    expert_memory.load_expert_data('train_set/laps', max_samples=350000)

    agent.actor_frozen = False

    # ─────────────────────────────────────────────────────────────────────────
    #  AUTO-REFINEMENT (Beeson & Montana 2022): di default, dopo un PLATEAU stabile
    #  disattiva l'aggiornamento del Critic e riduce il vincolo BC per spingere la
    #  policy deterministica oltre il muro. Si puo' spegnere con --no-auto-refine
    #  quando il Critic ha bisogno di recuperare stabilita' dopo rollback/corruzioni.
    #  --refine resta sempre manuale e immediato.
    # ─────────────────────────────────────────────────────────────────────────
    REFINE_BC_WEIGHT = 0.3         # vincolo BC ridotto durante la refinement
    REFINE_MAX_ATTEMPTS = 3        # oltre, resta in training normale (no loop)
    REFINE_COLLAPSE_FRAC = 0.6     # eval < 60% del LIVELLO RECENTE per 3 volte → rollback
    REFINE_WINDOW = 8              # ampiezza finestra eval per MEDIA/MAX recenti (statistica robusta)
    REFINE_PLATEAU_EVALS = 4       # valutazioni con MEDIA recente NON in salita → plateau (auto)
    REFINE_MIN_EP = 200            # episodio minimo per l'auto-trigger
    REFINE_IMPROVE_FRAC = 1.02     # la media deve salire >2% per contare come "miglioramento"
    REFINE_BREAKOUT_FRAC = 1.10    # breakout reale: eval > riferimento rollback del 10%
    REFINE_NEAR_BEST_MARGIN = 5.0  # metri: se il breakout è vicino al best globale, consolidiamo subito
    BEST_DIST_EPS = 1.0            # evita di risalvare/loggare record identici per jitter sub-metrico
    agent.refine_mode = False
    agent.refine_bc_weight = 1.0
    recent_eval_window = deque(maxlen=REFINE_WINDOW)  # ultimi eval → media/max recenti

    # Popoliamo la finestra leggendo i dati recenti direttamente dal log
    initial_evals = load_recent_evals_from_log('train_set/session_logs/td3_training.log', REFINE_WINDOW)
    for ev in initial_evals:
        recent_eval_window.append(ev)
    if len(recent_eval_window) > 0:
        print(f"Caricati {len(recent_eval_window)} eval precedenti dal log. Storico: {list(recent_eval_window)}")
    if auto_refine_enabled:
        print("Auto-refinement automatica: attiva di default.")
    else:
        print("Auto-refinement automatica: disattivata da --no-auto-refine. --refine manuale resta disponibile.")

    refine_best_mean = 0.0         # miglior MEDIA-finestra vista (segnale di plateau)
    if len(recent_eval_window) >= REFINE_WINDOW:
        refine_best_mean = sum(recent_eval_window) / len(recent_eval_window)

    refine_evals_no_improve = 0
    refine_attempts = 0
    refine_plateau_level = 0.0     # 0 = riferimento rollback non ancora impostato
    refine_collapse_count = 0
    refine_evals_count = 0
    refine_breakout_logged = False  # per loggare UNA volta il superamento del plateau
    if getattr(args, 'refine', False):
        # Avvio mirato quando l'operatore SA già che è in plateau: refinement attiva da subito.
        agent.refine_mode = True
        agent.refine_bc_weight = REFINE_BC_WEIGHT

        # Inizializziamo il livello di riferimento per il rollback di sicurezza
        # usando lo storico appena letto dal log, o il record caricato o td3_det_best_dist.
        # Se stiamo facendo un rollback, escludiamo lo storico recente (che è degradato)
        # e usiamo direttamente il best_eval_dist o il file td3_det_best_dist.
        if len(recent_eval_window) >= 4 and not getattr(args, 'rollback', False):
            refine_plateau_level = float(np.median(list(recent_eval_window)))
        else:
            refine_plateau_level = best_eval_dist
            det_best_dist_txt = 'train_set/checkpoints/td3_det_best_dist.txt'
            if refine_plateau_level <= 0.0:
                refine_plateau_level = safe_read_float(det_best_dist_txt, 0.0)

        if refine_plateau_level > 0.0:
            print(f"--refine attivo: refinement attiva da subito "
                  f"(aggiornamento Critic disattivato, peso Behavioral Cloning={REFINE_BC_WEIGHT}). "
                  f"Riferimento rollback={refine_plateau_level:.0f}m. Loss Critic solo diagnostica.")
        else:
            print(f"--refine attivo: refinement attiva da subito "
                  f"(aggiornamento Critic disattivato, peso Behavioral Cloning={REFINE_BC_WEIGHT}). "
                  f"Riferimento rollback impostato dopo i primi eval (mediana recente). Loss Critic solo diagnostica.")

    batch_size = 256

    # Warm-Start: se è il primo avvio (nessun checkpoint), inizializza l'Actor con i pesi BC.
    if start_episode == 0:
        agent.actor.load_bc_weights(args.bc_weights)
        agent.actor_target.load_state_dict(agent.actor.state_dict())
    else:
        # Rollback Actor: solo se esplicitamente richiesto da riga di comando.
        # PRIORITÀ DETERMINISTICA (la competizione/submission usa la policy senza rumore):
        # si riparte dalla migliore policy DETERMINISTICA, non dal giro esplorativo (rumoroso,
        # "fortunato"). Gli esplorativi (best_lap/best_dist) sono solo un ripiego estremo.
        # ── PROCEDURA DI EMERGENZA (SAFETY-NET) ──
        # Questa procedura scatta SOLO se viene esplicitamente passato il parametro --rollback.
        # Serve per recuperare da corruzioni del checkpoint principale (Critic degradato o crash)
        # riallineando il Critic sui dati offline storici prima di riprendere il training normale.
        if args.rollback:
            rollback_candidates = [
                'train_set/checkpoints/td3_det_best_lap.pth',  # 1) giro VALIDO deterministico più veloce
                'train_set/checkpoints/td3_det_best_dist.pth',          # 2) miglior DISTANZA deterministica (sopravvive a --clean)
                'train_set/checkpoints/td3_det_best_dist_run.pth',          # 3) miglior eval deterministico del run corrente
                'train_set/checkpoints/td3_expl_best_lap.pth',           # 4) giro ESPLORATIVO (rumoroso) — ripiego
                'train_set/checkpoints/td3_expl_best_dist.pth',          # 5) distanza ESPLORATIVA — ripiego
            ]
            best_path = next((p for p in rollback_candidates if os.path.exists(p)), None)
            if best_path:
                print(f"[EMERGENZA] Rollback Actor: caricamento della migliore policy deterministica da {best_path}")
                agent.actor.load_state_dict(torch.load(best_path, map_location=agent.device))
                agent.actor_target.load_state_dict(agent.actor.state_dict())
                import torch.optim as optim
                agent.actor_optimizer = optim.Adam(
                    [p for p in agent.actor.parameters() if p.requires_grad], lr=3e-4)
                # Attiviamo il congelamento temporaneo dell'Actor post-rollback, durata configurabile.
                if actor_freeze_episodes > 0:
                    agent.actor_frozen = True
                    print(f"Actor congelato per {actor_freeze_episodes} episodi: stabilizzazione post-rollback.")
                else:
                    agent.actor_frozen = False
                    print("Congelamento Actor post-rollback disattivato (--actor-freeze-episodes 0).")

                # Eseguiamo il pre-training del Critic sui dati offline del buffer (procedura di sicurezza una-tantum, solo con flag dedicato)
                if getattr(args, 'pretrain_critic', False) and (len(memory) > batch_size or len(expert_memory) > batch_size):
                    print("[EMERGENZA] Pre-addestramento del Critic in corso sui dati offline del Replay Buffer (50,000 passi)...")
                    for pretrain_step in range(50000):
                        critic_loss_val, _, _ = agent.update(memory, elite_memory, expert_memory, batch_size, global_step=0)
                        if (pretrain_step + 1) % 10000 == 0:
                            print(f"  [Pre-addestramento] Passo {pretrain_step + 1}/50000 | Loss del Critic: {critic_loss_val:.4f}")
                    print("Pre-addestramento del Critic completato con successo!")
            else:
                print("Rollback richiesto ma nessun checkpoint valido trovato! Avvio ripresa normale.")
        else:
            print("Ripresa regolare dal checkpoint (nessun rollback o congelamento Actor).")

    os.makedirs('train_set/checkpoints', exist_ok=True)
    os.makedirs('train_set/session_logs', exist_ok=True)
    log_file = 'train_set/session_logs/td3_training.log'
    # Conferma nel LOG (non solo stdout) se la refinement è stata armata da --refine, così è
    # tracciabile a posteriori senza ambiguità con l'AUTO-REFINEMENT che scatta da solo.
    if getattr(args, 'refine', False):
        with open(log_file, 'a', encoding='utf-8') as f:
            initial_ref = f"{refine_plateau_level:.0f}m" if refine_plateau_level > 0.0 else "da impostare"
            f.write(f"AVVIO con --refine: REFINEMENT armata da subito "
                    f"(aggiornamento Critic disattivato, loss Critic solo diagnostica, "
                    f"peso Behavioral Cloning={REFINE_BC_WEIGHT}, riferimento rollback={initial_ref}, "
                    f"episodio iniziale {start_episode})\n")
    if getattr(args, 'rollback', False):
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"AVVIO con --rollback: Actor congelato per {actor_freeze_episodes} episodi "
                    f"(0 = nessun congelamento), auto-refinement automatica="
                    f"{'attiva' if auto_refine_enabled else 'disattivata'}.\n")
    elif not auto_refine_enabled:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write("AVVIO con --no-auto-refine: refinement automatica disattivata; "
                    "--refine manuale resta disponibile.\n")

    batch_size = 256
    elite_threshold = 500.0

    print("Avvio training TD3+BC...")

    for episode in range(start_episode, args.episodes):
        # Gestione dello scongelamento dell'Actor dopo la fase di stabilizzazione post-rollback
        if agent.actor_frozen and episode >= start_episode + actor_freeze_episodes:
            agent.actor_frozen = False
            print(f"Actor scongelato dopo {actor_freeze_episodes} episodi: "
                  f"riavvio aggiornamenti Actor con gradienti del Critic stabilizzati.")

        ob = env.reset(relaunch=True)
        episode_transitions = []

        # Frame Stacking (87D): concatena 3 frame temporalmente distanziati
        # (t-12, t-6, t) per dare alla rete una percezione della velocità
        # e dell'accelerazione senza doverle calcolare esplicitamente.
        f_state = flatten_state(ob)
        state_stack = deque([f_state]*13, maxlen=13)
        stacked_state = np.concatenate([state_stack[0], state_stack[6], state_stack[12]])

        episode_reward, step, max_dist = 0, 0, 0.0
        critic_losses, actor_losses = [], []
        termination_reason = "TIMEOUT"
        new_record = False

        # Marcia deterministica (gearing.py): identica in training/eval/test → i checkpoint
        # (best_lap/best_eval) sono esattamente riproducibili in inferenza.
        current_gear = 1
        steps_since_shift = 999  # consenti il primo cambio subito
        cur_speed_kmh = float(np.array(ob.get('speedX', 0.0)).flat[0]) * 50.0
        cur_rpm = float(np.array(ob.get('rpm', 0.0)).flat[0])
        # Rilevamento robusto del completamento giro: confrontiamo lastLapTime
        # con il valore iniziale invece di assumere che il relaunch lo azzeri.
        prev_last_lap = float(np.array(ob.get('lastLapTime', 0.0)).flat[0])

        while True:
            # NB: niente actor.eval()/train() qui — sarebbe un no-op fuorviante (nessun dropout;
            # il LayerNorm è indipendente dal batch). select_action usa già torch.no_grad().
            cont_action, _raw_gear = agent.select_action(stacked_state, evaluate=False)

            # Mappatura Action Space: l'Actor emette azioni in [-1,1] (spazio tanh),
            # ma TORCS si aspetta accel/brake in [0,1]. La conversione (x+1)/2
            # preserva la simmetria del tanh per la backpropagation.
            env_action = np.zeros(4)
            env_action[0:3] = cont_action

            torcs_action = env_action.copy()
            torcs_action[1] = np.clip((torcs_action[1] + 1.0) / 2.0, 0.0, 1.0)  # accel: [-1,1] → [0,1]
            torcs_action[2] = np.clip((torcs_action[2] + 1.0) / 2.0, 0.0, 1.0)  # brake: [-1,1] → [0,1]
            # Mutual exclusion continua/moltiplicativa per prevenire stalli repentini
            torcs_action[1] = torcs_action[1] * (1.0 - torcs_action[2])

            # Marcia DETERMINISTICA (anti-hunting): da velocità/rpm correnti + il gas applicato.
            # _raw_gear (gear_head congelata) è ignorato. Vedi gearing.py.
            current_gear, _shifted = compute_gear(cur_speed_kmh, torcs_action[1], cur_rpm, current_gear, steps_since_shift)
            steps_since_shift = 0 if _shifted else steps_since_shift + 1
            torcs_action[3] = current_gear
            env_action[3] = current_gear

            next_ob, reward, env_done, info = env.step(torcs_action)
            cur_speed_kmh = float(np.array(next_ob.get('speedX', 0.0)).flat[0]) * 50.0
            cur_rpm = float(np.array(next_ob.get('rpm', 0.0)).flat[0])
            next_f_state = flatten_state(next_ob)
            state_stack.append(next_f_state)

            current_dist = float(np.array(next_ob.get('distRaced', 0.0)).flat[0])
            last_lap_time = float(np.array(next_ob.get('lastLapTime', 0.0)).flat[0])
            max_dist = max(max_dist, current_dist)

            done = False
            if last_lap_time > 0.0 and abs(last_lap_time - prev_last_lap) > 0.01 and step > 500:
                done, termination_reason = True, "SUCCESS"
                reward += 50.0
                if last_lap_time < best_lap_time:
                    best_lap_time = last_lap_time
                    new_record = True
                    safe_save(agent.actor.state_dict(), 'train_set/checkpoints/td3_expl_best_lap.pth')

            if info.get('crash', False):
                done, termination_reason = True, "CRASH"


            if max_dist > best_distance and max_dist > 500.0:
                best_distance = max_dist
                safe_save(agent.actor.state_dict(), 'train_set/checkpoints/td3_expl_best_dist.pth')

            next_stacked_state = np.concatenate([state_stack[0], state_stack[6], state_stack[12]])
            mask = 0.0 if info.get('crash', False) else 1.0

            time_limit_reached = (step >= args.max_steps)
            episode_transitions.append((stacked_state, cont_action, reward, next_stacked_state, mask))

            stacked_state = next_stacked_state
            episode_reward += reward
            step += 1
            global_step += 1

            # Update Frequency 1:1: un aggiornamento ad ogni step di simulazione (standard TD3).
            # L'expert buffer è sempre pieno → si aggiorna fin dal primo step (il Critic si
            # pre-allena sui dati umani, come la fase offline del TD3+BC) e l'ancora BC è garantita.
            if len(expert_memory) > batch_size:
                critic_loss_val, actor_loss_val, _ = agent.update(memory, elite_memory, expert_memory, batch_size, global_step)
                critic_losses.append(critic_loss_val)
                if actor_loss_val != 0.0:
                    actor_losses.append(actor_loss_val)

            if done or env_done or time_limit_reached:
                for t in episode_transitions:
                    memory.push(t[0], t[1], t[2], t[3], t[4], expert=0.0)

                # ── Elite Buffer Injection ──
                # Se la distanza supera la soglia (best_distance * 0.7), l'episodio viene clonato
                # nell'Elite Buffer per Self-Imitation. Gli ultimi 50 step di un crash sono flaggati
                # expert=0.0 per evitare Causal Confusion (imitare azioni pre-schianto).
                # NB: moltiplicatore abbassato da 0.9 a 0.7. Con 0.9 un singolo colpo fortunato
                # precoce (es. 2793m all'ep2) bloccava la soglia troppo in alto: l'Elite Buffer
                # si riempiva ~3 volte su 144 episodi e il Self-Imitation rinforzava solo poche
                # traiettorie irriproducibili (instabilità). Con 0.7 si riempie più spesso pur
                # richiedendo episodi "buoni", restando monotonicamente non decrescente.
                if max_dist >= elite_threshold:
                    n_trans = len(episode_transitions)
                    for i, t in enumerate(episode_transitions):
                        is_danger = (termination_reason == "CRASH") and (i >= n_trans - 50)
                        elite_memory.push(t[0], t[1], t[2], t[3], t[4], expert=0.0 if is_danger else 1.0)
                    elite_threshold = max(500.0, best_distance * 0.7)  # Soglia monotonicamente crescente
                break

        lap_time = step * 0.02
        time_str = datetime.now().strftime("%H:%M:%S")
        avg_critic_loss = np.mean(critic_losses) if len(critic_losses) > 0 else 0.0
        avg_actor_loss = np.mean(actor_losses) if len(actor_losses) > 0 else 0.0
        critic_status = "OFF" if getattr(agent, 'refine_mode', False) else "ON"
        if global_step < 15000:
            actor_status = "WARM"
        elif getattr(agent, 'actor_frozen', False):
            actor_status = "FREEZE"
        else:
            actor_status = "ON"
        log_msg = (f"[{time_str}] Ep {episode+1:03d} | [{termination_reason}] | "
                   f"Reward: {episode_reward:7.1f} | Steps: {step:4d} | "
                   f"Time: {lap_time:5.1f}s | Dist: {int(max_dist):5d}m | "
                   f"CriticL: {avg_critic_loss:.3f} ({critic_status}) | "
                   f"ActorL: {avg_actor_loss:.3f} ({actor_status})")
        if new_record: log_msg += f" | Record"
        print(f" {log_msg}")

        with open(log_file, 'a', encoding='utf-8') as f: f.write(log_msg + "\n")

        agent.save_checkpoint(checkpoint_path, episode + 1, global_step, memory, elite_memory,
                              best_lap_time=best_lap_time,
                              best_eval_dist=best_eval_dist,
                              best_distance=best_distance)
        safe_save(agent.actor.state_dict(), 'train_set/checkpoints/td3_policy.pth')

        if (episode + 1) % 5 == 0 and global_step > 15000:
            def _rlog(m):
                print(m)
                with open(log_file, 'a', encoding='utf-8') as f: f.write(m + "\n")
            _rlog("\n   [EVAL] Valutazione deterministica...")
            eval_ob = env.reset(relaunch=True)
            eval_stack = deque([flatten_state(eval_ob)]*13, maxlen=13)
            eval_stacked = np.concatenate([eval_stack[0], eval_stack[6], eval_stack[12]])
            eval_dist, eval_step, eval_reward = 0.0, 0, 0.0
            eval_current_gear = 1  # marcia deterministica anche in eval (gearing.py)
            eval_steps_since_shift = 999
            eval_cur_speed_kmh = float(np.array(eval_ob.get('speedX', 0.0)).flat[0]) * 50.0
            eval_cur_rpm = float(np.array(eval_ob.get('rpm', 0.0)).flat[0])
            # Cronometraggio del miglior giro VALIDO completato in questa valutazione deterministica.
            eval_prev_last_lap = float(np.array(eval_ob.get('lastLapTime', 0.0)).flat[0])
            eval_best_lap_in_run = float('inf')

            agent.actor.eval()
            while eval_step < args.max_steps:
                eval_step += 1
                with torch.no_grad():
                    eval_action, _eval_gear = agent.select_action(eval_stacked, evaluate=True)
                eval_env = np.zeros(4)
                eval_env[0:3] = eval_action
                eval_env[1], eval_env[2] = np.clip((eval_env[1]+1)/2, 0, 1), np.clip((eval_env[2]+1)/2, 0, 1)
                # Mutual exclusion continua/moltiplicativa per EVAL
                eval_env[1] = eval_env[1] * (1.0 - eval_env[2])
                # Marcia DETERMINISTICA (anti-hunting), identica a training/test (gearing.py)
                eval_current_gear, _esh = compute_gear(eval_cur_speed_kmh, eval_env[1], eval_cur_rpm, eval_current_gear, eval_steps_since_shift)
                eval_steps_since_shift = 0 if _esh else eval_steps_since_shift + 1
                eval_env[3] = eval_current_gear

                eval_ob, eval_r, eval_done, eval_info = env.step(eval_env)
                eval_cur_speed_kmh = float(np.array(eval_ob.get('speedX', 0.0)).flat[0]) * 50.0
                eval_cur_rpm = float(np.array(eval_ob.get('rpm', 0.0)).flat[0])
                eval_reward += eval_r
                eval_stack.append(flatten_state(eval_ob))
                eval_stacked = np.concatenate([eval_stack[0], eval_stack[6], eval_stack[12]])
                eval_dist = float(np.array(eval_ob.get('distRaced', 0.0)).flat[0])

                # Rilevamento giro VALIDO deterministico: TORCS aggiorna lastLapTime al traguardo.
                # L'eval prosegue oltre il traguardo (non termina sul giro), quindi può chiudere più
                # giri: teniamo il più veloce. Stesso criterio del loop di training (step>500).
                eval_last_lap = float(np.array(eval_ob.get('lastLapTime', 0.0)).flat[0])
                if eval_last_lap > 0.0 and abs(eval_last_lap - eval_prev_last_lap) > 0.01 and eval_step > 500:
                    eval_prev_last_lap = eval_last_lap
                    eval_best_lap_in_run = min(eval_best_lap_in_run, eval_last_lap)

                if eval_info.get('crash', False) or eval_done: break
            agent.actor.train()

            refine_status = ""
            if getattr(agent, 'refine_mode', False):
                riferimento_rollback = f"{refine_plateau_level:.0f}m" if refine_plateau_level > 0.0 else "none"
                refine_status = (f" | Refine: ON (BC={agent.refine_bc_weight:.1f}, "
                                 f"Critic=OFF, rollback_ref={riferimento_rollback})")
            else:
                refine_status = " | Refine: OFF"

            eval_msg = f"[{time_str}]  [EVAL] Result: Dist {int(eval_dist)}m | Reward: {eval_reward:.1f}{refine_status}"
            print(f"  {eval_msg}")
            with open(log_file, 'a', encoding='utf-8') as f: f.write(eval_msg + "\n")

            if eval_dist > best_eval_dist:
                best_eval_dist = eval_dist
                safe_save(agent.actor.state_dict(), 'train_set/checkpoints/td3_det_best_dist_run.pth')

            # ── Best-Ever (sopravvive a --clean) ──
            # td3_det_best_dist_run.pth viene cancellato da --clean. Per non perdere MAI la migliore
            # policy raggiunta tra run diversi, manteniamo td3_det_best_dist.pth + un sidecar .txt
            # con la sua distanza. train_rl.sh --clean NON cancella questi due file.
            det_best_dist_pth = 'train_set/checkpoints/td3_det_best_dist.pth'
            det_best_dist_txt = 'train_set/checkpoints/td3_det_best_dist.txt'
            prev_det_best_dist = safe_read_float(det_best_dist_txt, 0.0)
            if eval_dist > prev_det_best_dist + BEST_DIST_EPS:
                safe_save(agent.actor.state_dict(), det_best_dist_pth)
                safe_write_text(det_best_dist_txt, f"{eval_dist:.2f}")
                msg = (f"  NUOVO MIGLIOR DETERMINISTICO ASSOLUTO: {int(eval_dist)}m "
                       f"(precedente {int(prev_det_best_dist)}m, preservato anche dopo --clean)")
                print(msg)
                with open(log_file, 'a', encoding='utf-8') as f: f.write(msg + "\n")
                if getattr(agent, 'refine_mode', False):
                    agent.refine_mode = False
                    agent.refine_bc_weight = 1.0
                    agent.actor_frozen = actor_freeze_episodes > 0
                    start_episode = episode  # Congela l'Actor per la finestra configurata a partire da ora
                    refine_evals_no_improve = 0
                    refine_collapse_count = 0
                    refine_best_mean = 0.0
                    refine_plateau_level = 0.0
                    refine_evals_count = 0
                    recent_eval_window.clear()
                    _rlog("  REFINEMENT CONCLUSA CON SUCCESSO! Nuovo record deterministico rilevato.")
                    if actor_freeze_episodes > 0:
                        _rlog(f"  Rientro in modalità allineamento Critic: "
                              f"Actor congelato per {actor_freeze_episodes} episodi.")
                    else:
                        _rlog("  Rientro in training normale: congelamento Actor disattivato.")

            # ── Miglior tempo su giro valido in eval deterministica (candidato submission) ──
            # A differenza di td3_det_best_dist (basato sulla DISTANZA), questo cattura il GIRO VALIDO più
            # VELOCE chiuso in eval deterministica: esattamente la policy da sottomettere. Sidecar
            # .txt col tempo; train_rl.sh --clean NON lo cancella (preservato tra run).
            if eval_best_lap_in_run < float('inf'):
                det_best_lap_pth = 'train_set/checkpoints/td3_det_best_lap.pth'
                det_best_lap_txt = 'train_set/checkpoints/td3_det_best_lap.txt'
                prev_det_best_lap = safe_read_float(det_best_lap_txt, float('inf'))
                if eval_best_lap_in_run < prev_det_best_lap:
                    safe_save(agent.actor.state_dict(), det_best_lap_pth)
                    safe_write_text(det_best_lap_txt, f"{eval_best_lap_in_run:.3f}")
                    msg = f"  NUOVO MIGLIOR GIRO VALIDO (eval deterministica): {eval_best_lap_in_run:.3f}s (preservato anche dopo --clean)"
                    print(msg)
                    with open(log_file, 'a', encoding='utf-8') as f: f.write(msg + "\n")

            # ── AUTO-REFINEMENT: macchina a stati (plateau → refine; collasso → rollback) ──
            # Le valutazioni fatte con Actor congelato servono solo a monitorare la policy corrente:
            # non devono alimentare il rilevamento plateau, altrimenti il trigger può partire subito
            # dopo lo scongelamento usando episodi raccolti mentre l'Actor non poteva migliorare.
            actor_is_frozen = getattr(agent, 'actor_frozen', False)
            if actor_is_frozen and not agent.refine_mode:
                refine_evals_no_improve = 0
                refine_best_mean = 0.0
                recent_eval_window.clear()
                _rlog("  Auto-refinement sospesa: Actor congelato; eval ignorato per il plateau.")
            else:
                recent_eval_window.append(eval_dist)

            if auto_refine_enabled and not agent.refine_mode and not actor_is_frozen:
                # Rilevamento PLATEAU su STATISTICA (non sul singolo best, robusto ai colpi di
                # fortuna): la MEDIA della finestra recente smette di salire. Serve la finestra piena.
                if len(recent_eval_window) >= REFINE_WINDOW:
                    cur_mean = sum(recent_eval_window) / len(recent_eval_window)
                    if cur_mean > refine_best_mean * REFINE_IMPROVE_FRAC:
                        refine_best_mean = cur_mean          # la performance tipica sta ancora salendo
                        refine_evals_no_improve = 0
                    else:
                        refine_evals_no_improve += 1          # tipica ferma → conta verso il plateau
                    if (refine_evals_no_improve >= REFINE_PLATEAU_EVALS and (episode + 1) >= REFINE_MIN_EP
                            and refine_attempts < REFINE_MAX_ATTEMPTS):
                        agent.refine_mode = True
                        agent.refine_bc_weight = REFINE_BC_WEIGHT
                        # Riferimento rollback = MEDIANA recente (modo 'buono' del bimodale), non il singolo max stocastico
                        refine_plateau_level = float(np.median(list(recent_eval_window)))
                        refine_collapse_count = 0
                        refine_evals_count = 0
                        refine_breakout_logged = False
                        _rlog(f"  AUTO-REFINEMENT ATTIVA (tentativo {refine_attempts+1}/{REFINE_MAX_ATTEMPTS}): "
                              f"media recente in plateau a {cur_mean:.0f}m, aggiornamento Critic disattivato, "
                              f"loss Critic solo diagnostica, peso Behavioral Cloning→{REFINE_BC_WEIGHT}, "
                              f"riferimento rollback (mediana)={refine_plateau_level:.0f}m "
                              f"(max recente={max(recent_eval_window):.0f}m)")
            elif agent.refine_mode and refine_plateau_level <= 0.0:
                # --refine: refinement già attiva, ma il riferimento rollback si fissa dopo i primi eval
                # (MEDIANA recente per robustezza), così la rete di sicurezza non usa un valore casuale/fortunato.
                if len(recent_eval_window) >= 4:
                    refine_plateau_level = float(np.median(list(recent_eval_window)))
                    refine_evals_count = 0
                    _rlog(f"  refinement: riferimento rollback = {refine_plateau_level:.0f}m "
                          f"(MEDIANA degli ultimi {len(recent_eval_window)} eval, max={max(recent_eval_window):.0f}m)")
            elif agent.refine_mode:
                # In REFINEMENT.
                refine_evals_count += 1
                # ① Avviso di RECUPERO: la refinement ha rotto il plateau (eval
                # oltre +10% del riferimento) → log una-tantum, è il segnale che sta funzionando.
                breakout_detected = eval_dist > refine_plateau_level * REFINE_BREAKOUT_FRAC
                if not refine_breakout_logged and breakout_detected:
                    refine_breakout_logged = True
                    _rlog(f"  PLATEAU SUPERATO: la refinement funziona! eval {eval_dist:.0f}m "
                          f"> riferimento {refine_plateau_level:.0f}m (+{(eval_dist/refine_plateau_level-1)*100:.0f}%)")
                # Se il breakout è già vicino al miglior deterministico assoluto, uscire subito
                # dalla refinement è più sicuro che lasciare il Critic spento: consolidiamo con
                # bc_weight=1.0 e, se configurato, congeliamo l'Actor per riallineare il Critic.
                if breakout_detected and prev_det_best_dist > 0.0 and eval_dist >= prev_det_best_dist - REFINE_NEAR_BEST_MARGIN:
                    agent.refine_mode = False
                    agent.refine_bc_weight = 1.0
                    agent.actor_frozen = actor_freeze_episodes > 0
                    start_episode = episode
                    refine_evals_no_improve = 0
                    refine_collapse_count = 0
                    refine_best_mean = 0.0
                    refine_plateau_level = 0.0
                    refine_evals_count = 0
                    recent_eval_window.clear()
                    _rlog(f"  REFINEMENT CONSOLIDATA: breakout vicino al miglior deterministico "
                          f"(eval {eval_dist:.0f}m, best {prev_det_best_dist:.0f}m). "
                          "Peso Behavioral Cloning→1.0, aggiornamento Critic riattivato.")
                    if actor_freeze_episodes > 0:
                        _rlog(f"  Rientro in modalità allineamento Critic: "
                              f"Actor congelato per {actor_freeze_episodes} episodi.")
                    else:
                        _rlog("  Rientro in training normale: congelamento Actor disattivato.")
                # ② Rete di sicurezza. Se l'eval crolla sotto il 60% del livello recente per 3
                # valutazioni consecutive → ROLLBACK al miglior deterministico (td3_det_best_dist) e training normale.
                elif eval_dist < refine_plateau_level * REFINE_COLLAPSE_FRAC:
                    refine_collapse_count += 1
                else:
                    refine_collapse_count = 0
                if refine_collapse_count >= 3:
                    ref_lvl = refine_plateau_level
                    if os.path.exists(det_best_dist_pth):
                        agent.actor.load_state_dict(torch.load(det_best_dist_pth, map_location=agent.device))
                        agent.actor_target.load_state_dict(agent.actor.state_dict())
                    agent.refine_mode = False
                    agent.refine_bc_weight = 1.0
                    refine_attempts += 1
                    refine_evals_no_improve = 0
                    refine_collapse_count = 0
                    refine_evals_count = 0
                    refine_best_mean = 0.0          # ricomincia a misurare il plateau da capo
                    refine_plateau_level = 0.0
                    _rlog(f"  REFINEMENT collassata (<{int(REFINE_COLLAPSE_FRAC*100)}% di {ref_lvl:.0f}m) "
                          f"→ ROLLBACK al miglior deterministico (td3_det_best_dist), peso Behavioral Cloning→1.0, aggiornamento Critic riattivato. Tentativi: {refine_attempts}/{REFINE_MAX_ATTEMPTS}")
                # ③ Timeout del refinement: 40 episodi (8 valutazioni) senza nuovi record.
                elif refine_evals_count >= 8:
                    agent.refine_mode = False
                    agent.refine_bc_weight = 1.0
                    refine_attempts += 1
                    refine_evals_no_improve = 0
                    refine_collapse_count = 0
                    refine_evals_count = 0
                    refine_best_mean = 0.0
                    refine_plateau_level = 0.0
                    _rlog(f"  TIMEOUT REFINEMENT (40 episodi in refinement senza superare il record) "
                          f"→ Uscita automatica, peso Behavioral Cloning→1.0, aggiornamento Critic riattivato. Tentativi: {refine_attempts}/{REFINE_MAX_ATTEMPTS}")

    env.end()

if __name__ == '__main__':
    train()
