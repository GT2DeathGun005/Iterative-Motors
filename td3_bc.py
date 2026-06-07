"""
TD3+BC Fine-Tuning — Twin Delayed DDPG con Behavioral Cloning

Architettura Ibrida BC-RL per TORCS (Offline-to-Online), allineata al TD3+BC minimalista
(Fujimoto & Gu, 2021):
  - L'Actor eredita backbone + gear_head dal BC (warm-start). Il TD3 allena TUTTO l'Actor
    (backbone + continuous_head); resta congelata solo la gear_head (marcia discreta).
  - Il Critic (Twin Q-Network) è addestrato da zero.
  - BC weight COSTANTE = 1.0 → loss = -λ·Q + (π - a)²  (λ = 2.5 / mean|Q|).
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
import time
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
                
        print(f"  📥 [EXPERT INJECTION] Caricati {loaded} campioni esperti nel Replay Buffer.")

    def load(self, filepath: str):
        if not os.path.exists(filepath): return
        data = np.load(filepath)
        states, actions, rewards, next_states, dones = data['states'], data['actions'], data['rewards'], data['next_states'], data['dones']
        expert_masks_data = data.get('expert_masks', np.zeros(len(states)))
        for i in range(len(states)):
            self.buffer.append((states[i], actions[i], float(rewards[i]), next_states[i], float(dones[i])))
            self.expert_masks.append(float(expert_masks_data[i]))
        print(f"  📦 Replay Buffer caricato: {len(self.buffer)} transizioni")

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
        print(f"⚠️  flatten_state fallita (stato a zero): {e}")
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
        print(f"✅ Pesi BC caricati con successo da {bc_path} (compensato scaling 0.5 per accel/brake).")

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
        
        # Rimuoviamo il Frozen BC Anchor: il target d'imitazione sarà
        # solo l'azione empirica (expert_mask=1.0) e non la predizione OOD.
        
        self.critic = Critic().to(self.device)
        self.critic_target = Critic().to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Backbone SCONGELATO (TIER 3b): come in TD3+BC originale si allena tutta la rete
        # dell'Actor. È sicuro perché l'ancora BC è forte e costante (bc_weight=1.0, niente decay),
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

        # ── Critic Update (Bellman Equation con Twin Q-Network) ──
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

        # In REFINEMENT (Paper 2) il Critic è CONGELATO: l'Actor si raffina verso una value
        # function fissa con vincolo BC ridotto. critic_loss resta calcolata solo per logging.
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

            # ── BC weight COSTANTE = 1.0 (niente decay) ──
            # Questo riproduce ESATTAMENTE la loss del TD3+BC originale:
            #   L = -λ·Q + (π - a)²  (qui bc_penalty è la nostra MSE pesata).
            # Il decay precedente (1.0→0.5) indeboliva la BC nella fase fragile post-warm-up
            # → "troppo RL troppo presto" → collasso (Beeson & Montana 2022, Ablation 1;
            # Fujimoto & Gu 2021, ablation su α). L'eventuale rilassamento del vincolo va fatto
            # in una FASE separata dopo il training stabile, con il Critic congelato (vedi
            # AUTO-REFINEMENT nel loop di train): lì bc_weight scende a refine_bc_weight.
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
        torch.save(checkpoint, filepath)
        buffer_dir = os.path.join(os.path.dirname(filepath), 'buffers')
        os.makedirs(buffer_dir, exist_ok=True)
        base_name = os.path.basename(filepath).replace('.pth', '')
        memory.save(os.path.join(buffer_dir, f"{base_name}_buffer.npz"))
        if elite_memory: elite_memory.save(os.path.join(buffer_dir, f"{base_name}_elite_buffer.npz"))

    def load_checkpoint(self, filepath, memory, elite_memory=None):
        buffer_dir = os.path.join(os.path.dirname(filepath), 'buffers')
        base_name = os.path.basename(filepath).replace('.pth', '')
        if os.path.exists(os.path.join(buffer_dir, f"{base_name}_buffer.npz")):
            memory.load(os.path.join(buffer_dir, f"{base_name}_buffer.npz"))
        if elite_memory and os.path.exists(os.path.join(buffer_dir, f"{base_name}_elite_buffer.npz")):
            elite_memory.load(os.path.join(buffer_dir, f"{base_name}_elite_buffer.npz"))

        if not os.path.exists(filepath): return 0, 0, float('inf'), 0.0, 0.0

        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and 'actor' in checkpoint:
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
            print(f"✅ Checkpoint caricato: ripresa dall'Episodio {episode}")
        else:
            # È un file di soli pesi dell'actor (come td3_best_dist.pth)
            print("ℹ️ Checkpoint contiene solo pesi dell'Actor (formato weights-only). Inizializzazione degli altri componenti.")
            self.actor.load_state_dict(checkpoint)
            self.actor_target.load_state_dict(self.actor.state_dict())
            best_lap_time = float('inf')
            best_eval_dist = 0.0
            best_distance = 0.0
            episode = 0
            global_step = 0

        # NB: nessun fallback hardcoded sui record storici. Valori hardcoded (es. 84.3s/3619m
        # di una run specifica) corrompevano l'Elite Buffer su un resume weights-only:
        # best_distance alto → elite_threshold = best_distance*0.9 si alza subito e il buffer
        # non si riempie più (Self-Imitation spento). Su weights-only i record ripartono puliti.
        return episode, global_step, best_lap_time, best_eval_dist, best_distance

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bc_weights', type=str, default='train_set/checkpoints/bc_policy.pth')
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--max_steps', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--rollback', action='store_true', help="Forza il rollback dell'Actor all'ultimo miglior giro storico e lo congela per 10 episodi")
    parser.add_argument('--refine', action='store_true', help="Avvia subito in modalità REFINEMENT (Critic congelato + bc_weight ridotto): usare in resume quando il training è già in plateau stabile")
    args = parser.parse_args()

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
    #  AUTO-REFINEMENT (Beeson & Montana 2022): dopo un PLATEAU stabile, congela il
    #  Critic e riduce il vincolo BC per spingere la policy deterministica oltre il muro.
    #  Conservativo (refine troppo presto → collasso, Ablation 1) + rete di sicurezza:
    #  se l'eval crolla, rollback automatico al best_ever (mai perso su disco).
    # ─────────────────────────────────────────────────────────────────────────
    REFINE_BC_WEIGHT = 0.3         # vincolo BC ridotto durante la refinement
    REFINE_MAX_ATTEMPTS = 3        # oltre, resta in training normale (no loop)
    REFINE_COLLAPSE_FRAC = 0.6     # eval < 60% del LIVELLO RECENTE per 3 volte → rollback
    REFINE_WINDOW = 8              # ampiezza finestra eval per MEDIA/MAX recenti (statistica robusta)
    REFINE_PLATEAU_EVALS = 12      # valutazioni con MEDIA recente NON in salita → plateau (auto)
    REFINE_MIN_EP = 200            # episodio minimo per l'auto-trigger
    REFINE_IMPROVE_FRAC = 1.02     # la media deve salire >2% per contare come "miglioramento"
    agent.refine_mode = False
    agent.refine_bc_weight = 1.0
    recent_eval_window = deque(maxlen=REFINE_WINDOW)  # ultimi eval → media/max recenti
    refine_best_mean = 0.0         # miglior MEDIA-finestra vista (segnale di plateau)
    refine_evals_no_improve = 0
    refine_attempts = 0
    refine_plateau_level = 0.0     # 0 = riferimento rollback non ancora impostato
    refine_collapse_count = 0
    refine_breakout_logged = False  # per loggare UNA volta il superamento del plateau
    if getattr(args, 'refine', False):
        # Avvio mirato quando l'operatore SA già che è in plateau: refinement ON da subito.
        agent.refine_mode = True
        agent.refine_bc_weight = REFINE_BC_WEIGHT
        print(f"🔧 --refine attivo: refinement ON da subito (Critic congelato, bc_weight={REFINE_BC_WEIGHT}). "
              f"Riferimento rollback impostato dopo i primi eval (MAX recente).")

    # Warm-Start: se è il primo avvio (nessun checkpoint), inizializza l'Actor con i pesi BC.
    if start_episode == 0:
        agent.actor.load_bc_weights(args.bc_weights)
        agent.actor_target.load_state_dict(agent.actor.state_dict())
    else:
        # Rollback Actor: solo se esplicitamente richiesto da riga di comando
        if args.rollback:
            best_lap_path = 'train_set/checkpoints/td3_best_lap.pth'
            if os.path.exists(best_lap_path):
                print(f"♻️  Rollback Actor: caricamento dei pesi del miglior giro storico da {best_lap_path}")
                agent.actor.load_state_dict(torch.load(best_lap_path, map_location=agent.device))
                agent.actor_target.load_state_dict(agent.actor.state_dict())
                import torch.optim as optim
                agent.actor_optimizer = optim.Adam(
                    [p for p in agent.actor.parameters() if p.requires_grad], lr=3e-4)
                # Attiviamo il congelamento temporaneo dell'Actor post-rollback
                agent.actor_frozen = True
                print("🧊 Actor congelato temporaneamente per stabilizzazione post-rollback.")
            else:
                print("⚠️  Rollback richiesto ma train_set/checkpoints/td3_best_lap.pth non trovato! Avvio ripresa normale.")
        else:
            print("▶️  Ripresa regolare dal checkpoint (nessun rollback o congelamento Actor).")

    os.makedirs('train_set/checkpoints', exist_ok=True)
    os.makedirs('train_set/session_logs', exist_ok=True)
    log_file = 'train_set/session_logs/td3_training.log'
    # Conferma nel LOG (non solo stdout) se la refinement è stata armata da --refine, così è
    # tracciabile a posteriori senza ambiguità con l'AUTO-REFINEMENT che scatta da solo.
    if getattr(args, 'refine', False):
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"🔧 AVVIO con --refine: REFINEMENT armata da subito "
                    f"(Critic congelato, bc_weight={REFINE_BC_WEIGHT}, ep iniziale {start_episode})\n")

    batch_size = 256
    elite_threshold = 500.0

    print("🚀 Avvio training TD3+BC...")

    for episode in range(start_episode, args.episodes):
        # Gestione dello scongelamento dell'Actor dopo la fase di stabilizzazione post-rollback
        if agent.actor_frozen and episode >= start_episode + 10:
            agent.actor_frozen = False
            print("🔥 Actor scongelato: riavvio aggiornamenti Actor con gradienti del Critic stabilizzati.")

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
            cont_action, raw_gear = agent.select_action(stacked_state, evaluate=False)

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
            # raw_gear (gear_head congelata) è ignorato. Vedi gearing.py.
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
                    torch.save(agent.actor.state_dict(), 'train_set/checkpoints/td3_best_lap.pth')
                
            if info.get('crash', False):
                done, termination_reason = True, "CRASH"


            if max_dist > best_distance and max_dist > 500.0:
                best_distance = max_dist
                torch.save(agent.actor.state_dict(), 'train_set/checkpoints/td3_best_dist.pth')

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
        log_msg = (f"[{time_str}] Episode {episode+1:03d} | [{termination_reason}] | "
                   f"Reward: {episode_reward:7.1f} | Steps: {step:4d} | "
                   f"Time: {lap_time:5.1f}s | Dist: {int(max_dist):5d}m | "
                   f"CriticL: {avg_critic_loss:.3f} | ActorL: {avg_actor_loss:.3f}")
        if new_record: log_msg += f" | 🏆 NEW RECORD"
        print(f"🏁 {log_msg}")

        with open(log_file, 'a', encoding='utf-8') as f: f.write(log_msg + "\n")

        agent.save_checkpoint(checkpoint_path, episode + 1, global_step, memory, elite_memory,
                              best_lap_time=best_lap_time,
                              best_eval_dist=best_eval_dist,
                              best_distance=best_distance)
        torch.save(agent.actor.state_dict(), 'train_set/checkpoints/td3_policy.pth')

        if (episode + 1) % 5 == 0 and global_step > 15000:
            print(f"\n  🔍 [EVAL] Valutazione deterministica...")
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
                    eval_action, eval_gear = agent.select_action(eval_stacked, evaluate=True)
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

            eval_msg = f"[{time_str}] 🔍 [EVAL] Result: Dist {int(eval_dist)}m | Reward: {eval_reward:.1f}"
            print(f"  {eval_msg}")
            with open(log_file, 'a', encoding='utf-8') as f: f.write(eval_msg + "\n")
            
            if eval_dist > best_eval_dist:
                best_eval_dist = eval_dist
                torch.save(agent.actor.state_dict(), 'train_set/checkpoints/td3_best_eval.pth')

            # ── Best-Ever (sopravvive a --clean) ──
            # td3_best_eval.pth viene cancellato da --clean. Per non perdere MAI la migliore
            # policy raggiunta tra run diversi, manteniamo td3_best_ever.pth + un sidecar .txt
            # con la sua distanza. train_rl.sh --clean NON cancella questi due file.
            best_ever_pth = 'train_set/checkpoints/td3_best_ever.pth'
            best_ever_txt = 'train_set/checkpoints/td3_best_ever.txt'
            prev_best_ever = 0.0
            if os.path.exists(best_ever_txt):
                try:
                    with open(best_ever_txt) as f: prev_best_ever = float(f.read().strip())
                except Exception: pass
            if eval_dist > prev_best_ever:
                torch.save(agent.actor.state_dict(), best_ever_pth)
                with open(best_ever_txt, 'w') as f: f.write(f"{eval_dist:.2f}")
                msg = f"  🏅 NUOVO BEST-EVER: {int(eval_dist)}m (preservato anche dopo --clean)"
                print(msg)
                with open(log_file, 'a', encoding='utf-8') as f: f.write(msg + "\n")

            # ── Best-Eval-LapTime: miglior GIRO VALIDO deterministico (candidato submission) ──
            # A differenza di best_ever (basato sulla DISTANZA), questo cattura il GIRO VALIDO più
            # VELOCE chiuso in eval deterministica: esattamente la policy da sottomettere. Sidecar
            # .txt col tempo; train_rl.sh --clean NON lo cancella (preservato tra run).
            if eval_best_lap_in_run < float('inf'):
                best_lt_pth = 'train_set/checkpoints/td3_best_eval_laptime.pth'
                best_lt_txt = 'train_set/checkpoints/td3_best_eval_laptime.txt'
                prev_best_lt = float('inf')
                if os.path.exists(best_lt_txt):
                    try:
                        with open(best_lt_txt) as f: prev_best_lt = float(f.read().strip())
                    except Exception: pass
                if eval_best_lap_in_run < prev_best_lt:
                    torch.save(agent.actor.state_dict(), best_lt_pth)
                    with open(best_lt_txt, 'w') as f: f.write(f"{eval_best_lap_in_run:.3f}")
                    msg = f"  🏆 NUOVO MIGLIOR GIRO VALIDO (eval deterministica): {eval_best_lap_in_run:.3f}s (preservato anche dopo --clean)"
                    print(msg)
                    with open(log_file, 'a', encoding='utf-8') as f: f.write(msg + "\n")

            # ── AUTO-REFINEMENT: macchina a stati (plateau → refine; collasso → rollback) ──
            def _rlog(m):
                print(m)
                with open(log_file, 'a', encoding='utf-8') as f: f.write(m + "\n")

            recent_eval_window.append(eval_dist)
            if not agent.refine_mode:
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
                        # Riferimento rollback = MAX recente (modo "buono" del bimodale), non il best storico
                        refine_plateau_level = max(recent_eval_window)
                        refine_collapse_count = 0
                        refine_breakout_logged = False
                        _rlog(f"  🔧 AUTO-REFINEMENT ON (tentativo {refine_attempts+1}/{REFINE_MAX_ATTEMPTS}): "
                              f"media recente in plateau a {cur_mean:.0f}m, Critic congelato, "
                              f"bc_weight→{REFINE_BC_WEIGHT}, riferimento={refine_plateau_level:.0f}m")
            elif refine_plateau_level <= 0.0:
                # --refine: refinement già ON, ma il riferimento rollback si fissa dopo i primi eval
                # (MAX recente), così la rete di sicurezza non usa un valore casuale/fortunato.
                if len(recent_eval_window) >= 3:
                    refine_plateau_level = max(recent_eval_window)
                    _rlog(f"  🔧 refinement: riferimento rollback = {refine_plateau_level:.0f}m (MAX recente)")
            else:
                # In REFINEMENT. ① Avviso di RECUPERO: la refinement ha rotto il plateau (eval
                # oltre +10% del riferimento) → log una-tantum, è il segnale che sta funzionando.
                if not refine_breakout_logged and eval_dist > refine_plateau_level * 1.1:
                    refine_breakout_logged = True
                    _rlog(f"  🚀 PLATEAU SUPERATO: la refinement funziona! eval {eval_dist:.0f}m "
                          f"> riferimento {refine_plateau_level:.0f}m (+{(eval_dist/refine_plateau_level-1)*100:.0f}%)")
                # ② Rete di sicurezza. Se l'eval crolla sotto il 60% del livello recente per 3
                # valutazioni consecutive → ROLLBACK al best_ever e training normale.
                if eval_dist < refine_plateau_level * REFINE_COLLAPSE_FRAC:
                    refine_collapse_count += 1
                else:
                    refine_collapse_count = 0
                if refine_collapse_count >= 3:
                    ref_lvl = refine_plateau_level
                    if os.path.exists(best_ever_pth):
                        agent.actor.load_state_dict(torch.load(best_ever_pth, map_location=agent.device))
                        agent.actor_target.load_state_dict(agent.actor.state_dict())
                    agent.refine_mode = False
                    agent.refine_bc_weight = 1.0
                    refine_attempts += 1
                    refine_evals_no_improve = 0
                    refine_collapse_count = 0
                    refine_best_mean = 0.0          # ricomincia a misurare il plateau da capo
                    refine_plateau_level = 0.0
                    _rlog(f"  🛡️ REFINEMENT collassata (<{int(REFINE_COLLAPSE_FRAC*100)}% di {ref_lvl:.0f}m) "
                          f"→ ROLLBACK al best_ever, bc_weight→1.0, Critic scongelato. Tentativi: {refine_attempts}/{REFINE_MAX_ATTEMPTS}")

    env.end()

if __name__ == '__main__':
    train()
