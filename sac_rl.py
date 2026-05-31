"""
SAC Fine-Tuning — Soft Actor-Critic con Warm-Start da Behavioral Cloning

Architettura Ibrida BC-RL per TORCS:
  - L'Actor eredita backbone + gear_head dal BC (congelati via Gradient Freezing)
  - Il SAC aggiorna SOLO continuous_head e log_std_head
  - Il Critic (Twin Q-Network) è addestrato da zero
  - Critic Warm-Up: i primi 5000 step aggiornano solo il Critic
  - Fine-Tuning Conservativo: L'Actor usa un Learning Rate di 1e-6.
  - Entropia (Alpha): Si usa un Alpha fisso di 0.02 (innalzato per prevenire l'exploration dip). Rimosso l'Adaptive Alpha.
  - Tanh Explosion Prevention: Il `log_prob` è clippato matematicamente in [-20.0, 10.0] per evitare gradienti infiniti ai bordi della tanh.

Memory Safety (Gestione Memory Leak di TORCS):
  - Il noto memory leak del motore C++ di TORCS è bypassato forzando il kill/riavvio completo del processo server (`relaunch=True`) a ogni reset dell'episodio. La porta UDP viene chiusa e ricollegata per prevenire leak di rete.

Reward Reshaping (SAC-Compatible):
  - progress = (speedX/50.0) * cos(angle)
  - Dense Time Penalty e Steer Smoothness penalty per stabilizzare il veicolo
  - Tutte le penalità terminali (schianto, stallo, fuoripista) valgono -10.0

Replay Buffer:
  - Salvataggio su disco con np.savez_compressed (separato dal checkpoint PyTorch)
  - Previene catastrophic forgetting durante interruzioni

Done Masking:
  - done=True SOLO per schianti, fuoripista e spin
  - Il time-limit (max_steps) NON imposta done=True nel buffer
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
from torch.distributions import Normal
from collections import deque
from datetime import datetime

# Import gym_torcs
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'gym_torcs')))
try:
    from gym_torcs import TorcsEnv
except ImportError:
    print("Warning: gym_torcs non trovato.")

# ──────────────────────────────────────────────────────────────────────
#  Determinismo
# ──────────────────────────────────────────────────────────────────────
def set_seed(seed=42):
    """Garantisce il determinismo assoluto."""
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
#  Replay Buffer con Checkpointing su Disco
# ──────────────────────────────────────────────────────────────────────
class ReplayBuffer:
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
        """Salva il buffer su disco con compressione numpy."""
        if len(self.buffer) == 0:
            return
        states, actions, rewards, next_states, dones = zip(*self.buffer)
        np.savez_compressed(filepath,
            states=np.array(states, dtype=np.float32),
            actions=np.array(actions, dtype=np.float32),
            rewards=np.array(rewards, dtype=np.float32),
            next_states=np.array(next_states, dtype=np.float32),
            dones=np.array(dones, dtype=np.float32),
            expert_masks=np.array(list(self.expert_masks), dtype=np.float32))

    def load_expert_data(self, h5_dir_or_file: str, max_samples: int = None):
        """Carica dimostrazioni umane nel replay buffer per Expert Buffer Injection."""
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
                    states_np = h5f['states'][:]
                    actions_np = h5f['actions'][:] # steer, accel, brake, gear
                    
                length = len(states_np)
                k = 6
                for i in range(length - 1): # -1 per avere next_state
                    if max_samples and loaded >= max_samples: break
                    
                    idx_t6 = max(0, i - k)
                    idx_t12 = max(0, i - 2 * k)
                    
                    next_i = i + 1
                    n_idx_t6 = max(0, next_i - k)
                    n_idx_t12 = max(0, next_i - 2 * k)
                    
                    stacked_state = np.concatenate([states_np[idx_t12], states_np[idx_t6], states_np[i]])
                    next_stacked_state = np.concatenate([states_np[n_idx_t12], states_np[n_idx_t6], states_np[next_i]])
                    
                    cont_action = actions_np[i, 0:3]
                    # FIX ARCHITETTURALE: I dati BC hanno accel/brake in [0, 1].
                    # SAC usa tanh() per tutte le azioni in [-1, 1].
                    # Mappiamo accel/brake da [0, 1] a [-1, 1] per il ReplayBuffer.
                    cont_action[1] = (cont_action[1] * 2.0) - 1.0
                    cont_action[2] = (cont_action[2] * 2.0) - 1.0
                    
                    # Ricalcoliamo il reward con la nuova logica (Soft Shaping)
                    speedX = states_np[i, 21] * 50.0
                    angle = states_np[i, 0]
                    trackPos = states_np[i, 20]
                    
                    progress = (speedX / 50.0) * np.cos(angle)
                    pos_penalty = -1.0 * (trackPos ** 2)
                    steer_change = cont_action[0] - actions_np[i-1, 0] if i > 0 else 0.0
                        
                    reward = (progress * 1.5) + pos_penalty - (0.05 * abs(steer_change))
                    
                    done = (i == length - 2)
                    mask = 0.0 if done else 1.0
                    
                    self.push(stacked_state, cont_action, reward, next_stacked_state, mask, expert=1.0)
                    loaded += 1
            except Exception as e:
                print(f"Errore caricando {f} nel replay buffer: {e}")
                
        print(f"  📥 [EXPERT INJECTION] Caricati {loaded} campioni esperti nel Replay Buffer da {h5_dir_or_file}")

    def load(self, filepath: str):
        """Carica il buffer da disco."""
        if not os.path.exists(filepath):
            return
        data = np.load(filepath)
        states = data['states']
        actions = data['actions']
        rewards = data['rewards']
        next_states = data['next_states']
        dones = data['dones']
        expert_masks_data = data.get('expert_masks', np.zeros(len(states)))
        for i in range(len(states)):
            self.buffer.append((
                states[i], actions[i], float(rewards[i]),
                next_states[i], float(dones[i])
            ))
            self.expert_masks.append(float(expert_masks_data[i]))
        print(f"  📦 Replay Buffer caricato da disco: {len(self.buffer)} transizioni")

    def __len__(self):
        return len(self.buffer)

# ──────────────────────────────────────────────────────────────────────
#  State/Action Flattening
# ──────────────────────────────────────────────────────────────────────
def flatten_state(state_dict: dict) -> np.ndarray:
    """Appiattisce l'osservazione TORCS in un vettore 29D (come nel BC)."""
    def _s(key, default=0.0):
        v = state_dict.get(key, default)
        if isinstance(v, np.ndarray): return float(v.flat[0])
        return float(v) if v is not None else default

    def _a(key, size):
        v = state_dict.get(key, None)
        if v is None: return np.zeros(size, dtype=np.float32)
        return np.array(v, dtype=np.float32).flatten()[:size]

    try:
        return np.concatenate([
            [_s('angle')],
            _a('track', 19),
            [_s('trackPos'), _s('speedX'), _s('speedY'), _s('speedZ')],
            _a('wheelSpinVel', 4) / 100.0,
            [_s('rpm') / 10000.0]
        ]).astype(np.float32)
    except Exception:
        return np.zeros(29, dtype=np.float32)

def action_to_env(cont_action, gear_idx):
    """Converte l'output Tanh [-1, 1] dell'Actor SAC nel formato TORCS.

    Mappatura:
      - steer: [-1, 1] → [-1, 1]  (diretto, già Tanh)
      - accel: [-1, 1] → [0, 1]   (trasformazione affine (x+1)/2)
      - brake: [-1, 1] → [0, 1]   (trasformazione affine (x+1)/2)

    La trasformazione affine è stabile e priva degli asintoti dell'arctanh.
    """
    env_action = np.zeros(4, dtype=np.float32)
    env_action[0] = np.clip(cont_action[0], -1.0, 1.0)               # steer

    # Rimappatura affine: Tanh [-1, 1] → TORCS [0, 1]
    accel = np.clip((cont_action[1] + 1.0) / 2.0, 0.0, 1.0)
    brake = np.clip((cont_action[2] + 1.0) / 2.0, 0.0, 1.0)

    env_action[1] = accel
    env_action[2] = brake
    env_action[3] = float(max(1, min(6, gear_idx)))
    return env_action

# ──────────────────────────────────────────────────────────────────────
#  Architettura: Actor (ibrido BC) e Critic
# ──────────────────────────────────────────────────────────────────────
class Actor(nn.Module):
    def __init__(self, state_dim=87, hidden_size=512):
        super(Actor, self).__init__()
        # Backbone identico al Behavioral Cloning
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

        # Teste originali del BC
        self.continuous_head = nn.Linear(hidden_size, 3) # Steer, Accel, Brake
        self.gear_head = nn.Linear(hidden_size, 7)       # Cambio (discreto)

        # Nuova testa per la deviazione standard SAC
        self.log_std_head = nn.Linear(hidden_size, 3)

    def forward(self, state):
        features = self.backbone(state)
        mean = self.continuous_head(features)
        gear_logits = self.gear_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, min=-20.0, max=-2.0)
        return mean, log_std, gear_logits

    def sample(self, state, evaluate=False):
        mean, log_std, gear_logits = self.forward(state)

        # Gear: Scelta puramente deterministica basata sui pesi BC (congelati)
        gear_idx = torch.argmax(gear_logits, dim=-1)

        if evaluate:
            # Determinismo assoluto: restituisce tanh(mean), nessun campionamento
            action = torch.tanh(mean)
            return action, None, gear_idx

        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        action = torch.tanh(x_t)

        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        # Clamping log_prob per evitare l'esplosione numerica del tanh
        log_prob = torch.clamp(log_prob, min=-20.0, max=10.0)
        return action, log_prob, gear_idx

    def load_bc_weights(self, bc_path):
        """Carica backbone e teste dal BC e inizializza log_std a bassissima varianza."""
        if not os.path.exists(bc_path):
            print(f"⚠️ Checkpoint BC {bc_path} non trovato. Partenza da zero.")
            return

        bc_state = torch.load(bc_path, map_location='cpu', weights_only=True)
        self.load_state_dict(bc_state, strict=False)

        # Inizializza log_std per esplorazione infinitesimale (Warm-Start)
        nn.init.constant_(self.log_std_head.weight, 0.0)
        nn.init.constant_(self.log_std_head.bias, -3.0)
        print(f"✅ Pesi BC caricati con successo da {bc_path}. Log_std inizializzato a -3.0.")

class Critic(nn.Module):
    def __init__(self, state_dim=87, action_dim=3, hidden_size=512):
        super(Critic, self).__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        return self.q1(xu), self.q2(xu)

# ──────────────────────────────────────────────────────────────────────
#  SAC Agent
# ──────────────────────────────────────────────────────────────────────
class SACAgent:
    def __init__(self, device="cuda"):
        self.device = torch.device(device)
        # A 50Hz, gamma=0.99 dava un orizzonte di soli 2 secondi (100 step).
        # Aumentato a 0.999 per un orizzonte di 20 secondi (1000 step),
        # vitale per permettere al Critic di "vedere" la velocità oltre la linea di partenza.
        self.gamma = 0.999
        self.tau = 0.005

        self.actor = Actor().to(self.device)
        self.bc_policy = Actor().to(self.device)
        for p in self.bc_policy.parameters():
            p.requires_grad = False
        self.critic = Critic().to(self.device)
        self.critic_target = Critic().to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Gradient Freezing (Latent Shift prevention)
        for param in self.actor.backbone.parameters():
            param.requires_grad = False
        for param in self.actor.gear_head.parameters():
            param.requires_grad = False

        # Optimizer: aggiorna SOLO continuous_head e log_std_head
        # Actor Optimizer (Fase RL: solo continuous e log_std, backbone frozen)
        # 🛡️ TRUST REGION: LR microscopico (1e-5) per impedire il Policy Drift su stati OOD
        self.actor_optimizer = optim.Adam([
            {'params': self.actor.continuous_head.parameters()},
            {'params': self.actor.log_std_head.parameters()}
        ], lr=3e-4)

        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)

        # Entropy Auto-Tuning (Alpha)
        self.target_entropy = -3.0
        # Inizializziamo log_alpha a -3.91 (che corrisponde a un Alpha iniziale di ~0.02)
        init_log_alpha = -3.91
        self.log_alpha = torch.tensor([init_log_alpha], requires_grad=True, device=self.device)
        # LR standard (3e-4) per permettere ad Alpha di autoregolarsi correttamente
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)

    @property
    def alpha(self):
        # Restituisce l'alpha ottimizzato automaticamente per il bilanciamento esplorazione/exploit
        return self.log_alpha.exp().detach()

    def select_action(self, state, evaluate=False):
        state_t = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            cont_action, _, gear_idx = self.actor.sample(state_t, evaluate=evaluate)
        return cont_action.cpu().numpy()[0], gear_idx.cpu().item()

    def update(self, memory, elite_memory, batch_size, global_step):
        # ── Hybrid Sampling (Elite Buffer) ──
        if len(elite_memory.buffer) >= 64:
            b1 = int(batch_size * 0.75) # 192
            b2 = batch_size - b1        # 64
            
            s1, a1, r1, ns1, m1, em1 = memory.sample(b1)
            s2, a2, r2, ns2, m2, em2 = elite_memory.sample(b2)
            
            state_b = np.concatenate([s1, s2], axis=0)
            action_b = np.concatenate([a1, a2], axis=0)
            reward_b = np.concatenate([r1, r2], axis=0)
            next_state_b = np.concatenate([ns1, ns2], axis=0)
            mask_b = np.concatenate([m1, m2], axis=0)
            expert_mask_b = np.concatenate([em1, em2], axis=0)
        else:
            state_b, action_b, reward_b, next_state_b, mask_b, expert_mask_b = memory.sample(batch_size)

        # 🛡️ REWARD SCALING per prevenire il collasso dell'Actor
        # Gamma è 0.999 (orizzonte 1000 step), quindi i Q-Value sono 10 volte più grandi.
        # Riduciamo il reward_scale a 0.002 per mantenere bilanciati i gradienti del Critic
        # rispetto alla BC Penalty e all'Entropia.
        reward_scale = 0.002
        reward_b = reward_b * reward_scale

        state_b = torch.FloatTensor(state_b).to(self.device)
        next_state_b = torch.FloatTensor(next_state_b).to(self.device)
        action_b = torch.FloatTensor(action_b).to(self.device)
        reward_b = torch.FloatTensor(reward_b).to(self.device).unsqueeze(1)
        mask_b = torch.FloatTensor(mask_b).to(self.device).unsqueeze(1)
        expert_mask_b = torch.FloatTensor(expert_mask_b).to(self.device).unsqueeze(1)

        # Alpha calculation
        alpha = self.alpha

        # Critic Update
        with torch.no_grad():
            next_action, next_log_pi, _ = self.actor.sample(next_state_b)
            q1_next, q2_next = self.critic_target(next_state_b, next_action)
            min_q_next = torch.min(q1_next, q2_next) - alpha * next_log_pi
            target_q = reward_b + mask_b * self.gamma * min_q_next

        q1, q2 = self.critic(state_b, action_b)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        actor_loss_val = 0.0
        # Critic Warm-Up: Non aggiornare l'Actor per i primi 5000 step
        # Questo protegge i pesi pre-addestrati del BC dai gradienti randomici del Critic non addestrato
        if global_step >= 5000:
            # Actor Update
            pi, log_pi, _ = self.actor.sample(state_b)
            q1_pi, q2_pi = self.critic(state_b, pi)
            min_q_pi = torch.min(q1_pi, q2_pi)
            
            # L'Actor massimizza il Q-Value stimato e l'Entropia.
            actor_loss_sac = (alpha * log_pi - min_q_pi).mean()
            
            # --- PERMANENT BC ANCHOR & SELF-IMITATION ---
            # 1. Genera le azioni sicure dal modello BC umano (frozen)
            with torch.no_grad():
                bc_action, _, _ = self.bc_policy.sample(state_b, evaluate=True)

            # 2. Target ibrido:
            # - Se expert_mask_b == 1.0 (Elite), imita l'azione record (action_b)
            # - Se expert_mask_b == 0.0 (Standard/Schianto imminente), imita l'umano (bc_action)
            target_action = torch.where(expert_mask_b == 1.0, action_b, bc_action)

            # 3. La BC_Penalty si applica SEMPRE al 100% del batch.
            # Usiamo l'azione deterministica torch.tanh(mean) per isolare la varianza stocastica
            mean, _, _ = self.actor(state_b)
            deterministic_action = torch.tanh(mean)
            bc_penalty = F.mse_loss(deterministic_action, target_action)

            # Decay lineare del peso BC: Permanent BC Adherence (Residual RL)
            # Forza iniziale: 10.0 per proteggere i pesi BC dal Critic ancora acerbo.
            # Orizzonte lunghissimo: 500.000 step, per evitare Extrapolation Errors.
            # Hard Minimum: 2.0 (non arriva mai a zero) così l'Actor usa i Q-Value 
            # solo come correzioni locali (Residuals) della policy umana, senza sbandare.
            bc_weight = max(2.0, 10.0 * (1.0 - global_step / 500000.0))

            # Total Actor Loss
            total_actor_loss = actor_loss_sac + bc_weight * bc_penalty

            self.actor_optimizer.zero_grad()
            total_actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()
            actor_loss_val = total_actor_loss.item()

            # Alpha (Entropy) Update
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            # Evita che l'entropia crolli a zero: Alpha limitato a un minimo di ~0.007
            with torch.no_grad():
                self.log_alpha.clamp_(min=-5.0)

        # Target Soft Update
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        return critic_loss.item(), actor_loss_val, alpha

    def save_checkpoint(self, filepath, episode, global_step, memory, elite_memory=None):
        """Salva checkpoint PyTorch (reti + ottimizzatori) e buffer numpy separato."""
        checkpoint = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
            'alpha_optimizer': self.alpha_optimizer.state_dict(),
            'episode': episode,
            'global_step': global_step,
        }
        torch.save(checkpoint, filepath)

        # Salva i Replay Buffer separatamente nella sottocartella "buffers"
        buffer_dir = os.path.join(os.path.dirname(filepath), 'buffers')
        os.makedirs(buffer_dir, exist_ok=True)
        base_name = os.path.basename(filepath).replace('.pth', '')
        
        buffer_path = os.path.join(buffer_dir, f"{base_name}_buffer.npz")
        elite_buffer_path = os.path.join(buffer_dir, f"{base_name}_elite_buffer.npz")
        
        memory.save(buffer_path)
        if elite_memory:
            elite_memory.save(elite_buffer_path)

    def load_checkpoint(self, filepath, memory, elite_memory=None):
        # ── Carica i Replay Buffer INDIPENDENTEMENTE dal checkpoint di rete ──
        # In questo modo possiamo resettare i pesi della rete ma mantenere i record d'élite!
        buffer_dir = os.path.join(os.path.dirname(filepath), 'buffers')
        base_name = os.path.basename(filepath).replace('.pth', '')
        
        buffer_path = os.path.join(buffer_dir, f"{base_name}_buffer.npz")
        elite_buffer_path = os.path.join(buffer_dir, f"{base_name}_elite_buffer.npz")
        
        if os.path.exists(buffer_path):
            memory.load(buffer_path)
        if elite_memory and os.path.exists(elite_buffer_path):
            elite_memory.load(elite_buffer_path)

        if not os.path.exists(filepath):
            return 0, 0

        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])

        if 'log_alpha' in checkpoint:
            with torch.no_grad():
                self.log_alpha.copy_(checkpoint['log_alpha'])
            if 'alpha_optimizer' in checkpoint:
                self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])

        # Forza esplicitamente il Learning Rate a 3e-4 su tutti i param_groups
        # Altrimenti PyTorch ripristinerebbe il vecchio LR=1e-5 salvato nel file
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = 3e-4
        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] = 3e-4
        for param_group in self.alpha_optimizer.param_groups:
            param_group['lr'] = 3e-4

        # Carica i Replay Buffer dalla sottocartella "buffers"
        buffer_dir = os.path.join(os.path.dirname(filepath), 'buffers')
        base_name = os.path.basename(filepath).replace('.pth', '')
        
        buffer_path = os.path.join(buffer_dir, f"{base_name}_buffer.npz")
        elite_buffer_path = os.path.join(buffer_dir, f"{base_name}_elite_buffer.npz")
        
        memory.load(buffer_path)
        if elite_memory and os.path.exists(elite_buffer_path):
            elite_memory.load(elite_buffer_path)

        print(f"✅ Checkpoint caricato: ripresa dall'Episodio {checkpoint['episode']} "
              f"(Step {checkpoint['global_step']}). Buffer: {len(memory)} transizioni")
        return checkpoint['episode'], checkpoint['global_step']
def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bc_weights', type=str, default='train_set/checkpoints/bc_policy.pth')
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--max_steps', type=int, default=5000,
                        help='Max step per episodio (time-limit, non imposta done nel buffer)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    env = TorcsEnv(vision=False, throttle=True, gear_change=True, early_termination=True)
    print("  Inizializzazione Replay Buffer...")
    memory = ReplayBuffer(100000)
    elite_memory = ReplayBuffer(20000)

    agent = SACAgent()
    agent.bc_policy.load_bc_weights(args.bc_weights)
    checkpoint_path = 'train_set/checkpoints/sac_checkpoint.pth'
    start_episode, global_step = agent.load_checkpoint(checkpoint_path, memory, elite_memory)

    if start_episode == 0:
        agent.actor.load_bc_weights(args.bc_weights)
        print("  💉 Iniezione dell'Expert Buffer in corso...")
        memory.load_expert_data('train_set/laps', max_samples=50000)

    # Crea la directory di output
    os.makedirs('train_set/checkpoints', exist_ok=True)
    os.makedirs('train_set/session_logs', exist_ok=True)
    log_file = 'train_set/session_logs/sac_training.log'

    batch_size = 256
    if start_episode == 0:
        global_step = 0

    print("🚀 Avvio training SAC (Warm-Start)..." if start_episode == 0 else "🚀 Ripresa training SAC...")

    best_lap_time = float('inf')
    elite_threshold = 500.0

    for episode in range(start_episode, args.episodes):
        # Relaunch=True garantisce azzeramento residui fisici
        ob = env.reset(relaunch=True)

        episode_transitions = []

        # Inizializza stack con maxlen=13 per replicare k=6 (t-12, t-6, t)
        f_state = flatten_state(ob)
        state_stack = deque([f_state]*13, maxlen=13)
        stacked_state = np.concatenate([state_stack[0], state_stack[6], state_stack[12]])

        episode_reward = 0
        step = 0
        current_gear = 1
        critic_loss_val = 0.0
        actor_loss_val = 0.0
        current_alpha = 0.02
        prev_steer = 0.0
        max_dist = 0.0
        
        termination_reason = "TIMEOUT"
        new_record = False

        while True:
            agent.actor.eval()
            cont_action, raw_gear = agent.select_action(stacked_state, evaluate=False)

            # Nessun rumore manuale aggiunto. L'esplorazione e' gestita nativamente dal campionamento SAC.

            # Gestione marce semplificata
            current_gear = max(1, min(6, raw_gear))

            agent.actor.train()

            env_action = np.zeros(4)
            env_action[0:3] = cont_action
            env_action[3] = current_gear
            
            # FIX ARCHITETTURALE: L'Actor SAC lavora con azioni in [-1, 1].
            # TORCS richiede accel e brake in [0, 1].
            # Mappiamo le azioni prima di inviarle al simulatore.
            torcs_action = env_action.copy()
            torcs_action[1] = np.clip((torcs_action[1] + 1.0) / 2.0, 0.0, 1.0)
            torcs_action[2] = np.clip((torcs_action[2] + 1.0) / 2.0, 0.0, 1.0)
            
            # Affidiamoci alla reward di gym_torcs e al dict "info"
            next_ob, reward, env_done, info = env.step(torcs_action)
            
            next_f_state = flatten_state(next_ob)
            state_stack.append(next_f_state)

            current_dist = float(np.array(next_ob.get('distRaced', 0.0)).flat[0])
            last_lap_time = float(np.array(next_ob.get('lastLapTime', 0.0)).flat[0])
            max_dist = current_dist

            done = False
            # Check completamento giro
            if last_lap_time > 0.0 and step > 500:
                done = True  # L'episodio finisce perché hai vinto
                termination_reason = "SUCCESS"
                print(f"  🏎️  Giro completato: {last_lap_time:.2f}s!")
                if last_lap_time < best_lap_time:
                    best_lap_time = last_lap_time
                    new_record = True
                    torch.save(agent.actor.state_dict(), 'train_set/checkpoints/sac_best_policy.pth')
                
            # Anche uno schianto finisce l'episodio
            if info.get('crash', False):
                done = True
                termination_reason = "CRASH"

            # Salva i pesi per la distanza record
            global best_distance
            if 'best_distance' not in globals():
                best_distance = 0.0
                
            if max_dist > best_distance and max_dist > 500.0:
                best_distance = max_dist
                torch.save(agent.actor.state_dict(), 'train_set/checkpoints/sac_best_dist.pth')

            prev_dist = current_dist
            next_stacked_state = np.concatenate([state_stack[0], state_stack[6], state_stack[12]])

            # ── Done Masking Fix ──
            # mask=0.0 SOLO se ci siamo schiantati. Se scade il tempo o completiamo il giro, mask=1.0!
            mask = 0.0 if info.get('crash', False) else 1.0
            
            time_limit_reached = (step >= args.max_steps)
            episode_transitions.append((stacked_state, cont_action, reward, next_stacked_state, mask))

            stacked_state = next_stacked_state
            episode_reward += reward
            step += 1
            global_step += 1

            if len(memory) > batch_size:
                critic_loss_val, actor_loss_val, current_alpha = agent.update(memory, elite_memory, batch_size, global_step)

            if done or env_done or time_limit_reached:
                # ── Caching Episodico ──
                for t in episode_transitions:
                    memory.push(t[0], t[1], t[2], t[3], t[4], expert=0.0)
                
                # ── Elite Buffer Admission ──
                if max_dist >= elite_threshold:
                    n_transitions = len(episode_transitions)
                    danger_zone_steps = 50  # Sgancia il target d'élite 1 secondo prima del botto
                    for i, t in enumerate(episode_transitions):
                        is_danger_zone = (termination_reason == "CRASH") and (i >= n_transitions - danger_zone_steps)
                        exp_val = 0.0 if is_danger_zone else 1.0
                        elite_memory.push(t[0], t[1], t[2], t[3], t[4], expert=exp_val)
                    
                    # Manteniamo la soglia d'elite monotonicamente legata al record assoluto globale
                    elite_threshold = max(500.0, best_distance * 0.9)
                
                break

        # Tempo stimato (50Hz = 0.02s per step)
        lap_time = step * 0.02
        time_str = datetime.now().strftime("%H:%M:%S")
        log_msg = (f"[{time_str}] Episode {episode+1:03d} | [{termination_reason}] | "
                   f"Reward: {episode_reward:7.1f} | Steps: {step:4d} | "
                   f"Time: {lap_time:5.1f}s | Dist: {int(max_dist):5d}m | "
                   f"CriticL: {critic_loss_val:.3f} | ActorL: {actor_loss_val:.3f} | "
                   f"Alpha: {current_alpha:.3f}")
        
        if new_record:
            log_msg += f" | 🏆 NEW RECORD: {best_lap_time:.2f}s"
            
        print(f"🏁 {log_msg}")

        # Salva log testuale semplice
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_msg + "\n")

        # Salva i pesi aggiornati e il checkpoint integrale
        agent.save_checkpoint(checkpoint_path, episode + 1, global_step, memory, elite_memory)
        torch.save(agent.actor.state_dict(), 'train_set/checkpoints/sac_policy.pth')

    env.end()

if __name__ == '__main__':
    train()
