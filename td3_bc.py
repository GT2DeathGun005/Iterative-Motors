"""
TD3+BC Fine-Tuning — Twin Delayed DDPG con Behavioral Cloning

Architettura Ibrida BC-RL per TORCS (Offline-to-Online):
  - L'Actor eredita backbone + gear_head dal BC (congelati via Gradient Freezing)
  - Il TD3 aggiorna SOLO continuous_head
  - Il Critic (Twin Q-Network) è addestrato da zero
  - Delayed Policy Update: L'Actor e le reti target vengono aggiornati ogni 2 step del Critic.
  - Target Policy Smoothing: Rumore gaussiano clippato aggiunto alle azioni target.
  - BC Penalty: L'Actor massimizza il Q-Value restando ancorato ai dati esperti.

Retro-compatibilità:
  - La 'log_std_head' viene mantenuta nell'Actor unicamente per permettere a
    'test_agent.py' di caricare i vecchi checkpoint SAC senza crash, ma i suoi
    gradienti e output sono isolati e inutilizzati nel TD3.

Reward Reshaping:
  - progress = (speedX/50.0) * cos(angle)
  - Penalità terminali (schianto, stallo, fuoripista): -10.0
  - Bonus completamento giro: +50.0
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
                    states_np = h5f['states'][:]
                    actions_np = h5f['actions'][:]
                    
                length = len(states_np)
                k = 6
                for i in range(length - 1):
                    if max_samples and loaded >= max_samples: break
                    idx_t6 = max(0, i - k)
                    idx_t12 = max(0, i - 2 * k)
                    next_i = i + 1
                    n_idx_t6 = max(0, next_i - k)
                    n_idx_t12 = max(0, next_i - 2 * k)
                    
                    stacked_state = np.concatenate([states_np[idx_t12], states_np[idx_t6], states_np[i]])
                    next_stacked_state = np.concatenate([states_np[n_idx_t12], states_np[n_idx_t6], states_np[next_i]])
                    
                    cont_action = actions_np[i, 0:3]
                    cont_action[1] = (cont_action[1] * 2.0) - 1.0
                    cont_action[2] = (cont_action[2] * 2.0) - 1.0
                    
                    speedX = states_np[i, 21] * 50.0
                    angle = states_np[i, 0]
                    trackPos = states_np[i, 20]
                    
                    progress = (speedX / 50.0) * np.cos(angle)
                    pos_penalty = -1.0 * (trackPos ** 2)
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
        return np.concatenate([
            [_s('angle')], _a('track', 19), [_s('trackPos'), _s('speedX'), _s('speedY'), _s('speedZ')],
            _a('wheelSpinVel', 4) / 100.0, [_s('rpm') / 10000.0]
        ]).astype(np.float32)
    except Exception:
        return np.zeros(29, dtype=np.float32)

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
        self.load_state_dict(bc_state, strict=False)
        print(f"✅ Pesi BC caricati con successo da {bc_path}.")

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
        self.gamma = 0.999
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

        # Gradient Freezing: congela backbone e gear_head per preservare
        # la conoscenza pregressa del BC. Solo continuous_head viene aggiornato.
        for param in self.actor.backbone.parameters(): param.requires_grad = False
        for param in self.actor.gear_head.parameters(): param.requires_grad = False

        self.actor_optimizer = optim.Adam(self.actor.continuous_head.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)

    def select_action(self, state, evaluate=False):
        state_t = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            cont_action, _, gear_idx = self.actor.sample(state_t, evaluate=evaluate)
        return cont_action.cpu().numpy()[0], gear_idx.cpu().item()

    def update(self, memory, elite_memory, batch_size, global_step):
        # ── Hybrid Sampling (75% Standard + 25% Elite) ──
        # Il buffer standard contiene tutte le esperienze (anche crash e run mediocri),
        # essenziale per insegnare all'agente a generalizzare in stati "sporchi".
        # L'Elite Buffer contiene solo i giri da record (Self-Imitation Learning),
        # che forzano l'Actor a imitare le proprie migliori performance.
        # Il rapporto 75/25 bilancia generalizzazione vs ottimizzazione.
        if len(elite_memory.buffer) >= 64:
            b1, b2 = int(batch_size * 0.75), batch_size - int(batch_size * 0.75)
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

        # Reward Scaling: comprime i Q-values per compensare l'orizzonte lungo
        # di gamma=0.999 (che decuplica la magnitudo dei Q rispetto a gamma=0.99).
        # Senza questo scaling, i gradienti del Critic esploderebbero.
        reward_scale = 0.002
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
        if global_step >= 15000 and global_step % self.policy_freq == 0:
            pi, _ = self.actor(state_b)
            q1_pi, _ = self.critic(state_b, pi)
            
            # Componente RL: l'Actor massimizza il Q-Value stimato dal Critic
            actor_loss_td3 = -q1_pi.mean()
            
            # ── BC Penalty (Masking Rigoroso: Solo su stati Expert) ──
            # Per i campioni con expert_mask=1.0 (elite buffer + expert data),
            # il target è l'azione empirica registrata nel buffer.
            # Per i campioni con expert_mask=0.0 (esplorazione online),
            # la BC Penalty è azzerata: l'agente è libero di imparare manovre
            # di recupero tramite il solo gradiente RL.
            # Questo risolve il bug OOD: la bc_policy congelata non viene più
            # consultata per stati fuori distribuzione.
            det_steer = pi[:, 0]
            det_accel = (pi[:, 1] + 1.0) / 2.0
            det_brake = (pi[:, 2] + 1.0) / 2.0
            target_steer = action_b[:, 0]
            target_accel = (action_b[:, 1] + 1.0) / 2.0
            target_brake = (action_b[:, 2] + 1.0) / 2.0

            # Loss calcolata senza reduction per applicare il masking individuale
            steer_loss = F.mse_loss(det_steer, target_steer, reduction='none')
            accel_loss = F.mse_loss(det_accel, target_accel, reduction='none')
            brake_loss = F.mse_loss(det_brake, target_brake, reduction='none')

            # Loss direzionale di shape (batch_size,)
            directional_loss = (steer_loss * 2.0 + accel_loss + brake_loss * 2.0) / 5.0
            mutual_exclusion_penalty = (det_accel * det_brake).mean()

            # MASKING: La mask (B, 1) viene appiattita a (B,) per il broadcasting.
            # Il BC viene calcolato SOLO per i campioni con expert_mask == 1.0
            expert_mask_flat = expert_mask_b.squeeze(1)
            bc_penalty = (directional_loss * expert_mask_flat).mean() + (mutual_exclusion_penalty * 0.1)

            # ── Relaxed Policy Constraint (Beeson & Montana, 2022) ──
            # Decadimento esponenziale di lambda dopo i 15k step di warm-up.
            # Transizione da lambda=2.5 a lambda=0.25 su un orizzonte di 100k step.
            if global_step <= 15000:
                lambda_val = 2.5
            else:
                progress = min(1.0, (global_step - 15000) / 100000.0)
                lambda_val = 2.5 * (0.1 ** progress)

            Q_abs_mean = q1_pi.abs().mean().detach().clamp(min=1e-5)
            dynamic_alpha = lambda_val / Q_abs_mean
            
            total_actor_loss = dynamic_alpha * actor_loss_td3 + bc_penalty

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

    def save_checkpoint(self, filepath, episode, global_step, memory, elite_memory=None):
        checkpoint = {
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'episode': episode,
            'global_step': global_step,
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

        if not os.path.exists(filepath): return 0, 0

        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint['actor'])
        if 'actor_target' in checkpoint: self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])

        print(f"✅ Checkpoint caricato: ripresa dall'Episodio {checkpoint['episode']}")
        return checkpoint['episode'], checkpoint['global_step']

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bc_weights', type=str, default='train_set/checkpoints/bc_policy.pth')
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--max_steps', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    env = TorcsEnv(vision=False, throttle=True, gear_change=True, early_termination=True)
    memory = ReplayBuffer(100000)
    elite_memory = ReplayBuffer(20000)

    # ── Inizializzazione Agent ──
    agent = TD3BCAgent()
    agent.actor.load_bc_weights(args.bc_weights)  # Carica il backbone dal modello BC
    checkpoint_path = 'train_set/checkpoints/td3_checkpoint.pth'
    start_episode, global_step = agent.load_checkpoint(checkpoint_path, memory, elite_memory)

    # Warm-Start: se è il primo avvio (nessun checkpoint), inizializza
    # l'Actor con i pesi BC e inietta 50k campioni esperti nel buffer.
    if start_episode == 0:
        agent.actor.load_bc_weights(args.bc_weights)
        agent.actor_target.load_state_dict(agent.actor.state_dict())
        memory.load_expert_data('train_set/laps', max_samples=50000)

    os.makedirs('train_set/checkpoints', exist_ok=True)
    os.makedirs('train_set/session_logs', exist_ok=True)
    log_file = 'train_set/session_logs/td3_training.log'

    batch_size = 256
    best_lap_time = float('inf')
    elite_threshold = 500.0
    best_eval_dist = 0.0
    best_distance = 0.0

    print("🚀 Avvio training TD3+BC...")

    for episode in range(start_episode, args.episodes):
        ob = env.reset(relaunch=True)
        episode_transitions = []

        # Frame Stacking (87D): concatena 3 frame temporalmente distanziati
        # (t-12, t-6, t) per dare alla rete una percezione della velocità
        # e dell'accelerazione senza doverle calcolare esplicitamente.
        f_state = flatten_state(ob)
        state_stack = deque([f_state]*13, maxlen=13)
        stacked_state = np.concatenate([state_stack[0], state_stack[6], state_stack[12]])

        episode_reward, step, max_dist = 0, 0, 0.0
        critic_loss_val, actor_loss_val = 0.0, 0.0
        termination_reason = "TIMEOUT"
        new_record = False

        while True:
            agent.actor.eval()
            cont_action, raw_gear = agent.select_action(stacked_state, evaluate=False)
            agent.actor.train()

            # Mappatura Action Space: l'Actor emette azioni in [-1,1] (spazio tanh),
            # ma TORCS si aspetta accel/brake in [0,1]. La conversione (x+1)/2
            # preserva la simmetria del tanh per la backpropagation.
            env_action = np.zeros(4)
            env_action[0:3] = cont_action
            env_action[3] = max(1, min(6, raw_gear))  # Marcia: clamp a [1,6]
            
            torcs_action = env_action.copy()
            torcs_action[1] = np.clip((torcs_action[1] + 1.0) / 2.0, 0.0, 1.0)  # accel: [-1,1] → [0,1]
            torcs_action[2] = np.clip((torcs_action[2] + 1.0) / 2.0, 0.0, 1.0)  # brake: [-1,1] → [0,1]
            
            next_ob, reward, env_done, info = env.step(torcs_action)
            next_f_state = flatten_state(next_ob)
            state_stack.append(next_f_state)

            current_dist = float(np.array(next_ob.get('distRaced', 0.0)).flat[0])
            last_lap_time = float(np.array(next_ob.get('lastLapTime', 0.0)).flat[0])
            max_dist = max(max_dist, current_dist)

            done = False
            if last_lap_time > 0.0 and step > 500:
                done, termination_reason = True, "SUCCESS"
                reward += 50.0
                if last_lap_time < best_lap_time:
                    best_lap_time = last_lap_time
                    new_record = True
                    torch.save(agent.actor.state_dict(), 'train_set/checkpoints/td3_best_policy.pth')
                
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

            # Update Frequency 1:4: un aggiornamento ogni 4 step di simulazione.
            # Riduce l'overfitting su transizioni correlate e stabilizza i gradienti.
            if len(memory) > batch_size and global_step % 4 == 0:
                critic_loss_val, actor_loss_val, _ = agent.update(memory, elite_memory, batch_size, global_step)

            if done or env_done or time_limit_reached:
                for t in episode_transitions:
                    memory.push(t[0], t[1], t[2], t[3], t[4], expert=0.0)
                
                # ── Elite Buffer Injection ──
                # Se la distanza supera la soglia (best_distance * 0.9),
                # l'episodio viene clonato nell'Elite Buffer per Self-Imitation.
                # Gli ultimi 50 step di un crash vengono flaggati expert=0.0
                # per evitare Causal Confusion (imitare azioni pre-schianto).
                if max_dist >= elite_threshold:
                    n_trans = len(episode_transitions)
                    for i, t in enumerate(episode_transitions):
                        is_danger = (termination_reason == "CRASH") and (i >= n_trans - 50)
                        elite_memory.push(t[0], t[1], t[2], t[3], t[4], expert=0.0 if is_danger else 1.0)
                    elite_threshold = max(500.0, best_distance * 0.9)  # Soglia monotonicamente crescente
                break

        lap_time = step * 0.02
        time_str = datetime.now().strftime("%H:%M:%S")
        log_msg = (f"[{time_str}] Episode {episode+1:03d} | [{termination_reason}] | "
                   f"Reward: {episode_reward:7.1f} | Steps: {step:4d} | "
                   f"Time: {lap_time:5.1f}s | Dist: {int(max_dist):5d}m | "
                   f"CriticL: {critic_loss_val:.3f} | ActorL: {actor_loss_val:.3f}")
        if new_record: log_msg += f" | 🏆 NEW RECORD"
        print(f"🏁 {log_msg}")

        with open(log_file, 'a', encoding='utf-8') as f: f.write(log_msg + "\n")

        agent.save_checkpoint(checkpoint_path, episode + 1, global_step, memory, elite_memory)
        torch.save(agent.actor.state_dict(), 'train_set/checkpoints/td3_policy.pth')

        if (episode + 1) % 5 == 0 and global_step > 15000:
            print(f"\n  🔍 [EVAL] Valutazione deterministica...")
            eval_ob = env.reset(relaunch=True)
            eval_stack = deque([flatten_state(eval_ob)]*13, maxlen=13)
            eval_stacked = np.concatenate([eval_stack[0], eval_stack[6], eval_stack[12]])
            eval_dist, eval_step, eval_reward = 0.0, 0, 0.0

            agent.actor.eval()
            while eval_step < args.max_steps:
                eval_step += 1
                with torch.no_grad():
                    eval_action, eval_gear = agent.select_action(eval_stacked, evaluate=True)
                eval_env = np.zeros(4)
                eval_env[0:3], eval_env[3] = eval_action, max(1, min(6, eval_gear))
                eval_env[1], eval_env[2] = np.clip((eval_env[1]+1)/2, 0, 1), np.clip((eval_env[2]+1)/2, 0, 1)

                eval_ob, eval_r, eval_done, eval_info = env.step(eval_env)
                eval_reward += eval_r
                eval_stack.append(flatten_state(eval_ob))
                eval_stacked = np.concatenate([eval_stack[0], eval_stack[6], eval_stack[12]])
                eval_dist = float(np.array(eval_ob.get('distRaced', 0.0)).flat[0])

                if eval_info.get('crash', False) or eval_done: break
            agent.actor.train()

            eval_msg = f"[{time_str}] 🔍 [EVAL] Result: Dist {int(eval_dist)}m | Reward: {eval_reward:.1f}"
            print(f"  {eval_msg}")
            with open(log_file, 'a', encoding='utf-8') as f: f.write(eval_msg + "\n")
            
            if eval_dist > best_eval_dist:
                best_eval_dist = eval_dist
                torch.save(agent.actor.state_dict(), 'train_set/checkpoints/td3_best_eval.pth')

    env.end()

if __name__ == '__main__':
    train()
