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
#  Replay Buffer
# ──────────────────────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

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
    """Converte l'output dell'Actor (Tanh [-1, 1] e Gear Idx [0-6]) nel formato TORCS.
    
    Ricostruisce ESATTAMENTE l'attivazione Sigmoid che il backbone BC si aspetta.
    """
    env_action = np.zeros(4, dtype=np.float32)
    env_action[0] = np.clip(cont_action[0], -1.0, 1.0)               # steer
    
    # Inverte il Tanh dell'Actor SAC per recuperare i raw logits
    u_accel = np.clip(cont_action[1], -0.9999, 0.9999)
    u_brake = np.clip(cont_action[2], -0.9999, 0.9999)
    
    x_accel = np.arctanh(u_accel)
    x_brake = np.arctanh(u_brake)
    
    # Applica il Sigmoid per un matching 1:1 con il BC
    accel = 1.0 / (1.0 + np.exp(-x_accel))
    brake = 1.0 / (1.0 + np.exp(-x_brake))
    
    # Mutual exclusion (come l'esperto umano e il test_agent)
    if brake > 0.05:
        accel = 0.0
        
    env_action[1] = np.clip(accel, 0.0, 1.0)
    env_action[2] = np.clip(brake, 0.0, 1.0)
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
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std, gear_logits

    def sample(self, state, evaluate=False):
        mean, log_std, gear_logits = self.forward(state)
        
        # Gear: Scelta puramente deterministica basata sui pesi BC
        gear_idx = torch.argmax(gear_logits, dim=-1)
        
        if evaluate:
            # Determinismo assoluto per le azioni continue
            action = torch.tanh(mean)
            return action, None, gear_idx
            
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        action = torch.tanh(x_t)
        
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
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
        nn.init.constant_(self.log_std_head.bias, -6.0)
        print(f"✅ Pesi BC caricati con successo da {bc_path}. Log_std inizializzato a -6.0.")

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
        self.gamma = 0.99
        self.tau = 0.005
        self.alpha = 0.02  # Entropia fissa

        self.actor = Actor().to(self.device)
        self.critic = Critic().to(self.device)
        self.critic_target = Critic().to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Gradient Freezing (Latent Shift prevention)
        for param in self.actor.backbone.parameters():
            param.requires_grad = False
        for param in self.actor.gear_head.parameters():
            param.requires_grad = False

        # Optimizer: aggiorna SOLO continuous_head e log_std_head
        actor_params = list(self.actor.continuous_head.parameters()) + list(self.actor.log_std_head.parameters())
        self.actor_optimizer = optim.Adam(actor_params, lr=3e-5)
        
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)

    def select_action(self, state, evaluate=False):
        state_t = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            cont_action, _, gear_idx = self.actor.sample(state_t, evaluate=evaluate)
        return cont_action.cpu().numpy()[0], gear_idx.cpu().item()

    def update(self, memory, batch_size, global_step):
        state_b, action_b, reward_b, next_state_b, mask_b = memory.sample(batch_size)
        
        state_b = torch.FloatTensor(state_b).to(self.device)
        next_state_b = torch.FloatTensor(next_state_b).to(self.device)
        action_b = torch.FloatTensor(action_b).to(self.device)
        reward_b = torch.FloatTensor(reward_b).to(self.device).unsqueeze(1)
        mask_b = torch.FloatTensor(mask_b).to(self.device).unsqueeze(1)

        # Critic Update
        with torch.no_grad():
            next_action, next_log_pi, _ = self.actor.sample(next_state_b)
            q1_next, q2_next = self.critic_target(next_state_b, next_action)
            min_q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_pi
            target_q = reward_b + mask_b * self.gamma * min_q_next

        q1, q2 = self.critic(state_b, action_b)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss_val = 0.0
        # Critic Warm-Up: Non aggiornare l'Actor per i primi 5000 step
        # Questo protegge i pesi pre-addestrati del BC dai gradienti randomici del Critic non addestrato
        if global_step >= 5000:
            # Actor Update
            pi, log_pi, _ = self.actor.sample(state_b)
            q1_pi, q2_pi = self.critic(state_b, pi)
            min_q_pi = torch.min(q1_pi, q2_pi)
            actor_loss = (self.alpha * log_pi - min_q_pi).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            actor_loss_val = actor_loss.item()

        # Target Soft Update
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        return critic_loss.item(), actor_loss_val

# ──────────────────────────────────────────────────────────────────────
#  Reward Function e Loop
# ──────────────────────────────────────────────────────────────────────
def compute_reward(obs, prev_steer, cont_action, prev_damage):
    speed_x = float(np.array(obs.get('speedX', 0.0)).flat[0])
    angle = float(np.array(obs.get('angle', 0.0)).flat[0])
    track_pos = float(np.array(obs.get('trackPos', 0.0)).flat[0])
    damage = float(np.array(obs.get('damage', 0.0)).flat[0])
    steer = cont_action[0]

    # Penalità deterministica base
    progress = speed_x * np.cos(angle)
    angle_penalty = -2.0 * abs(angle)
    track_pos_penalty = -1.0 * (track_pos ** 2)
    
    # Penalità regolarizzante sullo sterzo (fluidità)
    steer_smoothness = -0.5 * abs(steer - prev_steer)
    
    # Penalità per collisione col muro o danno
    damage_penalty = 0.0
    if damage > prev_damage:
        damage_penalty = -50.0  # Punizione per impatto col muro

    reward = progress + angle_penalty + track_pos_penalty + steer_smoothness + damage_penalty
    
    done = False
    # Fuoripista critico / taglio curva estremo
    if track_pos > 1.50 or track_pos < -1.50:
        reward = -100.0
        done = True
        
    return reward, done, damage

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bc_weights', type=str, default='train_set/checkpoints/bc_policy.pth')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    
    # State Stacking (k=6, t-12, t-6, t) = 3x29 = 87
    env = TorcsEnv(vision=False, throttle=True, gear_change=True, early_termination=False)
    agent = SACAgent()
    agent.actor.load_bc_weights(args.bc_weights)
    memory = ReplayBuffer(capacity=100000)

    # Crea la directory di output
    os.makedirs('train_set/checkpoints', exist_ok=True)
    os.makedirs('train_set/session_logs', exist_ok=True)
    log_file = 'train_set/session_logs/sac_training.log'

    batch_size = 256
    global_step = 0

    print("🚀 Avvio training SAC (Warm-Start)...")
    
    for episode in range(args.episodes):
        # Relaunch=True garantisce azzeramento residui fisici
        ob = env.reset(relaunch=True)
        
        # Inizializza stack con maxlen=13 per replicare k=6 (t-12, t-6, t)
        f_state = flatten_state(ob)
        state_stack = deque([f_state]*13, maxlen=13)
        stacked_state = np.concatenate([state_stack[0], state_stack[6], state_stack[12]])
        
        episode_reward = 0
        step = 0
        prev_steer = 0.0
        prev_damage = 0.0
        prev_dist = 0.0
        current_gear = 1
        critic_loss_val = 0.0
        actor_loss_val = 0.0

        while True:
            # L'Actor DEVE essere in eval() durante l'inferenza anche in fase di esplorazione SAC
            # per disattivare eventuali dropout o BN (anche se non ci sono, è best practice).
            agent.actor.eval()
            cont_action, raw_gear = agent.select_action(stacked_state, evaluate=False)
            
            # Gear Hysteresis (Sequential Filter)
            if raw_gear > current_gear + 1:
                raw_gear = current_gear + 1
            elif raw_gear < current_gear - 1:
                raw_gear = current_gear - 1
            current_gear = max(1, raw_gear)
            
            agent.actor.train() # Riattiva train per i gradienti (su log_std_head e continuous_head)

            env_action = action_to_env(cont_action, current_gear)
            next_ob, _, env_done, _ = env.step(env_action)
            
            reward, done, current_damage = compute_reward(next_ob, prev_steer, cont_action, prev_damage)
            prev_steer = cont_action[0]
            prev_damage = current_damage
            
            next_f_state = flatten_state(next_ob)
            state_stack.append(next_f_state)
            
            # Controllo invalidazione giro (taglio curva o muro non rilevato)
            current_dist = float(np.array(next_ob.get('distFromStart', 0.0)).flat[0])
            last_lap_time = float(np.array(next_ob.get('lastLapTime', 0.0)).flat[0])
            
            # Se la distanza crolla improvvisamente (abbiamo tagliato il traguardo)
            # Aggiungiamo step > 500 per evitare che il passaggio della linea di partenza al via inneschi l'errore
            if prev_dist > 2500.0 and current_dist < 500.0 and step > 500:
                if last_lap_time <= 0.0:
                    # Giro invalidato dal simulatore (taglio curva o impatto)
                    reward -= 100.0
                    done = True
                else:
                    # Giro valido!
                    reward += 200.0
                    done = True
                    
            prev_dist = current_dist
            next_stacked_state = np.concatenate([state_stack[0], state_stack[6], state_stack[12]])

            # Ignora la termination artificiale se non è causata dal muro
            mask = 0.0 if done else 1.0
            memory.push(stacked_state, cont_action, reward, next_stacked_state, mask)

            stacked_state = next_stacked_state
            episode_reward += reward
            step += 1
            global_step += 1

            if len(memory) > batch_size:
                critic_loss_val, actor_loss_val = agent.update(memory, batch_size, global_step)

            if done or env_done or step >= 3000:
                break

        # Tempo stimato (50Hz = 0.02s per step)
        lap_time = step * 0.02
        log_msg = f"[{datetime.now().strftime('%H:%M:%S')}] Episode {episode+1:03d} | Reward: {episode_reward:7.1f} | Steps: {step:4d} | Time: {lap_time:5.1f}s"
        print(f"🏁 {log_msg}")

        # Salva log testuale semplice
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_msg + "\n")

        # Salva i pesi aggiornati
        torch.save(agent.actor.state_dict(), 'train_set/checkpoints/sac_policy.pth')

    env.end()

if __name__ == '__main__':
    train()
