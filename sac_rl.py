"""
Soft Actor-Critic (SAC) con RLPD — Fine-Tuning RL per Giro Secco TORCS

Addestra un agente SAC pre-inizializzato con i pesi del Behavioral Cloning
per battere i tempi umani su singolo giro con partenza da fermo.

Features:
  - RLPD: replay buffer pre-riempito con 71k transizioni umane
  - Warm Start: carica backbone + mean_linear dal BC checkpoint
  - BC Regularization: previene catastrophic forgetting dei pesi BC
  - λ_bc decay: il vincolo BC si rilassa quando l'agente migliora
  - Actor LR separato (1e-5): aggiornamenti lenti per preservare BC
  - Reward dinamica: basata sui tempi umani reali
  - Terminazione: uscita pista, spin, stallo prolungato
  - Checkpoint periodici in train_set/checkpoints/
  - GPU-optimized (CUDA)

Action de-normalization (Tanh [-1,1] → env ranges):
  steer = action[0]                              # [-1, 1]
  accel = (action[1] + 1) / 2                    # [0, 1]
  brake = (action[2] + 1) / 2                    # [0, 1]
  gear  = round((action[3] + 1) * 3)             # [0, 6] discretizzato
"""

import os
import sys
import glob
import argparse
import random
import time
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from collections import deque
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'gym_torcs')))

try:
    from gym_torcs import TorcsEnv
except ImportError:
    print("Warning: gym_torcs non trovato.")

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPSILON = 1e-6


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


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
#  Actor (Policy Network)
# ──────────────────────────────────────────────────────────────────────

class Actor(nn.Module):
    """Rete Actor SAC con output Gaussiano (mean, log_std).

    Backbone identico alla PolicyNetwork del BC (layers 0-5 del Sequential):
      0: Linear → 1: LayerNorm → 2: ReLU → 3: Linear → 4: LayerNorm → 5: ReLU
    Poi si biforca in mean_linear e log_std_linear.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 256):
        super(Actor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),    # 0
            nn.LayerNorm(hidden_size),             # 1
            nn.ReLU(),                             # 2
            nn.Linear(hidden_size, hidden_size),   # 3
            nn.LayerNorm(hidden_size),             # 4
            nn.ReLU()                              # 5
        )

        self.mean_linear = nn.Linear(hidden_size, action_dim)
        self.log_std_linear = nn.Linear(hidden_size, action_dim)
        self.apply(weights_init_)

    def forward(self, state):
        x = self.net(state)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t

        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + EPSILON)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, torch.tanh(mean)

    def load_bc_weights(self, bc_model_path: str, device: str = "cpu"):
        """Carica i pesi pre-addestrati dal Behavioral Cloning (Warm Start).

        Mappa il backbone (net.0-5) e l'output layer (net.6 → mean_linear).
        Il log_std_linear viene inizializzato a bassa varianza.
        """
        if not os.path.exists(bc_model_path):
            print(f"  ⚠️  File BC '{bc_model_path}' non trovato. Actor parte da zero.")
            return False

        print(f"  Caricamento pesi BC da: {bc_model_path}")
        bc_state = torch.load(bc_model_path, map_location=device, weights_only=True)

        # Mappatura BC → SAC Actor
        mapping = {
            'net.0.weight': ('net.0.weight', 'Linear layer 1'),
            'net.0.bias':   ('net.0.bias',   None),
            'net.1.weight': ('net.1.weight', 'LayerNorm 1'),
            'net.1.bias':   ('net.1.bias',   None),
            'net.3.weight': ('net.3.weight', 'Linear layer 2'),
            'net.3.bias':   ('net.3.bias',   None),
            'net.4.weight': ('net.4.weight', 'LayerNorm 2'),
            'net.4.bias':   ('net.4.bias',   None),
            'net.6.weight': ('mean_linear.weight', 'Output → Mean'),
            'net.6.bias':   ('mean_linear.bias',   None),
        }

        loaded = 0
        with torch.no_grad():
            for bc_key, (sac_key, label) in mapping.items():
                if bc_key not in bc_state:
                    print(f"    ⚠️  Chiave BC '{bc_key}' non trovata, skip.")
                    continue

                # Naviga nella struttura dell'Actor per trovare il parametro
                parts = sac_key.split('.')
                param = self
                for p in parts[:-1]:
                    param = getattr(param, p) if not p.isdigit() else param[int(p)]
                target = getattr(param, parts[-1])

                if target.shape != bc_state[bc_key].shape:
                    print(f"    ⚠️  Shape mismatch {bc_key}: BC={bc_state[bc_key].shape} vs SAC={target.shape}")
                    continue

                target.copy_(bc_state[bc_key])
                loaded += 1
                if label:
                    print(f"    ✓ {label}: {bc_key} → {sac_key}")

            # Log_std inizializzato basso per sfruttare il prior BC
            nn.init.constant_(self.log_std_linear.weight, -2.0)
            nn.init.constant_(self.log_std_linear.bias, -2.0)

        print(f"  Warm Start completato: {loaded}/10 parametri caricati.")
        print(f"  Log_std inizializzato a -2.0 (bassa varianza iniziale).")
        return True


# ──────────────────────────────────────────────────────────────────────
#  Critic (Twin Q-Network)
# ──────────────────────────────────────────────────────────────────────

class Critic(nn.Module):
    """Twin Q-Network per Double Q-learning (mitiga sovrastima)."""

    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 256):
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
        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        return self.q1(xu), self.q2(xu)


# ──────────────────────────────────────────────────────────────────────
#  SAC Agent
# ──────────────────────────────────────────────────────────────────────

class SACAgent:
    def __init__(self, state_dim: int, action_dim: int,
                 device: str = "cpu", gamma: float = 0.99,
                 tau: float = 0.005, critic_lr: float = 3e-4,
                 actor_lr: float = 1e-5, bc_lambda: float = 1.0):
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.bc_lambda = bc_lambda  # Coefficiente regolarizzazione BC

        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # LR separato: actor lento (preserva BC), critic veloce
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Alpha fisso (NO auto-tuning con BC warm start)
        # L'auto-tuning standard forza alpha in alto perché la policy BC
        # è quasi deterministica, il che distrugge i pesi BC.
        self.alpha = 0.01  # Basso: poca esplorazione, preserva BC

        # BC model congelato come riferimento (caricato dopo)
        self.bc_model = None

    def load_bc_reference(self, bc_path: str):
        """Carica una copia congelata del BC model per la regularization."""
        from behavioral_cloning import PolicyNetwork
        self.bc_model = PolicyNetwork().to(self.device)
        self.bc_model.load_state_dict(
            torch.load(bc_path, map_location=self.device, weights_only=True)
        )
        self.bc_model.eval()
        for p in self.bc_model.parameters():
            p.requires_grad = False
        print(f"  ✅ BC reference model caricato e congelato per regularization.")

    def select_action(self, state, evaluate=False):
        state_t = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate:
            _, _, action = self.actor.sample(state_t)
        else:
            action, _, _ = self.actor.sample(state_t)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory: ReplayBuffer, batch_size: int):
        state_b, action_b, reward_b, next_state_b, mask_b = memory.sample(batch_size)

        state_b = torch.FloatTensor(state_b).to(self.device)
        next_state_b = torch.FloatTensor(next_state_b).to(self.device)
        action_b = torch.FloatTensor(action_b).to(self.device)
        reward_b = torch.FloatTensor(reward_b).to(self.device).unsqueeze(1)
        mask_b = torch.FloatTensor(mask_b).to(self.device).unsqueeze(1)

        # Critic update
        with torch.no_grad():
            next_action, next_log_pi, _ = self.actor.sample(next_state_b)
            q1_next, q2_next = self.critic_target(next_state_b, next_action)
            min_q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_pi
            next_q = reward_b + mask_b * self.gamma * min_q_next

        q1, q2 = self.critic(state_b, action_b)
        qf_loss = F.mse_loss(q1, next_q) + F.mse_loss(q2, next_q)

        self.critic_optimizer.zero_grad()
        qf_loss.backward()
        self.critic_optimizer.step()

        # Actor update (con BC regularization)
        pi, log_pi, _ = self.actor.sample(state_b)
        q1_pi, q2_pi = self.critic(state_b, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)
        sac_loss = ((self.alpha * log_pi) - min_q_pi).mean()

        # BC regularization: penalizza la distanza dalla policy BC
        bc_loss = torch.tensor(0.0, device=self.device)
        if self.bc_model is not None and self.bc_lambda > 0:
            with torch.no_grad():
                bc_actions = self.bc_model(state_b)  # output tanh [-1,1]
            # Confronta con l'output deterministico dell'actor (tanh(mean))
            _, _, actor_det = self.actor.sample(state_b)
            bc_loss = F.mse_loss(actor_det, bc_actions)

        policy_loss = sac_loss + self.bc_lambda * bc_loss

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # Soft update target
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        return qf_loss.item(), policy_loss.item(), 0.0


# ──────────────────────────────────────────────────────────────────────
#  State Flattening & Action De-normalization
# ──────────────────────────────────────────────────────────────────────

def flatten_state(state_dict: dict) -> np.ndarray:
    """Appiattisce osservazione TORCS → vettore 29D."""
    def _s(key, default=0.0):
        v = state_dict.get(key, default)
        if isinstance(v, np.ndarray):
            return float(v.flat[0])
        return float(v) if v is not None else default

    def _a(key, size):
        v = state_dict.get(key, None)
        if v is None:
            return np.zeros(size, dtype=np.float32)
        return np.array(v, dtype=np.float32).flatten()[:size]

    try:
        return np.concatenate([
            [_s('angle')],
            _a('track', 19),
            [_s('trackPos'), _s('speedX'), _s('speedY'), _s('speedZ')],
            _a('wheelSpinVel', 4) / 100.0,
            [_s('rpm') / 10000.0],
        ]).astype(np.float32)
    except Exception:
        return np.zeros(29, dtype=np.float32)


def denormalize_action(action: np.ndarray) -> np.ndarray:
    """Converte azione SAC (Tanh [-1,1]) in formato env TORCS."""
    env_action = np.zeros(4, dtype=np.float32)
    env_action[0] = np.clip(action[0], -1.0, 1.0)               # steer
    env_action[1] = np.clip((action[1] + 1.0) / 2.0, 0.0, 1.0)  # accel
    env_action[2] = np.clip((action[2] + 1.0) / 2.0, 0.0, 1.0)  # brake
    gear = int(round((action[3] + 1.0) * 3.0))                   # gear
    env_action[3] = float(max(0, min(6, gear)))
    return env_action


def normalize_action(env_action: np.ndarray) -> np.ndarray:
    """Converte azione env TORCS → formato SAC Tanh [-1,1] (inversa di denormalize)."""
    return np.array([
        env_action[0],                          # steer: già [-1,1]
        env_action[1] * 2.0 - 1.0,              # accel: [0,1] → [-1,1]
        env_action[2] * 2.0 - 1.0,              # brake: [0,1] → [-1,1]
        env_action[3] / 3.0 - 1.0,              # gear: [0,6] → [-1,1]
    ], dtype=np.float32)


def prefill_buffer_from_demos(memory: ReplayBuffer, demo_dir: str):
    """Carica le transizioni dalle demo umane nel replay buffer.

    Ogni coppia (state_t, action_t) → (state_t+1) diventa una transizione.
    Le azioni vengono normalizzate in formato SAC (tanh [-1,1]).
    La reward è un valore neutro-positivo (0.5) per indicare che le demo
    sono "buone" senza distorcere la scala della reward online.
    """
    h5_files = sorted(glob.glob(os.path.join(demo_dir, "lap_*.h5")))
    if not h5_files:
        print(f"  ⚠️  Nessun file demo trovato in {demo_dir}")
        return 0

    total = 0
    for h5_path in h5_files:
        with h5py.File(h5_path, 'r') as h5f:
            states = h5f['states'][:]
            actions = h5f['actions'][:]

        for i in range(len(states) - 1):
            norm_action = normalize_action(actions[i])
            memory.push(states[i], norm_action, 0.5, states[i + 1], 1.0)
            total += 1

    print(f"  ✅ Buffer pre-riempito con {total:,} transizioni da {len(h5_files)} giri demo")
    return total


# ──────────────────────────────────────────────────────────────────────
#  Reward Function — Giro Secco
# ──────────────────────────────────────────────────────────────────────

# ── Tempi umani dai session_logs (riferimenti per la reward) ──
HUMAN_BEST_TIME = 71.038    # Miglior giro umano (lap_017.h5)
HUMAN_WORST_TIME = 77.146   # Peggior giro umano (lap_001.h5)


def compute_step_reward(obs: dict, prev_dist: float, raw_obs: dict) -> tuple:
    """Reward per singolo step. Ritorna (reward, done, dist_raced).

    Componenti:
      + progress: avanzamento sulla pista (Δ distRaced)
      + speed:    bonus velocità longitudinale
      - center:   penalità quadratica per distanza dal centro
      - angle:    penalità per disallineamento
      - offtrack: terminazione + altissima penalità
      - spin:     terminazione se l'auto si gira (cos(angle) < 0)
    """
    speed_x = float(np.array(obs.get('speedX', 0.0)).flat[0])
    track_pos = float(np.array(obs.get('trackPos', 0.0)).flat[0])
    angle = float(np.array(obs.get('angle', 0.0)).flat[0])

    # Distanza percorsa (raw, non normalizzata)
    dist_raced = float(raw_obs.get('distRaced', 0.0))
    if isinstance(dist_raced, list):
        dist_raced = dist_raced[0]
    delta_dist = dist_raced - prev_dist

    # ── Componenti reward ──
    progress = delta_dist * 0.1
    speed_bonus = max(0, speed_x) * 0.005
    center_penalty = -2.0 * (track_pos ** 2)
    angle_penalty = -5.0 * abs(angle)

    reward = progress + speed_bonus + center_penalty + angle_penalty

    # ── Terminazione ──
    done = False

    # Uscita di pista — ALTISSIMA PENALITÀ
    if abs(track_pos) > 1.0:
        done = True
        reward = -500.0

    # Spin: l'auto si è girata
    if np.cos(angle) < 0:
        done = True
        reward = -500.0

    return reward, done, dist_raced


def compute_lap_bonus(lap_time: float, best_time: float) -> float:
    """Bonus/penalità per completamento giro basati sui tempi umani.

    Soglie (dai session_logs):
      Best umano:  71.038s (lap_017)
      Worst umano: 77.146s (lap_001)

    Reward:
      - Batte il best umano (< 71.038s):     +1000 base + 200/s di miglioramento
      - Tra best e worst umano:               +500 base
      - Poco più lento del worst (77-82s):    -50  (media penalità)
      - Molto più lento del worst (> 82s):    -100 (alta penalità)
    """
    if lap_time < HUMAN_BEST_TIME:
        # Premio cospicuo: ha battuto il miglior giro umano!
        improvement = HUMAN_BEST_TIME - lap_time
        return 1000.0 + improvement * 200.0
    elif lap_time <= HUMAN_WORST_TIME:
        # Giro nella fascia umana: buono, bonus base
        return 500.0
    elif lap_time <= 82.0:
        # Poco più lento del peggior giro umano: media penalità
        return -50.0
    else:
        # Molto più lento del peggior giro umano: alta penalità
        return -100.0


# ──────────────────────────────────────────────────────────────────────
#  Main Training Loop
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SAC RL Training — TORCS Giro Secco (RLPD)")
    parser.add_argument("--episodes", type=int, default=1000, help="Episodi di training")
    parser.add_argument("--max_steps", type=int, default=10000, help="Max step per episodio")
    parser.add_argument("--bc_weights", type=str, default="train_set/checkpoints/bc_policy.pth",
                        help="Path ai pesi BC per warm start")
    parser.add_argument("--demo_dir", type=str, default="train_set/laps",
                        help="Directory con i file HDF5 delle demo umane")
    parser.add_argument("--save_dir", type=str, default="train_set/checkpoints",
                        help="Directory per checkpoint")
    parser.add_argument("--target_time", type=float, default=75.0,
                        help="Tempo target umano in secondi (default: 75s)")
    parser.add_argument("--buffer_size", type=int, default=200000,
                        help="Capacità replay buffer")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--warmup_steps", type=int, default=5000,
                        help="Campioni nel buffer prima di iniziare gli update (default: 5000)")
    parser.add_argument("--actor_lr", type=float, default=1e-5,
                        help="Learning rate dell'actor (basso per preservare BC)")
    parser.add_argument("--critic_lr", type=float, default=3e-4,
                        help="Learning rate del critic")
    parser.add_argument("--bc_lambda", type=float, default=1.0,
                        help="Coefficiente regolarizzazione BC (0=disabilitato)")
    parser.add_argument("--relaunch_every", type=int, default=20,
                        help="Rilancia TORCS ogni N episodi")
    parser.add_argument("--checkpoint_every", type=int, default=50,
                        help="Salva checkpoint ogni N episodi")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state_dim = 29
    action_dim = 4

    os.makedirs(args.save_dir, exist_ok=True)

    print(f"\n{'=' * 64}")
    print(f"  🏎️  SAC REINFORCEMENT LEARNING (RLPD) — Giro Secco TORCS")
    print(f"  Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Episodi: {args.episodes} | Buffer: {args.buffer_size}")
    print(f"  Actor LR: {args.actor_lr} | Critic LR: {args.critic_lr}")
    print(f"  BC λ: {args.bc_lambda} | Tempo target: {args.target_time:.1f}s")
    print(f"{'=' * 64}\n")

    # ── Agent (con LR separati e BC lambda) ──
    agent = SACAgent(state_dim, action_dim, device,
                     actor_lr=args.actor_lr,
                     critic_lr=args.critic_lr,
                     bc_lambda=args.bc_lambda)

    # ── Warm Start ──
    agent.actor.load_bc_weights(args.bc_weights, device)

    # ── BC Reference Model (congelato, per regularization) ──
    if args.bc_lambda > 0:
        agent.load_bc_reference(args.bc_weights)

    # ── Replay Buffer ──
    memory = ReplayBuffer(capacity=args.buffer_size)

    # ── Pre-fill buffer con demo umane (RLPD) ──
    print(f"\n  Caricamento demo umane nel replay buffer...")
    prefill_buffer_from_demos(memory, args.demo_dir)

    # ── Best time tracking (reward dinamica) ──
    best_lap_time = args.target_time
    total_updates = 0

    # ── Training log ──
    log_dir = os.path.join(os.path.dirname(args.save_dir), "session_logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"sac_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    # ── Ambiente ──
    print("  Inizializzazione TORCS...")
    env = TorcsEnv(vision=False, throttle=True, gear_change=True, early_termination=False)

    try:
        for ep in range(1, args.episodes + 1):
            # ── Reset ──
            need_relaunch = (ep == 1) or (ep % args.relaunch_every == 0)
            if ep == 1:
                obs = env.reset(relaunch=True)
            else:
                obs = env.reset(relaunch=need_relaunch)

            state = flatten_state(obs)
            episode_reward = 0.0

            # Raw obs per distRaced e lapTime
            raw = env.client.S.d
            prev_dist = float(raw.get('distRaced', 0.0))
            if isinstance(prev_dist, list):
                prev_dist = prev_dist[0]
            prev_last_lap = float(raw.get('lastLapTime', 0.0))
            if isinstance(prev_last_lap, list):
                prev_last_lap = prev_last_lap[0]

            stall_counter = 0
            lap_completed = False
            ep_lap_time = 0.0

            for step in range(1, args.max_steps + 1):
                # ── Selezione azione ──
                # Con BC warm start, l'Actor produce già azioni ragionevoli.
                # L'esplorazione è garantita dalla distribuzione stocastica del SAC
                # (log_std). Il warmup serve solo a riempire il buffer prima
                # di iniziare gli update, NON per esplorare con azioni random.
                action = agent.select_action(state)

                # ── De-normalizza e step ──
                env_action = denormalize_action(action)
                next_obs, _, env_done, _ = env.step(env_action)
                next_state = flatten_state(next_obs)

                # ── Reward ──
                raw = env.client.S.d
                reward, custom_done, prev_dist = compute_step_reward(
                    next_obs, prev_dist, raw
                )

                # ── Stallo detection (dopo i primi 200 step) ──
                raw_speed = float(raw.get('speedX', 0.0))
                if isinstance(raw_speed, list):
                    raw_speed = raw_speed[0]
                if step > 200 and abs(raw_speed) < 20.0:
                    stall_counter += 1
                    if stall_counter > 50:
                        custom_done = True
                        reward = -500.0  # Stessa penalità dell'uscita pista
                else:
                    stall_counter = 0

                # ── Lap completion ──
                current_last_lap = float(raw.get('lastLapTime', 0.0))
                if isinstance(current_last_lap, list):
                    current_last_lap = current_last_lap[0]
                if current_last_lap > 0.0 and abs(current_last_lap - prev_last_lap) > 0.01:
                    lap_completed = True
                    ep_lap_time = current_last_lap
                    bonus = compute_lap_bonus(ep_lap_time, best_lap_time)
                    reward += bonus

                    if ep_lap_time < best_lap_time:
                        old_best = best_lap_time
                        best_lap_time = ep_lap_time
                        print(f"  🏆 NUOVO BEST LAP: {ep_lap_time:.3f}s (precedente: {old_best:.3f}s)")

                        # ── Decay λ_bc basato sulla performance ──
                        # Quando l'agente si avvicina al HUMAN_BEST_TIME,
                        # riduciamo il vincolo BC per permettere di superarlo.
                        # λ = 1.0 quando best == worst, λ → 0.1 quando best ≈ human_best
                        if HUMAN_WORST_TIME > HUMAN_BEST_TIME:
                            progress = (HUMAN_WORST_TIME - best_lap_time) / (HUMAN_WORST_TIME - HUMAN_BEST_TIME)
                            progress = max(0.0, min(1.0, progress))  # clamp [0, 1]
                            new_lambda = max(0.1, args.bc_lambda * (1.0 - 0.9 * progress))
                            agent.bc_lambda = new_lambda
                            print(f"    📉 BC λ aggiornato: {new_lambda:.3f} (progress: {progress:.1%})")

                        # Salva i pesi migliori SUBITO
                        best_path = os.path.join(args.save_dir, "sac_actor_best.pth")
                        torch.save(agent.actor.state_dict(), best_path)
                        print(f"    💾 Best model salvato: {best_path}")

                done = custom_done or env_done or lap_completed
                mask = 0.0 if done else 1.0
                memory.push(state, action, reward, next_state, mask)

                state = next_state
                episode_reward += reward

                # ── Update ──
                # Ritarda gli update finché il buffer non ha abbastanza
                # transizioni dalla guida BC. Questo previene la distruzione
                # dei pesi BC con pochi campioni di bassa qualità.
                if len(memory) > max(args.warmup_steps, args.batch_size):
                    agent.update_parameters(memory, args.batch_size)
                    total_updates += 1

                if done:
                    break

            # ── Logging episodio ──
            status = "LAP" if lap_completed else "FAIL"
            lap_str = f"{ep_lap_time:.3f}s" if lap_completed else "N/A"

            print(
                f"  Ep {ep:4d}/{args.episodes} | {status} | "
                f"Reward: {episode_reward:8.1f} | Steps: {step:5d} | "
                f"LapTime: {lap_str} | Best: {best_lap_time:.3f}s | "
                f"Updates: {total_updates} | α: {agent.alpha:.4f} | "
                f"λ_bc: {agent.bc_lambda:.3f}"
            )

            with open(log_path, 'a') as f:
                f.write(
                    f"ep={ep},reward={episode_reward:.2f},steps={step},"
                    f"lap={status},lap_time={ep_lap_time:.3f},"
                    f"best={best_lap_time:.3f},updates={total_updates}\n"
                )

            # ── Checkpoint ──
            if ep % args.checkpoint_every == 0:
                path = os.path.join(args.save_dir, f"sac_actor_ep{ep:04d}.pth")
                torch.save(agent.actor.state_dict(), path)
                print(f"    💾 Checkpoint: {path}")

    except KeyboardInterrupt:
        print(f"\n\n  🛑 Training interrotto dall'utente all'episodio {ep}.")

    finally:
        # ── Salvataggio finale ──
        final_path = os.path.join(args.save_dir, "sac_actor_final.pth")
        torch.save(agent.actor.state_dict(), final_path)
        print(f"\n  Pesi finali salvati: {final_path}")
        print(f"  Best lap time raggiunto: {best_lap_time:.3f}s")
        print(f"  Log training: {log_path}")

        env.end()


if __name__ == "__main__":
    main()
