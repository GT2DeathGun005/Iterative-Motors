"""
Soft Actor-Critic (SAC) — Fine-Tuning RL per Giro Secco TORCS

Addestra un agente SAC pre-inizializzato con i pesi del Behavioral Cloning
per battere i tempi umani su singolo giro con partenza da fermo.

Features:
  - Warm Start: carica backbone + mean_linear dal BC checkpoint
  - Reward dinamica: l'agente cerca di battere il proprio best lap time
  - Terminazione: uscita pista, spin (cos(angle)<0), stallo prolungato
  - Checkpoint periodici in train_set/
  - GPU-optimized (CUDA)

Action de-normalization (Tanh [-1,1] → env ranges):
  steer = action[0]                              # [-1, 1]
  accel = (action[1] + 1) / 2                    # [0, 1]
  brake = (action[2] + 1) / 2                    # [0, 1]
  gear  = round((action[3] + 1) * 3)             # [0, 6] discretizzato
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
                 tau: float = 0.005, lr: float = 3e-4):
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau

        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # Auto-tuning alpha (entropia)
        self.target_entropy = -float(action_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=lr)
        self.alpha = self.log_alpha.exp().item()

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

        # Actor update
        pi, log_pi, _ = self.actor.sample(state_b)
        q1_pi, q2_pi = self.critic(state_b, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)
        policy_loss = ((self.alpha * log_pi) - min_q_pi).mean()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # Alpha update
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp().item()

        # Soft update target
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        return qf_loss.item(), policy_loss.item(), alpha_loss.item()


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


# ──────────────────────────────────────────────────────────────────────
#  Reward Function — Giro Secco
# ──────────────────────────────────────────────────────────────────────

def compute_step_reward(obs: dict, prev_dist: float, raw_obs: dict) -> tuple:
    """Reward per singolo step. Ritorna (reward, done, dist_raced).

    Componenti:
      + progress: avanzamento sulla pista (Δ distRaced)
      + speed:    bonus velocità longitudinale
      - center:   penalità quadratica per distanza dal centro
      - angle:    penalità per disallineamento
      - offtrack: terminazione + penalità pesante
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

    # Uscita di pista
    if abs(track_pos) > 1.0:
        done = True
        reward = -500.0

    # Spin: l'auto si è girata
    if np.cos(angle) < 0:
        done = True
        reward = -500.0

    return reward, done, dist_raced


def compute_lap_bonus(lap_time: float, best_time: float) -> float:
    """Bonus per completamento giro. Extra se batte il best time."""
    base = 500.0
    if lap_time < best_time:
        improvement = best_time - lap_time
        return base + improvement * 200.0  # 200 punti per ogni secondo risparmiato
    return base


# ──────────────────────────────────────────────────────────────────────
#  Main Training Loop
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SAC RL Training — TORCS Giro Secco")
    parser.add_argument("--episodes", type=int, default=1000, help="Episodi di training")
    parser.add_argument("--max_steps", type=int, default=10000, help="Max step per episodio")
    parser.add_argument("--bc_weights", type=str, default="train_set/bc_policy.pth",
                        help="Path ai pesi BC per warm start")
    parser.add_argument("--save_dir", type=str, default="train_set",
                        help="Directory per checkpoint")
    parser.add_argument("--target_time", type=float, default=75.0,
                        help="Tempo target umano in secondi (default: 75s)")
    parser.add_argument("--buffer_size", type=int, default=200000,
                        help="Capacità replay buffer")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--warmup_steps", type=int, default=1000,
                        help="Step di esplorazione random prima del training")
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
    print(f"  🏎️  SAC REINFORCEMENT LEARNING — Giro Secco TORCS")
    print(f"  Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Episodi: {args.episodes} | Buffer: {args.buffer_size}")
    print(f"  Tempo target iniziale: {args.target_time:.1f}s")
    print(f"{'=' * 64}\n")

    # ── Agent ──
    agent = SACAgent(state_dim, action_dim, device)

    # ── Warm Start ──
    agent.actor.load_bc_weights(args.bc_weights, device)

    # ── Replay Buffer ──
    memory = ReplayBuffer(capacity=args.buffer_size)

    # ── Best time tracking (reward dinamica) ──
    best_lap_time = args.target_time
    total_updates = 0

    # ── Training log ──
    log_path = os.path.join(args.save_dir, f"sac_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

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
                if len(memory) < args.warmup_steps:
                    # Esplorazione random durante il warmup
                    action = np.random.uniform(-1, 1, size=action_dim).astype(np.float32)
                else:
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
                if step > 200 and abs(raw_speed) < 5.0:
                    stall_counter += 1
                    if stall_counter > 100:
                        custom_done = True
                        reward = -200.0
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

                done = custom_done or env_done or lap_completed
                mask = 0.0 if done else 1.0
                memory.push(state, action, reward, next_state, mask)

                state = next_state
                episode_reward += reward

                # ── Update ──
                if len(memory) > args.batch_size:
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
                f"Updates: {total_updates} | Alpha: {agent.alpha:.4f}"
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
        best_path = os.path.join(args.save_dir, "sac_actor_best.pth")
        torch.save(agent.actor.state_dict(), final_path)
        torch.save(agent.actor.state_dict(), best_path)
        print(f"\n  Pesi finali salvati: {final_path}")
        print(f"  Best lap time raggiunto: {best_lap_time:.3f}s")
        print(f"  Log training: {log_path}")

        env.end()


if __name__ == "__main__":
    main()
