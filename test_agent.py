"""
Test Agent — Guida Autonoma su TORCS (BC Deterministico)

Carica i pesi del modello Behavioral Cloning (BC) e fa guidare l'agente
in modalità rigorosamente deterministica per replicare il giro perfetto.

Uso:
  python test_agent.py --weights train_set/checkpoints/bc_policy.pth
"""

import os
import sys
import argparse
import time
import datetime
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'gym_torcs')))

from gym_torcs import TorcsEnv


# ──────────────────────────────────────────────────────────────────────
#  Rete (identica a behavioral_cloning.py)
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
#  Utilities
# ──────────────────────────────────────────────────────────────────────

def flatten_state(state_dict: dict) -> np.ndarray:
    """Appiattisce osservazione TORCS → vettore 30D."""
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
            [_s('distFromStart') / 4000.0],
        ]).astype(np.float32)
    except Exception:
        return np.zeros(30, dtype=np.float32)


def denormalize_action(action: np.ndarray) -> np.ndarray:
    """Converte azione Tanh [-1,1] → formato env TORCS."""
    env_action = np.zeros(4, dtype=np.float32)
    env_action[0] = np.clip(action[0], -1.0, 1.0)               # steer
    env_action[1] = np.clip((action[1] + 1.0) / 2.0, 0.0, 1.0)  # accel
    env_action[2] = np.clip((action[2] + 1.0) / 2.0, 0.0, 1.0)  # brake
    gear = int(round((action[3] + 1.0) * 3.0))                   # gear
    env_action[3] = float(max(0, min(6, gear)))
    return env_action


# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test Agent Autonomo (BC) — TORCS")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path ai pesi del modello (.pth)")
    parser.add_argument("--laps", type=int, default=3,
                        help="Numero di giri da completare")
    parser.add_argument("--max_steps", type=int, default=15000,
                        help="Max step per giro (timeout)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'=' * 64}")
    print(f"  🏁 TEST AGENTE AUTONOMO (BC) — TORCS")
    print(f"  Device: {device}")
    print(f"  Pesi: {args.weights}")
    print(f"  🎯 Modalità: DETERMINISTICA (Zero Noise)")
    print(f"{'=' * 64}\n")

    # ── Carica modello ──
    model = PolicyNetwork().to(device)

    if not os.path.exists(args.weights):
        print(f"  ❌ File pesi non trovato: {args.weights}")
        sys.exit(1)

    try:
        state_dict = torch.load(args.weights, map_location=device, weights_only=True)
    except Exception:
        state_dict = torch.load(args.weights, map_location=device, weights_only=False)

    model.load_state_dict(state_dict)
    model.eval()
    print(f"  ✅ Pesi caricati correttamente.")

    # ── Ambiente ──
    print("  Inizializzazione TORCS...")
    env = TorcsEnv(vision=False, throttle=True, gear_change=True, early_termination=False)

    lap_times = []
    total_attempts = 0

    try:
        while len(lap_times) < args.laps:
            total_attempts += 1
            
            # Reset ambiente
            obs = env.reset(relaunch=(total_attempts == 1 or total_attempts % 10 == 0))
            state = flatten_state(obs)

            # Lap tracking
            raw = env.client.S.d
            prev_last_lap = float(raw.get('lastLapTime', 0.0))
            if isinstance(prev_last_lap, list): prev_last_lap = prev_last_lap[0]

            lap_completed = False
            lap_time = 0.0

            print(f"\n  {'─' * 50}")
            print(f"  🏁 Tentativo #{total_attempts} (giri completati: {len(lap_times)}/{args.laps})")

            for step in range(1, args.max_steps + 1):
                # ── Inferenza DETERMINISTICA ──
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).to(device).unsqueeze(0)
                    action_t = model(state_t)
                    action = action_t.cpu().numpy()[0]

                # ── Step nell'ambiente ──
                env_action = denormalize_action(action)
                next_obs, _, env_done, _ = env.step(env_action)
                next_state = flatten_state(next_obs)

                # ── Check fuoripista/spin ──
                track_pos = float(np.array(next_obs.get('trackPos', 0.0)).flat[0])
                angle = float(np.array(next_obs.get('angle', 0.0)).flat[0])

                if abs(track_pos) > 1.0:
                    print(f"  ⚠️  Fuori pista allo step {step} (trackPos={track_pos:.3f})")
                    break
                if np.cos(angle) < 0:
                    print(f"  ⚠️  Spin allo step {step} (angle={angle:.3f})")
                    break

                # ── Check completamento giro ──
                current_last_lap = float(raw.get('lastLapTime', 0.0))
                if isinstance(current_last_lap, list): current_last_lap = current_last_lap[0]

                if current_last_lap > 0.0 and abs(current_last_lap - prev_last_lap) > 0.01:
                    lap_completed = True
                    lap_time = current_last_lap
                    break

                state = next_state
                if env_done: break

            if lap_completed:
                lap_times.append(lap_time)
                print(f"  ✅ GIRO COMPLETATO: {lap_time:.3f}s")
            else:
                print(f"  ❌ Fallito (step: {step})")

    except KeyboardInterrupt:
        print(f"\n  🛑 Test interrotto.")
    finally:
        env.end()

    # ── Riepilogo ──
    print(f"\n{'=' * 64}")
    print(f"  📊 RIEPILOGO TEST")
    if lap_times:
        print(f"  Best:  {min(lap_times):.3f}s | Media: {sum(lap_times)/len(lap_times):.3f}s")
    else:
        print(f"  Nessun giro completato.")
    print(f"{'=' * 64}\n")


if __name__ == "__main__":
    main()
