"""
Test Agent — Guida Autonoma su TORCS

Carica i pesi di un modello addestrato (BC o SAC) e fa guidare l'agente
in modalità deterministica (senza esplorazione) per valutarne le prestazioni.

Uso:
  # Testa il modello BC (warm start)
  python test_agent.py --weights train_set/checkpoints/bc_policy.pth --model bc

  # Testa il modello SAC (dopo fine-tuning RL)
  python test_agent.py --weights train_set/checkpoints/sac_actor_final.pth --model sac

  # Più giri per valutare la consistenza
  python test_agent.py --weights train_set/checkpoints/sac_actor_best.pth --model sac --laps 5
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'gym_torcs')))

from gym_torcs import TorcsEnv


# ──────────────────────────────────────────────────────────────────────
#  Reti (identiche a quelle di training)
# ──────────────────────────────────────────────────────────────────────

class PolicyNetwork(nn.Module):
    """Rete BC: stato → azione deterministica (Tanh)."""
    def __init__(self, state_dim=29, action_dim=4, hidden_size=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        return self.net(state)


class Actor(nn.Module):
    """Rete SAC Actor: stato → (mean, log_std). Per il test usiamo solo la mean."""
    def __init__(self, state_dim=29, action_dim=4, hidden_size=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        )
        self.mean_linear = nn.Linear(hidden_size, action_dim)
        self.log_std_linear = nn.Linear(hidden_size, action_dim)

    def forward(self, state):
        x = self.net(state)
        mean = self.mean_linear(x)
        return torch.tanh(mean)  # Azione deterministica


# ──────────────────────────────────────────────────────────────────────
#  Utilities (identiche a sac_rl.py)
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
    parser = argparse.ArgumentParser(description="Test Agent Autonomo — TORCS")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path ai pesi del modello (.pth)")
    parser.add_argument("--model", type=str, choices=["bc", "sac"], default="sac",
                        help="Tipo di modello: 'bc' (PolicyNetwork) o 'sac' (Actor)")
    parser.add_argument("--laps", type=int, default=3,
                        help="Numero di giri da completare")
    parser.add_argument("--max_steps", type=int, default=15000,
                        help="Max step per giro (timeout)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'=' * 64}")
    print(f"  🏁 TEST AGENTE AUTONOMO — TORCS")
    print(f"  Device: {device}")
    print(f"  Modello: {args.model.upper()} | Pesi: {args.weights}")
    print(f"  Giri da completare: {args.laps}")
    print(f"{'=' * 64}\n")

    # ── Carica modello ──
    if args.model == "bc":
        model = PolicyNetwork().to(device)
    else:
        model = Actor().to(device)

    if not os.path.exists(args.weights):
        print(f"  ❌ File pesi non trovato: {args.weights}")
        sys.exit(1)

    checkpoint = torch.load(args.weights, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and "actor" in checkpoint:
        print("  ℹ️ Rilevato checkpoint di training completo. Caricamento dei pesi dell'Actor...")
        state_dict = checkpoint["actor"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()
    print(f"  ✅ Pesi caricati correttamente.\n")

    # ── Ambiente ──
    print("  Inizializzazione TORCS...")
    env = TorcsEnv(vision=False, throttle=True, gear_change=True, early_termination=False)

    lap_times = []
    total_laps_attempted = 0

    try:
        while len(lap_times) < args.laps:
            total_laps_attempted += 1
            need_relaunch = (total_laps_attempted == 1) or (total_laps_attempted % 10 == 0)

            if total_laps_attempted == 1:
                obs = env.reset(relaunch=True)
            else:
                obs = env.reset(relaunch=need_relaunch)

            state = flatten_state(obs)

            # Lap tracking
            raw = env.client.S.d
            prev_last_lap = float(raw.get('lastLapTime', 0.0))
            if isinstance(prev_last_lap, list):
                prev_last_lap = prev_last_lap[0]

            lap_completed = False
            lap_time = 0.0

            print(f"\n  {'─' * 50}")
            print(f"  🏁 Tentativo #{total_laps_attempted}  "
                  f"(giri completati: {len(lap_times)}/{args.laps})")

            for step in range(1, args.max_steps + 1):
                # ── Inferenza deterministica ──
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).to(device).unsqueeze(0)
                    action_t = model(state_t)
                    action = action_t.cpu().numpy()[0]

                # ── Step nell'ambiente ──
                env_action = denormalize_action(action)
                next_obs, _, env_done, _ = env.step(env_action)
                next_state = flatten_state(next_obs)

                # ── Check fuoripista/spin ──
                raw = env.client.S.d
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
                if isinstance(current_last_lap, list):
                    current_last_lap = current_last_lap[0]

                if current_last_lap > 0.0 and abs(current_last_lap - prev_last_lap) > 0.01:
                    lap_completed = True
                    lap_time = current_last_lap
                    break

                state = next_state

                if env_done:
                    print(f"  ⚠️  Episodio terminato dall'env allo step {step}")
                    break

            # ── Risultato del giro ──
            if lap_completed:
                lap_times.append(lap_time)
                print(f"  ✅ GIRO COMPLETATO: {lap_time:.3f}s  "
                      f"({len(lap_times)}/{args.laps})")
            else:
                print(f"  ❌ Giro non completato (step: {step})")

    except KeyboardInterrupt:
        print(f"\n\n  🛑 Test interrotto dall'utente.")

    finally:
        env.end()

    # ── Riepilogo ──
    print(f"\n{'=' * 64}")
    print(f"  📊 RIEPILOGO TEST")
    print(f"  Giri completati: {len(lap_times)}/{total_laps_attempted} tentativi")

    if lap_times:
        print(f"  Best:  {min(lap_times):.3f}s")
        print(f"  Worst: {max(lap_times):.3f}s")
        print(f"  Media: {sum(lap_times)/len(lap_times):.3f}s")
        print(f"  Tutti: {', '.join(f'{t:.3f}s' for t in lap_times)}")
    else:
        print(f"  Nessun giro completato.")

    print(f"{'=' * 64}\n")


if __name__ == "__main__":
    main()
