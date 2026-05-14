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
import datetime
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
    def __init__(self, state_dim=30, action_dim=4, hidden_size=256):
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
    def __init__(self, state_dim=30, action_dim=4, hidden_size=256):
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
    parser = argparse.ArgumentParser(description="Test Agent Autonomo — TORCS")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path ai pesi del modello (.pth)")
    parser.add_argument("--model", type=str, choices=["bc", "sac"], default="sac",
                        help="Tipo di modello: 'bc' (PolicyNetwork) o 'sac' (Actor)")
    parser.add_argument("--laps", type=int, default=3,
                        help="Numero di giri da completare")
    parser.add_argument("--max_steps", type=int, default=15000,
                        help="Max step per giro (timeout)")
    parser.add_argument("--sigma", type=float, default=None,
                        help="Exploration noise σ (default: auto dal checkpoint, 0=deterministico)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Fissa il seed per rendere il rumore perfettamente riproducibile")
    parser.add_argument("--resume", action="store_true",
                        help="Riprende il test da dove lasciato (legge test_results.log per escludere i seed e riprendere i lap completati)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'=' * 64}")
    print(f"  🏁 TEST AGENTE AUTONOMO — TORCS")
    print(f"  Device: {device}")
    print(f"  Modello: {args.model.upper()} | Pesi: {args.weights}")
    print(f"  Giri da completare: {args.laps}")

    # ── Carica modello ──
    if args.model == "bc":
        model = PolicyNetwork().to(device)
    else:
        model = Actor().to(device)

    if not os.path.exists(args.weights):
        print(f"  ❌ File pesi non trovato: {args.weights}")
        sys.exit(1)

    # weights_only=False necessario per checkpoint completi che contengono oggetti numpy
    try:
        checkpoint = torch.load(args.weights, map_location=device, weights_only=True)
    except Exception:
        print("  ℹ️  Caricamento con weights_only=False (checkpoint completo)...")
        checkpoint = torch.load(args.weights, map_location=device, weights_only=False)

    # Estrai sigma dal checkpoint se disponibile
    sigma = 0.0  # Default: deterministico
    if isinstance(checkpoint, dict) and "actor" in checkpoint:
        print("  ℹ️  Rilevato checkpoint completo. Caricamento pesi Actor...")
        state_dict = checkpoint["actor"]
        if "episode" in checkpoint:
            print(f"  📊 Episodio: {checkpoint['episode']}")
        if "best_completed_lap_time" in checkpoint:
            print(f"  📊 Best lap time: {checkpoint['best_completed_lap_time']:.3f}s")
        # Estrai sigma dallo scheduler salvato nel checkpoint
        if "scheduler" in checkpoint and "sigma" in checkpoint["scheduler"]:
            sigma = float(checkpoint["scheduler"]["sigma"])
            print(f"  📊 Sigma dal checkpoint: {sigma:.4f}")
    else:
        state_dict = checkpoint

    # Override manuale da CLI
    if args.sigma is not None:
        sigma = args.sigma
        print(f"  ℹ️  Sigma override da CLI: {sigma:.4f}")

    model.load_state_dict(state_dict)
    model.eval()
    print(f"  ✅ Pesi caricati correttamente.")

    mode_str = f"σ={sigma:.4f} (quasi-deterministico)" if sigma > 0 else "DETERMINISTICO"
    print(f"  🎯 Modalità: {mode_str}")
    print(f"{'=' * 64}\n")

    # ── Lettura History (Resume e Seed Esclusi) ──
    log_dir = os.path.join(os.path.dirname(__file__), "train_set", "session_logs")
    log_file = os.path.join(log_dir, "test_results.log")
    
    tested_seeds = set()
    lap_times = []
    total_laps_attempted = 0
    weights_basename = os.path.basename(args.weights)

    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            for line in f:
                if f"Weights: {weights_basename}," in line and f"Sigma: {sigma:.4f}," in line:
                    try:
                        seed_part = line.split("Seed: ")[1].split(",")[0]
                        seed_val = int(seed_part)
                        tested_seeds.add(seed_val)
                        
                        if args.resume:
                            total_laps_attempted += 1
                            if "Status: SUCCESS" in line:
                                time_part = line.split("LapTime: ")[1].split("s")[0]
                                lap_times.append(float(time_part))
                    except Exception:
                        pass
                        
    if len(tested_seeds) > 0:
        print(f"  📜 Trovati {len(tested_seeds)} seed già testati per questa configurazione.")
    if args.resume and len(lap_times) > 0:
        print(f"  ▶️  Resume attivo: trovati {len(lap_times)} giri completati in precedenza.")
        if len(lap_times) >= args.laps:
            print(f"  ✅ Obiettivo di {args.laps} giri completati già raggiunto!")
            
            # Stampa riepilogo rapido ed esci
            print(f"\n{'=' * 64}")
            print(f"  📊 RIEPILOGO TEST STORICO")
            print(f"  Sigma: {sigma:.4f}" + (" (deterministico)" if sigma == 0 else ""))
            print(f"  Best:  {min(lap_times):.3f}s | Media: {sum(lap_times)/len(lap_times):.3f}s")
            print(f"{'=' * 64}\n")
            sys.exit(0)

    # ── Ambiente ──
    print("  Inizializzazione TORCS...")
    env = TorcsEnv(vision=False, throttle=True, gear_change=True, early_termination=False)

    try:
        while len(lap_times) < args.laps:
            total_laps_attempted += 1
            
            # Imposta il seed per riproducibilità esatta
            import random
            if args.seed is not None:
                current_seed = args.seed
                if current_seed in tested_seeds:
                    print(f"  ⚠️  Attenzione: il seed {current_seed} è già stato testato in precedenza.")
            else:
                while True:
                    current_seed = random.randint(0, 1000000)
                    if current_seed not in tested_seeds:
                        break
            tested_seeds.add(current_seed)
            
            np.random.seed(current_seed)
            torch.manual_seed(current_seed)
            random.seed(current_seed)
            
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
            print(f"  🏁 Tentativo #{total_laps_attempted} (Seed: {current_seed}) "
                  f"(giri completati: {len(lap_times)}/{args.laps})")

            for step in range(1, args.max_steps + 1):
                # ── Inferenza ──
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).to(device).unsqueeze(0)
                    action_t = model(state_t)
                    action = action_t.cpu().numpy()[0]

                # ── Exploration noise (stesse condizioni del training) ──
                if sigma > 0:
                    noise = np.random.normal(0, sigma, size=3)
                    action[0] = np.clip(action[0] + noise[0], -1.0, 1.0)  # steer
                    action[1] = np.clip(action[1] + noise[1], -1.0, 1.0)  # accel
                    action[2] = np.clip(action[2] + noise[2], -1.0, 1.0)  # brake

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
                status_str = "SUCCESS"
                lap_str = f"{lap_time:.3f}s"
                print(f"  ✅ GIRO COMPLETATO: {lap_time:.3f}s  "
                      f"({len(lap_times)}/{args.laps})")
            else:
                status_str = "FAIL"
                lap_str = "N/A"
                print(f"  ❌ Giro non completato (step: {step})")
            
            # ── Logging su file ──
            log_dir = os.path.join(os.path.dirname(__file__), "train_set", "session_logs")
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, "test_results.log")
            with open(log_file, "a") as f:
                ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{ts}] Model: {args.model.upper()}, Weights: {os.path.basename(args.weights)}, "
                        f"Seed: {current_seed}, Sigma: {sigma:.4f}, Status: {status_str}, "
                        f"LapTime: {lap_str}, Steps: {step}\n")
                        
            if not lap_completed and args.seed is not None:
                print(f"  🛑 Il test con seed fisso ({args.seed}) è terminato. Esco per evitare un loop infinito.")
                break

    except KeyboardInterrupt:
        print(f"\n\n  🛑 Test interrotto dall'utente.")

    finally:
        env.end()

    # ── Riepilogo ──
    print(f"\n{'=' * 64}")
    print(f"  📊 RIEPILOGO TEST")
    print(f"  Sigma: {sigma:.4f}" + (" (deterministico)" if sigma == 0 else ""))
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
