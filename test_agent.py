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
import csv
import numpy as np
import torch
import torch.nn as nn
from collections import deque

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'gym_torcs')))

from gym_torcs import TorcsEnv


# ──────────────────────────────────────────────────────────────────────
#  Rete (identica a behavioral_cloning.py)
# ──────────────────────────────────────────────────────────────────────

class PolicyNetwork(nn.Module):
    """Rete Actor per Behavioral Cloning con architettura Multi-Head:
    stato (29D) → testa continua (steer, accel, brake) & testa discreta (gear).

    Il backbone estrae feature condivise. Le due teste separate evitano
    le oscillazioni e i ritardi tipici della regressione sul cambio marcia.
    """

    def __init__(self, state_dim: int = 87, hidden_size: int = 512):
        super(PolicyNetwork, self).__init__()

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

        # Testa continua per: steer (1), accel (1), brake (1)
        self.continuous_head = nn.Linear(hidden_size, 3)
        
        # Testa discreta per la marcia (7 classi: 0, 1, 2, 3, 4, 5, 6)
        self.gear_head = nn.Linear(hidden_size, 7)

    def forward(self, state: torch.Tensor):
        features = self.backbone(state)
        
        cont_out = self.continuous_head(features)
        
        # Separiamo e applichiamo le attivazioni corrette
        steer = torch.tanh(cont_out[:, 0:1])          # [-1, 1]
        accel_brake = torch.sigmoid(cont_out[:, 1:3])   # [0, 1]
        
        continuous = torch.cat([steer, accel_brake], dim=1) # 3D: [steer, accel, brake]
        
        gear_logits = self.gear_head(features)          # 7D logits
        
        return continuous, gear_logits


# ──────────────────────────────────────────────────────────────────────
#  Utilities
# ──────────────────────────────────────────────────────────────────────

def flatten_state(state_dict: dict) -> np.ndarray:
    """Appiattisce osservazione TORCS → vettore 29D.

    Deve essere identica a data_collection.flatten_state() per coerenza.
    """
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
            _a('track', 19),              # già /200 da make_observaton
            [_s('trackPos')],
            [_s('speedX')],               # già /50 da make_observaton
            [_s('speedY')],               # già /50 da make_observaton
            [_s('speedZ')],               # già /50 da make_observaton
            _a('wheelSpinVel', 4) / 100.0,
            [_s('rpm') / 10000.0],
        ]).astype(np.float32)
    except Exception:
        return np.zeros(29, dtype=np.float32)


def denormalize_action(cont_action: np.ndarray, gear: int) -> np.ndarray:
    """Converte l'azione continua (3D) + marcia (int) nel formato TORCS.

    Gli output di accel e brake derivano da un'attivazione Sigmoid [0, 1].
    """
    env_action = np.zeros(4, dtype=np.float32)
    env_action[0] = np.clip(cont_action[0], -1.0, 1.0)               # steer
    env_action[1] = np.clip(cont_action[1], 0.0, 1.0)                # accel
    env_action[2] = np.clip(cont_action[2], 0.0, 1.0)                # brake
    env_action[3] = float(max(0, min(6, gear)))                      # gear
    return env_action


def apply_tcs(action: np.ndarray, obs: dict, slip_threshold: float = 5.0) -> np.ndarray:
    wsv = obs.get('wheelSpinVel', None)
    if wsv is None:
        return action

    wsv = np.array(wsv, dtype=np.float64).flatten()
    if wsv.shape[0] < 4:
        return action

    # Slip = (rear avg) - (front avg)
    rear_avg = (wsv[2] + wsv[3]) / 2.0
    front_avg = (wsv[0] + wsv[1]) / 2.0
    slip = rear_avg - front_avg

    if slip > slip_threshold:
        reduction = max(0.2, 1.0 - (slip - slip_threshold) / 30.0)
        action = action.copy()
        action[1] *= reduction  # Scala l'acceleratore

    return action


def apply_esp(action, obs, step=0):
    """
    Active Safety Envelope (ESP / Lane Keep Assist) - Versione Leggera
    Agisce come fail-safe morbido solo in prossimità del limite estremo di pista (1.15).
    Evita interventi bruschi per non destabilizzare la fisica dell'auto.
    """
    track_pos = obs.get('trackPos', 0.0)
    if isinstance(track_pos, np.ndarray):
        track_pos = track_pos.flat[0]

    action = action.copy()

    # Intervento sterzo molto leggero sopra 1.15
    if abs(track_pos) > 1.15:
        # Nudge proporzionale molto dolce
        steer_nudge = -0.15 * (np.sign(track_pos) * (abs(track_pos) - 1.15))
        action[0] = np.clip(action[0] + steer_nudge, -1.0, 1.0)
        
        # Parzializzazione gas e freno leggerissimi solo sopra 1.25 (vicino all'offtrack 1.50)
        if abs(track_pos) > 1.25:
            # Parzializzazione del gas (riduzione max del 30% per non tagliare bruscamente)
            throttle_scale = max(0.70, 1.0 - 1.2 * (abs(track_pos) - 1.25))
            action[1] *= throttle_scale
            
            # Frenata stabilizzante minima (max 0.05) per stabilizzare il retrotreno
            brake_nudge = 0.20 * (abs(track_pos) - 1.25)
            if brake_nudge > 0.01:
                action[2] = max(action[2], min(0.05, brake_nudge))
            
            if step % 20 == 0:
                print(f"    [ESP Soft] tp={track_pos:+.3f} | nudge={steer_nudge:+.3f} | scale={throttle_scale:.2f} | brake={action[2]:.2f}")

    return action


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
    print(f"  Stride Type: static (k=6, 0.24s)")
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
            
            # Reset ambiente con relaunch forzato ad ogni tentativo per garantire uno stato fisico iniziale pulito ed identico
            obs = env.reset(relaunch=True)
            initial_state = flatten_state(obs)

            # Inizializza buffer storico per State Stacking (esattamente 13 elementi: t-12, t-6, t)
            state_buffer = deque(maxlen=13)
            for _ in range(13):
                state_buffer.append(initial_state)

            # Lap tracking
            raw = env.client.S.d
            prev_last_lap = float(raw.get('lastLapTime', 0.0))
            if isinstance(prev_last_lap, list): prev_last_lap = prev_last_lap[0]

            lap_completed = False
            lap_time = 0.0
            telemetry_data = []

            print(f"\n  {'─' * 50}")
            print(f"  🏁 Tentativo #{total_attempts} (giri completati: {len(lap_times)}/{args.laps})")

            for step in range(1, args.max_steps + 1):
                # Costruisce il vettore di stato 87D concatenando t-12 (index 0), t-6 (index 6), t (index 12)
                stacked_state = np.concatenate([
                    state_buffer[0],
                    state_buffer[6],
                    state_buffer[12]
                ])

                # ── Inferenza DETERMINISTICA (Pure BC, sterzata/acceleratore/freno + marcia dal modello) ──
                with torch.no_grad():
                    state_t = torch.FloatTensor(stacked_state).to(device).unsqueeze(0)
                    pred_cont, gear_logits = model(state_t)
                    cont_action = pred_cont.cpu().numpy()[0]          # [steer, accel, brake]
                    gear = int(gear_logits.argmax(dim=1).item())
                    if gear < 1: gear = 1  # Safety: no retromarcia/folle

                # ── Mutual exclusion accel/brake (come l'esperto umano) ──
                if cont_action[2] > 0.05:
                    cont_action[1] = 0.0  # Se freno, niente gas

                # ── Step nell'ambiente (azione pura dal modello + TCS + ESP) ──
                env_action = denormalize_action(cont_action, gear)
                env_action = apply_tcs(env_action, obs)
                env_action = apply_esp(env_action, obs, step)
                next_obs, _, env_done, _ = env.step(env_action)
                next_state = flatten_state(next_obs)

                # Salva telemetria step
                dist_raw = next_obs.get('distFromStart', 0.0)
                if isinstance(dist_raw, np.ndarray): dist_raw = float(dist_raw.flat[0])
                dist_m = dist_raw
                spd_kmh = float(next_state[21] * 50.0)
                track_pos = float(np.array(next_obs.get('trackPos', 0.0)).flat[0])
                angle = float(np.array(next_obs.get('angle', 0.0)).flat[0])
                
                telemetry_data.append({
                    'step': step,
                    'dist': dist_m,
                    'speed': spd_kmh,
                    'trackPos': track_pos,
                    'angle': angle,
                    'steer': float(env_action[0]),
                    'accel': float(env_action[1]),
                    'brake': float(env_action[2]),
                    'gear': int(env_action[3])
                })

                # ── Check fuoripista/spin (soglia a 1.50 per consentire l'uso delle vie di fuga asfaltate e dei cordoli estesi) ──
                if abs(track_pos) > 1.50:
                    print(f"  ⚠️  Fuori pista allo step {step} (trackPos={track_pos:.3f})")
                    break
                if np.cos(angle) < 0:
                    print(f"  ⚠️  Spin allo step {step} (angle={angle:.3f})")
                    break

                # ── Telemetria ogni 200 step ──
                if step % 200 == 0:
                    print(f"    [Step {step:4d}] tp={track_pos:+.3f} | spd={spd_kmh:.0f}km/h | steer={env_action[0]:+.3f} | accel={env_action[1]:.2f} | brake={env_action[2]:.2f} | gear={int(env_action[3])}")

                # ── Check completamento giro ──
                current_last_lap = float(raw.get('lastLapTime', 0.0))
                if isinstance(current_last_lap, list): current_last_lap = current_last_lap[0]

                if current_last_lap > 0.0 and abs(current_last_lap - prev_last_lap) > 0.01:
                    lap_completed = True
                    lap_time = current_last_lap
                    break

                state_buffer.append(next_state)
                obs = next_obs
                if env_done: break

            # Salva telemetria in CSV a fine tentativo
            telemetry_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'telemetry')
            os.makedirs(telemetry_dir, exist_ok=True)
            csv_path = os.path.join(telemetry_dir, f'telemetry_attempt_{total_attempts}.csv')
            with open(csv_path, 'w', newline='') as f_csv:
                writer = csv.DictWriter(f_csv, fieldnames=['step', 'dist', 'speed', 'trackPos', 'angle', 'steer', 'accel', 'brake', 'gear'])
                writer.writeheader()
                writer.writerows(telemetry_data)
            print(f"  📊 Telemetria del tentativo salvata in: {csv_path}")

            if lap_completed:
                lap_times.append(lap_time)
                print(f"  ✅ GIRO COMPLETATO: {lap_time:.3f}s")
            else:
                print(f"  ❌ Fallito (step: {step})")
                print("  ⚠️  Tentativo fallito. Prossimo tentativo...")
                # Continua con il prossimo tentativo (il while loop riproverà con relaunch=True)

    except KeyboardInterrupt:
        print(f"\n  🛑 Test interrotto.")
    finally:
        env.end()

    # ── Riepilogo ──
    print(f"\n{'=' * 64}")
    print(f"  📊 RIEPILOGO TEST")
    if lap_times:
        print(f"  Giri completati: {len(lap_times)}/{args.laps}")
        print(f"  Best:  {min(lap_times):.3f}s | Media: {sum(lap_times)/len(lap_times):.3f}s")
        for i, t in enumerate(lap_times):
            print(f"  Giro {i+1}: {t:.3f}s")
    else:
        print(f"  Nessun giro completato su {total_attempts} tentativi.")
    print(f"{'=' * 64}\n")


if __name__ == "__main__":
    main()
