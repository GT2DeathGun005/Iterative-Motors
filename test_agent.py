"""
Test Agent — Guida Autonoma su TORCS (Compatibile BC + TD3)

Carica i pesi del modello (BC o TD3) e fa guidare l'agente in modalità
rigorosamente deterministica.

La classe BCActor è compatibile con entrambi i formati:
  - bc_policy.pth  (senza log_std_head) — caricato con strict=False
  - td3_policy.pth (con log_std_head)   — caricato con strict=True

Priorità di caricamento automatica:
  1. td3_det_best_lap.pth (miglior GIRO VALIDO deterministico: candidato submission)
  2. td3_det_best_dist.pth   (miglior policy ASSOLUTA per distanza; sopravvive a --clean)
  3. td3_det_best_dist_run.pth   (miglior checkpoint deterministico TD3)
  4. td3_expl_best_lap.pth    (record sul giro TD3)
  5. td3_expl_best_dist.pth   (record di distanza TD3)
  6. td3_policy.pth      (ultimo step TD3)
  7. bc_policy.pth       (fallback supervisionato)
  --weights path      (override esplicito, se fornito)

Determinismo:
  - Determinismo per costruzione (policy evaluate=True senza rumore, gearing/fisica deterministici)
  - model.eval() per disabilitare dropout/batchnorm stocastiche
  - actor.sample(state, evaluate=True) bypassa il campionamento gaussiano

Uso:
  python test_agent.py --weights train_set/checkpoints/td3_policy.pth
  python test_agent.py --weights train_set/checkpoints/bc_policy.pth
  python test_agent.py  # auto-detect migliore checkpoint
"""

import os
import sys
import argparse
import csv
import numpy as np
import torch
import torch.nn as nn
from collections import deque

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'gym_torcs')))

from gym_torcs import TorcsEnv
from gearing import compute_gear  # cambio marcia deterministico (anti-hunting), condiviso col training

# ──────────────────────────────────────────────────────────────────────
#  Riproducibilità
# ──────────────────────────────────────────────────────────────────────
# Il test è già deterministico PER COSTRUZIONE: la policy gira con evaluate=True
# (output tanh(mean), zero rumore), il cambio marcia è deterministico (gearing.py) e
# la fisica TORCS è near-deterministica → NESSUN RNG nel percorso di inferenza, quindi
# non serve seedare random/numpy/torch. Pinniamo solo cuDNN per un forward-pass GPU
# bit-riproducibile (no-op su CPU).
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────────────────────────────
#  BCActor — Rete compatibile con BC e TD3
# ──────────────────────────────────────────────────────────────────────

class BCActor(nn.Module):
    """Actor ibrido BC-RL con architettura identica all'Actor TD3.

    Include log_std_head per compatibilità con vecchi pesi.
    In modalità evaluate=True (usata per il test), la log_std_head
    viene completamente ignorata: si usa solo tanh(mean).

    forward() restituisce (continuous, gear_logits) con le attivazioni
    originali del BC (Tanh steer, Sigmoid accel/brake) per compatibilità
    all'indietro.

    sample(state, evaluate=True) restituisce (tanh_action, None, gear_idx)
    per l'inferenza deterministica RL-style.
    """

    def __init__(self, state_dim: int = 87, hidden_size: int = 512):
        super(BCActor, self).__init__()

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

        # Testa log_std per compatibilità pesi RL (ignorata in evaluate mode)
        self.log_std_head = nn.Linear(hidden_size, 3)

    def forward(self, state: torch.Tensor):
        """Forward compatibile all'indietro col BC: Tanh steer, Sigmoid accel/brake.

        Usato SOLO quando si caricano pesi BC puri (bc_policy.pth).
        """
        features = self.backbone(state)

        cont_out = self.continuous_head(features)

        # Attivazioni originali del BC
        steer = torch.tanh(cont_out[:, 0:1])          # [-1, 1]
        accel_brake = torch.sigmoid(cont_out[:, 1:3])   # [0, 1]

        continuous = torch.cat([steer, accel_brake], dim=1)  # 3D: [steer, accel, brake]

        gear_logits = self.gear_head(features)

        return continuous, gear_logits

    def sample(self, state: torch.Tensor, evaluate: bool = False):
        """Campionamento RL-compatible. Con evaluate=True: determinismo assoluto.

        Restituisce (action, log_prob, gear_idx):
          - evaluate=True:  action = tanh(mean), log_prob = None
          - evaluate=False: action = tanh(rsample), log_prob calcolato

        In modalità evaluate, la log_std_head e la distribuzione gaussiana
        vengono completamente bypassate. L'output è deterministico al bit.
        """
        features = self.backbone(state)
        mean = self.continuous_head(features)
        gear_logits = self.gear_head(features)
        gear_idx = torch.argmax(gear_logits, dim=-1)

        if evaluate:
            # Determinismo assoluto: solo tanh(mean), nessun campionamento
            action = torch.tanh(mean)
            return action, None, gear_idx

        # Campionamento stocastico (non usato a test-time)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = log_std.exp()
        from torch.distributions import Normal
        normal = Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)

        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, gear_idx


# ──────────────────────────────────────────────────────────────────────
#  Utilities
# ──────────────────────────────────────────────────────────────────────

# Normalizzazione stati mean-0/std-1 (Fujimoto & Gu 2021): stesse statistiche del BC,
# salvate in state_norm.npz, applicate alla 29D prima dello stacking. DEVE coincidere
# con td3_bc.apply_state_norm e con quanto applicato in fase di training BC.
_STATE_NORM_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'train_set', 'checkpoints', 'state_norm.npz')
if os.path.exists(_STATE_NORM_PATH):
    _sn = np.load(_STATE_NORM_PATH)
    _STATE_MEAN, _STATE_STD = _sn['mean'].astype(np.float32), _sn['std'].astype(np.float32)
else:
    _STATE_MEAN, _STATE_STD = None, None


def apply_state_norm(s):
    if _STATE_MEAN is None:
        return s
    return ((s - _STATE_MEAN) / (_STATE_STD + 1e-3)).astype(np.float32)


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
        s = np.concatenate([
            [_s('angle')],
            _a('track', 19),              # già /200 da make_observaton
            [_s('trackPos')],
            [_s('speedX')],               # già /50 da make_observaton
            [_s('speedY')],               # già /50 da make_observaton
            [_s('speedZ')],               # già /50 da make_observaton
            _a('wheelSpinVel', 4) / 100.0,
            [_s('rpm') / 10000.0],
        ]).astype(np.float32)
        return apply_state_norm(s)  # mean-0/std-1, coerente col training
    except Exception as e:
        # NON silenziare: uno stato a zero falsa l'inferenza ed è difficilissimo da diagnosticare.
        print(f"flatten_state fallita (stato a zero): {e}")
        return apply_state_norm(np.zeros(29, dtype=np.float32))


def denormalize_action_bc(cont_action: np.ndarray, gear: int) -> np.ndarray:
    """Converte l'output BC (Tanh steer, Sigmoid accel/brake) nel formato TORCS."""
    env_action = np.zeros(4, dtype=np.float32)
    env_action[0] = np.clip(cont_action[0], -1.0, 1.0)               # steer
    env_action[1] = np.clip(cont_action[1], 0.0, 1.0)                # accel (già Sigmoid)
    env_action[2] = np.clip(cont_action[2], 0.0, 1.0)                # brake (già Sigmoid)
    env_action[3] = float(max(0, min(6, gear)))                      # gear
    return env_action


def denormalize_action_rl(cont_action: np.ndarray, gear: int) -> np.ndarray:
    """Converte l'output RL (tutto Tanh [-1, 1]) nel formato TORCS.

    Mappatura:
      - steer: [-1, 1] → [-1, 1]  (diretto)
      - accel: [-1, 1] → [0, 1]   (affine: (x+1)/2)
      - brake: [-1, 1] → [0, 1]   (affine: (x+1)/2)
    """
    env_action = np.zeros(4, dtype=np.float32)
    env_action[0] = np.clip(cont_action[0], -1.0, 1.0)               # steer
    env_action[1] = np.clip((cont_action[1] + 1.0) / 2.0, 0.0, 1.0) # accel
    env_action[2] = np.clip((cont_action[2] + 1.0) / 2.0, 0.0, 1.0) # brake
    env_action[3] = float(max(0, min(6, gear)))                      # gear
    return env_action


# ──────────────────────────────────────────────────────────────────────
#  Auto-detect e caricamento pesi
# ──────────────────────────────────────────────────────────────────────

def load_best_weights(model, weights_arg, device, kind='auto'):
    """Carica i migliori pesi disponibili con auto-detect del formato.

    Priorità (se --weights non è specificato):
      1. td3_det_best_lap.pth (miglior GIRO VALIDO deterministico: candidato submission)
      2. td3_det_best_dist.pth (miglior policy ASSOLUTA per distanza; sopravvive a --clean)
      3. td3_det_best_dist_run.pth (miglior checkpoint deterministico del run corrente)
      4. td3_expl_best_lap.pth  (record sul giro TD3)
      5. td3_expl_best_dist.pth (record di distanza TD3)
      6. td3_policy.pth    (ultimo step TD3)
      7. bc_policy.pth     (fallback supervisionato)

    Se --weights è specificato, usa quello direttamente.

    Returns:
        (model, is_rl_weights: bool)
    """
    checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'train_set', 'checkpoints')
    td3_det_best_lap_path = os.path.join(checkpoint_dir, 'td3_det_best_lap.pth')
    td3_det_best_dist_path = os.path.join(checkpoint_dir, 'td3_det_best_dist.pth')
    td3_det_best_dist_run_path = os.path.join(checkpoint_dir, 'td3_det_best_dist_run.pth')
    td3_expl_best_lap_path = os.path.join(checkpoint_dir, 'td3_expl_best_lap.pth')
    td3_expl_best_dist_path = os.path.join(checkpoint_dir, 'td3_expl_best_dist.pth')
    td3_path = os.path.join(checkpoint_dir, 'td3_policy.pth')


    bc_path = os.path.join(checkpoint_dir, 'bc_policy.pth')

    # Se l'utente ha specificato un path esplicito, usalo
    if weights_arg:
        load_path = weights_arg
    elif os.path.exists(td3_det_best_lap_path):
        load_path = td3_det_best_lap_path
        _lt = ''
        _lt_txt = os.path.join(checkpoint_dir, 'td3_det_best_lap.txt')
        if os.path.exists(_lt_txt):
            try:
                with open(_lt_txt) as f: _lt = f' ({float(f.read().strip()):.3f}s)'
            except Exception: pass
        print(f"  Auto-detect: trovato td3_det_best_lap.pth (Miglior GIRO VALIDO deterministico{_lt} — candidato submission!)")
    elif os.path.exists(td3_det_best_dist_path):
        load_path = td3_det_best_dist_path
        print(f"  Auto-detect: trovato td3_det_best_dist.pth (Miglior policy ASSOLUTA per distanza, sopravvive ai --clean!)")
    elif os.path.exists(td3_det_best_dist_run_path):
        load_path = td3_det_best_dist_run_path
        print(f"  Auto-detect: trovato td3_det_best_dist_run.pth (Miglior checkpoint deterministico TD3!)")
    elif os.path.exists(td3_expl_best_lap_path):
        load_path = td3_expl_best_lap_path
        print(f"  Auto-detect: trovato td3_expl_best_lap.pth (Record sul giro TD3!)")
    elif os.path.exists(td3_expl_best_dist_path):
        load_path = td3_expl_best_dist_path
        print(f"  Auto-detect: trovato td3_expl_best_dist.pth (Record di distanza TD3!)")
    elif os.path.exists(td3_path):
        load_path = td3_path
        print(f"  Auto-detect: trovato td3_policy.pth (Ultimo step TD3)")

    elif os.path.exists(bc_path):
        load_path = bc_path
        print(f"  Auto-detect: fallback su bc_policy.pth")
    else:
        print(f"  Nessun file pesi trovato!")
        sys.exit(1)

    if not os.path.exists(load_path):
        print(f"  File pesi non trovato: {load_path}")
        sys.exit(1)

    try:
        state_dict = torch.load(load_path, map_location=device, weights_only=True)
    except Exception:
        state_dict = torch.load(load_path, map_location=device, weights_only=False)

    # strict=True solo se il file contiene già log_std_head (evita errori sui BC puliti)
    has_log_std = any('log_std_head' in k for k in state_dict.keys())
    model.load_state_dict(state_dict, strict=has_log_std)
    model.eval()

    # BC vs RL determina la mappatura delle azioni (RL: tanh→[0,1]; BC: sigmoid).
    # NON si può dedurre dai pesi (sia BC legacy sia Actor TD3 possono avere log_std_head).
    # Override esplicito con --kind {rl,bc}; in 'auto' si usa l'euristica sul nome file.
    if kind == 'rl':
        is_rl = True
    elif kind == 'bc':
        is_rl = False
    else:
        fname = os.path.basename(load_path).lower()
        if 'td3' in fname or 'sac' in fname:
            is_rl = True
        elif 'bc' in fname:
            is_rl = False
        else:
            print("   Tipo pesi non deducibile dal nome file: assumo "
                  f"{'RL' if has_log_std else 'BC'}. Usa --kind rl|bc per essere esplicito.")
            is_rl = has_log_std

    weight_type = "RL (TD3)" if is_rl else "BC"
    print(f"  Pesi [{weight_type}] caricati da: {load_path}")

    return model, is_rl


# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test Agent Autonomo (BC/TD3) — TORCS")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path ai pesi del modello (.pth). Se omesso, auto-detect.")
    parser.add_argument("--laps", type=int, default=3,
                        help="Numero di giri da completare")
    parser.add_argument("--max_steps", type=int, default=15000,
                        help="Max step per giro (timeout)")
    parser.add_argument("--kind", choices=["auto", "rl", "bc"], default="auto",
                        help="Tipo di pesi: 'rl' (tanh→[0,1]) o 'bc' (sigmoid). 'auto' deduce dal nome file.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'=' * 64}")
    print(f"  TEST AGENTE AUTONOMO (BC/TD3) — TORCS")
    print(f"  Device: {device}")
    print(f"  Stride Type: static (k=6, 0.24s)")
    print(f"  Modalità: DETERMINISTICA (evaluate=True, Zero Noise)")
    print(f"{'=' * 64}\n")

    # ── Carica modello con auto-detect ──
    model = BCActor().to(device)
    model, is_rl = load_best_weights(model, args.weights, device, kind=args.kind)

    # Seleziona la funzione di denormalizzazione corretta
    denormalize_fn = denormalize_action_rl if is_rl else denormalize_action_bc
    inference_mode = "RL (sample evaluate=True)" if is_rl else "BC (forward diretto)"
    print(f"  Inference mode: {inference_mode}")

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

            # Marcia deterministica (gearing.py), identica a training/eval
            current_gear = 1
            steps_since_shift = 999
            cur_speed_kmh = float(np.array(obs.get('speedX', 0.0)).flat[0]) * 50.0
            cur_rpm = float(np.array(obs.get('rpm', 0.0)).flat[0])
            stall_low_speed_steps = 0  # contatore stallo (#3: semantica allineata al training)

            print(f"\n  {'─' * 50}")
            print(f"  Tentativo #{total_attempts} (giri completati: {len(lap_times)}/{args.laps})")

            for step in range(1, args.max_steps + 1):
                # Costruisce il vettore di stato 87D concatenando t-12 (index 0), t-6 (index 6), t (index 12)
                stacked_state = np.concatenate([
                    state_buffer[0],
                    state_buffer[6],
                    state_buffer[12]
                ])

                # ── Inferenza DETERMINISTICA ──
                with torch.no_grad():
                    state_t = torch.FloatTensor(stacked_state).to(device).unsqueeze(0)

                    if is_rl:
                        # RL: usa sample(evaluate=True) per determinismo assoluto
                        tanh_action, _, gear_idx = model.sample(state_t, evaluate=True)
                        cont_action = tanh_action.cpu().numpy()[0]
                        _raw_gear = int(gear_idx.item())
                    else:
                        # BC: usa forward() con Tanh steer + Sigmoid accel/brake
                        pred_cont, gear_logits = model(state_t)
                        cont_action = pred_cont.cpu().numpy()[0]
                        _raw_gear = int(gear_logits.argmax(dim=1).item())

                # ── Mutual exclusion accel/brake (come l'esperto umano) ──
                # Per i pesi BC, cont_action[1:3] sono già [0,1] (Sigmoid)
                # Per i pesi RL, cont_action[1:3] sono [-1,1] (Tanh) — denormalize_fn li converte
                if not is_rl:
                    cont_action[1] = cont_action[1] * (1.0 - cont_action[2])

                # ── Costruzione azione (la marcia viene sovrascritta sotto) ──
                env_action = denormalize_fn(cont_action, current_gear)

                # Mutual exclusion post-denormalize per RL
                if is_rl:
                    env_action[1] = env_action[1] * (1.0 - env_action[2])

                # ── Marcia DETERMINISTICA (anti-hunting), identica a training/eval (gearing.py) ──
                # _raw_gear (gear_head congelata) è ignorato. Usa il gas APPLICATO (env_action[1]).
                current_gear, _shifted = compute_gear(cur_speed_kmh, float(env_action[1]), cur_rpm, current_gear, steps_since_shift)
                steps_since_shift = 0 if _shifted else steps_since_shift + 1
                env_action[3] = current_gear

                next_obs, _, env_done, _ = env.step(env_action)
                next_state = flatten_state(next_obs)
                cur_speed_kmh = float(np.array(next_obs.get('speedX', 0.0)).flat[0]) * 50.0
                cur_rpm = float(np.array(next_obs.get('rpm', 0.0)).flat[0])

                # Salva telemetria step
                dist_raw = next_obs.get('distFromStart', 0.0)
                if isinstance(dist_raw, np.ndarray): dist_raw = float(dist_raw.flat[0])
                dist_m = dist_raw
                # Velocità in km/h dall'obs GREZZO (make_observaton ha già fatto speedX/50), NON da
                # next_state[21]: quest'ultimo è ora normalizzato (apply_state_norm, mean/std), quindi
                # *50 darebbe un valore senza senso (negativo a velocità sotto-media → falso stallo).
                spd_kmh = float(np.array(next_obs.get('speedX', 0.0)).flat[0]) * 50.0
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

                # ── Giro NON valido oltre |trackPos| > 1.25 (taglio/muro), coerente col training e coi limiti di raccolta dati ──
                if abs(track_pos) > 1.25:
                    print(f"   Fuori pista / giro non valido allo step {step} (trackPos={track_pos:.3f})")
                    break
                if np.cos(angle) < 0:
                    print(f"   Spin allo step {step} (angle={angle:.3f})")
                    break

                # ── Stallo (#3: semantica allineata al training) ──
                # Training: dopo ~10s (terminal_judge_start=500 step) se la velocità in avanti
                # (speedX·cos) < 5 km/h → terminale. Qui lo replichiamo con una finestra di
                # conferma di ~1s per evitare falsi positivi da letture momentanee.
                fwd_kmh = spd_kmh * float(np.cos(angle))
                if step > 500 and fwd_kmh < 5.0:
                    stall_low_speed_steps += 1
                else:
                    stall_low_speed_steps = 0
                if stall_low_speed_steps >= 50:
                    print(f"   Stallo allo step {step} (vel. avanti {fwd_kmh:.1f} km/h)")
                    break

                # ── Telemetria ogni 200 passi ──
                if step % 200 == 0:
                    print(
                        f"    [Passo {step:4d}] posizione pista={track_pos:+.3f} | "
                        f"velocità={spd_kmh:.0f} km/h | sterzo={env_action[0]:+.3f} | "
                        f"acceleratore={env_action[1]:.2f} | freno={env_action[2]:.2f} | "
                        f"marcia={int(env_action[3])}"
                    )

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
            print(f"  Telemetria del tentativo salvata in: {csv_path}")

            if lap_completed:
                lap_times.append(lap_time)
                print(f"  GIRO COMPLETATO: {lap_time:.3f}s")
            else:
                print(f"  Fallito (step: {step})")
                print("   Tentativo fallito. Prossimo tentativo...")
                # Continua con il prossimo tentativo (il while loop riproverà con relaunch=True)

    except KeyboardInterrupt:
        print(f"\n  Test interrotto.")
    finally:
        env.end()

    # ── Riepilogo ──
    print(f"\n{'=' * 64}")
    print(f"  RIEPILOGO TEST")
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
