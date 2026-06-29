"""
Script for deterministic testing of the agent's autonomous-driving performance on TORCS.

This script's purpose is to load a trained model (in Behavioral Cloning or TD3+BC format),
connect to the TORCS simulator via the Gym wrapper, and let the agent drive. The test loop runs
in deterministic mode to evaluate real performance, without stochastic exploration, allowing to
quantify metrics such as lap time, off-track or stalls.

Key mechanisms implemented to guarantee a deterministic, reliable test:
  1. Evaluation state (model.eval()): puts the model in inference mode. The Actor uses no Dropout;
     LayerNorm is deterministic, but eval mode keeps the path consistent with testing.
  2. Noise removal: the action is determined in pure tanh(mean) mode, excluding any Gaussian or
     Ornstein-Uhlenbeck exploration noise used during training.
  3. Deterministic gear shifting: gear selection is delegated entirely to the 'gearing.py' module,
     based on speed in km/h, engine revs (RPM) and the applied throttle level.
  4. Physical environment relaunch: at each lap attempt the TORCS environment is reset with
     relaunch=True to force a full simulator reload, cleaning the physics-engine state.

Weight auto-detection hierarchy (in the absence of an explicit --weights argument):
  1. td3_det_best_lap.pth      -> Fastest valid deterministic lap recorded (submission candidate).
  2. td3_det_best_dist.pth     -> Deterministic RL policy with the best eval score (distance or time-equivalent).
  3. td3_det_best_dist_run.pth -> Best deterministic checkpoint of the current training session.
  4. td3_expl_best_lap.pth     -> Best lap time obtained during the RL exploration phases.
  5. td3_expl_best_dist.pth    -> Max distance record obtained during the RL exploration phases.
  6. td3_policy.pth            -> Last policy saved at the end of the RL training steps.
  7. bc_policy.pth             -> Initial model trained with Behavioral Cloning only (supervised).

Usage examples:
  python test_agent.py                                           # Auto-detects and runs the best checkpoint
  python test_agent.py --weights train_set/checkpoints/bc_policy.pth  # Forces loading the BC weights
  python test_agent.py --laps 5                                  # Sets the test for 5 complete laps
"""

import os
import sys
import argparse
import csv
import numpy as np
import torch
import torch.nn as nn
from collections import deque

# Iterative Motors: package (environment, state utility, networks, action mapping)
_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
from iterative_motors.env.gym_torcs import TorcsEnv
from iterative_motors.env.gearing import compute_gear  # deterministic gear shifting (anti-hunting)
from iterative_motors.common.state import apply_state_norm, flatten_state_norm as flatten_state
from iterative_motors.common.constants import CHECKPOINT_ROOT, TELEMETRY_DIR
from iterative_motors.models.networks import PolicyNetwork as PolicyActor
from iterative_motors.models.action_mapping import (
    bc_to_pedals as denormalize_action_bc, rl_to_pedals as denormalize_action_rl,
)

# Reproducibility
# Forces the GPU to perform deterministic computations (slower)
torch.backends.cudnn.deterministic = True

# Prevents the GPU from choosing the best algorithm but with the possibility of changing it each time (non-deterministic)
torch.backends.cudnn.benchmark = False




import time


#  Auto-detect and weight loading

def load_best_weights(model, weights_arg, device, kind='auto'):
    """
    Automatically detects and loads the best available Actor weights, identifying their nature (BC or TD3).

    How it works:
      - If the user specifies a path via '--weights', that file is loaded directly.
      - If '--weights' is None, it scans the checkpoint folder in decreasing order of importance to find the best file.
      - It loads the state_dict and filters the keys (filtered_state) to load only the weights compatible with
        the current Actor architecture, ignoring any Critic weights present in the file.
      - It detects whether the model is TD3+BC or BC based on the file name (presence of the 'td3' or 'bc' string), or
        using the explicit '--kind' argument. This determines the subsequent action denormalization.

    Args:
        model: Instance of the PolicyActor class to load.
        weights_arg: String of the weights path (optional).
        device: Device on which to load the model ('cuda' or 'cpu').
        kind: Format-selection string ('auto', 'td3', 'rl', or 'bc').
    Returns:
        A tuple (model, is_rl: bool) indicating the loaded model and whether it is a TD3 policy.
    """
    checkpoint_dir = CHECKPOINT_ROOT
    td3_det_best_lap_path = os.path.join(checkpoint_dir, 'td3_det_best_lap.pth')
    td3_det_best_dist_path = os.path.join(checkpoint_dir, 'td3_det_best_dist.pth')
    td3_det_best_dist_run_path = os.path.join(checkpoint_dir, 'td3_det_best_dist_run.pth')
    td3_expl_best_lap_path = os.path.join(checkpoint_dir, 'td3_expl_best_lap.pth')
    td3_expl_best_dist_path = os.path.join(checkpoint_dir, 'td3_expl_best_dist.pth')
    td3_path = os.path.join(checkpoint_dir, 'td3_policy.pth')
    bc_path = os.path.join(checkpoint_dir, 'bc_policy.pth')

    # Priority 1: explicit path provided by the user
    if weights_arg:
        load_path = weights_arg
    # Priority 2: best valid deterministic lap (the ideal file for testing)
    elif os.path.exists(td3_det_best_lap_path):
        load_path = td3_det_best_lap_path
        _lt = ''
        _lt_txt = os.path.join(checkpoint_dir, 'td3_det_best_lap.txt')
        if os.path.exists(_lt_txt):
            try:
                with open(_lt_txt) as f: _lt = f' ({float(f.read().strip()):.3f}s)'
            except Exception: pass
        print(f"  Auto-detect: trovato td3_det_best_lap.pth (Miglior GIRO VALIDO deterministico{_lt} — candidato submission!)")
    # Priority 3: best overall distance (survived --clean cleanups)
    elif os.path.exists(td3_det_best_dist_path):
        load_path = td3_det_best_dist_path
        print(f"  Auto-detect: trovato td3_det_best_dist.pth (Miglior policy ASSOLUTA per distanza, sopravvive ai --clean!)")
    # Priority 4: best checkpoint of the current RL session
    elif os.path.exists(td3_det_best_dist_run_path):
        load_path = td3_det_best_dist_run_path
        print(f"  Auto-detect: trovato td3_det_best_dist_run.pth (Miglior checkpoint deterministico TD3+BC!)")
    # Priority 5: checkpoint with the best exploration lap
    elif os.path.exists(td3_expl_best_lap_path):
        load_path = td3_expl_best_lap_path
        print(f"  Auto-detect: trovato td3_expl_best_lap.pth (Record sul giro TD3+BC!)")
    # Priority 6: checkpoint with the best exploration distance
    elif os.path.exists(td3_expl_best_dist_path):
        load_path = td3_expl_best_dist_path
        print(f"  Auto-detect: trovato td3_expl_best_dist.pth (Record di distanza TD3+BC!)")
    # Priority 7: last saved policy
    elif os.path.exists(td3_path):
        load_path = td3_path
        print(f"  Auto-detect: trovato td3_policy.pth (Ultimo step TD3+BC)")
    # Priority 8: Behavioral Cloning weights (base supervised)
    elif os.path.exists(bc_path):
        load_path = bc_path
        print(f"  Auto-detect: uso bc_policy.pth come ultima priorità supervisionata")
    else:
        print(f"  Nessun file pesi trovato!")
        sys.exit(1)

    if not os.path.exists(load_path):
        print(f"  File pesi non trovato: {load_path}")
        sys.exit(1)

    # Safely load the PyTorch weights onto the specified device (GPU or CPU)
    try:
        loaded = torch.load(load_path, map_location=device, weights_only=True)
    except Exception:
        loaded = torch.load(load_path, map_location=device, weights_only=False)

    # Extract the Actor state (if saved inside a dictionary with additional keys like optim or epoch)
    state_dict = loaded.get('actor', loaded) if isinstance(loaded, dict) else loaded
    if not hasattr(state_dict, 'items'):
        print(f"  File pesi non adatto all'Actor corrente: {load_path}")
        sys.exit(1)

    # Filter and match the keys to ensure compatibility with the current module's Actor
    model_state = model.state_dict()
    filtered_state = {
        key: value for key, value in state_dict.items()
        if key in model_state and hasattr(value, 'shape') and model_state[key].shape == value.shape
    }
    if not filtered_state:
        print(f"  File pesi non adatto all'Actor corrente: {load_path}")
        sys.exit(1)

    model.load_state_dict(filtered_state, strict=False)
    model.eval()

    # Automatic or manual detection of the policy type (TD3 vs BC)
    if kind in ('td3', 'rl'):
        is_rl = True
    elif kind == 'bc':
        is_rl = False
    else:
        fname = os.path.basename(load_path).lower()
        if 'td3' in fname:
            is_rl = True
        elif 'bc' in fname:
            is_rl = False
        else:
            print("   Tipo pesi non deducibile dal nome file. Usa --kind td3|bc per essere esplicito.")
            sys.exit(1)

    weight_type = "TD3+BC" if is_rl else "BC"
    print(f"  Pesi [{weight_type}] caricati da: {load_path}")

    return model, is_rl


# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test Agent Autonomo (BC/TD3+BC) — TORCS")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path ai pesi del modello (.pth). Se omesso, auto-detect.")
    parser.add_argument("--laps", type=int, default=3,
                        help="Numero di giri da completare")
    parser.add_argument("--max_steps", type=int, default=15000,
                        help="Max step per giro (timeout)")
    parser.add_argument("--kind", choices=["auto", "td3", "rl", "bc"], default="auto",
                        help="Tipo di pesi: 'td3' (tanh→[0,1]) o 'bc' (sigmoid). 'rl' resta alias compatibile.")
    parser.add_argument("--gui_hold_seconds", type=float, default=None,
                        help="Secondi extra prima di chiudere TORCS dopo un giro completato con SHOW_GUI=1. "
                             "Default: env IM_GUI_HOLD_SECONDS o 8s in GUI, 0s headless.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    show_gui = os.environ.get('SHOW_GUI', '0') == '1'
    if args.gui_hold_seconds is None:
        gui_hold_seconds = float(os.environ.get('IM_GUI_HOLD_SECONDS', '8.0')) if show_gui else 0.0
    else:
        gui_hold_seconds = max(0.0, float(args.gui_hold_seconds))

    print(f"\n{'=' * 64}")
    print(f"  TEST AGENTE AUTONOMO (BC/TD3+BC) — TORCS")
    print(f"  Device: {device}")
    print(f"  Stride Type: static (k=6, 0.24s)")
    print(f"  Modalità: DETERMINISTICA (evaluate=True, Zero Noise)")
    print(f"{'=' * 64}\n")

    # Load the agent network and place the best weights (auto-detect)
    model = PolicyActor().to(device)
    model, is_rl = load_best_weights(model, args.weights, device, kind=args.kind)

    # Select the denormalization function and the description based on the loaded policy
    denormalize_fn = denormalize_action_rl if is_rl else denormalize_action_bc
    inference_mode = "TD3+BC (sample evaluate=True)" if is_rl else "BC (forward diretto)"
    print(f"  Inference mode: {inference_mode}")

    # Initialize the TORCS simulator connection client
    print("  Inizializzazione TORCS...")
    env = TorcsEnv(early_termination=False)

    lap_times = []
    total_attempts = 0

    try:
        # The loop continues until we complete the requested number of valid laps
        while len(lap_times) < args.laps:
            total_attempts += 1

            # Forced relaunch of the TORCS environment at every reset to prevent accumulation of physics errors
            obs = env.reset(relaunch=True)
            initial_state = flatten_state(obs)

            # State Stacking (Fujimoto 2021): the neural network needs 3 concatenated temporal frames.
            # The buffer stores the last 13 frames (corresponding to 0.24 seconds total, with k=6).
            # At startup, the buffer is initialized by replicating the initial state.
            state_buffer = deque(maxlen=13)
            for _ in range(13):
                state_buffer.append(initial_state)

            # Detection of the initial lap times
            raw = env.client.S.d
            prev_last_lap = float(raw.get('lastLapTime', 0.0))
            if isinstance(prev_last_lap, list): prev_last_lap = prev_last_lap[0]

            lap_completed = False
            lap_time = 0.0
            telemetry_data = []

            # Initial state of the deterministic gear (anti-hunting, preventing continuous gear oscillations)
            current_gear = 1
            steps_since_shift = 999
            cur_speed_kmh = float(np.array(obs.get('speedX', 0.0)).flat[0]) * 50.0
            cur_rpm = float(np.array(obs.get('rpm', 0.0)).flat[0])
            stall_low_speed_steps = 0  # Counter for prolonged low-speed stalls

            print(f"\n  {'─' * 50}")
            print(f"  Tentativo #{total_attempts} (giri completati: {len(lap_times)}/{args.laps})")

            for step in range(1, args.max_steps + 1):
                # We build the 87D input vector:
                # - state_buffer[0]  -> frame t-12 (0.24 seconds ago)
                # - state_buffer[6]  -> frame t-6 (0.12 seconds ago)
                # - state_buffer[12] -> frame t (current)
                stacked_state = np.concatenate([
                    state_buffer[0],
                    state_buffer[6],
                    state_buffer[12]
                ])

                # Pure deterministic inference, without adding exploration noise
                with torch.no_grad():
                    state_t = torch.FloatTensor(stacked_state).to(device).unsqueeze(0)

                    if is_rl:
                        # TD3+BC weights: produce 3 commands in [-1, 1] via hyperbolic tangent (Tanh)
                        tanh_action = model.sample(state_t, evaluate=True)
                        cont_action = tanh_action.cpu().numpy()[0]
                    else:
                        # Behavioral Cloning weights: produce steering in [-1, 1] and throttle/brake in [0, 1] (Sigmoid)
                        pred_cont = model(state_t)
                        cont_action = pred_cont.cpu().numpy()[0]

                # Human mutual-exclusion rule (we never press throttle and brake at the same time)
                # For the BC policy we apply mutual exclusion before denormalization since the values are already in [0,1]
                if not is_rl:
                    cont_action[1] = cont_action[1] * (1.0 - cont_action[2])

                # Convert the continuous outputs into the format compatible with the TORCS environment
                env_action = denormalize_fn(cont_action)

                # For the RL policy we apply mutual exclusion after denormalization (when the action is scaled to [0,1])
                if is_rl:
                    env_action[1] = env_action[1] * (1.0 - env_action[2])

                # Deterministic gear computation via gearing.py.
                # Note: we pass the actual APPLIED throttle command (env_action[1]) to avoid false shifts under braking.
                current_gear, _shifted = compute_gear(cur_speed_kmh, float(env_action[1]), cur_rpm, current_gear, steps_since_shift)
                steps_since_shift = 0 if _shifted else steps_since_shift + 1
                env_action[3] = current_gear

                # We execute the physics simulation step in TORCS
                next_obs, _, env_done, _ = env.step(env_action)
                next_state = flatten_state(next_obs)
                cur_speed_kmh = float(np.array(next_obs.get('speedX', 0.0)).flat[0]) * 50.0
                cur_rpm = float(np.array(next_obs.get('rpm', 0.0)).flat[0])

                # Data extraction for local telemetry
                dist_raw = next_obs.get('distFromStart', 0.0)
                if isinstance(dist_raw, np.ndarray): dist_raw = float(dist_raw.flat[0])
                dist_m = dist_raw

                # Raw speed in km/h. Note: we do not extract from next_state since it is normalized (mean/std)
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

                # Lap-invalidation criterion 1: full off-track (trackPos > 1.25, curb tolerance)
                if abs(track_pos) > 1.25:
                    print(f"   Fuori pista / giro non valido allo step {step} (trackPos={track_pos:.3f})")
                    break

                # Lap-invalidation criterion 2: car spin (negative cosine of the angle relative to the track)
                if np.cos(angle) < 0:
                    print(f"   Spin allo step {step} (angle={angle:.3f})")
                    break

                # Lap-invalidation criterion 3: vehicle stall.
                # If the car moves at less than 5 km/h for over 50 steps (~1 second of simulation) after the first 500 steps,
                # we consider the car stuck or bogged down and abort the attempt.
                fwd_kmh = spd_kmh * float(np.cos(angle))
                if step > 500 and fwd_kmh < 5.0:
                    stall_low_speed_steps += 1
                else:
                    stall_low_speed_steps = 0
                if stall_low_speed_steps >= 50:
                    print(f"   Stallo allo step {step} (vel. avanti {fwd_kmh:.1f} km/h)")
                    break

                # Print intermediate telemetry every 200 steps for live monitoring
                if step % 200 == 0:
                    print(
                        f"    [Passo {step:4d}] posizione pista={track_pos:+.3f} | "
                        f"velocità={spd_kmh:.0f} km/h | sterzo={env_action[0]:+.3f} | "
                        f"acceleratore={env_action[1]:.2f} | freno={env_action[2]:.2f} | "
                        f"marcia={int(env_action[3])}"
                    )

                # Check whether the agent crossed the finish line completing a lap.
                # Compares the current lastLapTime in the simulator against the one saved at the start of the lap.
                current_last_lap = float(raw.get('lastLapTime', 0.0))
                if isinstance(current_last_lap, list): current_last_lap = current_last_lap[0]

                if current_last_lap > 0.0 and abs(current_last_lap - prev_last_lap) > 0.01:
                    lap_completed = True
                    lap_time = current_last_lap
                    break

                # Advance the temporal buffer and update the state
                state_buffer.append(next_state)
                obs = next_obs
                if env_done: break

            # Write telemetry to a CSV file at the end of each attempt
            telemetry_dir = TELEMETRY_DIR
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

    except KeyboardInterrupt:
        print(f"\n  Test interrotto.")
    finally:
        if show_gui and lap_times and gui_hold_seconds > 0.0:
            print(
                f"\n  SHOW_GUI=1: tengo TORCS aperto per {gui_hold_seconds:.1f}s "
                "per mostrare/salvare il tempo del giro..."
            )
            time.sleep(gui_hold_seconds)
        env.end()

    # Final summary of the performance of all the attempts made
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
