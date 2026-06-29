"""
Data Collection, script for collecting human data on the track. It allows the use of a controller or the keyboard.

Records the single laps completed without going off track, the laps of the problematic zones (corners)
and also laps with data collection exclusive to some segments of the track.

The laps are automatically saved into separate HDF5 files.

Output format:
    lap_001.h5, lap_002.h5, ...              (one file per valid complete lap)
    lap_seg_001.h5, ...                  (targeted segments, only with --segment_only)
    lap_seg_550m_900m.h5, ...             (specific segments for targeted data collection)
    session_logs/giri/session_YYYYMMDD.log   (textual session log)

Each HDF5 saves the 29D state, the action [steer, accel, brake, gear] and dist_from_start as metadata.
dist_from_start is used for segmentation/analysis, it does not enter the network; this prevents the model from learning to correlate the position with the action.
ensuring that it learns to drive based on the sensors and not on the position on the track.
"""

import os
import sys
import time
import argparse
import numpy as np
import h5py
import pygame   #Library for interacting with the controller
from datetime import datetime

# Force the TORCS GUI to be shown for data collection
os.environ['SHOW_GUI'] = '1'

# Iterative Motors: package (environment + flatten_state RAW like the HDF5 files).
_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
from iterative_motors.common.state import flatten_state_raw as flatten_state

try:
    from iterative_motors.env.gym_torcs import TorcsEnv
except ImportError as e:
    print(f"ERRORE FATALE: Impossibile importare gym_torcs o una sua dipendenza.")
    print(f"Dettagli errore: {e}")
    sys.exit(1)


# Class for handling the PS5 DualSense controller
class DualSenseController:
    """Handles the polling of the PlayStation 5 controller via Pygame.

    Mapping:
        Left Stick X   → Continuous steering (with configurable deadzone)
        R2 (axis 5)    → Throttle [0, 1]
        L2 (axis 2)    → Brake [0, 1]
        Square         → Upshift
        X              → Downshift
    """

    # Axis mapping
    AXIS_STEER = 0
    AXIS_L2 = 2       # Brake
    AXIS_R2 = 5       # Throttle

    # Button mapping
    BTN_CROSS = 0      # Downshift
    BTN_SQUARE = 3     # Upshift

    # Debounce for the gear-change buttons
    DEBOUNCE_MS = 200

    def __init__(self, steering_deadzone: float = 0.05):
        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            raise RuntimeError("Nessun controller rilevato. Collega un controller e riprova.")

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        print(f"  Controller inizializzato: {self.joystick.get_name()}")

        self.steering_deadzone = steering_deadzone
        self.gear = 1  # Start in first gear

        # Warm-up flags for the triggers (prevents spurious values before the first press)
        self._r2_initialized = False
        self._l2_initialized = False

        # Timestamp of the last gear change (debounce)
        self._last_shift_time = 0

    # Function that reads the controller and returns the action [steering, throttle, brake, gear]
    def get_action(self) -> np.ndarray:

        # Drain the Pygame event queue to avoid saturating it (causes input lag)
        pygame.event.clear()

        # Steering handling with deadzone
        raw_steer = -self.joystick.get_axis(self.AXIS_STEER)
        if abs(raw_steer) < self.steering_deadzone:
            steering = 0.0
        else:
            # Rescale the post-deadzone range to [-1, 1]
            sign = 1.0 if raw_steer > 0 else -1.0
            steering = sign * (abs(raw_steer) - self.steering_deadzone) / (1.0 - self.steering_deadzone)

        # Throttle handling (R2) with warm-up protection
        raw_r2 = self.joystick.get_axis(self.AXIS_R2)
        if not self._r2_initialized:
            if abs(raw_r2) > 0.1:
                self._r2_initialized = True
            accel = 0.0
        else:
            accel = max(0.0, (raw_r2 + 1.0) / 2.0)
            if accel < 0.05:
                accel = 0.0

        # Brake handling (L2) with warm-up protection
        raw_l2 = self.joystick.get_axis(self.AXIS_L2)
        if not self._l2_initialized:
            if abs(raw_l2) > 0.1:
                self._l2_initialized = True
            brake = 0.0
        else:
            brake = max(0.0, (raw_l2 + 1.0) / 2.0)
            if brake < 0.05:
                brake = 0.0

        # Gear change with temporal debounce
        now = pygame.time.get_ticks()
        if now - self._last_shift_time > self.DEBOUNCE_MS:
            if self.joystick.get_button(self.BTN_SQUARE):
                if self.gear < 6:
                    self.gear += 1
                    print(f"  [Gear] ⬆ Marcia {self.gear}")
                self._last_shift_time = now
            elif self.joystick.get_button(self.BTN_CROSS):
                if self.gear > 1:  # Minimum gear 1: data collection does not use reverse
                    self.gear -= 1
                    print(f"  [Gear] ⬇ Marcia {self.gear}")
                self._last_shift_time = now

        return np.array([steering, accel, brake, float(self.gear)], dtype=np.float32)

    def rumble(self, intensity: float = 0.3, duration_ms: int = 180):
        """Triggers a short haptic feedback, if the controller supports it."""
        try:
            self.joystick.rumble(0.0, float(min(0.5, intensity)), int(duration_ms))
        except Exception:
            pass  # If rumble is not supported, do nothing


class KeyboardController:
    """Handles driving TORCS via the keyboard (WASD + Arrows).

    Requires a small Pygame window open and focused to record the keys.
    """
    DEBOUNCE_MS = 250

    def __init__(self):
        pygame.init()
        # Minimal window to capture the Pygame inputs
        self.screen = pygame.display.set_mode((100, 100))
        pygame.display.set_caption("Input Focus")

        self.gear = 1
        self.steer_val = 0.0
        self._last_shift_time = 0
        print("  [Keyboard] Inizializzato. MANTIENI IL FOCUS sulla finestra nera 'Input Focus' per guidare!")

    def rumble(self, intensity: float = 0.3, duration_ms: int = 180):
        """No-op: the keyboard has no haptic feedback."""
        pass

    def get_action(self) -> np.ndarray:
        # Process the Pygame events to keep the window active and responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)

        keys = pygame.key.get_pressed()

        # Gradual steering (smooth interpolation) for fluid driving
        steer_target = 0.0
        if keys[pygame.K_a]:
            steer_target = 1.0  # +1.0 in TORCS turns left (Left)
        elif keys[pygame.K_d]:
            steer_target = -1.0  # -1.0 in TORCS turns right (Right)

        # Move toward the target
        if self.steer_val < steer_target:
            self.steer_val = min(steer_target, self.steer_val + 0.08)
        elif self.steer_val > steer_target:
            self.steer_val = max(steer_target, self.steer_val - 0.08)

        # Responsive digital throttle and brake
        accel = 1.0 if keys[pygame.K_w] else 0.0
        brake = 1.0 if keys[pygame.K_s] else 0.0

        # Priority to the brake in case of simultaneous press
        if brake > 0.1:
            accel = 0.0

        # Gear change with debounce protection
        now = pygame.time.get_ticks()
        if now - self._last_shift_time > self.DEBOUNCE_MS:
            if keys[pygame.K_UP]:
                if self.gear < 6:
                    self.gear += 1
                    print(f"  [Gear] ⬆ Marcia {self.gear}")
                self._last_shift_time = now
            elif keys[pygame.K_DOWN]:
                if self.gear > 1:
                    self.gear -= 1
                    print(f"  [Gear] ⬇ Marcia {self.gear}")
                self._last_shift_time = now

        # Update the screen to prevent the operating system from seeing the window as blocked/frozen
        self.screen.fill((30, 30, 40))  # Minimal dark gray
        pygame.display.flip()

        return np.array([self.steer_val, accel, brake, float(self.gear)], dtype=np.float32)


# The track's problematic zones are the corners; we extracted these values from the torcs files,
# in particular from corkscrew.xml, and then slightly widened the zones to also capture the braking points and corner exits
PROBLEM_ZONES = [
    (340, 530), (670, 810), (940, 1070), (1420, 1590), (1870, 1980),
    (2380, 2530), (2570, 2780), (2890, 3020), (3190, 3300),
]

# Converts the zones written by the user (if specified at the start of data collection via --zones) into a list of (start,end) tuples
# Otherwise it uses the predefined PROBLEM_ZONES
def _parse_zones(spec):
    """Converts 'a:b,c:d' into [(a,b),(c,d)]. None/'' → default PROBLEM_ZONES."""
    if not spec:
        return list(PROBLEM_ZONES)
    out = []
    for part in spec.split(','):
        a, b = part.split(':')
        out.append((float(a), float(b)))
    return out

# Checks whether the distance falls within a problematic zone and returns the zone index
def _zone_index(dist, zones):
    """Index of the zone containing 'dist', otherwise None."""
    for zi, (a, b) in enumerate(zones):
        if a <= dist <= b:
            return zi
    return None

# Extracts the contiguous runs of in-zone steps, with an approach margin
def _extract_segments(dists, zones, margin_steps=15):
    """Contiguous runs of in-zone steps, with an approach margin. Returns [(start,end), ...] (end excluded)."""
    n = len(dists)
    in_zone = [(_zone_index(d, zones) is not None) for d in dists]
    segs = []
    i = 0
    while i < n:
        if in_zone[i]:
            j = i
            while j < n and in_zone[j]:
                j += 1
            segs.append((max(0, i - margin_steps), j))
            i = j
        else:
            i += 1
    return segs




# Extracts distFromStart as a scalar float from the observation.
def _get_dist_from_start(obs: dict) -> float:
    """Extracts distFromStart as a scalar float from the observation."""
    dfs = obs.get('distFromStart', 0.0)
    if isinstance(dfs, np.ndarray):
        return float(dfs.flat[0])
    return float(dfs)

# Extracts curLapTime as a scalar float from the observation.
def _get_cur_lap_time(obs: dict) -> float:
    """Extracts curLapTime as a scalar float from the observation."""
    clt = obs.get('curLapTime', 0.0)
    if isinstance(clt, np.ndarray):
        return float(clt.flat[0])
    return float(clt)

# Extracts lastLapTime as a scalar float from the observation.
def _get_last_lap_time(obs: dict) -> float:
    """Extracts lastLapTime as a scalar float from the observation."""
    llt = obs.get('lastLapTime', 0.0)
    if isinstance(llt, np.ndarray):
        return float(llt.flat[0])
    return float(llt)



# Function that applies the Traction Control System (TCS); it helps us do better laps during data collection
def apply_tcs(action: np.ndarray, obs: dict, slip_threshold: float = 5.0) -> np.ndarray:
    """Traction Control System — reduces the throttle in case of slipping.

    Compares the angular velocity of the rear vs front wheels.
    If the difference exceeds the threshold, it scales the throttle proportionally.

    Args:
        action: [steering, accel, brake, gear]
        obs: TORCS observation dictionary (contains wheelSpinVel)
        slip_threshold: spin difference beyond which the TCS intervenes

    Returns:
        the action modified with reduced throttle if necessary
    """
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
        # Progressive reduction: more slip → more cut
        # From 1.0 (no cut) to 0.2 (maximum cut 80%)
        reduction = max(0.2, 1.0 - (slip - slip_threshold) / 30.0)
        action = action.copy()
        action[1] *= reduction  # Scale the throttle

    return action


# Main data-collection loop
def main():
    # Parsing of the various arguments accepted by data_collection.py
    parser = argparse.ArgumentParser(
        description="Data Collection TORCS — Giro Secco con controller PS5"
    )
    parser.add_argument(
        "--output_dir", type=str, default="train_set",
        help="Directory di output per i file HDF5 e il log (default: directory corrente)"
    )
    parser.add_argument(
        "--device", type=str, choices=["controller", "keyboard"], default="controller",
        help="Dispositivo di input: 'controller' (PS5 DualSense) o 'keyboard' (tastiera WASD)"
    )
    parser.add_argument(
        "--steering_deadzone", type=float, default=0.05,
        help="Deadzone dello sterzo [0.0-0.2] (default: 0.05)"
    )
    parser.add_argument(
        "--relaunch_every", type=int, default=10,
        help="Rilancia TORCS ogni N giri per prevenire memory leak (default: 10)"
    )
    parser.add_argument(
        "--tcs", action="store_true", default=True,
        help="Abilita il Traction Control System (default: abilitato)"
    )
    parser.add_argument(
        "--no-tcs", dest="tcs", action="store_false",
        help="Disabilita il Traction Control System"
    )
    parser.add_argument(
        "--tcs_slip", type=float, default=5.0,
        help="Soglia di slip del TCS (default: 5.0)"
    )
    parser.add_argument(
        "--zones", type=str, default=None,
        help="Zone curva target (distFromStart in metri) come 'a:b,c:d'. Default: PROBLEM_ZONES auto-rilevate per geometria."
    )
    parser.add_argument(
        "--segment_only", action="store_true",
        help="Salva SOLO i segmenti dentro le zone (raccolta parziale): guidi giri interi, vengono tenute solo le curve strette."
    )
    args = parser.parse_args()


    sys.argv = [sys.argv[0]]

    # Lap and corner data folder
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Target corner zones (focused collection)
    zones = _parse_zones(args.zones)
    print(f"\n  Zone curva target ({len(zones)}): " + ", ".join(f"{int(a)}-{int(b)}m" for a, b in zones))
    if args.segment_only:
        print(f"  Modalità SEGMENT_ONLY: salvo solo i segmenti dentro le zone (guidi giri interi).")
    print(f"  Vibrazione gentile del controller all'ingresso di ogni zona.\n")
    laps_dir = os.path.join(output_dir, "laps")
    os.makedirs(laps_dir, exist_ok=True)

    # Data-collection session log: distinct from the three top-level logs of the training pipeline.
    log_dir = os.path.join(output_dir, "session_logs", "giri")
    os.makedirs(log_dir, exist_ok=True)
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"session_{session_id}.log")

    # Controller
    if args.device == "keyboard":
        controller = KeyboardController()
    else:
        try:
            controller = DualSenseController(steering_deadzone=args.steering_deadzone)
        except RuntimeError as e:
            print(f"  Errore controller: {e}")
            print("  Vuoi usare la tastiera? Avvia con: python data_collection.py --device keyboard")
            sys.exit(1)

    # Count the laps already existing in the directory for continuous numbering
    existing_laps = sorted([
        f for f in os.listdir(laps_dir)
        if f.startswith("lap_") and f.endswith(".h5")
    ])
    lap_counter = len(existing_laps)

    # Session statistics
    session_saved = 0
    session_discarded = 0

    print()
    print("=" * 64)
    print("   DATA COLLECTION — Giro Secco TORCS")
    print("   Premi Ctrl+C nel terminale per terminare la sessione")
    print("=" * 64)

    # Initialize the environment
    env = TorcsEnv(early_termination=False)

    TARGET_DT = 1.0 / 50.0  # 50 Hz target
    lap_attempt = 0
    force_relaunch = False

    try:
        while True:
            lap_attempt += 1

            # Environment reset
            # Periodic relaunch, on the first lap, or if requested (e.g. off track)
            need_relaunch = (lap_attempt == 1) or (lap_attempt % args.relaunch_every == 0) or force_relaunch
            if lap_attempt == 1:
                ob = env.reset(relaunch=True)
            else:
                ob = env.reset(relaunch=need_relaunch)

            force_relaunch = False  # Reset the flag after use

            state_vec = flatten_state(ob)

            # RAM buffer for this lap
            lap_states: list = []
            lap_actions: list = []
            lap_dists: list = []  # distFromStart per step (METADATA: does NOT enter the 29D states)
            active_zone_idx = None  # current zone index (for the rumble on entry)
            entered_any_zone = False  # Tracks whether we entered at least one target zone


            # Lap validity state
            lap_valid = True
            invalidation_reason = ""
            lap_completed = False
            lap_time = 0.0
            went_off_track = False

            # Initial snapshot of timing and position to detect the transition
            prev_last_lap_time = _get_last_lap_time(ob)
            prev_cur_lap_time = _get_cur_lap_time(ob)
            prev_dist = _get_dist_from_start(ob)

            # Gear reset
            controller.gear = 1

            print(f"\n{'─' * 64}")
            print(f"  TENTATIVO GIRO #{lap_attempt}  (giri salvati finora: {lap_counter})")
            print(f"  Status: [VALIDO]")
            print(f"{'─' * 64}")

            step = 0

            while True:
                loop_start = time.perf_counter()
                step += 1

                # Controller polling
                action = controller.get_action()

                # Apply the tcs
                if args.tcs:
                    action = apply_tcs(action, ob, slip_threshold=args.tcs_slip)

                # Simulation step
                ob_next, reward, done, info = env.step(action)
                next_state_vec = flatten_state(ob_next)

                # Accumulate the states, actions and distances in RAM
                lap_states.append(state_vec.copy())
                lap_actions.append(action.copy())
                lap_dists.append(_get_dist_from_start(ob))


                state_vec = next_state_vec
                ob = ob_next

                # Off-track check
                current_track_pos = ob_next.get('trackPos', 0.0)
                if isinstance(current_track_pos, np.ndarray):
                    current_track_pos = current_track_pos.flat[0]

                # Limit to allow more aggressive driving on the curbs.
                if abs(current_track_pos) > 1.25:
                    print(f"\n  [OFF-TRACK] trackPos: {current_track_pos:.2f} - Riavvio immediato simulazione.")
                    went_off_track = True
                    lap_completed = True
                    lap_valid = False
                    invalidation_reason = f"Fuori pista (trackPos: {current_track_pos:.2f})"
                    force_relaunch = True
                    break

                # Lap-completion detection
                current_last_lap = _get_last_lap_time(ob_next)
                current_cur_lap = _get_cur_lap_time(ob_next)
                current_dist = _get_dist_from_start(ob_next)

                # Focused collection in the zones
                cur_zone = _zone_index(current_dist, zones)
                if cur_zone is not None and cur_zone != active_zone_idx:
                    controller.rumble(intensity=0.3, duration_ms=180) # Vibration on entry of each target zone.
                active_zone_idx = cur_zone

                if cur_zone is not None:
                    entered_any_zone = True

                # Terminate right after the end of the indicated section (or after the end of all sections if there are several)
                if args.zones and entered_any_zone:
                    max_zone_bound = max(b for a, b in zones)
                    if current_dist > max_zone_bound + 10.0:
                        lap_completed = True
                        lap_valid = True
                        lap_time = current_cur_lap
                        print(f"\n  [ZONA COMPLETATA] Zona completata (distanza: {current_dist:.1f}m > limite: {max_zone_bound + 10.0:.1f}m). Termino il giro anticipatamente!")

                # Log about every 2 seconds (100 steps) — zone indicator (only for the record)
                if step % 100 == 0:
                    zone_tag = "  ZONA TARGET" if cur_zone is not None else ""
                    print(
                        f"    [Passo {step:4d}] Tempo giro corrente: {current_cur_lap:6.2f}s | "
                        f"Ultimo giro: {current_last_lap:6.2f}s | Distanza: {current_dist:7.1f}m"
                        f"{zone_tag} | Fuori pista: {went_off_track}",
                        end='\r'
                    )

                # Conditions to detect the passage over the finish line.
                # TORCS updates the lastLapTime
                if current_last_lap > 0.0 and abs(current_last_lap - prev_last_lap_time) > 0.0001:
                    lap_completed = True
                    if went_off_track:
                        lap_valid = False
                        invalidation_reason = "Giro invalidato (taglio curva o fuori pista)"
                        print(f"\n  TRAGUARDO (A)! {invalidation_reason}")
                    else:
                        lap_valid = True
                        lap_time = current_last_lap
                        print(f"\n  TRAGUARDO (A)! Lap time rilevato: {lap_time:.3f}s")

                # TORCS does not update lastLapTime if the lap is not valid (cut or off-track)
                elif current_cur_lap < 1.5 and prev_cur_lap_time > 5.0:
                    lap_completed = True
                    lap_valid = False
                    invalidation_reason = "Giro invalidato da TORCS (taglio o uscita)"
                    print(f"\n  TRAGUARDO (B)! {invalidation_reason} (CurTime resettato)")

                # Geometric finish-line detection in case the timers are not updated (fail-safe)
                elif current_dist < 50.0 and prev_dist > 500.0:
                    # We have passed the finish line (distance reset)
                    # We wait 10 steps to see if lastLapTime updates before closing
                    # But for safety, if after a while nothing happens, we close as invalid.
                    if step > 500: # Avoid spurious resets at the start
                        lap_completed = True
                        lap_valid = False
                        invalidation_reason = "Fine giro rilevata da posizione (timer TORCS non aggiornato)"
                        print(f"\n  TRAGUARDO (C)! {invalidation_reason}")

                prev_cur_lap_time = current_cur_lap
                prev_dist = current_dist

                # Exit from the lap loop
                if lap_completed or done:
                    if done and not lap_completed:
                        print("\n  [Info] Simulazione terminata esternamente (TORCS chiuso).")
                    break

                # Dynamic frame-rate control (50Hz)
                elapsed = time.perf_counter() - loop_start
                sleep_time = max(0.0, TARGET_DT - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            # End of lap
            print(f"\n  --- Fine Giro (step totali: {step}) ---")

            # Save the data only if the lap is valid
            if lap_completed and lap_valid:
                states_np = np.stack(lap_states)
                actions_np = np.stack(lap_actions)

                dists_np = np.asarray(lap_dists[:len(states_np)], dtype=np.float32)

                # Utility function to save to HDF5
                def _write_h5(path, st, ac, di):
                    with h5py.File(path, 'w') as h5f:
                        h5f.create_dataset('states', data=st, compression="gzip")
                        h5f.create_dataset('actions', data=ac, compression="gzip")
                        h5f.create_dataset('dist_from_start', data=di, compression="gzip")
                        h5f.attrs['lap_time'] = lap_time
                        h5f.attrs['num_steps'] = len(st)
                        h5f.attrs['has_dist_meta'] = True
                        h5f.attrs['timestamp'] = datetime.now().isoformat()

                # If in segment only mode, save only those
                if args.segment_only:
                    segs = [(s, e) for (s, e) in _extract_segments(dists_np, zones, margin_steps=15) if e - s >= 20]
                    saved_names = []

                    # For each segment
                    for (s, e) in segs:
                        lap_counter += 1
                        # Find the zone corresponding to the segment
                        target_dist = dists_np[(s + e) // 2]
                        matched_zone = None
                        for (za, zb) in zones:
                            if za <= target_dist <= zb:
                                matched_zone = (za, zb)
                                break

                        if matched_zone is None:
                            matched_zone = min(zones, key=lambda z: min(abs(z[0] - target_dist), abs(z[1] - target_dist)))

                        za_int, zb_int = int(matched_zone[0]), int(matched_zone[1])
                        filename = f"lap_seg_{za_int}m_{zb_int}m_{lap_counter:03d}.h5"
                        _write_h5(os.path.join(laps_dir, filename),
                                  states_np[s:e], actions_np[s:e], dists_np[s:e])
                        saved_names.append(filename)
                    print(f"  GIRO VALIDO — Salvati {len(segs)} segmenti curva ({', '.join(saved_names)})")
                    log_steps = sum(e - s for s, e in segs)
                else:
                    lap_counter += 1
                    filename = f"lap_{lap_counter:03d}.h5"
                    _write_h5(os.path.join(laps_dir, filename), states_np, actions_np, dists_np)
                    print(f"  GIRO VALIDO — Salvato: {filename}")
                    log_steps = len(lap_states)

                session_saved += 1

                log_entry = (
                    f"[SALVATO] | Tempo giro: {lap_time:.3f}s | "
                    f"Passi salvati: {log_steps} | {datetime.now().isoformat()}"
                )
                print(f"     Tempo giro: {lap_time:.3f}s | Passi salvati: {log_steps}")

            else:
                # Discarded lap
                session_discarded += 1
                if not lap_completed:
                    reason = "Giro non completato (interrotto o timeout)"
                else:
                    reason = invalidation_reason if invalidation_reason else "Tempo non valido"

                log_entry = (
                    f"[SCARTATO] Tentativo #{lap_attempt} | Motivo: {reason} | "
                    f"Passi: {len(lap_states)} | {datetime.now().isoformat()}"
                )
                print(f"  GIRO SCARTATO — {reason}")

            # Write the log to file
            with open(log_path, 'a') as f:
                f.write(log_entry + "\n")

    except KeyboardInterrupt:
        # Manual interruption: do NOT save the current lap
        print(f"\n\n{'=' * 64}")
        print(f"  SESSIONE TERMINATA (Ctrl+C)")
        print(f"     Giri salvati:   {session_saved}")
        print(f"     Giri scartati:  {session_discarded}")
        print(f"     Log sessione:   {log_path}")
        print(f"{'=' * 64}")
        print(f"  Giro corrente scartato (incompleto/interrotto).")

    finally:
        # Write the final summary to the log
        try:
            with open(log_path, 'a') as f:
                f.write(f"\n--- RIEPILOGO SESSIONE ---\n")
                f.write(f"Giri salvati: {session_saved}\n")
                f.write(f"Giri scartati: {session_discarded}\n")
                f.write(f"Fine sessione: {datetime.now().isoformat()}\n")
        except Exception:
            pass

        env.end()
        pygame.quit()


if __name__ == "__main__":
    main()
