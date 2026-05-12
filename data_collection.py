"""
Data Collection — Giro Secco TORCS (Human-in-the-Loop)

Registra singoli giri con partenza da fermo usando un controller PS5 DualSense.
Ogni giro viene validato (nessuna uscita di pista + lap time registrato).
Solo i giri validi vengono salvati in file HDF5 separati.

Loop infinito: registra → valida → salva (se valido) → riavvia → ripeti.
Interrompere con Ctrl+C. Il giro corrente incompleto NON viene salvato.

Formato output (se il giro è valido e completato):
    lap_001.h5, lap_002.h5, ...              (un file per giro valido)
    lap_curve_001_0001.h5, ...               (snippet separati per ogni curva affrontata nel giro)
    session_logs/session_YYYYMMDD.log        (log di sessione testuale)
"""

import os
import sys
import time
import argparse
import numpy as np
import h5py
import pygame
from datetime import datetime
from typing import Optional
from collections import deque

# Aggiungo gym_torcs al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'gym_torcs')))

try:
    from gym_torcs import TorcsEnv
except ImportError as e:
    print(f"ERRORE FATALE: Impossibile importare gym_torcs o una sua dipendenza.")
    print(f"Dettagli errore: {e}")
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────
#  Controller PS5 DualSense
# ──────────────────────────────────────────────────────────────────────

class DualSenseController:
    """Gestisce il polling del controller PlayStation 5 tramite Pygame.

    Mappatura:
        Left Stick X → Sterzo continuo (con deadzone configurabile)
        R2 (asse 5)  → Acceleratore [0, 1]
        L2 (asse 2)  → Freno [0, 1]
        Quadrato      → Upshift
        X (Cross)     → Downshift
    """

    # Axis mapping (DualSense su Linux/Pygame)
    AXIS_STEER = 0
    AXIS_L2 = 2       # Brake
    AXIS_R2 = 5       # Accel

    # Button mapping
    BTN_CROSS = 0      # Downshift
    BTN_SQUARE = 3     # Upshift

    DEBOUNCE_MS = 200  # Millisecondi di debounce per i pulsanti del cambio

    def __init__(self, steering_deadzone: float = 0.05):
        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            raise RuntimeError("Nessun controller rilevato. Collega un DualSense e riprova.")

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        print(f"  Controller inizializzato: {self.joystick.get_name()}")

        self.steering_deadzone = steering_deadzone
        self.gear = 1  # Partenza in prima marcia

        # Warm-up flags per i grilletti (previene valori spuri pre-primo press)
        self._r2_initialized = False
        self._l2_initialized = False

        # Timestamp dell'ultimo cambio marcia (debounce)
        self._last_shift_time = 0

    def get_action(self) -> np.ndarray:
        """Legge controller e ritorna [steering, accel, brake, gear] come float32."""
        # Svuota la coda eventi di Pygame per evitare che si saturi (causa input lag)
        pygame.event.clear()

        # ── Sterzo con deadzone ──
        raw_steer = -self.joystick.get_axis(self.AXIS_STEER)
        if abs(raw_steer) < self.steering_deadzone:
            steering = 0.0
        else:
            # Riscala il range post-deadzone su [-1, 1]
            sign = 1.0 if raw_steer > 0 else -1.0
            steering = sign * (abs(raw_steer) - self.steering_deadzone) / (1.0 - self.steering_deadzone)

        # ── Acceleratore (R2) con protezione warm-up ──
        raw_r2 = self.joystick.get_axis(self.AXIS_R2)
        if not self._r2_initialized:
            if abs(raw_r2) > 0.1:
                self._r2_initialized = True
            accel = 0.0
        else:
            accel = max(0.0, (raw_r2 + 1.0) / 2.0)
            if accel < 0.05:
                accel = 0.0

        # ── Freno (L2) con protezione warm-up ──
        raw_l2 = self.joystick.get_axis(self.AXIS_L2)
        if not self._l2_initialized:
            if abs(raw_l2) > 0.1:
                self._l2_initialized = True
            brake = 0.0
        else:
            brake = max(0.0, (raw_l2 + 1.0) / 2.0)
            if brake < 0.05:
                brake = 0.0

        # ── Cambio marcia con debounce temporale ──
        now = pygame.time.get_ticks()
        if now - self._last_shift_time > self.DEBOUNCE_MS:
            if self.joystick.get_button(self.BTN_SQUARE):
                if self.gear < 6:
                    self.gear += 1
                    print(f"  [Gear] ⬆ Marcia {self.gear}")
                self._last_shift_time = now
            elif self.joystick.get_button(self.BTN_CROSS):
                if self.gear > 1:  # Min gear 1 (niente retromarcia nella raccolta dati)
                    self.gear -= 1
                    print(f"  [Gear] ⬇ Marcia {self.gear}")
                self._last_shift_time = now

        return np.array([steering, accel, brake, float(self.gear)], dtype=np.float32)


class KeyboardController:
    """Gestisce la guida di TORCS tramite la tastiera (WASD + Frecce).
    
    Richiede una piccola finestra Pygame aperta e focalizzata per registrare i tasti.
    """
    DEBOUNCE_MS = 250

    def __init__(self):
        pygame.init()
        # Finestra minimale per catturare gli input di Pygame
        self.screen = pygame.display.set_mode((100, 100))
        pygame.display.set_caption("Input Focus")
        
        self.gear = 1
        self.steer_val = 0.0
        self._last_shift_time = 0
        print("  [Keyboard] Inizializzato. MANTIENI IL FOCUS sulla finestra nera 'Input Focus' per guidare!")

    def get_action(self) -> np.ndarray:
        # Processa gli eventi di Pygame per mantenere la finestra attiva e reattiva
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)

        keys = pygame.key.get_pressed()

        # 1. Sterzo graduale (Smooth interpolation) per una guida fluida
        steer_target = 0.0
        if keys[pygame.K_a]:
            steer_target = 1.0  # +1.0 in TORCS gira a sinistra (Sinistra)
        elif keys[pygame.K_d]:
            steer_target = -1.0  # -1.0 in TORCS gira a destra (Destra)

        # Muoviti verso il target
        if self.steer_val < steer_target:
            self.steer_val = min(steer_target, self.steer_val + 0.08)
        elif self.steer_val > steer_target:
            self.steer_val = max(steer_target, self.steer_val - 0.08)

        # 2. Acceleratore e Freno digitali reattivi
        accel = 1.0 if keys[pygame.K_w] else 0.0
        brake = 1.0 if keys[pygame.K_s] else 0.0

        # Priorità al freno in caso di pressione simultanea
        if brake > 0.1:
            accel = 0.0

        # 3. Cambio marcia con protezione debounce
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

        # Aggiorna lo schermo per evitare che il sistema operativo veda la finestra come bloccata/congelata
        self.screen.fill((30, 30, 40))  # Grigio scuro minimale
        pygame.display.flip()

        return np.array([self.steer_val, accel, brake, float(self.gear)], dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────
#  Utility: Flattening sicuro dello stato
# ──────────────────────────────────────────────────────────────────────

def flatten_state(state_dict: dict) -> np.ndarray:
    """Appiattisce il dizionario di osservazione TORCS in un vettore 1D (29D).

    Ordine: [angle(1), track(19), trackPos(1), speedX(1), speedY(1), speedZ(1),
             wheelSpinVel(4)/100, rpm(1)/10000]

    Usa .get() con default per evitare crash su chiavi mancanti.
    """
    def _scalar(key: str, default: float = 0.0) -> float:
        val = state_dict.get(key, default)
        if val is None:
            return default
        if isinstance(val, np.ndarray):
            return float(val.flat[0])
        return float(val)

    def _array(key: str, size: int) -> np.ndarray:
        val = state_dict.get(key, None)
        if val is None:
            return np.zeros(size, dtype=np.float32)
        arr = np.array(val, dtype=np.float32).flatten()
        if arr.shape[0] != size:
            padded = np.zeros(size, dtype=np.float32)
            padded[:min(size, arr.shape[0])] = arr[:min(size, arr.shape[0])]
            return padded
        return arr

    try:
        state_vec = np.concatenate([
            np.array([_scalar('angle')]),
            _array('track', 19),
            np.array([_scalar('trackPos')]),
            np.array([_scalar('speedX')]),
            np.array([_scalar('speedY')]),
            np.array([_scalar('speedZ')]),
            _array('wheelSpinVel', 4) / 100.0,
            np.array([_scalar('rpm') / 10000.0]),
        ])
        return state_vec.astype(np.float32)
    except Exception as e:
        print(f"  [WARN] Errore in flatten_state: {e}. Ritorno vettore zero (29D).")
        return np.zeros(29, dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────
#  Funzione helper: estrai trackPos come scalare
# ──────────────────────────────────────────────────────────────────────

def _get_track_pos(obs: dict) -> float:
    """Estrae trackPos come float scalare dall'osservazione."""
    tp = obs.get('trackPos', 0.0)
    if isinstance(tp, np.ndarray):
        return float(tp.flat[0])
    return float(tp)


def _get_last_lap_time(obs: dict) -> float:
    """Estrae lastLapTime come float scalare dall'osservazione."""
    llt = obs.get('lastLapTime', 0.0)
    if isinstance(llt, np.ndarray):
        return float(llt.flat[0])
    return float(llt)


def is_on_curve(obs: dict) -> bool:
    """Rileva se l'auto si trova in curva basandosi sulla geometria del tracciato.

    Analizza i sensori 'track' (19 range finders) e l'angolo 'angle' per determinare
    la curvatura della strada, indipendentemente dall'azione di sterzo del pilota.
    """
    track = obs.get('track', None)
    if track is None:
        return False

    track = np.array(track, dtype=np.float32).flatten()
    if track.shape[0] < 19:
        return False

    # 1. Distanza massima e indice del sensore più lungo
    max_idx = np.argmax(track)

    # 2. Distanza frontale (sensore 9, a 0 gradi rispetto all'auto)
    front_dist = track[9]

    # 3. Analisi di asimmetria dei sensori laterali
    left_sum = np.sum(track[0:9])
    right_sum = np.sum(track[10:19])
    asymmetry = abs(left_sum - right_sum) / (min(left_sum, right_sum) + 1e-5)

    # 4. Condizione di curva geometrica
    is_curved = False

    # Se la visuale davanti è chiusa, verifichiamo deviazione del percorso o asimmetria
    if front_dist < 130.0:
        if abs(max_idx - 9) >= 2 or asymmetry > 0.15:
            is_curved = True

    # Gestione curve cieche / tornanti molto stretti
    if front_dist < 60.0:
        is_curved = True

    return is_curved



def apply_tcs(action: np.ndarray, obs: dict, slip_threshold: float = 5.0) -> np.ndarray:
    """Traction Control System — riduce l'acceleratore in caso di slittamento.

    Confronta la velocità angolare delle ruote posteriori vs anteriori.
    Se la differenza supera la soglia, scala l'accel proporzionalmente.

    Args:
        action: [steering, accel, brake, gear]
        obs: dizionario di osservazione TORCS (contiene wheelSpinVel)
        slip_threshold: differenza di spin oltre cui il TCS interviene

    Returns:
        action modificata con accel ridotta se necessario
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
        # Riduzione progressiva: più slip → più taglio
        # Da 1.0 (nessun taglio) a 0.2 (taglio massimo 80%)
        reduction = max(0.2, 1.0 - (slip - slip_threshold) / 30.0)
        action = action.copy()
        action[1] *= reduction  # Scala l'acceleratore

    return action


# ──────────────────────────────────────────────────────────────────────
#  Main Loop di Data Collection
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Data Collection TORCS — Giro Secco con controller PS5"
    )
    parser.add_argument(
        "--output_dir", type=str, default="train_set",
        help="Directory di output per i file HDF5 e il log (default: directory corrente)"
    )
    parser.add_argument(
        "--mode", type=str, choices=["ideal", "diverse"], default="ideal",
        help="Modalità di raccolta: 'ideal' (traiettoria ideale, salva giro + curve) o 'diverse' (traiettorie diverse in curva, salva solo curve)"
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
    args = parser.parse_args()

    # ── Sanitizza sys.argv per evitare conflitti con getopt di snakeoil3 ──
    # snakeoil3_gym.Client.__init__ chiama parse_the_command_line() che usa
    # getopt su sys.argv e non conosce --output_dir / --steering_deadzone.
    sys.argv = [sys.argv[0]]

    # ── Cartella dati giri e curve ──
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    laps_dir = os.path.join(output_dir, "laps")
    os.makedirs(laps_dir, exist_ok=True)
    curves_dir = os.path.join(laps_dir, "curves")
    os.makedirs(curves_dir, exist_ok=True)

    # ── Session log ──
    log_dir = os.path.join(output_dir, "session_logs")
    os.makedirs(log_dir, exist_ok=True)
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"session_{session_id}.log")

    # ── Controller ──
    if args.device == "keyboard":
        controller = KeyboardController()
    else:
        try:
            controller = DualSenseController(steering_deadzone=args.steering_deadzone)
        except RuntimeError as e:
            print(f"  ❌ Errore controller: {e}")
            print("  👉 Vuoi usare la tastiera? Avvia con: python data_collection.py --device keyboard")
            sys.exit(1)

    # ── Conta i giri già esistenti nella directory per numerazione continua ──
    existing_laps = sorted([
        f for f in os.listdir(laps_dir)
        if f.startswith("lap_") and f.endswith(".h5")
    ])
    lap_counter = len(existing_laps)

    # ── Statistiche di sessione ──
    session_saved = 0
    session_discarded = 0

    print()
    print("=" * 64)
    print("   🏎️  DATA COLLECTION — Giro Secco TORCS")
    print("   Premi Ctrl+C nel terminale per terminare la sessione")
    print("=" * 64)

    # ── Inizializza l'ambiente ──
    env = TorcsEnv(vision=False, throttle=True, gear_change=True, early_termination=False)

    TARGET_DT = 1.0 / 50.0  # 50 Hz target
    lap_attempt = 0

    try:
        while True:
            lap_attempt += 1

            # ── Reset ambiente ──
            # Relaunch periodico per evitare memory leak, e al primo giro
            need_relaunch = (lap_attempt == 1) or (lap_attempt % args.relaunch_every == 0)
            if lap_attempt == 1:
                ob = env.reset(relaunch=True)
            else:
                ob = env.reset(relaunch=need_relaunch)

            state_vec = flatten_state(ob)

            # ── Buffer in RAM per questo giro ──
            lap_states: list = []
            lap_actions: list = []

            # ── Buffer per estrazione automatica curve ──
            curve_snippets = []
            is_in_curve = False
            pre_curve_buffer_states = deque(maxlen=100) # 2 sec a 50Hz
            pre_curve_buffer_actions = deque(maxlen=100)
            current_curve_states = []
            current_curve_actions = []
            curve_cooldown = 0
            POST_CURVE_FRAMES = 250 # 5 sec a 50Hz

            # ── Stato di validità del giro ──
            lap_valid = True
            invalidation_step: Optional[int] = None
            invalidation_reason = ""
            lap_completed = False
            lap_time = 0.0

            # ── Snapshot iniziale di lastLapTime per rilevare la transizione ──
            prev_last_lap_time = _get_last_lap_time(ob)

            # ── Reset marcia ──
            controller.gear = 1

            print(f"\n{'─' * 64}")
            print(f"  🏁 TENTATIVO GIRO #{lap_attempt}  (giri salvati finora: {lap_counter})")
            print(f"  Status: [VALIDO]")
            print(f"{'─' * 64}")

            step = 0

            while True:
                loop_start = time.perf_counter()
                step += 1

                # ── Poll controller ──
                action = controller.get_action()

                # ── Traction Control System ──
                if args.tcs:
                    action = apply_tcs(action, ob, slip_threshold=args.tcs_slip)

                # ── Step simulazione ──
                ob_next, reward, done, info = env.step(action)
                next_state_vec = flatten_state(ob_next)

                # ── Accumula in RAM ──
                lap_states.append(state_vec.copy())
                lap_actions.append(action.copy())

                # ── Estrazione automatica curve (Geometrica) ──
                curve_detected = is_on_curve(ob_next)
                if curve_detected:
                    if not is_in_curve:
                        is_in_curve = True
                        print(f"  [Curve] ↪️ Inizio curva rilevato geometricamente (distanza frontale: {ob_next.get('track', [0]*10)[9]:.1f}m)")
                        current_curve_states.extend(list(pre_curve_buffer_states))
                        current_curve_actions.extend(list(pre_curve_buffer_actions))
                    current_curve_states.append(state_vec.copy())
                    current_curve_actions.append(action.copy())
                    curve_cooldown = POST_CURVE_FRAMES
                elif is_in_curve:
                    current_curve_states.append(state_vec.copy())
                    current_curve_actions.append(action.copy())
                    curve_cooldown -= 1
                    if curve_cooldown <= 0:
                        is_in_curve = False
                        print(f"  [Curve] ↩️ Fine curva (cooldown scaduto).")
                        if lap_valid:
                            curve_snippets.append((
                                np.stack(current_curve_states),
                                np.stack(current_curve_actions)
                            ))
                        current_curve_states = []
                        current_curve_actions = []
                else:
                    pre_curve_buffer_states.append(state_vec.copy())
                    pre_curve_buffer_actions.append(action.copy())

                state_vec = next_state_vec
                ob = ob_next

                # ── Validazione: uscita di pista ──
                track_pos = _get_track_pos(ob_next)
                if lap_valid and abs(track_pos) > 1.0:
                    lap_valid = False
                    invalidation_step = step
                    invalidation_reason = (
                        f"Uscita di pista (trackPos = {track_pos:.3f})"
                    )
                    # Stampa SOLO al cambio di stato (una volta sola)
                    print(f"\n  ⚠️  GIRO INVALIDATO allo step {step}: {invalidation_reason}")
                    print(f"  Status: [INVALIDO] — i dati NON saranno salvati\n")

                # ── Rilevamento completamento giro ──
                current_last_lap = _get_last_lap_time(ob_next)
                if current_last_lap > 0.0 and abs(current_last_lap - prev_last_lap_time) > 0.01:
                    lap_completed = True
                    lap_time = current_last_lap

                # ── Uscita dal loop del giro ──
                if lap_completed or done or not lap_valid:
                    break

                # ── Frame rate control dinamico (50Hz) ──
                elapsed = time.perf_counter() - loop_start
                sleep_time = max(0.0, TARGET_DT - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            # ────────────────────────────────────────
            #  Fine giro: valutazione e salvataggio
            # ────────────────────────────────────────
            print(f"\n  --- Fine Giro (step totali: {step}) ---")

            if lap_completed and lap_valid:
                # ── Salvataggio HDF5 (batch unico) ──
                states_np = np.stack(lap_states)
                actions_np = np.stack(lap_actions)

                # Se siamo in modalità 'ideal', salviamo il giro completo in laps_dir
                if args.mode == "ideal":
                    lap_counter += 1
                    filename = f"lap_{lap_counter:03d}.h5"
                    filepath = os.path.join(laps_dir, filename)

                    with h5py.File(filepath, 'w') as h5f:
                        h5f.create_dataset('states', data=states_np, compression="gzip")
                        h5f.create_dataset('actions', data=actions_np, compression="gzip")
                        h5f.attrs['lap_time'] = lap_time
                        h5f.attrs['num_steps'] = len(lap_states)
                        h5f.attrs['timestamp'] = datetime.now().isoformat()
                    print(f"  ✅ GIRO VALIDO (Ideale) — Salvato giro completo: {filename}")
                else:
                    print(f"  ✅ GIRO VALIDO (Diverso) — Giro completato correttamente. Salvataggio solo dei dati curva...")

                # ── Salvataggio Snippet Curve ──
                if is_in_curve:
                    curve_snippets.append((
                        np.stack(current_curve_states),
                        np.stack(current_curve_actions)
                    ))

                saved_curves_count = 0
                for c_states, c_actions in curve_snippets:
                    existing_curves = [
                        f for f in os.listdir(curves_dir)
                        if f.startswith("lap_curve_") and f.endswith(".h5")
                    ]
                    c_idx = len(existing_curves) + 1

                    if args.mode == "ideal":
                        c_filename = f"lap_curve_{lap_counter:03d}_{c_idx:04d}.h5"
                    else:
                        c_filename = f"lap_curve_diverse_{lap_attempt:03d}_{c_idx:04d}.h5"

                    c_filepath = os.path.join(curves_dir, c_filename)
                    with h5py.File(c_filepath, 'w') as h5f:
                        h5f.create_dataset('states', data=c_states, compression="gzip")
                        h5f.create_dataset('actions', data=c_actions, compression="gzip")
                        h5f.attrs['num_steps'] = len(c_states)
                        h5f.attrs['timestamp'] = datetime.now().isoformat()
                    saved_curves_count += 1

                session_saved += 1

                log_entry = (
                    f"[{'SAVED_IDEAL' if args.mode == 'ideal' else 'SAVED_DIVERSE'}] | Lap Time: {lap_time:.3f}s | "
                    f"Steps: {len(lap_states)} | Curve: {saved_curves_count} | {datetime.now().isoformat()}"
                )
                print(f"     Lap Time: {lap_time:.3f}s | Steps: {len(lap_states)} | Curve salvate in 'curves/': {saved_curves_count}")

            else:
                # ── Giro scartato ──
                session_discarded += 1
                if not lap_completed:
                    reason = "Giro non completato (nessun lap time registrato)"
                else:
                    reason = f"Uscita di pista allo step {invalidation_step}"

                log_entry = (
                    f"[DISCARDED] Tentativo #{lap_attempt} | Motivo: {reason} | "
                    f"Steps: {len(lap_states)} | {datetime.now().isoformat()}"
                )
                print(f"  ❌ GIRO SCARTATO — {reason}")

            # ── Scrivi log su file ──
            with open(log_path, 'a') as f:
                f.write(log_entry + "\n")

    except KeyboardInterrupt:
        # ── Interruzione manuale: NON salvare il giro corrente ──
        print(f"\n\n{'=' * 64}")
        print(f"  🛑 SESSIONE TERMINATA (Ctrl+C)")
        print(f"     Giri salvati:   {session_saved}")
        print(f"     Giri scartati:  {session_discarded}")
        print(f"     Log sessione:   {log_path}")
        print(f"{'=' * 64}")
        print(f"  ⚠️  Giro corrente scartato (incompleto/interrotto).")

    finally:
        # ── Scrivi riepilogo finale nel log ──
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
