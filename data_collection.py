"""
Data Collection, script di raccolta dati umani sul tracciato. Consente l'utilizzo di un controller o della tastiera. 

Registra i singolo giri completati senza uscire di pista, i giri delle zone problematiche (curve)
e anche giri con raccolta dati esclusivi di alcuni segmenti del tracciato. 

I giri vengono salvati automaticamente in file HDF5 separati.

Formato output:
    lap_001.h5, lap_002.h5, ...              (un file per giro valido completo)
    lap_seg_001.h5, ...                  (segmenti mirati, solo con --segment_only)
    lap_seg_550m_900m.h5, ...             (segmenti specifici per raccolta dati mirata)
    session_logs/giri/session_YYYYMMDD.log   (log di sessione testuale)

Ogni HDF5 salva stato 29D, azione [steer, accel, brake, gear] e dist_from_start come metadato.
dist_from_start serve per segmentazione/analisi, non entra nella rete, questo impedisce che il modello impari a correlare la posizione con l'azione.
garantendo che impara a guidare in base ai sensori e non alla posizione sul tracciato.
"""

import os
import sys
import time
import argparse
import numpy as np
import h5py
import pygame   #Libreria per l'interazione con il controller
from datetime import datetime

# Forza la visualizzazione della GUI di TORCS per la data collection
os.environ['SHOW_GUI'] = '1'

# Aggiunge il wrapper TORCS locale al path di import.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'gym_torcs')))

try:
    from gym_torcs import TorcsEnv
except ImportError as e:
    print(f"ERRORE FATALE: Impossibile importare gym_torcs o una sua dipendenza.")
    print(f"Dettagli errore: {e}")
    sys.exit(1)


# Classe per la gestione del controller PS5 DualSense
class DualSenseController:
    """Gestisce il polling del controller PlayStation 5 tramite Pygame.

    Mappatura:
        Left Stick X   → Sterzo continuo (con deadzone configurabile)
        R2 (asse 5)    → Acceleratore [0, 1]
        L2 (asse 2)    → Freno [0, 1]
        Quadrato       → Upshift
        X              → Downshift
    """

    # Mappatura degli assi
    AXIS_STEER = 0
    AXIS_L2 = 2       # Freno
    AXIS_R2 = 5       # Acceleratore

    # Mappatura dei pulsanti
    BTN_CROSS = 0      # Downshift
    BTN_SQUARE = 3     # Upshift

    # Debounce per i pulsanti del cambio
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
        self.gear = 1  # Partenza in prima marcia

        # Warm-up flags per i grilletti (previene valori spuri pre-primo press)
        self._r2_initialized = False
        self._l2_initialized = False

        # Timestamp dell'ultimo cambio marcia (debounce)
        self._last_shift_time = 0

    # Funzione che legge il controller e restituisce l'azione [sterzo, acceleratore, freno, marcia]
    def get_action(self) -> np.ndarray:
    
        # Svuota la coda eventi di Pygame per evitare che si saturi (causa input lag)
        pygame.event.clear()

        # Gestione sterzo con deadzone 
        raw_steer = -self.joystick.get_axis(self.AXIS_STEER)
        if abs(raw_steer) < self.steering_deadzone:
            steering = 0.0
        else:
            # Riscala il range post-deadzone su [-1, 1]
            sign = 1.0 if raw_steer > 0 else -1.0
            steering = sign * (abs(raw_steer) - self.steering_deadzone) / (1.0 - self.steering_deadzone)

        # Gestione acceleratore (R2) con protezione warm-up 
        raw_r2 = self.joystick.get_axis(self.AXIS_R2)
        if not self._r2_initialized:
            if abs(raw_r2) > 0.1:
                self._r2_initialized = True
            accel = 0.0
        else:
            accel = max(0.0, (raw_r2 + 1.0) / 2.0)
            if accel < 0.05:
                accel = 0.0

        # Gestione freno (L2) con protezione warm-up 
        raw_l2 = self.joystick.get_axis(self.AXIS_L2)
        if not self._l2_initialized:
            if abs(raw_l2) > 0.1:
                self._l2_initialized = True
            brake = 0.0
        else:
            brake = max(0.0, (raw_l2 + 1.0) / 2.0)
            if brake < 0.05:
                brake = 0.0

        # Cambio marcia con debounce temporale
        now = pygame.time.get_ticks()
        if now - self._last_shift_time > self.DEBOUNCE_MS:
            if self.joystick.get_button(self.BTN_SQUARE):
                if self.gear < 6:
                    self.gear += 1
                    print(f"  [Gear] ⬆ Marcia {self.gear}")
                self._last_shift_time = now
            elif self.joystick.get_button(self.BTN_CROSS):
                if self.gear > 1:  # Gear minimo 1: la raccolta dati non usa la retromarcia
                    self.gear -= 1
                    print(f"  [Gear] ⬇ Marcia {self.gear}")
                self._last_shift_time = now

        return np.array([steering, accel, brake, float(self.gear)], dtype=np.float32)

    def rumble(self, intensity: float = 0.3, duration_ms: int = 180):
        """Attiva un breve feedback aptico, se il controller lo supporta."""
        try:
            self.joystick.rumble(0.0, float(min(0.5, intensity)), int(duration_ms))
        except Exception:
            pass  # Se il rumble non è supportato, non fa nulla


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

    def rumble(self, intensity: float = 0.3, duration_ms: int = 180):
        """No-op: la tastiera non ha feedback aptico."""
        pass

    def get_action(self) -> np.ndarray:
        # Processa gli eventi di Pygame per mantenere la finestra attiva e reattiva
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)

        keys = pygame.key.get_pressed()

        # Sterzo graduale (Smooth interpolation) per una guida fluida
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

        # Acceleratore e Freno digitali reattivi
        accel = 1.0 if keys[pygame.K_w] else 0.0
        brake = 1.0 if keys[pygame.K_s] else 0.0

        # Priorità al freno in caso di pressione simultanea
        if brake > 0.1:
            accel = 0.0

        # Cambio marcia con protezione debounce
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


# Le zone problematiche del tracciato sono le curve, abbiamo estratto questi valori dai file di torcs
# in particoalre da corkscrew.xml, abbiamo poi allargato leggermente le zone per catturare anche le staccate e le uscite di curva
PROBLEM_ZONES = [
    (340, 530), (670, 810), (940, 1070), (1420, 1590), (1870, 1980),
    (2380, 2530), (2570, 2780), (2890, 3020), (3190, 3300),
]

# Converte le zone scritte dall'utente (se specificate durante l'avvio della raccolta dati tramite --zones) in una lista di tuple (start,end)
# Altrinenti utilizza quelle predefinite PROBLEM_ZONES
def _parse_zones(spec):
    """Converte 'a:b,c:d' in [(a,b),(c,d)]. None/'' → PROBLEM_ZONES di default."""
    if not spec:
        return list(PROBLEM_ZONES)
    out = []
    for part in spec.split(','):
        a, b = part.split(':')
        out.append((float(a), float(b)))
    return out

# Verifica se la distanza rientra in una zona problematica e restituisce l'indice della zona
def _zone_index(dist, zones):
    """Indice della zona che contiene 'dist', altrimenti None."""
    for zi, (a, b) in enumerate(zones):
        if a <= dist <= b:
            return zi
    return None

# Estrae i run contigui di step in zona, con margine di approccio
def _extract_segments(dists, zones, margin_steps=15):
    """Run contigui di step in zona, con margine di approccio. Ritorna [(start,end), ...] (end escluso)."""
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


# Flattening dello stato, rende il dizionario di osservazione TORCS in un vettore 1D con 29 elementi.
# Le variabili sono normalizzate e scalate per essere compatibili con la rete neurale.
def flatten_state(state_dict: dict) -> np.ndarray:
    """Appiattisce il dizionario di osservazione TORCS in un vettore 1D (29D).

    Ordine: [angle(1), track(19), trackPos(1), speedX(1), speedY(1), speedZ(1),
             wheelSpinVel(4)/100, rpm(1)/10000]

    NOTA: distFromStart non fa parte dello stato 29D. Viene salvata solo come
    metadato per analisi/segmentazione, perché la policy deve guidare dai sensori.

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
        print(f"  Errore in flatten_state: {e}. Ritorno vettore zero (29D).")
        return np.zeros(29, dtype=np.float32)


# Estrae distFromStart come float scalare dall'osservazione.
def _get_dist_from_start(obs: dict) -> float:
    """Estrae distFromStart come float scalare dall'osservazione."""
    dfs = obs.get('distFromStart', 0.0)
    if isinstance(dfs, np.ndarray):
        return float(dfs.flat[0])
    return float(dfs)

# Estrae curLapTime come float scalare dall'osservazione.
def _get_cur_lap_time(obs: dict) -> float:
    """Estrae curLapTime come float scalare dall'osservazione."""
    clt = obs.get('curLapTime', 0.0)
    if isinstance(clt, np.ndarray):
        return float(clt.flat[0])
    return float(clt)

# Estrae lastLapTime come float scalare dall'osservazione.
def _get_last_lap_time(obs: dict) -> float:
    """Estrae lastLapTime come float scalare dall'osservazione."""
    llt = obs.get('lastLapTime', 0.0)
    if isinstance(llt, np.ndarray):
        return float(llt.flat[0])
    return float(llt)



# Funzione che applica il Traction Control System (TCS), ci aiuta a fare giri migliori in fase di raccolta dati
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


# Main loop di data collection
def main():
    # Parsing dei vari argomenti accettati da data_collection.py
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

    # Cartella dati giri e curve
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Zone curva target (raccolta mirata)
    zones = _parse_zones(args.zones)
    print(f"\n  Zone curva target ({len(zones)}): " + ", ".join(f"{int(a)}-{int(b)}m" for a, b in zones))
    if args.segment_only:
        print(f"  Modalità SEGMENT_ONLY: salvo solo i segmenti dentro le zone (guidi giri interi).")
    print(f"  Vibrazione gentile del controller all'ingresso di ogni zona.\n")
    laps_dir = os.path.join(output_dir, "laps")
    os.makedirs(laps_dir, exist_ok=True)

    # Session log
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

    # Conta i giri già esistenti nella directory per numerazione continua
    existing_laps = sorted([
        f for f in os.listdir(laps_dir)
        if f.startswith("lap_") and f.endswith(".h5")
    ])
    lap_counter = len(existing_laps)

    # Statistiche di sessione
    session_saved = 0
    session_discarded = 0

    print()
    print("=" * 64)
    print("   DATA COLLECTION — Giro Secco TORCS")
    print("   Premi Ctrl+C nel terminale per terminare la sessione")
    print("=" * 64)

    # Inizializza l'ambiente
    env = TorcsEnv(early_termination=False)

    TARGET_DT = 1.0 / 50.0  # 50 Hz target
    lap_attempt = 0
    force_relaunch = False

    try:
        while True:
            lap_attempt += 1

            # Reset ambiente
            # Relaunch periodico, al primo giro, o se richiesto (es. fuori pista)
            need_relaunch = (lap_attempt == 1) or (lap_attempt % args.relaunch_every == 0) or force_relaunch
            if lap_attempt == 1:
                ob = env.reset(relaunch=True)
            else:
                ob = env.reset(relaunch=need_relaunch)
            
            force_relaunch = False  # Reset flag dopo l'uso

            state_vec = flatten_state(ob)

            # Buffer in RAM per questo giro
            lap_states: list = []
            lap_actions: list = []
            lap_dists: list = []  # distFromStart per step (METADATO: NON entra negli stati 29D)
            active_zone_idx = None  # indice zona corrente (per il rumble all'ingresso)
            entered_any_zone = False  # Traccia se siamo entrati in almeno una zona target


            # Stato di validità del giro
            lap_valid = True
            invalidation_reason = ""
            lap_completed = False
            lap_time = 0.0
            went_off_track = False

            # Snapshot iniziale di timing e posizione per rilevare la transizione
            prev_last_lap_time = _get_last_lap_time(ob)
            prev_cur_lap_time = _get_cur_lap_time(ob)
            prev_dist = _get_dist_from_start(ob)

            # Reset marcia
            controller.gear = 1

            print(f"\n{'─' * 64}")
            print(f"  TENTATIVO GIRO #{lap_attempt}  (giri salvati finora: {lap_counter})")
            print(f"  Status: [VALIDO]")
            print(f"{'─' * 64}")

            step = 0

            while True:
                loop_start = time.perf_counter()
                step += 1

                # Polling del controller
                action = controller.get_action()

                # Applica il tcs
                if args.tcs:
                    action = apply_tcs(action, ob, slip_threshold=args.tcs_slip)

                # Passo di simulazione
                ob_next, reward, done, info = env.step(action)
                next_state_vec = flatten_state(ob_next)

                # Accumula gli stati, azioni e distanze in RAM
                lap_states.append(state_vec.copy())
                lap_actions.append(action.copy())
                lap_dists.append(_get_dist_from_start(ob))


                state_vec = next_state_vec
                ob = ob_next

                # Controllo fuori pista
                current_track_pos = ob_next.get('trackPos', 0.0)
                if isinstance(current_track_pos, np.ndarray):
                    current_track_pos = current_track_pos.flat[0]
                
                # Limite per permettere una guida più aggressiva sui cordoli.
                if abs(current_track_pos) > 1.25:
                    print(f"\n  [OFF-TRACK] trackPos: {current_track_pos:.2f} - Riavvio immediato simulazione.")
                    went_off_track = True
                    lap_completed = True
                    lap_valid = False
                    invalidation_reason = f"Fuori pista (trackPos: {current_track_pos:.2f})"
                    force_relaunch = True
                    break

                # Rilevamento completamento giro
                current_last_lap = _get_last_lap_time(ob_next)
                current_cur_lap = _get_cur_lap_time(ob_next)
                current_dist = _get_dist_from_start(ob_next)

                # Raccolta mirata nelle zone
                cur_zone = _zone_index(current_dist, zones)
                if cur_zone is not None and cur_zone != active_zone_idx:
                    controller.rumble(intensity=0.3, duration_ms=180) # Vibrazione all'ingresso di ogni zona target.
                active_zone_idx = cur_zone

                if cur_zone is not None:
                    entered_any_zone = True

                # Termina subito dopo la fine della sezione indicata (o dopo la fine di tutte le sezioni se ce ne sono di più)
                if args.zones and entered_any_zone:
                    max_zone_bound = max(b for a, b in zones)
                    if current_dist > max_zone_bound + 10.0:
                        lap_completed = True
                        lap_valid = True
                        lap_time = current_cur_lap
                        print(f"\n  [ZONA COMPLETATA] Zona completata (distanza: {current_dist:.1f}m > limite: {max_zone_bound + 10.0:.1f}m). Termino il giro anticipatamente!")

                # Log ogni 2 secondi circa (100 step) — indicatore zona (solo per il record)
                if step % 100 == 0:
                    zone_tag = "  ZONA TARGET" if cur_zone is not None else ""
                    print(
                        f"    [Passo {step:4d}] Tempo giro corrente: {current_cur_lap:6.2f}s | "
                        f"Ultimo giro: {current_last_lap:6.2f}s | Distanza: {current_dist:7.1f}m"
                        f"{zone_tag} | Fuori pista: {went_off_track}",
                        end='\r'
                    )

                # Condizioni per rilevare il passaggio dal traguardo.
                # TORCS aggiorna il lastLapTime 
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

                # Torcs non aggiorna lastLapTime se il giro non è valido (taglio o uscita)
                elif current_cur_lap < 1.5 and prev_cur_lap_time > 5.0:
                    lap_completed = True
                    lap_valid = False
                    invalidation_reason = "Giro invalidato da TORCS (taglio o uscita)"
                    print(f"\n  TRAGUARDO (B)! {invalidation_reason} (CurTime resettato)")

                # Rilevamento geometrico del traguardo in caso di mancato aggiornamento dei timer (fail-safe)
                elif current_dist < 50.0 and prev_dist > 500.0:
                    # Abbiamo passato il traguardo (distanza resettata)
                    # Aspettiamo 10 step per vedere se lastLapTime si aggiorna prima di chiudere
                    # Ma per sicurezza, se dopo un po' non succede nulla, chiudiamo come invalido.
                    if step > 500: # Evita reset spuri alla partenza
                        lap_completed = True
                        lap_valid = False
                        invalidation_reason = "Fine giro rilevata da posizione (timer TORCS non aggiornato)"
                        print(f"\n  TRAGUARDO (C)! {invalidation_reason}")

                prev_cur_lap_time = current_cur_lap
                prev_dist = current_dist

                # Uscita dal loop del giro
                if lap_completed or done:
                    if done and not lap_completed:
                        print("\n  [Info] Simulazione terminata esternamente (TORCS chiuso).")
                    break

                # Controllo del frame rate dinamico (50Hz)
                elapsed = time.perf_counter() - loop_start
                sleep_time = max(0.0, TARGET_DT - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            # Fine giro
            print(f"\n  --- Fine Giro (step totali: {step}) ---")

            # Salvataggio dei dati solo se il giro è valido
            if lap_completed and lap_valid:
                states_np = np.stack(lap_states)
                actions_np = np.stack(lap_actions)

                dists_np = np.asarray(lap_dists[:len(states_np)], dtype=np.float32)

                # Funzione di utilità per salvare in HDF5
                def _write_h5(path, st, ac, di):
                    with h5py.File(path, 'w') as h5f:
                        h5f.create_dataset('states', data=st, compression="gzip")
                        h5f.create_dataset('actions', data=ac, compression="gzip")
                        h5f.create_dataset('dist_from_start', data=di, compression="gzip")
                        h5f.attrs['lap_time'] = lap_time
                        h5f.attrs['num_steps'] = len(st)
                        h5f.attrs['has_dist_meta'] = True
                        h5f.attrs['timestamp'] = datetime.now().isoformat()

                # Se in modalità segment only salva solo quelli
                if args.segment_only:
                    segs = [(s, e) for (s, e) in _extract_segments(dists_np, zones, margin_steps=15) if e - s >= 20]
                    saved_names = []

                    # Per ogni segmento
                    for (s, e) in segs:
                        lap_counter += 1
                        # Trova la zona corrispondente al segmento
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
                # Giro scartato
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

            # Scrivi log su file
            with open(log_path, 'a') as f:
                f.write(log_entry + "\n")

    except KeyboardInterrupt:
        # Interruzione manuale: NON salvare il giro corrente
        print(f"\n\n{'=' * 64}")
        print(f"  SESSIONE TERMINATA (Ctrl+C)")
        print(f"     Giri salvati:   {session_saved}")
        print(f"     Giri scartati:  {session_discarded}")
        print(f"     Log sessione:   {log_path}")
        print(f"{'=' * 64}")
        print(f"  Giro corrente scartato (incompleto/interrotto).")

    finally:
        # Scrivi riepilogo finale nel log
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
