import snakeoil3_jm2 as snakeoil3
import time
import json
import os
import sys
import pygame # <--- AGGIUNTO PER IL CONTROLLER

# --- CONFIGURAZIONE ---
MAX_LAP_TIME = 100.0
MIN_SPEED_KMH = 10.0
STUCK_TIMEOUT_TICKS = 150
OFF_TRACK_THRESHOLD = 1.0
STEER_SMOOTHING = 0.25
RPM_UPSHIFT = 8000
RPM_DOWNSHIFT = 3500
MAX_GEAR = 6
SENSORS_TO_LOG = ['speedX', 'speedY', 'speedZ', 'trackPos', 'angle',
                  'rpm', 'track', 'wheelSpinVel', 'distFromStart',
                  'distRaced', 'gear', 'damage']
OUTPUT_DIR = "dataset_manuale"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class GamepadController:
    """Controller basato su Pygame per usare un Gamepad (es. Xbox Controller)"""
    def __init__(self):
        pygame.init()
        pygame.joystick.init()
        self.joystick = None
        
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print(f"🎮 Controller collegato: {self.joystick.get_name()}")
        else:
            print("⚠️ NESSUN CONTROLLER RILEVATO! (Collega il controller prima di avviare lo script)")
            sys.exit(1)

        self.controls = {
            'steer': 0.0,
            'accel': 0.0,
            'brake': 0.0,
            'gear': 1,
            'clutch': 0.0,
            'meta': 0,
        }

    def _auto_gear(self, rpm, current_gear, speed_kmh):
        if current_gear <= 0: return 1
        if rpm > RPM_UPSHIFT and current_gear < MAX_GEAR: return current_gear + 1
        if rpm < RPM_DOWNSHIFT and current_gear > 1 and speed_kmh > 5: return current_gear - 1
        return current_gear

    def get_input(self, sensors):
        pygame.event.pump() # Aggiorna lo stato di pygame
        
        if self.joystick:
            # 1. STERZO: Analogico Sinistro (Asse 0)
            raw_steer = self.joystick.get_axis(0)
            if abs(raw_steer) < 0.15: raw_steer = 0.0 # Deadzone per evitare drift
            
            # 2. ACCELERATORE/FRENO: Trigger (LT/RT) o Pulsanti (A/X)
            # Nota: Negli Xbox controller, Asse 5 = RT (Gas), Asse 4 = LT (Freno)
            # Vanno da -1 (rilasciato) a +1 (premuto), quindi li normalizziamo tra 0 e 1.
            try:
                accel = max(0.0, (self.joystick.get_axis(5) + 1.0) / 2.0)
                brake = max(0.0, (self.joystick.get_axis(4) + 1.0) / 2.0)
            except pygame.error:
                accel = 0.0
                brake = 0.0
            
            # Fallback ai tasti (es. A per gas, B/X per freno) se i trigger non vengono letti come assi
            if self.joystick.get_button(0): accel = 1.0
            if self.joystick.get_button(1) or self.joystick.get_button(2): brake = 1.0

            # Smoothing dello sterzo
            self.controls['steer'] += (raw_steer - self.controls['steer']) * STEER_SMOOTHING
            self.controls['accel'] = accel
            self.controls['brake'] = brake

        # Cambio automatico
        rpm = sensors.get('rpm', 0.0)
        speed_kmh = sensors.get('speedX', 0.0) * 3.6
        self.controls['gear'] = self._auto_gear(rpm, self.controls['gear'], speed_kmh)
        return self.controls

    def stop(self):
        pygame.quit()

def snapshot_sensors(sensors):
    snap = {}
    for k in SENSORS_TO_LOG:
        v = sensors.get(k)
        if isinstance(v, list):
            snap[k] = list(v)
        else:
            snap[k] = v
    return snap

def save_lap(lap_data, lap_time, lap_count, meta=None):
    fname = os.path.join(OUTPUT_DIR, f"lap_{lap_count:03d}.json")
    payload = {"lap_time": lap_time, "num_steps": len(lap_data), "meta": meta or {}, "data": lap_data}
    tmp = fname + ".tmp"
    try:
        with open(tmp, 'w') as f:
            json.dump(payload, f)
        os.replace(tmp, fname)
        return fname
    except Exception as e:
        print(f"❌ Errore salvataggio {fname}: {e}")
        if os.path.exists(tmp): os.remove(tmp)
        return None

def main():
    client = snakeoil3.Client(p=3001)
    driver = GamepadController() # <--- Usa il nuovo controller
    
    lap_data = []
    lap_is_valid = True
    invalid_reason = None
    prev_cur_lap_time = 0.0
    stuck_ticks = 0
    lap_count = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.json')])
    started_moving = False

    print("\n--- RACCOLTA DATI AVVIATA (MODALITÀ GAMEPAD) ---")
    
    try:
        while True:
            client.get_servers_input()
            sensors = client.S.d
            
            # --- Lettura Input Controller ---
            ctrl = driver.get_input(sensors)
            client.R.d.update(ctrl)

            # Estrazione metriche
            track_pos = sensors.get('trackPos', 0.0)
            speed_kmh = sensors.get('speedX', 0.0) * 3.6
            damage = sensors.get('damage', 0.0)

            # --- Validazione Danni (Forza Restart Immediato) ---
            if damage > 0:
                print(f"⚠️ DANNI SUBITI ({damage}). Restart immediato richiesto!")
                client.R.d['meta'] = 1 # Chiede il restart a TORCS
                client.respond_to_server() # Invia subito
                
                # Resetta variabili locali
                lap_data, stuck_ticks = [], 0
                lap_is_valid, started_moving = True, False
                prev_cur_lap_time = 0.0
                time.sleep(1) # Attendi che TORCS resetti
                continue # Salta il resto del loop per non inquinare i dati

            # --- Validazione Fuori Pista ---
            if lap_is_valid and abs(track_pos) > OFF_TRACK_THRESHOLD:
                lap_is_valid, invalid_reason = False, "Fuori pista"
                print("⚠️ FUORI PISTA! Giro invalidato.")

            # --- Validazione Auto Ferma ---
            if not started_moving and speed_kmh > MIN_SPEED_KMH:
                started_moving = True
            
            if started_moving:
                stuck_ticks = stuck_ticks + 1 if speed_kmh < MIN_SPEED_KMH else 0
                if lap_is_valid and stuck_ticks > STUCK_TIMEOUT_TICKS:
                    lap_is_valid, invalid_reason = False, "Auto ferma"
                    print("⚠️ AUTO FERMA TROPPO A LUNGO! Giro invalidato.")

            # Registrazione Step
            if lap_is_valid:
                lap_data.append({
                    "sensors": snapshot_sensors(sensors),
                    "actions": {'steer': ctrl['steer'], 'accel': ctrl['accel'], 'brake': ctrl['brake'], 'gear': ctrl['gear']}
                })

            # --- Detection fine giro ---
            cur_lap_time = sensors.get('curLapTime', 0.0)
            last_lap_time = sensors.get('lastLapTime', 0.0)
            lap_finished = (prev_cur_lap_time > 1.0 and cur_lap_time < prev_cur_lap_time - 0.5)
            prev_cur_lap_time = cur_lap_time

            if lap_finished:
                print(f"\n🏁 Giro completato in {last_lap_time:.2f}s")
                if lap_is_valid and 0 < last_lap_time <= MAX_LAP_TIME and len(lap_data) > 50:
                    lap_count += 1
                    meta = {"max_lap_time": MAX_LAP_TIME, "hz": len(lap_data) / last_lap_time if last_lap_time > 0 else 0}
                    fname = save_lap(lap_data, last_lap_time, lap_count, meta)
                    if fname: print(f"✅ SALVATO: {fname} ({len(lap_data)} step)\n")
                else:
                    motivo = invalid_reason or ("Troppo lento" if last_lap_time > MAX_LAP_TIME else "Dati insufficienti")
                    print(f"❌ SCARTATO: {motivo}\n")
                
                # Reset
                lap_data, stuck_ticks = [], 0
                lap_is_valid, started_moving = True, False

            # --- Auto-Restart per auto impantanate (FIXED BUG) ---
            if not lap_is_valid and started_moving and speed_kmh < 2.0 and stuck_ticks > STUCK_TIMEOUT_TICKS:
                print("🔄 Restart gara richiesto per auto impantanata...")
                client.R.d['meta'] = 1 # Settiamo meta=1 senza fare respond_to_server() doppio!
                lap_data, stuck_ticks = [], 0
                lap_is_valid, started_moving = True, False
                prev_cur_lap_time = 0.0
                time.sleep(0.5)
            else:
                client.R.d['meta'] = 0

            # Invia i comandi a TORCS (unica chiamata corretta)
            client.respond_to_server()

    except KeyboardInterrupt:
        print("\nChiusura richiesta dall'utente...")
    except Exception as e:
        print(f"\n❌ Errore inatteso: {e}", file=sys.stderr)
    finally:
        driver.stop()
        try: client.shutdown()
        except: pass
        print(f"Totale giri salvati: {lap_count}")

if __name__ == "__main__":
    main()