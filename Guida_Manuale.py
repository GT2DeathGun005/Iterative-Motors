import snakeoil3_jm2 as snakeoil3
import time
import json
import os
import sys
from pynput.keyboard import Key, Listener

# --- CONFIGURAZIONE ---
MAX_LAP_TIME = 100.0          
MIN_SPEED_KMH = 10.0          
STUCK_TIMEOUT_TICKS = 150     
OFF_TRACK_THRESHOLD = 1.0     
STEER_SMOOTHING = 0.25        
RPM_UPSHIFT = 8000            
RPM_DOWNSHIFT = 3500          
MAX_GEAR = 6

# Sensori da loggare (Ho lasciato i tuoi 12 + curLapTime per coerenza)
SENSORS_TO_LOG = ['speedX', 'speedY', 'speedZ', 'trackPos', 'angle',
                  'rpm', 'track', 'wheelSpinVel', 'distFromStart',
                  'distRaced', 'gear', 'damage', 'curLapTime']

OUTPUT_DIR = "dataset_manuale"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class ArcadeController:
    """Controller Tastiera con cambio automatico e smoothing."""
    def __init__(self):
        self.keys = set()
        self.controls = {
            'steer': 0.0,
            'accel': 0.0,
            'brake': 0.0,
            'gear': 1,
            'clutch': 0.0,
            'meta': 0,
        }
        self.listener = Listener(on_press=self._on_press, on_release=self._on_release)
        self.listener.daemon = True
        self.listener.start()

    def _on_press(self, key):
        self.keys.add(key)

    def _on_release(self, key):
        self.keys.discard(key)

    def _auto_gear(self, rpm, current_gear, speed_kmh):
        if current_gear <= 0: return 1
        if rpm > RPM_UPSHIFT and current_gear < MAX_GEAR: return current_gear + 1
        if rpm < RPM_DOWNSHIFT and current_gear > 1 and speed_kmh > 5: return current_gear - 1
        return current_gear

    def get_input(self, sensors):
        # Acceleratore / Freno
        accel = 1.0 if Key.up in self.keys else 0.0
        brake = 1.0 if Key.down in self.keys else 0.0

        # Sterzo target
        target_steer = 0.0
        if Key.left in self.keys: target_steer = 0.5
        if Key.right in self.keys: target_steer = -0.5

        # Smoothing sterzo
        self.controls['steer'] += (target_steer - self.controls['steer']) * STEER_SMOOTHING
        self.controls['accel'] = accel
        self.controls['brake'] = brake

        # Cambio automatico
        rpm = sensors.get('rpm', 0.0)
        speed_kmh = sensors.get('speedX', 0.0) * 3.6
        self.controls['gear'] = self._auto_gear(rpm, self.controls['gear'], speed_kmh)
        
        return self.controls

    def stop(self):
        try: self.listener.stop()
        except: pass

def snapshot_sensors(sensors):
    snap = {}
    for k in SENSORS_TO_LOG:
        v = sensors.get(k)
        if isinstance(v, list):
            snap[k] = list(v) # Copia profonda dei 19 sensori track
        else:
            snap[k] = v
    return snap

def save_lap(lap_data, lap_time, lap_count, meta=None):
    fname = os.path.join(OUTPUT_DIR, f"lap_{lap_count:03d}.json")
    payload = {
        "lap_time": lap_time,
        "num_steps": len(lap_data),
        "meta": meta or {},
        "data": lap_data,
    }
    tmp = fname + ".tmp"
    try:
        with open(tmp, 'w') as f:
            json.dump(payload, f)
        os.replace(tmp, fname)
        return fname
    except Exception as e:
        print(f"❌ Errore salvataggio {fname}: {e}")
        return None

def main():
    client = snakeoil3.Client(p=3001)
    driver = ArcadeController()
    
    lap_data = []
    lap_is_valid = True
    invalid_reason = None
    prev_cur_lap_time = 0.0
    stuck_ticks = 0
    lap_count = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.json')])
    started_moving = False

    print("--- RACCOLTA DATI TASTIERA AVVIATA ---")
    print("Comandi: ↑ accel | ↓ freno | ← → sterzo")

    try:
        while True:
            client.get_servers_input()
            sensors = client.S.d
            
            # 1. Ottieni input e resetta meta a ogni tick
            ctrl = driver.get_input(sensors)
            client.R.d.update(ctrl)
            client.R.d['meta'] = 0 

            # Parametri attuali
            speed_kmh = sensors.get('speedX', 0.0) * 3.6
            track_pos = sensors.get('trackPos', 0.0)
            damage = sensors.get('damage', 0)

            # 2. Gestione Danni (BUG FIX: Restart obbligatorio)
            if damage > 0:
                print(f"⚠️ DANNI RILEVATI ({damage}). Reset gara...")
                client.R.d['meta'] = 1
                client.respond_to_server()
                # Reset totale stato locale
                lap_data, stuck_ticks = [], 0
                lap_is_valid, started_moving = True, False
                prev_cur_lap_time = 0.0
                time.sleep(1.0) 
                continue

            # 3. Validazione Fuori Pista
            if lap_is_valid and abs(track_pos) > OFF_TRACK_THRESHOLD:
                lap_is_valid = False
                invalid_reason = "Fuori pista"
                print("⚠️ FUORI PISTA! Giro invalidato.")

            # 4. Validazione Auto Ferma
            if not started_moving and speed_kmh > MIN_SPEED_KMH:
                started_moving = True
            
            if started_moving:
                if speed_kmh < MIN_SPEED_KMH:
                    stuck_ticks += 1
                else:
                    stuck_ticks = 0
                
                if lap_is_valid and stuck_ticks > STUCK_TIMEOUT_TICKS:
                    lap_is_valid = False
                    invalid_reason = "Auto ferma/lenta"
                    print("⚠️ AUTO FERMA! Giro invalidato.")

            # 5. Registrazione step
            if lap_is_valid:
                lap_data.append({
                    "sensors": snapshot_sensors(sensors),
                    "actions": {
                        'steer': ctrl['steer'],
                        'accel': ctrl['accel'],
                        'brake': ctrl['brake'],
                        'gear': ctrl['gear'],
                    }
                })

            # 6. Detection fine giro
            cur_lap_time = sensors.get('curLapTime', 0.0)
            last_lap_time = sensors.get('lastLapTime', 0.0)
            lap_finished = (prev_cur_lap_time > 1.0 and cur_lap_time < prev_cur_lap_time - 0.5)
            prev_cur_lap_time = cur_lap_time

            if lap_finished:
                print(f"\n🏁 Giro completato in {last_lap_time:.2f}s")
                if lap_is_valid and 0 < last_lap_time <= MAX_LAP_TIME and len(lap_data) > 50:
                    lap_count += 1
                    meta = {"hz": len(lap_data)/last_lap_time if last_lap_time > 0 else 0}
                    fname = save_lap(lap_data, last_lap_time, lap_count, meta)
                    if fname: print(f"✅ SALVATO: {fname}\n")
                else:
                    motivo = invalid_reason or "Tempo non valido/Dati insufficienti"
                    print(f"❌ SCARTATO: {motivo}\n")
                
                # Reset per nuovo giro
                lap_data, stuck_ticks = [], 0
                lap_is_valid, started_moving = True, False

            # 7. Auto-restart se invalidato e fermo (BUG FIX: No doppie risposte)
            if not lap_is_valid and started_moving and speed_kmh < 2.0 and stuck_ticks > 50:
                print("🔄 Restarting...")
                client.R.d['meta'] = 1
                lap_data, stuck_ticks = [], 0
                lap_is_valid, started_moving = True, False
                prev_cur_lap_time = 0.0
                # La risposta verrà inviata riga sotto

            client.respond_to_server()

    except KeyboardInterrupt:
        print("\nChiusura...")
    finally:
        driver.stop()
        try: client.shutdown()
        except: pass
        print(f"Giri totali: {lap_count}")

if __name__ == "__main__":
    main()