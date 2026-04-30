"""
Modulo di Data Collection (Human-in-the-Loop)
Registra le interazioni di guida da un controller PS5 per Imitation Learning.
Ottimizzato per salvare lo stato e le azioni in un file HDF5.
"""

import os
import time
import argparse
import numpy as np
import h5py
import pygame
from typing import Tuple, Dict

import sys

# Aggiungo la directory di gym_torcs in cima al path per assicurarmi di usare la versione modificata
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../gym_torcs')))

# Assumiamo la presenza di gym_torcs. Se non disponibile, l'utente dovrà installarlo o fornirne il wrapper.
try:
    from gym_torcs import TorcsEnv
except ImportError:
    print("Warning: gym_torcs non trovato. Per l'esecuzione sarà necessario il modulo TorcsEnv.")

class DualSenseController:
    """Gestisce il polling del controller PlayStation 5 tramite Pygame."""
    def __init__(self):
        pygame.init()
        pygame.joystick.init()
        
        if pygame.joystick.get_count() == 0:
            raise RuntimeError("Nessun controller rilevato. Collega un DualSense e riprova.")
            
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        
        print(f"Controller inizializzato: {self.joystick.get_name()}")
        
        # Mappatura standard per DualSense su Linux/Pygame
        self.AXIS_STEER = 0       # Left Thumbstick X
        self.AXIS_L2 = 2          # Brake (L2)
        self.AXIS_R2 = 5          # Accel (R2)
        
        self.BTN_CROSS = 0        # X (Downshift)
        self.BTN_CIRCLE = 1       # Circle
        self.BTN_TRIANGLE = 2     # Triangle (Reverse)
        self.BTN_SQUARE = 3       # Square (Upshift)

        self.gear = 1  # Iniziamo in prima marcia

    def get_action(self) -> np.ndarray:
        """
        Legge gli assi e i bottoni dal controller e restituisce l'azione.
        Ritorna un array: [steering, accel, brake, gear]
        """
        pygame.event.pump()
        
        # Sterzo continuo: [-1.0, 1.0]. Invertito per correggere la direzione
        steering = -self.joystick.get_axis(self.AXIS_STEER)
        
        # I grilletti L2/R2 partono da -1.0 (rilasciati) a 1.0 (premuti).
        # Bug Pygame/Linux: all'avvio i grilletti restituiscono ~0.0 finché non vengono premuti la prima volta.
        raw_r2 = self.joystick.get_axis(self.AXIS_R2)
        if not hasattr(self, "r2_init"): self.r2_init = False
        if abs(raw_r2) > 0.1: self.r2_init = True
        accel = (raw_r2 + 1.0) / 2.0 if self.r2_init else 0.0
        if accel < 0.05: accel = 0.0
        
        raw_l2 = self.joystick.get_axis(self.AXIS_L2)
        if not hasattr(self, "l2_init"): self.l2_init = False
        if abs(raw_l2) > 0.1: self.l2_init = True
        brake = (raw_l2 + 1.0) / 2.0 if self.l2_init else 0.0
        if brake < 0.05: brake = 0.0
        
        # Gestione cambio
        if self.joystick.get_button(self.BTN_SQUARE):
            if self.gear < 6:
                self.gear += 1
                print(f"[Cambio] Upshift -> Marcia {self.gear}")
            time.sleep(0.2)  # Debounce rudimentale
        elif self.joystick.get_button(self.BTN_CROSS):
            if self.gear > 0:
                self.gear -= 1
                print(f"[Cambio] Downshift -> Marcia {self.gear}")
            time.sleep(0.2)
        elif self.joystick.get_button(self.BTN_TRIANGLE):
            self.gear = -1  # Retromarcia
            print(f"[Cambio] Retromarcia inserita (-1)")
            time.sleep(0.2)
            
        # Per semplicità in molti wrapper TORCS: [steering, accel, brake]
        # Includiamo il gear se il wrapper lo supporta, altrimenti si usa solo i primi 3.
        # Restituiamo un vettore continuo a 4 dimensioni (il gear viene castato a float)
        return np.array([steering, accel, brake, float(self.gear)], dtype=np.float32)

class DataCollector:
    """
    Raccoglie i dati dall'ambiente TORCS e li salva in HDF5.
    """
    def __init__(self, output_file: str = "dataset.h5"):
        self.output_file = output_file
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        
    def add_step(self, state: np.ndarray, action: np.ndarray, reward: float, done: bool):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        
    def save(self):
        """Salva i buffer in un file HDF5."""
        print(f"Salvataggio di {len(self.states)} step nel file {self.output_file}...")
        
        if not self.states:
            print("Nessun dato da salvare. File non creato.")
            return

        # Assicuriamoci che i vettori siano numpy arrays omogenei
        states_np = np.stack(self.states)
        actions_np = np.stack(self.actions)
        rewards_np = np.array(self.rewards, dtype=np.float32)
        dones_np = np.array(self.dones, dtype=np.bool_)
        
        with h5py.File(self.output_file, 'w') as h5f:
            h5f.create_dataset('states', data=states_np, compression="gzip")
            h5f.create_dataset('actions', data=actions_np, compression="gzip")
            h5f.create_dataset('rewards', data=rewards_np, compression="gzip")
            h5f.create_dataset('dones', data=dones_np, compression="gzip")
            
        print("Salvataggio completato con successo.")

def flatten_state(state_dict: Dict) -> np.ndarray:
    """
    Il wrapper gym_torcs solitamente ritorna un dizionario di sensori.
    Appiattisce i sensori in un singolo vettore 1D continuo.
    Dimensione tipica standard: 29 (1 angle, 3 speedX/Y/Z, 19 track, 4 wheelSpinVel, 1 rpm, 1 trackPos)
    """
    state_vec = np.hstack((
        state_dict['angle'],
        np.array(state_dict['track']),
        state_dict['trackPos'],
        state_dict['speedX'],
        state_dict['speedY'],
        state_dict['speedZ'],
        np.array(state_dict['wheelSpinVel']) / 100.0,
        state_dict['rpm'] / 10000.0
    ))
    return np.array(state_vec, dtype=np.float32)

def main():
    parser = argparse.ArgumentParser(description="Data Collection for TORCS")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to record")
    parser.add_argument("--output", type=str, default="human_expert.h5", help="Output HDF5 file")
    args = parser.parse_args()

    controller = DualSenseController()
    collector = DataCollector(output_file=args.output)
    
    # Inizializza l'ambiente TORCS con rendering attivo (vision=False è solo per le telecamere, 
    # di solito il gioco apre la finestra 3D da solo)
    env = TorcsEnv(vision=False, throttle=True, gear_change=True)
    
    print("Inizio fase di Data Collection.")
    print("Premi CTRL+C nel terminale per fermare e salvare anticipatamente.")

    try:
        for ep in range(args.episodes):
            print(f"--- Episodio {ep+1}/{args.episodes} ---")
            
            # Restart environment (launching/relaunching TORCS client)
            if ep == 0:
                ob = env.reset(relaunch=True)
            else:
                ob = env.reset()
                
            state_vec = flatten_state(ob)
            
            step_count = 0
            while True:
                step_count += 1
                action = controller.get_action()
                
                # Se l'azione è a 4 dimensioni [steer, accel, brake, gear]
                # gym_torcs step richiede tipicamente un dizionario o una lista specifica
                # L'implementazione standard accetta una lista
                
                ob_next, reward, done, info = env.step(action)
                next_state_vec = flatten_state(ob_next)
                
                collector.add_step(state_vec, action, reward, done)
                state_vec = next_state_vec
                
                # Sincronizzazione al frame rate di torcs (tipicamente 50Hz, sleep 20ms)
                time.sleep(0.02)
                
                if done:
                    print(f". Episodio concluso al passo {step_count}.")
                    break

    except KeyboardInterrupt:
        print("\nInterruzione manuale da tastiera.")
    finally:
        collector.save()
        env.end()
        pygame.quit()

if __name__ == "__main__":
    main()
