import gym
from gym import spaces
import numpy as np
import snakeoil3_gym as snakeoil3
import copy
import os
import time

# Directory di questo file (gym_torcs/) — usata per risolvere i path relativi
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_AUTOSTART_SH = os.path.join(_THIS_DIR, 'autostart.sh')


def _kill_torcs():
    """Termina le istanze TORCS.

    Di DEFAULT uccide *tutti* i processi torcs della macchina (`pkill -9 -f torcs`):
    è la base del workaround per il memory-leak di TORCS (vedi ARCHITECTURE §5) ed è
    corretto per il flusso single-instance di training/test.

    ⚠️ In scenari MULTI-istanza (run paralleli) o MULTI-vettura (video finale) questo
    ucciderebbe anche gli altri TORCS: imposta `TORCS_KILL_ALL=0` per disabilitare il
    kill globale (in quel caso gestisci tu la terminazione dell'istanza specifica).
    """
    if os.environ.get('TORCS_KILL_ALL', '1') != '0':
        os.system('pkill -9 -f torcs')


class TorcsEnv:
    terminal_judge_start = 500  # 10 secondi per consentire il transitorio di partenza
    termination_limit_progress = 5  # Soglia tollerante per non punire le incertezze
    default_speed = 50

    initial_reset = True


    def __init__(self, vision=False, throttle=False, gear_change=False, early_termination=True):
        import shutil
        if shutil.which('xvfb-run') is None:
            raise EnvironmentError("xvfb-run non trovato. Installa il pacchetto 'xvfb' per l'esecuzione headless isolata di TORCS.")
            
        self.vision = vision
        self.throttle = throttle
        self.gear_change = gear_change
        self.early_termination = early_termination

        self.initial_run = True

        _kill_torcs()
        time.sleep(1.5)
        
        # Costruisce il comando torcs base
        torcs_cmd = 'torcs -nofuel -nodamage -vision' if self.vision else 'torcs -nofuel -nodamage'
        
        # Se la variabile SHOW_GUI è settata a 1, avvia normalmente. Altrimenti usa Xvfb.
        if os.environ.get('SHOW_GUI', '0') == '1':
            os.system(f'sh -c "(sleep 1.5 && sh {_AUTOSTART_SH}) & exec {torcs_cmd} > /dev/null 2>&1" &')
        else:
            xvfb_cmd = f'xvfb-run -a -s "-screen 0 640x480x24" sh -c "(sleep 1.5 && sh {_AUTOSTART_SH}) & exec {torcs_cmd} > /dev/null 2>&1"'
            os.system(f"{xvfb_cmd} &")
        
        time.sleep(3.0)  # Attende Xvfb/TORCS e la macro di autostart.
        if throttle is False:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))

        if vision is False:
            high = np.array([1., np.inf, np.inf, np.inf, 1., np.inf, 1., np.inf])
            low = np.array([0., -np.inf, -np.inf, -np.inf, 0., -np.inf, 0., -np.inf])
            self.observation_space = spaces.Box(low=low, high=high)
        else:
            high = np.array([1., np.inf, np.inf, np.inf, 1., np.inf, 1., np.inf, 255])
            low = np.array([0., -np.inf, -np.inf, -np.inf, 0., -np.inf, 0., -np.inf, 0])
            self.observation_space = spaces.Box(low=low, high=high)

    def step(self, u):
        # Converte l'azione dell'agente nel formato richiesto dal server TORCS.
        client = self.client

        this_action = self.agent_to_torcs(u)

        # Apply Action
        action_torcs = client.R.d

        # Steering
        action_torcs['steer'] = this_action['steer']  # in [-1, 1]

        # Controllo automatico minimale di Snakeoil, usato solo se throttle=False.
        if self.throttle is False:
            target_speed = self.default_speed
            if client.S.d['speedX'] < target_speed - (client.R.d['steer']*50):
                client.R.d['accel'] += .01
            else:
                client.R.d['accel'] -= .01

            if client.R.d['accel'] > 0.2:
                client.R.d['accel'] = 0.2

            if client.S.d['speedX'] < 10:
                client.R.d['accel'] += 1/(client.S.d['speedX']+.1)

            # Traction Control System
            if ((client.S.d['wheelSpinVel'][2]+client.S.d['wheelSpinVel'][3]) -
               (client.S.d['wheelSpinVel'][0]+client.S.d['wheelSpinVel'][1]) > 5):
                action_torcs['accel'] -= .2
        else:
            action_torcs['accel'] = this_action['accel']
            action_torcs['brake'] = this_action.get('brake', 0.0)

        # Cambio marcia: in AIcar è gestito dall'agente/gearing.py quando gear_change=True.
        if self.gear_change is True:
            action_torcs['gear'] = this_action['gear']
        else:
            action_torcs['gear'] = 1


        # Osservazione precedente: serve a rilevare nuovo danno/muro nel reward.
        obs_pre = copy.deepcopy(client.S.d)

        # Step fisico: invia l'azione e legge la nuova telemetria dal server.
        client.respond_to_server()
        client.get_servers_input()

        obs = client.S.d

        # Converte la telemetria grezza TORCS nel dizionario normalizzato usato dagli script.
        self.observation = self.make_observaton(obs)

        # ─── Reward Reshaping condiviso dal TD3+BC ───────────────────────
        sp_norm = obs['speedX'] / 50.0  # Range ~[0, 6]
        progress = sp_norm * np.cos(obs['angle'])
        
        # Inizializza last_steer se non esiste
        if not hasattr(self, 'last_steer'):
            self.last_steer = 0.0
            
        steer_change = this_action['steer'] - self.last_steer
        self.last_steer = this_action['steer']

        # Penalità di posizione con DEADZONE: nessuna penalità entro |trackPos| < 1.0
        # (libertà piena sulla pista), poi una rampa morbida nella fascia dei cordoli
        # 1.0→1.25 come margine prima del limite di GIRO VALIDO. Oltre 1.25 = taglio/uscita
        # → terminale (sotto). Coerente coi limiti usati in raccolta dati (|trackPos| ≤ 1.25).
        tp = abs(float(obs['trackPos']))
        pos_penalty = -2.0 * (max(0.0, tp - 1.0) ** 2)

        # Reward da corsa: massimizza il progresso (velocità in avanti) lasciando l'agente
        # libero su staccate e velocità in curva; lo steer-smoothness è un lieve anti-zigzag.
        # NB: il bonus +50 per GIRO VALIDO completato è applicato nel loop di training di
        # td3_bc.py (dove si rileva il cambio di lastLapTime e si salva td3_expl_best_lap.pth),
        # NON qui — altrimenti si conterebbe due volte.
        reward = (progress * 1.5) + pos_penalty - (0.05 * abs(steer_change))

        # info dict comunicherà al Replay Buffer se il done è un vero "crash"
        info = {'crash': False}

        # ─── Termination Conditions ──────────────────────────────────
        episode_terminate = False
        
        # Danno / Muro. Penalità e flag crash SEMPRE attivi (coerenza reward/Critic). La
        # TERMINAZIONE invece solo se early_termination=True (training RL): così la transizione
        # con mask=0 corrisponde a un episodio realmente chiuso lato TORCS, senza l'incoerenza
        # segnalata. Durante la RACCOLTA DATI umana (early_termination=False) il danno NON termina,
        # altrimenti un contatto/cordolo ucciderebbe il giro del pilota.
        if obs['damage'] - obs_pre['damage'] > 0:
            reward = -10.0
            info['crash'] = True
            if self.early_termination:
                episode_terminate = True
                client.R.d['meta'] = True

        if self.early_termination:
            # Giro NON valido: oltre |trackPos| > 1.25 (taglio curva / muro). È lo stesso limite
            # usato in raccolta dati (cordoli consentiti fino a 1.25, oltre = invalido).
            if abs(obs['trackPos']) > 1.25:
                reward = -10.0
                info['crash'] = True
                episode_terminate = True
                client.R.d['meta'] = True

            # Stallo
            if self.terminal_judge_start < self.time_step:
                if progress < (self.termination_limit_progress / 50.0):
                    reward = -10.0
                    info['crash'] = True
                    episode_terminate = True
                    client.R.d['meta'] = True

            # Spin (Retromarcia)
            if np.cos(obs['angle']) < 0:
                reward = -10.0
                info['crash'] = True
                episode_terminate = True
                client.R.d['meta'] = True

        if client.R.d['meta'] is True:
            self.initial_run = False
            client.respond_to_server()

        self.time_step += 1

        return self.get_obs(), reward, client.R.d['meta'] or client.so is None, info

    def reset(self, relaunch=False):
        self.time_step = 0

        if self.initial_reset is not True:
            self.client.R.d['meta'] = True
            self.client.respond_to_server()

            if relaunch is True:
                # Chiudiamo esplicitamente il socket UDP client precedente per evitare conflitti di porta bindata
                if hasattr(self, 'client') and self.client is not None:
                    try:
                        self.client.so.close()
                    except Exception:
                        pass
                self.reset_torcs()
                print("### TORCS is RELAUNCHED ###")

        self.client = snakeoil3.Client(p=3001, vision=self.vision)  # Socket UDP SCR standard.
        self.client.MAX_STEPS = np.inf

        client = self.client
        client.get_servers_input()

        obs = client.S.d
        self.observation = self.make_observaton(obs)

        self.last_u = None
        self.last_steer = 0.0

        self.initial_reset = False
        return self.get_obs()

    def end(self):
        _kill_torcs()

    def get_obs(self):
        return self.observation

    def reset_torcs(self):
        _kill_torcs()
        time.sleep(1.5)  # Garantisce che il sistema operativo liberi la porta UDP
        
        torcs_cmd = 'torcs -nofuel -nodamage -vision' if self.vision else 'torcs -nofuel -nodamage'
        
        # Se la variabile SHOW_GUI è settata a 1, avvia normalmente. Altrimenti usa Xvfb.
        if os.environ.get('SHOW_GUI', '0') == '1':
            os.system(f'sh -c "(sleep 1.5 && sh {_AUTOSTART_SH}) & exec {torcs_cmd} > /dev/null 2>&1" &')
        else:
            xvfb_cmd = f'xvfb-run -a -s "-screen 0 640x480x24" sh -c "(sleep 1.5 && sh {_AUTOSTART_SH}) & exec {torcs_cmd} > /dev/null 2>&1"'
            os.system(f"{xvfb_cmd} &")
        
        time.sleep(3.0)  # Tempo combinato per avvio e macro

    def agent_to_torcs(self, u):
        torcs_action = {'steer': u[0]}

        if self.throttle is True:
            torcs_action.update({'accel': u[1]})
            torcs_action.update({'brake': u[2]})

        if self.gear_change is True:
            torcs_action.update({'gear': int(u[3])})

        return torcs_action


    def obs_vision_to_image_rgb(self, obs_image_vec):
        image_vec =  obs_image_vec
        rgb = []
        temp = []
        # Converte il vettore immagine 64x64x3 in righe RGB, formato Gym-like.
        for i in range(0,12286,3):
            temp.append(image_vec[i])
            temp.append(image_vec[i+1])
            temp.append(image_vec[i+2])
            rgb.append(temp)
            temp = []
        return np.array(rgb, dtype=np.uint8)

    def make_observaton(self, raw_obs):
        obs_dict = {
            'focus': np.array(raw_obs['focus'], dtype=np.float32)/200.,
            'speedX': np.array(raw_obs['speedX'], dtype=np.float32)/self.default_speed,
            'speedY': np.array(raw_obs['speedY'], dtype=np.float32)/self.default_speed,
            'speedZ': np.array(raw_obs['speedZ'], dtype=np.float32)/self.default_speed,
            'opponents': np.array(raw_obs['opponents'], dtype=np.float32)/200.,
            'rpm': np.array(raw_obs['rpm'], dtype=np.float32),
            'track': np.array(raw_obs['track'], dtype=np.float32)/200.,
            'wheelSpinVel': np.array(raw_obs['wheelSpinVel'], dtype=np.float32),
            'angle': np.array(raw_obs['angle'], dtype=np.float32),
            'trackPos': np.array(raw_obs['trackPos'], dtype=np.float32),
            'damage': np.array(raw_obs['damage'], dtype=np.float32),
            # Lap timing and distance sensors (raw, not normalized)
            'curLapTime': np.array(raw_obs.get('curLapTime', 0.0), dtype=np.float32),
            'lastLapTime': np.array(raw_obs.get('lastLapTime', 0.0), dtype=np.float32),
            'distFromStart': np.array(raw_obs.get('distFromStart', 0.0), dtype=np.float32),
            'distRaced': np.array(raw_obs.get('distRaced', 0.0), dtype=np.float32),
        }

        if self.vision is True:
            # Converte la visione grezza in RGB solo se vision=True.
            image_rgb = self.obs_vision_to_image_rgb(raw_obs['img'])
            obs_dict['img'] = image_rgb

        return obs_dict
