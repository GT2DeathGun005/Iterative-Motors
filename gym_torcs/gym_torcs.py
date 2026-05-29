import gym
from gym import spaces
import numpy as np
# from os import path
import snakeoil3_gym as snakeoil3
import numpy as np
import copy
import collections as col
import os
import time

# Directory di questo file (gym_torcs/) — usata per risolvere i path relativi
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_AUTOSTART_SH = os.path.join(_THIS_DIR, 'autostart.sh')


class TorcsEnv:
    terminal_judge_start = 150  # L'antistallo (speed limit) viene applicato dopo 150 step (3 secondi)
    termination_limit_progress = 20  # [km/h], episode terminates if car is running slower than this limit
    default_speed = 50

    initial_reset = True


    def __init__(self, vision=False, throttle=False, gear_change=False, early_termination=True):
       #print("Init")
        import shutil
        if shutil.which('xvfb-run') is None:
            raise EnvironmentError("xvfb-run non trovato. Installa il pacchetto 'xvfb' per l'esecuzione headless isolata di TORCS.")
            
        self.vision = vision
        self.throttle = throttle
        self.gear_change = gear_change
        self.early_termination = early_termination

        self.initial_run = True

        ##print("launch torcs")
        os.system('pkill -9 -f torcs')
        time.sleep(1.5)
        
        # Costruisce il comando torcs base
        torcs_cmd = 'torcs -nofuel -nodamage -vision' if self.vision else 'torcs -nofuel -nodamage'
        
        # Se la variabile SHOW_GUI è settata a 1, avvia normalmente. Altrimenti usa Xvfb.
        if os.environ.get('SHOW_GUI', '0') == '1':
            os.system(f'sh -c "(sleep 1.5 && sh {_AUTOSTART_SH}) & exec {torcs_cmd} > /dev/null 2>&1" &')
        else:
            xvfb_cmd = f'xvfb-run -a -s "-screen 0 640x480x24" sh -c "(sleep 1.5 && sh {_AUTOSTART_SH}) & exec {torcs_cmd} > /dev/null 2>&1"'
            os.system(f"{xvfb_cmd} &")
        
        time.sleep(3.0) # Attendi l'inizializzazione del server X virtuale, torcs e della macro

        """
        # Modify here if you use multiple tracks in the environment
        self.client = snakeoil3.Client(p=3101, vision=self.vision)  # Open new UDP in vtorcs
        self.client.MAX_STEPS = np.inf

        client = self.client
        client.get_servers_input()  # Get the initial input from torcs

        obs = client.S.d  # Get the current full-observation from torcs
        """
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
       #print("Step")
        # convert thisAction to the actual torcs actionstr
        client = self.client

        this_action = self.agent_to_torcs(u)

        # Apply Action
        action_torcs = client.R.d

        # Steering
        action_torcs['steer'] = this_action['steer']  # in [-1, 1]

        #  Simple Autnmatic Throttle Control by Snakeoil
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

        #  Automatic Gear Change by Snakeoil
        if self.gear_change is True:
            action_torcs['gear'] = this_action['gear']
        else:
            action_torcs['gear'] = 1


        # Save the privious full-obs from torcs for the reward calculation
        obs_pre = copy.deepcopy(client.S.d)

        # One-Step Dynamics Update #################################
        # Apply the Agent's action into torcs
        client.respond_to_server()
        # Get the response of TORCS
        client.get_servers_input()

        # Get the current full-observation from torcs
        obs = client.S.d

        # Make an obsevation from a raw observation vector from TORCS
        self.observation = self.make_observaton(obs)

        # ─── Reward Reshaping Unificato (SAC-Compatible) ───────────────────────
        sp_norm = obs['speedX'] / 50.0  # Range ~[0, 6]
        progress = sp_norm * np.cos(obs['angle'])
        
        # Inizializza last_steer se non esiste
        if not hasattr(self, 'last_steer'):
            self.last_steer = 0.0
            
        steer_smoothness = -0.1 * abs(this_action['steer'] - self.last_steer)
        self.last_steer = this_action['steer']

        time_penalty = -0.1
        reward = progress + time_penalty + steer_smoothness
        
        # info dict comunicherà al Replay Buffer se il done è un vero "crash"
        info = {'crash': False}

        # ─── Termination Conditions ──────────────────────────────────
        episode_terminate = False
        
        # Danno / Muro
        if obs['damage'] - obs_pre['damage'] > 0:
            reward = -10.0
            info['crash'] = True

        if self.early_termination:
            # Fuoripista (|trackPos| > 1.5)
            if abs(obs['trackPos']) > 1.5:
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

        if client.R.d['meta'] is True: # Send a reset signal
            self.initial_run = False
            client.respond_to_server()

        self.time_step += 1

        return self.get_obs(), reward, client.R.d['meta'] or client.so is None, info

    def reset(self, relaunch=False):
        #print("Reset")

        self.time_step = 0

        if self.initial_reset is not True:
            self.client.R.d['meta'] = True
            self.client.respond_to_server()

            ## TENTATIVE. Restarting TORCS every episode suffers the memory leak bug!
            if relaunch is True:
                # Chiudiamo esplicitamente il socket UDP client precedente per evitare conflitti di porta bindata
                if hasattr(self, 'client') and self.client is not None:
                    try:
                        self.client.so.close()
                    except Exception:
                        pass
                self.reset_torcs()
                print("### TORCS is RELAUNCHED ###")

        # Modify here if you use multiple tracks in the environment
        self.client = snakeoil3.Client(p=3001, vision=self.vision)  # Open new UDP in vtorcs
        self.client.MAX_STEPS = np.inf

        client = self.client
        client.get_servers_input()  # Get the initial input from torcs

        obs = client.S.d  # Get the current full-observation from torcs
        self.observation = self.make_observaton(obs)

        self.last_u = None
        self.last_steer = 0.0

        self.initial_reset = False
        return self.get_obs()

    def end(self):
        os.system('pkill -9 -f torcs')

    def get_obs(self):
        return self.observation

    def reset_torcs(self):
       #print("relaunch torcs")
        os.system('pkill -9 -f torcs')
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

        if self.throttle is True:  # throttle action is enabled
            torcs_action.update({'accel': u[1]})
            torcs_action.update({'brake': u[2]})

        if self.gear_change is True: # gear change action is enabled
            torcs_action.update({'gear': int(u[3])})

        return torcs_action


    def obs_vision_to_image_rgb(self, obs_image_vec):
        image_vec =  obs_image_vec
        rgb = []
        temp = []
        # convert size 64x64x3 = 12288 to 64x64=4096 2-D list 
        # with rgb values grouped together.
        # Format similar to the observation in openai gym
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
            # Get RGB from observation
            image_rgb = self.obs_vision_to_image_rgb(raw_obs['img'])
            obs_dict['img'] = image_rgb

        return obs_dict
