from gym import spaces 
import numpy as np
import snakeoil3_gym as snakeoil3
import copy
import os
import time

# Directory di questo file (gym_torcs/) — usata per risolvere i path relativi
_THIS_DIR = os.path.dirname(os.path.abspath(__file__)) # Variabile che contiene il percorso assoluto della directory in cui si trova questo file (gym_torcs/), usata per risolvere i path relativi in modo robusto.
_AUTOSTART_SH = os.path.join(_THIS_DIR, 'autostart.sh') # Path completo allo script di autostart.sh, che automatizza l'avvio di TORCS e la partenza della simulazione.


def _kill_torcs():
    """Termina le istanze TORCS.

    Di DEFAULT uccide *tutti* i processi torcs della macchina (`pkill -9 -f torcs`):
    è il workaround operativo contro il memory leak osservato nei run lunghi di TORCS
    ed è corretto per il flusso single-instance di training/test.

    """
    if os.environ.get('TORCS_KILL_ALL', '1') != '0':
        os.system('pkill -9 -f torcs')

class TorcsEnv:
    # Variabili usate per valutare la terminazione anticipata in caso di stallo della vettura
    terminal_judge_start = 500  # 10 secondi dopo la quale si inizia a valutare se la vettura è in stallo
    termination_limit_progress = 5  # Se la vettura non avanza di almeno 5 unità di distanza in 10 secondi, consideriamo che è in stallo e terminiamo l'episodio.
    
    default_speed = 50 # Velocità di riferimento per normalizzare speedX/Y/Z. Non è una velocità massima, ma un valore tipico di velocità in pista (50 m/s = 180 km/h) usato per scalare le osservazioni in modo che siano in un range più gestibile per l'allenamento degli agenti.
    # cambiando questo valore si scalano tutte le osservazioni di velocità (speedX/Y/Z) e anche il calcolo del reward (progress), quindi va scelto in modo coerente con le velocità tipiche che si vogliono raggiungere in pista. 
    # Un valore troppo basso potrebbe portare a osservazioni normalizzate troppo grandi, 
    # mentre un valore troppo alto potrebbe portare a osservazioni troppo piccole.
    # 50 m/s è una scelta buona perché rappresenta una velocità elevata ma raggiungibile in molte situazioni di gara.


    initial_reset = True    # Flag per indicare se è il primo reset (avvio) dell'ambiente. 

    # di default l'early termination è attivo, cioè l'episodio termina al primo contatto con muro/avversari o stallo.
    def __init__(self, early_termination=True):
        import shutil   # è una libreria utile all'elaborazione dei path nell'OS

        # Verifica che xvfb-run sia installato altrimenti lancia un errore di ambiente. 
        if shutil.which('xvfb-run') is None:
            raise EnvironmentError("xvfb-run non trovato. Installa il pacchetto 'xvfb' per l'esecuzione headless isolata di TORCS.")


        self.early_termination = early_termination
        self.initial_run = True


        _kill_torcs()
        time.sleep(1.5) #attende che il sistema operativo liberi la porta UDP usata da TORCS, altrimenti il successivo avvio fallisce.

        # Stringa che usiamo per lanciare torcs, in modalità no damage e no fuel
        torcs_cmd = 'torcs -nofuel -nodamage'
        

        # Se la variabile SHOW_GUI è settata a 1, avvia normalmente. Altrimenti usa Xvfb.
        if os.environ.get('SHOW_GUI', '0') == '1':
            os.system(f'sh -c "(sleep 1.5 && sh {_AUTOSTART_SH}) & exec {torcs_cmd} > /dev/null 2>&1" &')
        else:
            xvfb_cmd = f'xvfb-run -a -s "-screen 0 640x480x24" sh -c "(sleep 1.5 && sh {_AUTOSTART_SH}) & exec {torcs_cmd} > /dev/null 2>&1"'
            os.system(f"{xvfb_cmd} &")
        
        time.sleep(3.0)  # Attende Xvfb/TORCS e la macro di autostart.
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0, 1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 6.0], dtype=np.float32),
            dtype=np.float32,
        )

        #Dizionario con tutte le infomrazioni che invia TORCS, ogni infomrazione ha le sue dimensioni (visibili da shape) e range di valori
        self.observation_space = spaces.Dict({
            'focus': spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32),
            'speedX': spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float32),
            'speedY': spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float32),
            'speedZ': spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float32),
            'opponents': spaces.Box(low=-np.inf, high=np.inf, shape=(36,), dtype=np.float32),
            'rpm': spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float32),
            'track': spaces.Box(low=-np.inf, high=np.inf, shape=(19,), dtype=np.float32),
            'wheelSpinVel': spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32),
            'angle': spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float32),
            'trackPos': spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float32),
            'damage': spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float32),
            'curLapTime': spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float32),
            'lastLapTime': spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float32),
            'distFromStart': spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float32),
            'distRaced': spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float32),
        })

    # La funzione step prende in input l'azione dell'agente (u), la converte nel formato richiesto da TORCS, invia l'azione al server TORCS, riceve la nuova telemetria, calcola il reward e determina se l'episodio è terminato.
    def step(self, u):
        # Converte l'azione dell'agente nel formato richiesto dal server TORCS.
        client = self.client

        this_action = self.agent_to_torcs(u)

        # Apply Action
        action_torcs = client.R.d

        action_torcs['steer'] = this_action['steer']
        action_torcs['accel'] = this_action['accel']
        action_torcs['brake'] = this_action['brake']
        action_torcs['gear'] = this_action['gear']

        # Snapshot pre-step: serve a rilevare nuovo danno/muro nel reward.
        obs_pre = copy.deepcopy(client.S.d)

        # Step fisico: invia l'azione e legge la nuova telemetria dal server.
        client.respond_to_server()
        client.get_servers_input()

        obs = client.S.d

        # Converte la telemetria grezza TORCS nel dizionario normalizzato usato dagli script.
        self.observation = self.make_observaton(obs)

        # ─── Reward Reshaping condiviso dal TD3+BC ───────────────────────
        sp_norm = obs['speedX'] / self.default_speed  # Range ~[0, 6]
        progress = sp_norm * np.cos(obs['angle'])
        
        # Inizializza last_steer se non esiste cioè setta lo sterzo diritto al primo step
        if not hasattr(self, 'last_steer'):
            self.last_steer = 0.0

        # calcola la variazione di sterzo rispetto allo step precedente,
        # usata per penalizzare i cambi di direzione bruschi (zigzag) e
        # incentivare uno stile di guida più fluido. La penalità è proporzionale
        # alla variazione assoluta dello sterzo, con un coefficiente di 0.05 (usato sotto) che 
        # bilancia l'importanza di questo termine nel reward complessivo.
        steer_change = this_action['steer'] - self.last_steer
        self.last_steer = this_action['steer']

        # Calcolo della penalità per la posizione sul tracciato:
        # - Nessuna penalità se |trackPos| < 1.0 (vettura entro i bordi della pista)
        # - Penalità crescente (rampa quadratica) >= 1.0 e <= 1.25 (punisce il modello se va troppo fuori)
        # - Penalità massima se |trackPos| > 1.25 (considerato taglio curva/muro, episodio invalido)
        tp = abs(float(obs['trackPos']))
        pos_penalty = -2.0 * (max(0.0, tp - 1.0) ** 2)


        # Calcolo della reward complessiva per ogni step:
        # - La reward principale è il progresso in avanti
        # - A questo si aggiunge la penalità per la posizione fuori pista (pos_penalty)
        # - E si sottrae una penalità per i cambi di sterzo bruschi (zigzag) 
        # La scelta di mettere la reward in questo file è stata effettuata per convenienza
        # Avremmo potuto metterla in TD3+BC ma così è più semplice accedere alle variabili necessarie per il calcolo della stessa
        # Quali obs, obs_pre, last_steer, ecc... 
        # La reward 
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

            # Spin: auto rivolta nella direzione opposta al senso di marcia.
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
                # Chiudiamo esplicitamente il socket UDP aperto prima del relaunch.
                if hasattr(self, 'client') and self.client is not None:
                    try:
                        self.client.so.close()
                    except Exception:
                        pass
                self.reset_torcs()
                print("### TORCS is RELAUNCHED ###")

        self.client = snakeoil3.Client(p=3001, vision=False)  # Socket UDP SCR standard.
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
        
        torcs_cmd = 'torcs -nofuel -nodamage'
        
        # Se la variabile SHOW_GUI è settata a 1, avvia normalmente. Altrimenti usa Xvfb.
        if os.environ.get('SHOW_GUI', '0') == '1':
            os.system(f'sh -c "(sleep 1.5 && sh {_AUTOSTART_SH}) & exec {torcs_cmd} > /dev/null 2>&1" &')
        else:
            xvfb_cmd = f'xvfb-run -a -s "-screen 0 640x480x24" sh -c "(sleep 1.5 && sh {_AUTOSTART_SH}) & exec {torcs_cmd} > /dev/null 2>&1"'
            os.system(f"{xvfb_cmd} &")
        
        time.sleep(3.0)  # Tempo combinato per avvio e macro

    def agent_to_torcs(self, u):
        action = np.asarray(u, dtype=np.float32).flatten()
        if action.shape[0] != 4:
            raise ValueError(
                f"TorcsEnv.step richiede azioni [steer, accel, brake, gear], ricevuta shape {action.shape}."
            )
        return {
            'steer': float(action[0]),
            'accel': float(action[1]),
            'brake': float(action[2]),
            'gear': int(round(float(action[3]))),
        }

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

        return obs_dict
