from gym import spaces
import numpy as np
from . import snakeoil3_gym as snakeoil3
import os
import time

# Directory of this file (gym_torcs/) — used to resolve relative paths
_THIS_DIR = os.path.dirname(os.path.abspath(__file__)) # Variable holding the absolute path of the directory in which this file (gym_torcs/) resides, used to resolve relative paths robustly.
_AUTOSTART_SH = os.path.join(_THIS_DIR, 'autostart.sh') # Full path to the autostart.sh script, which automates the TORCS startup and the start of the simulation.
_REQUIRED_OBS_KEYS = (
    'focus', 'speedX', 'speedY', 'speedZ', 'opponents', 'rpm', 'track',
    'wheelSpinVel', 'angle', 'trackPos', 'damage',
)


def _kill_torcs():
    """Terminates the TORCS instances.

    By DEFAULT it kills *all* the machine's torcs processes (`pkill -9 -f torcs`):
    it is the operational workaround against the memory leak observed in long TORCS runs
    and is correct for the single-instance training/test flow.

    TODO(parallel): to speed up the RL with N TORCS environments in parallel (ports 3001+i,
    separate Xvfb displays) this global kill must be replaced with a per-instance teardown
    (tracked PID/port). It is the point to modify when evaluating the speedup; see also
    the ``EnvRunner`` seam and ``updates_per_step`` documented in train (deferred point of the plan).
    """
    if os.environ.get('TORCS_KILL_ALL', '1') != '0':
        os.system('pkill -9 -f torcs')
        # The pkill above also kills the xvfb-run wrapper (its command line
        # contains "torcs") before it can clean up, orphaning the Xvfb
        # server: without this line every relaunch would accumulate a dead Xvfb process.
        os.system('pkill -9 -f xvfb-run; pkill -9 Xvfb')

class TorcsEnv:
    # Variables used to evaluate the early termination in case the car stalls
    terminal_judge_start = 500  # 10 seconds after which we start evaluating whether the car is stalled
    termination_limit_progress = 5  # After 10 seconds, if the forward speed/progress drops below about 5 m/s, we consider the car stalled.
    off_track_limit = 1.25  # Beyond this value the lap is considered invalid.
    off_track_penalty_base = 5.0  # Minimum terminal penalty when off_track_limit is exceeded.
    off_track_penalty_extra = 5.0  # Additional progressive penalty, saturated within +1.0 trackPos.
    incomplete_lap_step_penalty = 5.0  # Local penalty for terminal failures not related to lap time.

    default_speed = 50 # Reference speed to normalize speedX/Y/Z. It is not a maximum speed, but a typical on-track speed value (50 m/s = 180 km/h) used to scale the observations into a more manageable range for training the agents.
    # changing this value rescales all the speed observations (speedX/Y/Z) and also the reward computation (progress), so it must be chosen consistently with the typical speeds one wants to reach on track.
    # A value that is too low could lead to normalized observations that are too large,
    # while a value that is too high could lead to observations that are too small.
    # 50 m/s is a good choice because it represents a high but reachable speed in many race situations.


    initial_reset = True    # Flag indicating whether it is the first reset (startup) of the environment.

    # by default early termination is active, i.e. the episode ends at the first contact with a wall/opponents or a stall.
    def __init__(self, early_termination=True):
        import shutil   # a library useful for path processing in the OS

        # Check that xvfb-run is installed, otherwise raise an environment error.
        if shutil.which('xvfb-run') is None:
            raise EnvironmentError("xvfb-run non trovato. Installa il pacchetto 'xvfb' per l'esecuzione headless isolata di TORCS.")


        self.early_termination = early_termination
        self.initial_run = True


        _kill_torcs()
        time.sleep(1.5) #waits for the operating system to release the UDP port used by TORCS, otherwise the next startup fails.

        # String we use to launch torcs, in no damage and no fuel mode
        torcs_cmd = 'torcs -nofuel -nodamage'


        # If the SHOW_GUI variable is set to 1, start normally. Otherwise use Xvfb.
        # setsid detaches TORCS from the terminal: the user's Ctrl+C (SIGINT to the whole foreground
        # process group) must not kill the simulator mid-episode. Cleanup remains
        # delegated to _kill_torcs (pkill by name, independent of the process group).
        if os.environ.get('SHOW_GUI', '0') == '1':
            os.system(f'setsid sh -c "(sleep 1.5 && sh {_AUTOSTART_SH}) & exec {torcs_cmd} > /dev/null 2>&1" &')
        else:
            xvfb_cmd = f'xvfb-run -a -s "-screen 0 640x480x24" sh -c "(sleep 1.5 && sh {_AUTOSTART_SH}) & exec {torcs_cmd} > /dev/null 2>&1"'
            os.system(f"setsid {xvfb_cmd} &")

        time.sleep(3.0)  # Waits for Xvfb/TORCS and the autostart macro.
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0, 1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 6.0], dtype=np.float32),
            dtype=np.float32,
        )

        #Dictionary with all the information TORCS sends; each piece of information has its own dimensions (visible from shape) and value range
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

    # The step function takes the agent's action (u), converts it into the format required by TORCS, sends the action to the TORCS server, receives the new telemetry, computes the reward and determines whether the episode has ended.
    def step(self, u):
        # Converts the agent's action into the format required by the TORCS server.
        client = self.client

        this_action = self.agent_to_torcs(u)

        # Apply Action
        action_torcs = client.R.d

        action_torcs['steer'] = this_action['steer']
        action_torcs['accel'] = this_action['accel']
        action_torcs['brake'] = this_action['brake']
        action_torcs['gear'] = this_action['gear']

        # Pre-step snapshot: used to detect whether TORCS updates lastLapTime at the finish line.
        prev_last_lap_time = float(np.array(client.S.d.get('lastLapTime', 0.0)).flat[0])

        # Physics step: sends the action and reads the new telemetry from the server.
        client.respond_to_server()
        client.get_servers_input()

        obs = client.S.d

        # Converts the raw TORCS telemetry into the normalized dictionary used by the scripts.
        self.observation = self.make_observaton(obs)

        # ─── Reward Reshaping shared by the TD3+BC ───────────────────────
        sp_norm = obs['speedX'] / self.default_speed  # Range ~[0, 6]
        progress = sp_norm * np.cos(obs['angle'])

        # Initialize last_steer if it does not exist, i.e. set the steering straight at the first step
        if not hasattr(self, 'last_steer'):
            self.last_steer = 0.0

        # computes the steering change relative to the previous step,
        # used to penalize abrupt direction changes (zigzag) and
        # encourage a smoother driving style. The penalty is proportional
        # to the absolute steering change, with a coefficient of 0.05 (used below) that
        # balances the importance of this term in the overall reward.
        steer_change = this_action['steer'] - self.last_steer
        self.last_steer = this_action['steer']

        # Computation of the track-position penalty:
        # - No penalty if |trackPos| < 1.0 (car within the track edges)
        # - Increasing penalty (quadratic ramp) >= 1.0 and <= 1.25 (punishes the model if it goes too far out)
        # - Beyond 1.25 the lap is invalid and below it a graduated terminal penalty is added.
        tp = abs(float(obs['trackPos']))
        pos_penalty = -2.0 * (max(0.0, tp - 1.0) ** 2)


        # Computation of the overall reward for each step:
        # - The main reward is the forward progress
        # - To this is added the off-track position penalty (pos_penalty)
        # - And a penalty for abrupt steering changes (zigzag) is subtracted
        # The choice to put the reward in this file was made for convenience
        # We could have put it in TD3+BC but this way it is easier to access the variables needed to compute it
        # Such as obs, last_steer, etc...
        # Computes the reward as a linear combination of the components
        reward = (progress * 1.5) + pos_penalty - (0.05 * abs(steer_change))

        # Extracts the lap time of the last completed lap
        last_lap_time = float(np.array(obs.get('lastLapTime', 0.0)).flat[0])

        # The lap is completed if laptime is > 0 and the time changed from the previous step and
        # we are past the 10 seconds of evaluation for early termination (terminal_judge_start)
        lap_completed = (
            last_lap_time > 0.0
            and abs(last_lap_time - prev_last_lap_time) > 0.01
            and self.time_step > self.terminal_judge_start
        )

        # Info dictionary containing diagnostic information about the episode,
        # such as whether there was a crash, whether the car went off track,
        # whether the lap was completed, the lap time, and the termination reason
        # (if applicable). This information is useful for analysis and debugging of the
        # agents' training.
        info = {
            'crash': False,
            'off_track': False,
            'lap_completed': lap_completed,
            'lap_time': last_lap_time if lap_completed else 0.0,
            'termination_reason': 'SUCCESS' if lap_completed else None,
        }


        # Variable indicating whether the episode must be terminated.
        episode_terminate = False


        # If early termination is active, we evaluate the early-termination conditions
        if self.early_termination:
            # INVALID lap: beyond |trackPos| > 1.25 (corner cut / wall). It is the same limit
            # used in data collection (curbs allowed up to 1.25, beyond = invalid).
            if tp > self.off_track_limit:
                excess = min(tp - self.off_track_limit, 1.0) #computes how far off track it is, capped at 1 because with value 1 you get the maximum off-track penalty
                reward -= self.off_track_penalty_base + (self.off_track_penalty_extra * excess) # Updating the reward by counting the penalty

                # Update the info flags to indicate there was a crash due to going off track
                info['crash'] = True
                info['off_track'] = True
                info['lap_completed'] = False
                info['lap_time'] = 0.0
                info['termination_reason'] = 'OFF_TRACK'
                episode_terminate = True
                client.R.d['meta'] = True # Flag to signal that the episode must terminate


            # Evaluate whether the car is stalled:
            # - if after 10 seconds (terminal_judge_start) it has not completed the lap
            # - If the episode has not terminated
            # - if the lap is not completed
            if not episode_terminate and not lap_completed and self.terminal_judge_start < self.time_step:
                # If the instantaneous forward progress is insufficient, we consider the car stalled and terminate the episode.
                if progress < (self.termination_limit_progress / 50.0):
                    reward -= self.incomplete_lap_step_penalty  #Updates the reward by counting the stall penalty

                    # Update the info flags to indicate there was a crash due to a stall
                    info['crash'] = True
                    info['termination_reason'] = 'STALL'
                    episode_terminate = True
                    client.R.d['meta'] = True

            # Evaluate whether the car has spun:
            # - if the episode has not terminated
            # - if the lap is not completed
            # - if the cosine of the angle between the car and the track axis is negative
            if not episode_terminate and not lap_completed and np.cos(obs['angle']) < 0:
                reward -= self.incomplete_lap_step_penalty  #Updates the reward by counting the spin penalty (the same as the stall one)

                # Update the info flags
                info['crash'] = True
                info['termination_reason'] = 'SPIN'
                episode_terminate = True
                client.R.d['meta'] = True

            # Evaluate whether the lap is completed: if the lap is completed but the episode has not terminated yet, then terminate the episode successfully.
            # The lap-end bonus reward is applied in TD3+BC; here we only apply the episode termination.
            if not episode_terminate and lap_completed:

                # Flag update
                episode_terminate = True
                client.R.d['meta'] = True

        # If the episode has terminated, change the initial run flag to False
        # and respond to the server by sending the R dictionary with meta=True, which is the signal for
        # TORCS to terminate the episode and prepare for the reset.
        if client.R.d['meta'] is True:
            self.initial_run = False
            client.respond_to_server()

        self.time_step += 1 # Increments the step counter

        return self.get_obs(), reward, client.R.d['meta'] or client.so is None, info # returns the state, the reward, whether the episode has ended and additional information.

    # The reset function restarts the episode. If the initial_reset flag is True, it restarts TORCS and clears the state variables.
    # If the initial_reset flag is False, it sets the R.d['meta'] flag to True to signal TORCS to terminate the current episode and prepare for the reset.
    def reset(self, relaunch=False):
        self.time_step = 0

        # If initial_reset is False, set the R.d['meta'] flag to True to signal TORCS to terminate the current episode and prepare for the reset.
        if self.initial_reset is not True:
            # If the client socket is dead (TORCS server not responding or process terminated),
            # a soft reset would block forever in setup_connection waiting for
            # a server that no longer exists: a full relaunch is forced.
            if getattr(self.client, 'so', None) is None:
                relaunch = True
            self.client.R.d['meta'] = True
            self.client.respond_to_server()

            # If the relaunch flag is True, restart TORCS and clear the state variables.
            if relaunch is True:
                # We explicitly close the open UDP socket before the relaunch.
                if hasattr(self, 'client') and self.client is not None:
                    try:
                        self.client.so.close()
                    except Exception:
                        pass
                self.reset_torcs()
                print("### TORCS is RELAUNCHED ###")

        # The connection can fail if TORCS stays stuck at the menu (the autostart xte
        # macro can lose timing at startup): in that case a full simulator relaunch is
        # forced and retried. Even an "identified" connection can then fail to produce
        # the first telemetry packet: without this validation we ended up with a
        # KeyError on raw_obs['focus'].
        connect_attempts = 3
        for attempt in range(1, connect_attempts + 1):
            try:
                self.client = snakeoil3.Client(p=3001, vision=False)  # Standard SCR UDP socket.
                self.client.MAX_STEPS = np.inf
                self.client.get_servers_input()

                obs = self.client.S.d
                missing = [key for key in _REQUIRED_OBS_KEYS if key not in obs]
                if getattr(self.client, 'so', None) is None or missing:
                    if attempt == connect_attempts:
                        raise snakeoil3.ServerTimeoutError(
                            "Il server TORCS si è connesso ma non ha inviato telemetria valida "
                            f"(chiavi mancanti: {', '.join(missing) if missing else 'socket chiuso'})."
                        )
                    print(
                        "### Telemetria iniziale TORCS incompleta "
                        f"(tentativo {attempt}/{connect_attempts}); relaunch completo ###"
                    )
                    self.reset_torcs()
                    continue

                break
            except snakeoil3.ServerTimeoutError as e:
                if e.aborted or attempt == connect_attempts:
                    raise
                print(f"### Server TORCS non raggiungibile (tentativo {attempt}/{connect_attempts}): relaunch completo ###")
                self.reset_torcs()

        obs = self.client.S.d
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
        time.sleep(1.5)  # Ensures the operating system releases the UDP port

        torcs_cmd = 'torcs -nofuel -nodamage'

        # If the SHOW_GUI variable is set to 1, start normally. Otherwise use Xvfb.
        # setsid: see __init__ — TORCS must not receive the Ctrl+C intended for the training.
        if os.environ.get('SHOW_GUI', '0') == '1':
            os.system(f'setsid sh -c "(sleep 1.5 && sh {_AUTOSTART_SH}) & exec {torcs_cmd} > /dev/null 2>&1" &')
        else:
            xvfb_cmd = f'xvfb-run -a -s "-screen 0 640x480x24" sh -c "(sleep 1.5 && sh {_AUTOSTART_SH}) & exec {torcs_cmd} > /dev/null 2>&1"'
            os.system(f"setsid {xvfb_cmd} &")

        time.sleep(3.0)  # Combined time for startup and macro

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
