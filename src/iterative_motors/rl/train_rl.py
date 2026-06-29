"""
TD3+BC Fine-Tuning module — Twin Delayed DDPG (TD3) integrated with Behavioral Cloning for TORCS.

The TD3 algorithm is a more robust variant of the well-known DDPG (Deep Deterministic Policy Gradient), introduced
in the paper "Addressing Function Approximation Error in Actor-Critic Methods" (Fujimoto et al., 2018).

This algorithm has as its main features:
    - Twin Critic: Compared to DDPG it introduces two independent networks, Q1 and Q2; the Bellman target is computed as the minimum of the two, reducing the probability of overestimation.
    - Delayed Policy Update: The Critic and Actor policy updates do not happen simultaneously; the actor is updated less frequently than the critic (policy_frequency = 2 steps)
    - Target Policy Smoothing: Adds clipped noise to the target action computed for the Critic's target networks
      to prevent overfitting on narrow peaks of the Q-function.

This module implements the TD3+BC algorithm (Fujimoto & Gu, 2021), an offline-to-online architecture
i.e. the model is first pre-trained on collected data without interacting with the environment, then it is inserted into the TORCS simulator
a hybrid architecture designed to train an autonomous-driving agent on the TORCS simulator. The main purpose
is to inherit the initial knowledge learned by imitation from a human driver (Behavioral Cloning - BC)
and refine it via Reinforcement Learning (RL) without incurring
policy degradation or gradient collapse caused by erroneous value estimates (Q-values) at startup.

SYSTEM ARCHITECTURE AND LOGIC

1. Model structure:
   - Actor (Driving Agent): Inherits the backbone and the continuous head of the pre-trained BC model
     (Warm-Start). During RL training, the entire Actor network is updated to optimize
     the trajectories. The generated continuous commands control steering, throttle and brake, while
     the gear change is governed by the external module 'gearing.py'.
   - Critic (Twin Q-Networks): Two identical and independent networks, trained from scratch. They
     receive as input the 87D environment state and the chosen 3D action, estimating the expected Q value.

2. TD3+BC Actor Loss function (these formulas were taken from the Fujimoto & Gu (2021) paper):
   The Actor loss is defined as:
       Loss = -lambda * Q(s, pi(s)) + BC_Penalty
   Where:
     - Q(s, pi(s)) is the Q value estimated by the Critic Q1.
     - BC_Penalty is the mean squared error (MSE) between the predicted action and the human expert's action.
     - lambda is a balancing coefficient computed dynamically at each batch as:
       lambda = alpha / mean(|Q(s, pi(s))|), with alpha = 2.5. This ensures that the reinforcement (RL)
       component keeps the same scale as the imitation (BC) term, making the gradient stable
       against strong variations in the magnitude of the Q-values.

3. Strict BC Penalty masking (Expert Masking):
   The BC Penalty is computed and applied exclusively on the share of batch samples that come
   from the expert dataset (marked with expert_mask = 1.0). This allows the model not to be
   penalized if it tries trajectories different from those of the human driver (expert) during online driving.

4. Three-Way Hybrid Sampling (Three-Way Replay Buffer):
   The training batches are composed by combining three distinct experience sources to maximize
   both fidelity to the expert and the ability to recover from errors:
     - 25% Human Dataset (Expert): Permanent buffer, prevents the detachment from the BC and the degeneration into pure RL.
     - 15% Best Runs (Elite): Replays transitions from the best autonomous attempts, encouraging Self-Imitation.
     - 60% Online Exploration: Allows the agent to learn to handle dirty, off-trajectory states.

    These percentages were chosen empirically.

5. Reward function and penalty handling:
   The reward code is contained inside the gym_torcs.py file; in this file it is invoked and any maluses or bonuses are added.

   At each simulation step, the environment computes a multi-objective reward to guide the agent's optimization:
     - Longitudinal Progress (Speed): Computed as `(progress * 1.5)`. Rewards the projection of the speed
       along the central axis of the track, penalizing transverse drifts. The scale factor `1.5` encourages
       the search for maximum speeds on the straights.
     - Road Position Penalty: `-2.0 * (max(0.0, |trackPos| - 1.0) ^ 2)`. A quadratic penalty applied
       exclusively when the car leaves the asphalt edges (`|trackPos| > 1.0`). If the car is inside
       the road lines, this term is zero (`0.0`), thus acting as a soft virtual barrier.
     - Direction Change Penalty (Anti-oscillation): `-0.05 * |steer_change|`. Penalizes abrupt changes
       in steering between two adjacent time instants. Avoids the zigzag behaviour on the straights
       and forces the Actor to learn smooth driving trajectories.
     - Off-Track Crash Penalty: If `|trackPos| > 1.25` (corner cut or barrier hit), the episode
       is interrupted prematurely and a cutoff penalty `-base_penalty - (extra_penalty * excess)` is applied.
     - Stall and Spin Penalty: If the car remains still for more than 10 seconds or spins (cosine of the angle
       relative to the track negative, `cos(angle) < 0`), the episode ends with a fixed collision penalty.
     - Lap-End Bonus (added in TD3+BC): `+50.0` if the finish line is crossed regularly and successfully.
       The time-proportional bonus (`+10.0` for every second below 80s) and the personal-record bonus
       are active ONLY in time-attack: in stabilization the completion is rewarded flatly, so a
       slow but clean lap is worth as much as a fast but risky one and the policy first learns to finish the lap.
     - Incomplete-Lap Malus (added in TD3+BC): `-25.0` if the episode ends prematurely due to a slide or crash,
       discouraging reckless driving in favour of completing the circuit.
     - Corridor Penalty (STABILIZATION): per-step quadratic penalty that kicks in already at `|trackPos| > 0.80`,
       i.e. BEFORE leaving the driving surface, to teach a safety margin from the edge. Full in
       stabilization (reliable complete laps), reduced in time-attack (the full track width is needed). See
       `reward.margin_penalty`.
     - Per-Sector Telemetry Reward (TIME-ATTACK): the track is divided into sectors; at the close of each one the
       agent is rewarded (or penalized) based on how much it beats its own best sector time. A dense signal that indicates
       WHERE to gain time, with a log of the "theoretical ideal lap" (sum of the best splits). See `sector_timer`.
"""

import os
import sys
import argparse
import random
import re
import signal
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from datetime import datetime

# ── Iterative Motors: package (ambiente, utility, reti condivise) ─────────
_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
from iterative_motors.env.gym_torcs import TorcsEnv
from iterative_motors.env import snakeoil3_gym as snakeoil3
from iterative_motors.env.gearing import compute_gear  # algorithmic gear shifting
from iterative_motors.common.constants import TRACK_LENGTH_M, LAPS_AUTO_DIR, SESSION_LOGS_DIR
from iterative_motors.common.checkpoint import (
    safe_save, safe_write_text, safe_read_float, safe_save_npz,
    _fsync_file, _fsync_dir, _backup_paths, _rotate_backup, _checkpoint_candidates,
)
from iterative_motors.common.state import apply_state_norm, flatten_state_norm as flatten_state
from iterative_motors.models.networks import Actor, Critic
from iterative_motors.data.replay_buffer import ReplayBuffer
from iterative_motors.data.lap_recorder import LapRecorder
from iterative_motors.rl.reward import (
    LAP_SUCCESS_BONUS, INCOMPLETE_LAP_PENALTY, EVAL_DISTANCE_SANITY_LIMIT,
    LAP_TIME_BONUS_REF_S, LAP_TIME_BONUS_PER_S, EVAL_SCORE_T_REF_S, EVAL_SCORE_SANITY_LIMIT,
    _is_plausible_eval_dist, _is_plausible_eval_score, _eval_score, _track_progress_from_start,
    personal_best_bonus, TIME_ATTACK_BC_ALPHA, TIME_ATTACK_NOISE_FLOOR, TIME_ATTACK_ENTRY_S,
    margin_penalty, MARGIN_PENALTY_COEF,
    TA_SECTORS_DEFAULT, TA_SECTOR_REWARD_K, TA_SECTOR_REWARD_CAP,
)
from iterative_motors.rl.agent import TD3BCAgent
from iterative_motors.rl.sector_timer import SectorTimer


# Full TORCS relaunch (kill + restart + autostart macro) only every N episodes:
# it costs ~6 real seconds versus the soft reset (in-place meta-restart) that is almost instantaneous.
# The periodic relaunch keeps the simulator state clean (memory leak,
# dirty sockets) without paying its cost every episode. The offset 2 keeps it away
# from the evals (episodes ≡ 4 mod 5), which already do full relaunches on their own.
# If the connection with the server drops, gym_torcs forces the relaunch by itself anyway.
RELAUNCH_EVERY_EPISODES = 5
RELAUNCH_EPISODE_OFFSET = 2

# Exploration-noise annealing: at the start of training wide exploration is needed (0.10),
# but at steady state such a large disturbance on the steering causes systematic crashes in fast corners;
# to shave the last tenths micro-variations of the trajectory are needed (0.04).
EXPL_NOISE_START = 0.10
EXPL_NOISE_END = 0.04
EXPL_NOISE_ANNEAL_EPISODES = 1500




# ──────────────────────────────────────────────────────────────────────
#  Determinism
# ──────────────────────────────────────────────────────────────────────
def set_seed(seed=42):
    """
    Sets the random-generation seeds to guarantee experiment reproducibility.

    Configures the random generators for:
      - Python standard library (random)
      - NumPy (np.random)
      - PyTorch CPU and CUDA (torch.manual_seed, torch.cuda.manual_seed)
      - cuDNN backend configuration in deterministic mode
      - PYTHONHASHSEED environment variable
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)




def load_recent_evals_from_log(log_path, max_len=8):
    """
    Parses the training log file to extract the last recorded evaluation scores
    ('Score' field in the current format; falls back to the 'Dist' distance for historical lines).

    This serves to populate the memory window for the Auto-Refinement at agent startup (Resume),
    preventing the plateau state from being lost or reset when the training process is restarted.

    Arguments:
        log_path: Path of the td3_training.log text file.
        max_len: Maximum length of the temporal window (default: 8 evaluations).

    Returns:
        A list containing the last max_len valid distances extracted.
    """
    evals = []
    if os.path.exists(log_path):
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    # Filters only the lines containing the deterministic-evaluation tag [EVAL]
                    if '[EVAL]' in line and 'Result' in line:
                        try:
                            # Prefers the Score field (new format: distance or time-equivalent);
                            # historical lines without Score fall back to the distance only.
                            match = re.search(r'\bScore\s+([0-9]+(?:\.[0-9]+)?)m', line)
                            if match:
                                eval_score = float(match.group(1))
                                if _is_plausible_eval_score(eval_score):
                                    evals.append(eval_score)
                                continue
                            match = re.search(r'\b(?:Dist|Distanza)\s+([0-9]+(?:\.[0-9]+)?)m', line)
                            if match:
                                eval_dist = float(match.group(1))
                                # Validates the value with the single-lap physical plausibility filter
                                if _is_plausible_eval_dist(eval_dist):
                                    evals.append(eval_dist)
                        except Exception:
                            pass
        except Exception as e:
            print(f"Impossibile leggere gli eval recenti dal log: {e}")
    return evals[-max_len:]

def train():
    """
    Main function that coordinates the online training loop of the TD3+BC agent on TORCS.

    Execution flow:
      1. Parsing of the command-line arguments (rollback configuration, critic pretrain,
         actor freezing and refinement settings).
      2. Initialization of the TorcsEnv simulation environment and the three replay buffers
         (Online, Elite, Expert).
      3. Restore (Resume) of the agent state from a checkpoint via `load_checkpoint`.
      4. Permanent in-memory loading of the human expert transition dataset.
      5. Main episode simulation loop:
         - Handling of the temporal frame stacking of the sensor states.
         - Interaction with TORCS to obtain the next state, gear computation (gearing.py) and reward.
         - Update of the agent weights at every simulation step.
         - Differentiated storage and filling of the replay buffers.
      6. Periodic deterministic evaluation of the agent (every 5 episodes) with tracking
         of the absolute records and saving of the optimal deterministic action weights for the submission.
      7. Finite state machine for the Auto-Refinement (activation/deactivation of the refinement,
         BC weight reduction, rollback handling and plateau timeout prevention).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--bc_weights', type=str, default='train_set/checkpoints/bc_policy.pth')
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--max_steps', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--rollback', action='store_true', help="Forza il rollback dell'Actor alla migliore policy deterministica e lo congela temporaneamente")
    parser.add_argument('--actor-freeze-episodes', '--actor_freeze_episodes', dest='actor_freeze_episodes',
                        type=int, default=30,
                        help="Numero di episodi di congelamento Actor dopo --rollback (default: 30; aumenta per dare piu' tempo al Critic)")
    parser.add_argument('--no-auto-refine', '--no_auto_refine', dest='no_auto_refine',
                        action='store_true',
                        help="Disattiva solo la refinement automatica da plateau; --refine manuale resta disponibile")
    parser.add_argument('--refine', action='store_true', help="Avvia subito la refinement: aggiornamento del Critic disattivato, loss Critic solo diagnostica, peso Behavioral Cloning ridotto")
    parser.add_argument('--pretrain_critic', action='store_true', help="Esegue il pre-training offline del Critic per 50k passi in caso di emergenza (da usare con --rollback)")
    parser.add_argument('--expert_max_lap_time', type=float, default=71.0,
                        help="Carica nel buffer expert solo i file con lap_time <= soglia (secondi); "
                             "<= 0 disattiva il filtro e carica tutti i giri (default: 71.0)")
    parser.add_argument('--bc_alpha', type=float, default=2.5,
                        help="Coefficiente alpha del TD3+BC: piu' alto = piu' peso al RL rispetto alla BC "
                             "(default: 2.5 come nel paper; 3.5-5.0 per spingere oltre l'esperto)")
    parser.add_argument('--trust_region', type=float, default=0.0,
                        help="Peso FISSO della trust region verso le azioni del buffer sui campioni "
                             "non-expert (0 = off; ~0.3 ancora l'Actor al supporto dati e cura il "
                             "collasso della policy quando l'Actor torna attivo)")
    parser.add_argument('--reseed_elite_max_lap_time', type=float, default=0.0,
                        help="Semina UNA TANTUM il buffer elite con i giri auto-registrati "
                             "(train_set/laps_auto) con lap_time <= soglia (0 = off). Rompe la "
                             "starvation dell'elite dando alla self-imitation giri completi da imitare; "
                             "i semi lenti invecchiano ed escono man mano che la policy cattura giri "
                             "più veloci. Passalo SOLO al primo lancio di semina (poi è nel checkpoint).")
    parser.add_argument('--capture_eval_elite', action='store_true',
                        help="Cattura nell'elite i giri COMPLETI dell'eval deterministica (la linea "
                             "pulita/veloce). Default OFF: in fase di stabilizzazione self-imitare la "
                             "linea-rasoio sovra-affila la policy e destabilizza l'esplorazione; tienila "
                             "spenta finché l'esplorazione non è stabile, riattivala nella limatura.")
    args = parser.parse_args()
    if args.actor_freeze_episodes < 0:
        parser.error("--actor-freeze-episodes deve essere >= 0")
    if args.trust_region < 0:
        parser.error("--trust_region deve essere >= 0")
    actor_freeze_episodes = args.actor_freeze_episodes
    auto_refine_enabled = not args.no_auto_refine

    set_seed(args.seed)

    env = TorcsEnv(early_termination=True)

    # ONLINE buffer (FIFO) for the agent's experience. 2M transitions: doubled horizon to
    # retain more recent history before eviction (user request: "do not forget the old").
    # The batch proportions stay 25/15/60: the capacity only decides how much horizon the online keeps.
    memory = ReplayBuffer(2000000)
    # ELITE buffer (self-imitation): capacity increased 20k→200k to host a VARIED collection of
    # good laps — initial seeding (--reseed_elite_max_lap_time) + captures of the complete laps (eval and
    # online) — without the FIFO evicting too soon. ~110 whole laps of breathing room.
    elite_memory = ReplayBuffer(200000)

    # SEPARATE and PERMANENT EXPERT buffer (human data): capacity > dataset so it is
    # NEVER emptied by the FIFO. Fixes the loss of the BC anchor and makes it present in every batch.
    expert_memory = ReplayBuffer(400000)

    # Agent initialization
    agent = TD3BCAgent()
    agent.bc_alpha = args.bc_alpha
    agent.trust_region_weight = args.trust_region
    if args.trust_region > 0.0:
        print(f"Trust region attiva: peso {args.trust_region} verso le azioni del buffer sui campioni "
              f"non-expert (ancora l'Actor al supporto dati, stabilizza la policy deterministica).")

    # Nota: i pesi BC pre-addestrati vengono caricati solo al fresh-start (blocco successivo); in caso di resume, sono ripristinati dal checkpoint.
    checkpoint_path = 'train_set/checkpoints/td3_checkpoint.pth'
    start_episode, global_step, best_lap_time, best_eval_dist, best_distance = agent.load_checkpoint(checkpoint_path, memory, elite_memory)

    # Loads the expert data into the permanent buffer both at startup and at restart, guaranteeing the BC anchor.
    # The lap_time filter keeps only the driver's best laps: the anchor must point at their best, not their average.
    expert_lap_filter = args.expert_max_lap_time if args.expert_max_lap_time > 0 else None
    expert_memory.load_expert_data('train_set/laps', max_samples=350000, max_lap_time=expert_lap_filter)

    # Seeding of the elite from the self-recorded laps (train_set/laps_auto) for the self-imitation. Two triggers:
    #  - EXPLICIT: --reseed_elite_max_lap_time S (seeds with the laps <= S);
    #  - AUTOMATIC (safety net): if after loading the elite is STARVED (< floor) — because
    #    starved or because a previous save was almost empty — it seeds by itself with a
    #    default threshold. This way the self-imitation NEVER runs dry between restarts, and the captures of
    #    the fast laps (eval/online) over time age out and evict in FIFO the slower seeds.
    ELITE_STARVATION_FLOOR = 8000
    DEFAULT_RESEED_LAP_TIME = 74.5
    elite_loaded = len(elite_memory)
    print(f"  [ELITE] {elite_loaded} transizioni caricate dal checkpoint.")
    seed_cut = args.reseed_elite_max_lap_time
    if seed_cut <= 0.0 and elite_loaded < ELITE_STARVATION_FLOOR:
        seed_cut = DEFAULT_RESEED_LAP_TIME
        print(f"  [ELITE] affamato (<{ELITE_STARVATION_FLOOR}): auto-semina di sicurezza dai giri auto <= {seed_cut}s.")
    if seed_cut > 0.0:
        n_before = len(elite_memory)
        elite_memory.load_expert_data(LAPS_AUTO_DIR, max_samples=150000, max_lap_time=seed_cut)
        print(f"  [ELITE SEED] {n_before} -> {len(elite_memory)} transizioni (giri auto <= {seed_cut:.1f}s).")

    agent.actor_frozen = False

    # Auto-Refinement configuration (Beeson & Montana, 2022).
    # Detects stall (plateau) situations of the results and optimizes the policy
    # by disabling the Critic update and reducing the BC constraint.
    # It can be disabled via the --no-auto-refine flag.
    REFINE_BC_WEIGHT = 0.3                 # Reduced weight for the BC penalty during refinement.
    REFINE_MAX_ATTEMPTS = 3                # Maximum number of refinement attempts before forcing standard training.
    REFINE_COLLAPSE_FRAC = 0.6             # Collapse threshold (60% of the recent value) to trigger a preventive rollback.
    REFINE_WINDOW = 8                      # Number of historical evaluations considered for the moving window.
    REFINE_PLATEAU_EVALS = 4               # Number of consecutive evaluations without increase needed to declare a plateau.
    REFINE_MIN_EP = 200                    # Minimum episode required to be able to activate the auto-refinement.
    REFINE_IMPROVE_FRAC = 1.02             # Minimum increase (+2%) to consider an evaluation a significant improvement.
    REFINE_BREAKOUT_FRAC = 1.10            # Plateau-breakout coefficient (+10%) to consider an attempt a "breakout".
    REFINE_NEW_PLATEAU_FRAC = 1.10         # Threshold (+10%) to establish a new plateau and reset the attempts.
    REFINE_GOOD_EVALS_TO_CONSOLIDATE = 3  # Consecutive positive evaluations required to consolidate and reactivate the Critic.
    REFINE_NEAR_BEST_MARGIN = 5.0          # Closeness margin to the historical record (in meters) to induce immediate consolidation.
    BEST_DIST_EPS = 1.0                    # Metric tolerance to ignore minor oscillations in the record-distance logs.

    # Thresholds for the "completed lap" regime: when the reference exceeds TRACK_LENGTH_M the score
    # is in time-equivalent (1% of score ≈ 0.7s of lap), so the percentages of the distance regime
    # would be unreachable: +10% would mean asking for 7 seconds of lap improvement.
    REFINE_LAP_IMPROVE_FRAC = 1.004        # +0.4% of score ≈ 0.3s of lap: significant improvement.
    REFINE_LAP_BREAKOUT_FRAC = 1.01        # +1% of score ≈ 0.7s of lap: breakout from the plateau.
    REFINE_LAP_NEW_PLATEAU_FRAC = 1.01     # +1% of score: new plateau, reset of the attempts.

    def _refine_frac(reference, dist_frac, lap_frac):
        """
        Selects the correct percentage threshold based on the reference regime:
        below TRACK_LENGTH_M the score is a distance (incomplete laps), above it is in
        time-equivalent (completed laps) and requires much finer thresholds.
        """
        return lap_frac if reference > TRACK_LENGTH_M else dist_frac
    agent.refine_mode = False
    agent.refine_bc_weight = 1.0
    recent_eval_window = deque(maxlen=REFINE_WINDOW)  # Moving queue for the last evaluations.
    time_attack = (os.environ.get('IM_TIME_ATTACK', '0') == '1')
    phase_log_file = os.path.join(
        SESSION_LOGS_DIR,
        'time-attack.log' if time_attack else 'td3_training.log',
    )

    # We populate the window by reading the recent data directly from the log
    initial_evals = load_recent_evals_from_log(phase_log_file, REFINE_WINDOW)
    for ev in initial_evals:
        recent_eval_window.append(ev)
    if len(recent_eval_window) > 0:
        print(f"Caricati {len(recent_eval_window)} eval recenti dal log: {list(recent_eval_window)}")
    if auto_refine_enabled:
        print("Auto-refinement automatica: attiva di default.")
    else:
        print("Auto-refinement automatica: disattivata da --no-auto-refine. --refine manuale resta disponibile.")

    refine_best_mean = 0.0                 # Best moving average of the recorded evaluation window (plateau signal).
    if len(recent_eval_window) >= REFINE_WINDOW:
        refine_best_mean = sum(recent_eval_window) / len(recent_eval_window)

    refine_evals_no_improve = 0            # Number of consecutive evaluations without significant increase.
    refine_attempts = 0                    # Total number of refinement attempts started.
    refine_attempt_plateau_ref = 0.0       # Reference plateau distance for the current attempt.
    refine_attempt_limit_logged = False    # Logging state for the maximum attempts limit.
    refine_plateau_level = 0.0             # Stored plateau level (0.0 if not yet established).
    refine_collapse_count = 0              # Counter of the detected performance collapses.
    refine_evals_count = 0                 # Number of evaluations performed during the refinement phase.
    refine_good_eval_count = 0             # Number of consecutive positive evaluations post-breakout.
    refine_breakout_logged = False         # Logging state for the first plateau breakout.

    if getattr(args, 'refine', False):
        # Immediate manual activation of the refinement (e.g. if a persistent stall is detected).
        agent.refine_mode = True
        agent.refine_bc_weight = REFINE_BC_WEIGHT

        # Estimate of the initial plateau level:
        # If the recent window contains enough data and we are not in a rollback phase,
        # the median is computed. Otherwise, the best historical deterministic record is used.
        if len(recent_eval_window) >= 4 and not getattr(args, 'rollback', False):
            refine_plateau_level = float(np.median(list(recent_eval_window)))
        else:
            refine_plateau_level = best_eval_dist
            det_best_dist_txt = 'train_set/checkpoints/td3_det_best_dist.txt'
            if refine_plateau_level <= 0.0:
                refine_plateau_level = safe_read_float(det_best_dist_txt, 0.0)

        if refine_plateau_level > 0.0:
            refine_attempt_plateau_ref = refine_plateau_level
            print(f"--refine attivo: refinement attiva da subito "
                  f"(aggiornamento Critic disattivato, peso Behavioral Cloning={REFINE_BC_WEIGHT}). "
                  f"Riferimento plateau={refine_plateau_level:.0f}m. Loss Critic solo diagnostica.")
        else:
            print(f"--refine attivo: refinement attiva da subito "
                  f"(aggiornamento Critic disattivato, peso Behavioral Cloning={REFINE_BC_WEIGHT}). "
                  f"Riferimento plateau impostato dopo i primi eval (mediana recente). Loss Critic solo diagnostica.")

    batch_size = 256

    # Warm-Start: initialization of the Actor with the BC weights if it is a fresh start (episode 0).
    if start_episode == 0:
        agent.actor.load_bc_weights(args.bc_weights)
        agent.actor_target.load_state_dict(agent.actor.state_dict())
    else:
        # Rollback procedure for the emergency restore of the Actor weights.
        # It tries to restore the model starting from the best available deterministic policy,
        # going down in priority order to the exploratory policies if the first ones are absent.
        if args.rollback:
            rollback_candidates = [
                'train_set/checkpoints/td3_det_best_lap.pth',      # 1) Fastest valid deterministic lap.
                'train_set/checkpoints/td3_det_best_dist.pth',     # 2) Best absolute deterministic distance.
                'train_set/checkpoints/td3_det_best_dist_run.pth', # 3) Best deterministic distance of the current run.
                'train_set/checkpoints/td3_expl_best_lap.pth',     # 4) Best exploratory lap (with noise active).
                'train_set/checkpoints/td3_expl_best_dist.pth',    # 5) Best exploratory distance (with noise active).
            ]
            best_path = next((p for p in rollback_candidates if os.path.exists(p)), None)
            if best_path:
                print(f"[EMERGENZA] Rollback Actor: caricamento della migliore policy deterministica da {best_path}")
                agent.actor.load_actor_weights(best_path, agent.device)
                agent.actor_target.load_state_dict(agent.actor.state_dict())
                import torch.optim as optim
                agent.actor_optimizer = optim.Adam(
                    [p for p in agent.actor.parameters() if p.requires_grad], lr=3e-4)
                
                # Temporary Actor freeze after rollback to stabilize the Critic convergence.
                if actor_freeze_episodes > 0:
                    agent.actor_frozen = True
                    print(f"Actor congelato per {actor_freeze_episodes} episodi: stabilizzazione post-rollback.")
                else:
                    agent.actor_frozen = False
                    print("Congelamento Actor post-rollback disattivato (--actor-freeze-episodes 0).")

                # Optional offline Critic pre-training to tune its weights on the buffer transitions.
                if getattr(args, 'pretrain_critic', False) and (len(memory) > batch_size or len(expert_memory) > batch_size):
                    print("[EMERGENZA] Pre-addestramento del Critic in corso sui dati offline del Replay Buffer (50,000 passi)...")
                    for pretrain_step in range(50000):
                        critic_loss_val, _, _ = agent.update(memory, elite_memory, expert_memory, batch_size, global_step=0)
                        if (pretrain_step + 1) % 10000 == 0:
                            print(f"  [Pre-addestramento] Passo {pretrain_step + 1}/50000 | Loss del Critic: {critic_loss_val:.4f}")
                    print("Pre-addestramento del Critic completato con successo!")
            else:
                print("Rollback richiesto ma nessun checkpoint valido trovato! Avvio ripresa normale.")
        else:
            print("Ripresa regolare dal checkpoint (nessun rollback o congelamento Actor).")

    os.makedirs('train_set/checkpoints', exist_ok=True)
    os.makedirs(SESSION_LOGS_DIR, exist_ok=True)
    log_file = os.devnull if os.environ.get('IM_WRAPPER_LOG_ONLY', '0') == '1' else phase_log_file

    def _control_log(message):
        """
        Prints a message to the terminal and writes it simultaneously to the log file.
        """
        print(message)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(message + "\n")

    stop_requested = False

    def _request_stop(signum, frame):
        """
        Handler for the SIGINT (Ctrl+C) and SIGTERM signals for a clean and orderly exit.

        Sets the stop_requested variable to True. The current episode is NOT
        interrupted: it continues to its natural outcome (lap completed, crash or
        step limit), so its data remains valid transitions. At the end of the
        episode the loop saves the full checkpoint and exits without restarting.
        """
        nonlocal stop_requested
        if not stop_requested:
            stop_requested = True
            print("\nRichiesta di arresto ricevuta: l'episodio corrente termina naturalmente, "
                  "poi checkpoint completo e uscita. Un secondo Ctrl+C forza l'uscita immediata.")
        else:
            print("\nSecondo Ctrl+C: uscita forzata immediata. L'ultimo checkpoint completo "
                  "resta quello salvato a fine dell'episodio precedente.")
            os._exit(130)

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    # The TORCS server wait in snakeoil checks this hook: a Ctrl+C given while
    # the client is blocked on "Waiting for server" aborts the wait with a clean exit,
    # instead of hanging until the server appears.
    snakeoil3.abort_check = lambda: stop_requested

    # Logging to the log file of the selected startup parameters to trace the session.
    if getattr(args, 'refine', False):
        initial_ref = f"{refine_plateau_level:.0f}m" if refine_plateau_level > 0.0 else "da impostare"
        _control_log(f"AVVIO con --refine: REFINEMENT armata da subito "
                     f"(aggiornamento Critic disattivato, loss Critic solo diagnostica, "
                     f"peso Behavioral Cloning={REFINE_BC_WEIGHT}, riferimento plateau={initial_ref}, "
                     f"episodio iniziale {start_episode})")
    if getattr(args, 'rollback', False):
        _control_log(f"AVVIO con --rollback: Actor congelato per {actor_freeze_episodes} episodi "
                     f"(0 = nessun congelamento), auto-refinement automatica="
                     f"{'attiva' if auto_refine_enabled else 'disattivata'}.")
    elif not auto_refine_enabled:
        _control_log("AVVIO con --no-auto-refine: refinement automatica disattivata; "
                     "--refine manuale resta disponibile.")

    elite_threshold = 500.0

    # ── TIME-ATTACK phase (opt-in via IM_TIME_ATTACK=1) ───────────────────────
    # To be activated AFTER collecting enough complete laps and retraining the BC: it reduces
    # the BC anchoring (higher alpha) and lowers the floor of the exploration noise to
    # shave the times. The time pressure (proportional bonus + personal record) is applied
    # ONLY in time-attack (see the SUCCESS block); in stabilization the completion is flat.
    noise_floor = TIME_ATTACK_NOISE_FLOOR if time_attack else EXPL_NOISE_END
    if time_attack:
        agent.bc_alpha = TIME_ATTACK_BC_ALPHA
        print(f"[TIME-ATTACK] Fase attiva: bc_alpha={agent.bc_alpha}, noise_floor={noise_floor}. "
              f"L'agente ottimizza il tempo sul giro battendo il proprio record.")

    # Manual override of the exploration noise (IM_EXPL_NOISE): fixes expl_noise to a constant value,
    # bypassing both the annealing and the floor. Useful for the STABILIZATION phase: forcing a low noise
    # on an already-converged policy (e.g. resume at low episodes) while staying in standard td3, without
    # going through time-attack which would also raise bc_alpha weakening the BC anchor. <=0 or absent => off.
    expl_noise_override = None
    _noise_override_env = os.environ.get('IM_EXPL_NOISE')
    if _noise_override_env is not None:
        try:
            _v = float(_noise_override_env)
            if _v > 0:
                expl_noise_override = _v
                print(f"[STABILIZZAZIONE] IM_EXPL_NOISE attivo: expl_noise fisso a {_v} "
                      f"(annealing e floor scavalcati).")
            else:
                print(f"IM_EXPL_NOISE={_noise_override_env} <= 0: override ignorato.")
        except ValueError:
            print(f"IM_EXPL_NOISE='{_noise_override_env}' non numerico: override ignorato.")

    # Lap recorder: collects the complete and clean laps driven by the agent during exploration
    # and saves them to train_set/laps_auto/ to enrich the BC dataset (data flywheel).
    # Time threshold configurable via IM_RECORD_MAX_LAP_TIME (default 80s); disable with IM_RECORD_LAPS=0.
    lap_recorder = LapRecorder(
        LAPS_AUTO_DIR,
        max_lap_time=float(os.environ.get('IM_RECORD_MAX_LAP_TIME', '80.0')),
        on_track_limit=1.0,
        enabled=(os.environ.get('IM_RECORD_LAPS', '1') != '0'),
    )

    # ── Corridor penalty (STABILIZATION) ──────────────────────────────────────
    # Pushes the policy to keep away from the edge BEFORE leaving it → reliable complete laps.
    # Full in stabilization; reduced to 25% in time-attack (there the full track width is needed to
    # shave the times). Overridable via IM_MARGIN_PENALTY (0 = disables it entirely).
    margin_coef_default = MARGIN_PENALTY_COEF * (0.25 if time_attack else 1.0)
    try:
        margin_coef = float(os.environ.get('IM_MARGIN_PENALTY', margin_coef_default))
    except ValueError:
        margin_coef = margin_coef_default
    if margin_coef > 0.0:
        print(f"[STABILIZZAZIONE] Penalità di corridoio attiva: coef={margin_coef:.1f} "
              f"(margine dal bordo per completare i giri).")

    # ── Per-sector telemetry (TIME-ATTACK) ────────────────────────────────────
    # Time-attack only: rewards beating one's own sector splits and logs where time is lost.
    sector_timer = None
    if time_attack:
        sector_timer = SectorTimer(
            TRACK_LENGTH_M,
            n_sectors=int(os.environ.get('IM_TA_SECTORS', str(TA_SECTORS_DEFAULT))),
            sidecar_path='train_set/checkpoints/td3_sector_best.json',
            reward_k=float(os.environ.get('IM_TA_SECTOR_K', str(TA_SECTOR_REWARD_K))),
            reward_cap=float(os.environ.get('IM_TA_SECTOR_CAP', str(TA_SECTOR_REWARD_CAP))),
            log=_control_log,
        )
        print(f"[TIME-ATTACK] Reward a settori attiva: {sector_timer.n} settori, "
              f"k={sector_timer.reward_k}, cap={sector_timer.reward_cap}.")

    print("Avvio training TD3+BC...")

    for episode in range(start_episode, args.episodes):
        # Linear annealing of the exploration noise: from EXPL_NOISE_START to the floor (EXPL_NOISE_END)
        # over EXPL_NOISE_ANNEAL_EPISODES episodes. At steady state micro-variations of the trajectory are needed,
        # not slides at race speed.
        # In time-attack the policy has already reached convergence (it closes the lap): the annealing — tied to the
        # ABSOLUTE episode number — would still impose ~0.065 at low episodes after a resume, i.e.
        # warmup noise on a mature policy, which throws it out at the first fast corner. So we go
        # straight to the floor (micro-variations around the optimal line), which is exactly the purpose
        # of the time-refinement phase.
        if expl_noise_override is not None:
            agent.expl_noise = expl_noise_override
        elif time_attack:
            agent.expl_noise = noise_floor
        else:
            agent.expl_noise = max(
                noise_floor,
                EXPL_NOISE_START - (EXPL_NOISE_START - noise_floor) * episode / EXPL_NOISE_ANNEAL_EPISODES
            )

        # Handling of the Actor unfreezing after the post-rollback stabilization phase
        if agent.actor_frozen and episode >= start_episode + actor_freeze_episodes:
            agent.actor_frozen = False
            print(f"Actor scongelato dopo {actor_freeze_episodes} episodi: "
                  f"riavvio aggiornamenti Actor con gradienti del Critic stabilizzati.")

        # Soft reset by default; periodic full relaunch (see RELAUNCH_EVERY_EPISODES).
        full_relaunch = (episode % RELAUNCH_EVERY_EPISODES == RELAUNCH_EPISODE_OFFSET)
        try:
            ob = env.reset(relaunch=full_relaunch)
        except snakeoil3.ServerTimeoutError as e:
            if e.aborted:
                _control_log(f"[{datetime.now().strftime('%H:%M:%S')}] STOP richiesto durante "
                             f"l'attesa del server TORCS: uscita pulita (ultimo checkpoint "
                             f"completo: episodio {episode}).")
                break
            raise
        episode_transitions = []

        # Lap recorder: new lap, and tracking of the raw pre-step observation.
        lap_recorder.start_episode()
        if sector_timer is not None:
            sector_timer.start_lap(float(np.array(ob.get('distFromStart', 0.0)).flat[0]))
        cur_ob = ob
        # Phase label (curriculum): time-attack if active, otherwise warmup (Critic
        # not yet warm) or online. Used for the recorder metadata and the logs.
        current_phase = "time_attack" if time_attack else ("warmup" if global_step < 15000 else "online")

        # Frame Stacking: concatenation of 3 time-spaced frames (t-12, t-6, t)
        # to provide information about the temporal dynamics (speed and acceleration).
        f_state = flatten_state(ob)
        state_stack = deque([f_state]*13, maxlen=13)
        stacked_state = np.concatenate([state_stack[0], state_stack[6], state_stack[12]])

        episode_reward, step, max_dist = 0, 0, 0.0
        critic_losses, actor_losses = [], []
        termination_reason = "TIMEOUT"
        new_record = False

        # Initial configuration of the gears and the telemetry variables.
        current_gear = 1
        steps_since_shift = 999  # allow the first shift immediately
        cur_speed_kmh = float(np.array(ob.get('speedX', 0.0)).flat[0]) * 50.0
        cur_rpm = float(np.array(ob.get('rpm', 0.0)).flat[0])
        # Reference of the last lap time to detect track completion.
        prev_last_lap = float(np.array(ob.get('lastLapTime', 0.0)).flat[0])
        torcs_lap_time = float(np.array(ob.get('curLapTime', 0.0)).flat[0])
        completed_lap_time = None
        episode_start_dist = float(np.array(ob.get('distFromStart', 0.0)).flat[0])

        while True:
            # Action selection via the current policy (with exploration noise).
            cont_action = agent.select_action(stacked_state, evaluate=False)

            # Continuous-action mapping: conversion of throttle and brake from [-1, 1] to [0, 1].
            env_action = np.zeros(4)
            env_action[0:3] = cont_action

            torcs_action = env_action.copy()
            torcs_action[1] = np.clip((torcs_action[1] + 1.0) / 2.0, 0.0, 1.0)  # accel: [-1,1] → [0,1]
            torcs_action[2] = np.clip((torcs_action[2] + 1.0) / 2.0, 0.0, 1.0)  # brake: [-1,1] → [0,1]
            # Continuous/multiplicative mutual exclusion to prevent sudden stalls
            torcs_action[1] = torcs_action[1] * (1.0 - torcs_action[2])

            # Algorithmic computation of the optimal gear based on speed, RPM and throttle.
            current_gear, _shifted = compute_gear(cur_speed_kmh, torcs_action[1], cur_rpm, current_gear, steps_since_shift)
            steps_since_shift = 0 if _shifted else steps_since_shift + 1
            torcs_action[3] = current_gear
            env_action[3] = current_gear

            # Lap recorder: raw pre-step state + action actually executed on TORCS.
            lap_recorder.record_step(cur_ob, torcs_action)

            next_ob, reward, env_done, info = env.step(torcs_action)
            cur_ob = next_ob
            cur_speed_kmh = float(np.array(next_ob.get('speedX', 0.0)).flat[0]) * 50.0
            cur_rpm = float(np.array(next_ob.get('rpm', 0.0)).flat[0])
            next_f_state = flatten_state(next_ob)
            state_stack.append(next_f_state)

            current_track_pos_m = float(np.array(next_ob.get('distFromStart', 0.0)).flat[0])
            current_dist = _track_progress_from_start(episode_start_dist, current_track_pos_m)
            last_lap_time = float(np.array(next_ob.get('lastLapTime', 0.0)).flat[0])
            torcs_lap_time = float(np.array(next_ob.get('curLapTime', 0.0)).flat[0])
            max_dist = max(max_dist, current_dist)

            # Per-step shaping on the ONLINE transitions (the agent's actual driving):
            #  - corridor: margin from the edge (stabilization → reliable complete laps);
            #  - sectors: reward for beating one's own splits (time-attack only).
            if margin_coef > 0.0:
                reward += margin_penalty(
                    float(np.array(next_ob.get('trackPos', 0.0)).flat[0]), coef=margin_coef)
            if sector_timer is not None:
                reward += sector_timer.update(current_track_pos_m, torcs_lap_time)

            lap_completed = bool(info.get('lap_completed', False))
            if not lap_completed:
                lap_completed = last_lap_time > 0.0 and abs(last_lap_time - prev_last_lap) > 0.01 and step > 500

            done = False
            if lap_completed and not info.get('crash', False):
                done, termination_reason = True, "SUCCESS"
                completed_lap_time = last_lap_time
                max_dist = max(max_dist, TRACK_LENGTH_M)
                # STABILIZATION: the completion is rewarded FLATLY (only LAP_SUCCESS_BONUS),
                # so a slow but clean lap is worth as much as a fast but risky lap → the policy
                # first learns to close the lap. The time pressure (proportional bonus and
                # personal record) is reserved for TIME-ATTACK, where the tenths need shaving.
                reward += LAP_SUCCESS_BONUS
                if time_attack:
                    reward += LAP_TIME_BONUS_PER_S * max(0.0, LAP_TIME_BONUS_REF_S - last_lap_time)
                if last_lap_time < best_lap_time:
                    # PERSONAL RECORD bonus: time-attack only (computed on the PREVIOUS best);
                    # in stabilization the best lap is recorded anyway, but without a time reward.
                    if time_attack:
                        reward += personal_best_bonus(best_lap_time, last_lap_time)
                    best_lap_time = last_lap_time
                    new_record = True
                    safe_save(agent.actor.state_dict(), 'train_set/checkpoints/td3_expl_best_lap.pth')

            if info.get('crash', False):
                done, termination_reason = True, "CRASH"


            if max_dist > best_distance and max_dist > 500.0:
                best_distance = max_dist
                safe_save(agent.actor.state_dict(), 'train_set/checkpoints/td3_expl_best_dist.pth')

            next_stacked_state = np.concatenate([state_stack[0], state_stack[6], state_stack[12]])

            time_limit_reached = (step >= args.max_steps)
            episode_finishes_now = done or env_done or time_limit_reached
            incomplete_lap = episode_finishes_now and termination_reason != "SUCCESS"
            if incomplete_lap:
                if termination_reason == "TIMEOUT":
                    termination_reason = "INCOMPLETE"
                reward -= INCOMPLETE_LAP_PENALTY

            mask = 0.0 if incomplete_lap else 1.0
            episode_transitions.append((stacked_state, cont_action, reward, next_stacked_state, mask))

            stacked_state = next_stacked_state
            episode_reward += reward
            step += 1
            global_step += 1

            # Update Frequency 1:1: one update at every simulation step (standard TD3).
            # The expert buffer is always full → it updates from the first step (the Critic
            # pre-trains on the human data, like the offline phase of TD3+BC) and the BC anchor is guaranteed.
            if len(expert_memory) > batch_size:
                critic_loss_val, actor_loss_val, _ = agent.update(memory, elite_memory, expert_memory, batch_size, global_step)
                critic_losses.append(critic_loss_val)
                if actor_loss_val != 0.0:
                    actor_losses.append(actor_loss_val)

            if done or env_done or time_limit_reached:
                for t in episode_transitions:
                    memory.push(t[0], t[1], t[2], t[3], t[4], expert=0.0)

                # Elite Buffer Injection:
                # If the distance covered exceeds the threshold (70% of the current distance record),
                # the episode transitions are inserted into the Elite Buffer for Self-Imitation.
                # To avoid storing wrong behaviours (Causal Confusion), the last 50 steps
                # before a crash are not marked as valid data for imitation.
                if max_dist >= elite_threshold:
                    n_trans = len(episode_transitions)
                    for i, t in enumerate(episode_transitions):
                        is_danger = (termination_reason == "CRASH") and (i >= n_trans - 50)
                        elite_memory.push(t[0], t[1], t[2], t[3], t[4], expert=0.0 if is_danger else 1.0)
                    elite_threshold = max(500.0, best_distance * 0.7)  # Monotonically increasing threshold

                # Lap recorder: saves the lap only if completed cleanly (internal quality gate).
                if termination_reason == "SUCCESS":
                    lap_recorder.finish_lap(completed_lap_time, phase=current_phase,
                                            episode=episode, global_step=global_step)
                else:
                    lap_recorder.discard()

                # Time-attack: close the sector lap (logs where it loses time + ideal lap) if
                # completed, otherwise discard the partial keeping the already-improved sector bests.
                if sector_timer is not None:
                    if termination_reason == "SUCCESS":
                        sector_timer.finish_lap(completed_lap_time)
                    else:
                        sector_timer.discard_lap()
                break

        # Detection of the lap time provided by the TORCS sensors.
        lap_time = completed_lap_time if completed_lap_time is not None else torcs_lap_time
        time_str = datetime.now().strftime("%H:%M:%S")
        avg_critic_loss = np.mean(critic_losses) if len(critic_losses) > 0 else 0.0
        avg_actor_loss = np.mean(actor_losses) if len(actor_losses) > 0 else 0.0
        critic_status = "OFF" if getattr(agent, 'refine_mode', False) else "ON"
        if global_step < 15000:
            actor_status = "WARM"
        elif getattr(agent, 'actor_frozen', False):
            actor_status = "FREEZE"
        else:
            actor_status = "ON"
        log_msg = (f"[{time_str}] Ep {episode+1:03d} | [{termination_reason}] | "
                   f"Reward: {episode_reward:7.1f} | Steps: {step:4d} | "
                   f"LapTime: {lap_time:5.1f}s | Dist: {int(max_dist):5d}m | "
                   f"CriticL: {avg_critic_loss:.3f} ({critic_status}) | "
                   f"ActorL: {avg_actor_loss:.3f} ({actor_status})")
        if new_record: log_msg += f" | Record"
        print(f" {log_msg}")

        with open(log_file, 'a', encoding='utf-8') as f: f.write(log_msg + "\n")

        agent.save_checkpoint(checkpoint_path, episode + 1, global_step, memory, elite_memory,
                              best_lap_time=best_lap_time,
                              best_eval_dist=best_eval_dist,
                              best_distance=best_distance)
        safe_save(agent.actor.state_dict(), 'train_set/checkpoints/td3_policy.pth')
        if stop_requested:
            _control_log(f"[{datetime.now().strftime('%H:%M:%S')}] STOP richiesto: "
                         f"checkpoint completo salvato all'episodio {episode + 1}; uscita pulita.")
            break

        if (episode + 1) % 5 == 0 and global_step > 15000:
            def _rlog(m):
                """
                Evaluation-specific (EVAL) helper that prints the message to the screen
                and appends it to the main log file.
                """
                print(m)
                with open(log_file, 'a', encoding='utf-8') as f: f.write(m + "\n")
            def _run_deterministic_eval():
                """
                Runs a single deterministic evaluation episode (without noise).

                Returns a tuple (eval_dist, eval_lap_time, eval_reward); eval_lap_time is None
                if the lap was not validly completed.
                """
                eval_ob = env.reset(relaunch=True)
                eval_stack = deque([flatten_state(eval_ob)]*13, maxlen=13)
                eval_stacked = np.concatenate([eval_stack[0], eval_stack[6], eval_stack[12]])
                eval_dist, eval_step, eval_reward = 0.0, 0, 0.0
                eval_lap_completed = False
                eval_lap_time = None
                eval_current_gear = 1  # Use of the algorithmic gear for the evaluation.
                eval_steps_since_shift = 999
                eval_cur_speed_kmh = float(np.array(eval_ob.get('speedX', 0.0)).flat[0]) * 50.0
                eval_cur_rpm = float(np.array(eval_ob.get('rpm', 0.0)).flat[0])
                # Timing of the best VALID lap completed in this deterministic evaluation.
                eval_prev_last_lap = float(np.array(eval_ob.get('lastLapTime', 0.0)).flat[0])
                eval_start_dist = float(np.array(eval_ob.get('distFromStart', 0.0)).flat[0])
                # State for the GEOMETRIC end-of-lap stop: track position and lap stopwatch
                # at the previous step, to detect the finish-line re-crossing regardless of the
                # lastLapTime sensor (which lags ~1 tick).
                eval_prev_track_pos_m = eval_start_dist
                eval_prev_cur_lap = float(np.array(eval_ob.get('curLapTime', 0.0)).flat[0])
                eval_transitions = []  # lap transitions, for the elite capture if completed

                agent.actor.eval()
                while eval_step < args.max_steps:
                    eval_step += 1
                    prev_eval_stacked = eval_stacked  # current state BEFORE the step (for the transition)
                    with torch.no_grad():
                        eval_action = agent.select_action(eval_stacked, evaluate=True)
                    eval_env = np.zeros(4)
                    eval_env[0:3] = eval_action
                    eval_env[1], eval_env[2] = np.clip((eval_env[1]+1)/2, 0, 1), np.clip((eval_env[2]+1)/2, 0, 1)
                    # Continuous/multiplicative mutual exclusion for EVAL
                    eval_env[1] = eval_env[1] * (1.0 - eval_env[2])
                    # Computation of the optimal gear for the evaluation phase.
                    eval_current_gear, _esh = compute_gear(eval_cur_speed_kmh, eval_env[1], eval_cur_rpm, eval_current_gear, eval_steps_since_shift)
                    eval_steps_since_shift = 0 if _esh else eval_steps_since_shift + 1
                    eval_env[3] = eval_current_gear

                    eval_ob, eval_r, eval_done, eval_info = env.step(eval_env)
                    eval_cur_speed_kmh = float(np.array(eval_ob.get('speedX', 0.0)).flat[0]) * 50.0
                    eval_cur_rpm = float(np.array(eval_ob.get('rpm', 0.0)).flat[0])
                    eval_reward += eval_r
                    eval_stack.append(flatten_state(eval_ob))
                    eval_stacked = np.concatenate([eval_stack[0], eval_stack[6], eval_stack[12]])
                    current_eval_track_pos_m = float(np.array(eval_ob.get('distFromStart', 0.0)).flat[0])
                    current_eval_dist = _track_progress_from_start(eval_start_dist, current_eval_track_pos_m)
                    eval_dist = max(eval_dist, current_eval_dist)
                    eval_cur_lap = float(np.array(eval_ob.get('curLapTime', 0.0)).flat[0])
                    # Record the transition (for the possible elite capture on lap completion)
                    eval_transitions.append((prev_eval_stacked, eval_action.copy(), eval_r, eval_stacked, 1.0))

                    # GEOMETRIC end-of-lap stop: independent of the lastLapTime sensor (which updates with
                    # ~1 tick of delay, and the distance clamp would mask an overrun into the 2nd lap).
                    # If the car has covered >=90% of the track and then distFromStart "jumps back" beyond
                    # half the track (finish-line re-crossing), the lap is complete: we stop IMMEDIATELY,
                    # no 2nd lap. Lap time from the sensor if updated, otherwise from the lap stopwatch
                    # at the last step before the wrap.
                    crossed_finish = (current_eval_track_pos_m + TRACK_LENGTH_M * 0.5 < eval_prev_track_pos_m)
                    if eval_dist >= TRACK_LENGTH_M * 0.9 and crossed_finish and not eval_info.get('crash', False):
                        eval_lap_completed = True
                        eval_dist = TRACK_LENGTH_M
                        eval_sensor_lap = float(np.array(eval_ob.get('lastLapTime', 0.0)).flat[0])
                        if eval_sensor_lap > 0.0 and abs(eval_sensor_lap - eval_prev_last_lap) > 0.01:
                            eval_lap_time = eval_sensor_lap
                        elif eval_prev_cur_lap > 0.0:
                            eval_lap_time = eval_prev_cur_lap
                        break
                    eval_prev_track_pos_m = current_eval_track_pos_m
                    eval_prev_cur_lap = eval_cur_lap

                    # Early stop of the evaluation at the completion of the first valid lap.
                    eval_last_lap = float(np.array(eval_ob.get('lastLapTime', 0.0)).flat[0])
                    eval_lap_completed = bool(eval_info.get('lap_completed', False))
                    if not eval_lap_completed:
                        eval_lap_completed = eval_last_lap > 0.0 and abs(eval_last_lap - eval_prev_last_lap) > 0.01 and eval_step > 500
                    if eval_lap_completed and not eval_info.get('crash', False):
                        eval_prev_last_lap = eval_last_lap
                        eval_lap_time = eval_last_lap
                        eval_dist = max(eval_dist, TRACK_LENGTH_M)
                        break

                    if eval_info.get('crash', False) or eval_done: break
                agent.actor.train()

                # Capture of the complete eval lap into the elite (self-imitation): if the lap was closed
                # cleanly, its deterministic transitions (the "clean line") enter the elite marked
                # expert=1.0, like the online injections. It is the source of FAST laps for the self-imitation, which the
                # self-recorded laps (slower) do not provide; as the policy improves, these captures
                # replace the slow seeds in FIFO.
                # Gated by --capture_eval_elite (default OFF): in stabilization self-imitating the razor line
                # of the eval over-sharpens the policy and destabilizes the exploration; reactivate it in the refinement.
                if args.capture_eval_elite and eval_lap_completed and len(eval_transitions) > 0:
                    for (s, a, r, ns, m) in eval_transitions:
                        elite_memory.push(s, a, r, ns, m, expert=1.0)
                    print(f"  [ELITE CAPTURE] Giro eval completo nell'elite: "
                          f"+{len(eval_transitions)} transizioni (totale {len(elite_memory)}).")

                if not _is_plausible_eval_dist(eval_dist):
                    _rlog(
                        f"  [EVAL] distanza {eval_dist:.1f}m non plausibile per un eval monogiro; "
                        "scartata da record/refinement."
                    )
                    eval_dist = 0.0
                return eval_dist, eval_lap_time, eval_reward

            _rlog("\n   [EVAL] Valutazione deterministica...")
            # Single run: a deterministic policy + deterministic simulator give results
            # reproducible to the meter (empirically verified: repeated runs always identical,
            # even on the anomalous crashes), so best-of-N adds no information.
            try:
                eval_dist, eval_lap_time, eval_reward = _run_deterministic_eval()
            except snakeoil3.ServerTimeoutError as e:
                if e.aborted:
                    _control_log(f"[{datetime.now().strftime('%H:%M:%S')}] STOP richiesto durante "
                                 f"l'attesa del server TORCS in eval: uscita pulita (ultimo "
                                 f"checkpoint completo: episodio {episode + 1}).")
                    break
                raise
            eval_score = _eval_score(eval_dist, eval_lap_time)
            eval_best_lap_in_run = eval_lap_time if eval_lap_time is not None else float('inf')

            refine_status = ""
            if getattr(agent, 'refine_mode', False):
                riferimento_plateau = f"{refine_plateau_level:.0f}m" if refine_plateau_level > 0.0 else "none"
                refine_status = (f" | Refine: ON (BC={agent.refine_bc_weight:.1f}, "
                                 f"Critic=OFF, plateau_ref={riferimento_plateau})")
            else:
                refine_status = " | Refine: OFF"

            lap_status = f" | Lap: {eval_lap_time:.3f}s" if eval_lap_time is not None else ""
            eval_msg = (f"[{time_str}]  [EVAL] Result: Dist {int(eval_dist)}m{lap_status} | "
                        f"Score {eval_score:.0f}m | Reward: {eval_reward:.1f}{refine_status}")
            print(f"  {eval_msg}")
            with open(log_file, 'a', encoding='utf-8') as f: f.write(eval_msg + "\n")

            # best_eval_dist contains the SCORE (name kept for checkpoint compatibility):
            # below 3608 it coincides with the distance, above it grows as the lap time improves.
            if eval_score > best_eval_dist:
                best_eval_dist = eval_score
                safe_save(agent.actor.state_dict(), 'train_set/checkpoints/td3_det_best_dist_run.pth')

            # Persistent saving of the historical absolute deterministic-score record
            # (distance for incomplete laps, time-equivalent for completed laps).
            det_best_dist_pth = 'train_set/checkpoints/td3_det_best_dist.pth'
            det_best_dist_txt = 'train_set/checkpoints/td3_det_best_dist.txt'
            prev_det_best_dist = safe_read_float(det_best_dist_txt, 0.0)
            if eval_score > prev_det_best_dist + BEST_DIST_EPS:
                safe_save(agent.actor.state_dict(), det_best_dist_pth)
                safe_write_text(det_best_dist_txt, f"{eval_score:.2f}")
                msg = (f"  NUOVO MIGLIOR DETERMINISTICO ASSOLUTO: score {int(eval_score)}m "
                       f"(precedente {int(prev_det_best_dist)}m, preservato anche dopo --clean)")
                print(msg)
                with open(log_file, 'a', encoding='utf-8') as f: f.write(msg + "\n")
                if getattr(agent, 'refine_mode', False):
                    agent.refine_mode = False
                    agent.refine_bc_weight = 1.0
                    agent.actor_frozen = actor_freeze_episodes > 0
                    start_episode = episode  # Freeze the Actor for the configured window from now on
                    refine_evals_no_improve = 0
                    refine_collapse_count = 0
                    refine_best_mean = 0.0
                    refine_plateau_level = 0.0
                    refine_attempts = 0
                    refine_attempt_plateau_ref = 0.0
                    refine_attempt_limit_logged = False
                    refine_evals_count = 0
                    refine_good_eval_count = 0
                    recent_eval_window.clear()
                    _rlog("  REFINEMENT CONCLUSA CON SUCCESSO! Nuovo record deterministico rilevato.")
                    if actor_freeze_episodes > 0:
                        _rlog(f"  Rientro in modalità allineamento Critic: "
                              f"Actor congelato per {actor_freeze_episodes} episodi.")
                    else:
                        _rlog("  Rientro in training normale: congelamento Actor disattivato.")

            # Persistent saving of the best valid deterministic lap (submission candidate).
            if eval_best_lap_in_run < float('inf'):
                det_best_lap_pth = 'train_set/checkpoints/td3_det_best_lap.pth'
                det_best_lap_txt = 'train_set/checkpoints/td3_det_best_lap.txt'
                prev_det_best_lap = safe_read_float(det_best_lap_txt, float('inf'))
                if eval_best_lap_in_run < prev_det_best_lap:
                    safe_save(agent.actor.state_dict(), det_best_lap_pth)
                    safe_write_text(det_best_lap_txt, f"{eval_best_lap_in_run:.3f}")
                    msg = f"  NUOVO MIGLIOR GIRO VALIDO (eval deterministica): {eval_best_lap_in_run:.3f}s (preservato anche dopo --clean)"
                    print(msg)
                    with open(log_file, 'a', encoding='utf-8') as f: f.write(msg + "\n")

            # Handling of the Auto-Refinement state machine (refinement activation or rollback).
            # If the Actor is frozen, the evaluations are ignored for the plateau computation.
            actor_is_frozen = getattr(agent, 'actor_frozen', False)
            if actor_is_frozen and not agent.refine_mode:
                refine_evals_no_improve = 0
                refine_best_mean = 0.0
                recent_eval_window.clear()
                _rlog("  Auto-refinement sospesa: Actor congelato; eval ignorato per il plateau.")
            else:
                recent_eval_window.append(eval_score)

            if auto_refine_enabled and not agent.refine_mode and not actor_is_frozen:
                # PLATEAU detection on a STATISTIC (not on the single best, robust to lucky
                # strikes): the MEAN of the recent window stops rising. The full window is needed.
                if len(recent_eval_window) >= REFINE_WINDOW:
                    cur_mean = sum(recent_eval_window) / len(recent_eval_window)
                    if cur_mean > refine_best_mean * _refine_frac(refine_best_mean, REFINE_IMPROVE_FRAC, REFINE_LAP_IMPROVE_FRAC):
                        refine_best_mean = cur_mean          # the typical performance is still rising
                        refine_evals_no_improve = 0
                    else:
                        refine_evals_no_improve += 1          # typical flat → counts toward the plateau
                    if refine_evals_no_improve >= REFINE_PLATEAU_EVALS and (episode + 1) >= REFINE_MIN_EP:
                        # Plateau reference = recent MEDIAN ('good' mode of the bimodal), not the single stochastic max.
                        candidate_plateau_level = float(np.median(list(recent_eval_window)))
                        reset_msg = None
                        if refine_attempt_plateau_ref <= 0.0:
                            refine_attempt_plateau_ref = candidate_plateau_level
                        elif candidate_plateau_level > refine_attempt_plateau_ref * _refine_frac(refine_attempt_plateau_ref, REFINE_NEW_PLATEAU_FRAC, REFINE_LAP_NEW_PLATEAU_FRAC):
                            previous_plateau_ref = refine_attempt_plateau_ref
                            refine_attempts = 0
                            refine_attempt_plateau_ref = candidate_plateau_level
                            refine_attempt_limit_logged = False
                            reset_msg = (f"  Nuovo plateau rilevato: riferimento {candidate_plateau_level:.0f}m "
                                         f"> riferimento attuale {previous_plateau_ref:.0f}m "
                                         f"(+{(candidate_plateau_level / previous_plateau_ref - 1.0) * 100:.0f}%). "
                                         "Contatore refinement azzerato per il nuovo regime.")

                        if refine_attempts < REFINE_MAX_ATTEMPTS:
                            agent.refine_mode = True
                            agent.refine_bc_weight = REFINE_BC_WEIGHT
                            refine_plateau_level = candidate_plateau_level
                            refine_collapse_count = 0
                            refine_evals_count = 0
                            refine_good_eval_count = 0
                            refine_breakout_logged = False
                            refine_attempt_limit_logged = False
                            if reset_msg is not None:
                                _rlog(reset_msg)
                            _rlog(f"  AUTO-REFINEMENT ATTIVA (tentativo {refine_attempts+1}/{REFINE_MAX_ATTEMPTS} "
                                  f"sul plateau {refine_attempt_plateau_ref:.0f}m): "
                                  f"media recente in plateau a {cur_mean:.0f}m, aggiornamento Critic disattivato, "
                                  f"loss Critic solo diagnostica, peso Behavioral Cloning→{REFINE_BC_WEIGHT}, "
                                  f"riferimento plateau (mediana)={refine_plateau_level:.0f}m "
                                  f"(max recente={max(recent_eval_window):.0f}m)")
                        elif not refine_attempt_limit_logged:
                            refine_attempt_limit_logged = True
                            _rlog(f"  AUTO-REFINEMENT non riattivata: limite {REFINE_MAX_ATTEMPTS}/{REFINE_MAX_ATTEMPTS} "
                                  f"raggiunto per il plateau {refine_attempt_plateau_ref:.0f}m. "
                                  "Training normale finché non emerge un plateau più alto.")
            elif agent.refine_mode and refine_plateau_level <= 0.0:
                # Determination of the initial plateau level for the manual refinement via the median.
                if len(recent_eval_window) >= 4:
                    refine_plateau_level = float(np.median(list(recent_eval_window)))
                    if refine_attempt_plateau_ref <= 0.0:
                        refine_attempt_plateau_ref = refine_plateau_level
                    refine_evals_count = 0
                    refine_good_eval_count = 0
                    _rlog(f"  refinement: riferimento plateau = {refine_plateau_level:.0f}m "
                          f"(MEDIANA degli ultimi {len(recent_eval_window)} eval, max={max(recent_eval_window):.0f}m)")
            elif agent.refine_mode:
                # In REFINEMENT.
                refine_evals_count += 1
                # Detection and logging of the reference-plateau breakout.
                # Dual-regime threshold: +10% in the distance regime, +1% (≈0.7s of lap) in the time regime.
                breakout_frac = _refine_frac(refine_plateau_level, REFINE_BREAKOUT_FRAC, REFINE_LAP_BREAKOUT_FRAC)
                breakout_detected = eval_score > refine_plateau_level * breakout_frac
                if breakout_detected:
                    refine_good_eval_count += 1
                else:
                    refine_good_eval_count = 0
                if not refine_breakout_logged and breakout_detected:
                    refine_breakout_logged = True
                    _rlog(f"  PLATEAU SUPERATO: eval score {eval_score:.0f}m "
                          f"> riferimento {refine_plateau_level:.0f}m (+{(eval_score/refine_plateau_level-1)*100:.1f}%)")
                # Restore of normal training (with weight consolidation and possible Actor freeze)
                # if the breakout is stable or close to the best absolute record.
                near_best_breakout = breakout_detected and prev_det_best_dist > 0.0 and eval_score >= prev_det_best_dist - REFINE_NEAR_BEST_MARGIN
                stable_breakout = refine_good_eval_count >= REFINE_GOOD_EVALS_TO_CONSOLIDATE
                if near_best_breakout or stable_breakout:
                    ref_lvl = refine_plateau_level
                    good_eval_count = refine_good_eval_count
                    agent.refine_mode = False
                    agent.refine_bc_weight = 1.0
                    agent.actor_frozen = actor_freeze_episodes > 0
                    start_episode = episode
                    refine_evals_no_improve = 0
                    refine_collapse_count = 0
                    refine_best_mean = 0.0
                    refine_plateau_level = 0.0
                    refine_attempts = 0
                    refine_attempt_plateau_ref = 0.0
                    refine_attempt_limit_logged = False
                    refine_evals_count = 0
                    refine_good_eval_count = 0
                    recent_eval_window.clear()
                    if near_best_breakout:
                        _rlog(f"  REFINEMENT CONSOLIDATA: breakout vicino al miglior deterministico "
                              f"(eval score {eval_score:.0f}m, best {prev_det_best_dist:.0f}m). "
                              "Peso Behavioral Cloning→1.0, aggiornamento Critic riattivato.")
                    else:
                        _rlog(f"  REFINEMENT CONSOLIDATA: {good_eval_count} eval buone consecutive "
                              f"sopra il riferimento plateau (ultima {eval_score:.0f}m, riferimento {ref_lvl:.0f}m). "
                              "Peso Behavioral Cloning→1.0, aggiornamento Critic riattivato.")
                    if actor_freeze_episodes > 0:
                        _rlog(f"  Rientro in modalità allineamento Critic: "
                              f"Actor congelato per {actor_freeze_episodes} episodi.")
                    else:
                        _rlog("  Rientro in training normale: congelamento Actor disattivato.")
                # Activation of the preventive rollback in case of a prolonged performance collapse.
                elif eval_score < refine_plateau_level * REFINE_COLLAPSE_FRAC:
                    refine_collapse_count += 1
                else:
                    refine_collapse_count = 0
                if refine_collapse_count >= 3:
                    ref_lvl = refine_plateau_level
                    if refine_attempt_plateau_ref <= 0.0:
                        refine_attempt_plateau_ref = ref_lvl
                    if os.path.exists(det_best_dist_pth):
                        agent.actor.load_actor_weights(det_best_dist_pth, agent.device)
                        agent.actor_target.load_state_dict(agent.actor.state_dict())
                    agent.refine_mode = False
                    agent.refine_bc_weight = 1.0
                    refine_attempts += 1
                    refine_evals_no_improve = 0
                    refine_collapse_count = 0
                    refine_evals_count = 0
                    refine_good_eval_count = 0
                    refine_best_mean = 0.0          # Reset of the parameters to measure the plateau again.
                    refine_plateau_level = 0.0
                    _rlog(f"  REFINEMENT collassata (<{int(REFINE_COLLAPSE_FRAC*100)}% di {ref_lvl:.0f}m) "
                          f"→ ROLLBACK al miglior deterministico (td3_det_best_dist), peso Behavioral Cloning→1.0, "
                          f"aggiornamento Critic riattivato. Tentativi sul plateau {refine_attempt_plateau_ref:.0f}m: "
                          f"{refine_attempts}/{REFINE_MAX_ATTEMPTS}")
                # Timeout exit if the refinement drags on for 40 episodes without improvements.
                elif refine_evals_count >= 8:
                    ref_lvl = refine_plateau_level
                    if refine_attempt_plateau_ref <= 0.0:
                        refine_attempt_plateau_ref = ref_lvl
                    agent.refine_mode = False
                    agent.refine_bc_weight = 1.0
                    refine_attempts += 1
                    refine_evals_no_improve = 0
                    refine_collapse_count = 0
                    refine_evals_count = 0
                    refine_good_eval_count = 0
                    refine_best_mean = 0.0
                    refine_plateau_level = 0.0
                    _rlog(f"  TIMEOUT REFINEMENT (40 episodi in refinement senza superare il record) "
                          f"→ Uscita automatica, peso Behavioral Cloning→1.0, aggiornamento Critic riattivato. "
                          f"Tentativi sul plateau {refine_attempt_plateau_ref:.0f}m: "
                          f"{refine_attempts}/{REFINE_MAX_ATTEMPTS}")

        if stop_requested:
            _control_log(f"[{datetime.now().strftime('%H:%M:%S')}] STOP richiesto durante/ dopo eval: "
                         f"ultimo checkpoint completo episodio {episode + 1}; uscita pulita.")
            break

    env.end()

if __name__ == '__main__':
    train()
