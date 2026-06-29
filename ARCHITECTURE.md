# Iterative Motors — Project architecture

Iterative Motors is an autonomous-driving agent for **TORCS** (the *corkscrew* circuit) built for the
**IBM AI Racing League**. The core idea: don't learn to drive "from scratch" with Reinforcement
Learning alone (slow and unstable), but **start from a human driver's competence** transferred into a
neural network via **Behavioral Cloning (BC)**, and then **surpass it** with **TD3+BC** fine-tuning
until the agent beats the best human time (**69.54s**), aiming at the track record (**target ~65s**).

> **Name and folder.** The project is called *Iterative Motors*; `AIcar` is only the name of the
> repository root folder. All reusable code lives in the `src/iterative_motors/` package.

> **IBM AI Racing League constraint.** TORCS physics and installation must NOT be modified. TORCS is
> used as an external simulator over the SCR protocol (UDP); the `env/` subpackage is the only point
> of contact and does not alter its configuration.

Index:
1. [Repository structure](#1-repository-structure)
2. [The pipeline and the data flywheel](#2-the-pipeline-and-the-data-flywheel)
3. [State representation](#3-state-representation)
4. [Neural networks](#4-neural-networks)
5. [TORCS environment](#5-torcs-environment)
6. [Behavioral Cloning](#6-behavioral-cloning)
7. [TD3+BC](#7-td3bc)
8. [Lap recorder and enrichment](#8-lap-recorder-and-enrichment)
9. [Time-attack and personal record](#9-time-attack-and-personal-record)
10. [Checkpoint system](#10-checkpoint-system)
11. [run.sh orchestrator](#11-runsh-orchestrator)
12. [Configuration](#12-configuration)
13. [RL speedup (deferred)](#13-rl-speedup-deferred)
14. [Operational notes](#14-operational-notes)

---

## 1. Repository structure

```
AIcar/                                  # repo root folder
  run.sh                                # single pipeline ORCHESTRATOR (CLI controller)
  requirements.txt                      # Python dependencies (tested versions)
  src/iterative_motors/                 # PACKAGE: all project logic
    common/                             # cross-cutting utilities (no dependency on other subpackages)
      constants.py    # paths, state/stack sizes, track geometry, the 19 sensor angles
      state.py        # flatten_state_raw / flatten_state_norm, apply_state_norm, FrameStacker
      checkpoint.py   # atomic safe_save/load, .bak/.prev rotation, sidecars (untouchable archive)
    env/                                # the only point of contact with TORCS
      gym_torcs.py    # Gym wrapper: observations, actions, per-step reward, terminations
      snakeoil3_gym.py# SCR client (UDP); the 19 sensor angles from SENSOR_ANGLES_DEG
      gearing.py      # algorithmic gear shifting with hysteresis (the net does NOT predict the gear)
      autostart.sh    # TORCS startup macro via xte
    models/                             # shared neural networks
      networks.py     # Actor (RL), PolicyNetwork (BC/eval), Critic — shared backbone and heads
      action_mapping.py# net-action <-> TORCS-pedals conversion, throttle/brake mutual exclusion
    data/                               # data: RL buffer, BC dataset, lap recorder
      replay_buffer.py# ReplayBuffer + load_expert_data (HDF5 -> buffer with stacking)
      hdf5_dataset.py # TorcsHDF5Dataset + load_dataset (with self-recorded laps for enrichment)
      lap_recorder.py # LapRecorder: saves the agent's clean laps to laps_auto/
      collection.py   # ENTRY POINT for human lap collection (PS5 controller / keyboard)
    bc/                                 # Behavioral Cloning
      augmentation.py # Bojarski-style data augmentation (AugmentConfig + augment_batch)
      train_bc.py     # ENTRY POINT for BC training (BehaviorCloningTrainer + main)
    rl/                                 # Reinforcement Learning TD3+BC
      agent.py        # TD3BCAgent: RL/BC update, robust save/load_checkpoint
      reward.py       # lap-end bonuses/penalties, corridor penalty, eval score, personal record
      sector_timer.py # SectorTimer: per-sector split times and telemetry reward (time-attack only)
      train_rl.py     # ENTRY POINT for TD3+BC fine-tuning (+ lap harvest, time-attack)
    eval/
      test_agent.py   # ENTRY POINT for deterministic evaluation with best auto-detect
  train_set/                            # local data (NOT tracked by git)
    laps/             # human laps (HDF5)
    laps_auto/        # laps self-recorded by the TD3 agent (flywheel)
    checkpoints/      # weights, buffers, record sidecars; state_norm.npz
    session_logs/  .run/  (PIDs of run.sh tasks)
  telemetry/                            # CSVs produced by tests
  ARCHITECTURE.md  README.md
```

**Why a package + an orchestrator.** The code used to live in monolithic scripts in the root with
heavy duplication (the state-flatten function was repeated 4 times, the network 3 times, etc.). Now
every responsibility has a single home in the package and entry points are launched as modules
(`python -m iterative_motors.<subpackage>.<module>`) or, more conveniently, via `run.sh`. The entry
points add `src/` to the path with a small bootstrap, so they work both as a module and when run
directly.

---

## 2. The pipeline and the data flywheel

```
1. HUMAN COLLECTION  collection.py  -> train_set/laps/lap_NNN.h5   (29-D state, 4-D action, 50Hz)
2. BEHAVIORAL CLON.  train_bc.py    -> bc_policy.pth + state_norm.npz
3. WARM-START + RL   train_rl.py    -> Actor initialized from BC, then online TD3+BC
       │
       ├── HARVEST   the LapRecorder saves the complete/clean laps -> train_set/laps_auto/
       │
4. ENRICHMENT        train_bc.py --auto_laps laps_auto -> BC retrained on (human ∪ auto)
5. TIME-ATTACK       train_rl.py (IM_TIME_ATTACK=1) -> the agent beats its own times
6. TEST              test_agent.py  -> deterministic evaluation, best auto-detect
```

The conceptual core is the **data flywheel (3 → 4 → 5)**. The initial BC is trained on the ~75 human
laps, whose best is 69.54s: imitating the *average* of those laps pulls the policy toward a mediocre
lap. With RL the agent learns to close laps **cleaner and more repeatable** than the human ones; the
`LapRecorder` captures them and feeds them back into the dataset. By retraining the BC on this
enriched dataset, the **starting point** of the next RL iteration is higher. Iterating, the system
lifts itself toward times that no human lap in the dataset contains.

---

## 3. State representation

The state is a **29-D** vector built by `common/state.py` in this order:

| Index | Feature | Scale | Notes |
|---|---|---|---|
| 0 | `angle` | rad | car angle relative to the track axis |
| 1–19 | `track[19]` | /200 | distance to the edges on 19 rays (see angles below) |
| 20 | `trackPos` | — | lateral position (0 = center, ±1 = edges) |
| 21–23 | `speedX/Y/Z` | /50 | longitudinal/lateral/vertical speed |
| 24–27 | `wheelSpinVel[4]` | /100 | rotation speed of the 4 wheels |
| 28 | `rpm` | /10000 | engine revs |

Gear and `distFromStart` are **not** part of the state: the gear is handled by `gearing.py`;
`distFromStart` is collected only as metadata, because correlating the action with the absolute
position would introduce a train-test mismatch (the policy must drive from the sensors, not "by
memory").

**Two forms of the state** (the most delicate distinction in the project):
- `flatten_state_raw(obs)` — **raw** vector (physical scale). This is what is written to the HDF5
  files (human and auto laps) and used by the `LapRecorder`.
- `flatten_state_norm(obs)` = `apply_state_norm(flatten_state_raw(obs))` — **z-scored** vector
  ((x − mean) / (std + 1e-3)), the form fed to the network in RL/eval.

The mean/std statistics are computed by the BC over the entire dataset, saved in
`train_set/checkpoints/state_norm.npz` and shared by training and eval (they must match).

**Temporal frame stacking.** The network does not see a single instant but 3 frames spaced
`FRAME_STRIDE_K = 6` steps apart (t-12, t-6, t), concatenated into an **87-D** input. This gives the
network implicit information about speed and acceleration, useful to predict the trajectory. The
`FrameStacker` (in `state.py`) encapsulates this logic.

**The 19 sensor angles** (`constants.SENSOR_ANGLES_DEG`, in degrees):
`-45 -19 -12 -7 -4 -2.5 -1.7 -1 -0.5 0 0.5 1 1.7 2.5 4 7 12 19 45`. Dense near 0° (look-ahead along
the axis) and sparse at ±45° (edge detection). They are used **both** by the SCR client (to
initialize the rays) **and** by the BC data augmentation (to perturb them coherently): hence they are
a single shared constant.

---

## 4. Neural networks

Defined in `models/networks.py`. All command-producing networks share the same **backbone** (4 blocks
of `Linear(512) → LayerNorm → ReLU`) and the same **continuous head** `Linear(512, 3)`; only the
output activation changes.

- **Actor** (RL): `forward` applies `tanh` to all 3 channels (outputs in [-1, 1]); `sample` adds
  clipped Gaussian exploration noise (annealed by training). `load_bc_weights` performs the warm-start
  from BC, rescaling throttle/brake by 0.5 (Sigmoid→Tanh).
- **PolicyNetwork** (BC and eval): `forward` with `tanh`(steering)+`sigmoid`(throttle/brake → [0,1]),
  the activation the BC is trained with; `sample` with `tanh` on all channels, to evaluate RL
  checkpoints with the same class.
- **Critic**: Twin Q-Network (independent q1, q2) over an 87-D + 3-D input; the Bellman target uses
  the minimum of the two to counter value overestimation.

**Checkpoint compatibility.** Backbone and head produce `state_dict` keys identical to the original
code (`backbone.0/1/3/4/6/7/9/10`, `continuous_head`, `q1/q2.0/2/4`). This is a non-negotiable
constraint: the historical records in `train_set/checkpoints/` must load without migration. Since
Actor and PolicyNetwork share the keys, BC and RL weights are interchangeable.

---

## 5. TORCS environment

`env/gym_torcs.py` is the Gym wrapper; `env/snakeoil3_gym.py` is the low-level SCR UDP client.

**Actions** (4-D): `[steer ∈ [-1,1], accel ∈ [0,1], brake ∈ [0,1], gear]`. The network predicts the
first 3; the gear is computed by `gearing.py` with speed/RPM thresholds and hysteresis (cooldown) to
avoid "hunting" (oscillating gear changes). `action_mapping.py` converts between network space and
pedals and applies the **mutual exclusion** (`accel ← accel·(1−brake)`) to avoid throttle and brake
pressed together.

**Per-step reward** (in `gym_torcs.py`):

```
reward = 1.5 · progress + pos_penalty − 0.05 · |Δsteer|
  progress    = (speedX / 50) · cos(angle)          # advance along the track axis
  pos_penalty = −2 · max(0, |trackPos| − 1)²         # soft barrier at the edges
```

`progress` rewards the speed projected along the track (≈0 if the car is sideways, negative if going
backward). `pos_penalty` is zero inside the edges and grows quadratically beyond |trackPos|=1. The
anti-zigzag term penalizes abrupt steering changes, favoring smooth trajectories.

**Early terminations** (training): off-track (|trackPos| > 1.25), stall (progress < 0.1 after 500
steps ≈ 10s), spin (cos(angle) < 0), lap completed (change of `lastLapTime` after step 500). The
terminal bonuses/penalties (completion, time, incomplete lap, personal record) and the steady-state
per-step shaping (corridor penalty, sector reward) are added by the training loop via `rl/reward.py`
and `rl/sector_timer.py`, with different signals depending on the regime (see §7).

**Control frequency**: the SCR protocol works nominally at 50Hz. The wrapper periodically relaunches
TORCS (kill + autostart) to counter a memory leak observed in long runs.

---

## 6. Behavioral Cloning

Entry point `bc/train_bc.py` (class `BehaviorCloningTrainer`). It trains the `PolicyNetwork` to
reproduce human actions with a **per-channel weighted MSE loss**:

- base weights `[steer=1, accel=1, brake=3]`;
- dynamic brake boost (×8 when the driver brakes) to learn braking points, rare but critical events;
- steering boost in corners (×4 when |steer| > 0.07) for trajectory precision.

> **Weight revision (vs. the original).** Brake used to weigh up to 5×25 = **125×** steering: the loss
> became almost a single braking regressor, at the expense of steering precision. Now the peak is ~24×
> (base 3 × boost 8), with more emphasis on steering in corners.

### Bojarski-style data augmentation

`bc/augmentation.py` (`AugmentConfig` + `augment_batch`). To mitigate *covariate shift* (at inference
the car ends up in states the driver never visited), lateral position and angle are synthetically
perturbed, correcting the targets to teach recovery toward the center:

- lateral `trackPos` perturbation (σ 0.22) and angular (σ 0.09 rad, clip ~10°);
- the 19 rays are perturbed in a way that is **geometrically consistent** with the simulated shift;
- the steering target is corrected toward the center (configurable gains), throttle reduced as a
  function of the perturbation magnitude, plus an "overspeed" branch that teaches braking earlier
  before high-speed corners.

> **Key fixes (vs. the original).**
> 1. **On-track clamp** (`on_track_limit = 0.95`): the perturbed `trackPos` never crosses the edge, so
>    the network learns to recover toward the center **from poses still on track** — it is never taught
>    to drive off track (a real risk of the previous version, without the clamp).
> 2. **Widened angular perturbation** from ~4.5° to **~10°**: covers realistic mid-corner
>    misalignments, from which the network previously did not learn to recover.

All parameters are in `AugmentConfig`, so they are tunable without touching the code.

---

## 7. TD3+BC

Entry point `rl/train_rl.py`, agent `rl/agent.py`. Combines TD3 (Fujimoto et al. 2018) with the BC
constraint of TD3+BC (Fujimoto & Gu 2021): an offline-to-online architecture that inherits the BC
knowledge (Actor warm-start) and refines it with RL without collapsing the policy.

**Actor loss:** `L = −λ · Q(s, π(s)) + BC_penalty`, with `λ = bc_alpha / mean(|Q(s, π(s))|)`. The
dynamic normalization of λ keeps the scale of the RL term and the BC term comparable, making the
gradient stable against Q-value variations. Higher `bc_alpha` = more weight to RL (useful to surpass
the expert). The **BC penalty** is an MSE between predicted action and expert action applied **only**
to samples with `expert_mask = 1` (strict masking): this lets the agent explore trajectories different
from the human ones without being penalized.

**Three-way hybrid sampling** (per batch): 25% **expert** (human), 15% **elite** (best autonomous
runs), 60% **online** (current exploration). If online/elite have little data, the quota is
compensated by the expert (always available). Capacities: online **2M** (wide horizon to avoid
forgetting recent history too soon), elite **200k**, expert 400k.

Stability features (all preserved from the original code):
- **Progressive anchor**: the expert buffer is permanent (capacity 400k >> dataset). The
  `--expert_max_lap_time` filter is *phase-dependent*: in **time-attack** it keeps only the *best*
  human laps (≤71s) so the BC anchor points at the human best; in **stabilization** (`td3` launcher)
  it is disabled (`0`) to load ALL human laps — there, finishing matters, not speed (see §7).
- **Trust region** (`--trust_region`, default 0.3 in the `td3` launcher): FIXED-weight MSE between the
  Actor's action and the buffer's action on **non-expert** samples (the 75% where the only force would
  be `max Q`, which would push the Actor outside the data support onto actions with overestimated Q).
  It anchors the Actor to the data support and cures deterministic-policy collapse when the Actor
  becomes active again.
- **Elite cycle** (self-imitation, fed by three sources so it is never starved):
  - *online gate*: an episode enters the elite if its distance exceeds 70% of the current record; the
    last 50 steps before a crash are excluded from imitation (anti causal-confusion);
  - *eval capture*: every lap **completed** in deterministic evaluation (detected via a geometric stop
    at the finish-line re-crossing, independent of the `lastLapTime` sensor lag) enters the elite — it
    is the source of *fast* laps (the "clean line") that the recorder, stuck on the slower exploration
    laps, does not provide;
  - *seeding* (`--reseed_elite_max_lap_time S`, + a safety auto-seed if the elite loads starved):
    injects the self-recorded laps ≤ S as a **consistency** bootstrap when the policy completes few of
    them. The (slower) seeds age out in FIFO as the fast captures replace them.
- **Refinement FSM**: on an evaluation plateau, it reduces the BC weight and freezes the Critic to
  refine the Actor toward a fixed value function; with rollback on collapse and exit on breakout.
- **Warm-up**: the Actor stays frozen until the Critic stabilizes (15000 steps), then updates with
  Delayed Policy Update (every 2 Critic steps) and Target Policy Smoothing.

### Two-regime reward: stabilization → time-attack

The progress reward (`1.5·progress`) accumulates ~8000 points per lap: the terminal bonuses (~150) are
<2% of the return, so on their own they do **not** teach *finishing* instead of risking. Therefore the
shaping changes by phase (gating on the `time_attack` / `IM_TIME_ATTACK` variable):

- **STABILIZATION** (`td3` launcher, default — goal: finish *almost every* lap, even slower):
  - **Corridor penalty** (`reward.margin_penalty`): a per-step quadratic penalty that kicks in already
    at `|trackPos| > 0.80`, i.e. *before* leaving the driving surface (the environment's `pos_penalty`
    only intervenes beyond 1.0, when the car is already at the edge and a micro-perturbation sends it
    into a crash). It teaches a **safety margin**; being per-step it weighs as much as progress and,
    generalizing over the sensor state, transfers the lesson to the whole track. Coefficient via
    `IM_MARGIN_PENALTY`.
  - **Flat completion**: the time-proportional bonus and the personal-record bonus are **disabled**;
    only `LAP_SUCCESS_BONUS` remains. A slow but clean lap is worth as much as a fast but risky one →
    the policy first learns to finish the lap.
  - Applied to **online** transitions only (the agent's actual driving), not to expert/elite.
- **TIME-ATTACK** (`time-attack` launcher, `IM_TIME_ATTACK=1` — goal: shave tenths, see §9): time +
  personal-record bonuses re-enabled, corridor penalty reduced to 25% (the full track width is needed)
  and a **per-sector telemetry reward** (`rl/sector_timer.py`) that rewards beating its own splits.

---

## 8. Lap recorder and enrichment

`data/lap_recorder.py` (`LapRecorder`). During training, for each step it captures the **raw pre-step
state** (`flatten_state_raw`) and the **action actually executed** on TORCS (`[steer, applied accel,
brake, gear]`). At the end of an episode it saves the lap to `train_set/laps_auto/lap_auto_NNN.h5`
**only if**:

- the lap was **completed** (SUCCESS, not crash/incomplete);
- it is **clean** (`max|trackPos| ≤ on_track_limit`, default 1.0 = never off track);
- it falls within the time threshold `IM_RECORD_MAX_LAP_TIME` (default 80s; the `td3` launcher raises
  it to `999` = no time filter in stabilization, so every clean lap is collected even if slow);
- it has a minimum length and no NaN/Inf values.

The HDF5 format is **identical** to that of the human laps (datasets `states`/`actions`/
`dist_from_start` + attributes), so the auto laps are loadable both by `TorcsHDF5Dataset` (for the BC)
and by `load_expert_data` (for the RL buffer). Enrichment is triggered with
`train_bc.py --auto_laps laps_auto`, which merges human+auto and **recomputes** `state_norm.npz` over
the union.

---

## 9. Time-attack and personal record

The time-attack phase is enabled **after** the policy is stable (finishes almost every lap, §7) and
aims at shaving tenths down to the track limit. It is enabled with `IM_TIME_ATTACK=1` (`time-attack`
launcher).

**Time pressure** (in `rl/reward.py`, active *only* in time-attack):
- **Time-proportional bonus**: `+10` for every second below the 80s reference.
- **Personal-record bonus**: when the agent sets a new best time it receives `+30 + 15·(seconds
  gained)` on top of the completion bonus. A direct incentive to shave times when the distance is
  already saturated at lap end. In stabilization both are off (§7).

**Per-sector telemetry reward** (`rl/sector_timer.py`, class `SectorTimer`): the track is divided into
N contiguous sectors over `distFromStart` (default 18). For each sector the **best time ever driven**
(best split) is kept, persisted in the sidecar `td3_sector_best.json` so it survives restarts. At the
close of each sector the agent is rewarded (or penalized) based on how much it beats its own record
for that sector (`reward_k·Δt`, clamped to `±reward_cap`): a **dense** signal indicating *where* to
gain time, with good credit assignment. In the time-attack regime the improvements are on the order of
tenths, so the signal stays in the linear, gentle region.

- The best splits update **in real time** at the close of each sector (even from different laps), so
  the **theoretical ideal lap** = sum of the best splits is the optimal line assembled piece by piece:
  chasing it brings the agent closer to the track limit.
- At a completed lap, the log shows the breakdown: lap time, ideal lap, gap, and the sectors where it
  **loses the most time** (`S07(+0.22s) ...`) — telemetry useful to understand where to intervene.
- Tunable via env: `IM_TA_SECTORS` (number of sectors), `IM_TA_SECTOR_K`, `IM_TA_SECTOR_CAP`.

**Other phase parameters**:
- It reduces the BC anchoring (`bc_alpha` → 4.0, more weight to RL) and brings the exploration noise
  **straight to the floor 0.02**, without annealing: on an already-converged policy the annealing —
  tied to the *absolute* episode number — would still impose ~0.065 after a resume at low episodes,
  destroying the trajectory at the first fast corner.
- **Noise override** (`IM_EXPL_NOISE=<v>`): fixes `expl_noise` to a constant value, bypassing
  annealing and floor. Useful to force low noise on a mature policy while staying in standard `td3`
  (default 0.04 in the launcher) without going through time-attack, which would also raise `bc_alpha`.

---

## 10. Checkpoint system

`common/checkpoint.py`. **Atomic** save resistant to interruptions: write to a temporary file →
`fsync` → backup rotation (`.bak` → `.prev`) → atomic `os.replace` → directory `fsync`. On load the
candidates are scanned in order (main file, `.bak`, `.prev`), so a mid-write interruption never leaves
a corrupted checkpoint.

**Untouchable archive.** The deterministic records (`td3_det_best_lap.pth`, `td3_det_best_dist.pth`)
are accompanied by text sidecars (`.txt`) with the value, are overwritten only on improvement and
survive `--clean`. Resume temporally aligns the buffers with the checkpoint (it does not load buffers
more recent than the `.pth`, a hint of a later interrupted save).

Main files in `train_set/checkpoints/`:

| File | Meaning |
|---|---|
| `bc_policy.pth` | policy trained with BC only (RL warm-start) |
| `state_norm.npz` | state mean/std for normalization |
| `td3_checkpoint.pth` | full checkpoint for TD3+BC resume |
| `td3_det_best_lap.pth` | best valid deterministic lap (submission candidate) |
| `td3_det_best_dist.pth` | best absolute deterministic score |
| `td3_expl_best_lap/dist.pth` | best results found during exploration |
| `td3_sector_best.json` | best per-sector times (time-attack); defines the "ideal lap" |

---

## 11. run.sh orchestrator

`run.sh` is the single pipeline controller. It sets `PYTHONPATH=src`, launches long tasks in the
background saving PIDs and logs, and offers a status dashboard.

```
./run.sh collect [args]       # human lap collection (foreground)
./run.sh bc [args]            # BC training (background)
./run.sh bc-enriched [args]   # BC training on human+auto (background)
./run.sh td3 [args]           # TD3+BC + lap harvest (background)
./run.sh time-attack [args]   # time-attack phase (background)
./run.sh test [args]          # deterministic evaluation (foreground)
./run.sh status               # active processes, dataset, records, last logs
./run.sh logs <task> [n]      # last n log lines of a task
./run.sh stop [task]          # clean stop (SIGINT) of one task or all
```

`status` shows: running tasks (with uptime), number of human/auto laps, best deterministic time,
checkpoint episode, and the last log line of each task. `stop` sends SIGINT, which the training entry
points intercept to save a full checkpoint before exiting.

---

## 12. Configuration

| Where | Name | Default | Meaning |
|---|---|---|---|
| env | `IM_RECORD_LAPS` | 1 | enable the lap recorder during training |
| env | `IM_RECORD_MAX_LAP_TIME` | 80.0 | time threshold (s) to save an auto lap (the `td3` launcher raises it to 999 = no filter) |
| env | `IM_TIME_ATTACK` | 0 | enable the time-attack phase (time pressure + sector reward) |
| env | `IM_MARGIN_PENALTY` | 12.0 / 3.0 | corridor-penalty coef. (stabilization / time-attack; 0 = off) |
| env | `IM_TA_SECTORS` | 18 | number of sectors for the telemetry reward (time-attack) |
| env | `IM_TA_SECTOR_K` | 30.0 | reward scale per second gained over the sector record |
| env | `IM_TA_SECTOR_CAP` | 8.0 | clamp of the per-sector reward/penalty |
| env | `IM_EXPL_NOISE` | 0.04 (td3) | exploration-noise override (bypasses annealing/floor) |
| env | `TORCS_KILL_ALL` | 1 | global TORCS kill (memory-leak workaround) |
| env | `SHOW_GUI` | 0 | show the TORCS window instead of headless Xvfb |
| rl | `--episodes` | 1000 | final episode (must exceed the resume one) |
| rl | `--bc_alpha` | 2.5 | RL/BC balance (time-attack raises it to 4.0) |
| rl | `--expert_max_lap_time` | 71.0 | BC-anchor quality filter; `≤0` = off. The `td3` launcher sets it to 0 (loads all human laps) |
| rl | `--reseed_elite_max_lap_time` | 0 | seed the elite with auto laps ≤ threshold (0 = off); the `td3` launcher sets it to 999 = all |
| bc | `--auto_laps DIR` | — | directory of auto laps for enrichment |
| bc | `--epochs / --batch_size / --lr` | 300 / 256 / 3e-4 | training hyperparameters |

The data-augmentation parameters are in `AugmentConfig` (`bc/augmentation.py`).

---

## 13. RL speedup (deferred)

A point intentionally left unimplemented (to be evaluated in the future), with the hooks already in
place:
- `env/gym_torcs.py::_kill_torcs` documents where to replace the global kill with a per-instance
  teardown, needed to run N TORCS environments in parallel (ports 3001+i, separate Xvfb displays);
- the update/step ratio (UTD) is concentrated in a single point of the loop (`agent.update`), easy to
  parametrize.

---

## 14. Operational notes

- **Checkpoint lineage after enrichment.** Recomputing `state_norm.npz` changes the normalization: TD3
  checkpoints trained with the old one are no longer consistent. To adopt an enriched BC it is best to
  start a **new lineage**: copy the new `bc_policy.pth` + `state_norm.npz` into `checkpoints/`, then
  `./run.sh td3` after resetting the TD3 checkpoints (the deterministic records are protected by the
  sidecars). The old checkpoints remain as an archive.
- The data and checkpoints in `train_set/` are not tracked by git (a project choice): they must be
  saved with external backups.
- Prerequisites: Python 3, TORCS with the SCR server, `xvfb-run` (headless), `xte` (autostart), and
  the libraries `torch`, `numpy`, `h5py`, `pygame`, `gym` (see `requirements.txt`).
```

