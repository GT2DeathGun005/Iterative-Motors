# Iterative Motors

Iterative Motors is an autonomous-driving project for TORCS, built for the **IBM AI Racing League**.
The goal: an agent that beats human lap performance. The model does not learn "from scratch": first
it **imitates** a real driver via Behavioral Cloning (BC), then it pushes past that baseline with
Reinforcement Learning (**TD3+BC**), producing faster and more repeatable laps.

The best human lap in the dataset is **69.54 s**; our agent's best deterministic lap is **68.838 s**,
and we are pushing toward the track limit (**~65 s**).

For the full technical details see **[ARCHITECTURE.md](ARCHITECTURE.md)**.

## Key features

- TORCS simulator driven over the SCR/UDP protocol (without changing its physics — IBM League rule).
- Human data collection with a PS5 DualSense controller or keyboard.
- 29-D sensor state with temporal frame stacking (t-12, t-6, t → 87-D); gearing handled separately.
- Behavioral Cloning with a per-channel weighted loss (steering / throttle / brake).
- Bojarski-style data augmentation with an **on-track clamp** (never teaches the car to drive off
  track) and a widened angular perturbation (~10°).
- TD3+BC fine-tuning: Actor warm-started from BC, Twin Critic, hybrid expert/online/elite sampling.
- **Data flywheel**: the agent records its own clean laps (`laps_auto/`) to retrain a stronger BC.
- **Two-regime reward**: stabilization (corridor margin penalty + flat completion → finish almost
  every lap) and time-attack (time bonus + **per-sector split telemetry** → shave tenths by beating
  its own splits, toward a theoretical ideal lap).
- Deterministic gear shifting decoupled from the network; atomic checkpoints with backups and robust resume.
- Deterministic evaluation with automatic best-checkpoint detection.
- **Single orchestrator `run.sh`** for the whole pipeline, with a status dashboard.

## Project layout

```text
.
|-- run.sh                          # single pipeline orchestrator (CLI controller)
|-- requirements.txt                # Python dependencies (tested versions)
|-- src/iterative_motors/           # PACKAGE: project logic
|   |-- common/                    # constants, state/normalization, checkpointing
|   |-- env/                       # TORCS wrapper, SCR client, gearing, autostart
|   |-- models/                    # networks (Actor/PolicyNetwork/Critic) and action mapping
|   |-- data/                      # replay buffer, HDF5 dataset, lap recorder, data collection
|   |-- bc/                        # data augmentation + Behavioral Cloning training
|   |-- rl/                        # TD3+BC agent, reward/time-attack, training loop
|   `-- eval/                      # deterministic agent test
|-- train_set/                      # dataset (laps/, laps_auto/), checkpoints, logs (NOT in git)
`-- telemetry/                      # CSVs produced by tests
```

Entry points live in the package and are launched via `run.sh` or as modules
(`PYTHONPATH=src python -m iterative_motors.<subpackage>.<module>`).

## Requirements

Linux with:
- Python 3; TORCS with the SCR server; `xvfb-run` (headless); `xte` (TORCS autostart menu, package `xautomation`);
- Python libraries: see `requirements.txt` (`torch`, `numpy`, `h5py`, `pygame`, `gym`).

```bash
sudo apt install torcs xvfb xautomation
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # for CUDA, see the note in requirements.txt
```

## Quick start (via the orchestrator)

```bash
./run.sh             # interactive menu (arrows + Enter)
./run.sh help        # full list of direct commands
./run.sh status      # dashboard: running tasks, dataset, records, last logs
```

The `run.sh` menu works like a small pit wall: an ASCII Formula 1, a summary of
processes/dataset/records, and arrow-selectable options. When you start a command it suggests common
presets and a free field for args/env vars, e.g. `SHOW_GUI=1 --laps 1`. The direct commands remain
available for automation and scripting.

### 1. Collect human demonstrations

```bash
./run.sh collect --device controller     # or --device keyboard
```

Valid laps land in `train_set/laps/lap_NNN.h5`. Useful options: `--segment_only` (corner segments
only), `--zones "670:900,2380:2530"` (specific zones).

### 2. Train Behavioral Cloning

```bash
./run.sh bc
```

Produces `train_set/checkpoints/bc_policy.pth` and `state_norm.npz` (runs in the background; follow
with `./run.sh logs bc` and `./run.sh status`).

### 3. TD3+BC fine-tuning + lap harvesting

```bash
./run.sh td3 --episodes 2500
```

The `td3` launcher applies the **stabilization defaults**: trust region `--trust_region 0.3` (anchors
the Actor to the data support on non-expert samples, anti policy-collapse) and `IM_EXPL_NOISE=0.04`
(fixed exploration noise, overrides annealing). Both overridable (e.g. `--trust_region 0`,
`IM_EXPL_NOISE=0.02`).

**No time filters** in this phase: only *finishing* the lap matters, not speed. The launcher
therefore loads ALL human laps (`--expert_max_lap_time 0`), seeds the elite with ALL self-recorded
laps (`--reseed_elite_max_lap_time 999`) and records every clean lap regardless of time
(`IM_RECORD_MAX_LAP_TIME=999`). The speed filters return automatically in time-attack.

The reward favors **robust completion**: a corridor penalty (margin from the track edge,
`IM_MARGIN_PENALTY`) and a *flat* completion bonus (no time pressure — that is reserved for
time-attack). Keep `--capture_eval_elite` **off**: self-imitating the razor-edge eval line
over-sharpens the policy, the opposite of what stabilization needs (see §7 of ARCHITECTURE.md).

During training the agent automatically records its own clean, completed laps in
`train_set/laps_auto/` (disable with `IM_RECORD_LAPS=0`). Clean stop: `./run.sh stop td3` (saves a
checkpoint before exiting).

### 4. Enrich the BC (flywheel) and time-attack

```bash
# Retrain the BC on human + self-recorded laps
./run.sh bc-enriched --output train_set/checkpoints/enriched/bc_policy.pth

# Time-attack: the agent optimizes lap time by beating its own record
./run.sh time-attack --episodes 4000
```

In time-attack the time pressure (time bonus + personal-record bonus) is re-enabled and a **per-sector
split telemetry** reward is added (`rl/sector_timer.py`): the agent is rewarded for beating its own
sector splits, and at lap completion the log shows where it loses time (`S07(+0.22s) …`) and the
**theoretical ideal lap** (the sum of the best splits). The best splits are persisted to
`checkpoints/td3_sector_best.json` and survive restarts; tunable via `IM_TA_SECTORS`,
`IM_TA_SECTOR_K`, `IM_TA_SECTOR_CAP`.

To adopt the enriched BC in a new TD3 lineage: copy the new `bc_policy.pth` + `state_norm.npz` from
`enriched/` into `train_set/checkpoints/`, reset the TD3 checkpoints (the deterministic records stay
protected), and relaunch `./run.sh td3`.

### 5. Test the agent

```bash
./run.sh test --laps 3                # auto-detect the best checkpoint
SHOW_GUI=1 ./run.sh test --laps 1     # with the TORCS window visible
./run.sh test --weights train_set/checkpoints/td3_det_best_lap.pth --laps 5
```

Telemetry CSVs are written to `telemetry/`.

## The idea behind the model

The policy sees 3 temporal frames of the sensor state (87-D input) and outputs steering, throttle and
brake; the gear is computed by `gearing.py`. BC provides the initial competence; TD3+BC keeps that
expert anchor while optimizing the racing reward (advance, stay on track, smoothness, finish the lap,
lower the time). The data flywheel feeds the agent's best laps back into the BC dataset, raising the
starting point on every iteration.

## References

- Lillicrap et al., *Continuous Control with Deep RL*, 2015 — https://arxiv.org/abs/1509.02971
- Fujimoto, van Hoof, Meger, *Addressing Function Approximation Error in Actor-Critic Methods*, 2018 — https://arxiv.org/abs/1802.09477
- Fujimoto, Gu, *A Minimalist Approach to Offline RL*, 2021 — https://arxiv.org/abs/2106.06860
- Beeson, Montana, *Improving TD3-BC*, 2022 — https://arxiv.org/abs/2211.11802
- Bojarski et al., *End to End Learning for Self-Driving Cars*, 2016 — https://arxiv.org/abs/1604.07316
- Loiacono, Cardamone, Lanzi, *SCR Championship: Competition Software Manual*, 2013 — https://arxiv.org/abs/1304.1672
