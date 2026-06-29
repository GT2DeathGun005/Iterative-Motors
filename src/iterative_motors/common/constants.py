"""Global constants and paths of the Iterative Motors project.

Centralizes values previously duplicated across the various monolithic scripts: checkpoint/
dataset paths, state and temporal-stack sizes, track geometry and the 19 track-sensor angles
(which must match between the SCR client and the BC data augmentation).
"""

import os

# ── Paths ───────────────────────────────────────────────────────────────────
# Repository root (the ``AIcar`` folder): three levels above this file
# (common/ -> iterative_motors/ -> src/ -> AIcar/).
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))

TRAIN_SET_DIR = os.path.join(PROJECT_ROOT, 'train_set')
CHECKPOINT_ROOT = os.path.join(TRAIN_SET_DIR, 'checkpoints')
CHECKPOINT_BACKUP_ROOT = os.path.join(CHECKPOINT_ROOT, 'backups')
BUFFERS_DIR = os.path.join(CHECKPOINT_ROOT, 'buffers')
STATE_NORM_PATH = os.path.join(CHECKPOINT_ROOT, 'state_norm.npz')

LAPS_DIR = os.path.join(TRAIN_SET_DIR, 'laps')            # human laps (data_collection)
LAPS_AUTO_DIR = os.path.join(TRAIN_SET_DIR, 'laps_auto')  # laps self-recorded by the TD3 agent
SESSION_LOGS_DIR = os.path.join(TRAIN_SET_DIR, 'session_logs')
TELEMETRY_DIR = os.path.join(PROJECT_ROOT, 'telemetry')

# ── State and temporal stacking ─────────────────────────────────────────────
STATE_DIM = 29                 # sensor features per frame
NUM_STACK_FRAMES = 3           # stacked frames (t-12, t-6, t)
STACK_DIM = STATE_DIM * NUM_STACK_FRAMES  # 87 = network input
FRAME_STRIDE_K = 6             # temporal distance between stacked frames
STACK_LEN = 2 * FRAME_STRIDE_K + 1        # 13 = length of the deque buffer

# ── Track geometry ──────────────────────────────────────────────────────────
TRACK_LENGTH_M = 3608.0        # circuit length (corkscrew)

# ── Track sensors (SCR) ─────────────────────────────────────────────────────
# The 19 angles (degrees) of the distance rays to the track edges. They must be
# IDENTICAL between the SCR client init string (snakeoil) and the geometric
# perturbation of the Bojarski-style data augmentation in the BC.
SENSOR_ANGLES_DEG = (
    -45.0, -19.0, -12.0, -7.0, -4.0, -2.5, -1.7, -1.0, -0.5, 0.0,
    0.5, 1.0, 1.7, 2.5, 4.0, 7.0, 12.0, 19.0, 45.0,
)
