"""Utility trasversali di Iterative Motors (costanti, stato, checkpoint).

Questo sottopacchetto raccoglie ciò che è condiviso da tutti gli altri (env, models, data,
bc, rl) e che NON dipende da loro, evitando duplicazione e import circolari:

  - ``constants``  : percorsi del progetto, dimensioni di stato/stack, geometria della pista,
                     angoli dei 19 sensori track (condivisi tra client SCR e augmentation BC).
  - ``state``      : costruzione del vettore di stato 29D grezzo / normalizzato, statistiche di
                     normalizzazione (state_norm.npz) e frame stacking temporale (t-12, t-6, t).
  - ``checkpoint`` : salvataggio/caricamento atomico e resistente alle interruzioni, con
                     rotazione dei backup (.bak/.prev) e sidecar testuali (archivio intoccabile).

L'API pubblica più usata è ri-esportata qui per comodità
(``from iterative_motors.common import flatten_state_norm, safe_save, ...``).
"""

from .constants import (
    PROJECT_ROOT,
    TRAIN_SET_DIR,
    CHECKPOINT_ROOT,
    CHECKPOINT_BACKUP_ROOT,
    BUFFERS_DIR,
    STATE_NORM_PATH,
    LAPS_DIR,
    LAPS_AUTO_DIR,
    SESSION_LOGS_DIR,
    TELEMETRY_DIR,
    STATE_DIM,
    NUM_STACK_FRAMES,
    STACK_DIM,
    FRAME_STRIDE_K,
    STACK_LEN,
    TRACK_LENGTH_M,
    SENSOR_ANGLES_DEG,
)
from .state import (
    flatten_state_raw,
    flatten_state_norm,
    apply_state_norm,
    load_state_norm,
    reload_state_norm,
    save_state_norm,
    FrameStacker,
)
from .checkpoint import (
    safe_save,
    safe_save_npz,
    safe_write_text,
    safe_read_float,
)

__all__ = [
    "PROJECT_ROOT", "TRAIN_SET_DIR", "CHECKPOINT_ROOT", "CHECKPOINT_BACKUP_ROOT",
    "BUFFERS_DIR", "STATE_NORM_PATH", "LAPS_DIR", "LAPS_AUTO_DIR", "SESSION_LOGS_DIR",
    "TELEMETRY_DIR", "STATE_DIM", "NUM_STACK_FRAMES", "STACK_DIM", "FRAME_STRIDE_K",
    "STACK_LEN", "TRACK_LENGTH_M", "SENSOR_ANGLES_DEG",
    "flatten_state_raw", "flatten_state_norm", "apply_state_norm",
    "load_state_norm", "reload_state_norm", "save_state_norm", "FrameStacker",
    "safe_save", "safe_save_npz", "safe_write_text", "safe_read_float",
]
