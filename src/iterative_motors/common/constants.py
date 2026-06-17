"""Costanti globali e percorsi del progetto Iterative Motors.

Centralizza i valori prima duplicati nei vari script monolitici: percorsi di
checkpoint/dataset, dimensioni dello stato e dello stack temporale, geometria
del tracciato e gli angoli dei 19 sensori track (che devono coincidere tra il
client SCR e la data augmentation della BC).
"""

import os

# ── Percorsi ──────────────────────────────────────────────────────────────
# Radice del repository (cartella ``AIcar``): tre livelli sopra questo file
# (common/ -> iterative_motors/ -> src/ -> AIcar/).
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))

TRAIN_SET_DIR = os.path.join(PROJECT_ROOT, 'train_set')
CHECKPOINT_ROOT = os.path.join(TRAIN_SET_DIR, 'checkpoints')
CHECKPOINT_BACKUP_ROOT = os.path.join(CHECKPOINT_ROOT, 'backups')
BUFFERS_DIR = os.path.join(CHECKPOINT_ROOT, 'buffers')
STATE_NORM_PATH = os.path.join(CHECKPOINT_ROOT, 'state_norm.npz')

LAPS_DIR = os.path.join(TRAIN_SET_DIR, 'laps')            # giri umani (data_collection)
LAPS_AUTO_DIR = os.path.join(TRAIN_SET_DIR, 'laps_auto')  # giri auto-raccolti dalla TD3
SESSION_LOGS_DIR = os.path.join(TRAIN_SET_DIR, 'session_logs')
TELEMETRY_DIR = os.path.join(PROJECT_ROOT, 'telemetry')

# ── Stato e stacking temporale ────────────────────────────────────────────
STATE_DIM = 29                 # feature sensoriali per frame
NUM_STACK_FRAMES = 3           # frame impilati (t-12, t-6, t)
STACK_DIM = STATE_DIM * NUM_STACK_FRAMES  # 87 = input della rete
FRAME_STRIDE_K = 6             # distanza temporale tra i frame impilati
STACK_LEN = 2 * FRAME_STRIDE_K + 1        # 13 = lunghezza del buffer deque

# ── Geometria del tracciato ───────────────────────────────────────────────
TRACK_LENGTH_M = 3608.0        # lunghezza del circuito (corkscrew)

# ── Sensori track (SCR) ───────────────────────────────────────────────────
# I 19 angoli (gradi) dei raggi di distanza dai bordi pista. Devono essere
# IDENTICI tra la init string del client SCR (snakeoil) e la perturbazione
# geometrica della data augmentation Bojarski-style nella BC.
SENSOR_ANGLES_DEG = (
    -45.0, -19.0, -12.0, -7.0, -4.0, -2.5, -1.7, -1.0, -0.5, 0.0,
    0.5, 1.0, 1.7, 2.5, 4.0, 7.0, 12.0, 19.0, 45.0,
)
