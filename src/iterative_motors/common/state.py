"""State representation: sensor flattening, normalization, frame stacking.

z-score boundary (important): the lap HDF5 files contain **raw-scaled, NOT normalized** states.
For this reason the state has two forms:

  - ``flatten_state_raw(obs)``  -> raw 29D vector (track/200, speed/50 already done by the
    wrapper; here additionally wheelSpinVel/100 and rpm/10000). It is the form written to disk by
    the human laps (data_collection) and by the lap recorder.
  - ``flatten_state_norm(obs)`` -> ``apply_state_norm(flatten_state_raw(obs))``, the z-scored form
    fed to the network during RL/eval.

The statistics (mean/std) live in ``state_norm.npz`` and are loaded at import;
``reload_state_norm()`` allows reloading them after a recompute (enrichment step).
"""

from collections import deque

import numpy as np

from .constants import STATE_NORM_PATH, STATE_DIM, STACK_LEN, FRAME_STRIDE_K

# Normalization statistics loaded at module level (mean-0/std-1).
_STATE_MEAN = None
_STATE_STD = None


def load_state_norm(path=STATE_NORM_PATH):
    """Reads (mean, std) from ``state_norm.npz``; returns (None, None) if absent."""
    import os
    if os.path.exists(path):
        d = np.load(path)
        return d['mean'].astype(np.float32), d['std'].astype(np.float32)
    return None, None


def reload_state_norm(path=STATE_NORM_PATH):
    """Reloads the statistics from disk, updating the module state."""
    global _STATE_MEAN, _STATE_STD
    _STATE_MEAN, _STATE_STD = load_state_norm(path)
    return _STATE_MEAN, _STATE_STD


def save_state_norm(mean, std, path=STATE_NORM_PATH):
    """Saves the normalization statistics and updates the module state."""
    import os
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    np.savez(path, mean=np.asarray(mean, dtype=np.float32), std=np.asarray(std, dtype=np.float32))
    reload_state_norm(path)


# Load at import (consistent with the historical behaviour of the monoliths).
reload_state_norm()


def apply_state_norm(s):
    """Standardizes a raw state vector: ``(s - mean) / (std + 1e-3)``.

    If the statistics are not loaded it returns the vector unchanged (no-op).
    The 1e-3 epsilon avoids divisions by zero on static sensors.
    """
    if _STATE_MEAN is None:
        return s
    return ((s - _STATE_MEAN) / (_STATE_STD + 1e-3)).astype(np.float32)


def flatten_state_raw(state_dict: dict) -> np.ndarray:
    """Flattens the TORCS observation dictionary into the **raw** 29D vector.

    Order: [angle(1), track(19), trackPos(1), speedX(1), speedY(1), speedZ(1),
            wheelSpinVel(4)/100, rpm(1)/10000]. distFromStart is NOT part of the
            state (the policy must drive from the sensors only).
    """
    def _scalar(key: str, default: float = 0.0) -> float:
        val = state_dict.get(key, default)
        if val is None:
            return default
        if isinstance(val, np.ndarray):
            return float(val.flat[0])
        return float(val)

    def _array(key: str, size: int) -> np.ndarray:
        val = state_dict.get(key, None)
        if val is None:
            return np.zeros(size, dtype=np.float32)
        arr = np.array(val, dtype=np.float32).flatten()
        if arr.shape[0] != size:
            padded = np.zeros(size, dtype=np.float32)
            n = min(size, arr.shape[0])
            padded[:n] = arr[:n]
            return padded
        return arr

    try:
        return np.concatenate([
            np.array([_scalar('angle')]),
            _array('track', 19),
            np.array([_scalar('trackPos')]),
            np.array([_scalar('speedX')]),
            np.array([_scalar('speedY')]),
            np.array([_scalar('speedZ')]),
            _array('wheelSpinVel', 4) / 100.0,
            np.array([_scalar('rpm') / 10000.0]),
        ]).astype(np.float32)
    except Exception as e:
        print(f"flatten_state_raw fallita (stato a zero): {e}")
        return np.zeros(STATE_DIM, dtype=np.float32)


def flatten_state_norm(state_dict: dict) -> np.ndarray:
    """Normalized (z-score) 29D vector for network inference."""
    return apply_state_norm(flatten_state_raw(state_dict))


class FrameStacker:
    """Sliding buffer that stacks 3 time-spaced frames (t-12, t-6, t).

    Reproduces ``deque([f]*STACK_LEN, maxlen=STACK_LEN)`` with concatenation of indices
    0, FRAME_STRIDE_K, 2*FRAME_STRIDE_K.
    """

    def __init__(self, init_frame: np.ndarray):
        self._dq = deque([init_frame] * STACK_LEN, maxlen=STACK_LEN)

    def append(self, frame: np.ndarray) -> None:
        self._dq.append(frame)

    @property
    def stacked(self) -> np.ndarray:
        return np.concatenate([self._dq[0], self._dq[FRAME_STRIDE_K], self._dq[2 * FRAME_STRIDE_K]])
