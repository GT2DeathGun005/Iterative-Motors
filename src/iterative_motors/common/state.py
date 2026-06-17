"""Rappresentazione dello stato: flatten dei sensori, normalizzazione, frame stacking.

Confine z-score (importante): gli HDF5 dei giri contengono stati **raw-scaled, NON
normalizzati**. Per questo lo stato ha due forme:

  - ``flatten_state_raw(obs)``  -> vettore 29D grezzo (track/200, speed/50 già fatti dal
    wrapper; qui in più wheelSpinVel/100 e rpm/10000). È la forma scritta su disco dai
    giri umani (data_collection) e dal lap recorder.
  - ``flatten_state_norm(obs)`` -> ``apply_state_norm(flatten_state_raw(obs))``, la forma
    z-scored data in pasto alla rete durante RL/eval.

Le statistiche (mean/std) vivono in ``state_norm.npz`` e sono caricate all'import;
``reload_state_norm()`` permette di ricaricarle dopo un ricalcolo (Step enrichment).
"""

from collections import deque

import numpy as np

from .constants import STATE_NORM_PATH, STATE_DIM, STACK_LEN, FRAME_STRIDE_K

# Statistiche di normalizzazione caricate a livello di modulo (mean-0/std-1).
_STATE_MEAN = None
_STATE_STD = None


def load_state_norm(path=STATE_NORM_PATH):
    """Legge (mean, std) da ``state_norm.npz``; ritorna (None, None) se assente."""
    import os
    if os.path.exists(path):
        d = np.load(path)
        return d['mean'].astype(np.float32), d['std'].astype(np.float32)
    return None, None


def reload_state_norm(path=STATE_NORM_PATH):
    """Ricarica le statistiche dal disco aggiornando lo stato del modulo."""
    global _STATE_MEAN, _STATE_STD
    _STATE_MEAN, _STATE_STD = load_state_norm(path)
    return _STATE_MEAN, _STATE_STD


def save_state_norm(mean, std, path=STATE_NORM_PATH):
    """Salva le statistiche di normalizzazione e aggiorna lo stato del modulo."""
    import os
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    np.savez(path, mean=np.asarray(mean, dtype=np.float32), std=np.asarray(std, dtype=np.float32))
    reload_state_norm(path)


# Caricamento all'import (coerente col comportamento storico dei monoliti).
reload_state_norm()


def apply_state_norm(s):
    """Standardizza un vettore di stato grezzo: ``(s - mean) / (std + 1e-3)``.

    Se le statistiche non sono caricate restituisce il vettore invariato (no-op).
    L'epsilon 1e-3 evita divisioni per zero su sensori statici.
    """
    if _STATE_MEAN is None:
        return s
    return ((s - _STATE_MEAN) / (_STATE_STD + 1e-3)).astype(np.float32)


def flatten_state_raw(state_dict: dict) -> np.ndarray:
    """Appiattisce il dizionario di osservazione TORCS nel vettore 29D **grezzo**.

    Ordine: [angle(1), track(19), trackPos(1), speedX(1), speedY(1), speedZ(1),
             wheelSpinVel(4)/100, rpm(1)/10000]. distFromStart NON fa parte dello
            stato (la policy deve guidare dai soli sensori).
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
    """Vettore 29D normalizzato (z-score) per l'inferenza della rete."""
    return apply_state_norm(flatten_state_raw(state_dict))


class FrameStacker:
    """Buffer scorrevole che impila 3 frame distanziati nel tempo (t-12, t-6, t).

    Riproduce ``deque([f]*STACK_LEN, maxlen=STACK_LEN)`` con concatenazione degli
    indici 0, FRAME_STRIDE_K, 2*FRAME_STRIDE_K.
    """

    def __init__(self, init_frame: np.ndarray):
        self._dq = deque([init_frame] * STACK_LEN, maxlen=STACK_LEN)

    def append(self, frame: np.ndarray) -> None:
        self._dq.append(frame)

    @property
    def stacked(self) -> np.ndarray:
        return np.concatenate([self._dq[0], self._dq[FRAME_STRIDE_K], self._dq[2 * FRAME_STRIDE_K]])
