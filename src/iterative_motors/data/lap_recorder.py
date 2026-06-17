"""Registratore dei giri guidati dalla TD3 in esplorazione, per arricchire il dataset BC.

Cattura, per ogni step, lo stato grezzo 29D (``flatten_state_raw``, NON normalizzato come
gli HDF5 umani) e l'azione *realmente eseguita* su TORCS (``torcs_action`` =
[steer, accel applicato, brake, gear algoritmico]). A fine giro, se il giro è completo,
pulito (mai oltre ``on_track_limit`` di trackPos) e abbastanza veloce, scrive un file HDF5
nello stesso formato di ``data_collection`` dentro ``train_set/laps_auto/``.

Così i giri buoni dell'agente rientrano nel riaddestramento della BC (flywheel dati).
"""

import os
import glob
from datetime import datetime

import numpy as np
import h5py

from ..common.state import flatten_state_raw
from ..common.constants import STATE_DIM


def _scalar(obs, key, default=0.0):
    v = obs.get(key, default)
    if v is None:
        return default
    if isinstance(v, np.ndarray):
        return float(v.flat[0])
    return float(v)


class LapRecorder:
    """Accumula i giri dell'agente e ne salva solo quelli completi/puliti/veloci in HDF5."""

    def __init__(self, out_dir, max_lap_time=80.0, on_track_limit=1.0, min_steps=500, enabled=True):
        self.out_dir = out_dir
        self.max_lap_time = float(max_lap_time)
        self.on_track_limit = float(on_track_limit)
        self.min_steps = int(min_steps)
        self.enabled = bool(enabled)
        self.saved_count = 0
        if self.enabled:
            os.makedirs(self.out_dir, exist_ok=True)
        self._reset()

    def _reset(self):
        self._states = []
        self._actions = []
        self._dists = []
        self._max_abs_tp = 0.0

    def start_episode(self):
        """Azzera il buffer del giro corrente (da chiamare a ogni reset dell'ambiente)."""
        self._reset()

    def record_step(self, raw_obs, executed_action):
        """Registra uno step: stato grezzo (pre-step) + azione 4D eseguita su TORCS."""
        if not self.enabled:
            return
        self._states.append(flatten_state_raw(raw_obs))
        self._actions.append(np.asarray(executed_action, dtype=np.float32)[:4])
        self._dists.append(_scalar(raw_obs, 'distFromStart'))
        self._max_abs_tp = max(self._max_abs_tp, abs(_scalar(raw_obs, 'trackPos')))

    def discard(self):
        """Scarta il giro corrente (episodio non completato o sporco)."""
        self._reset()

    def set_gate(self, max_lap_time=None, on_track_limit=None):
        """Aggiorna dinamicamente la soglia di qualità (es. stringere col PB in time-attack)."""
        if max_lap_time is not None:
            self.max_lap_time = float(max_lap_time)
        if on_track_limit is not None:
            self.on_track_limit = float(on_track_limit)

    def _next_path(self):
        existing = glob.glob(os.path.join(self.out_dir, "lap_auto_*.h5"))
        idx = len(existing) + 1
        # Evita collisioni se la numerazione è frammentata.
        while os.path.exists(os.path.join(self.out_dir, f"lap_auto_{idx:04d}.h5")):
            idx += 1
        return os.path.join(self.out_dir, f"lap_auto_{idx:04d}.h5")

    def finish_lap(self, lap_time, phase="online", episode=-1, global_step=-1):
        """Valuta il quality gate e, se passato, salva il giro in HDF5. Ritorna True se salvato."""
        if not self.enabled:
            self._reset()
            return False
        n = len(self._states)
        ok = (
            n >= self.min_steps
            and lap_time is not None and 0.0 < float(lap_time) <= self.max_lap_time
            and self._max_abs_tp <= self.on_track_limit
        )
        if not ok:
            self._reset()
            return False

        states = np.asarray(self._states, dtype=np.float32)
        actions = np.asarray(self._actions, dtype=np.float32)
        dists = np.asarray(self._dists, dtype=np.float32)
        if states.shape[1] != STATE_DIM or not np.all(np.isfinite(states)) or not np.all(np.isfinite(actions)):
            self._reset()
            return False

        try:
            path = self._next_path()
            tmp = path + ".tmp"
            with h5py.File(tmp, 'w') as h5f:
                h5f.create_dataset('states', data=states, compression='gzip')
                h5f.create_dataset('actions', data=actions, compression='gzip')
                h5f.create_dataset('dist_from_start', data=dists, compression='gzip')
                h5f.attrs['lap_time'] = float(lap_time)
                h5f.attrs['num_steps'] = int(n)
                h5f.attrs['has_dist_meta'] = True
                h5f.attrs['timestamp'] = datetime.now().isoformat()
                h5f.attrs['source'] = 'td3_auto'
                h5f.attrs['phase'] = str(phase)
                h5f.attrs['episode'] = int(episode)
                h5f.attrs['global_step'] = int(global_step)
            os.replace(tmp, path)
            self.saved_count += 1
            print(f"  [LAP RECORDER] Salvato giro auto #{self.saved_count}: {os.path.basename(path)} "
                  f"(lap_time={lap_time:.3f}s, steps={n}, max|trackPos|={self._max_abs_tp:.2f})")
            saved = True
        except Exception as e:
            print(f"  [LAP RECORDER] Errore nel salvataggio del giro: {e}")
            saved = False

        self._reset()
        return saved
