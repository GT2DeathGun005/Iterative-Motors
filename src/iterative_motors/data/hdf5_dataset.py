"""HDF5 dataset for Behavioral Cloning, with temporal frame stacking.

Loads the human laps (``lap_[0-9]*.h5``, excluding the ``lap_seg_*`` segments) and, optionally,
the laps self-recorded by the TD3 agent (``extra_dirs``, e.g. ``train_set/laps_auto``) for the
enriched BC retraining.
"""

import os
import glob

import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, ConcatDataset

from ..common.constants import FRAME_STRIDE_K


class TorcsHDF5Dataset(Dataset):
    """PyTorch dataset over a single lap HDF5 file, with temporal frame stacking.

    Features:
      - Frame stacking of 3 frames spaced ``FRAME_STRIDE_K`` apart (=6, i.e. 0.12s at 50Hz):
        ``__getitem__`` returns the concatenation of the states at times (t-12, t-6, t), bringing
        the state from 29D to 87D. This makes the model aware of the car dynamics (implicit speed
        and acceleration), helping it predict the future trajectory.
      - Sanity check at initialization: verifies the presence of the ``states`` and ``actions``
        datasets and the absence of NaN/Inf values. It is deliberate: invalid data in training would
        cause instability or policy collapse, so it is better to fail immediately with a clear error.

    The states on disk are raw (raw-scaled, NOT z-scored): normalization is applied downstream by
    the trainer after any data augmentation.
    """

    def __init__(self, file_path: str):
        super().__init__()
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File dataset non trovato: {file_path}")
        self.file_path = file_path

        with h5py.File(self.file_path, 'r') as h5f:
            if 'states' not in h5f:
                raise KeyError(f"Gruppo 'states' mancante in {file_path}")
            if 'actions' not in h5f:
                raise KeyError(f"Gruppo 'actions' mancante in {file_path}")

            states_np = h5f['states'][:]
            actions_np = h5f['actions'][:]

            if np.any(np.isnan(states_np)):
                raise ValueError(f"NaN rilevati in 'states' di {file_path}")
            if np.any(np.isinf(states_np)):
                raise ValueError(f"Inf rilevati in 'states' di {file_path}")
            if np.any(np.isnan(actions_np)):
                raise ValueError(f"NaN rilevati in 'actions' di {file_path}")
            if np.any(np.isinf(actions_np)):
                raise ValueError(f"Inf rilevati in 'actions' di {file_path}")

            self.states = torch.tensor(states_np, dtype=torch.float32)
            self.actions = torch.tensor(actions_np, dtype=torch.float32)

        self.length = self.states.shape[0]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        k = FRAME_STRIDE_K
        idx_t6 = max(0, idx - k)
        idx_t12 = max(0, idx - 2 * k)
        stacked = torch.cat([
            self.states[idx_t12],
            self.states[idx_t6],
            self.states[idx],
        ])
        return stacked, self.actions[idx]


def load_dataset(path: str, extra_dirs=None):
    """Loads the complete laps from ``path`` (and from ``extra_dirs``) into a single concatenated dataset.

    - ``path``: directory of the human laps (glob ``lap_[0-9]*.h5``, excludes the segments) or a
      single ``.h5`` file.
    - ``extra_dirs``: optional list of additional directories (e.g. ``train_set/laps_auto``) from
      which to load the ``lap_*.h5`` laps for BC dataset enrichment.

    Returns (dataset, total_num_samples).
    """
    if os.path.isdir(path):
        h5_files = sorted(glob.glob(os.path.join(path, "**/lap_[0-9]*.h5"), recursive=True))
        if not h5_files:
            raise FileNotFoundError(
                f"Nessun file lap_[0-9]*.h5 (giro intero) trovato in {path} o nelle sue sottocartelle"
            )

        # Additional self-recorded laps (e.g. laps_auto/lap_auto_*.h5 and lap_*.h5).
        for extra in (extra_dirs or []):
            if extra and os.path.isdir(extra):
                extra_files = sorted(glob.glob(os.path.join(extra, "**/lap_*.h5"), recursive=True))
                if extra_files:
                    print(f"  + {len(extra_files)} giri aggiuntivi da {extra}")
                    h5_files.extend(extra_files)

        print(f"Trovati {len(h5_files)} file HDF5. Carico l'intero dataset...")

        datasets = []
        total_samples = 0
        for f in h5_files:
            try:
                ds = TorcsHDF5Dataset(f)
                datasets.append(ds)
                total_samples += len(ds)
            except Exception as e:
                print(f"Warning: Impossibile leggere {f}: {e}")

        if not datasets:
            raise ValueError("Nessun dataset valido trovato.")
        print(f"Dataset caricato: {len(datasets)} giri, {total_samples} campioni totali.")
        return ConcatDataset(datasets), total_samples

    ds = TorcsHDF5Dataset(path)
    return ds, len(ds)
