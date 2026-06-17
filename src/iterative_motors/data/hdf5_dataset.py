"""Dataset HDF5 per la Behavioral Cloning, con frame stacking temporale.

Carica i giri umani (``lap_[0-9]*.h5``, esclusi i segmenti ``lap_seg_*``) e, in modo
opzionale, i giri auto-raccolti dalla TD3 (``extra_dirs``, es. ``train_set/laps_auto``)
per il riaddestramento arricchito della BC.
"""

import os
import glob

import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, ConcatDataset

from ..common.constants import FRAME_STRIDE_K


class TorcsHDF5Dataset(Dataset):
    """Dataset PyTorch su un singolo file HDF5 di un giro, con frame stacking temporale.

    Caratteristiche:
      - Frame stacking di 3 frame distanziati di ``FRAME_STRIDE_K`` (=6, cioè 0.12s a 50Hz):
        ``__getitem__`` restituisce la concatenazione degli stati ai tempi (t-12, t-6, t), portando
        lo stato da 29D a 87D. Questo rende il modello consapevole della dinamica della vettura
        (velocità e accelerazione implicite), aiutandolo a prevedere la traiettoria futura.
      - Sanity check all'inizializzazione: verifica la presenza dei dataset ``states`` e ``actions``
        e l'assenza di valori NaN/Inf. È deliberato: dati non validi nel training provocherebbero
        instabilità o collasso della policy, quindi è meglio fallire subito con un errore chiaro.

    Gli stati su disco sono grezzi (raw-scaled, NON z-scored): la normalizzazione viene applicata a
    valle dal trainer dopo l'eventuale data augmentation.
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
    """Carica i giri completi da ``path`` (e da ``extra_dirs``) in un unico dataset concatenato.

    - ``path``: directory dei giri umani (glob ``lap_[0-9]*.h5``, esclude i segmenti) o un
      singolo file ``.h5``.
    - ``extra_dirs``: lista opzionale di directory aggiuntive (es. ``train_set/laps_auto``)
      da cui caricare i giri ``lap_*.h5`` per l'arricchimento del dataset BC.

    Ritorna (dataset, num_campioni_totali).
    """
    if os.path.isdir(path):
        h5_files = sorted(glob.glob(os.path.join(path, "**/lap_[0-9]*.h5"), recursive=True))
        if not h5_files:
            raise FileNotFoundError(
                f"Nessun file lap_[0-9]*.h5 (giro intero) trovato in {path} o nelle sue sottocartelle"
            )

        # Giri aggiuntivi auto-raccolti (es. laps_auto/lap_auto_*.h5 e lap_*.h5).
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
