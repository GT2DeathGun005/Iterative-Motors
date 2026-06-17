"""Dati: replay buffer dell'RL, dataset HDF5 della BC e registratore dei giri.

  - ``replay_buffer`` : ``ReplayBuffer`` (coda circolare di transizioni con maschera esperto) e
                        ``load_expert_data`` che carica i giri HDF5 nel buffer permanente con
                        normalizzazione e frame stacking.
  - ``hdf5_dataset``  : ``TorcsHDF5Dataset`` e ``load_dataset`` per il training della BC, con
                        supporto a directory aggiuntive (giri auto-raccolti) per l'arricchimento.
  - ``lap_recorder``  : ``LapRecorder`` che registra i giri completi/puliti guidati dall'agente
                        in TORCS e li salva in ``train_set/laps_auto/`` (flywheel dati BC↔RL).
  - ``collection``    : entrypoint di raccolta dei giri umani (controller PS5/tastiera), non
                        ri-esportato qui perché dipende da ``pygame`` e da una GUI.
"""

from .replay_buffer import ReplayBuffer
from .hdf5_dataset import TorcsHDF5Dataset, load_dataset
from .lap_recorder import LapRecorder

__all__ = ["ReplayBuffer", "TorcsHDF5Dataset", "load_dataset", "LapRecorder"]
