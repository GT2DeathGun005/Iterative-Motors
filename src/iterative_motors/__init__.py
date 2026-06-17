"""Iterative Motors — guida autonoma su TORCS con Behavioral Cloning + TD3+BC.

Package modulare del progetto. La cartella root del repository si chiama ``AIcar``
ed è solo il contenitore: il progetto è *Iterative Motors*.

Sottopacchetti:
  - ``common``  costanti, normalizzazione/flatten dello stato, utility checkpoint, config.
  - ``env``     wrapper Gym di TORCS, client SCR (snakeoil), cambio marcia algoritmico.
  - ``models``  reti neurali condivise (Actor/Critic/PolicyNetwork) e mapping delle azioni.
  - ``data``    dataset HDF5, replay buffer, registratore dei giri auto-raccolti.
  - ``bc``      Behavioral Cloning (augmentation, trainer, entrypoint).
  - ``rl``      agente TD3+BC, fasi del curriculum, refinement, reward shaping, training.
  - ``eval``    valutazione deterministica e test dell'agente.
"""

__version__ = "1.0.0"
