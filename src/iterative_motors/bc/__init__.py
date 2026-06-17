"""Behavioral Cloning: data augmentation e addestramento supervisionato.

  - ``augmentation`` : data augmentation Bojarski-style (``AugmentConfig`` + ``augment_batch``)
                       con clamp on-track e perturbazione angolare configurabili.
  - ``train_bc``     : entrypoint di addestramento (classe ``BehaviorCloningTrainer`` + ``main``),
                       lanciabile con ``python -m iterative_motors.bc.train_bc`` o via ``run.sh``.

Si ri-esporta la sola API dell'augmentation; l'entrypoint di training non viene importato qui
per non caricare l'intera pipeline di dati all'import del package.
"""

from .augmentation import AugmentConfig, augment_batch

__all__ = ["AugmentConfig", "augment_batch"]
