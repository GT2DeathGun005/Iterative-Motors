"""Ambiente di simulazione: wrapper Gym di TORCS, client SCR e cambio marcia.

  - ``gym_torcs``     : wrapper Gym che avvia/gestisce TORCS, espone le osservazioni sensoriali,
                        applica le azioni, calcola la reward per-step e le condizioni di terminazione.
  - ``snakeoil3_gym`` : client UDP a basso livello del protocollo SCR (parsing telemetria, formato
                        azione, gestione socket). Definisce gli angoli dei 19 sensori track.
  - ``gearing``       : selezione algoritmica della marcia (con isteresi) — la rete NON la predice.

La regola IBM AI Racing League vieta di modificare la fisica/installazione di TORCS: questo
sottopacchetto è l'unico punto di contatto col simulatore e non ne altera la configurazione.

``TorcsEnv`` e ``compute_gear`` sono ri-esportati per comodità. NB: importare questo package
carica ``gym`` (il wrapper ne dipende).
"""

from .gym_torcs import TorcsEnv
from .gearing import compute_gear

__all__ = ["TorcsEnv", "compute_gear"]
