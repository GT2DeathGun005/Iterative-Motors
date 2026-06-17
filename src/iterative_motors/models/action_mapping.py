"""Mapping delle azioni tra spazio della rete e pedali TORCS.

La rete RL produce [-1,1] su tutti i canali (Tanh); la BC produce sterzo in [-1,1] e
accel/freno in [0,1] (Sigmoid). TORCS vuole sterzo [-1,1] e pedali [0,1]. Inoltre
acceleratore e freno non devono essere premuti insieme (mutual exclusion).
"""

import numpy as np


def rl_to_pedals(cont_action) -> np.ndarray:
    """Azione RL ([-1,1]^3) -> [steer, accel, brake, gear=0]; accel/freno mappati a [0,1]."""
    env_action = np.zeros(4, dtype=np.float32)
    env_action[0] = np.clip(cont_action[0], -1.0, 1.0)
    env_action[1] = np.clip((cont_action[1] + 1.0) / 2.0, 0.0, 1.0)
    env_action[2] = np.clip((cont_action[2] + 1.0) / 2.0, 0.0, 1.0)
    return env_action


def bc_to_pedals(cont_action) -> np.ndarray:
    """Azione BC ([steer in [-1,1], accel/brake in [0,1]]) -> [steer, accel, brake, gear=0]."""
    env_action = np.zeros(4, dtype=np.float32)
    env_action[0] = np.clip(cont_action[0], -1.0, 1.0)
    env_action[1] = np.clip(cont_action[1], 0.0, 1.0)
    env_action[2] = np.clip(cont_action[2], 0.0, 1.0)
    return env_action


def apply_mutual_exclusion(accel: float, brake: float) -> float:
    """Riduce l'acceleratore in modo continuo in funzione del freno: ``accel * (1 - brake)``.

    Previene la pressione simultanea gas+freno (stalli) mantenendo una transizione morbida.
    """
    return accel * (1.0 - brake)
