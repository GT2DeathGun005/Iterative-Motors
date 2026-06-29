"""Action mapping between the network space and TORCS pedals.

The RL network outputs [-1,1] on all channels (Tanh); the BC outputs steering in [-1,1] and
throttle/brake in [0,1] (Sigmoid). TORCS wants steering [-1,1] and pedals [0,1]. In addition,
throttle and brake must not be pressed together (mutual exclusion).
"""

import numpy as np


def rl_to_pedals(cont_action) -> np.ndarray:
    """RL action ([-1,1]^3) -> [steer, accel, brake, gear=0]; throttle/brake mapped to [0,1]."""
    env_action = np.zeros(4, dtype=np.float32)
    env_action[0] = np.clip(cont_action[0], -1.0, 1.0)
    env_action[1] = np.clip((cont_action[1] + 1.0) / 2.0, 0.0, 1.0)
    env_action[2] = np.clip((cont_action[2] + 1.0) / 2.0, 0.0, 1.0)
    return env_action


def bc_to_pedals(cont_action) -> np.ndarray:
    """BC action ([steer in [-1,1], accel/brake in [0,1]]) -> [steer, accel, brake, gear=0]."""
    env_action = np.zeros(4, dtype=np.float32)
    env_action[0] = np.clip(cont_action[0], -1.0, 1.0)
    env_action[1] = np.clip(cont_action[1], 0.0, 1.0)
    env_action[2] = np.clip(cont_action[2], 0.0, 1.0)
    return env_action


def apply_mutual_exclusion(accel: float, brake: float) -> float:
    """Continuously reduces the throttle as a function of the brake: ``accel * (1 - brake)``.

    Prevents simultaneous throttle+brake (stalls) while keeping a smooth transition.
    """
    return accel * (1.0 - brake)
