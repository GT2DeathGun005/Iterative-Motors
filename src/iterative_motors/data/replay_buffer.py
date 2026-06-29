"""Replay buffer for off-policy TD3+BC training and loading of expert data.

Stores transitions (87D stacked state, 3D action, reward, next_state, done) with a parallel
``expert`` flag marking the human samples (=1.0) vs autonomous ones (=0.0), used to apply the
BC Penalty only to expert samples. ``load_expert_data`` reads the HDF5 files (human and/or
self-recorded), normalizes, applies temporal stacking and recomputes the reward with the same
formula as gym_torcs.
"""

import os
from collections import deque

import numpy as np

from ..common.state import apply_state_norm
from ..common.constants import FRAME_STRIDE_K


class ReplayBuffer:
    """Transition-storage buffer for off-policy training.

    Stores experiences as ``(state, action, reward, next_state, done)`` tuples in a FIFO circular
    queue: once capacity is exceeded, the oldest sample is dropped. In parallel it keeps an
    ``expert`` mask marking each transition as coming from the human driver (``1.0``) or collected
    autonomously by the agent (``0.0``). This marker is essential because the TD3+BC BC penalty is
    computed EXCLUSIVELY on expert samples, leaving the agent free to explore trajectories different
    from the human ones without being penalized.

    The project uses three distinct instances of this buffer (three-way hybrid sampling):
    expert (human data, permanent), elite (best autonomous runs) and online (current exploration).
    """

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        self.expert_masks = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, expert=0.0):
        """Inserts a transition; beyond capacity removes the oldest (FIFO)."""
        self.buffer.append((state, action, reward, next_state, done))
        self.expert_masks.append(expert)

    def sample(self, batch_size: int):
        """Draws a random batch: (state, action, reward, next_state, done, expert_mask)."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        expert_masks_batch = [self.expert_masks[i] for i in indices]
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done, np.array(expert_masks_batch, dtype=np.float32)

    def save(self, filepath: str):
        """Saves the buffer to a compressed .npz for training resume/restart."""
        if len(self.buffer) == 0:
            return
        states, actions, rewards, next_states, dones = zip(*self.buffer)
        np.savez_compressed(filepath,
            states=np.array(states, dtype=np.float32),
            actions=np.array(actions, dtype=np.float32),
            rewards=np.array(rewards, dtype=np.float32),
            next_states=np.array(next_states, dtype=np.float32),
            dones=np.array(dones, dtype=np.float32),
            expert_masks=np.array(list(self.expert_masks), dtype=np.float32))

    def load_expert_data(self, h5_dir_or_file, max_samples: int = None, max_lap_time: float = None):
        """Loads the HDF5 laps (human/auto), normalizes, stacks and recomputes the reward.

        - ``max_lap_time``: discards files with a ``lap_time`` attribute above the threshold (raises
          the BC anchor toward the best laps, not the average).
        - Temporal stacking t-12, t-6, t (stride ``FRAME_STRIDE_K``) for the 87D states.
        - Expert throttle/brake action mapped from [0,1] (Sigmoid) to [-1,1] (Tanh).
        - Reward recomputed with the gym_torcs formula; samples marked expert=1.0.

        Accepts a directory (loads ``lap_*.h5`` recursively) or a single file.
        """
        import glob
        import h5py

        if os.path.isdir(h5_dir_or_file):
            h5_files = sorted(glob.glob(os.path.join(h5_dir_or_file, "**/lap_*.h5"), recursive=True))
        else:
            h5_files = [h5_dir_or_file]

        loaded = 0
        skipped_slow = 0
        for f in h5_files:
            if max_samples and loaded >= max_samples:
                break
            try:
                with h5py.File(f, 'r') as h5f:
                    if max_lap_time is not None:
                        file_lap_time = h5f.attrs.get('lap_time', None)
                        if file_lap_time is not None and float(file_lap_time) > max_lap_time:
                            skipped_slow += 1
                            continue
                    states_np = h5f['states'][:]
                    actions_np = h5f['actions'][:]

                states_norm = apply_state_norm(states_np)
                length = len(states_np)
                k = FRAME_STRIDE_K

                for i in range(length - 1):
                    if max_samples and loaded >= max_samples:
                        break
                    idx_t6 = max(0, i - k)
                    idx_t12 = max(0, i - 2 * k)
                    next_i = i + 1
                    n_idx_t6 = max(0, next_i - k)
                    n_idx_t12 = max(0, next_i - 2 * k)

                    stacked_state = np.concatenate([states_norm[idx_t12], states_norm[idx_t6], states_norm[i]])
                    next_stacked_state = np.concatenate([states_norm[n_idx_t12], states_norm[n_idx_t6], states_norm[next_i]])

                    cont_action = actions_np[i, 0:3].copy()
                    cont_action[1] = (cont_action[1] * 2.0) - 1.0
                    cont_action[2] = (cont_action[2] * 2.0) - 1.0

                    speedX = states_np[i, 21] * 50.0
                    angle = states_np[i, 0]
                    trackPos = states_np[i, 20]

                    progress = (speedX / 50.0) * np.cos(angle)
                    tp = abs(trackPos)
                    pos_penalty = -2.0 * (max(0.0, tp - 1.0) ** 2)
                    steer_change = cont_action[0] - actions_np[i - 1, 0] if i > 0 else 0.0
                    reward = (progress * 1.5) + pos_penalty - (0.05 * abs(steer_change))

                    mask = 1.0

                    self.push(stacked_state, cont_action, reward, next_stacked_state, mask, expert=1.0)
                    loaded += 1
            except Exception as e:
                print(f"Errore caricando {f}: {e}")

        filtro_msg = ""
        if max_lap_time is not None:
            filtro_msg = f" (filtro lap_time <= {max_lap_time:.1f}s: scartati {skipped_slow} file più lenti)"
        print(f"  [EXPERT INJECTION] Caricati {loaded} campioni esperti nel Replay Buffer.{filtro_msg}")

    def load(self, filepath: str):
        """Loads transitions from a .npz into the in-memory buffer."""
        if not os.path.exists(filepath):
            return
        with np.load(filepath) as data:
            states = data['states']
            actions = data['actions']
            rewards = data['rewards']
            next_states = data['next_states']
            dones = data['dones']
            expert_masks_data = data['expert_masks'] if 'expert_masks' in data.files else np.zeros(len(states))
            states, actions, rewards, next_states, dones, expert_masks_data = [
                np.asarray(x) for x in (states, actions, rewards, next_states, dones, expert_masks_data)
            ]
        lengths = {len(states), len(actions), len(rewards), len(next_states), len(dones), len(expert_masks_data)}
        if len(lengths) != 1:
            raise ValueError(f"Replay Buffer non coerente in {filepath}: lunghezze diverse {sorted(lengths)}")

        new_buffer = deque(maxlen=self.buffer.maxlen)
        new_expert_masks = deque(maxlen=self.expert_masks.maxlen)
        for i in range(len(states)):
            new_buffer.append((states[i], actions[i], float(rewards[i]), next_states[i], float(dones[i])))
            new_expert_masks.append(float(expert_masks_data[i]))
        self.buffer = new_buffer
        self.expert_masks = new_expert_masks
        print(f"  Replay Buffer caricato: {len(self.buffer)} transizioni")

    def __len__(self):
        return len(self.buffer)
