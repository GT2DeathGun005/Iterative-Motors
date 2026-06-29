"""TD3+BC agent: Actor + Twin Critic, hybrid RL/BC update, checkpoint with robust resume.

Implements Twin Critic, Delayed Policy Update, Target Policy Smoothing and Polyak averaging
(Fujimoto et al. 2018) with the BC Penalty masked on expert samples only and the TD3+BC λ
normalization (Fujimoto & Gu 2021). The ``refine_mode``, ``actor_frozen`` and ``refine_bc_weight``
attributes are controlled by the training loop (curriculum/refinement).
"""

import os
import re
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from ..models.networks import Actor, Critic
from ..common.constants import TRACK_LENGTH_M
from ..common.checkpoint import safe_save, safe_save_npz, safe_read_float, _checkpoint_candidates
from .reward import _is_plausible_eval_dist, _is_plausible_eval_score

# Initial exploration noise (annealed by the training loop from 0.10 to 0.04).
_EXPL_NOISE_START = 0.10


class TD3BCAgent:
    """TD3+BC agent: coordinates the neural models and the whole optimization cycle.

    Implements TD3 (Fujimoto et al. 2018) with the Behavioral Cloning constraint of TD3+BC
    (Fujimoto & Gu 2021). It manages the interaction between Actor and Twin Critic through the steps:

      - Action selection, with or without Gaussian exploration noise (``select_action``).
      - Critic optimization by minimizing the temporal-difference error (TD error), i.e. the
        discrepancy between the current Q estimate and the Bellman target computed with the target nets.
      - Actor optimization by minimizing the hybrid loss ``-λ·Q(s,π(s)) + BC_penalty``, where the
        BC penalty is applied ONLY to expert samples (strict masking).
      - Stabilization via Polyak averaging (soft update of the target nets with rate τ).
      - Handling of the special curriculum states: Critic warm-up, temporary Actor freeze after a
        rollback, and refinement mode (reduced BC constraint, Critic stopped).

    The ``refine_mode``, ``actor_frozen`` and ``refine_bc_weight`` attributes are driven by the
    training loop (refinement/curriculum); here they have robust defaults.
    """

    def __init__(self, device="cuda"):
        self.device = torch.device(device)
        self.gamma = 0.99   # Temporal discount factor for computing the future Q value
        self.tau = 0.005    # Parameter for the Polyak soft update of the target nets
        self.policy_freq = 2  # Actor-vs-Critic update frequency (Delayed Policy Update)
        self.expl_noise = _EXPL_NOISE_START  # exploration noise std, annealed by the training loop
        self.bc_alpha = 2.5  # TD3+BC alpha: higher = more weight to RL relative to BC
        # Trust-region weight toward the buffer actions on non-expert samples (0 = off).
        # Set by the training loop via --trust_region; cures Actor collapse after re-activation.
        self.trust_region_weight = 0.0

        # Curriculum/refinement state (set by the training loop; robust defaults).
        self.refine_mode = False
        self.actor_frozen = False
        self.refine_bc_weight = 0.3

        # Actor initialization (online and target)
        self.actor = Actor().to(self.device)
        self.actor_target = Actor().to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Critic initialization (online and target)
        self.critic = Critic().to(self.device)
        self.critic_target = Critic().to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Adam optimizers
        actor_params = [p for p in self.actor.parameters() if p.requires_grad]
        self.actor_optimizer = optim.Adam(actor_params, lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)

    def select_action(self, state, evaluate=False):
        """Continuous 3D action for the current state (deterministic if evaluate=True)."""
        state_t = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            cont_action = self.actor.sample(state_t, evaluate=evaluate, noise_std=self.expl_noise)
        return cont_action.cpu().numpy()[0]

    def update(self, online_memory, elite_memory, expert_memory, batch_size, global_step):
        """Performs a single training step of the Critic and (possibly) the Actor.

        Steps:

          1. Samples a three-way hybrid batch: 25% expert (human driver), 15% elite (best autonomous
             runs) and 60% online (current exploration). If online or elite have little data, the
             missing quota is compensated with expert samples (always available).
          2. Rescales the reward (``reward_scale = 0.02``) to keep the Q values in a numerically
             stable range.
          3. Updates the Critic (Twin Q):
             - computes the next-state action with Target Policy Smoothing (clipped noise);
             - extracts Q1_target(s', a') and Q2_target(s', a') from the target nets;
             - takes the MINIMUM of the two (anti-overestimation) and forms the Bellman target
               ``r + γ·mask·min(Q1, Q2)``;
             - minimizes the MSE of the current estimates against the target (with gradient clipping).
             In refinement mode this update is DISABLED: the Actor refines toward a fixed value
             function.
          4. Updates the Actor (Delayed Policy Update, every ``policy_freq`` steps) only after the
             15000-step warm-up and if it is not frozen:
             - RL component: maximizes Q1(s, π(s));
             - BC penalty: MSE between predicted action and expert action, computed ONLY on samples
               with ``expert_mask > 0.5``, plus a throttle/brake mutual-exclusion penalty;
             - dynamic coefficient ``λ = bc_alpha / mean(|Q(s, π(s))|)`` that keeps the scale of the
               RL term and the BC term comparable (Fujimoto & Gu 2021);
             - total loss ``λ·(-Q) + bc_weight·BC_penalty`` (``bc_weight`` reduced in refinement).
          5. Soft update (Polyak, τ) of the target nets every ``policy_freq`` steps, REGARDLESS of
             warm-up and freeze (as in the original TD3): keeping it tied to the Actor update left the
             Critic targets stuck for tens of thousands of steps, making current estimates and targets
             diverge.

        Returns ``(critic_loss, actor_loss, 0.0)``; ``actor_loss`` is 0 if the Actor was not updated.
        """
        # Three-way Hybrid Sampling: Expert + Online + Elite
        b_expert = int(batch_size * 0.25)
        b_elite = min(int(batch_size * 0.15), len(elite_memory.buffer))
        b_online = min(batch_size - b_expert - b_elite, len(online_memory.buffer))
        b_expert = batch_size - b_online - b_elite  # the rest from expert (always available)

        parts = [expert_memory.sample(b_expert)]

        if b_online > 0:
            parts.append(online_memory.sample(b_online))
        if b_elite > 0:
            parts.append(elite_memory.sample(b_elite))

        state_b = np.concatenate([p[0] for p in parts], axis=0)
        action_b = np.concatenate([p[1] for p in parts], axis=0)
        reward_b = np.concatenate([p[2] for p in parts], axis=0)
        next_state_b = np.concatenate([p[3] for p in parts], axis=0)
        mask_b = np.concatenate([p[4] for p in parts], axis=0)
        expert_mask_b = np.concatenate([p[5] for p in parts], axis=0)

        # Reward rescaling to keep the Critic magnitude in a healthy range
        reward_scale = 0.02
        reward_b = reward_b * reward_scale

        state_b = torch.FloatTensor(state_b).to(self.device)
        next_state_b = torch.FloatTensor(next_state_b).to(self.device)
        action_b = torch.FloatTensor(action_b).to(self.device)
        reward_b = torch.FloatTensor(reward_b).to(self.device).unsqueeze(1)
        mask_b = torch.FloatTensor(mask_b).to(self.device).unsqueeze(1)
        expert_mask_b = torch.FloatTensor(expert_mask_b).to(self.device).unsqueeze(1)

        # Critic update (Bellman with Twin Q-Network)
        with torch.no_grad():
            noise = (torch.randn_like(action_b) * 0.2).clamp(-0.5, 0.5)  # Target Policy Smoothing
            next_action = self.actor_target(next_state_b)
            next_action = (next_action + noise).clamp(-1.0, 1.0)

            q1_next, q2_next = self.critic_target(next_state_b, next_action)
            min_q_next = torch.min(q1_next, q2_next)
            target_q = reward_b + mask_b * self.gamma * min_q_next

        q1, q2 = self.critic(state_b, action_b)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        # In refinement the Critic update is disabled (fixed value function, reduced BC constraint).
        if not getattr(self, 'refine_mode', False):
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.critic_optimizer.step()

        actor_loss_val = 0.0

        # Delayed Policy Update (every 2 Critic steps) after the 15000-step warm-up.
        if global_step >= 15000 and global_step % self.policy_freq == 0 and not getattr(self, 'actor_frozen', False):
            pi = self.actor(state_b)
            q1_pi, _ = self.critic(state_b, pi)

            actor_loss_td3 = -q1_pi.mean()  # RL component: maximizes Q1(s, pi(s))

            # BC Penalty (strict masking: Expert sub-batch only)
            det_steer = pi[:, 0]
            det_accel = (pi[:, 1] + 1.0) / 2.0
            det_brake = (pi[:, 2] + 1.0) / 2.0
            target_steer = action_b[:, 0]
            target_accel = (action_b[:, 1] + 1.0) / 2.0
            target_brake = (action_b[:, 2] + 1.0) / 2.0

            expert_mask_flat = expert_mask_b.squeeze(1)
            expert_indices = torch.where(expert_mask_flat > 0.5)[0]

            if len(expert_indices) > 0:
                steer_loss = F.mse_loss(det_steer[expert_indices], target_steer[expert_indices])
                accel_loss = F.mse_loss(det_accel[expert_indices], target_accel[expert_indices])
                brake_loss = F.mse_loss(det_brake[expert_indices], target_brake[expert_indices])
                bc_penalty = (steer_loss * 2.0 + accel_loss + brake_loss * 2.0)
            else:
                bc_penalty = torch.tensor(0.0, device=self.device)

            # Penalty to avoid throttle and brake pressed together
            mutual_exclusion_penalty = (det_accel * det_brake).mean()
            bc_penalty = bc_penalty + (mutual_exclusion_penalty * 0.1)

            # Trust region (FIXED weight, it does not melt with the dynamic anchor nor with refine):
            # it anchors the Actor to the actions actually present in the buffer on NON-expert samples
            # (75% of the batch). Without it, on that 75% the only force is max Q, which pushes the
            # Actor onto out-of-distribution actions where the Critic overestimates Q → the
            # deterministic policy collapses as soon as the Actor is active. Anchoring it to the data
            # support stabilizes it (it was the lineage's TRUST_REGION_WEIGHT that consolidated to
            # identical evals, lost in the refactor).
            trust_region_penalty = torch.tensor(0.0, device=self.device)
            if getattr(self, 'trust_region_weight', 0.0) > 0.0:
                non_expert_indices = torch.where(expert_mask_flat <= 0.5)[0]
                if len(non_expert_indices) > 0:
                    trust_region_penalty = F.mse_loss(
                        pi[non_expert_indices], action_b[non_expert_indices])

            # TD3+BC λ normalization (Fujimoto & Gu, 2021)
            Q_abs_mean = q1_pi.abs().mean().detach().clamp(min=1e-5)
            dynamic_alpha = self.bc_alpha / Q_abs_mean

            bc_weight = self.refine_bc_weight if getattr(self, 'refine_mode', False) else 1.0

            total_actor_loss = (dynamic_alpha * actor_loss_td3
                                + (bc_weight * bc_penalty)
                                + (self.trust_region_weight * trust_region_penalty))

            self.actor_optimizer.zero_grad()
            total_actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()
            actor_loss_val = total_actor_loss.item()

        # Soft Update (Polyak Averaging, τ) every policy_freq steps, regardless of warm-up/freeze.
        if global_step % self.policy_freq == 0:
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return critic_loss.item(), actor_loss_val, 0.0

    def save_checkpoint(self, filepath, episode, global_step, memory, elite_memory=None,
                        best_lap_time=float('inf'), best_eval_dist=0.0, best_distance=0.0):
        """Saves the full agent and replay-buffer state atomically.

        To guarantee integrity and avoid misalignments in case of a sudden stop:
          1. creates the ``buffers/`` folder next to the checkpoint;
          2. saves the Replay Buffers (online and elite) to ``.npz`` BEFORE the weights, since they
             are the most expensive I/O operation;
          3. builds the dictionary with Actor/Critic weights, target nets, optimizers, episode,
             global_step and record metrics;
          4. saves it to ``.pth`` with ``safe_save`` (temporary write, fsync, backup rotation).

        If the interruption happens midway, the absence of the updated ``.pth`` signals to resume that
        the new buffers are not aligned, so they will be ignored in favor of the consistent backups.
        """
        checkpoint = {
            'checkpoint_version': 2,
            'saved_at': datetime.now().isoformat(),
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'episode': episode,
            'global_step': global_step,
            'best_lap_time': best_lap_time,
            'best_eval_dist': best_eval_dist,
            'best_distance': best_distance,
        }
        buffer_dir = os.path.join(os.path.dirname(filepath), 'buffers')
        os.makedirs(buffer_dir, exist_ok=True)
        base_name = os.path.basename(filepath).replace('.pth', '')

        safe_save_npz(memory, os.path.join(buffer_dir, f"{base_name}_buffer.npz"))
        if elite_memory:
            safe_save_npz(elite_memory, os.path.join(buffer_dir, f"{base_name}_elite_buffer.npz"))
        safe_save(checkpoint, filepath)

    def load_checkpoint(self, filepath, memory, elite_memory=None):
        """Restores the agent and buffer state from a checkpoint, robustly (resume).

        Restore handling:
          1. scans the candidates (including ``backups/``) to find a readable ``.pth``;
          2. if the file contains the full training state, restores weights, target nets, optimizers
             and progress variables (episode, global_step, records);
          3. validates the stored records: if ``best_eval_dist`` (a score) or ``best_distance`` are
             implausible, recovers them from the text sidecars (``td3_det_best_dist.txt``);
          4. if the file contains ONLY the Actor weights (e.g. a checkpoint extracted for testing),
             performs a warm-start of the driving parameters only, zeroing optimizers and buffers;
          5. loads the Replay Buffers forcing temporal consistency: it does not load buffers with a
             timestamp later than the ``.pth`` (which would indicate a later interrupted save),
             falling back to the aligned backups; failing that, an emergency recovery from the newest.

        If it finds no checkpoint, it tries to recover the last episode from the training log.

        Returns ``(episode, global_step, best_lap_time, best_eval_dist, best_distance)``.
        """
        buffer_dir = os.path.join(os.path.dirname(filepath), 'buffers')
        base_name = os.path.basename(filepath).replace('.pth', '')
        buffer_path = os.path.join(buffer_dir, f"{base_name}_buffer.npz")
        elite_buffer_path = os.path.join(buffer_dir, f"{base_name}_elite_buffer.npz")

        def _load_buffer_aligned(buffer_obj, path, label, loaded_checkpoint_path=None):
            """Loads the most recent buffer not later than the .pth checkpoint (anti-misalignment)."""
            if buffer_obj is None:
                return False
            max_mtime = None
            if loaded_checkpoint_path and os.path.exists(loaded_checkpoint_path):
                max_mtime = os.path.getmtime(loaded_checkpoint_path)

            skipped_newer = []
            for candidate in _checkpoint_candidates(path):
                if not os.path.exists(candidate):
                    continue
                if max_mtime is not None and os.path.getmtime(candidate) > max_mtime + 1e-3:
                    skipped_newer.append(candidate)
                    continue
                try:
                    buffer_obj.load(candidate)
                    if candidate != path:
                        print(f"{label} recuperato dal backup coerente: {candidate}")
                    if skipped_newer:
                        print(f"{label}: ignorati file più nuovi del checkpoint caricato: {skipped_newer}")
                    return True
                except Exception as e:
                    print(f"Impossibile caricare {label} da {candidate}: {e}")

            # Emergency recovery: no intact aligned backup -> use the newest available
            for candidate in skipped_newer:
                try:
                    buffer_obj.load(candidate)
                    print(f"{label}: nessun backup allineato trovato; uso {candidate} (più nuovo del checkpoint).")
                    return True
                except Exception as e:
                    print(f"Impossibile caricare {label} da {candidate}: {e}")
            return False

        loaded_ok = False
        loaded_checkpoint_path = None
        for candidate in _checkpoint_candidates(filepath):
            if not os.path.exists(candidate):
                continue
            try:
                checkpoint = torch.load(candidate, map_location=self.device, weights_only=False)
                if isinstance(checkpoint, dict) and 'actor' in checkpoint:
                    required_keys = ['actor', 'critic', 'critic_target', 'actor_optimizer', 'critic_optimizer', 'episode', 'global_step']
                    missing_keys = [k for k in required_keys if k not in checkpoint]
                    if missing_keys:
                        raise KeyError(f"checkpoint incompleto, chiavi mancanti: {missing_keys}")
                    self.actor.load_state_dict(checkpoint['actor'], strict=False)
                    if 'actor_target' in checkpoint:
                        self.actor_target.load_state_dict(checkpoint['actor_target'], strict=False)
                    self.critic.load_state_dict(checkpoint['critic'])
                    self.critic_target.load_state_dict(checkpoint['critic_target'])
                    self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
                    self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])

                    best_lap_time = checkpoint.get('best_lap_time', float('inf'))
                    best_eval_dist = checkpoint.get('best_eval_dist', 0.0)
                    best_distance = checkpoint.get('best_distance', 0.0)
                    # best_eval_dist is a SCORE (distance or time-equivalent): score threshold.
                    if not _is_plausible_eval_score(best_eval_dist):
                        det_best_dist_txt = 'train_set/checkpoints/td3_det_best_dist.txt'
                        sidecar_best_eval_dist = safe_read_float(det_best_dist_txt, 0.0)
                        print(
                            f"best_eval_dist={best_eval_dist:.2f} non plausibile come score di eval; "
                            f"uso sidecar {sidecar_best_eval_dist:.2f}."
                        )
                        best_eval_dist = sidecar_best_eval_dist if _is_plausible_eval_score(sidecar_best_eval_dist) else 0.0
                    if not _is_plausible_eval_dist(best_distance):
                        best_distance = min(best_eval_dist, TRACK_LENGTH_M)
                    episode = checkpoint['episode']
                    global_step = checkpoint['global_step']
                    if candidate != filepath:
                        print(f"Checkpoint principale non usato: recupero da backup {candidate}")
                    print(f"Checkpoint caricato: ripresa dall'Episodio {episode}")
                else:
                    # Actor-weights-only file (e.g. td3_expl_best_dist.pth).
                    print(f"{candidate} contiene solo pesi dell'Actor. Inizializzazione degli altri componenti.")
                    self.actor.load_state_dict(checkpoint, strict=False)
                    self.actor_target.load_state_dict(self.actor.state_dict())
                    best_lap_time = float('inf')
                    best_eval_dist = 0.0
                    best_distance = 0.0
                    episode = 0
                    global_step = 0
                loaded_ok = True
                loaded_checkpoint_path = candidate
                break
            except Exception as e:
                print(f"Checkpoint non utilizzabile da {candidate}: {e}")

        if not loaded_ok:
            if not any(os.path.exists(candidate) for candidate in _checkpoint_candidates(filepath)):
                return 0, 0, float('inf'), 0.0, 0.0
            print("Nessun checkpoint completo valido trovato tra principale e backup recenti.")
            print("Tentativo di recupero minimo delle informazioni dal log...")
            log_file = 'train_set/session_logs/td3_training.log'
            last_ep = 0
            if os.path.exists(log_file):
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            match = re.search(r'\b(?:Episode|Episodio|Ep)\s+(\d+)', line)
                            if match:
                                ep_num = int(match.group(1))
                                last_ep = max(last_ep, ep_num)
                except Exception:
                    pass
            print(f"Ripristinato ultimo episodio: {last_ep}. Il training riprenderà dall'episodio {last_ep + 1}.")
            episode = last_ep + 1
            global_step = episode * 1500
            best_lap_time = float('inf')

            best_eval_dist = 0.0
            det_best_dist_txt = 'train_set/checkpoints/td3_det_best_dist.txt'
            best_eval_dist = safe_read_float(det_best_dist_txt, 0.0)
            best_distance = min(best_eval_dist, TRACK_LENGTH_M)

        _load_buffer_aligned(memory, buffer_path, "Replay Buffer", loaded_checkpoint_path if loaded_ok else None)
        if elite_memory:
            _load_buffer_aligned(elite_memory, elite_buffer_path, "Elite Buffer", loaded_checkpoint_path if loaded_ok else None)

        return episode, global_step, best_lap_time, best_eval_dist, best_distance
