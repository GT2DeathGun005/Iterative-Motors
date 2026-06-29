"""Iterative Motors shared neural networks: backbone, Actor (RL), PolicyNetwork (BC), Critic.

This module centralizes ALL the project's neural architectures, which used to be duplicated
across three different files (BC training, TD3+BC training, evaluation). Having a single
definition removes the risk of misalignment between the three phases and guarantees that weights
trained in one phase are loadable in the others.

All networks producing driving commands share the same "body" (backbone) and the same continuous
head: only the output activation function changes.

------------------------------------------------------------------------------------------
state_dict KEY CONTRACT — DO NOT MODIFY WITHOUT MIGRATING THE CHECKPOINTS
------------------------------------------------------------------------------------------
The historical checkpoints saved in ``train_set/checkpoints/`` (the "untouchable archive" of
records) must keep loading without conversions. For this reason the backbone and the head are
built EXACTLY as in the original code, producing the following keys:

    Actor / PolicyNetwork (same structure):
        backbone.0.{weight,bias}    backbone.1.{weight,bias}      # Linear(87,512) + LayerNorm
        backbone.3.{weight,bias}    backbone.4.{weight,bias}      # Linear(512,512) + LayerNorm
        backbone.6.{weight,bias}    backbone.7.{weight,bias}      # Linear(512,512) + LayerNorm
        backbone.9.{weight,bias}    backbone.10.{weight,bias}     # Linear(512,512) + LayerNorm
        continuous_head.{weight,bias}                              # Linear(512,3)
    Critic:
        q1.0/2/4.{weight,bias}      q2.0/2/4.{weight,bias}

The "missing" backbone indices (2, 5, 8, 11) are the ReLU layers, which have no parameters and
are therefore absent from the state_dict. Since Actor and PolicyNetwork have the same keys, the BC
and TD3+BC weights are interchangeable: only the output activations applied at runtime differ
(see below).
"""

import os

import torch
import torch.nn as nn

from ..common.constants import STACK_DIM


def make_backbone(state_dim: int = STACK_DIM, hidden_size: int = 512) -> nn.Sequential:
    """Builds the shared backbone: 4 blocks of {Linear -> LayerNorm -> ReLU}.

    The depth (4 layers) and width (512 neurons) were chosen based on the offline-RL literature
    (Fujimoto & Gu 2021; Beeson & Montana 2022): enough capacity to capture non-linear relations
    among the 19 sensors, the speed and the car attitude, without being so much as to memorize the
    dataset (overfitting).

    The Layer Normalization after each Linear stabilizes training by keeping mean 0 and variance 1
    per feature, preventing some dimensions from dominating the gradient; the ReLU introduces
    non-linearity by activating only for positive inputs.

    Args:
        state_dim: input size (default 87 = 3 stacked frames of 29 sensors).
        hidden_size: neurons per hidden layer (default 512).
    Returns:
        An ``nn.Sequential`` with keys ``backbone.{0,1,3,4,6,7,9,10}`` (see the contract above).
    """
    return nn.Sequential(
        nn.Linear(state_dim, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU(),
        nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU(),
        nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU(),
        nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU(),
    )


def make_q_net(state_dim: int = STACK_DIM, action_dim: int = 3, hidden_size: int = 512) -> nn.Sequential:
    """Builds a single Critic Q-network: {Linear -> ReLU -> Linear -> ReLU -> Linear(.,1)}.

    It takes the concatenation of state (87D) and action (3D) and produces a single scalar, the
    estimate of the expected value Q(s, a). The Critic uses two independent ones (Twin Q) to reduce
    value overestimation (see ``Critic``).
    """
    return nn.Sequential(
        nn.Linear(state_dim + action_dim, hidden_size), nn.ReLU(),
        nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1),
    )


class Actor(nn.Module):
    """Deterministic policy of the RL agent (TD3+BC): from 87D state to 3D continuous commands.

    The input is the concatenated 87-dimensional state (3 temporal stacks of 29 sensors at t-12,
    t-6, t). The backbone extracts the driving features; the continuous head produces 3 outputs
    brought into [-1, 1] by a Tanh:

        - output 0: steering (negative = left, positive = right)
        - output 1: throttle (remapped to [0, 1] by the downstream action mapping)
        - output 2: brake    (remapped to [0, 1] by the downstream action mapping)

    Unlike the BC ``PolicyNetwork``, here throttle and brake also use Tanh: it is the representation
    the Critic learned to estimate Q on, and the mapping to pedals [0, 1] happens only when sending
    the command to TORCS (see ``action_mapping.rl_to_pedals``).
    """

    def __init__(self, state_dim: int = STACK_DIM, hidden_size: int = 512):
        super(Actor, self).__init__()
        self.backbone = make_backbone(state_dim, hidden_size)
        self.continuous_head = nn.Linear(hidden_size, 3)  # steer, accel, brake

    def forward(self, state):
        """Raw deterministic action (no noise): Tanh applied to all 3 channels.

        It is the π(s) used both in evaluation and, internally, in the TD3+BC Actor-loss
        computation (maximization of Q(s, π(s))).
        """
        features = self.backbone(state)
        mean = self.continuous_head(features)
        return torch.tanh(mean)

    def sample(self, state, evaluate=False, noise_std=0.1):
        """Action to execute on the environment, deterministic or with exploration noise.

        - ``evaluate=True``: returns the pure deterministic action produced by the network, for
          stable and reproducible driving (evaluation/submission).
        - ``evaluate=False``: adds Gaussian noise (standard deviation ``noise_std``, clipped to
          [-2σ, +2σ]) for off-policy exploration. σ is annealed by the training loop (from
          EXPL_NOISE_START to EXPL_NOISE_END, or to the time-attack floor) because at steady state
          micro-variations of the trajectory are needed, not slides. The final action is saturated
          to [-1, 1].
        """
        action = self.forward(state)
        if not evaluate:
            noise = torch.randn_like(action) * noise_std
            noise = torch.clamp(noise, -2.0 * noise_std, 2.0 * noise_std)
            action = torch.clamp(action + noise, -1.0, 1.0)
        return action

    def load_bc_weights(self, bc_path):
        """Initialize the Actor (warm-start) from the BC pre-trained weights.

        It is the bridge between the two phases: the TD3+BC Actor inherits the backbone and the
        continuous head from the BC so it starts "already knowing how to drive". Since the BC uses
        the Sigmoid [0, 1] for throttle/brake while the Actor uses Tanh [-1, 1], the weights and
        biases of channels 1:3 of the head are multiplied by 0.5: it is the linear transformation
        y = 0.5·x that preserves the order of magnitude of the initial outputs in the new range.
        ``strict=False`` ignores any extra keys (e.g. an old ``gear_head`` no longer used).
        """
        if not os.path.exists(bc_path):
            return
        bc_state = torch.load(bc_path, map_location='cpu', weights_only=True)
        if 'continuous_head.weight' in bc_state:
            bc_state['continuous_head.weight'][1:3] = bc_state['continuous_head.weight'][1:3] * 0.5
        if 'continuous_head.bias' in bc_state:
            bc_state['continuous_head.bias'][1:3] = bc_state['continuous_head.bias'][1:3] * 0.5
        self.load_state_dict(bc_state, strict=False)
        print(f"Pesi BC caricati con successo da {bc_path} (compensato scaling 0.5 per accel/brake).")

    def load_actor_weights(self, path, device):
        """Load the Actor weights from a file (full checkpoint or Actor-only).

        Filters the keys keeping only the parameters present in the current model with a compatible
        shape: this makes it possible to load even files that contain other components (e.g. the
        Critic) or that come from architecture variants, ignoring the rest.
        """
        if not os.path.exists(path):
            return
        try:
            loaded = torch.load(path, map_location=device, weights_only=True)
        except Exception:
            loaded = torch.load(path, map_location=device, weights_only=False)
        state_dict = loaded.get('actor', loaded) if isinstance(loaded, dict) else loaded
        model_state = self.state_dict()
        filtered_state = {
            k: v for k, v in state_dict.items()
            if k in model_state and hasattr(v, 'shape') and model_state[k].shape == v.shape
        }
        self.load_state_dict(filtered_state, strict=False)


class PolicyNetwork(nn.Module):
    """Behavioral Cloning and evaluation-inference network.

    Exactly the same architecture as the ``Actor`` (hence the same state_dict keys, interchangeable
    weights), but with two output modes designed to cover both checkpoint types without having to
    instantiate different classes:

      - ``forward`` (BC activation): steering (channel 0) goes through a Tanh in [-1, 1], while
        throttle and brake (channels 1, 2) go through a Sigmoid in [0, 1]. It is the form the BC is
        trained with, because it matches exactly the codomain of the recorded human actions
        (steering in [-1, 1], pedals in [0, 1]).
      - ``sample`` (RL activation): Tanh on all 3 channels ([-1, 1]). It serves to evaluate a
        TD3+BC checkpoint with the same class used for the BC.

    In ``test_agent`` the choice between ``forward`` and ``sample`` depends on the type of weights
    loaded (BC or RL), detected from the file name or from the ``--kind`` argument.
    """

    def __init__(self, state_dim: int = STACK_DIM, hidden_size: int = 512):
        super(PolicyNetwork, self).__init__()
        self.backbone = make_backbone(state_dim, hidden_size)
        self.continuous_head = nn.Linear(hidden_size, 3)

    def forward(self, state):
        """BC activation: Tanh on steering ([-1, 1]), Sigmoid on throttle/brake ([0, 1])."""
        features = self.backbone(state)
        cont_out = self.continuous_head(features)
        steer = torch.tanh(cont_out[:, 0:1])
        accel_brake = torch.sigmoid(cont_out[:, 1:3])
        return torch.cat([steer, accel_brake], dim=1)

    def sample(self, state, evaluate: bool = False):
        """Deterministic RL activation: Tanh on all 3 channels (outputs in [-1, 1])."""
        features = self.backbone(state)
        mean = self.continuous_head(features)
        return torch.tanh(mean)


class Critic(nn.Module):
    """Twin Critic Network for estimating the value Q(s, a) in TD3+BC.

    Implements two independent Q-networks (Q1 and Q2), trained from scratch, that receive the
    concatenation of the 87D state and the 3D action. Using two distinct estimates counters the
    Overestimation Bias typical of actor-critic methods: the Bellman target uses the MINIMUM of Q1
    and Q2, yielding a more conservative and stable value estimate.
    """

    def __init__(self, state_dim: int = STACK_DIM, action_dim: int = 3, hidden_size: int = 512):
        super(Critic, self).__init__()
        self.q1 = make_q_net(state_dim, action_dim, hidden_size)
        self.q2 = make_q_net(state_dim, action_dim, hidden_size)

    def forward(self, state, action):
        """Returns the pair of estimates (Q1(s, a), Q2(s, a))."""
        xu = torch.cat([state, action], 1)
        return self.q1(xu), self.q2(xu)
