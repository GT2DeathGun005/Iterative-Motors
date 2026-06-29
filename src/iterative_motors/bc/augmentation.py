"""Bojarski-style data augmentation for Behavioral Cloning.

Simulates lateral/angular viewpoint shifts on the 29D states (for each of the 3 stacked frames)
and corrects the targets to teach recovery toward the track center. All parameters are in
``AugmentConfig`` (CLI/config), so they can be tuned.

``on_track_limit`` (on-track clamp): if set, the perturbed trackPos never exceeds the edge
(|trackPos| <= limit), so the network learns recovery toward the center from poses *still on
track*, never to drive off track. With ``None`` the clamp is disabled (historical behaviour).
"""

import math
from dataclasses import dataclass

import torch

from ..common.constants import SENSOR_ANGLES_DEG, STATE_DIM

# Angles of the 19 rays in radians (consistent with the SCR client), for the geometric perturbation.
_ALPHA_RAD = tuple(math.radians(a) for a in SENSOR_ANGLES_DEG)
_DEG45_RAD = math.radians(45.0)


@dataclass
class AugmentConfig:
    """Bojarski-style data-augmentation parameters.

    Updated defaults (Iterative Motors): on-track clamp active and wider angular perturbation
    (~10°). Compared to the historical version (angle σ 0.04 / clip 0.08 ≈ 4.5°, no clamp) this
    (a) avoids teaching to drive off track — the perturbed trackPos stays within |0.95| — and
    (b) widens recovery from realistic mid-corner misalignments.
    """
    pos_sigma: float = 0.22            # lateral perturbation std (trackPos)
    pos_clip: float = 0.45             # clip of the lateral perturbation
    angle_sigma: float = 0.09          # angular perturbation std (rad)
    angle_clip: float = 0.175          # clip of the angular perturbation (rad) ~10°
    aug_prob: float = 0.55            # fraction of samples to which the augmentation is applied
    w_half_min: float = 4.0           # minimum track half-width (m)
    w_half_max: float = 10.0          # maximum track half-width (m)
    dy_scale: float = 0.5             # scale of the physical lateral shift
    steer_corr_pos_gain: float = 0.30  # steering-correction gain for the lateral shift
    steer_corr_angle_gain: float = 1.6  # steering-correction gain for the angular misalignment
    accel_reduce_gain: float = 0.15    # throttle reduction proportional to the perturbation
    angle_combine_weight: float = 5.0  # weight of the angle in the combined perturbation
    on_track_limit: float = 0.95       # on-track clamp of the perturbed trackPos (None = off)
    # Overspeed recovery (preventive braking entering a high-speed corner)
    overspeed_prob: float = 0.5
    overspeed_speed_kmh: float = 90.0
    overspeed_steer_thr: float = 0.10
    overspeed_front_thr: float = 0.60
    overspeed_factor_min: float = 0.10
    overspeed_factor_range: float = 0.20
    overspeed_accel_reduce: float = 0.7
    overspeed_brake_add: float = 0.8


def augment_batch(states, targets, cfg: AugmentConfig):
    """Applies the augmentation to a batch of states (B,3,29) and targets (B,3). Modifies in-place.

    Returns (states, targets). The state is raw-scaled (not z-scored): normalization must be applied
    AFTER this function.
    """
    batch_size = states.size(0)
    device = states.device

    # Lateral and angular perturbations, with gating (aug_prob fraction of the samples).
    delta_pos = torch.clamp(torch.randn(batch_size, device=device) * cfg.pos_sigma, -cfg.pos_clip, cfg.pos_clip)
    delta_angle = torch.clamp(torch.randn(batch_size, device=device) * cfg.angle_sigma, -cfg.angle_clip, cfg.angle_clip)
    aug_mask = (torch.rand(batch_size, device=device) < cfg.aug_prob).float()
    delta_pos = delta_pos * aug_mask
    delta_angle = delta_angle * aug_mask

    alpha = torch.tensor(_ALPHA_RAD, device=device)

    # Effective lateral shift (clamped on-track if requested); the last frame (t) provides the
    # value used for the target correction, consistent with the current pose.
    dpos_eff_last = delta_pos
    for f_idx in range(3):
        frame_states = states[:, f_idx, :]

        angle = frame_states[:, 0]
        L_0 = frame_states[:, 1] * 200.0    # sensor -45°
        L_18 = frame_states[:, 19] * 200.0  # sensor +45°
        W_L = L_18 * torch.sin(angle + _DEG45_RAD)
        W_R = L_0 * torch.sin(_DEG45_RAD - angle)
        W_half = torch.clamp((W_L + W_R) / 2.0, cfg.w_half_min, cfg.w_half_max)

        orig_pos = frame_states[:, 20]
        if cfg.on_track_limit is not None:
            # On-track clamp: the perturbed trackPos stays within the edges; the effective
            # perturbation is the realized difference (it never pushes beyond the limit).
            target_pos = torch.clamp(orig_pos + delta_pos, -cfg.on_track_limit, cfg.on_track_limit)
            dpos_eff = target_pos - orig_pos
        else:
            dpos_eff = delta_pos
            target_pos = orig_pos + delta_pos
        dpos_eff_last = dpos_eff

        dy = dpos_eff * W_half * cfg.dy_scale
        frame_states[:, 20] = target_pos
        frame_states[:, 0] = frame_states[:, 0] + delta_angle

        # Geometrically consistent perturbation of the 19 rays (lateral + angular).
        perturbed_angle = frame_states[:, 0]
        beta = perturbed_angle.unsqueeze(1) + alpha.unsqueeze(0)
        dL = -dy.unsqueeze(1) * torch.sin(beta)
        frame_states[:, 1:20] = torch.clamp(frame_states[:, 1:20] + dL / 200.0, 0.0, 1.0)

    # Steering target correction (recovery toward the center + angular realignment).
    targets[:, 0] = targets[:, 0] - cfg.steer_corr_pos_gain * dpos_eff_last - cfg.steer_corr_angle_gain * delta_angle
    targets[:, 0] = torch.clamp(targets[:, 0], -1.0, 1.0)

    # Throttle partialization as a function of the perturbation magnitude.
    combined_perturbation = dpos_eff_last.abs() + delta_angle.abs() * cfg.angle_combine_weight
    targets[:, 1] = targets[:, 1] * (1.0 - cfg.accel_reduce_gain * combined_perturbation)
    targets[:, 1] = torch.clamp(targets[:, 1], 0.0, 1.0)

    # Overspeed recovery: high speed entering a corner -> less throttle, more brake.
    if torch.rand(1).item() < cfg.overspeed_prob:
        speedX_latest = states[:, 2, 21] * 50.0
        steer_target_abs = targets[:, 0].abs()
        sensor_front_latest = states[:, 2, 10]  # ray at 0°
        is_speed_critical = (speedX_latest > cfg.overspeed_speed_kmh) & (
            (steer_target_abs > cfg.overspeed_steer_thr) | (sensor_front_latest < cfg.overspeed_front_thr)
        )
        if is_speed_critical.any():
            speed_factor = cfg.overspeed_factor_min + cfg.overspeed_factor_range * torch.rand(batch_size, device=device)
            speed_factor = speed_factor * is_speed_critical.float()
            for f_idx in range(3):
                states[:, f_idx, 21] = states[:, f_idx, 21] * (1.0 + speed_factor)
            targets[:, 1] = torch.clamp(targets[:, 1] * (1.0 - cfg.overspeed_accel_reduce * speed_factor), 0.0, 1.0)
            targets[:, 2] = torch.clamp(targets[:, 2] + cfg.overspeed_brake_add * speed_factor, 0.0, 1.0)

    return states, targets
