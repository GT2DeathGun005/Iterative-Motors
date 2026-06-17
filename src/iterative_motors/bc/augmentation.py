"""Data augmentation Bojarski-style per la Behavioral Cloning.

Simula viewpoint shift laterali/angolari sugli stati 29D (per ciascuno dei 3 frame
impilati) e corregge i target per insegnare il rientro verso il centro pista. Tutti i
parametri sono in ``AugmentConfig`` (CLI/config), così da poterli tarare.

``on_track_limit`` (clamp on-track): se impostato, il trackPos perturbato non supera mai
il bordo (|trackPos| <= limite), quindi la rete impara il recupero verso il centro da pose
*ancora in pista*, mai a guidare fuori pista. Con ``None`` il clamp è disattivato
(comportamento storico).
"""

import math
from dataclasses import dataclass

import torch

from ..common.constants import SENSOR_ANGLES_DEG, STATE_DIM

# Angoli dei 19 raggi in radianti (coerenti col client SCR), per la perturbazione geometrica.
_ALPHA_RAD = tuple(math.radians(a) for a in SENSOR_ANGLES_DEG)
_DEG45_RAD = math.radians(45.0)


@dataclass
class AugmentConfig:
    """Parametri della data augmentation Bojarski-style.

    Default aggiornati (Iterative Motors): clamp on-track attivo e perturbazione angolare
    più ampia (~10°). Rispetto allo storico (σ angolo 0.04 / clip 0.08 ≈ 4.5°, nessun clamp)
    questo (a) evita di insegnare a guidare fuori pista — il trackPos perturbato resta entro
    |0.95| — e (b) amplia il recupero da disallineamenti realistici di metà curva.
    """
    pos_sigma: float = 0.22            # std perturbazione laterale (trackPos)
    pos_clip: float = 0.45             # clip della perturbazione laterale
    angle_sigma: float = 0.09          # std perturbazione angolare (rad)
    angle_clip: float = 0.175          # clip della perturbazione angolare (rad) ~10°
    aug_prob: float = 0.55            # frazione di campioni a cui applicare l'augmentation
    w_half_min: float = 4.0           # semi-larghezza pista minima (m)
    w_half_max: float = 10.0          # semi-larghezza pista massima (m)
    dy_scale: float = 0.5             # scala dello spostamento laterale fisico
    steer_corr_pos_gain: float = 0.30  # guadagno correzione sterzo per spostamento laterale
    steer_corr_angle_gain: float = 1.6  # guadagno correzione sterzo per disallineamento angolare
    accel_reduce_gain: float = 0.15    # riduzione acceleratore proporzionale alla perturbazione
    angle_combine_weight: float = 5.0  # peso dell'angolo nella perturbazione combinata
    on_track_limit: float = 0.95       # clamp on-track del trackPos perturbato (None = off)
    # Overspeed recovery (frenata preventiva in ingresso curva ad alta velocità)
    overspeed_prob: float = 0.5
    overspeed_speed_kmh: float = 90.0
    overspeed_steer_thr: float = 0.10
    overspeed_front_thr: float = 0.60
    overspeed_factor_min: float = 0.10
    overspeed_factor_range: float = 0.20
    overspeed_accel_reduce: float = 0.7
    overspeed_brake_add: float = 0.8


def augment_batch(states, targets, cfg: AugmentConfig):
    """Applica l'augmentation a un batch di stati (B,3,29) e ai target (B,3). Modifica in-place.

    Ritorna (states, targets). Lo stato è raw-scaled (non z-scored): la normalizzazione va
    applicata DOPO questa funzione.
    """
    batch_size = states.size(0)
    device = states.device

    # Perturbazioni laterale e angolare, con gating (frazione aug_prob dei campioni).
    delta_pos = torch.clamp(torch.randn(batch_size, device=device) * cfg.pos_sigma, -cfg.pos_clip, cfg.pos_clip)
    delta_angle = torch.clamp(torch.randn(batch_size, device=device) * cfg.angle_sigma, -cfg.angle_clip, cfg.angle_clip)
    aug_mask = (torch.rand(batch_size, device=device) < cfg.aug_prob).float()
    delta_pos = delta_pos * aug_mask
    delta_angle = delta_angle * aug_mask

    alpha = torch.tensor(_ALPHA_RAD, device=device)

    # Spostamento laterale effettivo (clampato on-track se richiesto); l'ultimo frame (t)
    # fornisce il valore usato per la correzione dei target, coerente con la posa corrente.
    dpos_eff_last = delta_pos
    for f_idx in range(3):
        frame_states = states[:, f_idx, :]

        angle = frame_states[:, 0]
        L_0 = frame_states[:, 1] * 200.0    # sensore -45°
        L_18 = frame_states[:, 19] * 200.0  # sensore +45°
        W_L = L_18 * torch.sin(angle + _DEG45_RAD)
        W_R = L_0 * torch.sin(_DEG45_RAD - angle)
        W_half = torch.clamp((W_L + W_R) / 2.0, cfg.w_half_min, cfg.w_half_max)

        orig_pos = frame_states[:, 20]
        if cfg.on_track_limit is not None:
            # Clamp on-track: il trackPos perturbato resta entro i bordi; la perturbazione
            # effettiva è la differenza realizzata (non spinge mai oltre il limite).
            target_pos = torch.clamp(orig_pos + delta_pos, -cfg.on_track_limit, cfg.on_track_limit)
            dpos_eff = target_pos - orig_pos
        else:
            dpos_eff = delta_pos
            target_pos = orig_pos + delta_pos
        dpos_eff_last = dpos_eff

        dy = dpos_eff * W_half * cfg.dy_scale
        frame_states[:, 20] = target_pos
        frame_states[:, 0] = frame_states[:, 0] + delta_angle

        # Perturbazione geometricamente coerente dei 19 raggi (laterale + angolare).
        perturbed_angle = frame_states[:, 0]
        beta = perturbed_angle.unsqueeze(1) + alpha.unsqueeze(0)
        dL = -dy.unsqueeze(1) * torch.sin(beta)
        frame_states[:, 1:20] = torch.clamp(frame_states[:, 1:20] + dL / 200.0, 0.0, 1.0)

    # Correzione del target di sterzo (rientro verso il centro + riallineamento angolare).
    targets[:, 0] = targets[:, 0] - cfg.steer_corr_pos_gain * dpos_eff_last - cfg.steer_corr_angle_gain * delta_angle
    targets[:, 0] = torch.clamp(targets[:, 0], -1.0, 1.0)

    # Parzializzazione dell'acceleratore in funzione dell'entità della perturbazione.
    combined_perturbation = dpos_eff_last.abs() + delta_angle.abs() * cfg.angle_combine_weight
    targets[:, 1] = targets[:, 1] * (1.0 - cfg.accel_reduce_gain * combined_perturbation)
    targets[:, 1] = torch.clamp(targets[:, 1], 0.0, 1.0)

    # Overspeed recovery: alta velocità in ingresso curva -> meno gas, più freno.
    if torch.rand(1).item() < cfg.overspeed_prob:
        speedX_latest = states[:, 2, 21] * 50.0
        steer_target_abs = targets[:, 0].abs()
        sensor_front_latest = states[:, 2, 10]  # raggio a 0°
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
