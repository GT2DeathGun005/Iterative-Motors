"""TD3+BC reward shaping: lap-end bonuses/penalties and evaluation score.

The per-step reward (progress + position penalty + anti-zigzag) lives in
``env/gym_torcs.py``; here are the terminal terms added by training (completion and
time-proportional bonus, incomplete-lap penalty) and the unified distance/time eval score
used by records and refinement.
"""

from ..common.constants import TRACK_LENGTH_M

# Terminal bonuses/penalties (added by the training loop on top of the gym_torcs reward).
LAP_SUCCESS_BONUS = 50.0
INCOMPLETE_LAP_PENALTY = 25.0

# Time-proportional lap bonus: +10 points for every second below 80s.
LAP_TIME_BONUS_REF_S = 80.0
LAP_TIME_BONUS_PER_S = 10.0

# Plausibility limits and unified distance/time evaluation score.
EVAL_DISTANCE_SANITY_LIMIT = 3800.0
EVAL_SCORE_T_REF_S = 90.0
EVAL_SCORE_SANITY_LIMIT = 6500.0  # corresponds to a lap < 50s, physically implausible

# ── Time-attack ───────────────────────────────────────────────────────────
# Bonus for beating one's OWN lap-time record (incentivizes shaving times even when the
# distance is saturated). Triggered when a new best lap is set.
PERSONAL_BEST_BONUS = 30.0       # fixed bonus for each new personal record
PERSONAL_BEST_PER_S = 15.0       # extra bonus for each second gained over the record
# TIME_ATTACK phase parameters (enabled via IM_TIME_ATTACK=1): less BC anchoring
# (higher alpha) and finer exploration noise for micro-optimization of the trajectory.
TIME_ATTACK_BC_ALPHA = 4.0
TIME_ATTACK_NOISE_FLOOR = 0.02
TIME_ATTACK_ENTRY_S = 70.5       # indicative "learned driving" threshold to promote to time-attack

# ── Stabilization: corridor penalty (margin from the track edge) ───────────
# The gym_torcs pos_penalty only kicks in at |trackPos| > 1.0, i.e. when the car is ALREADY
# off the driving surface: at that point a micro-perturbation sends it beyond 1.25 (crash).
# This penalty anticipates the signal, discouraging approaching the edge already from
# |trackPos| > MARGIN_PENALTY_START. It teaches a safety corridor → reliable complete laps
# (even if slower). In time-attack it is reduced (the full track width is needed).
MARGIN_PENALTY_START = 0.80      # |trackPos| beyond which the corridor penalty begins
MARGIN_PENALTY_COEF = 12.0       # quadratic coefficient (at |trackPos|=1.0 → ~-0.75/step)

# ── Time-attack: per-sector telemetry reward (split times) ─────────────────
# Defaults for the SectorTimer: number of sectors and the scale of the reward for beating
# one's own best sector time. Overridable via env (IM_TA_SECTORS/_K/_CAP).
TA_SECTORS_DEFAULT = 18
TA_SECTOR_REWARD_K = 30.0        # points per second gained over the sector record
TA_SECTOR_REWARD_CAP = 8.0       # clamp of the per-sector reward/penalty (≈0.27s)


def margin_penalty(track_pos, coef=MARGIN_PENALTY_COEF, start=MARGIN_PENALTY_START):
    """Quadratic corridor penalty: discourages approaching the edge BEFORE leaving it.

    Zero if ``|track_pos| <= start``; grows as ``-coef*(|track_pos|-start)^2`` toward the
    edge. Applied per-step to online transitions only (the agent's risky driving), it pushes
    the policy to keep a safety margin and to complete the lap repeatably.
    """
    tp = abs(float(track_pos))
    if tp <= start:
        return 0.0
    return -float(coef) * (tp - start) ** 2


def personal_best_bonus(prev_best_s, lap_time_s):
    """Bonus for a new personal record: fixed + proportional to the seconds gained.

    Returns 0 if ``lap_time_s`` does not improve ``prev_best_s`` (or if there is no previous best).
    """
    if prev_best_s is None or lap_time_s is None:
        return 0.0
    gain = float(prev_best_s) - float(lap_time_s)
    if gain <= 0.0 or not (gain < 1e4):  # ignore an infinite/anomalous previous best
        return 0.0
    return PERSONAL_BEST_BONUS + PERSONAL_BEST_PER_S * gain


def _is_plausible_eval_dist(value):
    """True if the eval distance is physically plausible (<= 3800m)."""
    return 0.0 <= float(value) <= EVAL_DISTANCE_SANITY_LIMIT


def _is_plausible_eval_score(value):
    """True if the eval score is plausible (allows > track length for fast laps)."""
    return 0.0 <= float(value) <= EVAL_SCORE_SANITY_LIMIT


def _eval_score(eval_dist, lap_time=None):
    """Comparable scalar score: distance (incomplete lap) or TRACK_LENGTH*T_REF/time (complete).

    When the agent completes the lap the distance saturates at 3608m and stops giving signal;
    the score keeps growing as the time improves, feeding records and refinement.
    """
    if lap_time is not None and 30.0 < float(lap_time) < EVAL_SCORE_T_REF_S * 4:
        return TRACK_LENGTH_M * max(1.0, EVAL_SCORE_T_REF_S / float(lap_time))
    return max(0.0, min(float(eval_dist), TRACK_LENGTH_M))


def _track_progress_from_start(start_dist, current_dist):
    """Progress along the track from a starting point, handling the wrap at the finish line."""
    start = float(start_dist)
    current = float(current_dist)
    progress = current - start
    if progress < 0.0:
        progress += TRACK_LENGTH_M
    return max(0.0, min(progress, TRACK_LENGTH_M))
