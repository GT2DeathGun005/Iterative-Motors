"""
Module for the deterministic, algorithmic gear shifting (1..6) in TORCS for the AI agent.

Pipeline note:
  - Data Collection: the human driver drives with a manual gearbox (e.g. controller buttons or
    keyboard arrows). The manually engaged gears are recorded directly into the HDF5 dataset for the
    initial supervised learning.
  - RL Training and Evaluation (TD3+BC / Test): gear selection is handled automatically and
    deterministically by this module. This lets us exclude the gear from the action space predicted
    by the neural network (which controls only steer, accel and brake), simplifying training and
    avoiding behavioural mismatches.

How it works:
  - Downshift: based solely on vehicle speed, which decreases monotonically during braking. Avoids
    oscillations caused by temporary RPM spikes.
  - Upshift: allowed only with the throttle pressed (> 40%), high engine revs (> 15500 RPM) and speed
    above the threshold specific to the current gear.
  - Jitter prevention: structural hysteresis (downshift thresholds lower than upshift ones) and a
    temporal lockout cooldown (SHIFT_COOLDOWN) prevent repeated gear changes or oscillations.

The thresholds are calibrated on expert-driver telemetry and validated on the circuit.
"""

# Speed thresholds (km/h) to shift up: g1→2, g2→3, g3→4, g4→5, g5→6.
UP_SPEED = [55.0, 118.0, 200.0, 258.0, 286.0]
# Speed thresholds (km/h) to shift down (hysteresis: < UP_SPEED): g2→1, g3→2, g4→3, g5→4, g6→5.
DN_SPEED = [40.0, 92.0, 165.0, 232.0, 272.0]

UP_RPM_GATE = 15500.0   # do not shift up if the revs are not already high
UP_ACCEL_GATE = 0.4     # do not shift up if not on the throttle
SHIFT_COOLDOWN = 5      # lockout steps after a shift


def compute_gear(speed_kmh, accel, rpm, current_gear, steps_since_shift):
    """Robust deterministic gear. Changes by at most ±1 per call.

    Args:
        speed_kmh: forward speed in km/h (= obs['speedX'] * 50).
        accel: APPLIED throttle pedal in [0,1] (after mutual exclusion).
        rpm: raw engine revs (= obs['rpm']).
        current_gear: current gear (1..6).
        steps_since_shift: steps elapsed since the last shift.

    Returns:
        (gear: int, shifted: bool)
    """
    g = int(current_gear)
    if g < 1:
        g = 1
    if steps_since_shift < SHIFT_COOLDOWN:
        return g, False

    # UPSHIFT: only on the throttle + high revs + above the gear's speed threshold.
    if g < 6 and speed_kmh > UP_SPEED[g - 1] and accel > UP_ACCEL_GATE and rpm > UP_RPM_GATE:
        return g + 1, True

    # DOWNSHIFT: speed dropped below the gear's threshold (under braking or deceleration).
    if g > 1 and speed_kmh < DN_SPEED[g - 2]:
        return g - 1, True

    return g, False
