"""
gearing.py - cambio marcia deterministico velocita-primario.

La marcia non e' predetta dalla rete. Training, eval e test usano questa stessa
funzione, cosi' non esiste mismatch tra la policy salvata e la guida live.

Principio:
  - il downshift guarda la velocita', che in frenata cala in modo monotono;
  - l'upshift richiede gas applicato e rpm alti;
  - isteresi e cooldown impediscono jitter al confine delle soglie.

Le soglie sono derivate dai giri umani in train_set/laps e validate live sulla
policy RL: circa 10.5 cambi ogni 1000 step, senza oscillazioni rapide.
"""

# Soglie di velocità (km/h) per salire di marcia: g1→2, g2→3, g3→4, g4→5, g5→6.
UP_SPEED = [55.0, 118.0, 200.0, 258.0, 286.0]
# Soglie di velocità (km/h) per scendere di marcia (isteresi: < UP_SPEED): g2→1, g3→2, g4→3, g5→4, g6→5.
DN_SPEED = [40.0, 92.0, 165.0, 232.0, 272.0]

UP_RPM_GATE = 15500.0   # non salire di marcia se gli rpm non sono già alti (evita di "tirare corto")
UP_ACCEL_GATE = 0.4     # non salire se non si è sul gas (chiave anti-hunting in staccata)
SHIFT_COOLDOWN = 5      # step di lockout dopo un cambio (anti-jitter)


def compute_gear(speed_kmh, accel, rpm, current_gear, steps_since_shift):
    """Marcia deterministica robusta. Cambia al massimo di ±1 per chiamata.

    Args:
        speed_kmh: velocità in avanti in km/h (= obs['speedX'] * 50).
        accel: pedale acceleratore APPLICATO in [0,1] (dopo mutual exclusion).
        rpm: giri motore grezzi (= obs['rpm']).
        current_gear: marcia attuale (1..6).
        steps_since_shift: step trascorsi dall'ultimo cambio.

    Returns:
        (gear: int, shifted: bool)
    """
    g = int(current_gear)
    if g < 1:
        g = 1
    if steps_since_shift < SHIFT_COOLDOWN:
        return g, False

    # UPSHIFT: solo sul gas + rpm alti + sopra la soglia di velocità della marcia.
    if g < 6 and speed_kmh > UP_SPEED[g - 1] and accel > UP_ACCEL_GATE and rpm > UP_RPM_GATE:
        return g + 1, True

    # DOWNSHIFT: la velocità è scesa sotto la soglia della marcia (in frenata o decelerazione).
    if g > 1 and speed_kmh < DN_SPEED[g - 2]:
        return g - 1, True

    return g, False
