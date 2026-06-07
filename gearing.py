"""
gearing.py — Cambio marcia DETERMINISTICO (velocità-primario, anti-hunting).

Sostituisce la testa gear appresa (gear_head, congelata durante l'RL e soggetta a
hunting: fino a 322 cambi ogni 1000 step, con assurdità tipo 1ª a 150 km/h).

PROBLEMA classico degli auto-shifter algoritmici: il downshift in staccata fa SALIRE
gli rpm → uno shifter rpm-based crede di dover risalire di marcia → oscilla.

SOLUZIONE qui adottata:
  - Il DOWNSHIFT guarda la VELOCITÀ (monotòna decrescente in frenata), NON gli rpm:
    il picco di rpm nel downshift diventa irrilevante → niente oscillazione.
  - L'UPSHIFT scatta solo SE SUL GAS (accel alto) e con rpm alti: durante la staccata
    (gas≈0) l'upshift è bloccato anche se gli rpm superano la soglia.
  - Isteresi (UP_SPEED > DN_SPEED) + cooldown post-cambio → zero jitter al confine.

Soglie DERIVATE e VALIDATE sui 75 giri umani (train_set/laps):
  - accordo ±1 marcia con la guida umana: 99.3%
  - cambi marcia: 9.7 ogni 1000 step (umano reale 7.8; policy rotta 322)
  - upshift umano: accel~1.00, rpm~19400 | downshift umano: brake~1.00 (conferma il design)
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
