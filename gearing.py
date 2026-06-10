"""
Modulo per la gestione deterministica e algoritmica del cambio marcia (1..6) in TORCS per l'agente IA.

Nota sulla pipeline:
  - Raccolta Dati (Data Collection): Il pilota umano guida utilizzando il cambio manuale (es. pulsanti
    del controller o frecce della tastiera). Le marce inserite manualmente vengono registrate direttamente
    nel dataset HDF5 per l'apprendimento supervisionato iniziale.
  - Addestramento RL ed Evaluation (TD3+BC / Test): La selezione della marcia viene gestita in modo
    automatico e deterministico da questo modulo. Questo permette di escludere la marcia dallo spazio delle
    azioni predette dalla rete neurale (che controlla unicamente steer, accel e brake), semplificando
    l'addestramento ed evitando mismatch comportamentali.

Meccanismi di funzionamento:
  - Downshift (Scalata): Basato esclusivamente sulla velocità del veicolo, che cala in modo
    monotono durante le frenate. Evita le oscillazioni dovute ai picchi temporanei di RPM.
  - Upshift (Salita): Consentito solo in presenza di acceleratore premuto (> 40%), giri motore
    elevati (> 15500 RPM) e velocità superiore alla soglia specifica della marcia corrente.
  - Prevenzione Jitter: Isteresi strutturale (soglie di scalata inferiori a quelle di salita)
    e cooldown temporale di lockout (SHIFT_COOLDOWN) impediscono cambi marcia ripetuti o oscillazioni.

Le soglie sono calibrate sui dati di telemetria dei piloti esperti e validate sul circuito.
"""

# Soglie di velocità (km/h) per salire di marcia: g1→2, g2→3, g3→4, g4→5, g5→6.
UP_SPEED = [55.0, 118.0, 200.0, 258.0, 286.0]
# Soglie di velocità (km/h) per scendere di marcia (isteresi: < UP_SPEED): g2→1, g3→2, g4→3, g5→4, g6→5.
DN_SPEED = [40.0, 92.0, 165.0, 232.0, 272.0]

UP_RPM_GATE = 15500.0   # non salire di marcia se gli rpm non sono già alti
UP_ACCEL_GATE = 0.4     # non salire se non si è sul gas
SHIFT_COOLDOWN = 5      # step di lockout dopo un cambio


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
