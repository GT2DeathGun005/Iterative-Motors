"""Reward shaping del TD3+BC: bonus/penalità di fine giro e score di valutazione.

La reward per-step (progresso + penalità posizione + anti-zigzag) vive in
``env/gym_torcs.py``; qui stanno i termini terminali aggiunti dal training (bonus di
completamento e proporzionale al tempo, malus giro incompleto) e lo score di eval
unificato distanza/tempo usato da record e refinement.
"""

from ..common.constants import TRACK_LENGTH_M

# Bonus/penalità terminali (sommati dal training loop alla reward di gym_torcs).
LAP_SUCCESS_BONUS = 50.0
INCOMPLETE_LAP_PENALTY = 25.0

# Bonus proporzionale al tempo sul giro: +10 punti per ogni secondo sotto gli 80s.
LAP_TIME_BONUS_REF_S = 80.0
LAP_TIME_BONUS_PER_S = 10.0

# Limiti di plausibilità e score di valutazione unificato distanza/tempo.
EVAL_DISTANCE_SANITY_LIMIT = 3800.0
EVAL_SCORE_T_REF_S = 90.0
EVAL_SCORE_SANITY_LIMIT = 6500.0  # corrisponde a un giro < 50s, fisicamente implausibile

# ── Time-attack ───────────────────────────────────────────────────────────
# Premio per aver battuto il PROPRIO record di tempo sul giro (incentiva a limare i tempi
# anche quando la distanza è saturata). Si attiva quando si stabilisce un nuovo best lap.
PERSONAL_BEST_BONUS = 30.0       # bonus fisso per ogni nuovo record personale
PERSONAL_BEST_PER_S = 15.0       # bonus aggiuntivo per ogni secondo guadagnato sul record
# Parametri della fase TIME_ATTACK (attivata via IM_TIME_ATTACK=1): meno ancoraggio alla BC
# (alpha più alto) e rumore esplorativo più fine per micro-ottimizzazione della traiettoria.
TIME_ATTACK_BC_ALPHA = 4.0
TIME_ATTACK_NOISE_FLOOR = 0.02
TIME_ATTACK_ENTRY_S = 70.5       # soglia indicativa di "guida appresa" per promuovere a time-attack


def personal_best_bonus(prev_best_s, lap_time_s):
    """Bonus per un nuovo record personale: fisso + proporzionale ai secondi guadagnati.

    Ritorna 0 se ``lap_time_s`` non migliora ``prev_best_s`` (o se non c'è un best precedente).
    """
    if prev_best_s is None or lap_time_s is None:
        return 0.0
    gain = float(prev_best_s) - float(lap_time_s)
    if gain <= 0.0 or not (gain < 1e4):  # ignora best precedente infinito/anomalo
        return 0.0
    return PERSONAL_BEST_BONUS + PERSONAL_BEST_PER_S * gain


def _is_plausible_eval_dist(value):
    """True se la distanza di eval è fisicamente plausibile (<= 3800m)."""
    return 0.0 <= float(value) <= EVAL_DISTANCE_SANITY_LIMIT


def _is_plausible_eval_score(value):
    """True se lo score di eval è plausibile (ammette > lunghezza pista per giri veloci)."""
    return 0.0 <= float(value) <= EVAL_SCORE_SANITY_LIMIT


def _eval_score(eval_dist, lap_time=None):
    """Score scalare confrontabile: distanza (giro incompleto) o TRACK_LENGTH*T_REF/tempo (completo).

    Quando l'agente completa il giro la distanza satura a 3608m e smette di dare segnale;
    lo score continua a crescere al migliorare del tempo, alimentando record e refinement.
    """
    if lap_time is not None and 30.0 < float(lap_time) < EVAL_SCORE_T_REF_S * 4:
        return TRACK_LENGTH_M * max(1.0, EVAL_SCORE_T_REF_S / float(lap_time))
    return max(0.0, min(float(eval_dist), TRACK_LENGTH_M))


def _track_progress_from_start(start_dist, current_dist):
    """Progresso lungo il tracciato da un punto iniziale, gestendo il wrap al traguardo."""
    start = float(start_dist)
    current = float(current_dist)
    progress = current - start
    if progress < 0.0:
        progress += TRACK_LENGTH_M
    return max(0.0, min(progress, TRACK_LENGTH_M))
