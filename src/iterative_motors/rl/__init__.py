"""Reinforcement Learning TD3+BC: agente, reward shaping e training loop.

  - ``agent``    : ``TD3BCAgent`` — Actor + Twin Critic, update ibrida RL/BC (loss del paper
                   Fujimoto & Gu 2021), salvataggio/resume robusto dei checkpoint.
  - ``reward``   : termini terminali della reward (bonus di completamento e proporzionale al
                   tempo, malus giro incompleto), score di valutazione distanza/tempo e bonus
                   di record personale per la fase time-attack.
  - ``train_rl`` : entrypoint del fine-tuning TD3+BC (warm-start da BC, campionamento ibrido,
                   elite gate, ancora progressiva, refinement, harvest dei giri, time-attack).

Si ri-esportano l'agente e l'API di reward; l'entrypoint ``train_rl`` non è importato qui per
non caricare l'ambiente TORCS all'import del package.
"""

from .agent import TD3BCAgent
from .reward import (
    LAP_SUCCESS_BONUS,
    INCOMPLETE_LAP_PENALTY,
    LAP_TIME_BONUS_REF_S,
    LAP_TIME_BONUS_PER_S,
    EVAL_SCORE_T_REF_S,
    personal_best_bonus,
    TIME_ATTACK_BC_ALPHA,
    TIME_ATTACK_NOISE_FLOOR,
    TIME_ATTACK_ENTRY_S,
)

__all__ = [
    "TD3BCAgent",
    "LAP_SUCCESS_BONUS", "INCOMPLETE_LAP_PENALTY", "LAP_TIME_BONUS_REF_S",
    "LAP_TIME_BONUS_PER_S", "EVAL_SCORE_T_REF_S", "personal_best_bonus",
    "TIME_ATTACK_BC_ALPHA", "TIME_ATTACK_NOISE_FLOOR", "TIME_ATTACK_ENTRY_S",
]
