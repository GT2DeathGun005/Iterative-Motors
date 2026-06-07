#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  train_rl.sh — Avvia il training TD3+BC (Reinforcement Learning)
#
#  L'agente parte dai pesi del Behavioral Cloning (Warm-Start) e li
#  affina tramite TD3+BC per correggere il Covariate Shift.
#
#  Uso:
#    ./train_rl.sh                      # 1000 episodi (default)
#    TD3_EPISODES=500 ./train_rl.sh     # Override episodi
#    ./train_rl.sh --clean              # Riparte da zero (cancella checkpoint TD3)
#    ./train_rl.sh --rollback           # Rollback alla migliore policy DETERMINISTICA (det_best_lap →
#                                       #   det_best_dist → det_best_dist_run) e congela l'Actor per ~10 ep (recupero)
#    ./train_rl.sh --refine             # Avvia in REFINEMENT (Critic congelato + bc_weight ridotto): usare in
#                                       #   resume quando il training è già in plateau stabile (vedi ARCHITECTURE §17.2)
#
#  Per interrompere il training in sicurezza:
#    Ctrl+C  oppure  ./stop_training.sh
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

# ── Colori ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ── Directory del progetto ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Configurazione ──
BC_WEIGHTS="${BC_WEIGHTS:-train_set/checkpoints/bc_policy.pth}"
TD3_CHECKPOINT="train_set/checkpoints/td3_checkpoint.pth"
TD3_BUFFER="train_set/checkpoints/buffers/td3_checkpoint_buffer.npz"
TD3_ELITE_BUFFER="train_set/checkpoints/buffers/td3_checkpoint_elite_buffer.npz"
TD3_POLICY="train_set/checkpoints/td3_policy.pth"
LOG_DIR="train_set/session_logs"
CHECKPOINT_DIR="train_set/checkpoints"
BUFFER_DIR="train_set/checkpoints/buffers"

# TD3 Hyperparameters (override con variabili d'ambiente)
TD3_EPISODES="${TD3_EPISODES:-1000}"
TD3_SEED="${TD3_SEED:-42}"
TD3_MAX_STEPS="${TD3_MAX_STEPS:-5000}"

# ── Funzioni utility ──
timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

log_info()  { echo -e "${CYAN}[$(timestamp)]${NC} ${BLUE}ℹ${NC}  $1"; }
log_ok()    { echo -e "${CYAN}[$(timestamp)]${NC} ${GREEN}✅${NC} $1"; }
log_warn()  { echo -e "${CYAN}[$(timestamp)]${NC} ${YELLOW}⚠️${NC}  $1"; }
log_error() { echo -e "${CYAN}[$(timestamp)]${NC} ${RED}❌${NC} $1"; }
log_phase() { echo -e "\n${BOLD}${BLUE}══════════════════════════════════════════${NC}"; \
              echo -e "${BOLD}${BLUE}  $1${NC}"; \
              echo -e "${BOLD}${BLUE}══════════════════════════════════════════${NC}\n"; }

# ── Gestione flag --clean ──
if [[ "${1:-}" == "--clean" ]]; then
    log_warn "Flag --clean rilevato: cancellazione checkpoint TD3 precedenti..."
    rm -f "$TD3_CHECKPOINT" "$TD3_BUFFER" "$TD3_ELITE_BUFFER" "$TD3_POLICY" "train_set/checkpoints/td3_expl_best_lap.pth" "train_set/checkpoints/td3_expl_best_dist.pth" "train_set/checkpoints/td3_det_best_dist_run.pth"
    log_ok "Checkpoint TD3 cancellati. Ripartenza pulita."
fi

# Filtra gli argomenti per python (rimuove --clean)
PY_ARGS=()
for arg in "$@"; do
    if [[ "$arg" != "--clean" ]]; then
        PY_ARGS+=("$arg")
    fi
done

# ── Pre-check ──
log_phase "🧠  AIcar TD3+BC Training (Reinforcement Learning)"

# Crea directory necessarie
mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR"

# ═══════════════════════════════════════════════════════════════════════
#  Warm-Start Detection
# ═══════════════════════════════════════════════════════════════════════

if [[ -f "$TD3_CHECKPOINT" ]]; then
    log_phase "♻️  Ripresa Training (Resume)"
    TD3_SIZE=$(du -h "$TD3_CHECKPOINT" | cut -f1)
    log_info "Checkpoint TD3 trovato: ${BOLD}$TD3_CHECKPOINT${NC} ($TD3_SIZE)"
    if [[ -f "$TD3_BUFFER" ]]; then
        BUF_SIZE=$(du -h "$TD3_BUFFER" | cut -f1)
        log_info "Replay Buffer trovato: ${BOLD}$TD3_BUFFER${NC} ($BUF_SIZE)"
    else
        log_warn "Replay Buffer non trovato. Il buffer ripartirà vuoto."
    fi
    log_info "Il training riprenderà dall'ultimo episodio salvato."
elif [[ -f "$BC_WEIGHTS" ]]; then
    log_phase "🚀  Warm-Start da Behavioral Cloning"
    BC_SIZE=$(du -h "$BC_WEIGHTS" | cut -f1)
    log_info "Pesi BC trovati: ${BOLD}$BC_WEIGHTS${NC} ($BC_SIZE)"
    log_info "L'Actor TD3 inizializzerà backbone e teste dal BC."
    log_info "Il Critic partirà da zero (Twin Q-Network)."
    log_info "Gradient Freezing attivo: backbone + gear_head congelati."
else
    log_phase "⚠️  Cold-Start (Nessun Peso Trovato)"
    log_warn "Nessun peso BC trovato in: $BC_WEIGHTS"
    log_warn "Il TD3 partirà da ZERO — l'addestramento sarà molto più lungo."
    log_warn "Consiglio: esegui prima './train_all.sh' per addestrare il BC."
fi

# ═══════════════════════════════════════════════════════════════════════
#  TD3 Training
# ═══════════════════════════════════════════════════════════════════════

log_info "Episodi: ${BOLD}$TD3_EPISODES${NC} | Seed: $TD3_SEED | Max Steps/ep: $TD3_MAX_STEPS"
log_info "Output policy: $TD3_POLICY"
log_info "Output checkpoint: $TD3_CHECKPOINT"
log_info ""
log_info "Per interrompere il training: Ctrl+C o ./stop_training.sh"
log_info "Il checkpoint viene salvato ad ogni episodio (resume-safe)."
echo ""

python -u td3_bc.py \
    --bc_weights "$BC_WEIGHTS" \
    --episodes "$TD3_EPISODES" \
    --max_steps "$TD3_MAX_STEPS" \
    --seed "$TD3_SEED" \
    ${PY_ARGS[@]:+"${PY_ARGS[@]}"}

if [[ $? -eq 0 ]] && [[ -f "$TD3_POLICY" ]]; then
    TD3_SIZE=$(du -h "$TD3_POLICY" | cut -f1)
    log_ok "TD3 completato con successo!"
    log_info "Pesi policy salvati in: ${BOLD}$TD3_POLICY${NC} ($TD3_SIZE)"
    log_info ""
    log_info "Per testare l'agente esegui:"
    log_info "${BOLD}python test_agent.py --weights $TD3_POLICY${NC}"
    log_info ""
    log_info "Oppure lascia che test_agent auto-rilevi i pesi migliori:"
    log_info "${BOLD}python test_agent.py${NC}"
else
    log_error "TD3 Training fallito!"
    exit 1
fi

echo ""
log_ok "${BOLD}Training TD3 completato!${NC}"
