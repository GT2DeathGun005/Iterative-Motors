#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  train_rl.sh — Avvia il training SAC (Reinforcement Learning)
#
#  L'agente parte dai pesi del Behavioral Cloning (Warm-Start) e li
#  affina tramite Soft Actor-Critic per correggere il Covariate Shift.
#
#  Uso:
#    ./train_rl.sh                      # 1000 episodi (default)
#    SAC_EPISODES=500 ./train_rl.sh     # Override episodi
#    ./train_rl.sh --clean              # Riparte da zero (cancella checkpoint SAC)
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
BC_WEIGHTS="train_set/checkpoints/bc_policy.pth"
SAC_CHECKPOINT="train_set/checkpoints/sac_checkpoint.pth"
SAC_BUFFER="train_set/checkpoints/sac_checkpoint_buffer.npz"
SAC_POLICY="train_set/checkpoints/sac_policy.pth"
LOG_DIR="train_set/session_logs"
CHECKPOINT_DIR="train_set/checkpoints"

# SAC Hyperparameters (override con variabili d'ambiente)
SAC_EPISODES="${SAC_EPISODES:-1000}"
SAC_SEED="${SAC_SEED:-42}"
SAC_MAX_STEPS="${SAC_MAX_STEPS:-5000}"

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
    log_warn "Flag --clean rilevato: cancellazione checkpoint SAC precedenti..."
    rm -f "$SAC_CHECKPOINT" "$SAC_BUFFER" "$SAC_POLICY" "train_set/checkpoints/sac_best_policy.pth" "train_set/checkpoints/sac_best_dist.pth"
    log_ok "Checkpoint SAC cancellati. Ripartenza pulita."
fi

# ── Pre-check ──
log_phase "🧠  AIcar SAC Training (Reinforcement Learning)"

# Crea directory necessarie
mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR"

# ═══════════════════════════════════════════════════════════════════════
#  Warm-Start Detection
# ═══════════════════════════════════════════════════════════════════════

if [[ -f "$SAC_CHECKPOINT" ]]; then
    log_phase "♻️  Ripresa Training (Resume)"
    SAC_SIZE=$(du -h "$SAC_CHECKPOINT" | cut -f1)
    log_info "Checkpoint SAC trovato: ${BOLD}$SAC_CHECKPOINT${NC} ($SAC_SIZE)"
    if [[ -f "$SAC_BUFFER" ]]; then
        BUF_SIZE=$(du -h "$SAC_BUFFER" | cut -f1)
        log_info "Replay Buffer trovato: ${BOLD}$SAC_BUFFER${NC} ($BUF_SIZE)"
    else
        log_warn "Replay Buffer non trovato. Il buffer ripartirà vuoto."
    fi
    log_info "Il training riprenderà dall'ultimo episodio salvato."
elif [[ -f "$BC_WEIGHTS" ]]; then
    log_phase "🚀  Warm-Start da Behavioral Cloning"
    BC_SIZE=$(du -h "$BC_WEIGHTS" | cut -f1)
    log_info "Pesi BC trovati: ${BOLD}$BC_WEIGHTS${NC} ($BC_SIZE)"
    log_info "L'Actor SAC inizializzerà backbone e teste dal BC."
    log_info "Il Critic partirà da zero (Twin Q-Network)."
    log_info "Gradient Freezing attivo: backbone + gear_head congelati."
else
    log_phase "⚠️  Cold-Start (Nessun Peso Trovato)"
    log_warn "Nessun peso BC trovato in: $BC_WEIGHTS"
    log_warn "Il SAC partirà da ZERO — l'addestramento sarà molto più lungo."
    log_warn "Consiglio: esegui prima './train_all.sh' per addestrare il BC."
fi

# ═══════════════════════════════════════════════════════════════════════
#  SAC Training
# ═══════════════════════════════════════════════════════════════════════

log_info "Episodi: ${BOLD}$SAC_EPISODES${NC} | Seed: $SAC_SEED | Max Steps/ep: $SAC_MAX_STEPS"
log_info "Output policy: $SAC_POLICY"
log_info "Output checkpoint: $SAC_CHECKPOINT"
log_info ""
log_info "Per interrompere il training: Ctrl+C o ./stop_training.sh"
log_info "Il checkpoint viene salvato ad ogni episodio (resume-safe)."
echo ""

python -u sac_rl.py \
    --bc_weights "$BC_WEIGHTS" \
    --episodes "$SAC_EPISODES" \
    --max_steps "$SAC_MAX_STEPS" \
    --seed "$SAC_SEED"

if [[ $? -eq 0 ]] && [[ -f "$SAC_POLICY" ]]; then
    SAC_SIZE=$(du -h "$SAC_POLICY" | cut -f1)
    log_ok "SAC completato con successo!"
    log_info "Pesi policy salvati in: ${BOLD}$SAC_POLICY${NC} ($SAC_SIZE)"
    log_info ""
    log_info "Per testare l'agente esegui:"
    log_info "${BOLD}python test_agent.py --weights $SAC_POLICY${NC}"
    log_info ""
    log_info "Oppure lascia che test_agent auto-rilevi i pesi migliori:"
    log_info "${BOLD}python test_agent.py${NC}"
else
    log_error "SAC Training fallito!"
    exit 1
fi

echo ""
log_ok "${BOLD}Training SAC completato!${NC}"
