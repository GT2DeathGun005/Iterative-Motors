#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  train_all.sh — Avvia l'addestramento AIcar (Behavioral Cloning)
#
#  L'agente imita i dati esperti raccolti durante la Fase 1.
#
#  Uso:
#    ./train_all.sh                         # Avvia il training BC
#
#  Nota: La Fase 1 (Data Collection) richiede guida umana e va eseguita
#        manualmente con: python data_collection.py
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
DEMO_DIR="train_set/laps"
BC_WEIGHTS="train_set/checkpoints/bc_policy.pth"
LOG_DIR="train_set/session_logs"
CHECKPOINT_DIR="train_set/checkpoints"

# BC Hyperparameters
BC_EPOCHS=300
BC_BATCH_SIZE=256

# ── Funzioni utility ──
timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

log_info()  { echo -e "${CYAN}[$(timestamp)]${NC} ${BLUE}ℹ${NC}  $1"; }
log_ok()    { echo -e "${CYAN}[$(timestamp)]${NC} ${GREEN}✅${NC} $1"; }
log_warn()  { echo -e "${CYAN}[$(timestamp)]${NC} ${YELLOW}⚠️${NC}  $1"; }
log_error() { echo -e "${CYAN}[$(timestamp)]${NC} ${RED}❌${NC} $1"; }
log_phase() { echo -e "\n${BOLD}${BLUE}══════════════════════════════════════════${NC}"; \
              echo -e "${BOLD}${BLUE}  $1${NC}"; \
              echo -e "${BOLD}${BLUE}══════════════════════════════════════════${NC}\n"; }

# ── Pre-check ──
log_phase "🏎️  AIcar BC Training Pipeline"

# Controlla che i demo esistano
if [[ ! -d "$DEMO_DIR" ]] || [[ -z "$(ls "$DEMO_DIR"/lap_*.h5 2>/dev/null)" ]]; then
    log_error "Nessun file demo trovato in $DEMO_DIR"
    log_error "Esegui prima la Fase 1: python data_collection.py"
    exit 1
fi

DEMO_COUNT=$(ls "$DEMO_DIR"/lap_*.h5 2>/dev/null | wc -l)
log_info "Demo trovate: ${BOLD}${DEMO_COUNT} giri${NC} in $DEMO_DIR"

# Crea directory necessarie
mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR"

# ═══════════════════════════════════════════════════════════════════════
#  Behavioral Cloning Training
# ═══════════════════════════════════════════════════════════════════════
log_phase "🧠  Training: Behavioral Cloning"
log_info "Dataset: $DEMO_DIR ($DEMO_COUNT giri)"
log_info "Epochs: $BC_EPOCHS | Batch: $BC_BATCH_SIZE"
log_info "Output: $BC_WEIGHTS"

python -u behavioral_cloning.py \
    --dataset "$DEMO_DIR" \
    --epochs "$BC_EPOCHS" \
    --batch_size "$BC_BATCH_SIZE" \
    --output "$BC_WEIGHTS"

if [[ $? -eq 0 ]] && [[ -f "$BC_WEIGHTS" ]]; then
    BC_SIZE=$(du -h "$BC_WEIGHTS" | cut -f1)
    log_ok "Behavioral Cloning completato con successo!"
    log_info "Pesi salvati in: $BC_WEIGHTS ($BC_SIZE)"
    log_info ""
    log_info "Per testare l'agente esegui:"
    log_info "${BOLD}python test_agent.py --weights $BC_WEIGHTS${NC}"
else
    log_error "Behavioral Cloning fallito!"
    exit 1
fi

echo ""

# ═══════════════════════════════════════════════════════════════════════
#  Prossimo Step: SAC Reinforcement Learning
# ═══════════════════════════════════════════════════════════════════════
log_info ""
log_info "Per avviare il Fine-Tuning SAC (Warm-Start dal BC appena addestrato):"
log_info "${BOLD}./train_rl.sh${NC}"
log_info ""
log_ok "${BOLD}Pipeline BC completata!${NC}"

