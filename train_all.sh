#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  train_all.sh — Avvia/riprendi l'addestramento AIcar
#
#  Comportamento automatico:
#    1. Se esiste un checkpoint SAC → riprende da lì
#    2. Se esistono i pesi BC ma non un checkpoint → avvia SAC da zero
#    3. Se non esistono neanche i pesi BC → addestra BC e poi avvia SAC
#
#  Uso:
#    ./train_all.sh                         # Auto-detect e riprendi/avvia
#    ./train_all.sh --fresh                 # Forza ripartenza da zero (BC → SAC)
#    ./train_all.sh --bc-only               # Solo Behavioral Cloning
#    ./train_all.sh --resume <checkpoint>   # Riprendi da un checkpoint specifico
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

# ── Directory del progetto (dove si trova questo script) ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Configurazione ──
DEMO_DIR="train_set/laps"
BC_WEIGHTS="train_set/checkpoints/bc_policy.pth"
LOG_DIR="train_set/session_logs"
CHECKPOINT_DIR="train_set/checkpoints"

# BC
BC_EPOCHS=200
BC_BATCH_SIZE=256

# SAC
SAC_EPISODES=2500
SAC_MAX_STEPS=5000
SAC_TARGET_TIME=71.038
SAC_CRITIC_WARMUP=10000
SAC_BC_LAMBDA=1.0
SAC_FREEZE_EPISODES=5
SAC_BATCH_SIZE=256
SAC_RELAUNCH_EVERY=20
SAC_CHECKPOINT_EVERY=50

# ── Parsing argomenti ──
MODE="auto"         # auto | fresh | bc-only | resume
RESUME_PATH=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --fresh)
            MODE="fresh"
            shift ;;
        --bc-only)
            MODE="bc-only"
            shift ;;
        --resume)
            MODE="resume"
            RESUME_PATH="$2"
            shift 2 ;;
        -h|--help)
            echo "Uso: $0 [--fresh | --bc-only | --resume <checkpoint>]"
            echo ""
            echo "  (nessun flag)          Auto-detect: riprende se esiste un checkpoint,"
            echo "                         altrimenti avvia da zero"
            echo "  --fresh                Forza ripartenza completa (BC → SAC)"
            echo "  --bc-only              Solo Behavioral Cloning"
            echo "  --resume <checkpoint>  Riprendi da un checkpoint specifico"
            exit 0 ;;
        *)
            echo -e "${RED}Argomento sconosciuto: $1${NC}"
            exit 1 ;;
    esac
done

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
log_phase "🏎️  AIcar Training Pipeline"

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

# Controlla che non ci sia già un training in corso
if pgrep -f "sac_rl.py" > /dev/null 2>&1; then
    log_warn "Un processo sac_rl.py è già in esecuzione!"
    log_warn "Usa ./stop_training.sh per fermarlo prima di rilanciare."
    exit 1
fi

# ═══════════════════════════════════════════════════════════════════════
#  AUTO-DETECT: cosa fare?
# ═══════════════════════════════════════════════════════════════════════
RUN_BC=false
RUN_SAC=false

if [[ "$MODE" == "auto" ]]; then
    # Cerca il checkpoint più recente (periodici + latest, per data di modifica)
    LATEST_CKPT=$(ls -t "$CHECKPOINT_DIR"/sac_checkpoint_*.pth 2>/dev/null | head -1 || true)

    if [[ -n "$LATEST_CKPT" ]]; then
        # Esiste un checkpoint → riprendi
        RESUME_PATH="$LATEST_CKPT"
        RUN_SAC=true
        CKPT_NAME=$(basename "$LATEST_CKPT")
        log_info "Checkpoint trovato: ${BOLD}$CKPT_NAME${NC}"
        log_info "Modalità: ${BOLD}RIPRENDI${NC} training da checkpoint"
    elif [[ -f "$BC_WEIGHTS" ]]; then
        # Esistono pesi BC ma nessun checkpoint SAC → avvia SAC da zero
        RUN_SAC=true
        log_info "Pesi BC trovati, nessun checkpoint SAC"
        log_info "Modalità: ${BOLD}AVVIA SAC${NC} da zero con warm start BC"
    else
        # Niente esiste → pipeline completa
        RUN_BC=true
        RUN_SAC=true
        log_info "Nessun checkpoint trovato"
        log_info "Modalità: ${BOLD}PIPELINE COMPLETA${NC} (BC → SAC)"
    fi

elif [[ "$MODE" == "fresh" ]]; then
    RUN_BC=true
    RUN_SAC=true
    RESUME_PATH=""
    log_info "Modalità: ${BOLD}RIPARTENZA DA ZERO${NC} (BC → SAC)"

elif [[ "$MODE" == "bc-only" ]]; then
    RUN_BC=true
    log_info "Modalità: ${BOLD}SOLO BC${NC}"

elif [[ "$MODE" == "resume" ]]; then
    RUN_SAC=true
    log_info "Modalità: ${BOLD}RESUME MANUALE${NC} da $RESUME_PATH"
fi

# ═══════════════════════════════════════════════════════════════════════
#  FASE 2: Behavioral Cloning
# ═══════════════════════════════════════════════════════════════════════
if $RUN_BC; then
    log_phase "🧠  Fase 2: Behavioral Cloning"
    log_info "Dataset: $DEMO_DIR ($DEMO_COUNT giri)"
    log_info "Epochs: $BC_EPOCHS | Batch: $BC_BATCH_SIZE | Patience: 15"
    log_info "Output: $BC_WEIGHTS"

    python -u behavioral_cloning.py \
        --dataset "$DEMO_DIR" \
        --epochs "$BC_EPOCHS" \
        --batch_size "$BC_BATCH_SIZE" \
        --output "$BC_WEIGHTS"

    if [[ $? -eq 0 ]] && [[ -f "$BC_WEIGHTS" ]]; then
        BC_SIZE=$(du -h "$BC_WEIGHTS" | cut -f1)
        log_ok "Behavioral Cloning completato! Pesi: $BC_WEIGHTS ($BC_SIZE)"
    else
        log_error "Behavioral Cloning fallito!"
        exit 1
    fi
fi

# ═══════════════════════════════════════════════════════════════════════
#  FASE 3: SAC RL (headless)
# ═══════════════════════════════════════════════════════════════════════
if $RUN_SAC; then
    log_phase "🔥  Fase 3: SAC Reinforcement Learning"

    # Verifica che i pesi BC esistano
    if [[ -z "$RESUME_PATH" ]] && [[ ! -f "$BC_WEIGHTS" ]]; then
        log_error "Pesi BC non trovati: $BC_WEIGHTS"
        log_error "Esegui prima: ./train_all.sh --bc-only"
        exit 1
    fi

    # Verifica xvfb
    if ! command -v xvfb-run &> /dev/null; then
        log_error "xvfb-run non trovato! Installa con: sudo apt install xvfb"
        exit 1
    fi

    # Timestamp per i log
    TS=$(date '+%Y%m%d_%H%M%S')
    SAC_LOG="$LOG_DIR/sac_stdout_${TS}.log"

    # Costruisci il comando
    SAC_CMD="python -u sac_rl.py \
        --episodes $SAC_EPISODES \
        --max_steps $SAC_MAX_STEPS \
        --bc_weights $BC_WEIGHTS \
        --demo_dir $DEMO_DIR \
        --target_time $SAC_TARGET_TIME \
        --critic_warmup_steps $SAC_CRITIC_WARMUP \
        --actor_freeze_episodes $SAC_FREEZE_EPISODES \
        --bc_lambda $SAC_BC_LAMBDA \
        --batch_size $SAC_BATCH_SIZE \
        --relaunch_every $SAC_RELAUNCH_EVERY \
        --checkpoint_every $SAC_CHECKPOINT_EVERY"

    # Aggiungi --resume se specificato
    if [[ -n "$RESUME_PATH" ]]; then
        if [[ ! -f "$RESUME_PATH" ]]; then
            log_error "Checkpoint non trovato: $RESUME_PATH"
            exit 1
        fi
        SAC_CMD="$SAC_CMD --resume $RESUME_PATH"
        log_info "Ripresa da: ${BOLD}$(basename "$RESUME_PATH")${NC}"
    fi

    log_info "Episodi: $SAC_EPISODES | Max steps: $SAC_MAX_STEPS"
    log_info "Target time: ${SAC_TARGET_TIME}s | Freeze: ${SAC_FREEZE_EPISODES} ep"
    log_info "Log: $SAC_LOG"
    log_info ""
    log_info "Avvio training headless con xvfb-run..."

    # Lancia in background
    nohup xvfb-run -a -s "-screen 0 800x600x24" $SAC_CMD \
        > "$SAC_LOG" 2>&1 &
    SAC_PID=$!

    # Aspetta un attimo per verificare che parta
    sleep 3
    if kill -0 "$SAC_PID" 2>/dev/null; then
        log_ok "SAC RL avviato in background (PID: $SAC_PID)"
        log_info ""
        log_info "Monitoraggio:  ${BOLD}./monitor.sh${NC}"
        log_info "Fermare:       ${BOLD}./stop_training.sh${NC}"

        # Salva il PID per stop_training.sh
        echo "$SAC_PID" > "$LOG_DIR/.sac_pid"
    else
        log_error "SAC RL non si è avviato! Controlla: $SAC_LOG"
        tail -20 "$SAC_LOG" 2>/dev/null
        exit 1
    fi
fi

echo ""
log_ok "${BOLD}Pipeline avviata con successo!${NC}"
