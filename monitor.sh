#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  monitor.sh — Monitoraggio live del training AIcar
#
#  Mostra lo stato attuale e poi segue i log in tempo reale.
#
#  Uso:
#    ./monitor.sh            # Status + follow del training log
#    ./monitor.sh --status   # Solo status (senza follow)
# ═══════════════════════════════════════════════════════════════════════
set -uo pipefail

# ── Colori ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/train_set/session_logs"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

echo -e "\n${BOLD}${BLUE}══════════════════════════════════════════${NC}"
echo -e "${BOLD}${BLUE}  📊  AIcar Training Monitor${NC}"
echo -e "${BOLD}${BLUE}══════════════════════════════════════════${NC}\n"

# ── Processi attivi ──
echo -e "${BOLD}Processi:${NC}"
SAC_PID=$(pgrep -f "sac_rl.py" 2>/dev/null || true)
TORCS_PID=$(pgrep -f "torcs-bin" 2>/dev/null || true)
BC_PID=$(pgrep -f "behavioral_cloning.py" 2>/dev/null || true)

if [[ -n "$SAC_PID" ]]; then
    echo -e "  ${GREEN}●${NC} SAC RL         PID: $SAC_PID"
else
    echo -e "  ${RED}○${NC} SAC RL         non attivo"
fi

if [[ -n "$TORCS_PID" ]]; then
    echo -e "  ${GREEN}●${NC} TORCS          PID: $TORCS_PID"
else
    echo -e "  ${RED}○${NC} TORCS          non attivo"
fi

if [[ -n "$BC_PID" ]]; then
    echo -e "  ${GREEN}●${NC} BC Training    PID: $BC_PID"
else
    echo -e "  ${RED}○${NC} BC Training    non attivo"
fi

# ── Checkpoints ──
echo -e "\n${BOLD}Checkpoints:${NC}"
if [[ -f "$SCRIPT_DIR/train_set/checkpoints/bc_policy.pth" ]]; then
    BC_DATE=$(stat -c '%y' "$SCRIPT_DIR/train_set/checkpoints/bc_policy.pth" 2>/dev/null | cut -d. -f1)
    BC_SIZE=$(du -h "$SCRIPT_DIR/train_set/checkpoints/bc_policy.pth" | cut -f1)
    echo -e "  ${GREEN}✓${NC} bc_policy.pth          ${BC_SIZE}  ($BC_DATE)"
else
    echo -e "  ${RED}✗${NC} bc_policy.pth          non trovato"
fi

if [[ -f "$SCRIPT_DIR/train_set/checkpoints/sac_actor_best.pth" ]]; then
    BEST_DATE=$(stat -c '%y' "$SCRIPT_DIR/train_set/checkpoints/sac_actor_best.pth" 2>/dev/null | cut -d. -f1)
    BEST_SIZE=$(du -h "$SCRIPT_DIR/train_set/checkpoints/sac_actor_best.pth" | cut -f1)
    echo -e "  ${GREEN}✓${NC} sac_actor_best.pth     ${BEST_SIZE}  ($BEST_DATE)"
else
    echo -e "  ${YELLOW}–${NC} sac_actor_best.pth     non ancora generato"
fi

LATEST_CKPT=$(ls -t "$SCRIPT_DIR"/train_set/checkpoints/sac_checkpoint_*.pth 2>/dev/null | head -1)
if [[ -n "$LATEST_CKPT" ]]; then
    CKPT_DATE=$(stat -c '%y' "$LATEST_CKPT" 2>/dev/null | cut -d. -f1)
    CKPT_SIZE=$(du -h "$LATEST_CKPT" | cut -f1)
    CKPT_NAME=$(basename "$LATEST_CKPT")
    echo -e "  ${GREEN}✓${NC} $CKPT_NAME  ${CKPT_SIZE}  ($CKPT_DATE)"
fi

# ── Ultimo training log ──
TRAINING_LOG=$(ls -t "$LOG_DIR"/sac_training_*.log 2>/dev/null | head -1)
if [[ -n "$TRAINING_LOG" ]]; then
    echo -e "\n${BOLD}Ultimi episodi:${NC}"
    tail -10 "$TRAINING_LOG" | while IFS= read -r line; do
        # Colora in base al risultato
        if echo "$line" | grep -q "lap=DONE"; then
            echo -e "  ${GREEN}$line${NC}"
        elif echo "$line" | grep -q "reward=-"; then
            echo -e "  ${RED}$line${NC}"
        else
            echo -e "  ${YELLOW}$line${NC}"
        fi
    done

    # Statistiche rapide
    TOTAL_EP=$(wc -l < "$TRAINING_LOG")
    if [[ $TOTAL_EP -gt 0 ]]; then
        AVG_REWARD=$(awk -F'reward=' '{split($2,a,","); sum+=a[1]; n++} END {if(n>0) printf "%.1f", sum/n; else print "N/A"}' "$TRAINING_LOG")
        MAX_REWARD=$(awk -F'reward=' '{split($2,a,","); if(a[1]+0 > max+0) max=a[1]} END {printf "%.1f", max}' "$TRAINING_LOG")
        AVG_STEPS=$(awk -F'steps=' '{split($2,a,","); sum+=a[1]; n++} END {if(n>0) printf "%.0f", sum/n; else print "N/A"}' "$TRAINING_LOG")
        LAST_MASTERY=$(tail -1 "$TRAINING_LOG" | awk -F'mastery=' '{print $2}' | tr -d '\n')

        echo -e "\n${BOLD}Statistiche sessione:${NC}"
        echo -e "  Episodi completati: ${BOLD}$TOTAL_EP${NC}"
        echo -e "  Reward media:       ${BOLD}$AVG_REWARD${NC}"
        echo -e "  Reward massima:     ${BOLD}$MAX_REWARD${NC}"
        echo -e "  Steps medi:         ${BOLD}$AVG_STEPS${NC}"
        if [[ -n "$LAST_MASTERY" ]]; then
            echo -e "  Mastery attuale:    ${BOLD}$LAST_MASTERY${NC}"
        fi
    fi
fi

# ── Follow mode ──
if [[ "${1:-}" != "--status" ]]; then
    if [[ -n "$TRAINING_LOG" ]] && [[ -n "$SAC_PID" ]]; then
        echo -e "\n${CYAN}━━━ Follow mode (Ctrl+C per uscire) ━━━${NC}\n"
        tail -f "$TRAINING_LOG"
    elif [[ -z "$SAC_PID" ]]; then
        echo -e "\n${YELLOW}Training non attivo. Usa --status per solo status.${NC}"
    fi
fi

echo ""
