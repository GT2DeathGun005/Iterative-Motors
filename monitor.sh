#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  monitor.sh — Monitoraggio live del training AIcar (BC)
#
#  Mostra lo stato attuale del training Behavioral Cloning.
#
#  Uso:
#    ./monitor.sh            # Status + follow del training log
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

echo -e "\n${BOLD}${BLUE}══════════════════════════════════════════${NC}"
echo -e "${BOLD}${BLUE}  📊  AIcar BC Training Monitor${NC}"
echo -e "${BOLD}${BLUE}══════════════════════════════════════════${NC}\n"

# ── Processi attivi ──
echo -e "${BOLD}Processi:${NC}"
BC_PID=$(pgrep -f "behavioral_cloning.py" 2>/dev/null || true)
TORCS_PID=$(pgrep -f "torcs-bin" 2>/dev/null || true)

if [[ -n "$BC_PID" ]]; then
    echo -e "  ${GREEN}●${NC} BC Training    PID: $BC_PID"
else
    echo -e "  ${RED}○${NC} BC Training    non attivo"
fi

if [[ -n "$TORCS_PID" ]]; then
    echo -e "  ${GREEN}●${NC} TORCS          PID: $TORCS_PID"
else
    echo -e "  ${RED}○${NC} TORCS          non attivo (non richiesto per BC training)"
fi

# ── Checkpoints ──
echo -e "\n${BOLD}Checkpoints:${NC}"
BC_PATH="$SCRIPT_DIR/train_set/checkpoints/bc_policy.pth"
if [[ -f "$BC_PATH" ]]; then
    BC_DATE=$(stat -c '%y' "$BC_PATH" 2>/dev/null | cut -d. -f1)
    BC_SIZE=$(du -h "$BC_PATH" | cut -f1)
    echo -e "  ${GREEN}✓${NC} bc_policy.pth          ${BC_SIZE}  ($BC_DATE)"
else
    echo -e "  ${RED}✗${NC} bc_policy.pth          non trovato"
fi

# ── Follow mode ──
if [[ -n "$BC_PID" ]]; then
    echo -e "\n${CYAN}━━━ Log di training (Ctrl+C per uscire) ━━━${NC}\n"
    # Il log del BC va su stdout di train_all.sh solitamente, 
    # ma behavioral_cloning.py stampa a video.
    # Se train_all.sh fosse rediretto potremmo seguirlo.
    echo -e "${YELLOW}Monitoraggio in tempo reale non disponibile via file log per BC.${NC}"
    echo -e "${YELLOW}Il training BC stampa direttamente nel terminale di train_all.sh.${NC}"
fi

echo ""

