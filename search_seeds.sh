#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  search_seeds.sh — Automatizza la ricerca di seed in background
#
#  Avvia test_agent.py in modalità headless (con Xvfb) per trovare i seed
#  che permettono di completare un giro con il miglior checkpoint salvato.
#
#  Uso:
#    ./search_seeds.sh            # Cerca fino a 50 giri completati (resume)
#    ./search_seeds.sh 10         # Cerca fino a 10 giri completati (resume)
#    ./search_seeds.sh --stop     # Ferma la ricerca corrente
#    ./search_seeds.sh --status   # Mostra lo stato della ricerca
# ═══════════════════════════════════════════════════════════════════════
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Configurazione ──
WEIGHTS="train_set/checkpoints/sac_checkpoint_best.pth"
LOG_DIR="train_set/session_logs"
STDOUT_LOG="$LOG_DIR/test_stdout.log"
RESULTS_LOG="$LOG_DIR/test_results.log"

LAPS=50

# ── Colori ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

if [[ "${1:-}" == "--stop" ]]; then
    ./stop_training.sh
    exit 0
fi

if [[ "${1:-}" == "--status" ]]; then
    TEST_PID=$(pgrep -f "test_agent.py" 2>/dev/null | head -n 1 || true)
    echo -e "\n${BOLD}${CYAN}📊 STATO RICERCA SEED${NC}"
    if [[ -n "$TEST_PID" ]]; then
        echo -e "${GREEN}In esecuzione${NC} (PID: $TEST_PID)"
        echo -e "\nUltimi risultati da $RESULTS_LOG:"
        tail -n 5 "$RESULTS_LOG" 2>/dev/null || echo "(Nessun risultato ancora)"
    else
        echo -e "${RED}Fermato${NC} (nessun processo test_agent.py in esecuzione)"
    fi
    echo ""
    exit 0
fi

if [[ -n "${1:-}" && "$1" =~ ^[0-9]+$ ]]; then
    LAPS="$1"
fi

# Controllo se è già in esecuzione
if pgrep -f "test_agent.py" > /dev/null; then
    echo -e "${YELLOW}⚠️  Attenzione: La ricerca seed (test_agent.py) è già in esecuzione.${NC}"
    echo "Usa ./search_seeds.sh --stop per fermarla prima di avviarne una nuova."
    exit 1
fi

if [[ ! -f "$WEIGHTS" ]]; then
    echo -e "${RED}❌ Errore: File dei pesi non trovato ($WEIGHTS)${NC}"
    echo "Devi completare almeno un giro durante il training per avere il best checkpoint."
    exit 1
fi

echo -e "\n${BOLD}${CYAN}🏎️  AVVIO RICERCA SEED HEADLESS${NC}"
echo -e "Obiettivo totale: trovare ${BOLD}$LAPS${NC} giri di successo (con --resume automatico)."
echo -e "Checkpoint: $WEIGHTS"

mkdir -p "$LOG_DIR"

nohup xvfb-run -a -s "-screen 0 800x600x24" python test_agent.py \
  --weights "$WEIGHTS" \
  --model sac \
  --laps "$LAPS" \
  --resume \
  > "$STDOUT_LOG" 2>&1 &

PID=$!
echo -e "${GREEN}✅ Processo avviato in background con PID $PID${NC}"
echo -e "Puoi consultare comodamente i risultati e i seed ottimali con:"
echo -e "    ${BOLD}tail -f $RESULTS_LOG${NC}"
echo -e "Per monitorare la sessione in corso:"
echo -e "    ${BOLD}tail -f $STDOUT_LOG${NC}\n"
