#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  stop_training.sh — Ferma tutti i processi di training AIcar
#
#  Ferma in modo pulito:
#    - sac_rl.py (SAC training)
#    - torcs-bin (simulatore TORCS)
#    - xvfb-run (display virtuale)
#
#  Uso:
#    ./stop_training.sh           # Ferma tutto
#    ./stop_training.sh --force   # Kill forzato (SIGKILL)
# ═══════════════════════════════════════════════════════════════════════
set -uo pipefail

# ── Colori ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/train_set/session_logs"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
log_info()  { echo -e "${CYAN}[$(timestamp)]${NC}  $1"; }
log_ok()    { echo -e "${CYAN}[$(timestamp)]${NC} ${GREEN}✅${NC} $1"; }
log_warn()  { echo -e "${CYAN}[$(timestamp)]${NC} ${YELLOW}⚠️${NC}  $1"; }

SIGNAL="TERM"
if [[ "${1:-}" == "--force" ]]; then
    SIGNAL="KILL"
    log_warn "Modalità forzata (SIGKILL)"
fi

echo -e "\n${BOLD}${RED}══════════════════════════════════════════${NC}"
echo -e "${BOLD}${RED}  🛑  Stop Training AIcar${NC}"
echo -e "${BOLD}${RED}══════════════════════════════════════════${NC}\n"

KILLED=0

# ── SAC RL ──
SAC_PIDS=$(pgrep -f "sac_rl.py" 2>/dev/null || true)
if [[ -n "$SAC_PIDS" ]]; then
    for pid in $SAC_PIDS; do
        log_info "Fermando sac_rl.py (PID: $pid)..."
        kill -"$SIGNAL" "$pid" 2>/dev/null && KILLED=$((KILLED + 1))
    done
else
    log_info "Nessun processo sac_rl.py trovato."
fi

# ── TORCS ──
TORCS_PIDS=$(pgrep -f "torcs-bin" 2>/dev/null || true)
if [[ -n "$TORCS_PIDS" ]]; then
    for pid in $TORCS_PIDS; do
        log_info "Fermando torcs-bin (PID: $pid)..."
        kill -"$SIGNAL" "$pid" 2>/dev/null && KILLED=$((KILLED + 1))
    done
else
    log_info "Nessun processo torcs-bin trovato."
fi

# ── Xvfb ──
XVFB_PIDS=$(pgrep -f "Xvfb.*:99" 2>/dev/null || true)
if [[ -n "$XVFB_PIDS" ]]; then
    for pid in $XVFB_PIDS; do
        log_info "Fermando Xvfb (PID: $pid)..."
        kill -"$SIGNAL" "$pid" 2>/dev/null && KILLED=$((KILLED + 1))
    done
fi

# ── Behavioral Cloning ──
BC_PIDS=$(pgrep -f "behavioral_cloning.py" 2>/dev/null || true)
if [[ -n "$BC_PIDS" ]]; then
    for pid in $BC_PIDS; do
        log_info "Fermando behavioral_cloning.py (PID: $pid)..."
        kill -"$SIGNAL" "$pid" 2>/dev/null && KILLED=$((KILLED + 1))
    done
fi

# ── Aspetta che i processi terminino ──
if [[ $KILLED -gt 0 ]]; then
    log_info "Attesa terminazione processi..."
    sleep 3

    # Verifica che siano effettivamente terminati
    REMAINING=$(pgrep -f "sac_rl.py|torcs-bin" 2>/dev/null || true)
    if [[ -n "$REMAINING" ]]; then
        log_warn "Alcuni processi ancora attivi. Forzo la chiusura..."
        for pid in $REMAINING; do
            kill -9 "$pid" 2>/dev/null
        done
        sleep 1
    fi
fi

# ── Pulizia PID file ──
rm -f "$LOG_DIR/.sac_pid" 2>/dev/null

# ── Report ──
echo ""
if [[ $KILLED -gt 0 ]]; then
    log_ok "${BOLD}$KILLED processi fermati.${NC}"
else
    log_ok "Nessun processo di training in esecuzione."
fi

# ── Stato finale ──
REMAINING_CHECK=$(pgrep -f "sac_rl.py|torcs-bin|behavioral_cloning.py" 2>/dev/null || true)
if [[ -z "$REMAINING_CHECK" ]]; then
    log_ok "Tutti i processi di training sono stati fermati."
else
    log_warn "Processi ancora attivi:"
    ps -p $(echo "$REMAINING_CHECK" | tr '\n' ',') -o pid,cmd 2>/dev/null
fi
echo ""
