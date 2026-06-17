#!/usr/bin/env bash
# =============================================================================
#  Iterative Motors — orchestratore unico della pipeline (CLI controller)
# =============================================================================
#
#  Punto di ingresso unico per TUTTA la pipeline del progetto: raccolta dati,
#  Behavioral Cloning, fine-tuning TD3+BC (con harvest dei giri), fase time-attack,
#  valutazione, oltre a un cruscotto di STATO dei processi e allo STOP pulito.
#
#  Sostituisce i vecchi train_bc.sh / train_rl.sh / stop_training.sh.
#
#  I task lunghi (bc, rl, time-attack) girano in BACKGROUND: lo script salva PID e
#  log in train_set/.run/ e in train_set/session_logs/, così "status" e "stop"
#  funzionano anche tra invocazioni diverse del terminale. I task interattivi o brevi
#  (collect, test) girano in FOREGROUND, così vedi l'output dal vivo.
#
#  USO:
#    ./run.sh <comando> [opzioni]
#
#  COMANDI:
#    collect [args...]        Raccolta giri umani in TORCS (foreground, controller/tastiera)
#    bc [args...]             Addestra la BC sui giri umani (background)
#    bc-enriched [args...]    Addestra la BC su giri umani + auto-raccolti (background)
#    rl [args...]             Fine-tuning TD3+BC con harvest dei giri (background)
#    time-attack [args...]    Fase time-attack: l'agente batte i propri tempi (background)
#    test [args...]           Valutazione deterministica dell'agente (foreground)
#    status                   Cruscotto: processi attivi, dataset, checkpoint, ultimi log
#    logs <task> [n]          Mostra le ultime n righe (default 40) del log di un task
#    stop [task]              Stop PULITO (SIGINT) di un task (o di tutti se omesso)
#    help                     Questo messaggio
#
#  ESEMPI:
#    ./run.sh bc
#    ./run.sh bc-enriched --output train_set/checkpoints/enriched/bc_policy.pth
#    ./run.sh rl --episodes 2500
#    ./run.sh time-attack --episodes 4000
#    ./run.sh test --laps 3
#    ./run.sh status
#    ./run.sh stop rl
# =============================================================================

set -u

# Radice del progetto = cartella di questo script. Tutti i path sono relativi a essa.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

RUN_DIR="$ROOT/train_set/.run"          # pidfile dei task in background
LOG_DIR="$ROOT/train_set/session_logs"  # log dei task
LAPS_DIR="$ROOT/train_set/laps"
LAPS_AUTO_DIR="$ROOT/train_set/laps_auto"
CKPT_DIR="$ROOT/train_set/checkpoints"
mkdir -p "$RUN_DIR" "$LOG_DIR"

# Colori (disattivati se non TTY)
if [ -t 1 ]; then B="\033[1m"; G="\033[32m"; Y="\033[33m"; R="\033[31m"; C="\033[36m"; N="\033[0m"; else B=""; G=""; Y=""; R=""; C=""; N=""; fi
log()  { echo -e "${C}[run]${N} $*"; }
err()  { echo -e "${R}[run]${N} $*" >&2; }

PYTHON="${PYTHON:-python}"

# --- gestione task in background -------------------------------------------
# pid_file <task> -> percorso del pidfile; log_file <task> -> percorso del log
pid_file() { echo "$RUN_DIR/$1.pid"; }
log_file() { echo "$LOG_DIR/$1.log"; }

is_running() {  # is_running <task> -> 0 se vivo
    local pf; pf="$(pid_file "$1")"
    [ -f "$pf" ] || return 1
    local pid; pid="$(cat "$pf" 2>/dev/null)"
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

start_bg() {  # start_bg <task> <comando...>
    local task="$1"; shift
    if is_running "$task"; then
        err "Il task '${task}' è già in esecuzione (PID $(cat "$(pid_file "$task")")). Usa './run.sh stop ${task}' prima."
        return 1
    fi
    local lf; lf="$(log_file "$task")"
    log "Avvio '${task}' in background → log: ${lf}"
    # setsid stacca il processo dal terminale: chiudere la shell non lo uccide.
    setsid bash -c "exec $* >'$lf' 2>&1" &
    local pid=$!
    echo "$pid" > "$(pid_file "$task")"
    sleep 1
    if is_running "$task"; then
        log "${G}'${task}' avviato${N} (PID ${pid}). Stato: './run.sh status' | Log: './run.sh logs ${task}'"
    else
        err "${task} è uscito subito. Controlla il log:"; tail -n 20 "$lf" 2>/dev/null
        return 1
    fi
}

cmd_stop() {  # stop pulito (SIGINT) di un task o di tutti
    local targets
    if [ $# -ge 1 ]; then targets="$1"; else targets="rl time-attack bc bc-enriched"; fi
    for task in $targets; do
        if is_running "$task"; then
            local pid; pid="$(cat "$(pid_file "$task")")"
            log "SIGINT a '${task}' (PID ${pid}) — uscita pulita con salvataggio del checkpoint..."
            kill -INT "$pid" 2>/dev/null
        else
            log "'${task}' non in esecuzione."
        fi
    done
}

# --- cruscotto di stato ----------------------------------------------------
cmd_status() {
    echo -e "${B}== Iterative Motors — stato pipeline ==${N}"
    echo -e "${B}Processi:${N}"
    local any=0
    for task in collect bc bc-enriched rl time-attack test; do
        if is_running "$task"; then
            local pid; pid="$(cat "$(pid_file "$task")")"
            echo -e "  ${G}● ${task}${N}  PID ${pid}  $(ps -o etime= -p "$pid" 2>/dev/null | tr -d ' ' | sed 's/^/uptime /')"
            any=1
        fi
    done
    [ "$any" -eq 0 ] && echo "  (nessun task gestito in esecuzione)"

    echo -e "${B}Dataset:${N}"
    printf "  giri umani:  %s\n" "$(ls -1 "$LAPS_DIR"/lap_[0-9]*.h5 2>/dev/null | wc -l)"
    printf "  giri auto:   %s\n" "$(ls -1 "$LAPS_AUTO_DIR"/*.h5 2>/dev/null | wc -l)"

    echo -e "${B}Checkpoint / record:${N}"
    if [ -f "$CKPT_DIR/td3_det_best_lap.txt" ]; then echo "  best giro deterministico: $(cat "$CKPT_DIR/td3_det_best_lap.txt") s"; fi
    if [ -f "$CKPT_DIR/td3_checkpoint.pth" ]; then
        "$PYTHON" - <<PY 2>/dev/null
import torch
c = torch.load("$CKPT_DIR/td3_checkpoint.pth", map_location="cpu", weights_only=False)
print(f"  td3_checkpoint: episodio {c.get('episode','?')}, best_lap {float(c.get('best_lap_time',0)):.3f}s")
PY
    fi

    echo -e "${B}Ultima riga di log dei task:${N}"
    for task in bc bc-enriched rl time-attack; do
        local lf; lf="$(log_file "$task")"
        if [ -f "$lf" ]; then
            local line; line="$(grep -v 'Gym has been\|Please upgrade\|Users of this\|migration guide\|Waiting for server\|Client connected' "$lf" 2>/dev/null | tail -n 1)"
            [ -n "$line" ] && echo "  [${task}] ${line}"
        fi
    done
}

cmd_logs() {
    local task="${1:-}"; local n="${2:-40}"
    [ -z "$task" ] && { err "Uso: ./run.sh logs <task> [n_righe]"; return 1; }
    local lf; lf="$(log_file "$task")"
    [ -f "$lf" ] || { err "Nessun log per '${task}' ($lf)"; return 1; }
    grep -v 'Gym has been\|Please upgrade\|Users of this\|migration guide' "$lf" | tail -n "$n"
}

# --- comandi della pipeline ------------------------------------------------
cmd_collect()     { log "Raccolta giri umani (foreground)…"; exec "$PYTHON" -m iterative_motors.data.collection "$@"; }
cmd_test()        { log "Valutazione deterministica (foreground)…"; exec "$PYTHON" -m iterative_motors.eval.test_agent "$@"; }
cmd_bc()          { start_bg bc          "$PYTHON" -u -m iterative_motors.bc.train_bc "$@"; }
cmd_bc_enriched() { start_bg bc-enriched "$PYTHON" -u -m iterative_motors.bc.train_bc --auto_laps "$LAPS_AUTO_DIR" "$@"; }
cmd_rl()          { start_bg rl          env IM_RECORD_LAPS=1 "$PYTHON" -u -m iterative_motors.rl.train_rl "$@"; }
cmd_time_attack() { start_bg time-attack env IM_TIME_ATTACK=1 IM_RECORD_LAPS=1 "$PYTHON" -u -m iterative_motors.rl.train_rl "$@"; }

usage() { sed -n '2,46p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

# --- dispatch --------------------------------------------------------------
cmd="${1:-help}"; shift || true
case "$cmd" in
    collect)      cmd_collect "$@" ;;
    bc)           cmd_bc "$@" ;;
    bc-enriched)  cmd_bc_enriched "$@" ;;
    rl)           cmd_rl "$@" ;;
    time-attack)  cmd_time_attack "$@" ;;
    test)         cmd_test "$@" ;;
    status)       cmd_status ;;
    logs)         cmd_logs "$@" ;;
    stop)         cmd_stop "$@" ;;
    help|-h|--help) usage ;;
    *) err "Comando sconosciuto: '${cmd}'"; usage; exit 1 ;;
esac
