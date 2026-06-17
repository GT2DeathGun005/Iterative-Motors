#!/usr/bin/env bash
# =============================================================================
#  Iterative Motors — orchestratore unico della pipeline (menu + CLI)
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
#    ./run.sh                  Menu interattivo con frecce + Enter
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
#    ./run.sh                  apre il menu interattivo
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
if [ -t 1 ]; then
    B="\033[1m"; D="\033[2m"; U="\033[4m"; INV="\033[7m"
    G="\033[32m"; Y="\033[33m"; R="\033[31m"; C="\033[36m"; M="\033[35m"; W="\033[37m"; K="\033[90m"; N="\033[0m"
else
    B=""; D=""; U=""; INV=""; G=""; Y=""; R=""; C=""; M=""; W=""; K=""; N=""
fi
log()  { echo -e "${C}[run]${N} $*"; }
err()  { echo -e "${R}[run]${N} $*" >&2; }

PYTHON="${PYTHON:-python}"
MENU_ARGS=()

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

usage() {
    awk '
        NR == 1 { next }
        /^#/ { sub(/^# ?/, ""); print; next }
        { exit }
    ' "${BASH_SOURCE[0]}"
}

# --- menu interattivo ------------------------------------------------------
menu_pause() {
    echo
    read -r -p "Premi Enter per tornare al menu..." _
}

menu_prompt_args() {
    local label="$1"
    echo
    echo -e "${B}${C}${label}${N}"
    echo "Puoi aggiungere opzioni come faresti da CLI."
    echo -e "${K}Esempio: --episodes 4000${N}"
    MENU_ARGS=()
    read -r -p "Argomenti extra per ${label} (Enter = default): " -a MENU_ARGS
}

menu_count_glob() {
    local pattern="$1"
    compgen -G "$pattern" | wc -l
}

menu_running_tasks() {
    local out="" task
    for task in bc bc-enriched rl time-attack; do
        if is_running "$task"; then
            out="${out}${out:+ }${task}"
        fi
    done
    [ -n "$out" ] && echo "$out" || echo "nessuno"
}

menu_f1_art() {
    printf "%b" "${G}"
    cat <<'EOF'
                         __
                   _.--""  |
    .----.     _.-'   |/\| |.--.
    | IBM|__.-'   _________|  |_)  _______________
    |  .-""-.""""" ___,    `----'"))   __   .-""-.""""--._
    '-' ,--. `    |   |   .---.       |:.| ' ,--. `      _`.
     ( (    ) ) __|   |__ \\|// _..--  \/ ( (    ) )--._".-.
      . `--' ;\__________________..--------. `--' ;--------'
       `-..-'                               `-..-'
EOF
    printf "%b" "${N}"
}

menu_header() {
    local human_laps auto_laps running best_lap
    human_laps="$(menu_count_glob "$LAPS_DIR/lap_[0-9]*.h5")"
    auto_laps="$(menu_count_glob "$LAPS_AUTO_DIR/*.h5")"
    running="$(menu_running_tasks)"
    best_lap="n/d"
    [ -f "$CKPT_DIR/td3_det_best_lap.txt" ] && best_lap="$(cat "$CKPT_DIR/td3_det_best_lap.txt" 2>/dev/null)s"

    menu_f1_art
    echo
    echo -e "${B}${W}ITERATIVE MOTORS PIT WALL${N}  ${K}TORCS | BC -> TD3+BC -> TIME ATTACK${N}"
    echo -e "${K}────────────────────────────────────────────────────────────────────────${N}"
    echo -e "${C}Processi:${N} ${running}   ${C}Giri umani:${N} ${human_laps}   ${C}Giri auto:${N} ${auto_laps}   ${C}Best lap:${N} ${best_lap}"
    echo -e "${K}────────────────────────────────────────────────────────────────────────${N}"
}

menu_easter_egg() {
    clear 2>/dev/null || true
    printf "%b" "${G}"
    cat <<'EOF'
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠀⠀⠐⣆⢠⡈⠂⠀⢻⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠃⠀⠀⠀⢹⡈⢿⡄⠀⠘⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠂⠀⠀⠀⠈⣧⠘⢧⠀⠀⢻⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡄⠀⠀⠀⠀⠈⠀⠀⠀⠀⢸⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠃⠠⠀⠀⠀⠀⠀⠂⠙⠀⢰⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⡿⠿⠟⠁⠀⡇⡀⠀⠀⠀⣶⣰⠀⠀⢸⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⡟⠀⠀⢀⡄⠀⠃⠿⠀⠀⠀⠘⡿⠀⠀⣾⣿⣿⣿
⣿⣿⣿⣿⣿⣿⡿⢠⠀⣠⡿⠁⠀⢃⠀⣸⡇⠀⠀⠀⠀⠀⢹⣿⣿⣿
⣿⣿⣿⣿⣿⣿⢁⡿⠀⣿⠷⠀⠀⠸⠄⢻⣿⣄⠀⠀⠀⠀⣾⣿⣿⣿
⣿⣿⣿⣿⣿⡇⢸⣇⠀⣿⣷⠀⠀⠀⠀⢸⣿⣿⣷⣶⣆⢠⣿⣿⣿⣿
⣿⡿⠟⠻⣿⡇⢸⣿⠀⣿⣷⡂⠀⠀⠀⢸⡿⣿⡿⢿⡟⢸⣿⣿⣿⣿
⠛⠁⠀⢀⣀⠁⠸⠟⠀⣿⣿⡇⠀⢀⠀⢸⣾⣿⡇⠺⠇⣸⣿⣿⣿⣿
⣷⡀⢀⣿⠏⠀⣦⣤⣼⣿⣿⡇⠀⣸⠀⠸⣿⠟⠓⠀⠀⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣴⣶⣿⣿⣿⣿⡟⠀⠀⠹⠄⠀⢻⣄⡐⠀⢠⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠁⣰⡄⠀⠀⠀⠀⣿⠉⠓⢸⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⡟⠀⣇⠸⠀⠀⠀⠀⠻⠟⠀⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⠇⠀⢿⣶⣤⠆⠀⠀⠐⠛⠀⠻⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⠏⡀⠀⠈⡿⠛⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⠟⠀⢀⣾⡄⠃⢀⠀⠀⠀⠀⠀⠀⢠⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⡟⣰⢀⣾⡿⢹⡀⠻⠀⠀⠂⠀⠀⠀⣾⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⡟⢴⣣⢾⣾⠇⢸⣷⡀⠀⠀⠀⠰⠀⣦⡹⣿⣿⣿⣿⣿
⣿⣿⣿⣿⡟⠀⠐⠘⠇⠋⢠⢿⣿⣷⠀⠀⠀⠀⠑⢸⡇⠙⣿⣿⣿⣿
⣿⣿⣿⡿⠀⡄⠲⠄⠀⠀⣾⠘⣿⣿⡆⠀⠀⡀⠀⠀⡇⠀⠘⣿⣿⣿
⣿⣿⣿⡇⠘⣇⡀⠈⡇⠀⠀⠀⣿⢿⣿⠀⠀⢻⣷⡀⢸⠀⠀⢻⣿⣿
⣿⣿⣿⠀⡈⢛⣩⡆⢸⣦⠀⠀⠈⣸⣿⡇⢹⠘⣿⣵⠘⣇⠀⢸⣿⣿
⣿⣿⣿⠀⢁⣼⠟⠡⣆⢻⣧⠀⢰⣸⠟⣷⠘⡇⣿⣿⡆⢿⡄⠀⣿⣿
⣿⣿⣟⠀⢀⣤⠠⣦⣿⠀⠈⠄⠈⠁⠈⣻⡀⣧⢸⣿⣧⢸⠀⠀⢹⣿
⣿⣿⡯⠀⡄⡅⢧⣿⣿⡆⡇⠀⠀⡸⢰⣿⡇⢹⡀⣿⣿⡘⡇⠀⢸⣿
⣿⣿⡇⠀⢣⠡⠘⢃⣿⣣⠙⠀⠀⢁⠀⣤⣤⠘⠇⠙⣿⣧⠙⠀⠀⣿
⣿⣿⡇⠀⠈⠁⠡⠈⢿⠸⠀⠀⠀⠜⠁⣿⣿⡀⢰⡷⢸⣿⠀⠀⠀⣿
⣿⣿⠃⠀⠘⠂⠀⣾⣜⠃⢰⠀⠀⠘⡇⣿⡿⠃⠘⡇⢸⣿⡄⠈⠀⢸
⣿⣿⠀⠀⣴⣖⡲⣿⣿⣿⠀⠀⠁⢀⡇⠆⠀⠀⡆⠀⣿⣿⡇⠰⡀⣴
⣿⣿⠀⠀⡏⣹⣿⣄⠘⢿⡇⠀⠀⢸⡇⢀⠀⠰⠗⠀⣈⢻⣿⠀⠀⢻
⣿⣿⠀⠀⠀⣿⣿⡍⢷⠘⡇⠀⠀⠘⠇⢸⠄⠠⣄⠀⣿⡌⣿⡇⠀⣼
⣿⣿⠀⠀⠀⢀⣀⡀⠀⠃⠁⠀⠀⠀⠀⠸⠀⡆⠉⠀⠈⢃⢛⣧⡄⢹
⣿⣿⠀⠐⣫⣿⣿⣻⣦⡀⠀⠀⠀⢠⠀⠀⠀⠁⠀⠀⢠⣈⣸⣿⣧⠸
⣿⣿⠀⣼⣿⣿⣿⣿⣿⣹⡆⠀⠀⠘⠗⠀⠈⠀⠀⠀⢸⣿⣿⣿⣿⢀
⣿⣿⠃⣛⠛⠛⢿⣉⠙⢿⡏⠀⠀⠀⠀⠀⠀⠀⠐⠒⠚⣹⣿⣿⡟⠈
⣿⡏⠀⢻⣷⣦⡠⠈⠀⠈⠳⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⣿⣿⠋⠁⢰
EOF
    printf "%b" "${N}"
    echo
    read -r -p "Premi Enter per tornare al pit wall..." _
}

konami_feed() {
    local token="$1"
    local expected="${KONAMI_CODE[$KONAMI_POS]}"

    if [ "$token" = "$expected" ]; then
        KONAMI_POS=$((KONAMI_POS + 1))
        if [ "$KONAMI_POS" -eq "${#KONAMI_CODE[@]}" ]; then
            KONAMI_POS=0
            menu_easter_egg
        fi
    elif [ "$token" = "${KONAMI_CODE[0]}" ]; then
        KONAMI_POS=1
    else
        KONAMI_POS=0
    fi
}

menu_draw() {
    local selected="$1"
    clear 2>/dev/null || true
    menu_header
    echo -e "${D}Freccia su/giu per muoverti, Enter per selezionare, q o Esc per uscire.${N}"
    echo

    local i
    for i in "${!MENU_LABELS[@]}"; do
        if [ "$i" -eq "$selected" ]; then
            echo -e " ${INV}${B}  ${MENU_LABELS[$i]}  ${N}"
        else
            echo -e "   ${MENU_LABELS[$i]}"
        fi
    done

    echo
    echo -e "${K}────────────────────────────────────────────────────────────────────────${N}"
    echo -e "${Y}Scelta:${N} ${MENU_DESCRIPTIONS[$selected]}"
}

menu_run_action() {
    local action="$1"
    clear 2>/dev/null || true
    case "$action" in
        status)
            cmd_status
            menu_pause
            ;;
        logs-bc)
            cmd_logs bc 80
            menu_pause
            ;;
        logs-bc-enriched)
            cmd_logs bc-enriched 80
            menu_pause
            ;;
        logs-rl)
            cmd_logs rl 80
            menu_pause
            ;;
        logs-time-attack)
            cmd_logs time-attack 80
            menu_pause
            ;;
        collect-controller)
            cmd_collect --device controller
            ;;
        collect-keyboard)
            cmd_collect --device keyboard
            ;;
        bc)
            menu_prompt_args "bc"
            cmd_bc "${MENU_ARGS[@]}"
            menu_pause
            ;;
        bc-enriched)
            menu_prompt_args "bc-enriched"
            cmd_bc_enriched "${MENU_ARGS[@]}"
            menu_pause
            ;;
        rl)
            menu_prompt_args "rl"
            cmd_rl "${MENU_ARGS[@]}"
            menu_pause
            ;;
        time-attack)
            menu_prompt_args "time-attack"
            cmd_time_attack "${MENU_ARGS[@]}"
            menu_pause
            ;;
        test)
            menu_prompt_args "test"
            cmd_test "${MENU_ARGS[@]}"
            ;;
        stop-all)
            cmd_stop
            menu_pause
            ;;
        stop-bc)
            cmd_stop bc
            menu_pause
            ;;
        stop-bc-enriched)
            cmd_stop bc-enriched
            menu_pause
            ;;
        stop-rl)
            cmd_stop rl
            menu_pause
            ;;
        stop-time-attack)
            cmd_stop time-attack
            menu_pause
            ;;
        help)
            usage
            menu_pause
            ;;
        exit)
            return 1
            ;;
    esac
}

cmd_menu() {
    if [ ! -t 0 ] || [ ! -t 1 ]; then
        usage
        return 0
    fi

    MENU_LABELS=(
        "[Dashboard] Stato pipeline"
        "[Log] Time attack"
        "[Log] BC arricchita"
        "[Log] RL"
        "[Log] BC"
        "[Dati] Raccolta con controller"
        "[Dati] Raccolta con tastiera"
        "[Train] Behavioral Cloning"
        "[Train] BC arricchita"
        "[Train] TD3+BC"
        "[Race] Time attack"
        "[Eval] Test deterministico"
        "[Stop] Tutti i task"
        "[Stop] BC"
        "[Stop] BC arricchita"
        "[Stop] RL"
        "[Stop] Time attack"
        "[Info] Help"
        "[Exit] Esci"
    )
    MENU_ACTIONS=(
        "status"
        "logs-time-attack"
        "logs-bc-enriched"
        "logs-rl"
        "logs-bc"
        "collect-controller"
        "collect-keyboard"
        "bc"
        "bc-enriched"
        "rl"
        "time-attack"
        "test"
        "stop-all"
        "stop-bc"
        "stop-bc-enriched"
        "stop-rl"
        "stop-time-attack"
        "help"
        "exit"
    )
    MENU_DESCRIPTIONS=(
        "Mostra processi, dataset, checkpoint e ultime righe dei log."
        "Apre le ultime 80 righe del log time-attack."
        "Apre le ultime 80 righe del training BC su umano + auto-laps."
        "Apre le ultime 80 righe del training TD3+BC."
        "Apre le ultime 80 righe del training BC base."
        "Avvia la raccolta dati in foreground usando il controller."
        "Avvia la raccolta dati in foreground usando la tastiera."
        "Avvia il training BC in background."
        "Avvia il training BC arricchita in background."
        "Avvia TD3+BC con harvest dei giri in background."
        "Avvia la fase time-attack in background."
        "Avvia il test deterministico in foreground."
        "Invia SIGINT pulito a tutti i task gestiti."
        "Ferma solo il training BC."
        "Ferma solo il training BC arricchita."
        "Ferma solo il training TD3+BC."
        "Ferma solo il time-attack."
        "Mostra la guida testuale dei comandi."
        "Chiude il menu."
    )

    KONAMI_CODE=(UP UP DOWN DOWN LEFT RIGHT LEFT RIGHT B A)
    KONAMI_POS=0

    local selected=0
    local key=""
    while true; do
        menu_draw "$selected"
        IFS= read -rsn1 key || break
        case "$key" in
            q|Q)
                break
                ;;
            "")
                konami_feed OTHER
                menu_run_action "${MENU_ACTIONS[$selected]}" || break
                ;;
            $'\x1b')
                IFS= read -rsn2 -t 0.1 key || break
                case "$key" in
                    "[A")
                        konami_feed UP
                        if [ "$selected" -le 0 ]; then
                            selected=$((${#MENU_LABELS[@]} - 1))
                        else
                            selected=$((selected - 1))
                        fi
                        ;;
                    "[B")
                        konami_feed DOWN
                        selected=$(((selected + 1) % ${#MENU_LABELS[@]}))
                        ;;
                    "[D")
                        konami_feed LEFT
                        ;;
                    "[C")
                        konami_feed RIGHT
                        ;;
                    *)
                        konami_feed OTHER
                        ;;
                esac
                ;;
            b|B)
                konami_feed B
                ;;
            a|A)
                konami_feed A
                ;;
            *)
                konami_feed OTHER
                ;;
        esac
    done
    clear 2>/dev/null || true
}

# --- dispatch --------------------------------------------------------------
cmd="${1:-menu}"; shift || true
case "$cmd" in
    menu)         cmd_menu ;;
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
