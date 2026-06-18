#!/usr/bin/env bash
# =============================================================================
#  Iterative Motors — orchestratore unico della pipeline (menu + CLI)
# =============================================================================
#
#  Punto di ingresso unico per TUTTA la pipeline del progetto: raccolta dati,
#  Behavioral Cloning, fine-tuning TD3+BC (con harvest dei giri), fase time-attack,
#  valutazione, oltre a un cruscotto di STATO dei processi e allo STOP pulito.
#
#  I task lunghi (bc, td3, time-attack) girano in BACKGROUND: lo script salva PID e
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
#    td3 [args...]            Fine-tuning TD3+BC con harvest dei giri (background)
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
#    ./run.sh td3 --episodes 2500
#    ./run.sh time-attack --episodes 4000
#    ./run.sh test --laps 3
#    ./run.sh status
#    ./run.sh stop td3
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
MENU_ENV=()
CMD_ARGS=()
CMD_ENV=()

# --- gestione task in background -------------------------------------------
# pid_file <task> -> percorso del pidfile; log_file <task> -> percorso del log
canonical_task() {
    case "$1" in
        rl) echo "td3" ;;
        *)  echo "$1" ;;
    esac
}

pid_file() {
    local task; task="$(canonical_task "$1")"
    echo "$RUN_DIR/$task.pid"
}
log_file() {
    case "$1" in
        bc|bc-enriched) echo "$LOG_DIR/bc.log" ;;
        rl|td3)         echo "$LOG_DIR/td3_training.log" ;;
        *)              echo "$LOG_DIR/$1.log" ;;
    esac
}

is_running() {  # is_running <task> -> 0 se vivo
    local pf; pf="$(pid_file "$1")"
    [ -f "$pf" ] || return 1
    local pid; pid="$(cat "$pf" 2>/dev/null)"
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

task_pids() {
    local task; task="$(canonical_task "$1")"
    local pid args has_time_attack
    ps -eo pid=,args= | while read -r pid args; do
        case "$args" in
            *"python"*"-m iterative_motors.rl.train_rl"*)
                has_time_attack=0
                if [ -r "/proc/$pid/environ" ] && tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | grep -qx 'IM_TIME_ATTACK=1'; then
                    has_time_attack=1
                fi
                if { [ "$task" = "time-attack" ] && [ "$has_time_attack" -eq 1 ]; } ||
                   { [ "$task" = "td3" ] && [ "$has_time_attack" -eq 0 ]; }; then
                    echo "$pid"
                fi
                ;;
            *"python"*"-m iterative_motors.bc.train_bc"*)
                if [ "$task" = "bc-enriched" ] && [[ "$args" == *"--auto_laps"* ]]; then
                    echo "$pid"
                elif [ "$task" = "bc" ] && [[ "$args" != *"--auto_laps"* ]]; then
                    echo "$pid"
                fi
                ;;
        esac
    done
}

start_bg() {  # start_bg <task> <comando...>
    local task; task="$(canonical_task "$1")"; shift
    if is_running "$task"; then
        err "Il task '${task}' è già in esecuzione (PID $(cat "$(pid_file "$task")")). Usa './run.sh stop ${task}' prima."
        return 1
    fi
    local lf; lf="$(log_file "$task")"
    log "Avvio '${task}' in background → log: ${lf}"
    # setsid stacca il wrapper dal terminale. Il comando gira nello stesso process group:
    # stop puo' quindi inviare SIGINT al gruppo e lasciare al trainer il salvataggio pulito.
    setsid bash -c '
        lf="$1"; shift
        "$@" > >(
            sed -u -E \
                -e "/Gym has been|Please upgrade|Users of this|migration guide|Waiting for server|Client connected/d" \
                -e "/^[.][[:space:]]*$/d" \
                -e "/^### TORCS is RELAUNCHED ###$/d" \
                -e "s/^[.][[:space:]]+//" >> "$lf"
        ) 2>&1
    ' run-wrapper "$lf" "$@" &
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
    if [ $# -ge 1 ]; then targets="$1"; else targets="td3 time-attack bc bc-enriched"; fi
    for task in $targets; do
        task="$(canonical_task "$task")"
        local pids=""
        if is_running "$task"; then
            local pid; pid="$(cat "$(pid_file "$task")")"
            pids="$pid"
        else
            pids="$(task_pids "$task" | tr '\n' ' ')"
        fi

        if [ -n "$pids" ]; then
            local pid pgid
            for pid in $pids; do
                pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')"
                log "SIGINT a '${task}' (PID ${pid}${pgid:+, PGID ${pgid}}) — uscita pulita con salvataggio del checkpoint..."
                if [ -n "$pgid" ]; then
                    kill -INT "-$pgid" 2>/dev/null || kill -INT "$pid" 2>/dev/null
                else
                    kill -INT "$pid" 2>/dev/null
                fi
            done
        else
            log "'${task}' non in esecuzione."
        fi
    done
}

clean_task_log() {
    sed -E \
        -e '/Gym has been|Please upgrade|Users of this|migration guide|Waiting for server|Client connected/d' \
        -e '/^[.][[:space:]]*$/d' \
        -e '/^### TORCS is RELAUNCHED ###$/d' \
        -e 's/^[.][[:space:]]+//'
}

log_status_line() {
    local lf="$1"
    clean_task_log < "$lf" | grep -E 'Epoch [0-9]+/[0-9]+|Training completato|Addestramento|Pesi salvati|Ep [0-9]+|\[EVAL\] Result|SUCCESS|CRASH|STOP|NUOVO|Record|Checkpoint|Traceback|Errore|ERROR|Exception' | tail -n 1
}

split_env_args() {
    CMD_ENV=("${MENU_ENV[@]}")
    CMD_ARGS=()
    local token
    for token in "$@"; do
        if [[ "$token" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then
            CMD_ENV+=("$token")
        else
            CMD_ARGS+=("$token")
        fi
    done
}

# --- cruscotto di stato ----------------------------------------------------
cmd_status() {
    echo -e "${B}== Iterative Motors — stato pipeline ==${N}"
    echo -e "${B}Processi:${N}"
    local any=0
    for task in collect bc bc-enriched td3 time-attack test; do
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
    for task in bc td3 time-attack; do
        local lf; lf="$(log_file "$task")"
        if [ -f "$lf" ]; then
            local line; line="$(log_status_line "$lf")"
            [ -n "$line" ] && echo "  [${task}] ${line}"
        fi
    done
}

cmd_logs() {
    local task="${1:-}"; local n="${2:-40}"
    [ -z "$task" ] && { err "Uso: ./run.sh logs <task> [n_righe]"; return 1; }
    local lf; lf="$(log_file "$task")"
    [ -f "$lf" ] || { err "Nessun log per '${task}' ($lf)"; return 1; }
    clean_task_log < "$lf" | tail -n "$n"
}

# --- comandi della pipeline ------------------------------------------------
cmd_collect()     { split_env_args "$@"; log "Raccolta giri umani (foreground)…"; exec env "${CMD_ENV[@]}" "$PYTHON" -m iterative_motors.data.collection "${CMD_ARGS[@]}"; }
cmd_test()        { split_env_args "$@"; log "Valutazione deterministica (foreground)…"; exec env "${CMD_ENV[@]}" "$PYTHON" -m iterative_motors.eval.test_agent "${CMD_ARGS[@]}"; }
cmd_bc()          { split_env_args "$@"; start_bg bc          env "${CMD_ENV[@]}" "$PYTHON" -u -m iterative_motors.bc.train_bc "${CMD_ARGS[@]}"; }
cmd_bc_enriched() { split_env_args "$@"; start_bg bc-enriched env "${CMD_ENV[@]}" "$PYTHON" -u -m iterative_motors.bc.train_bc --auto_laps "$LAPS_AUTO_DIR" "${CMD_ARGS[@]}"; }
# Default di STABILIZZAZIONE per il td3: trust region 0.3 (ancora l'Actor al supporto dati) e rumore
# esplorativo fisso 0.04. Entrambi PRIMA di CMD_ENV/CMD_ARGS, così un IM_EXPL_NOISE=... o --trust_region ...
# passato dall'utente (env e argparse: vince l'ultimo) li sovrascrive.
cmd_td3()         { split_env_args "$@"; start_bg td3         env IM_WRAPPER_LOG_ONLY=1 IM_RECORD_LAPS=1 IM_EXPL_NOISE=0.04 "${CMD_ENV[@]}" "$PYTHON" -u -m iterative_motors.rl.train_rl --trust_region 0.3 "${CMD_ARGS[@]}"; }
cmd_rl()          { cmd_td3 "$@"; }
cmd_time_attack() { split_env_args "$@"; start_bg time-attack env IM_WRAPPER_LOG_ONLY=1 IM_TIME_ATTACK=1 IM_RECORD_LAPS=1 "${CMD_ENV[@]}" "$PYTHON" -u -m iterative_motors.rl.train_rl "${CMD_ARGS[@]}"; }

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
    local preset_labels=()
    local preset_values=()
    local example="--episodes 4000"
    local default_note=""
    local selection custom_line token idx value

    case "$label" in
        collect|collect-controller|collect-keyboard)
            preset_labels=(
                "Segmenti curva only"
                "Zone corkscrew 670:900,2380:2530"
                "Rilancia TORCS ogni 5 giri"
                "Deadzone sterzo 0.03"
                "TCS disabilitato"
                "Soglia TCS slip 4.0"
                "Output train_set"
            )
            preset_values=(
                "--segment_only"
                "--zones 670:900,2380:2530"
                "--relaunch_every 5"
                "--steering_deadzone 0.03"
                "--no-tcs"
                "--tcs_slip 4.0"
                "--output_dir train_set"
            )
            example="--zones 670:900,2380:2530 --segment_only"
            default_note="Default: output_dir=train_set, device scelto dal menu, deadzone=0.05, relaunch_every=10, TCS attivo, zone auto-rilevate, salva giri completi."
            ;;
        bc|bc-enriched)
            preset_labels=(
                "Default: 300 epoche"
                "Override: 500 epoche"
                "Override: batch 512"
                "Override: batch 128"
                "Override: LR 1e-4"
                "Override: LR 5e-4"
                "Output: BC standard"
                "Output: BC arricchita"
            )
            preset_values=(
                "--epochs 300"
                "--epochs 500"
                "--batch_size 512"
                "--batch_size 128"
                "--lr 1e-4"
                "--lr 5e-4"
                "--output train_set/checkpoints/bc_policy.pth"
                "--output train_set/checkpoints/enriched/bc_policy.pth"
            )
            example="--epochs 500 --batch_size 512"
            default_note="Default: dataset=train_set/laps, epochs=300, batch_size=256, lr=3e-4, output=train_set/checkpoints/bc_policy.pth."
            ;;
        td3|time-attack)
            preset_labels=(
                "Override: 2500 episodi"
                "Override: 4000 episodi"
                "Override: max_steps 15000"
                "Default: seed 42"
                "Rollback al best deterministico"
                "Refinement subito"
                "Disattiva auto-refine"
                "Disattiva trust region"
                "Override: rumore esplorativo 0.02"
                "Non registrare giri auto"
                "Override: registra solo giri auto <= 75s"
                "GUI visibile"
            )
            preset_values=(
                "--episodes 2500"
                "--episodes 4000"
                "--max_steps 15000"
                "--seed 42"
                "--rollback"
                "--refine"
                "--no-auto-refine"
                "--trust_region 0"
                "IM_EXPL_NOISE=0.02"
                "IM_RECORD_LAPS=0"
                "IM_RECORD_MAX_LAP_TIME=75.0"
                "SHOW_GUI=1"
            )
            example="--episodes 2500 SHOW_GUI=1"
            if [ "$label" = "time-attack" ]; then
                default_note="Default time-attack: episodes=1000, max_steps=5000, seed=42, auto-refine attivo, registra giri auto <=80s, noise floor/time-attack attivi, headless salvo SHOW_GUI=1."
            else
                default_note="Default TD3: episodes=1000, max_steps=5000, seed=42, trust_region=0.3 + IM_EXPL_NOISE=0.04 (stabilizzazione Actor/Critic), auto-refine attivo, no rollback/refine immediato, registra giri auto <=80s, headless salvo SHOW_GUI=1."
            fi
            ;;
        test)
            preset_labels=(
                "GUI visibile"
                "Override: 1 giro"
                "Default: 3 giri"
                "Override: 5 giri"
                "Override: max_steps 20000"
                "Default: auto-detect checkpoint"
                "Forza pesi TD3"
                "Forza pesi BC"
                "Best lap TD3"
                "Best dist TD3"
                "BC arricchita"
            )
            preset_values=(
                "SHOW_GUI=1"
                "--laps 1"
                "--laps 3"
                "--laps 5"
                "--max_steps 20000"
                "--kind auto"
                "--kind td3"
                "--kind bc"
                "--weights train_set/checkpoints/td3_det_best_lap.pth --kind td3"
                "--weights train_set/checkpoints/td3_det_best_dist.pth --kind td3"
                "--weights train_set/checkpoints/enriched/bc_policy.pth --kind bc"
            )
            example="SHOW_GUI=1 --laps 1 --kind td3"
            default_note="Default test: laps=3, max_steps=15000, kind=auto, weights auto-detect, headless salvo SHOW_GUI=1."
            ;;
    esac

    echo
    echo -e "${B}${C}${label}${N}"
    [ -n "$default_note" ] && echo -e "${Y}${default_note}${N}"
    MENU_ARGS=()
    MENU_ENV=()

    if [ "${#preset_labels[@]}" -gt 0 ]; then
        echo "Preset comuni (scrivi numeri separati da spazio o virgola, Enter = nessuno):"
        for idx in "${!preset_labels[@]}"; do
            printf "  %2d) %s\n" "$((idx + 1))" "${preset_labels[$idx]}"
        done
        echo
        read -r -p "Preset per ${label}: " selection
        menu_add_preset_selection "$selection"
    fi

    echo
    echo "Puoi aggiungere opzioni personalizzate come faresti da CLI."
    echo "Le variabili tipo SHOW_GUI=1 vengono applicate all'ambiente."
    echo -e "${K}Esempio: ${example}${N}"
    read -r -p "Argomenti extra per ${label} (Enter = default): " custom_line
    if [ "${#preset_labels[@]}" -gt 0 ] && [[ "$custom_line" =~ ^[[:space:]]*[0-9]+([,[:space:]]+[0-9]+)*[[:space:]]*$ ]]; then
        menu_add_preset_selection "$custom_line"
    else
        read -r -a MENU_CUSTOM_WORDS <<< "$custom_line"
        menu_add_tokens "${MENU_CUSTOM_WORDS[@]}"
    fi

    if [ "${#MENU_ENV[@]}" -gt 0 ] || [ "${#MENU_ARGS[@]}" -gt 0 ]; then
        echo
        [ "${#MENU_ENV[@]}" -gt 0 ] && echo -e "${C}Ambiente:${N} ${MENU_ENV[*]}"
        [ "${#MENU_ARGS[@]}" -gt 0 ] && echo -e "${C}Argomenti:${N} ${MENU_ARGS[*]}"
    fi
}

menu_add_preset_selection() {
    local selection="$1"
    local token value
    selection="${selection//,/ }"
    for token in $selection; do
        if [[ "$token" =~ ^[0-9]+$ ]] && [ "$token" -ge 1 ] && [ "$token" -le "${#preset_values[@]}" ]; then
            value="${preset_values[$((token - 1))]}"
            read -r -a MENU_PRESET_WORDS <<< "$value"
            menu_add_tokens "${MENU_PRESET_WORDS[@]}"
        fi
    done
}

menu_add_tokens() {
    local token
    for token in "$@"; do
        [ -z "$token" ] && continue
        if [[ "$token" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then
            MENU_ENV+=("$token")
        else
            MENU_ARGS+=("$token")
        fi
    done
}

menu_count_glob() {
    local pattern="$1"
    compgen -G "$pattern" | wc -l
}

menu_running_tasks() {
    local out="" task
    for task in bc bc-enriched td3 time-attack; do
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
        logs-td3)
            cmd_logs td3 80
            menu_pause
            ;;
        logs-time-attack)
            cmd_logs time-attack 80
            menu_pause
            ;;
        collect-controller)
            menu_prompt_args "collect-controller"
            cmd_collect --device controller "${MENU_ARGS[@]}"
            ;;
        collect-keyboard)
            menu_prompt_args "collect-keyboard"
            cmd_collect --device keyboard "${MENU_ARGS[@]}"
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
        td3)
            menu_prompt_args "td3"
            cmd_td3 "${MENU_ARGS[@]}"
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
        stop-td3)
            cmd_stop td3
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
        "[Log] TD3"
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
        "[Stop] TD3"
        "[Stop] Time attack"
        "[Info] Help"
        "[Exit] Esci"
    )
    MENU_ACTIONS=(
        "status"
        "logs-time-attack"
        "logs-td3"
        "logs-bc"
        "collect-controller"
        "collect-keyboard"
        "bc"
        "bc-enriched"
        "td3"
        "time-attack"
        "test"
        "stop-all"
        "stop-bc"
        "stop-bc-enriched"
        "stop-td3"
        "stop-time-attack"
        "help"
        "exit"
    )
    MENU_DESCRIPTIONS=(
        "Mostra processi, dataset, checkpoint e ultime righe dei log."
        "Apre le ultime 80 righe del log time-attack."
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
        IFS= read -rsn1 key || return 0
        case "$key" in
            q|Q)
                return 0
                ;;
            "")
                konami_feed OTHER
                menu_run_action "${MENU_ACTIONS[$selected]}" || return 0
                ;;
            $'\x1b')
                IFS= read -rsn2 -t 0.1 key || return 0
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
    td3)          cmd_td3 "$@" ;;
    rl)           cmd_rl "$@" ;;
    time-attack)  cmd_time_attack "$@" ;;
    test)         cmd_test "$@" ;;
    status)       cmd_status ;;
    logs)         cmd_logs "$@" ;;
    stop)         cmd_stop "$@" ;;
    help|-h|--help) usage ;;
    *) err "Comando sconosciuto: '${cmd}'"; usage; exit 1 ;;
esac
