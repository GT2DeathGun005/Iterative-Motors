#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  watchdog.sh — Sorveglia il training TD3+BC notturno e interviene su collasso
#
#  Gira come processo INDIPENDENTE (non serve Claude Code né un terminale aperto):
#      nohup ./watchdog.sh > train_set/session_logs/watchdog.out 2>&1 &
#  Per fermarlo:
#      pkill -f watchdog.sh
#
#  Cosa fa, ad ogni ciclo (default ogni 90s):
#    1. Legge la coda di td3_training.log.
#    2. Rileva i segnali di PERDITA LETALE della Q-function:
#         - CriticL esplosa (> SOGLIA) o NaN/Inf
#         - eval deterministico crollato (< MIN_EVAL_M per N eval consecutivi)
#         - processo di training morto inaspettatamente
#         - log fermo da troppo tempo (training appeso / TORCS impiccato)
#    3. Su rilevamento:
#         - fa SEMPRE un backup timestamped di td3_best_ever.pth (+ checkpoint)
#         - se MODE=active → ferma il run e lo riavvia con --rollback (o resume)
#         - se MODE=alert  → non tocca nulla, scrive solo l'allarme (+ notify-send)
#    4. Cooldown + tetto massimo interventi → niente loop di restart.
# ═══════════════════════════════════════════════════════════════════════
set -uo pipefail

# ── Directory del progetto ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ════════════════════ CONFIGURAZIONE ════════════════════
LOG="train_set/session_logs/td3_training.log"   # log scritto da td3_bc.py
WD_LOG="train_set/session_logs/watchdog.log"     # log del watchdog stesso
CKPT_DIR="train_set/checkpoints"
BACKUP_DIR="train_set/checkpoints/watchdog_backups"

MODE="${WD_MODE:-active}"        # "active" = backup+rollback automatico | "alert" = solo backup+allarme
POLL_SECONDS="${WD_POLL:-90}"    # ogni quanto controlla
CRITIC_MAX="${WD_CRITIC_MAX:-50}"   # CriticL oltre questo = esplosione (sana è 0.01–5)
MIN_EVAL_M="${WD_MIN_EVAL:-50}"     # eval sotto questi metri = sospetto collasso
EVAL_STREAK="${WD_EVAL_STREAK:-4}"  # quanti eval bassi consecutivi servono per confermare
STALE_SECONDS="${WD_STALE:-600}"    # log fermo da >10 min con processo vivo = appeso
COOLDOWN_SECONDS="${WD_COOLDOWN:-1800}"  # 30 min tra un intervento e l'altro
MAX_INTERVENTIONS="${WD_MAX_INT:-3}"     # oltre questo, smette di riavviare e resta in sola allerta

# ════════════════════ STATO INTERNO ════════════════════
interventions=0
last_intervention_ts=0

mkdir -p "$BACKUP_DIR" "$(dirname "$WD_LOG")"

wlog() { echo "[$(date '+%F %T')] $1" | tee -a "$WD_LOG"; }

notify() {   # best-effort: notifica desktop se disponibile, altrimenti silenzioso
    command -v notify-send >/dev/null 2>&1 && notify-send "AIcar Watchdog" "$1" || true
}

backup_best() {   # salva una copia timestamped dei pesi migliori e del checkpoint
    local ts; ts="$(date '+%Y%m%d_%H%M%S')"
    for f in td3_best_ever.pth td3_best_ever.txt td3_checkpoint.pth td3_best_lap.pth; do
        [[ -f "$CKPT_DIR/$f" ]] && cp -p "$CKPT_DIR/$f" "$BACKUP_DIR/${f%.pth}_$ts.${f##*.}" 2>/dev/null
    done
    wlog "🛡️  Backup dei pesi migliori salvato in $BACKUP_DIR (suffisso $ts)"
}

# ── Estrattori dal log ──
latest_critic() {   # ultimo valore di CriticL (numero, o 'nan'/'inf')
    grep 'CriticL:' "$LOG" 2>/dev/null | tail -1 | grep -oE 'CriticL: *[^ |]+' | awk '{print $2}'
}

low_eval_streak() {   # 1 se gli ultimi EVAL_STREAK eval sono TUTTI < MIN_EVAL_M
    local vals n low
    vals=$(grep '\[EVAL\]' "$LOG" 2>/dev/null | tail -"$EVAL_STREAK" | grep -oE 'Dist [0-9]+m' | grep -oE '[0-9]+')
    n=$(echo "$vals" | grep -c .); [[ "$n" -lt "$EVAL_STREAK" ]] && { echo 0; return; }
    low=$(echo "$vals" | awk -v t="$MIN_EVAL_M" '$1 < t' | grep -c .)
    [[ "$low" -ge "$EVAL_STREAK" ]] && echo 1 || echo 0
}

critic_is_lethal() {   # 1 se CriticL è NaN/Inf o oltre la soglia
    local v="$1"
    [[ -z "$v" ]] && { echo 0; return; }
    awk -v v="$v" -v m="$CRITIC_MAX" 'BEGIN{
        lv=tolower(v);
        if (lv=="nan"||lv=="-nan"||lv=="inf"||lv=="-inf") {print 1; exit}
        if (v+0 > m) {print 1} else {print 0}
    }'
}

training_running() { pgrep -f 'td3_bc.py' >/dev/null 2>&1; }

log_age() {   # secondi dall'ultima modifica del log
    [[ -f "$LOG" ]] || { echo 999999; return; }
    echo $(( $(date +%s) - $(stat -c %Y "$LOG") ))
}

intervene() {   # $1 = motivo, $2 = modalità riavvio ("rollback" | "resume")
    local reason="$1" restart_mode="$2" now; now=$(date +%s)
    wlog "🚨 COLLASSO RILEVATO: $reason"
    notify "Collasso: $reason"
    backup_best

    if [[ "$MODE" != "active" ]]; then
        wlog "ℹ️  MODE=alert → nessun riavvio automatico. Backup fatto, in attesa di te."
        return
    fi
    if (( now - last_intervention_ts < COOLDOWN_SECONDS )); then
        wlog "⏳ Cooldown attivo (<$((COOLDOWN_SECONDS/60)) min dall'ultimo intervento). Salto il riavvio."
        return
    fi
    if (( interventions >= MAX_INTERVENTIONS )); then
        wlog "🛑 Raggiunto il tetto di $MAX_INTERVENTIONS interventi. Passo in sola allerta (niente più riavvii)."
        return
    fi

    wlog "🔧 Intervento: stop del training e riavvio in modalità '$restart_mode'..."
    ./stop_training.sh >> "$WD_LOG" 2>&1
    sleep 8   # lascia liberare porte UDP / processi TORCS

    local flag=""; [[ "$restart_mode" == "rollback" ]] && flag="--rollback"
    nohup ./train_rl.sh $flag >> "train_set/session_logs/td3_resumed_$(date '+%Y%m%d_%H%M%S').log" 2>&1 &
    wlog "▶️  Training riavviato (./train_rl.sh $flag) — PID $!"

    interventions=$((interventions+1))
    last_intervention_ts=$now
}

# ════════════════════ LOOP PRINCIPALE ════════════════════
wlog "👁️  Watchdog avviato — MODE=$MODE | poll=${POLL_SECONDS}s | CriticL>$CRITIC_MAX | eval<${MIN_EVAL_M}m×$EVAL_STREAK"
while true; do
    if [[ ! -f "$LOG" ]]; then
        wlog "⚠️  Log non ancora presente ($LOG). Attendo l'avvio del training..."
        sleep "$POLL_SECONDS"; continue
    fi

    cv="$(latest_critic)"
    age="$(log_age)"

    if [[ "$(critic_is_lethal "$cv")" == "1" ]]; then
        intervene "CriticL letale = $cv (soglia $CRITIC_MAX)" "rollback"
    elif [[ "$(low_eval_streak)" == "1" ]]; then
        intervene "eval deterministico crollato (<${MIN_EVAL_M}m per $EVAL_STREAK valutazioni)" "rollback"
    elif ! training_running; then
        # Processo assente: o finito normalmente, o morto. Se il log non segna completamento → morto.
        if grep -q "Training TD3 completato" "$LOG" 2>/dev/null; then
            wlog "✅ Training terminato normalmente. Watchdog si ferma."
            break
        else
            intervene "processo td3_bc.py morto inaspettatamente" "resume"
        fi
    elif (( age > STALE_SECONDS )); then
        intervene "log fermo da $((age/60)) min con processo vivo (appeso/TORCS impiccato)" "resume"
    else
        wlog "✓ sano — CriticL=$cv | ultimo aggiornamento ${age}s fa"
    fi

    sleep "$POLL_SECONDS"
done
