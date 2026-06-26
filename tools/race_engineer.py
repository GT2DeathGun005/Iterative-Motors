#!/usr/bin/env python3
"""AI Race Engineer — IBM Granite 8B che legge la telemetria di training e la trasforma in
una diagnosi a parole + interventi consigliati.

NON guida e NON analizza la fisica in tempo reale: estrae poche righe strutturate dai log
(eval deterministiche, split a settori, esiti+loss recenti), costruisce un prompt e lo manda a
Granite 8B in locale. È summarization + reasoning su testo — quello che un 8B fa bene.

Uso:
    # Vedi SOLO il prompt che verrebbe inviato (nessun Granite necessario):
    python tools/race_engineer.py --dry-run

    # Esegui davvero su Granite via Ollama (default):
    python tools/race_engineer.py --model granite3.1-dense:8b

    # Altro runtime: genera il prompt e pipalo dove vuoi:
    python tools/race_engineer.py --dry-run | <il-tuo-comando-granite>

Dipende solo dalla standard library (urllib) — nessuna installazione extra.
"""
import argparse
import json
import os
import re
import sys
import urllib.request
from datetime import datetime

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOGS = os.path.join(_ROOT, 'train_set', 'session_logs')
_BEST_LAP_TXT = os.path.join(_ROOT, 'train_set', 'checkpoints', 'td3_det_best_lap.txt')
# Traccia verificabile: ogni consultazione live di Granite viene salvata qui (audit trail
# dell'uso reale del modello per guidare il tuning — onesto e dimostrabile).
_REPORTS = os.path.join(_LOGS, 'race_engineer_reports')


def _tail_matching(path, pattern, n):
    """Ultime n righe del file che contengono il pattern regex (stripate)."""
    if not path or not os.path.exists(path):
        return []
    rx = re.compile(pattern)
    hits = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if rx.search(line):
                hits.append(line.strip())
    return hits[-n:]


def _read_best_lap():
    try:
        with open(_BEST_LAP_TXT, 'r', encoding='utf-8') as f:
            return float(f.read().strip())
    except Exception:
        return None


def _completion_rate(episode_lines):
    """(success, total) sugli esiti episodio estratti."""
    succ = sum(1 for l in episode_lines if '[SUCCESS]' in l)
    return succ, len(episode_lines)


def build_prompt(log_path, n_eval=6, n_sector=4, n_ep=12):
    eval_lines = _tail_matching(log_path, r'\[EVAL\] Result', n_eval)
    sector_lines = _tail_matching(log_path, r'Giro [0-9.]+s \| ideale', n_sector)
    ep_lines = _tail_matching(log_path, r'Ep [0-9]+ \| \[(SUCCESS|CRASH|INCOMPLETE|TIMEOUT)\]', n_ep)
    best = _read_best_lap()
    succ, tot = _completion_rate(ep_lines)

    parts = []
    parts.append(
        "You are a senior race engineer for a TD3+BC reinforcement-learning agent racing on the "
        "TORCS Corkscrew circuit (3608 m). Analyse the telemetry below and answer concisely.\n"
    )
    parts.append("=== TELEMETRY (from the live training logs) ===")
    if best is not None:
        parts.append(f"- Best deterministic lap so far: {best:.3f} s  (best human lap: 69.54 s).")
    parts.append(
        "- Recent deterministic evaluations (Dist = metres reached before crashing; a full lap is "
        "3608 m; 'Lap:' appears only if the lap was completed):"
    )
    parts += [f"    {l}" for l in eval_lines] or ["    (none)"]
    parts.append(
        "- Sector split breakdown of COMPLETED laps (Italian labels: 'Giro' = lap time, 'ideale' = "
        "ideal lap from best-ever sectors, 'gap' = lap minus ideal, 'perde tempo: Sxx(+y s)' = the "
        "sectors losing the most time):"
    )
    parts += [f"    {l}" for l in sector_lines] or ["    (no completed lap with sectors yet)"]
    parts.append(
        "- Recent episode outcomes (SUCCESS = full lap, CRASH = off-track; CriticL = critic loss, "
        "ActorL = actor loss):"
    )
    parts += [f"    {l}" for l in ep_lines] or ["    (none)"]
    if tot:
        parts.append(f"- Completion rate in this window: {succ}/{tot} laps finished.")
    parts.append(
        "\n=== QUESTIONS ===\n"
        "1) What is the agent's main failure mode right now?\n"
        "2) Is the bottleneck the critic (value function) or the policy?\n"
        "3) The three highest-impact changes to try next, most important first.\n"
        "Answer in under 150 words, specific and actionable."
    )
    return "\n".join(parts)


def save_report(prompt, answer, model, log_path):
    """Salva la consultazione (telemetria + risposta di Granite) come report markdown datato."""
    os.makedirs(_REPORTS, exist_ok=True)
    ts = datetime.now()
    path = os.path.join(_REPORTS, f"report_{ts:%Y%m%d_%H%M%S}.md")
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f"# Race Engineer report — {ts:%Y-%m-%d %H:%M:%S}\n\n")
        f.write(f"- Model: `{model}` (IBM Granite, local via Ollama)\n")
        f.write(f"- Log analysed: `{log_path}`\n\n")
        f.write("## Telemetry snapshot sent to Granite\n\n```\n" + prompt + "\n```\n\n")
        f.write("## IBM Granite 8B — analysis & recommendations\n\n" + answer + "\n")
    return path


def call_ollama(prompt, model, url, timeout=120):
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
    return data.get("response", "").strip()


def main():
    default_log = os.path.join(_LOGS, 'time-attack.log')
    if not os.path.exists(default_log):
        default_log = os.path.join(_LOGS, 'td3_training.log')

    ap = argparse.ArgumentParser(description="AI Race Engineer (IBM Granite 8B over training logs).")
    ap.add_argument('--log', default=default_log, help="file di log da analizzare")
    ap.add_argument('--model', default=os.environ.get('GRANITE_MODEL', 'granite3.1-dense:8b'),
                    help="nome del modello Granite su Ollama (o env GRANITE_MODEL)")
    ap.add_argument('--ollama-url', default=os.environ.get('OLLAMA_URL', 'http://localhost:11434/api/generate'))
    ap.add_argument('--dry-run', action='store_true', help="stampa solo il prompt, non chiama Granite")
    ap.add_argument('--no-save', action='store_true', help="non salvare il report su disco")
    args = ap.parse_args()

    prompt = build_prompt(args.log)

    if args.dry_run:
        print(prompt)
        return

    print(f"[race-engineer] log: {args.log}\n[race-engineer] modello: {args.model}\n"
          f"[race-engineer] interrogo Granite...\n", file=sys.stderr)
    try:
        answer = call_ollama(prompt, args.model, args.ollama_url)
    except Exception as e:
        print(f"[race-engineer] Granite non raggiungibile ({e}).\n"
              f"Avvia il modello (es. 'ollama run {args.model}') oppure usa --dry-run e pipa il "
              f"prompt nel tuo runtime.", file=sys.stderr)
        sys.exit(1)
    print("=== IBM Granite 8B — Race Engineer report ===\n")
    print(answer)
    if not args.no_save:
        path = save_report(prompt, answer, args.model, args.log)
        print(f"\n[race-engineer] report salvato in {path}", file=sys.stderr)


if __name__ == '__main__':
    main()
