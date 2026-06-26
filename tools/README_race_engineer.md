# AI Race Engineer — IBM Granite 8B over training telemetry

A small, honest pipeline tool: it reads the **real** training logs, builds a compact prompt, and
sends it to **IBM Granite 8B running locally**. Granite returns a plain-language diagnosis of the
agent's behaviour plus a ranked list of concrete tuning interventions.

It does **not** drive or touch physics in real time — it is summarization + reasoning over text,
which an 8B instruction model does well locally (no cloud, no data leaving the machine).

## What it sends to Granite

Extracted from `train_set/session_logs/*.log`:
- recent deterministic evaluations (distance reached / lap time),
- per-sector split breakdowns of completed laps (where time is lost vs the ideal lap),
- recent episode outcomes + critic/actor loss, and the completion rate,
- the current best lap (from `td3_det_best_lap.txt`).

## Usage

```bash
# See exactly what would be sent (no Granite needed):
python tools/race_engineer.py --dry-run

# Real consultation via Ollama (saves a dated report by default):
ollama list                                   # find your exact Granite tag
python tools/race_engineer.py --model <granite-tag>

# Different runtime? Generate the prompt and pipe it wherever you run Granite:
python tools/race_engineer.py --dry-run | <your-granite-command>
```

Options: `--log <file>`, `--model <tag>` (or env `GRANITE_MODEL`), `--ollama-url`, `--no-save`.

## Audit trail

Every live consultation is saved to `train_set/session_logs/race_engineer_reports/report_<ts>.md`
— the telemetry snapshot **and** Granite's answer. That folder is a verifiable record that the
model was actually consulted to guide tuning decisions. 

## Workflow

Run it at each decision point (e.g. when an eval plateau appears), read Granite's top
recommendation, and apply it to the next run. The reports document the loop.
