# Iterative Motors — IBM AI Racing League — Video Script & Production Pack

> **NOT tracked by git** (scratch deliverable). Language: **English**. Hard limit: **3:00**.
> Fill the placeholders in **[BRACKETS]** before recording.

---

## 0. Requirements checklist (judging criteria → where we satisfy it)

| Requirement | Where in the video |
|---|---|
| Language: English | Entire VO + on-screen text |
| Max 3:00 | ~326 spoken words (~2:10 pure VO; ~2:45–2:55 with footage/pauses) — under 3:00 with margin |
| Identification (team + university) | Cold open on-screen lower-third + VO at 0:00 |
| Technical defense, not a vlog | Architecture + reward engineering + results |
| Who is behind the code? | Per-member VO inserts + closing roll (§3 maps it to real modules) |
| Why did you build it this way? | "Philosophy" beat (0:12) + two-phase reward beat (1:05) |
| IBM Granite usage | Dedicated beat at 1:45 (detail in §4) |
| IBM SkillsBuild usage | 1:45 beat (detail in §5) |
| Show the best result | Results beat at 2:10 — the **68.838s** lap, live |
| Bonus: social/blog of the journey | See §6 checklist |

---

## 1. THE SCRIPT (timecoded)

> Format per beat: **[time] — SECTION** · *VISUAL* · **SPEAKER:** "voice-over".
> Keep the agent's best lap footage running under most beats for energy.

**[0:00–0:12] — COLD OPEN / IDENTIFICATION**
*VISUAL:* onboard + trackside of the agent's best TORCS lap, fast cut. Lower-third on-screen: `ITERATIVE MOTORS · Università degli studi di Salerno · IBM AI Racing League`.
**FRANCESCO:** "This is a neural network driving a race car — and it laps faster than every human lap we recorded. We are Iterative Motors, from Università degli studi di Salerno."

**[0:12–0:35] — WHY WE BUILT IT THIS WAY**
*VISUAL:* split screen — human controller laps vs. agent; simple diagram `Human demo → Behavioral Cloning → Reinforcement Learning`.
**FRANCESCO:** "Learning to race from scratch with reinforcement learning is slow and unstable. So we don't. First we clone a human driver with Behavioral Cloning; then reinforcement learning surpasses that human. Offline-to-online: inherit the skill, then push past it."

**[0:35–1:05] — ARCHITECTURE PILLARS**
*VISUAL:* 19-ray sensor fan, the 29→87 frame-stack, network heads.
**MARIARITA:** "The agent reads the track through nineteen distance sensors, plus speed and angle — twenty-nine values, stacked over three time frames so it perceives motion. We train it on two hundred human laps with Bojarski-style augmentation, so it learns to recover toward the racing line, not just copy it."
**JACOPO:** "It outputs steering, throttle and brake. Gear changes run on a separate deterministic module — keeping the network focused on the driving line."

**[1:05–1:45] — THE CORE CONTRIBUTION (the hard part)**
*VISUAL:* montage of the agent crashing the *same* corner; cut to the two-phase reward diagram; sector-split overlay on the track map.
**FRANCESCO:** "Reinforcement learning found a fast line — but a brittle one, crashing the same corner every lap. The value network was healthy; the policy simply had no safety margin. So we engineered a two-phase reward. Phase one, stabilization: a corridor penalty that rewards keeping a margin, so the car finishes every lap, even if slower. Phase two, time-attack: we split the track into sectors and reward the agent for beating its own best split — chasing a theoretical ideal lap."
**ANDREA:** "And every clean lap the agent drives is recorded and fed back to retrain the imitation model — a data flywheel that raises the baseline on every iteration."

**[1:45–2:10] — IBM GRANITE 8B + SKILLSBUILD**
*VISUAL:* terminal — IBM Granite 8B reading a telemetry log and printing a diagnosis + recommendation; then IBM SkillsBuild course badges.
**ANDREA:** "We run IBM Granite 8B locally as our AI race engineer. It ingests the raw telemetry and the per-sector logs, explains where and why the car loses time, and recommends concrete changes — turning numbers into engineering decisions."
**Matteo:** "We built those skills on IBM SkillsBuild — machine-learning and Python tracks that let our whole team, engineers and communicator alike, contribute."

**[2:10–2:40] — RESULTS**
*VISUAL:* the **68.838s** lap full-screen with a running timer; bar chart `Best human 69.54s  vs  Our agent 68.84s`; the 3.6 km Corkscrew map.
**FRANCESCO:** "The result: our agent laps the Corkscrew circuit in sixty-eight point eight seconds — beating our best human lap of sixty-nine point five. Two hundred human laps became three hundred and thirty, and we are now pushing toward the sixty-five-second track limit."

**[2:40–2:55] — TEAM & CLOSE**
*VISUAL:* team names + roles; final hero shot of the lap; end card `ITERATIVE MOTORS · Università degli studi di Salerno`.
**Matteo:** "Four engineers and one communicator. Francesco on the reinforcement-learning core, Mariarita on imitation learning, Jacopo on the simulator and controls, Andrea on data and tooling, [MARKETING NAME] on communication. Iterative Motors — we taught a machine to find the perfect lap."

> **Timing note:** ~326 spoken words ≈ 2:10 of pure VO; with footage and pauses it lands ~2:45–2:55.
> Under the 3:00 cap with margin. If you add lines, trim the architecture beat (0:35) first.

---

## 2. Speaker map (quick reference for the editor)

- **Francesco** — spine of the technical narrative (philosophy, core contribution, results).
- **Mariarita** — imitation-learning pillar.
- **Jacopo** — simulator & controls pillar.
- **Andrea** — data flywheel + IBM Granite.
- **Matteo** — identification/close + SkillsBuild + storytelling.

---

## 3. WHO IS BEHIND THE CODE — ownership map (technical defense)

> This is the honest division to present. Francesco owns the algorithmic core; the other
> three own well-bounded, self-contained subsystems; the communicator owns the video & journey.

### Francesco — Reinforcement-Learning core (most technical / lead)
- `rl/agent.py` — TD3+BC agent: Twin Critic, Delayed Policy Update, target policy smoothing,
  λ-normalized BC loss, Polyak averaging.
- `rl/reward.py` + `rl/sector_timer.py` — reward engineering: corridor/margin penalty,
  sector-split telemetry, terminal bonuses.
- `rl/train_rl.py` — training loop, **two-phase curriculum** (stabilization → time-attack),
  auto-refinement state machine.
- `models/networks.py` — Actor / Twin-Critic architecture.
- **Strategy:** diagnosing the brittle-policy failure mode and designing the two-phase fix.

### Mariarita — Imitation Learning
- `bc/train_bc.py` — Behavioral Cloning trainer (per-channel weighted loss).
- `bc/augmentation.py` — Bojarski-style data augmentation (lateral/angular perturbation, on-track clamp).
- `common/state.py` — state representation: flatten, z-score normalization, temporal frame stacking.

### Jacopo — Simulator & Controls
- `env/gym_torcs.py` — Gym wrapper: observations, actions, per-step reward, early terminations.
- `env/snakeoil3_gym.py` — SCR/UDP client.
- `env/gearing.py` — algorithmic gear-shifting with hysteresis.
- `env/autostart.sh` + headless TORCS (Xvfb) setup.
- `models/action_mapping.py` — action ↔ pedal mapping, throttle/brake mutual exclusion.

### Andrea — Data, Flywheel & Tooling
- `data/collection.py` — human-lap recording (PS5 DualSense / keyboard).
- `data/lap_recorder.py` — harvesting clean agent laps (the flywheel).
- `data/replay_buffer.py` + `data/hdf5_dataset.py` — replay buffers & dataset.
- `common/checkpoint.py` — atomic checkpointing & robust resume.
- `run.sh` — pipeline orchestrator.
- `eval/test_agent.py` — deterministic evaluation.
- **IBM Granite 8B integration** — the AI race-engineer log-analysis tool (§4).

### Matteo — Digital Marketing
- Video production, editing, on-screen graphics & data visualization.
- Social/blog documentation of the journey (bonus criterion, §6).
- IBM SkillsBuild learning coordination across the team.

---

## 4. IBM GRANITE 8B — "AI Race Engineer" (real tool, for the technical defense)

**What it is.** A real pipeline tool — `tools/race_engineer.py` — runs **IBM Granite 8B locally**
over our training telemetry. Reinforcement learning produces a flood of raw logs; Granite turns
them into engineering decisions. It does **not** drive or touch physics in real time: it is
summarization + reasoning over text, which an 8B instruction model does well locally (no cloud, no
data leaving the machine).

**Pipeline (what the tool actually does).**
1. It extracts structured signals from the training logs: deterministic-eval lap times and crash
   distances, the per-sector split breakdown from our `SectorTimer` (`perde tempo: S07(+0.22s) …`),
   critic/actor loss, and the completion rate.
2. Granite 8B receives these as context and returns: (a) a plain-language **diagnosis**, (b) the
   most likely **failure mode**, and (c) a ranked list of **concrete interventions** (e.g., raise
   the corridor penalty, lower exploration noise, switch phase).
3. We read the top recommendation and apply it to the next run — a human-readable bridge between
   telemetry and tuning.

**How we run it (and why the claim is honest, not a prop).**
```bash
ollama list                                   # exact Granite tag
python tools/race_engineer.py --model <granite-tag>
```
Every live consultation is auto-saved to `train_set/session_logs/race_engineer_reports/` — the
telemetry snapshot **and** Granite's answer, timestamped. That folder is a **verifiable record**
that the model was actually consulted to guide our tuning. We claim only what those reports show.

> **⚠️ Make it real before recording (this IS the plan we chose).** Training is still running, so
> run the tool at the next decision point, let Granite's recommendation inform the next config, and
> keep the generated reports. Capture one run in the terminal for the 1:45 B-roll. The statement
> "we use Granite to analyse our telemetry and guide tuning" is then simply true. (Andrea owns this;
> see `tools/README_race_engineer.md`.)

---

## 5. IBM SKILLSBUILD — usage to state

The team upskilled through **IBM SkillsBuild** before and during the project: machine-learning,
deep-learning and Python tracks. This is what let a **cross-functional** team — four engineers plus
a digital-marketing member — share a common technical vocabulary, so the communicator could
genuinely explain the system rather than narrate over it.

> Fill in the **exact course/credential names** each member completed (judges value specifics):
> - Francesco: [SkillsBuild course(s)]
> - Mariarita: [SkillsBuild course(s)]
> - Jacopo: [SkillsBuild course(s)]
> - Andrea: [SkillsBuild course(s)]
> - Matteo: [SkillsBuild course(s) — e.g., AI fundamentals]

---

## 6. PRODUCTION & DELIVERY CHECKLIST

**Capture the hero lap (the 68.838s result):**
```bash
SHOW_GUI=1 ./run.sh test --weights train_set/checkpoints/td3_det_best_lap.pth --laps 1
```
Screen-record this for the cold open and the results beat. Overlay the live timer ending at 68.84s.

**On-screen identification:** team name `ITERATIVE MOTORS` + `Università degli studi di Salerno` in the first 5
seconds AND on the end card (covers the requirement even if audio is muted).

**Numbers to keep accurate (verified from the repo):**
- Best agent lap: **68.838 s** · Best human lap: **69.54 s** · Track limit target: **~65 s**
- Circuit length: **3608 m** · Dataset: **205** human laps + **134** self-recorded laps
- Algorithm: **TD3+BC** (Fujimoto 2018 / Fujimoto & Gu 2021) · augmentation: Bojarski 2016

**Bonus — social/blog documentation:** publish a short build-journey thread/post (the diagnosis →
two-phase reward → result arc makes a strong story) and link it in the submission. Matteo owns this.

**Final pass:** confirm runtime ≤ 3:00, English audio + English on-screen text, and that every
spoken claim is backed by something visible on screen.

---

## 7. Placeholders to fill
- Exact IBM SkillsBuild course/credential names (§5)
- Captured Granite 8B terminal output for the 1:45 B-roll (§4)
