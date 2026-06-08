# AIcar — Hybrid BC-RL Architecture (IBM AI Racing League 2026)

**Agente autonomo che impara a guidare tramite Behavioral Cloning (BC) e Twin Delayed DDPG (TD3+BC).**

Questo repository implementa una pipeline end-to-end per addestrare un agente di guida autonoma nell'ambiente di simulazione **TORCS** (The Open Racing Car Simulator). L'obiettivo: **fittare perfettamente i dati esperti** e poi **affinare la policy tramite RL** per sconfiggere il Covariate Shift e ottimizzare il tempo sul giro sul circuito **Corkscrew** con una vettura **F1**.

---

## Filosofia del Progetto: Architettura Ibrida BC-RL

Questo progetto supera il classico Behavioral Cloning tramite un'architettura **Ibrida BC-RL**. L'agente parte con una **Deep Policy Network Multi-Head** addestrata offline per imitare l'esperto umano. Per sconfiggere il temuto *Covariate Shift* (che fa deragliare l'agente non appena si discosta millimetricamente dalla traiettoria ottimale), la pipeline prosegue con un **TD3+BC Fine-Tuning**.

Questa fase RL sfrutta il **Warm-Start** dal BC e segue il **TD3+BC minimalista** (Fujimoto & Gu, 2021): l'Actor (backbone incluso) viene allenato con un termine di Behavioral Cloning **costante** che lo ancora ai dati umani, mentre il TD3 massimizza la velocità longitudinale. Stati normalizzati mean-0/std-1 e ancora BC sempre presente in ogni batch → stabilità.

### Punti di forza della pipeline Ibrida BC-RL:
1. **Sample Efficiency**: Il BC fornisce un ottimo punto di partenza, abbattendo drasticamente i tempi di esplorazione del RL.
2. **Ancora Behavioral Cloning stabile**: il TD3 allena backbone e testa continua, ma ogni batch contiene dati umani e mantiene il peso Behavioral Cloning a `1.0` nel training normale. La marcia è fuori dalla rete e viene calcolata da `gearing.py`.
3. **Determinismo della Policy**: la policy di inferenza è deterministica (output `tanh(mean)` senza rumore, `LayerNorm` indipendente dal batch, nessun dropout). L'**ambiente** TORCS, comunicando via UDP real-time con relaunch ad ogni episodio, è *near-deterministico* (stessa griglia di partenza, ma soggetto a jitter di timing): la riproducibilità è alta ma **non bit-esatta**.

### Stato locale verificato (2026-06-08)
- Miglior checkpoint deterministico per distanza: `td3_det_best_dist.pth` (~3290m, preservato anche dopo `--clean`).
- Giro valido deterministico: non ancora presente (`td3_det_best_lap.pth` verrà creato al primo giro valido in eval).
- Giro completo esplorativo: già osservato (`td3_expl_best_lap.pth`, solo riferimento non deterministico).

---

## Architettura del Sistema

La pipeline si compone di quattro fasi sequenziali:

```
┌──────────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
│  Fase 1              │     │  Fase 2              │     │  Fase 3              │     │  Fase 4              │
│  RACCOLTA DATI       │────>│  TRAINING BC         │────>│  RIFINITURA TD3+BC   │────>│  TEST / INFERENZA    │
│                      │     │                      │     │                      │     │                      │
│  PS5 / Tastiera      │     │  behavioral_cloning  │     │  td3_bc.py           │     │  test_agent.py       │
│  data_collection.py  │     │  .py                 │     │  (Warm-Start)        │     │  Deterministico      │
│                      │     │                      │     │                      │     │                      │
│  Output:             │     │  Output:             │     │  Output:             │     │  Valutazione live    │
│  train_set/laps/     │     │  bc_policy.pth       │     │  td3_policy.pth      │     │  su TORCS            │
└──────────────────────┘     └──────────────────────┘     └──────────────────────┘     └──────────────────────┘
```

| Fase | Script | Descrizione |
|------|--------|-------------|
| 1. Data Collection | `data_collection.py` | Raccolta di giri guidati da umano con controller PS5 o tastiera WASD. Salva solo i giri puliti. |
| 2. BC Training | `behavioral_cloning.py` | Addestramento della PolicyNetwork sui dati esperti. Produce una policy che imita l'esperto (`bc_policy.pth`). |
| 3. TD3 RL | `td3_bc.py` | Fine-tuning del modello tramite TD3+BC. Massimizza la velocità; salva i record **deterministici** (`td3_det_best_dist.pth`, e il giro valido `td3_det_best_lap.pth` = submission) e quelli esplorativi di riferimento (`td3_expl_best_*.pth`). Vedi ARCHITECTURE §14. |
| 4. Test & Eval | `test_agent.py` | Esecuzione deterministica del modello finale su TORCS per valutare la capacità di completare giri autonomi. |

---

## Struttura del Repository

```
AIcar/
├── data_collection.py         # Fase 1: Raccolta dati umani (PS5 / Tastiera)
├── behavioral_cloning.py      # Fase 2: Training della Deep Policy Network
├── td3_bc.py                  # Fase 3: TD3 Fine-Tuning (Warm-Start da BC)
├── test_agent.py              # Fase 4: Inferenza deterministica (BC o RL)
├── train_bc.sh                # Script per lanciare il training BC
├── train_rl.sh                # Script per lanciare il training TD3
├── stop_training.sh           # Ferma i processi di training/TORCS
├── README.md
├── gym_torcs/                 # Wrapper Python per TORCS
│   ├── gym_torcs.py           #   Ambiente OpenAI Gym con Reward Reshaping
│   ├── snakeoil3_gym.py       #   Client UDP per comunicazione con TORCS
│   └── autostart.sh           #   Automazione menu TORCS (via xte/xautomation)
├── telemetry/                 # Telemetria CSV dei test agent (auto-generata)
└── train_set/                 # Dati e Checkpoint
    ├── laps/                  #   File HDF5 dei giri registrati (lap_001.h5 ...)
    ├── checkpoints/           #   Pesi: bc_policy.pth, td3_policy.pth, td3_expl_best_lap.pth, td3_expl_best_dist.pth, td3_det_best_dist_run.pth, td3_det_best_dist.pth, td3_det_best_lap.pth
    │   ├── backups/           #   Copie .bak/.prev ordinate per recupero anti-interruzione
    │   │   └── buffers/       #   Backup .bak/.prev dei replay buffer
    │   └── buffers/
    │       ├── td3_checkpoint_buffer.npz        # Replay Buffer standard compresso (numpy)
    │       └── td3_checkpoint_elite_buffer.npz  # Elite Buffer compresso (numpy)
    └── session_logs/          #   Log delle sessioni di training
```

---

## Istruzioni d'Uso

### Prerequisiti

- Python 3.8+ con PyTorch, h5py, numpy, pygame
- TORCS installato con circuito **Corkscrew** e vettura **F1**
- `xautomation` (per `xte`, usato da `autostart.sh` per navigare i menu TORCS)
- Controller PS5 DualSense (opzionale, altrimenti tastiera WASD)

### 1. Raccolta Dati (Data Collection)

Registra giri puliti guidando manualmente. Solo i giri completati senza uscire di pista (`|trackPos| < 1.25`) vengono salvati come file HDF5 separati.

```bash
# Con controller PS5 DualSense
python data_collection.py --output_dir train_set --device controller

# Con tastiera (WASD + frecce per le marce)
python data_collection.py --output_dir train_set --device keyboard

# Raccolta MIRATA: guidi giri interi, il controller VIBRA (gentile) all'ingresso di
# ogni curva stretta, e con --segment_only vengono salvati SOLO i segmenti di quelle curve.
python data_collection.py --output_dir train_set --device controller --segment_only

# Override delle zone (default = PROBLEM_ZONES auto-rilevate per geometria della pista)
python data_collection.py --output_dir train_set --device controller --segment_only --zones "670:810,940:1070"
```

**Feedback aptico (non log)**: entrando in una zona-curva target il **controller vibra brevemente** (gentile, ~30%) — così sai quando sei nella curva senza leggere lo schermo mentre guidi. Le zone sono le **curve strette auto-rilevate dalla geometria** della pista (sensore frontale `<0.25`, ~9 tornanti su Corkscrew), **robuste ai tuoi errori di guida** (frenate/sterzate fuori posto: contano solo la forma della pista, non i tuoi input). Con `--segment_only` la raccolta è **parziale**: guidi giri interi ma vengono tenuti solo i segmenti dentro le zone (file `lap_seg_*.h5`, con margine di approccio per lo stacking).

**Output:** un file `train_set/laps/lap_NNN.h5` per ogni giro valido. Ogni file contiene `states` (29D), `actions` e il **metadato `dist_from_start`** (posizione per step). `dist_from_start` è solo un'etichetta di posizione per le analisi — **NON** entra nella rete, che resta **29D**.

### 2. Addestramento BC (Behavioral Cloning)

```bash
# Metodo rapido
./train_bc.sh

# Oppure direttamente con parametri personalizzati
python behavioral_cloning.py \
    --dataset train_set/laps \
    --epochs 300 \
    --batch_size 256 \
    --lr 3e-4 \
    --output train_set/checkpoints/bc_policy.pth
```

**Output:** `train_set/checkpoints/bc_policy.pth`

### 3. TD3+BC Fine-Tuning (Reinforcement Learning)

```bash
# Metodo rapido (1000 episodi, Warm-Start automatico dal BC)
./train_rl.sh

# Override del numero di episodi
TD3_EPISODES=500 ./train_rl.sh

# Ripartenza pulita (cancella checkpoint TD3 precedenti)
./train_rl.sh --clean

# Ripartenza di recupero: carica la migliore policy deterministica e congela l'Actor per 30 episodi
./train_rl.sh --rollback

# Recupero conservativo: piu' tempo al Critic e nessuna refinement automatica
./train_rl.sh --rollback --actor-freeze-episodes 100 --no-auto-refine

# Training normale senza trigger automatico di refinement
./train_rl.sh --no-auto-refine

# Refinement immediata: aggiornamento del Critic disattivato, loss Critic solo diagnostica, peso Behavioral Cloning ridotto
./train_rl.sh --refine

# Lancio diretto
python td3_bc.py \
    --bc_weights train_set/checkpoints/bc_policy.pth \
    --episodes 1000 \
    --max_steps 5000 \
    --seed 42
```

Il training è **resume-safe**: ad ogni episodio salva buffer e checkpoint in modo atomico, mantenendo anche `*.bak` e `*.prev` come ultime due copie complete recuperabili dentro `train_set/checkpoints/backups/`. Puoi interromperlo con `Ctrl+C` e riprenderlo in qualsiasi momento; se il checkpoint principale è corrotto, il loader prova automaticamente i backup recenti prima di cadere sul recupero minimo da log.

Il training avviene in modo isolato in un Virtual Framebuffer (`Xvfb`) per prevenire problemi di focus con il desktop dell'host.

**Output:** `train_set/checkpoints/td3_policy.pth` + `td3_expl_best_lap.pth` + `td3_expl_best_dist.pth` + `td3_det_best_dist_run.pth` + `td3_det_best_dist.pth` + `td3_det_best_lap.pth` (quando chiude un giro valido deterministico) + `td3_checkpoint.pth` + `buffers/td3_checkpoint_buffer.npz` + `buffers/td3_checkpoint_elite_buffer.npz` + backup automatici in `backups/`.

### 4. Test Deterministico (Inference)

Il test agent auto-rileva i migliori pesi disponibili: `td3_det_best_lap.pth` → `td3_det_best_dist.pth` → `td3_det_best_dist_run.pth` → `td3_expl_best_lap.pth` → `td3_expl_best_dist.pth` → `td3_policy.pth` → `bc_policy.pth`.

> `td3_det_best_lap.pth` è il **miglior giro VALIDO completato in valutazione deterministica** (il più veloce), con sidecar `.txt` che ne riporta il tempo: è il **candidato diretto per la submission** (giro valido + tempo minimo + riproducibile). Anch'esso **sopravvive a `--clean`**.

> `td3_det_best_dist.pth` è la **migliore policy assoluta tra tutti i run** e — a differenza degli altri — **sopravvive a `--clean`** (così non si perde mai un buon risultato per un restart sfortunato).

> **BC vs RL — rilevamento per nome file**: la mappatura azioni (RL: `tanh→[0,1]`; BC: `sigmoid`) viene scelta in base al **nome del file** (`td3_*`/`sac_*` = RL, `bc_*` = BC), **non** dalla presenza di `log_std_head` (che i vecchi checkpoint BC possono contenere). Caricare un BC come se fosse RL applicherebbe la de-normalizzazione sbagliata su gas/freno.

```bash
# Esecuzione standard con bypass Xvfb (visibile a schermo)
SHOW_GUI=1 python test_agent.py

# Vedere a schermo il miglior giro valido deterministico, quando disponibile
SHOW_GUI=1 python test_agent.py --weights train_set/checkpoints/td3_det_best_lap.pth --laps 1

# Vedere la migliore policy deterministica per distanza
SHOW_GUI=1 python test_agent.py --weights train_set/checkpoints/td3_det_best_dist.pth --laps 1

# Vedere la policy migliore ottenuta nel run corrente
SHOW_GUI=1 python test_agent.py --weights train_set/checkpoints/td3_det_best_dist_run.pth --laps 3
```

### Script di Supporto

```bash
./stop_training.sh      # Ferma training e TORCS
```

---

## Interpretazione dei Log di Addestramento TD3+BC

Durante il training RL, il log stampa metriche fondamentali per diagnosticare la salute dell'addestramento. Ecco i valori corretti da aspettarsi:

Il campo `LapTime` arriva dalla telemetria TORCS: `lastLapTime` quando l'episodio chiude un giro valido (`[SUCCESS]`), `curLapTime` negli altri casi. Non è derivato da `Steps * 0.02`, perché il passo del server non è una garanzia affidabile del tempo gara.

### 1. Loss del Critic
* **Cos'è:** Misura l'errore (MSE) del Critic nel prevedere le reward future.
* **Valori Sani:** Grazie al *Reward Scaling* implementato, i valori ottimali oscillano **tra `0.01` e `5.0`** (con occasionali picchi isolati a `10-20` quando la macchina scopre porzioni di pista inedite).
* **Diagnosi:** Un valore stabilmente basso significa che il Critic sta fornendo stime coerenti. Se la loss del Critic schizza permanentemente a centinaia, c'è un'esplosione dei gradienti (o mancano i dati Behavioral Cloning nel replay buffer). In refinement le righe episodio mostrano `CriticL: ... (OFF)`: in quel caso la loss è calcolata e mostrata solo come diagnostica, ma il Critic **non** viene aggiornato.

### 2. Loss dell'Actor
* **Cos'è:** Misura quanto l'Actor sta massimizzando le reward stimate dal Critic.
* **Valori Sani:** con peso Behavioral Cloning costante a `1.0`, dopo il warm-up la loss dell'Actor si assesta intorno a **`-2.5`** (= `-λ`, dominanza del termine RL normalizzato). È il comportamento atteso del TD3+BC.
* **Diagnosi:** Una discesa progressiva della loss è segno che l'Actor sta abbandonando l'imitazione forte iniziale per capitalizzare sul Q-Value.

### 3. Dinamiche TD3+BC (Decay di Lambda)
* In TD3+BC l'entropia del SAC è rimossa, l'agente è completamente deterministico.
* **Peso Behavioral Cloning costante = 1.0:** $\lambda$ resta **fisso a `2.5`** (normalizzazione di Fujimoto & Gu, 2021) e il peso della penalità imitativa è **costante** — niente decay. La loss è esattamente quella del TD3+BC originale ($-\lambda Q + (\pi-a)^2$). Un decay del vincolo nella fase fragile causava "troppo RL troppo presto" → collasso (Beeson & Montana 2022, Ablation 1). L'ancora umana è inoltre **garantita in ogni batch** dal buffer expert separato (quota 25%).

### 4. Refinement (`--refine`)
La refinement è una fase separata per plateau stabili: il peso Behavioral Cloning scende a `0.3` e l'aggiornamento del Critic viene disattivato. Il trigger automatico è attivo di default dopo plateau statistico, ma viene sospeso quando l'Actor è congelato: quelle valutazioni servono solo a monitorare la policy corrente e non vengono usate per dichiarare plateau. Il trigger può essere disattivato con `./train_rl.sh --no-auto-refine` quando il Critic deve recuperare stabilità dopo rollback, checkpoint corrotto o nuovi dati expert. `./train_rl.sh --refine` resta il comando manuale per avviarla subito; nelle righe `[EVAL]` il log usa `rollback_ref=...` per indicare la soglia della rete di sicurezza, non un target da inseguire. Se la refinement produce un breakout vicino al miglior deterministico già preservato, esce subito in consolidamento: peso Behavioral Cloning torna a `1.0`, Critic riattivo, Actor temporaneamente congelato se configurato.

---

## Dettagli Tecnici

### PolicyNetwork / Actor Multi-Head

La rete adotta un'architettura **Multi-Head** per elaborare lo storico temporale:

| Parametro | Valore |
|-----------|--------|
| Input | 87 neuroni (vettore di osservazione 29D × 3 frame stacked) |
| Hidden Layers (Backbone) | 4 × 512 neuroni con LayerNorm + ReLU |
| Continuous Head | 3 neuroni (steer, accel, brake) |
| Gear Head (Discreta) | 7 neuroni (logits marcia per CrossEntropy) |
| Log Std Head (Legacy) | 3 neuroni (mantenuta per retrocompatibilità coi vecchi test_agent, ma isolata in TD3) |
| Parametri totali | ~843,000 |

**Perché 3 frame stacked:** l'input concatena `t-12`, `t-6`, `t` (circa 0.24s a 50Hz). Lo stato 29D contiene già velocità e rpm, ma lo stacking aggiunge la tendenza recente dei sensori pista, della posizione laterale e delle rotazioni ruota: aiuta a distinguere ingresso curva, correzione in corso e uscita curva senza introdurre feature derivate a mano. È una scelta compatibile con tutti i checkpoint attuali (BC, TD3, test); passare a 29D single-frame richiederebbe retrain BC + rivalidazione completa, quindi resta un'ablazione futura.

### TD3+BC Architecture

```
Actor (Warm-Start da BC)                    Critic (Twin Q-Network, da zero)
┌─────────────────────────┐                 ┌──────────────────────────┐
│  backbone [ALLENATO]    │                 │  Q1: (state+action) → 1 │
│  4×512 LayerNorm+ReLU   │                 │  512 → 512 → 1          │
│                         │                 ├──────────────────────────┤
│  continuous_head [ALLENATA] │ ←── TD3 ───→  │  Q2: (state+action) → 1 │
│  log_std_head   [CONGELATA] │    aggiorna    │  512 → 512 → 1          │
│  gear_head      [CONGELATA] │                 └──────────────────────────┘
└─────────────────────────┘                 + Target Q (Polyak τ=0.005)
```

**Training dell'Actor:** come nel TD3+BC originale, il TD3 allena **tutto l'Actor** (backbone + `continuous_head`) con LR=`3e-4`. La `gear_head` resta congelata ed è ormai **inutilizzata** (la marcia è calcolata dalla logica deterministica `gearing.py`, non più dalla rete); la `log_std_head` è legacy (retro-compatibilità). L'ancora Behavioral Cloning costante (`bc_weight=1.0` nel codice) previene il *Latent Shift* del backbone.

> **Marcia deterministica (`gearing.py`)**: la marcia non è predetta dalla rete ma da una funzione velocità-primaria anti-hunting (downshift sulla velocità, upshift solo sul gas+rpm). Validazione offline sui giri umani: ±1 marcia 99%, ~9.7 cambi/1000 step vs 322 della testa appresa; validazione live sulla policy RL: ~10.5 cambi/1000 step, senza oscillazioni rapide. Identica in training/eval/test. Vedi ARCHITECTURE §17.

**Critic Warm-Up (15.000 step):** I primi 15.000 step aggiornano solo il Critic. Questo protegge i pesi BC dai gradienti randomici di un Critic non ancora calibrato.

**Update Frequency 1:1:** un aggiornamento ad ogni step di simulazione (standard TD3). Con l'expert buffer sempre pieno il Critic si pre-allena sui dati umani già dal primo step.

### Reward Reshaping (da corsa, minimalista)

Formula per timestep in `gym_torcs.py`:

$$r_t = \underbrace{\tfrac{v_x}{50} \cos(\theta) \times 1.5}_{\text{progress}} \underbrace{- 2\,\max(0, |\text{trackPos}|-1)^2}_{\text{pos penalty (deadzone)}} \underbrace{- 0.05|\delta_t - \delta_{t-1}|}_{\text{steer smooth}}$$

- **Progress**: velocità in avanti normalizzata × `cos(angle)` × 1.5. È il termine dominante → giri veloci.
- **Pos Penalty (deadzone)**: **0** entro `|trackPos| < 1.0` (libertà piena), rampa morbida sui cordoli `1.0→1.25`. Lascia l'agente libero su staccate e linea.
- **Steer Smoothness**: lieve anti-zigzag (coeff. 0.05).
- **Terminali -10.0**: danno/muro, **`|trackPos| > 1.25` (giro non valido)**, spin, stallo.
- **Bonus giro VALIDO: +50.0**: TORCS aggiorna `lastLapTime` solo per giri senza tagli/uscite.
- *(La vecchia "Corner Overspeed Penalty" è stata RIMOSSA: creava un attrattore "vai piano" → collasso. L'agente è ora libero di scegliere le velocità in curva, purché resti valido e veloce.)*

### Replay Buffer Checkpointing

Il Replay Buffer viene salvato separatamente in formato `np.savez_compressed`:
- **File**: `buffers/td3_checkpoint_buffer.npz` e `buffers/td3_checkpoint_elite_buffer.npz` (~50-100MB compressi vs >1GB con pickle)
- **Previene il Catastrophic Forgetting** quando il training viene interrotto e ripreso
- **Resume-safe reale**: i buffer vengono salvati prima del checkpoint `.pth`, che agisce da commit finale. Ogni file mantiene `main` nella sua posizione operativa e `.bak`/`.prev` in `train_set/checkpoints/backups/`; al resume vengono provati in questo ordine.

### Done Masking

Nel TD3+BC il buffer salva un `mask` per la Bellman equation:
- **`mask=0.0`** → solo per terminazioni reali: fuoripista, spin, stallo o collisione.
- **`mask=1.0`** → time-limit (`max_steps`) e completamento giro, perché non sono crash.
- **Dati expert** → `mask=1.0` per tutti i campioni: un giro umano completato non azzera il valore futuro.

### Vettore di Osservazione (29D)

| Indice | Feature | Normalizzazione | Range tipico |
|--------|---------|-----------------|--------------|
| 0 | `angle` | nessuna (radianti) | [-0.6, 0.4] |
| 1–19 | `track[19]` (sensori LIDAR) | /200 (via `gym_torcs`) | [0, 1] |
| 20 | `trackPos` | nessuna | [-1, 1] |
| 21 | `speedX` | /50 (via `gym_torcs`) | [0, ~5.7] |
| 22 | `speedY` | /50 | [-0.6, 0.8] |
| 23 | `speedZ` | /50 | [-0.4, 0.7] |
| 24–27 | `wheelSpinVel[4]` | /100 | [0, ~2.6] |
| 28 | `rpm` | /10000 | [0.5, 2.0] |

> **IMPORTANTE:** ci sono due livelli distinti: scaling fisso fisico (`gym_torcs.make_observaton()` + `flatten_state()`) e normalizzazione statistica mean/std (`state_norm.npz`) applicata prima della rete. Non aggiungere nuove normalizzazioni ad hoc.

---

## Strategia di Ottimizzazione Dataset

### Il Limite dei Dati Troppo Omogenei
Se il dataset contiene unicamente giri perfetti lungo l'identica traiettoria ideale, l'agente non apprenderà mai cosa fare fuori da quella linea → **Covariate Shift**.

### Come Raccogliere Dati di Recupero Efficaci
Registrare **5-10 giri aggiuntivi** con:
1. **Partenze Fuori Asse**: `trackPos ≈ ±0.8`, guidando verso il centro
2. **Correzioni in Rettilineo**: Oscillare dolcemente a destra e sinistra
3. **Ingressi Curva Alternativi**: Inserimenti larghi a velocità sub-ottimali

### Bojarski-Style Recovery Augmentation (con gating 50%)
Il training BC include perturbazione laterale dello stato (`trackPos ±0.4`) e angolare (`angle ±0.08 rad`) con correzione proporzionale dello sterzo e del freno target. **La perturbazione è applicata solo al 50% di ogni batch** (gating per-campione): l'altra metà resta pulita, così la rete impara *sia* la guida precisa sulla linea ideale *sia* il recupero da stati OOD. Senza il gating (perturbazione al 100%) la fedeltà di sterzo degradava (steer MAE 0.116 → 0.062 col gating).

### Nota storica: oversampling pesato per posizione (rimosso)
Un meccanismo che pesava di più i campioni di una curva nella loss BC è stato **testato e RIMOSSO**: degradava il closed-loop (la policy regrediva, ~236m invece di ~811m). Il bilanciamento si fa con la **quantità di dati reali** raccolti sulla curva (`data_collection --segment_only` → solo expert buffer RL). Il metadato `dist_from_start` resta nei giri solo come etichetta per analisi: **NON** entra nella rete, che resta **29D**.

---

## Bug Risolti (Workflow Tracking)

Le voci più vecchie sono cronologia tecnica: possono citare SAC o nomi checkpoint storici, ma la fonte di verità operativa attuale è la sezione TD3+BC sopra e ARCHITECTURE.md.

### [2026-06-07] Cambio Marcia Deterministico + Falso Stallo in Test

**Problema 1 — Hunting del cambio:** La `gear_head` appresa (congelata durante l'RL) produceva hunting estremo (fino a ~322 cambi ogni 1000 step, con assurdità tipo 1ª a 150 km/h) che destabilizzava l'intero giro e spezzava la trazione. Essendo congelata, l'RL non poteva correggerla.

**Fix:** Sostituita con `gearing.compute_gear`, logica deterministica **velocità-primaria** (vedi ARCHITECTURE §17). Il downshift guarda la velocità (monotòna in frenata) → il classico problema del picco-rpm in staccata (che fa ri-salire di marcia gli shifter ingenui) sparisce per costruzione; l'upshift scatta solo sul gas. Soglie derivate e validate sui 75 giri umani (**±1 marcia 99%**, 0% fuorigiri) **e dal vivo sulla policy RL** (10.5 cambi/1000, 0 oscillazioni rapide). Usata identica in training/eval/test.

**Problema 2 — Falso stallo in test:** `test_agent.py` rilevava uno stallo fasullo a ~550 step. Causa: leggeva la velocità da `next_state[21]*50`, ma `next_state` è ora **normalizzato** (`apply_state_norm`, mean/std) → a velocità sotto-media il valore diventa negativo → `fwd_kmh < 5` fasullo. **Fix:** leggere la velocità dall'obs grezzo `next_obs['speedX']*50`. *(Bug presente solo nel test, non nel training.)*

**Problema 3 — Segmenti concentrati avvelenano il BC:** Aggiungendo 18 segmenti della sola Corkscrew al dataset BC, gli eval di warm-up sono crollati da ~400-818m a ~19-188m (l'agente usciva di pista già a curva 1). **Root cause:** il BC è cieco alla posizione (29D, no `distFromStart`) e minimizza l'errore medio → la sterzata pesante di una curva concentrata "trabocca" su stati simili altrove. **Fix — Split dati BC/RL** (ARCHITECTURE §7, Livello 3): il BC carica solo i **giri interi** (`lap_[0-9]*.h5`, distribuzione bilanciata), l'**RL expert buffer** carica anche i **segmenti** (`lap_seg_*.h5`). La quota **25% Expert** è applicata durante il sampling del batch TD3+BC, non come quota separata di caricamento dei segmenti. Verificato: BC ri-allenato sui soli giri interi → guida di nuovo bene. *(Escluso anche il mismatch marce manuali/algoritmiche come causa: il BC pulito + `compute_gear` guida bene da subito → covariate shift tollerabile.)*

### [2026-06-04] Risoluzione OOD BC Bug e Relaxed Policy Constraint

**Problema:** L'Actor dimenticava come guidare in modo deterministico (Catastrophic Forgetting), esibendo un plateau fisso della loss dell'Actor a ~2.516 durante i crash.

**Root Cause:** L'ancora Behavioral Cloning congelata (la `bc_policy`) veniva interrogata per calcolare la penalità imitativa anche durante gli stati OOD (fuoripista, muri). La rete restituiva azioni allucinate, bloccando l'apprendimento delle manovre di recupero.

**Fix applicati:**
1. **Masking Rigoroso**: La BC Penalty è calcolata esclusivamente sui campioni empirici registrati nell'Elite Buffer (dove `expert_mask=1.0`). È azzerata durante le fasi esplorative online, liberando l'Actor.
2. **Rimozione bc_policy**: Eliminato del tutto il clone congelato dell'Actor (-2.3M parametri in VRAM).
3. **Relaxed Policy Constraint (Decay Esponenziale)**: Transizione del coefficiente imitativo $\lambda$ da 2.5 a 0.25 su 100k step, ispirato a Beeson & Montana (2022).

**Nota stato attuale:** questa variante con decay è storica ed è stata sostituita dal TD3+BC corrente: `bc_weight=1.0` costante nel training normale, con allentamento solo nella fase separata di refinement (`bc_weight=0.3`, Critic non aggiornato).

### [2026-06-03] Risoluzione Definitiva del Collasso della Policy (6 Bug Fix)

**Problema:** La policy collassava dopo 100-1000 episodi di training, entrando in un loop di reward negative e stalli. L'agente "dimenticava come guidare" in modo irreversibile.

**Root Cause:** Catena di 6 bug interconnessi:
1. LR Actor troppo alto (`3e-4`) distruggeva i pesi BC in 50-100 episodi
2. BC Penalty con decay esponenziale lasciava la policy senza àncora
3. Done masking contraddittorio tra dati expert (mask=0.0 per completamento) e dati online (mask=0.0 per crash) — il Critic riceveva segnali opposti
4. Alpha auto-tuning causava escalation entropica (0.02 → 0.05 → ...) che aggiungeva rumore crescente
5. Update ratio 1:1 (ogni step) causava overfitting sulle stesse transizioni
6. Nessun segnale di completamento giro — il Critic non distingueva successo da crash

**Fix applicati:**
1. **LR Differenziati**: `continuous_head` = `1e-5`, `log_std_head` = `1e-4`. Protegge i pesi BC calibrati.
2. **Peso Behavioral Cloning fisso = 5.0**: Nessun decay. L'Actor resta permanentemente ancorato al BC (Residual RL).
3. **Done Masking Corretto**: I dati expert usano `mask=1.0` per tutti i campioni (giri completati ≠ crash).
4. **Alpha Fisso = 0.01**: Disattivato l'auto-tuning dell'entropia.
5. **Update Ratio 1:4**: Aggiornamento ogni 4 step per ridurre l'overfitting.
6. **Bonus Completamento Giro = +50.0**: Segnale esplicito per il Critic.
7. **Evaluation Periodica Deterministica**: Ogni 25 episodi, checkpoint deterministico riproducibile per il sistema SAC storico.

**Nota stato attuale:** i valori di questa voce descrivono una configurazione superata. Il TD3+BC corrente usa LR `3e-4` per Actor e Critic, `bc_weight=1.0`, aggiornamento Critic 1:1 e Actor ritardato ogni 2 update (`policy_freq=2`).

### [2026-05-31] Risoluzione del Collasso della Policy (Stall Trap & Q-Value Explosion)

**Problema:** L'agente soffriva di uno "Stall Trap" alla partenza a causa di un *overestimation bias* critico (loss del Critic > 2000), seguito dal collasso della rete. Questo era indotto da Alpha azzerato forzatamente, iniezione di rumore scorretta e una soglia dell'elite buffer degenerata.

**Fix applicati:**
1. **Riattivazione Auto-Tuning Entropia**: Ripristinata l'ottimizzazione dinamica di `alpha`. L'entropia agisce ora come regolarizzatore nell'equazione di Bellman per frenare l'overestimation bias causata da Q-Values asintotici in stati *OOD*.
2. **Rimozione Rumore Manuale**: Eliminato l'uso di `np.random.normal(0, 0.05)` a valle dell'Actor. L'esplorazione è ora gestita interamente in modo nativo dal campionamento del SAC (`log_std`), rimuovendo il *distillation error* che forzava l'Actor a imparare il proprio tremolio.
3. **Calcolo Deterministico BC Penalty**: La distorsione introdotta dall'uso stocastico è stata risolta calcolando la loss (MSE) tra l'azione umana e l'azione deterministica pre-tanh (`torch.tanh(mean)`), separando la varianza esplorativa dal target direzionale.
4. **Monotonicità Elite Threshold**: La soglia per l'ammissione nell'Elite Buffer non decade più progressivamente, ma dipende strettamente dal record assoluto globale (`best_distance * 0.9`), prevenendo avvelenamenti causati da runs mediocri.
5. **Soft Mutual Exclusion e Critic LR**: Modificata la BC Penalty per penalizzare esplicitamente la pressione simultanea di acceleratore e freno, proteggendo l'addestramento. Sostituito il decay lineare del peso BC con un decadimento Esponenziale Smorzato e abbassato il Learning Rate del Critic a `1e-4` per rallentare l'assimilazione del gradiente ed evitare l'Extrapolation Error.
6. **Risoluzione Scalar Shock (BC Penalty Normalization)**: Normalizzata la loss direzionale e ridotta la magnitudine della *Mutual Exclusion Penalty* per prevenire l'overpowering del gradiente. Questo impedisce il crollo della deviazione standard (Variance Collapse) che precedentemente induceva un panico entropico e l'esplosione distruttiva di Alpha.

### [2026-05-30] Stabilizzazione SAC: Expert Buffer Injection, Gradient Clipping e Reward Scaling

**Problema:** L'Actor collassava istantaneamente (Catastrophic Forgetting) al termine dei 5000 step di Warm-Up del Critic, incapace di guidare oltre i primi metri a causa di gradienti esplosivi e di un crollo deterministico indotto dai Q-values sbilanciati.

**Fix applicati:**
1. **Expert Buffer Injection**: Precaricamento di 50.000 memorie BC nel Replay Buffer per addestrare il Critic su traiettorie ottimali fin dal primo step.
2. **Gradient Clipping**: Capping della norma dei gradienti a 1.0 (tramite `clip_grad_norm_`) per Actor e Critic, mitigando il "Critic Shock".
3. **Alpha Autotuning Fix**: Aumento del learning rate di `log_alpha` a `3e-4` per permettere al termostato entropico di reagire tempestivamente alla perdita di stochasticità.
4. **Reward Scaling**: Scalate le ricompense a `reward * 0.02` per bilanciare matematicamente i Q-values con la loss entropica (`alpha * log_pi`), prevenendo l'**Entropy Annihilation** e garantendo un'esplorazione stabile a lungo termine.
5. **Alpha Poisoning Clamp**: Abbassato l'hard clamp di `log_alpha` a `-3.0` (Alpha max **5%**) per impedire che l'entropia inietti un rumore fatale (>20%) per la precisione di guida di una Formula 1, proteggendo il Replay Buffer da schianti continui.
6. **Actor Trust Region (Micro-LR)**: Abbassato drasticamente il Learning Rate dell'Actor da `3e-4` a `1e-5`. Senza una regolarizzazione BC esplicita, questo impedisce l'**Extrapolation Error** e il Policy Drift, facendo sì che l'Actor compia passi microscopici e sicuri quando valuta gradienti OOD calcolati dal Critic.
7. **Architectural Action Space Mismatch**: Risolto un bug critico di mappatura dove il BC model emetteva valori `[0, 1]` (tramite sigmoide) ma l'Actor SAC emetteva valori `[-1, 1]` (tramite tanh), causando output nulli e stalli continui. Le azioni vengono ora ri-mappate istantaneamente a `[0, 1]` appena prima dell'invio al simulatore, preservando la simmetria del SAC e i pesi originali del BC.

### [2026-05-30] Risoluzione Definitiva dello "Stall Trap" (Differentiability Cliff & Tanh Paradosso)

**Fix applicati:**
1. **Rimozione Mutual Exclusion**: Eliminato l'`if brake > 0.05: accel = 0.0` in `action_to_env`. Questa regola, pensata per gli umani, era un "Differentiability Cliff" che uccideva il flusso del gradiente e spegneva inaspettatamente il motore durante le esplorazioni incerte dell'agente.
2. **Ambiente Esplorativo (Anti-Stall Relaxed)**: In `gym_torcs.py`, il timer di antistallo originale uccideva spietatamente l'agente a 3.0 secondi esatti (150 step) se andava a meno di 20 km/h. La regola è stata allentata (10 secondi, 5 km/h) per permettere all'Actor di muovere i primi passi con cautela senza subire falsi negativi fatali.
3. **Disattivazione Alpha Auto-Tuning**: Risolto il paradosso dello schiacciamento del `tanh` ai limiti. Nelle corse ("bang-bang" actions come gas a 1.0), lo Jacobiano esplode penalizzando infinitamente l'Actor e forzandolo a stallare. L'Alpha è stato fissato a `0.02` (Entropia Costante) disattivando l'ottimizzatore, tecnica standard per il RL in ambienti limitati, stabilizzando permanentemente i gradienti esplosivi.

### [2026-05-29] Finalizzazione Architettura (Xvfb, Best Lap, Pure SAC)

**Fix applicati:**
1. **Ambiente Isolato Xvfb:** TORCS e le macro girano confinati in un virtual display senza rubare focus. Usare `SHOW_GUI=1` per lo sblocco in rendering locale.
2. **Auto-Entropy Tuning**: Introdotto tuning automatico del `log_alpha` per un corretto calcolo del SAC.
3. **Loss dell'Actor SAC puro**: Rimossa la logica fallata TD3+BC dalla fase Online. L'Actor massimizza unicamente entropia e Q-Value target senza auto-imitare il proprio rumore di addestramento.
4. **Dual Checkpointing**: Il sistema SAC storico salvava checkpoint separati per giro da record e distanza massima prima dello schianto. Logging semantico per gli episodi `[SUCCESS]`, `[CRASH]` o `[TIMEOUT]`.
5. **Permanent BC Adherence (Residual RL)**: Implementata una strategia di "guinzaglio" asintotico per la penalità di Behavioral Cloning. Il peso decade lentissimamente (da 10.0 a 2.0 in 500.000 step) ma non arriva *mai* a zero. Questo previene il *Policy Collapse* causato dall'Extrapolation Error, obbligando l'Actor a restare ancorato alla fisica della policy BC, e sfruttando i gradienti Q-Value del SAC unicamente come affinamento locale (Residual RL) per ottimizzare le curve in cui la media umana fallisce.
6. **Alpha Math-Fix e Reward Scaling**: Disattivata l'Entropia SAC (`alpha = 0.0`) per curare in via definitiva l'esplosione dei gradienti ai confini del dominio `tanh` (`gas a tavoletta`). Per compensare l'iper-ottimismo del Critic (che porta all'Extrapolation Error summenzionato), la `reward_scale` è stata abbattuta a `0.002`, armonizzando i Q-value generati da un `gamma = 0.999`.
7. **Gamma Horizon Fix (50Hz Myopia)**: Aumentato il discount factor `gamma` da `0.99` a `0.999`. In un simulatore a 50Hz, `gamma=0.99` limitava l'orizzonte visivo del Q-Value a soli 2 secondi (100 step), rendendo la partenza (speed=0) indistinguibile da uno stallo e portando l'Actor a massimizzare l'entropia (0.5 gas, 0.5 freno). Con `gamma=0.999`, l'orizzonte si espande a 20 secondi, permettendo al Critic di ricompensare l'accelerazione a lungo termine.

### [2026-05-29] Migrazione Ibrida BC-RL (SAC) — ARCHITETTURALE

**Problema:** Il sistema SAC soffriva di molteplici bug che impedivano la convergenza:
1. Penalità terminali a `-100.0` e bonus sparsi a `+200.0` → esplosione del TD-Error
2. `action_to_env` con round-trip `arctanh → sigmoid` → asintoti e NaN
3. Replay Buffer salvato come pickle nella deque → checkpoint >1GB
4. `test_agent.py` incompatibile con pesi SAC (manca `log_std_head`)
5. Time-limit imposta `done=True` nel buffer → bias nel Critic

**Fix applicati:**
1. **Reward Reshaping**: Progress ×10, time penalty -1.0, tutte le penalità cappate a -50.0
2. **action_to_env**: Sostituito con trasformazione affine `(x+1)/2` stabile
3. **Replay Buffer**: Salvataggio separato con `np.savez_compressed`
4. **BCActor**: Nuova classe con `log_std_head` + `sample(evaluate=True)` deterministico
5. **Done Masking**: Time-limit non imposta `done=True` nel buffer
6. **train_rl.sh**: Script dedicato con Warm-Start detection e `--clean` flag
7. **test_agent.py**: Auto-detect formato pesi (BC vs SAC), dual denormalize

### [2026-05-28] Ottimizzazione Stride Temporale e Definizione dell'Architettura 29D (v4)

**Soluzione Definitiva:** Rimozione `distFromStart`, consolidamento a k=6 (0.24s), stacking 87D.

### [2026-05-24] Covariate Shift e Bojarski-Style Augmentation

**Soluzione:** Perturbazione laterale + correzione proporzionale sterzo + deprecazione heuristics.

### [2026-05-19] Doppia Normalizzazione Features — CRITICO

**Fix:** Rimossa doppia normalizzazione da `TorcsHDF5Dataset`, allineato `flatten_state()`.

---

**IBM AI Racing League 2026** — *Precision Driving through Hybrid BC-RL.*
