# 🏎️ AIcar — Behavioral Cloning Architecture (IBM AI Racing League 2026)

**Agente autonomo che impara a replicare il "giro perfetto" tramite Behavioral Cloning (BC).**

Questo repository implementa una pipeline end-to-end per addestrare un agente di guida autonoma nell'ambiente di simulazione **TORCS** (The Open Racing Car Simulator). L'obiettivo: **fittare perfettamente i dati esperti** per produrre una policy deterministica capace di completare un giro di qualifica pulito (senza collisioni o track-cut) sul circuito **Corkscrew** con una vettura **F1**.

---

## 🧠 Filosofia del Progetto: Pure Behavioral Cloning

A differenza degli approcci ibridi, questo progetto punta sulla **massima fedeltà ai dati esperti**. Invece di esplorare traiettorie casuali tramite RL, l'agente utilizza una **Deep Policy Network** (4 layer nascosti, 30D → 512D) per mappare esattamente ogni sensore alla risposta corretta del pilota.

### Punti di forza della pipeline BC:
1. **Stabilità Assoluta**: Nessun rischio di *catastrophic forgetting* o divergenza tipica del RL.
2. **Determinismo**: A parità di stato iniziale, l'agente produrrà sempre la stessa traiettoria ideale.
3. **Efficienza**: Il training richiede pochi minuti su GPU anziché ore di interazione con il simulatore.

---

## 🏛️ Architettura del Sistema

La pipeline si compone di tre fasi sequenziali:

```
┌──────────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
│  Fase 1              │     │  Fase 2              │     │  Fase 3              │
│  DATA COLLECTION     │────▶│  BC TRAINING         │────▶│  TEST / INFERENCE    │
│                      │     │                      │     │                      │
│  🎮 PS5 / Tastiera   │     │  behavioral_cloning  │     │  test_agent.py       │
│  data_collection.py  │     │  .py                 │     │  Zero-Noise          │
│                      │     │                      │     │  Deterministico      │
│  Output:             │     │  Output:             │     │                      │
│  train_set/laps/     │     │  train_set/          │     │  Valutazione live    │
│  lap_001.h5 ...      │     │  checkpoints/        │     │  su TORCS            │
│                      │     │  bc_policy.pth       │     │                      │
└──────────────────────┘     └──────────────────────┘     └──────────────────────┘
```

| Fase | Script | Descrizione |
|------|--------|-------------|
| 1. Data Collection | `data_collection.py` | Raccolta di giri guidati da umano (esperto) con controller PS5 DualSense o tastiera WASD. Solo i giri completati senza uscite di pista vengono salvati. |
| 2. BC Training | `behavioral_cloning.py` | Addestramento della PolicyNetwork sui dati esperti, con validation split 80/20 e early stopping. |
| 3. Test & Eval | `test_agent.py` | Esecuzione deterministica del modello BC su TORCS per valutare la capacità di completare giri autonomi. |

---

## 📁 Struttura del Repository

```
AIcar/
├── data_collection.py         # Fase 1: Raccolta dati umani (PS5 / Tastiera)
├── behavioral_cloning.py      # Fase 2: Training della Deep Policy Network
├── test_agent.py              # Fase 3: Inferenza deterministica su TORCS
├── train_all.sh               # 🚀 Script unico per lanciare il training BC
├── stop_training.sh           # 🛑 Ferma i processi di training/TORCS
├── monitor.sh                 # 📊 Monitoraggio status processi e checkpoint
├── README.md
├── gym_torcs/                 # Wrapper Python per TORCS
│   ├── gym_torcs.py           #   Ambiente OpenAI Gym per TORCS
│   ├── snakeoil3_gym.py       #   Client UDP per comunicazione con TORCS
│   └── autostart.sh           #   Automazione menu TORCS (via xte/xautomation)
└── train_set/                 # Dati e Checkpoint
    ├── laps/                  #   File HDF5 dei giri registrati (lap_001.h5 ...)
    ├── checkpoints/           #   Pesi del modello (bc_policy.pth)
    └── session_logs/          #   Log delle sessioni di data collection
```

---

## ⚙️ Istruzioni d'Uso

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
```

**Opzioni utili:**
- `--steering_deadzone 0.05` — Deadzone dello sterzo per il controller (default: 0.05)
- `--tcs` / `--no-tcs` — Abilita/disabilita il Traction Control System (default: abilitato)
- `--tcs_slip 5.0` — Soglia di slip per il TCS (default: 5.0)
- `--relaunch_every 10` — Rilancia TORCS ogni N giri per prevenire memory leak

**Output:** un file `train_set/laps/lap_NNN.h5` per ogni giro valido, contenente:
- `states`: matrice `(N_steps, 30)` — vettore di osservazione 30D normalizzato
- `actions`: matrice `(N_steps, 4)` — `[steering, accel, brake, gear]`
- Attributi: `lap_time`, `num_steps`, `timestamp`

### 2. Addestramento BC (Training)

```bash
# Metodo rapido (usa gli hyperparameter di default)
./train_all.sh

# Oppure direttamente con parametri personalizzati
python behavioral_cloning.py \
    --dataset train_set/laps \
    --epochs 300 \
    --batch_size 256 \
    --lr 3e-4 \
    --output train_set/checkpoints/bc_policy.pth
```

Il training utilizza:
- **Validation Split 80/20** con **Early Stopping** (patience 30 epoche) per evitare overfitting.
- **Cosine Annealing LR** da `3e-4` fino a `1e-6` per una convergenza stabile.
- **Loss Combinata Multi-Head**: $\mathcal{L} = \mathcal{L}_{\text{MSE continua}} + 2 \times \mathcal{L}_{\text{CrossEntropy gear}}$.
- **Dynamic Brake Boost (25x)**: l'errore sul canale del freno è pesato 25× nei campioni con frenata attiva dell'umano (`brake_target > 0.05`) per forzare staccate vigorose a runtime ed eliminare lo sbilanciamento del dataset (dove il freno è spento per il 95% del tempo).
- **Steer Boost (3x)**: errore del canale di sterzata pesato 3× nelle curve strette.
- **Data Augmentation con Rumore Strutturato Coerente**: rumore su `trackPos` calibrato a `0.05` per forzare la robustezza al *covariate shift* spaziale (politica di recupero aderenza), e rumore su `speedX` confinato a `0.01` per preservare il tempismo delle marce e delle staccate.
- **Gradient Clipping** (max_norm=1.0) per la stabilità dei gradienti.

**Output:** `train_set/checkpoints/bc_policy.pth`

### 3. Test Deterministico (Inference)

Avvia TORCS e lancia l'agente autonomo. Il modello guida in modalità interamente deterministica ed end-to-end (senza alcuna limitazione o guardrail euristico a runtime).

```bash
python test_agent.py --weights train_set/checkpoints/bc_policy.pth --laps 3
```

**Opzioni:**
- `--laps N` — Numero di giri da completare (default: 3)
- `--max_steps N` — Timeout per giro in step (default: 15000)

### Script di Supporto

```bash
./monitor.sh            # Mostra stato dei processi e checkpoint
./stop_training.sh      # Ferma training e TORCS (SIGTERM)
./stop_training.sh --force  # Kill forzato (SIGKILL)
```

---

## 🔧 Dettagli Tecnici

### PolicyNetwork Multi-Head

La rete Actor non è più a regressione continua singola (che portava a predizioni decimali confuse per la marcia come `2.5`, ritardando le scalate e le staccate). Adotta ora un'architettura **Multi-Head**:
- **Backbone Comune**: 4 layer densi (512 unità ciascuno) con `LayerNorm` e attivazione `ReLU`. Estrae feature spaziali e cinematiche condivise dallo stato 30D.
- **Testa Continua**: output a 3 dimensioni per il controllo dello sterzo e dei pedali:
  - `steer`: attivato via **Tanh** in $[-1, 1]$ per sterzate simmetriche.
  - `accel` e `brake`: attivati via **Sigmoid** in $[0, 1]$ per mappare naturalmente i pedali.
- **Testa Discreta**: output a 7 logit discreti per il cambio marcia (`gear` in $\{0, 1, 2, 3, 4, 5, 6\}$), addestrata tramite `CrossEntropyLoss`.

| Parametro | Valore |
|-----------|--------|
| Input | 30 neuroni (vettore di osservazione) |
| Hidden Layers (Backbone) | 4 × 512 neuroni con LayerNorm + ReLU |
| Continuous Head | 3 neuroni (steer [Tanh], accel [Sigmoid], brake [Sigmoid]) |
| Gear Head (Discreta) | 7 neuroni (logits marcia per CrossEntropy) |
| Parametri totali | ~813,000 |

### Vettore di Osservazione (30D)

Lo stato è un vettore 1D di 30 valori, costruito da `flatten_state()`:

| Indice | Feature | Normalizzazione | Range tipico |
|--------|---------|-----------------|--------------|
| 0 | `angle` | nessuna (radianti) | [-0.6, 0.4] |
| 1–19 | `track[19]` (sensori LIDAR) | /200 (via `gym_torcs`) | [0, 1] |
| 20 | `trackPos` | nessuna | [-1, 1] |
| 21 | `speedX` | /50 (via `gym_torcs`) | [0, ~5.7] |
| 22 | `speedY` | /50 (via `gym_torcs`) | [-0.6, 0.8] |
| 23 | `speedZ` | /50 (via `gym_torcs`) | [-0.4, 0.7] |
| 24–27 | `wheelSpinVel[4]` | /100 | [0, ~2.6] |
| 28 | `rpm` | /10000 | [0.5, 2.0] |
| 29 | `distFromStart` | /4000 | [0, ~0.9] |

> ⚠️ **IMPORTANTE:** La normalizzazione dei sensori avviene in due punti della pipeline e NON deve essere duplicata:
> - `gym_torcs.make_observaton()` normalizza `track/200` e `speed/default_speed(50)`
> - `data_collection.flatten_state()` normalizza `wheelSpinVel/100`, `rpm/10000`, `distFromStart/4000`
>
> `TorcsHDF5Dataset` e `test_agent.flatten_state()` NON devono ri-normalizzare track e speed.

### Mapping delle Azioni

Gli output della PolicyNetwork Multi-Head vengono convertiti in azioni TORCS in modo simmetrico all'addestramento:

| Azione | Provenienza Rete | Range Rete | → Range TORCS | Decodifica |
|--------|------------------|------------|---------------|------------|
| Steering | Testa Continua (Tanh) | [-1, 1] | [-1, 1] | Diretto |
| Accelerator | Testa Continua (Sigmoid) | [0, 1] | [0, 1] | Diretto |
| Brake | Testa Continua (Sigmoid) | [0, 1] | [0, 1] | Diretto |
| Gear | Testa Discreta (Argmax Logits) | 7 classi | {0, 1, ..., 6} | Indice del logit massimo |

---

## 🐛 Bug Risolti (Workflow Tracking)

### [2026-05-19] Doppia Normalizzazione Features — CRITICO

**Problema:** `TorcsHDF5Dataset` ri-divideva `track/200` e `speed/100`, ma i dati in HDF5 erano già normalizzati da `gym_torcs.make_observaton()` durante la data collection.

**Impatto:** I 19 sensori LIDAR venivano compressi da range [0, 1] a [0, 0.005], rendendo l'agente cieco. La velocità passava da [0, 5.75] a [0, 0.057]. Il threshold `speed_x < 0.05` nella loss pesata era sempre vero (85.6% dei campioni), applicando un boost 5× indiscriminato.

**Fix applicato:**
- Rimossa la doppia normalizzazione da `TorcsHDF5Dataset` (`behavioral_cloning.py`)
- Allineato `test_agent.py/flatten_state()` a `data_collection.py/flatten_state()` (nessuna ri-divisione di track e speed)
- Sostituita la loss con `_weighted_mse` bilanciata: sterzo curva 3×, freno 2×, cambio 3× (era: sterzo 15×, cambio 10×, con low-speed boost 5× buggy)
- Aggiunta validation split 80/20 + early stopping + cosine LR + gradient clipping

---

**IBM AI Racing League 2026** — *Precision Driving through Behavioral Cloning.*
