# 🏎️ AIcar — Behavioral Cloning Architecture (IBM AI Racing League 2026)

**Agente autonomo che impara a replicare il "giro perfetto" tramite Behavioral Cloning (BC).**

Questo repository implementa una pipeline end-to-end per addestrare un agente di guida autonoma nell'ambiente di simulazione **TORCS** (The Open Racing Car Simulator). L'obiettivo: **fittare perfettamente i dati esperti** per produrre una policy deterministica capace di completare un giro di qualifica pulito (senza collisioni o track-cut) sul circuito **Corkscrew** con una vettura **F1**.

---

## 🧠 Filosofia del Progetto: Pure Behavioral Cloning

A differenza degli approcci ibridi, questo progetto punta sulla **massima fedeltà ai dati esperti**. Invece di esplorare traiettorie casuali tramite RL, l'agente utilizza una **Deep Policy Network Multi-Head** con **State Stacking Temporale Statico** per fittare l'orizzonte cinematico ideale.

L'agente utilizza un input di **87D** composto dalla concatenazione di 3 frame temporali con **passo di stride statico $k = 6$ ($0.24\text{ secondi}$ totali)**:
- $t - 12$ (passato)
- $t - 6$ (passato recente)
- $t$ (presente)

Questo orizzonte consente alla rete di calcolare in modo stabile i trend macroscopici e la derivata di avvicinamento ai bordi della pista.

### Punti di forza della pipeline BC:
1. **Stabilità Assoluta**: Nessun rischio di *catastrophic forgetting* o divergenza tipica del RL.
2. **Determinismo**: A parità di stato iniziale, l'agente produrrà sempre la stessa traiettoria ideale.
3. **Reattività Dinamica**: Lo stacking temporale mitiga la latenza fisica, mentre l'orizzonte a $0.24\text{s}$ rappresenta il perfetto punto di equilibrio cinematico (un orizzonte superiore come $0.40\text{s}$ introduce latenza di controllo, mentre uno consecutivo fallisce a percepire le variazioni dei sensori).

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
├── README.md
├── gym_torcs/                 # Wrapper Python per TORCS
│   ├── gym_torcs.py           #   Ambiente OpenAI Gym per TORCS
│   ├── snakeoil3_gym.py       #   Client UDP per comunicazione con TORCS
│   └── autostart.sh           #   Automazione menu TORCS (via xte/xautomation)
├── telemetry/                 # Telemetria CSV dei test agent (auto-generata)
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
- `states`: matrice `(N_steps, 29)` — vettore di osservazione 29D normalizzato (dopo il drop v4 di `distFromStart`)
- `actions`: matrice `(N_steps, 4)` — `[steering, accel, brake, gear]`
- Attributi: `lap_time`, `num_steps`, `timestamp`, `preprocessing_version`

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
- **Validation Split 80/20** con **Early Stopping** (patience 100 epoche) per evitare overfitting.
- **Cosine Annealing LR** da `3e-4` fino a `1e-6` per una convergenza stabile.
- **Loss Combinata Multi-Head**: $\mathcal{L} = \mathcal{L}_{\text{MSE continua}} + 2 \times \mathcal{L}_{\text{CrossEntropy gear}}$.
- **Dynamic Brake Boost (25x)**: l'errore sul canale del freno è pesato 25× nei campioni con frenata attiva dell'umano (`brake_target > 0.05`) per forzare staccate vigorose a runtime ed eliminare lo sbilanciamento del dataset (dove il freno è spento per il 95% del tempo).
- **Steer Boost (3x)**: errore del canale di sterzata pesato 3× nelle curve strette.
- **Bojarski-Style Synthetic Recovery Augmentation (NVIDIA Autopilot)**: Insegna all'agente come correggere in modo proattivo gli scostamenti di traiettoria dovuti al *covariate shift*. Durante il training, applichiamo:
  1. Una perturbazione laterale dello spazio di stato (`trackPos`, indice 20) via `delta_pos` in `[-0.15, 0.15]`.
  2. Una correzione proporzionale sullo sterzo target: `new_steer = target_steer - 0.12 * delta_pos`. Questo realizza una legge di controllo autocentrante neurale estremamente stabile sia in rettilineo che in curva.
- **Gradient Clipping** (max_norm=1.0) per la stabilità dei gradienti.

**Output:** `train_set/checkpoints/bc_policy.pth`

### 3. Test Deterministico (Inference)

Avvia TORCS e lancia l'agente autonomo. Il modello guida in modalità interamente deterministica ed end-to-end, gestendo lo sterzo, l'acceleratore, il freno e il cambio discreto interamente con la rete neurale. Tre moduli di post-processing fisico stabilizzano l'output della rete senza mai sovrascriverne le decisioni.

```bash
python test_agent.py --weights train_set/checkpoints/bc_policy.pth --laps 1
```

**Sistemi di Controllo Attivi a Runtime:**

| Sistema | Descrizione | Parametri Chiave |
|---------|-------------|------------------|
| **Gear Hysteresis Filter** | La predizione neurale della marcia passa per un filtro di isteresi che impone due vincoli fisici: (1) vincolo sequenziale ±1 (impedisce salti come G1→G4), (2) conferma temporale di 3 step consecutivi prima di adottare un cambio. Questo elimina le oscillazioni ad alta frequenza mantenendo la marcia 100% neurale. | `confirm_steps=3`, `±1 sequential` |

### Script di Supporto

```bash
./stop_training.sh      # Ferma training e TORCS (SIGTERM)
./stop_training.sh --force  # Kill forzato (SIGKILL)
```

---

## 🔧 Dettagli Tecnici

### PolicyNetwork Multi-Head

La rete Actor adotta un'architettura **Multi-Head** progettata per elaborare lo storico temporale:
- **Backbone Comune**: 4 layer densi (512 unità ciascuno) con `LayerNorm` e attivazione `ReLU`. Estrae feature condivise dallo stack temporale 87D ($29 \times 3$ frame).
- **Testa Continua**: output a 3 dimensioni per il controllo dello sterzo e dei pedali:
  - `steer`: attivato via **Tanh** in $[-1, 1]$ per sterzate simmetriche.
  - `accel` e `brake`: attivati via **Sigmoid** in $[0, 1]$ per mappare naturalmente i pedali.
- **Testa Discreta**: output a 7 logit discreti per il cambio marcia (`gear` in $\{0, 1, 2, 3, 4, 5, 6\}$), addestrata tramite `CrossEntropyLoss`.

| Parametro | Valore |
|-----------|--------|
| Input | 87 neuroni (vettore di osservazione 29D × 3 frame stacked) |
| Hidden Layers (Backbone) | 4 × 512 neuroni con LayerNorm + ReLU |
| Continuous Head | 3 neuroni (steer [Tanh], accel [Sigmoid], brake [Sigmoid]) |
| Gear Head (Discreta) | 7 neuroni (logits marcia per CrossEntropy) |
| Parametri totali | 842,250 |

### Vettore di Osservazione (29D)

Lo stato è un vettore 1D di 29 valori, costruito da `flatten_state()`:

| Indice | Feature | Normalizzazione | Range tipico | Stato |
|--------|---------|-----------------|--------------|-------|
| 0 | `angle` | nessuna (radianti) | [-0.6, 0.4] | Attivo |
| 1–19 | `track[19]` (sensori LIDAR) | /200 (via `gym_torcs`) | [0, 1] | Attivo |
| 20 | `trackPos` | nessuna | [-1, 1] | Attivo |
| 21 | `speedX` | /50 (via `gym_torcs`) | [0, ~5.7] | Attivo |
| 22 | `speedY` | /50 (via `gym_torcs`) | [-0.6, 0.8] | Attivo |
| 23 | `speedZ` | /50 (via `gym_torcs`) | [-0.4, 0.7] | Attivo |
| 24–27 | `wheelSpinVel[4]` | /100 | [0, ~2.6] | Attivo |
| 28 | `rpm` | /10000 | [0.5, 2.0] | Attivo |
| - | *distFromStart* | - | - | **Rimosso (v4)** |

> ⚠️ **IMPORTANTE:** La normalizzazione dei sensori avviene in due punti della pipeline e NON deve essere duplicata:
> - `gym_torcs.make_observaton()` normalizza `track/200` e `speed/default_speed(50)`
> - `data_collection.flatten_state()` normalizza `wheelSpinVel/100` e `rpm/10000`
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

## 📈 Strategia di Ottimizzazione Dataset: Come superare il Covariate Shift con più Dati

Nello sviluppo di un modello di **Pure Behavioral Cloning**, la qualità e la diversità del dataset sono infinitamente più importanti della complessità dell'architettura di rete. Se l'agente mostra comportamenti instabili (come finire dritto nelle vie di fuga o innescare testacoda improvvisi), **la soluzione definitiva è arricchire il dataset con dati non-omogenei e manovre di recupero.**

### 1. Il Limite dei Dati Troppo Omogenei
Se il dataset contiene unicamente giri perfetti lungo l'identica traiettoria ideale (dati omogenei):
- L'agente non apprenderà mai cosa fare al di fuori di quella linea.
- A causa di piccoli disturbi fisici accumulati (ritardi di rete, sfrizionamenti), l'auto devierà inevitabilmente di pochi centimetri dalla traiettoria ideale.
- Trovandosi in uno stato mai visto prima (**Covariate Shift**), la rete predirrà azioni errate (come sterzare a sinistra in una curva a destra per "raddrizzarsi", provocando scivolamenti o uscite).

### 2. Come Raccogliere Dati di Recupero Efficaci (Recovery Dataset Protocol)
Per rendere l'agente solido e capace di auto-correggersi, ti consigliamo di registrare **5-10 giri aggiuntivi dedicati esclusivamente alle correzioni di traiettoria**:
1. **Partenze Fuori Asse**: Avvia il giro posizionandoti volutamente tutto a destra (`trackPos ≈ -0.8`) o tutto a sinistra (`trackPos ≈ 0.8`) e guida puntando attivamente al rientro verso il centro della pista.
2. **Correzioni in Rettilineo**: Durante i rettilinei, oscilla dolcemente a destra e sinistra rispetto alla mezzeria, registrando le azioni correttive per tornare al centro.
3. **Ingressi Curva Alternativi**: Esegui degli inserimenti in curva volutamente larghi o a velocità sub-ottimali, mostrando all'agente come decelerare e stringere lo sterzo in sicurezza per ritrovare il punto di corda ottimale.

### 3. Evitare il Bloccaggio Ruote (Braking and Physics Guide)
TORCS non possiede un sistema ABS attivo per impostazione predefinita sulla vettura F1. Di conseguenza:
- Frenate repentine al $90\%+$ mentre si accenna a sterzare bloccano all'istante le ruote anteriori, provocando un **sottosterzo terminale** che spinge l'auto dritta fuori pista.
- Frenate brusche in curva alleggeriscono il retrotreno causando **sovrasterzi repentini** e testacoda (spin).
- **Consiglio per il Pilota Esperto**: Durante la raccolta dati, esercita una frenata fluida e progressiva (**Threshold Braking**), evitando di schiacciare il pedale oltre il $50-60\%$ se non sei perfettamente dritto. La rete clonerà questa fluidità, mantenendo l'agente sempre all'interno del limite di aderenza fisica delle gomme!

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

### [2026-05-24] Covariate Shift, Allineamento Curve e Traiettorie Limite

**Problema:** L'agente soffriva di covariate shift nelle curve veloci e sul rettilineo iniziale, allontanandosi millimetricamente dalla linea ideale e superando la soglia di fuoripista (1.25). L'utilizzo di launch helper euristiche rompeva la purezza della guida autonoma neurale. Inoltre, la stabilità del cambio automatico precedente interferiva negativamente con la dinamica di frenata.

**Soluzioni Applicate:**
1. **Bojarski-Style Unified Recovery Augmentation**: Introdotta perturbazione laterale dello stato (`trackPos` ±0.15) durante l'addestramento, con correzione proporzionale sullo sterzo target (`new_steer = target_steer - 0.12 * delta_pos`) e parzializzazione dell'acceleratore. L'agente ha così appreso una forza autocentrante e stabilizzante nativa.
2. **Active Safety Envelope (ESP / Lane Keep Assist)**: Aggiunta una rete di protezione a runtime per `|trackPos| > 1.15` che corregge fluidamente la sterzata per evitare infrazioni millimetriche, simulando i controlli di stabilità ESC delle vetture reali.
3. **Deprecazione Heuristics**: Rimosso completamente il Launch Helper iniziale. L'agente ora gestisce la partenza da fermo e tutte le curve del circuito al 100% tramite la rete neurale.
4. **Cambio Manuale Ad Alti Giri (18,000 RPM)**: Il cambio discreto predittivo (testa discrete gear della rete) lavora coordinato sulla soglia di potenza dell'esperto (18,000 RPM).

### [2026-05-25] ~~Inquinamento Dataset — Anomalie Sterzata in Curva 10~~ → Invalidato

**Nota:** Una precedente analisi aveva identificato 19 giri su 66 come anomali per sottosterzo in Curva 10 (3175m-3255m). Un'analisi statistica successiva più approfondita (confronto profilo sterzo vs media con MSE + metriche globali su tutti i 66 giri) ha dimostrato che:
- **Nessun giro supera `|trackPos| > 1.25`** (il limite di pista TORCS)
- I 5 outlier statistici rilevati (lap_017, 056, 048, 060, 013) rappresentano semplicemente traiettorie più variate (linee larghe, velocità diverse)
- Anche i giri "normali" raggiungono `max|tp| > 1.20` (es. lap_008)
- **Tutti i 66 giri sono validi e utilizzati per il training** — la variazione è intenzionale e benefica per la robustezza del modello

### [2026-05-28] Ottimizzazione Stride Temporale e Definizione dell'Architettura 29D (v4)

**Problema:** L'aumento temporaneo a $k=10$ ($0.40\text{s}$) dell'orizzonte di State Stacking ha introdotto una latenza eccessiva di controllo (delay), provocando reazioni ritardate e conseguenti fuori pista. Al contempo, il passaggio ad uno stato puramente geometrico (24D) ha ridotto la sensibilità dell'agente sulle derapate in curva.

**Soluzione Definitiva Applicata:**
1. **Stabilizzazione su 29D**: Rimozione della sola feature discontinua `distFromStart` (causa di covariate shift al traguardo) per via del preprocessing v4, conservando le feature dinamiche di trazione (`wheelSpinVel` e `rpm`).
2. **Consolidamento a $k=6$ ($0.24\text{s}$)**: Ripristinato lo stride temporale a $k=6$ (orizzonte temporale totale di 0.24 secondi) tramite stacking 87D degli stati $t-12$, $t-6$, $t$. Questo rappresenta il perfetto punto di equilibrio dinamico nel controllo deterministico a 50Hz.
3. **Risultato**: Compilazione e integrità di tutta la codebase verificate con successo. Raggiunto minimo storico di validation loss pari a `0.1305`.

### [2026-05-29] Stabilizzazione Runtime — Gear Hysteresis, EMA Steering, ESP Progressivo

**Problema:** L'agente usciva sistematicamente di pista nei primi ~200m dopo la partenza. L'analisi della telemetria ha rivelato tre cause concatenate:
1. **Gear jitter**: La testa discreta oscillava tra marce non sequenziali (es. G1→G4→G3→G1), causando shock di coppia che destabilizzavano l'asse posteriore.
2. **Oscillazione sterzo**: Lo sterzo oscillava ±0.05 ad alta frequenza anche nei rettilinei, accumulando errore laterale progressivo (covariate shift residuo).
3. **ESP troppo debole**: L'intervento (soglia 1.15, gain 0.15) era insufficiente — il car passava da `tp=1.15` a `tp=1.58` in soli 5 step a 170+ km/h.

**Soluzioni Applicate (Runtime-Only — nessuna modifica al training):**
1. **Gear Hysteresis Filter**: Aggiunto filtro di isteresi sulla predizione neurale del gear con vincolo sequenziale ±1 (impedisce salti G1→G4) e conferma temporale di 3 step consecutivi. La marcia resta 100% neurale, ma fisicamente plausibile.
2. **Pure Neural Control**: Rimosso qualsiasi filtro artificiale sullo sterzo, ESP e TCS. Il modello guida in purezza (Direct Control) per massimizzare la fedeltà (Behavioral Cloning) ai dati originali. Rimane attivo solo il filtro antidisturbo neurale sulle marce.

---

**IBM AI Racing League 2026** — *Precision Driving through Behavioral Cloning.*
