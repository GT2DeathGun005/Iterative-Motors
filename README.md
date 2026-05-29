# 🏎️ AIcar — Hybrid BC-RL Architecture (IBM AI Racing League 2026)

**Agente autonomo che impara a guidare tramite Behavioral Cloning (BC) e Soft Actor-Critic (SAC).**

Questo repository implementa una pipeline end-to-end per addestrare un agente di guida autonoma nell'ambiente di simulazione **TORCS** (The Open Racing Car Simulator). L'obiettivo: **fittare perfettamente i dati esperti** e poi **affinare la policy tramite RL** per sconfiggere il Covariate Shift e ottimizzare il tempo sul giro sul circuito **Corkscrew** con una vettura **F1**.

---

## 🧠 Filosofia del Progetto: Architettura Ibrida BC-RL

Questo progetto supera il classico Behavioral Cloning tramite un'architettura **Ibrida BC-RL**. L'agente parte con una **Deep Policy Network Multi-Head** addestrata offline per imitare l'esperto umano. Per sconfiggere il temuto *Covariate Shift* (che fa deragliare l'agente non appena si discosta millimetricamente dalla traiettoria ottimale), la pipeline prosegue con un **Soft Actor-Critic (SAC) Fine-Tuning**.

Questa fase RL sfrutta la tecnica del **Warm-Start** e il **Gradient Freezing**: il backbone estratto dal BC viene congelato (per prevenire il *Latent Shift*), mentre il SAC esplora l'ambiente penalizzando duramente gli errori di traiettoria e massimizzando la velocità longitudinale.

### Punti di forza della pipeline Ibrida BC-RL:
1. **Sample Efficiency**: Il BC fornisce un ottimo punto di partenza, abbattendo drasticamente i tempi di esplorazione del RL.
2. **Prevenzione del Latent Shift**: Il backbone e il cambio marce rimangono quelli perfetti del BC. Il RL affina unicamente sterzo, acceleratore e freno.
3. **Determinismo Assoluto**: La policy finale, e l'inferenza, godono di seed statici e reset fisici per una riproducibilità matematica esatta.

---

## 🏛️ Architettura del Sistema

La pipeline si compone di quattro fasi sequenziali:

```
┌──────────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
│  Fase 1              │     │  Fase 2              │     │  Fase 3              │     │  Fase 4              │
│  DATA COLLECTION     │────▶│  BC TRAINING         │────▶│  SAC RL FINE-TUNING  │────▶│  TEST / INFERENCE    │
│                      │     │                      │     │                      │     │                      │
│  🎮 PS5 / Tastiera   │     │  behavioral_cloning  │     │  sac_rl.py           │     │  test_agent.py       │
│  data_collection.py  │     │  .py                 │     │  (Warm-Start)        │     │  Deterministico      │
│                      │     │                      │     │                      │     │                      │
│  Output:             │     │  Output:             │     │  Output:             │     │  Valutazione live    │
│  train_set/laps/     │     │  bc_policy.pth       │     │  sac_policy.pth      │     │  su TORCS            │
└──────────────────────┘     └──────────────────────┘     └──────────────────────┘     └──────────────────────┘
```

| Fase | Script | Descrizione |
|------|--------|-------------|
| 1. Data Collection | `data_collection.py` | Raccolta di giri guidati da umano con controller PS5 o tastiera WASD. Salva solo i giri puliti. |
| 2. BC Training | `behavioral_cloning.py` | Addestramento della PolicyNetwork sui dati esperti. Produce una policy che imita l'esperto (`bc_policy.pth`). |
| 3. SAC RL | `sac_rl.py` | Fine-tuning del modello tramite Soft Actor-Critic puro con Auto-Entropy tuning e Warm-Start. Massimizza la velocità, salva il *Best Lap* (`sac_best_policy.pth`). L'esecuzione avviene in un display virtuale invisibile tramite `Xvfb`. |
| 4. Test & Eval | `test_agent.py` | Esecuzione deterministica del modello finale su TORCS per valutare la capacità di completare giri autonomi. |

---

## 📁 Struttura del Repository

```
AIcar/
├── data_collection.py         # Fase 1: Raccolta dati umani (PS5 / Tastiera)
├── behavioral_cloning.py      # Fase 2: Training della Deep Policy Network
├── sac_rl.py                  # Fase 3: SAC Fine-Tuning (Warm-Start da BC)
├── test_agent.py              # Fase 4: Inferenza deterministica (BC o SAC)
├── train_all.sh               # 🚀 Script per lanciare il training BC
├── train_rl.sh                # 🚀 Script per lanciare il training SAC
├── stop_training.sh           # 🛑 Ferma i processi di training/TORCS
├── README.md
├── gym_torcs/                 # Wrapper Python per TORCS
│   ├── gym_torcs.py           #   Ambiente OpenAI Gym con Reward Reshaping SAC
│   ├── snakeoil3_gym.py       #   Client UDP per comunicazione con TORCS
│   └── autostart.sh           #   Automazione menu TORCS (via xte/xautomation)
├── telemetry/                 # Telemetria CSV dei test agent (auto-generata)
└── train_set/                 # Dati e Checkpoint
    ├── laps/                  #   File HDF5 dei giri registrati (lap_001.h5 ...)
    ├── checkpoints/           #   Pesi: bc_policy.pth, sac_policy.pth, sac_best_policy.pth, sac_checkpoint.pth
    │   └── sac_checkpoint_buffer.npz  # Replay Buffer compresso (numpy)
    └── session_logs/          #   Log delle sessioni di training
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

**Output:** un file `train_set/laps/lap_NNN.h5` per ogni giro valido.

### 2. Addestramento BC (Behavioral Cloning)

```bash
# Metodo rapido
./train_all.sh

# Oppure direttamente con parametri personalizzati
python behavioral_cloning.py \
    --dataset train_set/laps \
    --epochs 300 \
    --batch_size 256 \
    --lr 3e-4 \
    --output train_set/checkpoints/bc_policy.pth
```

**Output:** `train_set/checkpoints/bc_policy.pth`

### 3. SAC Fine-Tuning (Reinforcement Learning)

```bash
# Metodo rapido (1000 episodi, Warm-Start automatico dal BC)
./train_rl.sh

# Override del numero di episodi
SAC_EPISODES=500 ./train_rl.sh

# Ripartenza pulita (cancella checkpoint SAC precedenti)
./train_rl.sh --clean

# Lancio diretto
python sac_rl.py \
    --bc_weights train_set/checkpoints/bc_policy.pth \
    --episodes 1000 \
    --max_steps 5000 \
    --seed 42
```

Il training è **resume-safe**: il checkpoint viene salvato ad ogni episodio. Puoi interromperlo con `Ctrl+C` e riprenderlo in qualsiasi momento.

Il training avviene in modo isolato in un Virtual Framebuffer (`Xvfb`) per prevenire problemi di focus con il desktop dell'host. 

**Output:** `train_set/checkpoints/sac_policy.pth` + `sac_best_policy.pth` + `sac_checkpoint.pth` + `sac_checkpoint_buffer.npz`

### 4. Test Deterministico (Inference)

Il test agent auto-rileva i migliori pesi disponibili: `sac_best_policy.pth` → `sac_policy.pth` → `bc_policy.pth`.

```bash
# Esecuzione standard con bypass Xvfb (visibile a schermo)
SHOW_GUI=1 python test_agent.py

# Specificare esplicitamente i pesi
SHOW_GUI=1 python test_agent.py --weights train_set/checkpoints/sac_best_policy.pth --laps 3
SHOW_GUI=1 python test_agent.py --weights train_set/checkpoints/bc_policy.pth --laps 1
```

### Script di Supporto

```bash
./stop_training.sh      # Ferma training e TORCS
```

---

## 🔧 Dettagli Tecnici

### PolicyNetwork / Actor Multi-Head

La rete adotta un'architettura **Multi-Head** per elaborare lo storico temporale:

| Parametro | Valore |
|-----------|--------|
| Input | 87 neuroni (vettore di osservazione 29D × 3 frame stacked) |
| Hidden Layers (Backbone) | 4 × 512 neuroni con LayerNorm + ReLU |
| Continuous Head | 3 neuroni (steer, accel, brake) |
| Gear Head (Discreta) | 7 neuroni (logits marcia per CrossEntropy) |
| Log Std Head (SAC) | 3 neuroni (deviazione standard per campionamento gaussiano) |
| Parametri totali | ~843,000 |

### SAC Architecture

```
Actor (Warm-Start da BC)                    Critic (Twin Q-Network, da zero)
┌─────────────────────────┐                 ┌──────────────────────────┐
│  backbone [FROZEN]      │                 │  Q1: (state+action) → 1 │
│  4×512 LayerNorm+ReLU   │                 │  512 → 512 → 1          │
│                         │                 ├──────────────────────────┤
│  continuous_head [TRAIN] │ ←── SAC ───→  │  Q2: (state+action) → 1 │
│  log_std_head   [TRAIN] │    updates     │  512 → 512 → 1          │
│  gear_head      [FROZEN]│                 └──────────────────────────┘
└─────────────────────────┘                 + Target Q (Polyak τ=0.005)
```

**Gradient Freezing:** Il backbone e la gear_head hanno `requires_grad=False`. L'ottimizzatore aggiorna SOLO `continuous_head` e `log_std_head`. Questo previene il *Latent Shift* (distruzione delle feature estratte dal BC).

**Critic Warm-Up:** I primi 5000 step aggiornano solo il Critic. Questo protegge i pesi BC dai gradienti randomici di un Critic non ancora calibrato.

### Reward Reshaping Unificato (SAC-Compatible)

La formula del calcolo della ricompensa per timestep in `gym_torcs.py`:

$$r_t = \underbrace{\frac{v_x}{50} \cos(\theta)}_{\text{progress}} \underbrace{- 0.1}_{\text{time penalty}} \underbrace{- 0.1|\delta_t - \delta_{t-1}|}_{\text{steer smooth}}$$

- **Progress**: Basato sulla velocità in avanti normalizzata diviso 50.
- **Time Penalty -0.1**: Costante per incentivare il completamento del tracciato rapido.
- **Terminali cappati a -10.0**: Danno al veicolo, fuoripista, spin (retromarcia) e stallo, configurando il dizionario `info['crash'] = True`.
- **Nessuna sparse reward**: Reward densa per evitare distorsioni del gradiente del Critic.

### Replay Buffer Checkpointing

Il Replay Buffer viene salvato separatamente in formato `np.savez_compressed`:
- **File**: `sac_checkpoint_buffer.npz` (~50-100MB compressi vs >1GB con pickle)
- **Previene il Catastrophic Forgetting** quando il training viene interrotto e ripreso
- **Resume-safe**: Ad ogni episodio vengono salvati sia il checkpoint PyTorch che il buffer numpy

### Done Masking

Nel SAC, il flag `done` nel Replay Buffer è cruciale per la Bellman equation:
- **done=True** → Solo per terminazioni reali (fuoripista, spin, stallo, collisione)
- **done=False** → Per il time-limit (`max_steps`), perché il vero state-value non è zero

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

> ⚠️ **IMPORTANTE:** La normalizzazione avviene in due punti: `gym_torcs.make_observaton()` e `flatten_state()`. NON duplicare!

---

## 📈 Strategia di Ottimizzazione Dataset

### Il Limite dei Dati Troppo Omogenei
Se il dataset contiene unicamente giri perfetti lungo l'identica traiettoria ideale, l'agente non apprenderà mai cosa fare fuori da quella linea → **Covariate Shift**.

### Come Raccogliere Dati di Recupero Efficaci
Registrare **5-10 giri aggiuntivi** con:
1. **Partenze Fuori Asse**: `trackPos ≈ ±0.8`, guidando verso il centro
2. **Correzioni in Rettilineo**: Oscillare dolcemente a destra e sinistra
3. **Ingressi Curva Alternativi**: Inserimenti larghi a velocità sub-ottimali

### Bojarski-Style Recovery Augmentation
Il training BC include perturbazione laterale dello stato (`trackPos ±0.15`) con correzione proporzionale dello sterzo target, implementando una legge di controllo autocentrante neurale.

---

## 🐛 Bug Risolti (Workflow Tracking)

### [2026-05-29] Finalizzazione Architettura (Xvfb, Best Lap, Pure SAC)

**Fix applicati:**
1. **Ambiente Isolato Xvfb:** TORCS e le macro girano confinati in un virtual display senza rubare focus. Usare `SHOW_GUI=1` per lo sblocco in rendering locale.
2. **Auto-Entropy Tuning**: Introdotto tuning automatico del `log_alpha` per un corretto calcolo del SAC.
3. **Pure SAC Actor Loss**: Rimossa la logica fallata TD3+BC dalla fase Online. L'Actor massimizza unicamente entropia e Q-Value target senza auto-imitare il proprio rumore di addestramento.
4. **Early Checkpointing**: L'agente salva il `sac_best_policy.pth` a ogni giro da record. Logging semantico per gli episodi `[SUCCESS]`, `[CRASH]` o `[TIMEOUT]`.

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
