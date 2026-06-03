# 🏎️ AIcar — Hybrid BC-RL Architecture (IBM AI Racing League 2026)

**Agente autonomo che impara a guidare tramite Behavioral Cloning (BC) e Twin Delayed DDPG (TD3+BC).**

Questo repository implementa una pipeline end-to-end per addestrare un agente di guida autonoma nell'ambiente di simulazione **TORCS** (The Open Racing Car Simulator). L'obiettivo: **fittare perfettamente i dati esperti** e poi **affinare la policy tramite RL** per sconfiggere il Covariate Shift e ottimizzare il tempo sul giro sul circuito **Corkscrew** con una vettura **F1**.

---

## 🧠 Filosofia del Progetto: Architettura Ibrida BC-RL

Questo progetto supera il classico Behavioral Cloning tramite un'architettura **Ibrida BC-RL**. L'agente parte con una **Deep Policy Network Multi-Head** addestrata offline per imitare l'esperto umano. Per sconfiggere il temuto *Covariate Shift* (che fa deragliare l'agente non appena si discosta millimetricamente dalla traiettoria ottimale), la pipeline prosegue con un **TD3+BC Fine-Tuning**.

Questa fase RL sfrutta la tecnica del **Warm-Start** e il **Gradient Freezing**: il backbone estratto dal BC viene congelato (per prevenire il *Latent Shift*), mentre il TD3 esplora l'ambiente penalizzando duramente gli errori di traiettoria e massimizzando la velocità longitudinale.

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
│  DATA COLLECTION     │────▶│  BC TRAINING         │────▶│  TD3 RL FINE-TUNING  │────▶│  TEST / INFERENCE    │
│                      │     │                      │     │                      │     │                      │
│  🎮 PS5 / Tastiera   │     │  behavioral_cloning  │     │  td3_bc.py           │     │  test_agent.py       │
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
| 3. TD3 RL | `td3_bc.py` | Fine-tuning del modello tramite TD3+BC e Residual RL. Massimizza la velocità, salva il *Best Lap* (`td3_best_policy.pth`) e il *Best Eval* deterministico (`td3_best_eval.pth`). |
| 4. Test & Eval | `test_agent.py` | Esecuzione deterministica del modello finale su TORCS per valutare la capacità di completare giri autonomi. |

---

## 📁 Struttura del Repository

```
AIcar/
├── data_collection.py         # Fase 1: Raccolta dati umani (PS5 / Tastiera)
├── behavioral_cloning.py      # Fase 2: Training della Deep Policy Network
├── td3_bc.py                  # Fase 3: TD3 Fine-Tuning (Warm-Start da BC)
├── test_agent.py              # Fase 4: Inferenza deterministica (BC o RL)
├── train_all.sh               # 🚀 Script per lanciare il training BC
├── train_rl.sh                # 🚀 Script per lanciare il training TD3
├── stop_training.sh           # 🛑 Ferma i processi di training/TORCS
├── README.md
├── gym_torcs/                 # Wrapper Python per TORCS
│   ├── gym_torcs.py           #   Ambiente OpenAI Gym con Reward Reshaping SAC
│   ├── snakeoil3_gym.py       #   Client UDP per comunicazione con TORCS
│   └── autostart.sh           #   Automazione menu TORCS (via xte/xautomation)
├── telemetry/                 # Telemetria CSV dei test agent (auto-generata)
└── train_set/                 # Dati e Checkpoint
    ├── laps/                  #   File HDF5 dei giri registrati (lap_001.h5 ...)
    ├── checkpoints/           #   Pesi: bc_policy.pth, td3_policy.pth, td3_best_policy.pth, td3_best_dist.pth, td3_best_eval.pth
    │   └── buffers/
    │       ├── td3_checkpoint_buffer.npz        # Replay Buffer standard compresso (numpy)
    │       └── td3_checkpoint_elite_buffer.npz  # Elite Buffer compresso (numpy)
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

### 3. TD3+BC Fine-Tuning (Reinforcement Learning)

```bash
# Metodo rapido (1000 episodi, Warm-Start automatico dal BC)
./train_rl.sh

# Override del numero di episodi
TD3_EPISODES=500 ./train_rl.sh

# Ripartenza pulita (cancella checkpoint TD3 precedenti)
./train_rl.sh --clean

# Lancio diretto
python td3_bc.py \
    --bc_weights train_set/checkpoints/bc_policy.pth \
    --episodes 1000 \
    --max_steps 5000 \
    --seed 42
```

Il training è **resume-safe**: il checkpoint viene salvato ad ogni episodio. Puoi interromperlo con `Ctrl+C` e riprenderlo in qualsiasi momento.

Il training avviene in modo isolato in un Virtual Framebuffer (`Xvfb`) per prevenire problemi di focus con il desktop dell'host. 

**Output:** `train_set/checkpoints/td3_policy.pth` + `td3_best_policy.pth` + `td3_best_dist.pth` + `td3_best_eval.pth` + `td3_checkpoint.pth` + `buffers/td3_checkpoint_buffer.npz` + `buffers/td3_checkpoint_elite_buffer.npz`

### 4. Test Deterministico (Inference)

Il test agent auto-rileva i migliori pesi disponibili: `td3_best_eval.pth` → `td3_best_policy.pth` → `td3_best_dist.pth` → `td3_policy.pth` → `sac_best_eval.pth` → `bc_policy.pth`.

```bash
# Esecuzione standard con bypass Xvfb (visibile a schermo)
SHOW_GUI=1 python test_agent.py

# Specificare esplicitamente i pesi
SHOW_GUI=1 python test_agent.py --weights train_set/checkpoints/td3_best_eval.pth --laps 3
SHOW_GUI=1 python test_agent.py --weights train_set/checkpoints/bc_policy.pth --laps 1
```

### Script di Supporto

```bash
./stop_training.sh      # Ferma training e TORCS
```

---

## 📊 Interpretazione dei Log di Addestramento TD3+BC

Durante il training RL, il log stampa metriche fondamentali per diagnosticare la salute dell'addestramento. Ecco i valori corretti da aspettarsi:

### 1. Critic Loss (`CriticL`)
* **Cos'è:** Misura l'errore (MSE) del Critic nel prevedere le reward future.
* **Valori Sani:** Grazie al *Reward Scaling* implementato, i valori ottimali oscillano **tra `0.01` e `5.0`** (con occasionali picchi isolati a `10-20` quando la macchina scopre porzioni di pista inedite). 
* **Diagnosi:** Un valore stabilmente basso significa che il Critic comprende perfettamente la fisica del gioco e sta fornendo valutazioni accurate. Se la `CriticL` schizza permanentemente a centinaia, c'è un'esplosione dei gradienti (o mancano i dati BC nel replay buffer).

### 2. Actor Loss (`ActorL`)
* **Cos'è:** Misura quanto l'Actor sta massimizzando le reward stimate dal Critic. Essendo calcolata in PyTorch (che sa solo minimizzare) come `ActorL = BC_Penalty - Q_Value`, invertire il segno è necessario.
* **Valori Sani:** L'Actor Loss **deve diventare negativa**. Non esiste un limite inferiore, più scende sotto lo zero, più punti l'Actor si aspetta di guadagnare.
* **Diagnosi:** Una discesa dolce e lineare (es. da `0.0` a `-0.8` e oltre) è segno di un apprendimento sanissimo, in cui l'Actor sta capitalizzando sul Q-Value. Salti "positivi" giganteschi in un singolo step denotano un gradiente "sledgehammer" (solitamente causato dall'entropia o dalla BC Penalty) che punisce l'Actor.

### 3. Nessuna Entropia (Differenza con SAC)
* In TD3+BC non c'è più il parametro `Alpha` (entropia) nei log. 
* L'Actor usa azioni completamente deterministiche per la backpropagation, riducendo drasticamente il Catastrophic Forgetting.
* La componente BC è gestita strutturalmente e regolata dal moltiplicatore del reward.

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
| Log Std Head (Legacy) | 3 neuroni (mantenuta per retrocompatibilità coi vecchi test_agent, ma isolata in TD3) |
| Parametri totali | ~843,000 |

### TD3+BC Architecture

```
Actor (Warm-Start da BC)                    Critic (Twin Q-Network, da zero)
┌─────────────────────────┐                 ┌──────────────────────────┐
│  backbone [FROZEN]      │                 │  Q1: (state+action) → 1 │
│  4×512 LayerNorm+ReLU   │                 │  512 → 512 → 1          │
│                         │                 ├──────────────────────────┤
│  continuous_head [TRAIN] │ ←── TD3 ───→  │  Q2: (state+action) → 1 │
│  log_std_head   [TRAIN] │    updates     │  512 → 512 → 1          │
│  gear_head      [FROZEN]│                 └──────────────────────────┘
└─────────────────────────┘                 + Target Q (Polyak τ=0.005)
```

**Gradient Freezing:** Il backbone e la gear_head hanno `requires_grad=False`. L'ottimizzatore aggiorna SOLO `continuous_head` (LR=1e-5) e `log_std_head` (LR=1e-4). LR differenziati per proteggere i pesi BC calibrati.

**Critic Warm-Up:** I primi 5000 step aggiornano solo il Critic. Questo protegge i pesi BC dai gradienti randomici di un Critic non ancora calibrato.

**Update Frequency 1:4:** L'aggiornamento avviene ogni 4 step, non ad ogni step. Riduce l'overfitting su transizioni correlate.

### Reward Reshaping Unificato (SAC-Compatible)

La formula del calcolo della ricompensa per timestep in `gym_torcs.py`:

$$r_t = \underbrace{\frac{v_x}{50} \cos(\theta)}_{\text{progress}} \underbrace{- 0.1}_{\text{time penalty}} \underbrace{- 0.1|\delta_t - \delta_{t-1}|}_{\text{steer smooth}}$$

- **Progress**: Basato sulla velocità in avanti normalizzata diviso 50.
- **Terminali cappati a -10.0**: Danno al veicolo, fuoripista, spin e stallo.
- **Bonus completamento giro: +50.0**: Segnale esplicito per il Critic.
- **Nessuna sparse reward**: Reward densa per evitare distorsioni del gradiente del Critic.

### Replay Buffer Checkpointing

Il Replay Buffer viene salvato separatamente in formato `np.savez_compressed`:
- **File**: `buffers/sac_checkpoint_buffer.npz` e `buffers/sac_checkpoint_elite_buffer.npz` (~50-100MB compressi vs >1GB con pickle)
- **Previene il Catastrophic Forgetting** quando il training viene interrotto e ripreso
- **Resume-safe**: Ad ogni episodio vengono salvati sia il checkpoint PyTorch che il buffer numpy

### Done Masking

Nel SAC, il flag `done` nel Replay Buffer è cruciale per la Bellman equation:
- **done=True** → Solo per terminazioni reali (fuoripista, spin, stallo, collisione)
- **done=False** → Per il time-limit (`max_steps`) e il completamento giro, perché il vero state-value non è zero
- **Dati expert** → `mask=1.0` per tutti i campioni (giri completati, non crash)

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
Il training BC include perturbazione laterale dello stato (`trackPos ±0.4`) e angolare (`angle ±0.08 rad`) con correzione proporzionale dello sterzo e del freno target, implementando una legge di controllo autocentrante neurale avanzata.

---

## 🐛 Bug Risolti (Workflow Tracking)

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
2. **BC Weight Fisso = 5.0**: Nessun decay. L'Actor resta permanentemente ancorato al BC (Residual RL).
3. **Done Masking Corretto**: I dati expert usano `mask=1.0` per tutti i campioni (giri completati ≠ crash).
4. **Alpha Fisso = 0.01**: Disattivato l'auto-tuning dell'entropia.
5. **Update Ratio 1:4**: Aggiornamento ogni 4 step per ridurre l'overfitting.
6. **Bonus Completamento Giro = +50.0**: Segnale esplicito per il Critic.
7. **Evaluation Periodica Deterministica**: Ogni 25 episodi, checkpoint riproducibile `sac_best_eval.pth`.

### [2026-05-31] Risoluzione del Collasso della Policy (Stall Trap & Q-Value Explosion)

**Problema:** L'agente soffriva di uno "Stall Trap" alla partenza a causa di un *overestimation bias* critico (Critic Loss > 2000), seguito dal collasso della rete. Questo era indotto da Alpha azzerato forzatamente, iniezione di rumore scorretta e una soglia dell'elite buffer degenerata.

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
3. **Pure SAC Actor Loss**: Rimossa la logica fallata TD3+BC dalla fase Online. L'Actor massimizza unicamente entropia e Q-Value target senza auto-imitare il proprio rumore di addestramento.
4. **Dual Checkpointing**: L'agente salva `sac_best_policy.pth` ad ogni giro da record. Durante l'esplorazione, salva anche `sac_best_dist.pth` ad ogni nuovo record di distanza percorsa prima dello schianto (se > 500m). Logging semantico per gli episodi `[SUCCESS]`, `[CRASH]` o `[TIMEOUT]`.
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
