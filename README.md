# Iterative Motors

Iterative Motors è un progetto di guida autonoma per TORCS, nato per la **IBM AI Racing League**.
L'obiettivo: costruire un agente che superi le prestazioni umane sul giro. Il modello non guida
"da zero": prima **imita** un pilota reale tramite Behavioral Cloning (BC), poi spinge oltre quella
base con Reinforcement Learning **TD3+BC**, generando giri sempre più veloci e ripetibili.

Il miglior tempo umano nel dataset è **69.54s**; l'obiettivo è batterlo e avvicinare il record della
pista (**~65s**).

Per i dettagli tecnici completi vedi **[ARCHITECTURE.md](ARCHITECTURE.md)**.

## Caratteristiche principali

- Simulatore TORCS controllato via protocollo SCR/UDP (senza modificarne la fisica — regola IBM League).
- Raccolta dati umana con controller PS5 DualSense o tastiera.
- Stato sensoriale 29D con frame stacking temporale (t-12, t-6, t → 87D); marcia gestita a parte.
- Behavioral Cloning con loss pesata per sterzo/freno/acceleratore.
- Data augmentation Bojarski-style con **clamp on-track** (non insegna a guidare fuori pista) e
  perturbazione angolare ampliata (~10°).
- Fine-tuning TD3+BC: Actor con warm-start da BC, Twin Critic, campionamento ibrido expert/online/elite.
- **Flywheel dati**: la TD3 registra i propri giri puliti (`laps_auto/`) per riaddestrare una BC più forte.
- **Time-attack** con bonus di record personale: l'agente cerca di battere i propri tempi.
- Cambio marcia deterministico separato dalla rete; checkpoint atomici con backup e resume robusto.
- Valutazione deterministica con auto-detect del miglior checkpoint.
- **Orchestratore unico `run.sh`** per l'intera pipeline, con cruscotto di stato.

## Struttura del progetto

```text
.
|-- run.sh                          # orchestratore unico della pipeline (CLI controller)
|-- src/iterative_motors/           # PACKAGE: logica del progetto
|   |-- common/                    # costanti, stato/normalizzazione, checkpoint
|   |-- env/                       # wrapper TORCS, client SCR, gearing, autostart
|   |-- models/                    # reti (Actor/PolicyNetwork/Critic) e mapping azioni
|   |-- data/                      # replay buffer, dataset HDF5, lap recorder, raccolta dati
|   |-- bc/                        # data augmentation + training Behavioral Cloning
|   |-- rl/                        # agente TD3+BC, reward/time-attack, training loop
|   `-- eval/                      # test deterministico dell'agente
|-- train_set/                      # dataset (laps/, laps_auto/), checkpoint, log (NON in git)
`-- telemetry/                      # CSV generati dai test
```

Gli entrypoint vivono nel package e si lanciano via `run.sh` o come moduli
(`PYTHONPATH=src python -m iterative_motors.<sottopacchetto>.<modulo>`).

## Prerequisiti

Linux con:
- Python 3; TORCS con server SCR; `xvfb-run` (headless); `xte` (autostart menu TORCS, pacchetto `xautomation`);
- librerie Python: `torch`, `numpy`, `h5py`, `pygame`, `gym`.

```bash
sudo apt install torcs xvfb xautomation
python -m venv .venv && source .venv/bin/activate
pip install numpy torch h5py pygame gym   # per CUDA usare il comando ufficiale PyTorch
```

## Uso rapido (via orchestratore)

```bash
./run.sh             # menu interattivo con frecce + Enter
./run.sh help        # elenco completo dei comandi diretti
./run.sh status      # cruscotto: processi attivi, dataset, record, ultimi log
```

Il menu di `run.sh` funziona come un piccolo pit wall: mostra una Formula 1 in ASCII, riepilogo
di processi/dataset/record e opzioni selezionabili con le frecce. I comandi diretti restano
disponibili per automazione e script.

### 1. Raccogliere dimostrazioni umane

```bash
./run.sh collect --device controller     # oppure --device keyboard
```

I giri validi finiscono in `train_set/laps/lap_NNN.h5`. Opzioni utili: `--segment_only` (solo
segmenti di curve), `--zones "670:900,2380:2530"` (zone specifiche).

### 2. Addestrare la Behavioral Cloning

```bash
./run.sh bc
```

Produce `train_set/checkpoints/bc_policy.pth` e `state_norm.npz` (background; segui con
`./run.sh logs bc` e `./run.sh status`).

### 3. Fine-tuning TD3+BC + raccolta giri (harvest)

```bash
./run.sh rl --episodes 2500
```

Durante il training l'agente registra automaticamente i propri giri completi e puliti in
`train_set/laps_auto/` (disattivabile con `IM_RECORD_LAPS=0`). Stop pulito: `./run.sh stop rl`
(salva il checkpoint prima di uscire).

### 4. Arricchire la BC (flywheel) e time-attack

```bash
# Ri-addestra la BC su giri umani + auto-raccolti
./run.sh bc-enriched --output train_set/checkpoints/enriched/bc_policy.pth

# Fase time-attack: l'agente ottimizza il tempo battendo il proprio record
./run.sh time-attack --episodes 4000
```

Per adottare la BC arricchita in una nuova lineage RL: copia i nuovi `bc_policy.pth` +
`state_norm.npz` da `enriched/` in `train_set/checkpoints/`, azzera i checkpoint TD3 (i record
deterministici restano protetti) e rilancia `./run.sh rl`.

### 5. Testare l'agente

```bash
./run.sh test --laps 3                # auto-detect del miglior checkpoint
SHOW_GUI=1 ./run.sh test --laps 1     # con finestra TORCS visibile
./run.sh test --weights train_set/checkpoints/td3_det_best_lap.pth --laps 5
```

I CSV di telemetria vengono scritti in `telemetry/`.

## Idea del modello

La policy vede 3 istanti temporali dello stato sensoriale (input 87D) e produce sterzo, acceleratore
e freno; la marcia è calcolata da `gearing.py`. La BC dà la competenza iniziale; il TD3+BC conserva
quell'ancora esperta ma ottimizza la reward racing (avanzare, restare in pista, fluidità, completare
il giro, abbassare il tempo). Il flywheel dei dati reimmette i giri migliori dell'agente nel dataset
BC, alzando progressivamente il punto di partenza.

## Riferimenti

- Lillicrap et al., *Continuous Control with Deep RL*, 2015 — https://arxiv.org/abs/1509.02971
- Fujimoto, van Hoof, Meger, *Addressing Function Approximation Error in Actor-Critic Methods*, 2018 — https://arxiv.org/abs/1802.09477
- Fujimoto, Gu, *A Minimalist Approach to Offline RL*, 2021 — https://arxiv.org/abs/2106.06860
- Beeson, Montana, *Improving TD3-BC*, 2022 — https://arxiv.org/abs/2211.11802
- Bojarski et al., *End to End Learning for Self-Driving Cars*, 2016 — https://arxiv.org/abs/1604.07316
- Loiacono, Cardamone, Lanzi, *SCR Championship: Competition Software Manual*, 2013 — https://arxiv.org/abs/1304.1672
