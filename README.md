# Iterative Motors

Iterative Motors è un progetto di guida autonoma per TORCS nato con un obiettivo preciso: partecipare alla IBM AI Racing League 2026 e costruire un agente capace di superare i limiti e le prestazioni umane sul giro. L'idea è partire dalla competenza di un pilota reale, trasferirla in una rete neurale tramite Behavioral Cloning e poi spingere oltre quella base con Reinforcement Learning TD3+BC.

In pratica, il modello non prova a guidare "da zero". Prima impara a imitare traiettorie, accelerazioni e staccate umane; poi usa il simulatore per esplorare varianti, correggere errori e cercare una guida più veloce e robusta.

Per i dettagli tecnici completi vedi [ARCHITECTURE.md](ARCHITECTURE.md).

## Caratteristiche Principali

- Simulatore TORCS controllato tramite protocollo SCR/UDP.
- Raccolta dati umana con controller PS5 DualSense o tastiera.
- Dataset HDF5 con giri completi e segmenti mirati sulle curve difficili.
- Policy neurale MLP 87D -> 3D con frame stacking temporale.
- Behavioral Cloning con loss pesata per sterzo, freno e acceleratore.
- Data augmentation per recupero laterale, errore angolare e overspeed in curva.
- Fine-tuning TD3+BC con Actor warm-start da BC e Twin Critic.
- Replay buffer ibrido: expert, online ed elite.
- Cambio marcia deterministico separato dalla rete.
- Checkpoint atomici con backup `.bak`/`.prev` e resume.
- Evaluation deterministica e auto-detect del miglior checkpoint.

## Struttura Del Progetto

```text
.
|-- data_collection.py        # raccolta dati umani in TORCS
|-- behavioral_cloning.py     # training supervisionato BC
|-- td3_bc.py                 # fine-tuning TD3+BC
|-- test_agent.py             # test deterministico dell'agente
|-- gearing.py                # cambio marcia algoritmico
|-- gym_torcs/                # wrapper TORCS e client SCR
|-- train_bc.sh               # avvio training BC
|-- train_rl.sh               # avvio/resume training TD3+BC
|-- stop_training.sh          # stop processi training/TORCS
|-- train_set/                # dataset, checkpoint e log locali
`-- telemetry/                # CSV generati dai test
```

`train_set/` e `telemetry/` sono ignorate da Git tranne i `.gitkeep`: i dati raccolti, i checkpoint e la telemetria sono artefatti locali.

## Prerequisiti

Il progetto è pensato per Linux. Servono:

- Python 3;
- TORCS con server SCR disponibile;
- `xvfb-run` per training/test headless;
- `xte`, fornito di solito da `xautomation`, per l'autostart dei menu TORCS;
- librerie Python: `torch`, `numpy`, `h5py`, `pygame`, `gym`.

Setup indicativo:

```bash
sudo apt install torcs xvfb xautomation
python -m venv .venv
source .venv/bin/activate
pip install numpy torch h5py pygame gym
```

Per PyTorch con CUDA conviene usare il comando ufficiale adatto alla propria GPU.

## Uso Rapido

### 1. Raccogliere Dimostrazioni Umane

Con controller:

```bash
python data_collection.py --device controller
```

Con tastiera:

```bash
python data_collection.py --device keyboard
```

I giri validi vengono salvati in `train_set/laps/lap_XXX.h5`. Per salvare solo segmenti di curve:

```bash
python data_collection.py --device controller --segment_only
```

Per indicare zone specifiche:

```bash
python data_collection.py --device controller --segment_only --zones "670:900,2380:2530"
```

### 2. Addestrare Il Behavioral Cloning

```bash
./train_bc.sh
```

Output principali:

- `train_set/checkpoints/bc_policy.pth`;
- `train_set/checkpoints/state_norm.npz`;
- log in `train_set/session_logs/`.

Comando equivalente manuale:

```bash
python behavioral_cloning.py \
  --dataset train_set/laps \
  --epochs 300 \
  --batch_size 256 \
  --output train_set/checkpoints/bc_policy.pth
```

### 3. Avviare Il Fine-Tuning TD3+BC

```bash
./train_rl.sh
```

Variabili utili:

```bash
TD3_EPISODES=500 TD3_MAX_STEPS=5000 TD3_SEED=42 ./train_rl.sh
```

Flag principali:

```bash
./train_rl.sh --clean
./train_rl.sh --rollback
./train_rl.sh --rollback --actor-freeze-episodes 100
./train_rl.sh --no-auto-refine
./train_rl.sh --refine
```

`--clean` elimina i checkpoint TD3 del run corrente, ma preserva i record deterministici assoluti gestiti dai sidecar dedicati.

Per fermare in modo ordinato:

```bash
./stop_training.sh
```

Oppure usa `Ctrl+C`: `td3_bc.py` intercetta il segnale e prova a salvare un checkpoint completo prima di uscire.

### 4. Testare L'Agente

Auto-detect del miglior checkpoint:

```bash
python test_agent.py
```

Per guardare TORCS durante il test:

```bash
SHOW_GUI=1 python test_agent.py
```

Checkpoint esplicito:

```bash
python test_agent.py --weights train_set/checkpoints/td3_det_best_lap.pth --laps 5
```

Se il nome file non contiene `td3` o `bc`, specifica il tipo:

```bash
python test_agent.py --weights mio_actor.pth --kind rl
python test_agent.py --weights mio_bc.pth --kind bc
```

I CSV di telemetria vengono scritti in `telemetry/`.

## Checkpoint Importanti

| File | Significato |
| --- | --- |
| `bc_policy.pth` | Policy addestrata solo con Behavioral Cloning |
| `state_norm.npz` | Media e deviazione standard degli stati |
| `td3_checkpoint.pth` | Checkpoint completo per resume TD3+BC |
| `td3_policy.pth` | Ultima policy TD3 salvata |
| `td3_det_best_lap.pth` | Miglior giro valido deterministico, candidato submission |
| `td3_det_best_dist.pth` | Miglior distanza deterministica assoluta |
| `td3_expl_best_lap.pth` | Miglior giro trovato durante esplorazione |
| `td3_expl_best_dist.pth` | Miglior distanza trovata durante esplorazione |

Quando `test_agent.py` viene lanciato senza `--weights`, cerca i checkpoint in ordine di priorità e usa il migliore disponibile.

## Idea Del Modello

La policy vede tre istanti temporali dello stato sensoriale TORCS, concatenati in un input 87D. Produce sterzo, acceleratore e freno. La marcia non è appresa: `gearing.py` la calcola con soglie robuste di velocità, RPM e cooldown, riducendo lo spazio d'azione e migliorando la stabilità.

Il Behavioral Cloning fornisce una guida iniziale umana. TD3+BC conserva quell'ancora esperta ma permette alla policy di ottimizzare la reward racing: avanzare lungo la pista, restare entro i limiti, evitare oscillazioni di sterzo, completare il giro e migliorare tempo/distanza in valutazione deterministica.

## Riferimenti

- Lillicrap et al., "Continuous Control with Deep Reinforcement Learning", 2015: https://arxiv.org/abs/1509.02971
- Fujimoto, van Hoof, Meger, "Addressing Function Approximation Error in Actor-Critic Methods", 2018: https://arxiv.org/abs/1802.09477
- Fujimoto, Gu, "A Minimalist Approach to Offline Reinforcement Learning", 2021: https://arxiv.org/abs/2106.06860
- Beeson, Montana, "Improving TD3-BC: Relaxed Policy Constraint for Offline Learning and Stable Online Fine-Tuning", 2022: https://arxiv.org/abs/2211.11802
- Bojarski et al., "End to End Learning for Self-Driving Cars", 2016: https://arxiv.org/abs/1604.07316
- Loiacono, Cardamone, Lanzi, "Simulated Car Racing Championship: Competition Software Manual", 2013: https://arxiv.org/abs/1304.1672
