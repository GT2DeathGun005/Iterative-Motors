# Iterative Motors — Architettura del progetto

Iterative Motors è un agente di guida autonoma per **TORCS** (circuito *corkscrew*) sviluppato
per la **IBM AI Racing League**. L'idea portante: non imparare a guidare "da zero" con il solo
Reinforcement Learning (lento e instabile), ma **partire dalla competenza di un pilota umano**
trasferita in una rete neurale tramite **Behavioral Cloning (BC)**, e poi **superarla** con il
fine-tuning **TD3+BC** finché l'agente non batte il miglior tempo umano (**69.54s**) puntando al
record della pista (**target ~65s**).

> **Nome e cartella.** Il progetto si chiama *Iterative Motors*; `AIcar` è solo il nome della
> cartella root del repository. Tutto il codice riusabile vive nel package `src/iterative_motors/`.

> **Vincolo IBM AI Racing League.** La fisica e l'installazione di TORCS NON possono essere
> modificate. TORCS è usato come simulatore esterno via protocollo SCR (UDP); il sottopacchetto
> `env/` è l'unico punto di contatto e non ne altera la configurazione.

Indice:
1. [Struttura del repository](#1-struttura-del-repository)
2. [La pipeline e il flywheel dei dati](#2-la-pipeline-e-il-flywheel-dei-dati)
3. [Rappresentazione dello stato](#3-rappresentazione-dello-stato)
4. [Reti neurali](#4-reti-neurali)
5. [Ambiente TORCS](#5-ambiente-torcs)
6. [Behavioral Cloning](#6-behavioral-cloning)
7. [TD3+BC](#7-td3bc)
8. [Lap recorder ed enrichment](#8-lap-recorder-ed-enrichment)
9. [Time-attack e record personale](#9-time-attack-e-record-personale)
10. [Sistema di checkpoint](#10-sistema-di-checkpoint)
11. [Orchestratore run.sh](#11-orchestratore-runsh)
12. [Configurazione](#12-configurazione)
13. [Speedup RL (differito)](#13-speedup-rl-differito)
14. [Note operative](#14-note-operative)

---

## 1. Struttura del repository

```
AIcar/                                  # cartella root del repo
  run.sh                                # ORCHESTRATORE unico della pipeline (CLI controller)
  src/iterative_motors/                 # PACKAGE: tutta la logica del progetto
    common/                             # utility trasversali (nessuna dipendenza dagli altri sottopacchetti)
      constants.py    # percorsi, dimensioni stato/stack, geometria pista, angoli dei 19 sensori
      state.py        # flatten_state_raw / flatten_state_norm, apply_state_norm, FrameStacker
      checkpoint.py   # safe_save/load atomici, rotazione .bak/.prev, sidecar (archivio intoccabile)
    env/                                # unico punto di contatto con TORCS
      gym_torcs.py    # wrapper Gym: osservazioni, azioni, reward per-step, terminazioni
      snakeoil3_gym.py# client SCR (UDP); angoli dei 19 sensori da SENSOR_ANGLES_DEG
      gearing.py      # cambio marcia algoritmico con isteresi (la rete NON predice la marcia)
      autostart.sh    # macro di avvio TORCS via xte
    models/                             # reti neurali condivise
      networks.py     # Actor (RL), PolicyNetwork (BC/eval), Critic — backbone e teste condivisi
      action_mapping.py# conversione azioni rete <-> pedali TORCS, mutual exclusion gas/freno
    data/                               # dati: buffer RL, dataset BC, registratore giri
      replay_buffer.py# ReplayBuffer + load_expert_data (HDF5 -> buffer con stacking)
      hdf5_dataset.py # TorcsHDF5Dataset + load_dataset (con giri auto per l'arricchimento)
      lap_recorder.py # LapRecorder: salva i giri puliti dell'agente in laps_auto/
      collection.py   # ENTRYPOINT raccolta giri umani (controller PS5 / tastiera)
    bc/                                 # Behavioral Cloning
      augmentation.py # data augmentation Bojarski-style (AugmentConfig + augment_batch)
      train_bc.py     # ENTRYPOINT training BC (BehaviorCloningTrainer + main)
    rl/                                 # Reinforcement Learning TD3+BC
      agent.py        # TD3BCAgent: update RL/BC, save/load_checkpoint robusto
      reward.py       # bonus/penalità fine giro, score di eval, bonus di record personale
      train_rl.py     # ENTRYPOINT fine-tuning TD3+BC (+ harvest dei giri, time-attack)
    eval/
      test_agent.py   # ENTRYPOINT valutazione deterministica con auto-detect del best
  train_set/                            # dati locali (NON tracciati da git)
    laps/             # giri umani (HDF5)
    laps_auto/        # giri auto-raccolti dalla TD3 (flywheel)
    checkpoints/      # pesi, buffer, sidecar dei record; state_norm.npz
    session_logs/  .run/  (pid dei task di run.sh)
  telemetry/                            # CSV prodotti dai test
  ARCHITECTURE.md  README.md
```

**Perché un package + un orchestratore.** Prima il codice era in script monolitici nella root con
forte duplicazione (la funzione di flatten dello stato era ripetuta 4 volte, la rete 3 volte, ecc.).
Ora ogni responsabilità ha un'unica casa nel package e gli entrypoint si lanciano come moduli
(`python -m iterative_motors.<sottopacchetto>.<modulo>`) o, più comodamente, tramite `run.sh`.
Gli entrypoint aggiungono `src/` al path con un piccolo bootstrap, quindi funzionano sia da modulo
sia eseguiti direttamente.

---

## 2. La pipeline e il flywheel dei dati

```
1. RACCOLTA UMANA   collection.py  -> train_set/laps/lap_NNN.h5   (stato 29D, azione 4D, 50Hz)
2. BEHAVIORAL CLON. train_bc.py    -> bc_policy.pth + state_norm.npz
3. WARM-START + RL  train_rl.py    -> Actor inizializzato dalla BC, poi TD3+BC online
       │
       ├── HARVEST  il LapRecorder salva i giri completi/puliti -> train_set/laps_auto/
       │
4. ENRICHMENT       train_bc.py --auto_laps laps_auto -> BC ri-addestrata su (umano ∪ auto)
5. TIME-ATTACK      train_rl.py (IM_TIME_ATTACK=1) -> l'agente batte i propri tempi
6. TEST             test_agent.py  -> valutazione deterministica, auto-detect del best
```

Il cuore concettuale è il **flywheel dati (3 → 4 → 5)**. La BC iniziale è addestrata sui ~75 giri
umani, il cui migliore è 69.54s: imitare la *media* di quei giri tira la policy verso un giro
mediocre. Con l'RL l'agente impara a chiudere giri **più puliti e ripetibili** di quelli umani;
il `LapRecorder` li cattura e li reimmette nel dataset. Riaddestrando la BC su questo dataset
arricchito, il **punto di partenza** della prossima iterazione RL è più alto. Iterando, il sistema
si solleva da solo verso tempi che nessun giro umano del dataset contiene.

---

## 3. Rappresentazione dello stato

Lo stato è un vettore **29D** costruito da `common/state.py` nell'ordine:

| Indice | Feature | Scala | Note |
|---|---|---|---|
| 0 | `angle` | rad | angolo vettura rispetto all'asse pista |
| 1–19 | `track[19]` | /200 | distanza dai bordi su 19 raggi (vedi angoli sotto) |
| 20 | `trackPos` | — | posizione trasversale (0 = centro, ±1 = bordi) |
| 21–23 | `speedX/Y/Z` | /50 | velocità longitudinale/laterale/verticale |
| 24–27 | `wheelSpinVel[4]` | /100 | velocità di rotazione delle 4 ruote |
| 28 | `rpm` | /10000 | giri motore |

La marcia e `distFromStart` **non** fanno parte dello stato: la marcia è gestita da `gearing.py`;
`distFromStart` è raccolta solo come metadato, perché far correlare l'azione alla posizione assoluta
introdurrebbe un train-test mismatch (la policy deve guidare dai sensori, non "a memoria").

**Due forme dello stato** (è la distinzione più delicata del progetto):
- `flatten_state_raw(obs)` — vettore **grezzo** (scala fisica). È ciò che viene scritto negli HDF5
  (giri umani e auto) e usato dal `LapRecorder`.
- `flatten_state_norm(obs)` = `apply_state_norm(flatten_state_raw(obs))` — vettore **z-scored**
  ((x − media) / (std + 1e-3)), la forma data in input alla rete in RL/eval.

Le statistiche media/std sono calcolate dalla BC sull'intero dataset, salvate in
`train_set/checkpoints/state_norm.npz` e condivise da training ed eval (devono coincidere).

**Frame stacking temporale.** La rete non vede un singolo istante ma 3 frame distanziati di
`FRAME_STRIDE_K = 6` step (t-12, t-6, t), concatenati in un input **87D**. Questo dà alla rete
informazione implicita su velocità e accelerazione, utile a prevedere la traiettoria. La
`FrameStacker` (in `state.py`) incapsula questa logica.

**Angoli dei 19 sensori** (`constants.SENSOR_ANGLES_DEG`, in gradi):
`-45 -19 -12 -7 -4 -2.5 -1.7 -1 -0.5 0 0.5 1 1.7 2.5 4 7 12 19 45`. Distribuzione fitta vicino a 0°
(lookahead lungo l'asse) e rada a ±45° (rilevamento bordi). Sono usati **sia** dal client SCR (per
inizializzare i raggi) **sia** dalla data augmentation della BC (per perturbarli in modo coerente):
per questo sono un'unica costante condivisa.

---

## 4. Reti neurali

Definite in `models/networks.py`. Tutte le reti che producono comandi condividono lo stesso
**backbone** (4 blocchi `Linear(512) → LayerNorm → ReLU`) e la stessa **testa continua**
`Linear(512, 3)`; cambia solo l'attivazione delle uscite.

- **Actor** (RL): `forward` applica `tanh` a tutti e 3 i canali (uscite in [-1, 1]); `sample` aggiunge
  rumore esplorativo gaussiano clippato (annealato dal training). `load_bc_weights` esegue il
  warm-start dalla BC riscalando gas/freno di 0.5 (Sigmoid→Tanh).
- **PolicyNetwork** (BC ed eval): `forward` con `tanh`(sterzo)+`sigmoid`(gas/freno → [0,1]),
  l'attivazione con cui si addestra la BC; `sample` con `tanh` su tutti i canali, per valutare anche
  i checkpoint RL con la stessa classe.
- **Critic**: Twin Q-Network (q1, q2 indipendenti) su input 87D+3D; nel target di Bellman si usa il
  minimo tra i due per contrastare la sovrastima del valore.

**Compatibilità dei checkpoint.** Backbone e testa producono chiavi `state_dict` identiche a quelle
del codice originale (`backbone.0/1/3/4/6/7/9/10`, `continuous_head`, `q1/q2.0/2/4`). Questo è un
vincolo non negoziabile: i record storici in `train_set/checkpoints/` devono caricarsi senza
migrazione. Poiché Actor e PolicyNetwork condividono le chiavi, i pesi BC e RL sono interscambiabili.

---

## 5. Ambiente TORCS

`env/gym_torcs.py` è il wrapper Gym; `env/snakeoil3_gym.py` il client UDP SCR a basso livello.

**Azioni** (4D): `[steer ∈ [-1,1], accel ∈ [0,1], brake ∈ [0,1], gear]`. La rete predice i primi 3;
la marcia è calcolata da `gearing.py` con soglie di velocità/RPM e isteresi (cooldown) per evitare
il "hunting" (cambi marcia oscillanti). `action_mapping.py` converte tra spazio rete e pedali e
applica la **mutual exclusion** (`accel ← accel·(1−brake)`) per evitare gas e freno premuti insieme.

**Reward per-step** (in `gym_torcs.py`):

```
reward = 1.5 · progress + pos_penalty − 0.05 · |Δsteer|
  progress    = (speedX / 50) · cos(angle)          # avanzamento lungo l'asse pista
  pos_penalty = −2 · max(0, |trackPos| − 1)²         # barriera morbida ai bordi
```

`progress` premia la velocità proiettata lungo la pista (≈0 se la vettura è di traverso, negativo se
va all'indietro). `pos_penalty` è nulla dentro i bordi e cresce quadraticamente oltre |trackPos|=1.
Il termine anti-zigzag penalizza i cambi di sterzo bruschi, favorendo traiettorie fluide.

**Terminazioni anticipate** (training): uscita di pista (|trackPos| > 1.25), stallo (progresso < 0.1
dopo 500 step ≈ 10s), testacoda (cos(angle) < 0), giro completato (cambio di `lastLapTime` dopo lo
step 500). I bonus/malus terminali (completamento, tempo, giro incompleto, record personale) sono
aggiunti dal training loop tramite `rl/reward.py`.

**Frequenza di controllo**: il protocollo SCR lavora nominalmente a 50Hz. Il wrapper rilancia
periodicamente TORCS (kill + autostart) per contrastare un memory leak osservato nei run lunghi.

---

## 6. Behavioral Cloning

Entrypoint `bc/train_bc.py` (classe `BehaviorCloningTrainer`). Addestra la `PolicyNetwork` a
riprodurre le azioni umane con una **loss MSE pesata per canale**:

- pesi base `[steer=1, accel=1, brake=3]`;
- boost dinamico del freno (×8 quando il pilota frena) per imparare le staccate, eventi rari ma critici;
- boost dello sterzo in curva (×4 quando |steer| > 0.07) per la precisione di traiettoria.

> **Revisione dei pesi (rispetto all'originale).** Il freno prima pesava fino a 5×25 = **125×** lo
> sterzo: la loss diventava quasi un solo regressore di frenata, a scapito della precisione di sterzo.
> Ora il picco è ~24× (base 3 × boost 8), con più enfasi sullo sterzo in curva.

### Data augmentation Bojarski-style

`bc/augmentation.py` (`AugmentConfig` + `augment_batch`). Per mitigare il *covariate shift* (in
inferenza la vettura finisce in stati che il pilota non ha mai visitato), si perturbano sinteticamente
posizione laterale e angolo, correggendo i target per insegnare il rientro verso il centro:

- perturbazione laterale `trackPos` (σ 0.22) e angolare (σ 0.09 rad, clip ~10°);
- i 19 raggi vengono perturbati in modo **geometricamente coerente** con lo spostamento simulato;
- il target di sterzo è corretto verso il centro (guadagni configurabili), l'acceleratore ridotto in
  funzione dell'entità della perturbazione, più un ramo "overspeed" che insegna a frenare prima delle
  curve ad alta velocità.

> **Correzioni chiave (rispetto all'originale).**
> 1. **Clamp on-track** (`on_track_limit = 0.95`): il `trackPos` perturbato non supera mai il bordo,
>    quindi la rete impara a recuperare verso il centro **da pose ancora in pista** — non le si insegna
>    mai a guidare fuori pista (era un rischio reale della versione precedente, senza clamp).
> 2. **Perturbazione angolare ampliata** da ~4.5° a **~10°**: copre disallineamenti realistici di metà
>    curva, da cui prima la rete non imparava a rientrare.

Tutti i parametri sono in `AugmentConfig`, quindi tarabili senza toccare il codice.

---

## 7. TD3+BC

Entrypoint `rl/train_rl.py`, agente `rl/agent.py`. Combina TD3 (Fujimoto et al. 2018) con il vincolo
BC del TD3+BC (Fujimoto & Gu 2021): architettura offline-to-online che eredita la conoscenza della BC
(warm-start dell'Actor) e la raffina con l'RL senza far collassare la policy.

**Loss dell'Actor:** `L = −λ · Q(s, π(s)) + BC_penalty`, con `λ = bc_alpha / mean(|Q(s, π(s))|)`. La
normalizzazione dinamica di λ mantiene confrontabili la scala del termine RL e di quello BC, rendendo
il gradiente stabile rispetto a variazioni dei Q-value. `bc_alpha` più alto = più peso al RL (utile a
superare l'esperto). La **BC penalty** è un MSE tra azione predetta e azione esperta applicato
**solo** ai campioni con `expert_mask = 1` (mascheramento rigoroso): così l'agente può esplorare
traiettorie diverse da quelle umane senza essere penalizzato.

**Campionamento ibrido a tre vie** (per ogni batch): 25% **expert** (umano), 15% **elite** (migliori
run autonome), 60% **online** (esplorazione corrente). Se online/elite hanno pochi dati, la quota è
compensata dall'expert (sempre disponibile).

Caratteristiche di stabilità (tutte preservate dal codice originale):
- **Ancora progressiva**: il buffer expert è permanente (capacità 400k >> dataset) e filtrato sui
  *migliori* giri umani (`--expert_max_lap_time`), così l'ancora BC punta al best umano, non alla media.
- **Elite gate**: un episodio entra nel buffer elite solo se la distanza percorsa supera il 70% del
  record corrente; gli ultimi 50 step prima di un crash sono esclusi dall'imitazione (anti causal-confusion).
- **Refinement FSM**: su plateau della valutazione, riduce il peso della BC e congela il Critic per
  raffinare l'Actor verso una value function fissa; con rollback su collasso e uscita su breakout.
- **Warm-up**: l'Actor resta congelato finché il Critic non si stabilizza (15000 step), aggiornandosi
  poi con Delayed Policy Update (ogni 2 step del Critic) e Target Policy Smoothing.

---

## 8. Lap recorder ed enrichment

`data/lap_recorder.py` (`LapRecorder`). Durante il training, per ogni step cattura lo **stato grezzo
pre-step** (`flatten_state_raw`) e l'**azione realmente eseguita** su TORCS (`[steer, accel applicato,
brake, gear]`). A fine episodio salva il giro in `train_set/laps_auto/lap_auto_NNN.h5` **solo se**:

- il giro è stato **completato** (SUCCESS, non crash/incompleto);
- è **pulito** (`max|trackPos| ≤ on_track_limit`, default 1.0 = mai fuori pista);
- è abbastanza **veloce** (`lap_time ≤` soglia, default 80s, `IM_RECORD_MAX_LAP_TIME`);
- ha lunghezza minima e nessun valore NaN/Inf.

Il formato HDF5 è **identico** a quello dei giri umani (dataset `states`/`actions`/`dist_from_start`
+ attributi), quindi i giri auto sono caricabili sia da `TorcsHDF5Dataset` (per la BC) sia da
`load_expert_data` (per il buffer RL). L'enrichment si attiva con `train_bc.py --auto_laps laps_auto`,
che unisce umano+auto e **ricalcola** `state_norm.npz` sull'unione.

---

## 9. Time-attack e record personale

In `rl/reward.py`:
- **Bonus di record personale** (sempre attivo): quando l'agente stabilisce un nuovo miglior tempo,
  riceve `+30 + 15·(secondi guadagnati)` oltre al bonus di completamento. È l'incentivo diretto a
  limare i tempi anche quando la distanza è ormai saturata a fine giro.
- **Fase TIME-ATTACK** (`IM_TIME_ATTACK=1`): da attivare *dopo* aver raccolto abbastanza giri e
  ri-addestrato la BC. Riduce l'ancoraggio alla BC (`bc_alpha` → 4.0, più peso al RL) e abbassa il
  floor del rumore esplorativo (0.04 → 0.02) per la micro-ottimizzazione della traiettoria.

---

## 10. Sistema di checkpoint

`common/checkpoint.py`. Salvataggio **atomico** e resistente alle interruzioni: scrittura su file
temporaneo → `fsync` → rotazione backup (`.bak` → `.prev`) → `os.replace` atomico → `fsync` della
directory. Al caricamento si scandiscono in ordine i candidati (file principale, `.bak`, `.prev`),
così un'interruzione a metà scrittura non lascia mai un checkpoint corrotto.

**Archivio intoccabile.** I record deterministici (`td3_det_best_lap.pth`, `td3_det_best_dist.pth`)
sono accompagnati da sidecar testuali (`.txt`) col valore, vengono sovrascritti solo su miglioramento
e sopravvivono a `--clean`. Il resume allinea temporalmente i buffer al checkpoint (non carica buffer
più recenti del `.pth`, indizio di un salvataggio successivo interrotto).

File principali in `train_set/checkpoints/`:

| File | Significato |
|---|---|
| `bc_policy.pth` | policy addestrata in sola BC (warm-start dell'RL) |
| `state_norm.npz` | media/std degli stati per la normalizzazione |
| `td3_checkpoint.pth` | checkpoint completo per il resume del TD3+BC |
| `td3_det_best_lap.pth` | miglior giro valido deterministico (candidato submission) |
| `td3_det_best_dist.pth` | miglior score deterministico assoluto |
| `td3_expl_best_lap/dist.pth` | migliori risultati trovati in esplorazione |

---

## 11. Orchestratore run.sh

`run.sh` è il controller unico della pipeline (sostituisce i vecchi `train_bc.sh`/`train_rl.sh`/
`stop_training.sh`). Imposta `PYTHONPATH=src`, lancia i task lunghi in background salvando PID e log,
e offre un cruscotto di stato.

```
./run.sh collect [args]       # raccolta giri umani (foreground)
./run.sh bc [args]            # training BC (background)
./run.sh bc-enriched [args]   # training BC su umano+auto (background)
./run.sh rl [args]            # TD3+BC + harvest dei giri (background)
./run.sh time-attack [args]   # fase time-attack (background)
./run.sh test [args]          # valutazione deterministica (foreground)
./run.sh status               # processi attivi, dataset, record, ultimi log
./run.sh logs <task> [n]      # ultime n righe del log di un task
./run.sh stop [task]          # stop pulito (SIGINT) di un task o di tutti
```

`status` mostra: task in esecuzione (con uptime), numero di giri umani/auto, miglior tempo
deterministico, episodio del checkpoint, e l'ultima riga di log di ogni task. `stop` invia SIGINT, che
gli entrypoint di training intercettano per salvare un checkpoint completo prima di uscire.

---

## 12. Configurazione

| Dove | Nome | Default | Significato |
|---|---|---|---|
| env | `IM_RECORD_LAPS` | 1 | abilita il lap recorder durante il training |
| env | `IM_RECORD_MAX_LAP_TIME` | 80.0 | soglia tempo (s) per salvare un giro auto |
| env | `IM_TIME_ATTACK` | 0 | attiva la fase time-attack |
| env | `TORCS_KILL_ALL` | 1 | kill globale di TORCS (workaround memory leak) |
| env | `SHOW_GUI` | 0 | mostra la finestra TORCS invece di Xvfb headless |
| rl | `--episodes` | 1000 | episodio finale (deve superare quello di resume) |
| rl | `--bc_alpha` | 2.5 | bilanciamento RL/BC |
| rl | `--expert_max_lap_time` | 71.0 | filtro qualità dell'ancora BC |
| bc | `--auto_laps DIR` | — | directory dei giri auto per l'arricchimento |
| bc | `--epochs / --batch_size / --lr` | 300 / 256 / 3e-4 | iperparametri di training |

I parametri della data augmentation sono in `AugmentConfig` (`bc/augmentation.py`).

---

## 13. Speedup RL (differito)

Punto lasciato volutamente non implementato (da valutare in futuro), con i ganci già predisposti:
- `env/gym_torcs.py::_kill_torcs` documenta dove sostituire il kill globale con un teardown
  per-istanza, necessario per far girare N ambienti TORCS in parallelo (porte 3001+i, display Xvfb
  separati);
- il rapporto update/step (UTD) è concentrato in un unico punto del loop (`agent.update`), facile da
  parametrizzare.

---

## 14. Note operative

- **Lineage dei checkpoint dopo l'enrichment.** Ricalcolare `state_norm.npz` cambia la normalizzazione:
  i checkpoint TD3 addestrati con la vecchia non sono più coerenti. Per adottare una BC arricchita
  conviene avviare una **nuova lineage**: copiare i nuovi `bc_policy.pth` + `state_norm.npz` in
  `checkpoints/`, poi `./run.sh rl` dopo aver azzerato i checkpoint TD3 (i record deterministici sono
  protetti dai sidecar). I checkpoint vecchi restano come archivio.
- I dati e i checkpoint in `train_set/` non sono tracciati da git (scelta del progetto): vanno
  salvati con backup esterni.
- Prerequisiti: Python 3, TORCS con server SCR, `xvfb-run` (headless), `xte` (autostart), e le
  librerie `torch`, `numpy`, `h5py`, `pygame`, `gym`.
