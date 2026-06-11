# Architettura di Iterative Motors

Iterative Motors è un agente di guida autonoma sviluppato per partecipare alla IBM AI Racing League 2026. L'obiettivo progettuale è costruire una policy capace di guidare in TORCS con prestazioni superiori a quelle umane: non solo imitare un pilota, ma usare l'apprendimento per rinforzo per rifinire traiettorie, staccate e gestione degli errori oltre il limite raggiungibile con la sola raccolta dati manuale.

Il progetto segue una pipeline "human-to-agent": prima acquisisce dimostrazioni umane, poi addestra una rete con Behavioral Cloning, infine usa TD3+BC per migliorare la policy dentro il simulatore. La scelta non è casuale: una guida racing richiede continuità nei controlli, stabilità numerica e recupero da stati fuori traiettoria. Per questo il sistema combina imitazione, reinforcement learning off-policy, buffer esperti permanenti, buffer elite e valutazioni deterministiche periodiche.

## Vista D'insieme

Il flusso completo è composto da cinque blocchi principali.

1. Raccolta dati umana: `data_collection.py` usa TORCS in modalità grafica, legge controller PS5 DualSense o tastiera, applica un TCS opzionale e salva giri validi o segmenti mirati in HDF5.
2. Pretraining supervisionato: `behavioral_cloning.py` addestra una `PolicyNetwork` che imita sterzo, acceleratore e freno del pilota umano. La marcia viene registrata ma non viene appresa dalla rete.
3. Fine-tuning TD3+BC: `td3_bc.py` inizializza l'Actor dai pesi BC, addestra un Twin Critic e ottimizza la policy con una loss ibrida RL/BC.
4. Ambiente TORCS: `gym_torcs/gym_torcs.py` espone un wrapper Gym-like sopra il client UDP SCR di `snakeoil3_gym.py`, normalizza le osservazioni e calcola la reward online.
5. Test deterministico: `test_agent.py` carica automaticamente il miglior checkpoint disponibile, esegue giri senza rumore esplorativo e salva telemetria CSV.

La separazione è intenzionale: raccolta dati, imitazione, ottimizzazione RL, gestione ambiente e testing hanno responsabilità diverse. Questo permette di modificare, per esempio, la reward o la strategia di cambio marcia senza riscrivere la rete neurale.

## Mappa Dei File

| File | Ruolo |
| --- | --- |
| `data_collection.py` | Raccolta HDF5 da pilota umano, con controller/tastiera, TCS, validazione del giro e segmentazione delle curve |
| `behavioral_cloning.py` | Addestramento supervisionato della policy iniziale su giri completi, con loss pesata e augmentation Bojarski-style |
| `td3_bc.py` | Fine-tuning TD3+BC: tre replay buffer, loss ibrida RL/BC, checkpoint atomici, evaluation periodica e auto-refinement |
| `test_agent.py` | Inferenza deterministica, auto-detect del checkpoint migliore e telemetria CSV per ogni tentativo |
| `gearing.py` | Cambio marcia algoritmico condiviso da training RL e test (soglie, isteresi e cooldown) |
| `gym_torcs/gym_torcs.py` | Wrapper ambiente: avvio/relaunch TORCS, normalizzazione osservazioni, reward shaping e condizioni di terminazione |
| `gym_torcs/snakeoil3_gym.py` | Client UDP SCR a basso livello: handshake, parsing sensori e invio azioni |
| `gym_torcs/autostart.sh` | Automazione menu TORCS tramite `xte` (Race → Practice → New Race → Start) |
| `train_bc.sh` | Script operativo per il Behavioral Cloning: pre-check del dataset e avvio con iperparametri di default |
| `train_rl.sh` | Script operativo per il TD3+BC: rileva resume/warm-start/cold-start, gestisce `--clean` e inoltra i flag a `td3_bc.py` |
| `stop_training.sh` | Arresto via `pkill` dei processi di training/test/TORCS/Xvfb |

Le directory `train_set/` e `telemetry/` sono pensate per artefatti locali. La `.gitignore` mantiene versionate solo le sottodirectory tramite `.gitkeep`, mentre dataset, checkpoint e CSV restano fuori dal versionamento.

## Ambiente TORCS E Comunicazione

### Il client SCR (`snakeoil3_gym.py`)

Il progetto usa TORCS come simulatore fisico e il protocollo SCR (Simulated Car Racing) su UDP. `snakeoil3_gym.py` è una versione adattata della libreria Snakeoil: tutta la logica di guida vive negli script di livello superiore, mentre qui restano solo gestione del socket, parsing della telemetria e formato dell'azione. Il client espone due dizionari:

- `ServerState.d`: telemetria ricevuta da TORCS (`angle`, `curLapTime`, `damage`, `distFromStart`, `distRaced`, `focus`, `fuel`, `gear`, `lastLapTime`, `opponents`, `racePos`, `rpm`, `speedX/Y/Z`, `track`, `trackPos`, `wheelSpinVel`, `z`);
- `DriverAction.d`: comandi da inviare (`accel`, `brake`, `clutch`, `gear`, `steer`, `focus`, `meta`). Il flag `meta=True` è il segnale con cui il wrapper chiede a TORCS di terminare l'episodio.

La connessione avviene sulla porta `3001`. Il messaggio di init definisce gli angoli dei 19 sensori `track`: `-45 -19 -12 -7 -4 -2.5 -1.7 -1 -.5 0 .5 1 1.7 2.5 4 7 12 19 45` gradi. La distribuzione non è uniforme: i raggi sono molto più fitti vicino allo zero, dove serve risoluzione per anticipare le curve guardando lontano lungo l'asse pista, e più radi verso ±45° dove basta percepire i bordi. Questi stessi angoli sono riutilizzati dall'augmentation del BC per perturbare i sensori in modo geometricamente coerente.

Il socket ha timeout di 1 secondo. Se TORCS non risponde per 5 letture consecutive (`missed_packets`), il client chiude il socket e segnala l'uscita controllata: questo impedisce a un training notturno di restare bloccato all'infinito su un simulatore morto.

### Avvio, headless e relaunch (`gym_torcs.py`)

Quando viene istanziato `TorcsEnv`, il wrapper:

- verifica che `xvfb-run` sia installato (errore d'ambiente esplicito in caso contrario);
- termina eventuali processi TORCS rimasti attivi con `pkill -9 -f torcs`, salvo `TORCS_KILL_ALL=0`. Il kill aggressivo è il workaround operativo contro il memory leak osservato nei run lunghi di TORCS;
- attende 1.5 secondi perché il sistema operativo liberi la porta UDP (altrimenti il riavvio fallisce);
- avvia `torcs -nofuel -nodamage` — direttamente se `SHOW_GUI=1`, altrimenti dentro `xvfb-run -a -s "-screen 0 640x480x24"` per l'esecuzione headless;
- lancia in parallelo `autostart.sh`, che con `xte` invia la sequenza di tasti dei menu (Race → Practice → New Race → Start) con pause da 200 ms;
- attende altri 3 secondi per avvio e macro, poi espone `reset()`, `step()`, `get_obs()` ed `end()`.

Durante la data collection `data_collection.py` forza `SHOW_GUI=1`, perché il pilota deve vedere la pista. Durante training e test il sistema gira headless: nel TD3+BC non serve renderizzare per un umano, e si evita di sprecare risorse grafiche. `reset(relaunch=True)` chiude esplicitamente il socket UDP e ripete l'intera procedura di kill/avvio: ripartire da un processo pulito riduce la probabilità che stato sporco, socket bloccati o memory leak del simulatore falsino il training.

### Normalizzazione delle osservazioni

`make_observaton()` converte la telemetria grezza nel dizionario usato dagli script, applicando già alcune scale:

| Campo | Scala applicata dal wrapper |
| --- | --- |
| `track` (19), `focus` (5), `opponents` (36) | `/200` (i sensori di distanza hanno portata 200 m) |
| `speedX`, `speedY`, `speedZ` | `/default_speed` con `default_speed = 50` |
| `angle`, `trackPos`, `rpm`, `wheelSpinVel`, `damage` | grezzi |
| `curLapTime`, `lastLapTime`, `distFromStart`, `distRaced` | grezzi (timing/posizione) |

`focus` e `opponents` vengono normalizzati dal wrapper ma non entrano nello stato 29D della policy: in modalità pratica non ci sono avversari, e il blocco `track` copre già la percezione della pista.

`default_speed = 50` non è una velocità massima ma un valore di riferimento tipico in pista, scelto in modo che le velocità normalizzate restino in un range gestibile (gli script downstream ricostruiscono i km/h come `speedX * 50`). Cambiarlo scalerebbe sia le osservazioni sia il termine di progresso della reward, quindi va tenuto coerente.

### Reward online

La reward per step è calcolata direttamente in `step()`:

```text
progress       = (speedX / default_speed) * cos(angle)
pos_penalty    = -2.0 * max(0, |trackPos| - 1.0)^2
smooth_penalty = -0.05 * |steer_t - steer_t-1|
reward         = 1.5 * progress + pos_penalty + smooth_penalty
```

- `progress` premia la proiezione della velocità lungo l'asse pista, non il semplice movimento: `cos(angle)` annulla il contributo se l'auto è di traverso e lo rende negativo se è girata.
- `pos_penalty` è nulla finché `|trackPos| < 1.0` (vettura entro i bordi) e cresce con rampa quadratica tra 1.0 e 1.25: agisce come barriera virtuale morbida.
- `smooth_penalty` penalizza la variazione di sterzo tra step adiacenti: scoraggia lo zigzag nei rettilinei e spinge verso traiettorie fluide. `last_steer` viene inizializzato a 0 al primo step (sterzo dritto).

La reward è volutamente minimale: non impone una traiettoria, una velocità target o un punto di frenata, perché farlo limiterebbe la possibilità di superare il comportamento umano. Il calcolo sta in `gym_torcs.py` (e non in `td3_bc.py`) per convenienza: qui sono direttamente accessibili `obs`, `last_steer` e le variabili di stato necessarie.

### Terminazione dell'episodio

Con `early_termination=True` (training RL), l'episodio termina nei casi seguenti, ciascuno con la sua penalità locale e il suo tag diagnostico in `info['termination_reason']`:

| Condizione | Soglia | Penalità sullo step | Tag |
| --- | --- | --- | --- |
| Fuori pista | `\|trackPos\| > 1.25` (`off_track_limit`) | `-5.0` base `- 5.0 * eccesso` (eccesso saturato a 1.0) | `OFF_TRACK` |
| Stallo | `progress < 5/50` dopo i primi 500 step (`terminal_judge_start`, ~10 s a 50 Hz) | `-5.0` (`incomplete_lap_step_penalty`) | `STALL` |
| Testacoda | `cos(angle) < 0` | `-5.0` | `SPIN` |
| Giro completato | `lastLapTime` cambia di oltre 0.01 s con `time_step > 500` | nessuna (il bonus è aggiunto in `td3_bc.py`) | `SUCCESS` |

Il limite `|trackPos| > 1.25` è lo stesso usato in raccolta dati: i cordoli sono consentiti fino a 1.25, oltre il giro è invalido. Il completamento del giro è rilevato confrontando `lastLapTime` con uno snapshot pre-step: TORCS aggiorna quel campo solo al passaggio sul traguardo di un giro valido. Il dizionario `info` riporta anche `crash`, `off_track`, `lap_completed` e `lap_time`, usati da `td3_bc.py` per il done masking e i record.

In `data_collection.py` e `test_agent.py` l'ambiente è creato con `early_termination=False`: la validazione del giro è gestita dagli script stessi, con criteri propri descritti più avanti.

## Stato Sensoriale

La policy non riceve immagini. Riceve un vettore sensoriale compatto a 29 dimensioni costruito da `flatten_state()` (implementata in modo identico in `data_collection.py`, `td3_bc.py` e `test_agent.py`):

| Indici | Blocco | Dim. | Descrizione |
| --- | --- | ---: | --- |
| 0 | `angle` | 1 | Angolo tra asse vettura e asse pista (radianti) |
| 1–19 | `track` | 19 | Sensori di distanza dal bordo pista, già scalati `/200` dal wrapper |
| 20 | `trackPos` | 1 | Posizione laterale rispetto al centro pista |
| 21–23 | `speedX/Y/Z` | 3 | Velocità già scalate `/50` dal wrapper |
| 24–27 | `wheelSpinVel` | 4 | Velocità angolari delle ruote, riscalate `/100` in `flatten_state()` |
| 28 | `rpm` | 1 | Regime motore, riscalato `/10000` in `flatten_state()` |

Gli indici sono rilevanti perché l'augmentation del BC e il ricalcolo offline della reward vi accedono per posizione (es. `trackPos` all'indice 20, `speedX` all'indice 21, sensore frontale a 0° all'indice 10). `flatten_state()` usa accessi con default (`.get()`) e padding a zero su chiavi mancanti, e in caso di errore restituisce un vettore di zeri segnalandolo a video: uno stato silenziosamente corrotto falserebbe l'inferenza e causerebbe incidenti.

Questa scelta privilegia controllo e campionamento rispetto alla percezione visiva. Le immagini richiederebbero una CNN, molti più dati, più GPU e introdurrebbero un problema di visione che non è centrale per l'obiettivo: in TORCS i sensori SCR forniscono già geometria pista, velocità e stato meccanico. Usare sensori numerici rende il learning più sample-efficient e permette al TD3+BC di concentrarsi sulle decisioni racing.

`distFromStart` viene salvata negli HDF5 come metadato (`dist_from_start`), ma non entra nello stato della rete. Questa scelta evita un train-test mismatch: se la rete vedesse la posizione assoluta, potrebbe imparare un'associazione rigida tra metro del tracciato e comando, cioè "ricordare" una sequenza di azioni invece di guidare dai sensori. Il metadato resta utile per segmentazione e analisi.

### Frame stacking

La rete usa frame stacking: lo stato finale ha 87 dimensioni, ottenute concatenando tre frame da 29 dimensioni agli istanti `t-12`, `t-6` e `t` (`k=6`, circa 0.24 secondi di storia a 50 Hz). In pratica gli script mantengono una `deque` di 13 frame e concatenano gli elementi alle posizioni 0, 6 e 12; al reset la deque è inizializzata replicando 13 volte lo stato iniziale, e nel dataset BC gli indici `t-6`/`t-12` sono clampati a 0 a inizio giro. Questo aggiunge informazione temporale senza passare a una rete ricorrente: l'Actor vede non solo dove si trova l'auto, ma anche come ci sta arrivando (deriva, accelerazione, tendenza dello sterzo).

Il frame stacking è stato preferito a LSTM/GRU perché mantiene l'Actor semplice e deterministico. Una rete ricorrente potrebbe modellare dinamiche più lunghe, ma complicherebbe il replay buffer, il resume e la stabilità del Critic. Tre frame distanziati sono un compromesso pragmatico.

### Normalizzazione mean/std

Al termine del BC, `behavioral_cloning.py` calcola media e deviazione standard di ciascuna delle 29 feature sull'intero dataset esperto e le salva in `train_set/checkpoints/state_norm.npz`. TD3+BC e test caricano lo stesso file e applicano:

```text
s_norm = (s - mean) / (std + 1e-3)
```

Il termine `1e-3` evita divisioni per zero su sensori statici. Se il file non esiste, la normalizzazione è un no-op (fallback). L'approccio — statistiche globali pre-calcolate sull'intero dataset — segue Fujimoto & Gu (2021): stabilizza l'offline RL e ne migliora le prestazioni, perché le feature derivano da sensori con scale e distribuzioni molto diverse. Il file usato in training e in test DEVE essere lo stesso, altrimenti la rete riceve input in una scala mai vista.

## Azioni E Cambio Marcia

TORCS riceve un'azione 4D:

```text
[steer, accel, brake, gear]
```

La rete neurale predice solo i primi tre comandi. La marcia è gestita da `gearing.py` (vedi sotto): questo riduce lo spazio d'azione e impedisce alla rete di sprecare capacità su una decisione discreta facilmente descrivibile con soglie.

### Attivazioni e conversione BC → TD3

Nel Behavioral Cloning le attivazioni rispecchiano i vincoli fisici dei comandi registrati:

- `steer` usa `tanh` → `[-1, 1]` (lo sterzo è simmetrico intorno a zero);
- `accel` e `brake` usano `sigmoid` → `[0, 1]` (i pedali sono non negativi).

Nel TD3+BC l'Actor usa `tanh` su tutti e tre i canali, perché l'algoritmo lavora in uno spazio continuo normalizzato e simmetrico (rumore esplorativo, smoothing del target e clipping operano tutti in `[-1, 1]`). La conversione dei pedali avviene solo al momento dell'interazione con TORCS:

```text
accel_torcs = clip((accel_tanh + 1) / 2, 0, 1)
brake_torcs = clip((brake_tanh + 1) / 2, 0, 1)
accel_torcs = accel_torcs * (1 - brake_torcs)   # mutua esclusione
```

La mutua esclusione moltiplicativa evita una condizione poco realistica e dannosa — accelerare e frenare insieme — con una formula continua che non crea salti bruschi nella policy.

Quando l'Actor TD3 parte dai pesi BC, `load_bc_weights()` moltiplica per `0.5` pesi e bias dei canali acceleratore/freno della `continuous_head`. Il motivo è matematico: `tanh(z/2) = 2*sigmoid(z) - 1`, quindi dimezzare i logits trasferisce esattamente la funzione appresa con la sigmoide nel nuovo range tanh, senza rompere il comportamento iniziale.

### Cambio marcia algoritmico (`gearing.py`)

`compute_gear(speed_kmh, accel, rpm, current_gear, steps_since_shift)` restituisce la marcia (1..6) cambiando al massimo di ±1 per chiamata. Le soglie sono calibrate sulla telemetria dei piloti esperti:

| Parametro | Valore | Significato |
| --- | --- | --- |
| `UP_SPEED` | `[55, 118, 200, 258, 286]` km/h | Velocità minima per salire da 1→2, 2→3, … 5→6 |
| `DN_SPEED` | `[40, 92, 165, 232, 272]` km/h | Velocità sotto cui scendere da 2→1, 3→2, … 6→5 |
| `UP_RPM_GATE` | `15500` RPM | Non salire se i giri non sono già alti |
| `UP_ACCEL_GATE` | `0.4` | Non salire se non si è sul gas |
| `SHIFT_COOLDOWN` | `5` step | Lockout dopo ogni cambio |

L'upshift richiede tutte e tre le condizioni (velocità, RPM, gas): impedisce cambiate spurie in rilascio o in frenata. Il downshift si basa solo sulla velocità, che in frenata cala in modo monotono, evitando le oscillazioni dovute ai picchi temporanei di RPM. L'anti-jitter è duplice: isteresi strutturale (`DN_SPEED < UP_SPEED` per ogni marcia) e cooldown temporale. Il parametro `accel` passato deve essere quello effettivamente applicato dopo la mutua esclusione, per evitare false cambiate in staccata.

In raccolta dati, invece, il cambio è manuale (pulsanti/frecce) e la marcia inserita viene registrata nel dataset; la rete la ignora.

## Raccolta Dati

`data_collection.py` genera il dataset esperto. Il loop gira a 50 Hz con controllo dinamico del frame rate (misura il tempo del ciclo e dorme per la differenza rispetto a `1/50 s`).

### Dispositivi di input

Controller PS5 DualSense (`DualSenseController`, via Pygame):

- stick sinistro (asse 0, negato) → sterzo continuo, con deadzone configurabile (default 0.05) e riscalatura del range residuo su `[-1, 1]`, così la deadzone non crea un gradino;
- R2 (asse 5) → acceleratore `[0, 1]`; L2 (asse 2) → freno `[0, 1]`. Entrambi hanno una protezione warm-up: i grilletti riportano valori spuri prima della prima pressione, quindi l'asse è considerato valido solo dopo aver superato `|raw| > 0.1` una prima volta; sotto 0.05 il pedale è azzerato;
- quadrato → upshift, X → downshift, con debounce di 200 ms;
- la coda eventi Pygame viene svuotata ad ogni poll per evitare input lag;
- `rumble()` fornisce un feedback aptico all'ingresso delle zone target.

Tastiera (`KeyboardController`): apre una finestra Pygame 100×100 che deve mantenere il focus. A/D pilotano un target di sterzo ±1 raggiunto con interpolazione incrementale di 0.08 per step (sterzo fluido nonostante l'input digitale); W/S sono acceleratore e freno on/off con priorità al freno in caso di pressione simultanea; frecce su/giù cambiano marcia con debounce di 250 ms.

### TCS (Traction Control System)

Il TCS opzionale (`--tcs`, attivo di default) confronta lo spin medio delle ruote posteriori con quello delle anteriori:

```text
slip = (wsv[2] + wsv[3])/2 - (wsv[0] + wsv[1])/2
se slip > 5.0:  accel *= max(0.2, 1 - (slip - 5.0)/30)
```

Più slittamento, più taglio (fino all'80%). Il TCS non è parte della policy finale: serve solo a rendere più pulite le dimostrazioni umane e a ridurre i giri scartati.

### Validazione del giro

L'ambiente è creato con `early_termination=False`; la validazione è interna allo script:

- fuori pista: se `|trackPos| > 1.25` il giro è immediatamente invalidato e viene forzato un relaunch di TORCS;
- traguardo (A): `lastLapTime` cambia rispetto allo snapshot → giro completato; valido solo se non c'è stata uscita di pista, e il lap time è quello riportato da TORCS;
- traguardo (B): `curLapTime < 1.5` mentre il valore precedente era `> 5.0` → TORCS ha resettato il cronometro senza aggiornare `lastLapTime`, cioè ha invalidato il giro (taglio o uscita);
- traguardo (C, fail-safe geometrico): `distFromStart < 50` con valore precedente `> 500` e `step > 500` → passaggio sul traguardo rilevato dalla posizione quando i timer non si aggiornano; il giro è chiuso come invalido.

Solo i giri completati e validi vengono scritti su disco; Ctrl+C scarta il giro corrente. TORCS viene rilanciato al primo giro, ogni `--relaunch_every` giri (default 10, contro il memory leak) e dopo ogni uscita di pista.

### Zone problematiche e segmenti

`PROBLEM_ZONES` elenca 9 intervalli di `distFromStart` (in metri) corrispondenti alle curve del tracciato, estratti da `corkscrew.xml` e leggermente allargati per includere staccate e uscite di curva. `--zones "a:b,c:d"` permette di sovrascriverli. All'ingresso di ogni zona il controller vibra (puro feedback per il pilota). Con `--zones` specificate il giro termina anticipatamente 10 m dopo la fine dell'ultima zona attraversata.

Con `--segment_only`, a fine giro valido vengono salvati solo i run contigui di step interni alle zone, estesi di 15 step di margine in approccio e scartati se più corti di 20 step. Ogni segmento è nominato `lap_seg_{za}m_{zb}m_{NNN}.h5` con la zona di appartenenza nel nome.

La distinzione tra giri completi e segmenti è importante. I giri completi rappresentano la distribuzione globale di guida e sono adatti al Behavioral Cloning. I segmenti aumentano la densità di dati nelle curve problematiche: usarli nel BC sbilancerebbe la media delle azioni, ma usarli come expert data nel TD3+BC aiuta Critic e Actor a rivedere proprio gli stati più difficili.

### Formato HDF5

Il formato HDF5 salva array numerici compressi, attributi e metadati nello stesso file, restando facile da leggere con `h5py`: per sequenze dense di stati/azioni è più adatto di CSV o JSON. Ogni file contiene:

- `states`: sequenza di stati 29D (compressione gzip);
- `actions`: sequenza di azioni registrate `[steer, accel, brake, gear]`;
- `dist_from_start`: metadato per analisi e segmentazione;
- attributi `lap_time`, `num_steps`, `has_dist_meta`, `timestamp`.

I log testuali di sessione finiscono in `train_set/session_logs/giri/session_YYYYMMDD_HHMMSS.log`, con una riga `[SALVATO]`/`[SCARTATO]` per tentativo e un riepilogo finale.

## Behavioral Cloning

### Dataset (`TorcsHDF5Dataset`, `load_dataset`)

Il training usa solo i giri completi `lap_[0-9]*.h5` (ricerca ricorsiva), escludendo i segmenti `lap_seg_*.h5`: la BC minimizza l'errore medio sull'azione esperta, e un dataset pieno di soli segmenti di curva sbilancerebbe la policy. I file vengono concatenati in un `ConcatDataset`; un file illeggibile produce un warning, non un crash.

All'inizializzazione ogni file subisce sanity check: presenza dei gruppi `states` e `actions`, assenza di NaN e Inf in entrambi. Il check è bloccante perché valori non validi nel dataset provocherebbero instabilità o collasso della policy in addestramento. `__getitem__` restituisce lo stack `(t-12, t-6, t)` con clamping a 0 ai bordi del giro e l'azione target dell'istante `t`.

### Rete (`PolicyNetwork`)

La rete è una MLP (Multi-Layer Perceptron): una sequenza di strati fully-connected in cui ogni neurone riceve in input tutte le uscite dello strato precedente. È una MLP — e non una CNN o una rete ricorrente — perché l'input è già un vettore numerico strutturato: non c'è un'immagine da cui estrarre feature spaziali, e l'informazione temporale è fornita dal frame stacking.

Struttura completa, strato per strato:

| # | Strato | Dimensioni | Parametri |
| --- | --- | --- | ---: |
| 1 | `Linear` → `LayerNorm` → `ReLU` | 87 → 512 | 45 056 + 1 024 |
| 2 | `Linear` → `LayerNorm` → `ReLU` | 512 → 512 | 262 656 + 1 024 |
| 3 | `Linear` → `LayerNorm` → `ReLU` | 512 → 512 | 262 656 + 1 024 |
| 4 | `Linear` → `LayerNorm` → `ReLU` | 512 → 512 | 262 656 + 1 024 |
| 5 | `continuous_head` (`Linear`) + attivazioni | 512 → 3 | 1 539 |
| | **Totale** | | **≈ 838 700** |

Che cosa fa ogni componente e perché è lì:

- **`Linear` (strato fully-connected)**: ogni neurone calcola una somma pesata di tutti gli input più un bias (`y = W·x + b`). I pesi `W` e i bias `b` sono i parametri che il training modifica. Il primo strato elabora per la prima volta le 87 feature sensoriali e le proietta in uno spazio a 512 dimensioni; gli strati successivi ricombinano queste rappresentazioni in pattern sempre più astratti (es. "ingresso curva veloce con auto in deriva").
- **`LayerNorm` (Layer Normalization)**: normalizza le 512 attivazioni di ogni campione portandole a media 0 e deviazione standard 1, poi applica una scala e uno shift appresi (i 1 024 parametri per strato: 512 + 512). Serve perché le feature derivano da sensori con scale e distribuzioni molto diverse: senza normalizzazione alcune attivazioni dominerebbero le altre, destabilizzando i gradienti. A differenza della BatchNorm, la statistica è calcolata sul singolo campione e non sul batch, quindi il comportamento è identico in training e in inferenza — coerente con il requisito di guida deterministica.
- **`ReLU` (Rectified Linear Unit)**: la funzione di attivazione `max(0, x)` — restituisce l'input se positivo, zero altrimenti. Introduce la non linearità: senza, i quattro strati `Linear` collasserebbero matematicamente in un'unica trasformazione lineare, incapace di rappresentare relazioni come "frena solo se la velocità è alta E la curva è vicina". Il fatto che un neurone si attivi solo su input positivi produce pattern sparsi: la rete impara feature che non usano sempre tutti i neuroni. È inoltre semplice, veloce e ben supportata da PyTorch.
- **`continuous_head` e attivazioni di uscita**: lo strato finale proietta le 512 feature sui 3 comandi. Le attivazioni mappano l'output illimitato dello strato lineare sul range fisico di ciascun comando: `tanh` (tangente iperbolica, codominio `[-1, 1]`, simmetrica intorno a zero) per lo sterzo, `sigmoid` (codominio `[0, 1]`) per acceleratore e freno, che sono pedali non negativi. Coincidono esattamente con i range delle azioni registrate in raccolta dati.

Profondità (4 strati) e larghezza (512 neuroni) seguono i riferimenti di Fujimoto & Gu (2021) e Beeson & Montana (2022): meno capacità porterebbe a underfitting — la rete non riuscirebbe a catturare le relazioni complesse tra sensori e comandi in curva, staccata e recupero — mentre più capacità porterebbe a overfitting, cioè a memorizzare il dataset di dimostrazioni senza generalizzare a stati nuovi.

### Loss pesata (`_combined_loss`)

L'errore è un MSE per canale con pesi statici e boost dinamici:

| Canale | Peso base | Boost dinamico |
| --- | ---: | --- |
| `steer` | 1.0 | ×3 quando `\|steer_target\| > 0.10` (`STEER_CURVE_THRESHOLD`: siamo in curva) |
| `accel` | 1.0 | — |
| `brake` | 5.0 | fino a ×25 quando il pilota frena oltre `0.05` (`BRAKE_ACTIVE_THRESHOLD`) |

Il freno è pesato molto perché nelle dimostrazioni racing è un evento raro ma cruciale: sbagliare una staccata costa più che sbagliare lievemente il gas in rettilineo. Il boost sterzo in curva migliora la fedeltà della traiettoria dove conta.

### Data augmentation Bojarski-style

L'augmentation risponde al limite classico del Behavioral Cloning (covariate shift): il modello vede soprattutto stati puliti generati dal pilota, ma in inferenza si trova fuori traiettoria a causa dei propri piccoli errori. Tre meccanismi, applicati per batch durante `train_epoch()`:

1. Perturbazione laterale e angolare. Per ogni campione si campiona `delta_pos ~ N(0, 0.20)` clampato a `±0.40` (fino al 40% di `trackPos`) e `delta_angle ~ N(0, 0.04)` clampato a `±0.08` rad (~4.5°). Una maschera casuale applica la perturbazione solo al 50% del batch: perturbare tutto significherebbe non vedere mai la traiettoria ideale, quindi metà batch resta pulito per mantenere l'equilibrio tra robustezza e fedeltà.

2. Correzione geometrica coerente. La perturbazione non tocca solo `trackPos` e `angle`: anche i 19 sensori `track` vengono aggiornati. Per ogni frame si stima la semi-larghezza pista dai due raggi a ±45° (riportati in metri ×200, clampata in `[4, 10]` m), si converte `delta_pos` in uno spostamento fisico `dy` e si applica ad ogni raggio la correzione lineare `dL = -dy * sin(beta)`, dove `beta` è l'angolo assoluto del raggio (angle perturbato + angolo del sensore). I target vengono corretti di conseguenza: lo sterzo riceve `-0.25 * delta_pos - 1.5 * delta_angle` (riporta l'auto al centro e la riallinea alla mezzeria), l'acceleratore viene parzializzato in proporzione alla perturbazione combinata (insegna a rilasciare il gas quando si è fuori traiettoria). Poiché i 3 frame dello stack vengono perturbati con lo stesso delta, la perturbazione è coerente nel tempo.

3. Overspeed recovery. Con probabilità 50% per batch, i campioni "critici" — `speedX > 90` km/h e (sterzo target `> 0.10` oppure sensore frontale `< 0.60`) — ricevono un incremento artificiale di `speedX` del 10–30% su tutti e 3 i frame, con riduzione proporzionale dell'acceleratore target (`×(1 - 0.7·f)`) e incremento del freno target (`+0.8·f`). Questo insegna preventivamente alla policy a rallentare e frenare quando entra in curva troppo veloce, una situazione che il pilota esperto per definizione non produce mai nei dati.

La standardizzazione mean/std viene applicata dopo l'augmentation, perché le perturbazioni devono operare sulle grandezze fisiche reali (metri, radianti, km/h); solo a quel punto lo stato torna piatto a 87D per il forward.

### Loop di training (`BehaviorCloningTrainer`)

- Ottimizzatore Adam, `lr = 3e-4` di default, `weight_decay = 1e-5` contro i pesi troppo grandi.
- Split train/validation 80/20 con seed fisso 42 (riproducibile); DataLoader con shuffle, 2 worker e `pin_memory` su GPU; batch 256.
- Scheduler `CosineAnnealingLR` con `eta_min = 1e-6`: learning rate alto all'inizio, decadimento dolce a coseno fino al minimo.
- Gradient clipping a norma 1.0 contro l'esplosione del gradiente.
- Nessuna augmentation in validazione (solo normalizzazione), per misurare la loss su dati reali.
- Salvataggio del checkpoint solo quando la validation loss migliora; early stopping dopo `patience` epoche senza miglioramento (100 nel main, contro le 300 epoche massime di `train_bc.sh`).

Output: `train_set/checkpoints/bc_policy.pth` (miglior modello su validation), `train_set/checkpoints/state_norm.npz` (statistiche mean/std calcolate su tutti gli stati 29D del dataset) e un log testuale per epoca in `train_set/session_logs/`.

## Fine-Tuning TD3+BC

`td3_bc.py` implementa il cuore del progetto: l'algoritmo TD3+BC, cioè TD3 con un termine di Behavioral Cloning nella loss della policy.

### Da DDPG a TD3: caratteristiche e vantaggi

Per capire TD3 serve partire da DDPG (Deep Deterministic Policy Gradient, Lillicrap et al., 2015), di cui TD3 è l'evoluzione diretta. DDPG è un algoritmo actor-critic off-policy per azioni continue:

- l'**Actor** è una policy deterministica `pi(s)` che, dato lo stato, produce direttamente l'azione (non una distribuzione di probabilità: per i comandi continui di guida non si può "enumerare" le azioni come nel Q-learning discreto);
- il **Critic** è una rete `Q(s, a)` che stima il ritorno atteso dell'azione nello stato, addestrata a minimizzare l'errore di differenza temporale (TD error) rispetto al target di Bellman `r + gamma * Q_target(s', pi_target(s'))`;
- l'Actor viene aggiornato salendo il gradiente di `Q(s, pi(s))`: la policy si sposta verso le azioni che il Critic giudica migliori;
- entrambe le reti hanno una copia **target** aggiornata lentamente (Polyak averaging), che fornisce un bersaglio stabile per il bootstrap.

DDPG funziona ma è notoriamente fragile, per una ragione strutturale: l'Actor è un ottimizzatore del Critic. Qualunque errore di approssimazione della rete Q — e una rete neurale addestrata su un replay buffer ne ha sempre — viene attivamente cercato e sfruttato dalla policy. Se il Critic sovrastima il valore di una zona dello spazio d'azione, l'Actor ci si dirige; il bootstrap propaga la sovrastima ai target successivi; e il ciclo si autoalimenta fino a Q-value gonfiati e policy che "inseguono fantasmi" (in pista: comandi che il Critic giudica ottimi ma che sono fisicamente pessimi). È l'overestimation bias, aggravato dal fatto che critic e actor si aggiornano alla stessa frequenza, quindi la policy sfrutta stime che non hanno ancora avuto tempo di correggersi.

TD3 (Twin Delayed DDPG, Fujimoto et al., 2018) mantiene l'impianto di DDPG e aggiunge tre contromisure, ognuna mirata a un pezzo del problema:

1. **Twin Critic (Clipped Double Q-learning)**. Due reti Q indipendenti, `Q1` e `Q2`, inizializzate diversamente e addestrate sugli stessi target. Poiché le due reti sbagliano in modo diverso, il target di Bellman usa il minimo delle due stime: `r + gamma * min(Q1_target, Q2_target)`. Una sovrastima per essere dannosa deve ora presentarsi in entrambe le reti contemporaneamente, evento molto più raro: il target diventa sistematicamente conservativo. Vantaggio su DDPG: elimina la spirale di Q-value gonfiati al costo di un secondo Critic (accettabile: il Critic è la rete piccola del sistema).

2. **Delayed Policy Update**. L'Actor viene aggiornato una volta ogni `policy_freq = 2` aggiornamenti del Critic (e le reti target si muovono solo insieme all'Actor). L'idea: prima di permettere alla policy di sfruttare le stime di valore, si lascia al Critic il tempo di correggerle più volte. Vantaggio su DDPG: riduce la varianza degli update della policy e spezza il circolo vizioso "policy che insegue un Critic ancora sbagliato → target peggiori → Critic ancora più sbagliato".

3. **Target Policy Smoothing**. L'azione target usata nel bootstrap non è `pi_target(s')` pura: le viene aggiunto rumore gaussiano clippato (`N(0, 0.2)` saturato a `[-0.5, 0.5]`, poi azione clippata a `[-1, 1]`). In pratica il Critic viene addestrato a dare valori simili ad azioni simili. Una policy deterministica, altrimenti, può fare overfitting sui picchi stretti della Q-function — punte di valore alte e strettissime che sono artefatti di approssimazione, non azioni davvero migliori. Vantaggio su DDPG: la value function diventa più liscia, e la policy converge verso azioni robuste anziché verso spilli di stima.

Iterative Motors usa esattamente questi tre meccanismi con i valori del paper (`policy_freq = 2`, rumore di smoothing 0.2/0.5, `tau = 0.005`, `gamma = 0.99`), e vi aggiunge un warm-up di 15000 step prima del primo update dell'Actor — un'estensione nello stesso spirito del delayed update: all'avvio il Critic non è semplicemente "in ritardo", è del tutto non informativo, quindi non ha senso ottimizzarci contro la policy.

Per un dominio racing la stabilità di TD3 non è un lusso: gli episodi durano centinaia di step, un singolo comando sbagliato a 250 km/h termina l'episodio, e il training gira per ore senza supervisione. Con DDPG puro un collasso della policy a metà run butterebbe via la sessione.

### Da TD3 a TD3+BC

TD3+BC (Fujimoto & Gu, 2021) aggiunge alla loss dell'Actor un termine di imitazione (MSE rispetto all'azione dell'esperto) bilanciato dal coefficiente adattivo `lambda`. Il razionale per questo progetto: il solo RL da zero sarebbe costoso e instabile in un simulatore racing — l'agente passerebbe migliaia di episodi a sbattere prima di completare un giro — mentre la BC fornisce un comportamento iniziale umano che il TD3 raffina. I dettagli della loss ibrida sono nella sezione sull'aggiornamento, più sotto.

Costanti di modulo: `LAP_SUCCESS_BONUS = 50.0`, `INCOMPLETE_LAP_PENALTY = 25.0`, `TRACK_LENGTH_M = 3608.0`, `EVAL_DISTANCE_SANITY_LIMIT = 3800.0`.

All'avvio `set_seed()` (default 42, configurabile con `--seed`) fissa i generatori casuali di Python, NumPy e PyTorch (CPU e CUDA), imposta cuDNN in modalità deterministica (`deterministic=True`, `benchmark=False`) e definisce `PYTHONHASHSEED`: a parità di seed e dataset gli esperimenti sono riproducibili.

### Actor

Stessa architettura della rete BC descritta sopra — 87D → 4 blocchi `Linear(512) → LayerNorm → ReLU` → `continuous_head` 3D, ≈ 838 700 parametri — con un'unica differenza: `tanh` su tutti e tre i canali di uscita invece di tanh/sigmoid, perché TD3 lavora in uno spazio d'azione normalizzato e simmetrico `[-1, 1]`. L'identità strutturale rende il warm-start diretto: backbone e testa si trasferiscono senza adattatori, e la policy parte da una guida plausibile invece che da azioni casuali.

- `forward(state)` restituisce l'azione deterministica `tanh(mean)`.
- `sample(state, evaluate)` aggiunge in training rumore gaussiano `N(0, 0.1)` clippato a `±0.2`, poi clippa l'azione a `[-1, 1]`; con `evaluate=True` restituisce l'azione pura (usata in eval e submission).
- `load_bc_weights()` applica la compensazione ×0.5 descritta sopra; al fresh-start (episodio 0) l'Actor e il suo target partono da `bc_policy.pth`, in resume dal checkpoint TD3 completo.
- `load_actor_weights()` carica pesi filtrando solo i parametri con nome e shape compatibili: serve a recuperare file di soli pesi (es. i record `.pth`) dentro l'Actor corrente.

### Critic

Twin Q-Network: due reti indipendenti e identiche, `Q1(s, a)` e `Q2(s, a)`, che stimano il valore atteso (ritorno scontato) dell'azione `a` nello stato `s`. Ciascuna riceve la concatenazione dello stato 87D e dell'azione 3D (90 input) e produce un singolo valore scalare:

| # | Strato | Dimensioni | Parametri |
| --- | --- | --- | ---: |
| 1 | `Linear` → `ReLU` | 90 → 512 | 46 592 |
| 2 | `Linear` → `ReLU` | 512 → 512 | 262 656 |
| 3 | `Linear` | 512 → 1 | 513 |
| | **Totale per rete** | | **≈ 309 800 (×2 ≈ 619 500)** |

Le differenze strutturali rispetto all'Actor sono deliberate. Il Critic è più corto (2 strati nascosti invece di 4) e non usa LayerNorm: valuta coppie stato-azione ed è addestrato continuamente sul replay ad ogni step di simulazione, quindi una rete più semplice riduce costo computazionale e superfici di instabilità. L'uscita è lineare pura, senza attivazione finale: un valore Q può essere qualsiasi numero reale, quindi non va schiacciato in un range fisso.

I due Critic gemelli sono invece essenziali: nelle azioni continue una sovrastima del valore (overestimation bias) spinge l'Actor verso comandi apparentemente ottimi ma fisicamente pessimi; le due reti, inizializzate diversamente, sbagliano in modo diverso, e prendere `min(Q1, Q2)` nel target di Bellman rende la stima conservativa.

### Replay Buffer a tre vie

`ReplayBuffer` è una deque FIFO che, accanto alle tuple `(stato, azione, reward, stato_successivo, done_mask)`, mantiene una deque parallela di `expert_mask`, il flag che marca i campioni su cui calcolare la BC penalty.

| Buffer | Capacità | Origine | Uso |
| --- | ---: | --- | --- |
| Expert | 400000 | HDF5 umani (giri completi e segmenti, `max_samples=350000`) | Ancora BC permanente: capacità > dataset, quindi mai svuotato dalla FIFO |
| Online | 1000000 | Episodi generati dall'agente | Esplorazione e correzione off-distribution (~1400 episodi da ~700 step senza evizione) |
| Elite | 20000 | Episodi autonomi sopra soglia | Self-imitation delle traiettorie migliori |

Caricamento expert (`load_expert_data`): legge gli HDF5 (qui entrano anche i segmenti, glob `lap_*.h5`), normalizza gli stati 29D, applica lo stacking `(t-12, t-6, t)`, converte acceleratore e freno dal range sigmoid `[0,1]` al range tanh `[-1,1]` per coerenza con la testa dell'Actor, e ricalcola a posteriori la reward di ogni transizione con la stessa formula di `gym_torcs.py`. Ogni campione è inserito con `expert=1.0` e `mask=1.0`. Il caricamento avviene sia al fresh-start sia al resume, così l'ancora BC è sempre garantita.

Ogni batch da 256 transizioni punta a 25% expert, 15% elite, 60% online; se online o elite non hanno abbastanza dati, la quota mancante viene compensata con campioni expert (il resto del batch è sempre `batch − online − elite` dall'expert). Questo rende il training possibile fin dal primo step — di fatto il Critic si pre-allena sui dati umani, come la fase offline del TD3+BC — e mantiene sempre una quota di dimostrazione umana nella loss.

Le percentuali 25/15/60 sono empiriche ma hanno una logica precisa: l'expert mantiene l'ancora umana, l'online insegna a gestire gli stati realmente visitati dall'agente, l'elite preserva le scoperte autonome promettenti. Troppo expert e l'Actor resta incollato al pilota; troppo online e il Critic insegue rumore esplorativo; senza elite, le buone traiettorie verrebbero diluite nella FIFO.

Iniezione elite: a fine episodio, se `max_dist ≥ elite_threshold` (500 m iniziali, poi `max(500, best_distance * 0.7)`, soglia monotonicamente crescente), tutte le transizioni dell'episodio entrano nell'elite buffer con `expert=1.0` — cioè diventano anche bersagli della BC penalty (self-imitation) — tranne gli ultimi 50 step prima di un crash, marcati `expert=0.0` per non imitare proprio le azioni che hanno causato l'incidente (causal confusion).

Done masking: `mask = 0.0` per fallimenti terminali o episodi incompleti (il valore futuro non va propagato oltre un crash), `mask = 1.0` per transizioni non terminali e per i giri completati validamente; i dati expert hanno sempre `mask = 1.0`, perché una transizione di un giro umano valido non deve essere confusa con un crash nel target di Bellman.

### Reward, bonus e tracking della distanza

Alla reward online di `gym_torcs.py` (descritta sopra), `td3_bc.py` aggiunge:

- `+50` (`LAP_SUCCESS_BONUS`) quando il giro è completato validamente;
- `-25` (`INCOMPLETE_LAP_PENALTY`) se l'episodio termina in modo incompleto (crash, stallo, spin, timeout);
- record di distanza basati su `distFromStart` tramite `_track_progress_from_start()`, che gestisce il wrap al traguardo usando `TRACK_LENGTH_M = 3608.0` e misura il progresso a partire dal punto di spawn dell'episodio. Non si usa `distRaced` perché misura la distanza realmente percorsa anche quando l'auto sbanda o allunga la traiettoria: per il record interessa il progresso lungo il tracciato.

Prima di aggiornare il Critic, le reward campionate vengono moltiplicate per `reward_scale = 0.02`. La scala della reward determina la magnitudo dei Q-value: se i Q crescono troppo, la parte RL della loss dell'Actor domina la BC penalty o produce gradienti instabili. Il fattore 0.02 mantiene le stime in un range sano.

Una guardia di plausibilità (`_is_plausible_eval_dist`, limite 3800 m) filtra distanze di valutazione fisicamente impossibili per un eval monogiro, evitando che anomalie nei log inquinino record e refinement.

### Aggiornamento TD3+BC (`TD3BCAgent.update`)

Iperparametri: `gamma = 0.99`, `tau = 0.005`, `policy_freq = 2`, Adam `3e-4` per entrambe le reti, batch 256, frequenza update 1:1 (un update per ogni step di simulazione, appena l'expert buffer supera la batch size).

Per il Critic, ad ogni update:

1. l'Actor target calcola l'azione sul prossimo stato;
2. viene aggiunto rumore gaussiano `N(0, 0.2)` clippato a `[-0.5, 0.5]` (target policy smoothing: regolarizza le stime contro i picchi stretti della Q-function);
3. l'azione target viene clippata a `[-1, 1]`;
4. si calcola il target di Bellman e si minimizza l'MSE di entrambe le stime:

```text
target_q    = reward_scaled + mask * gamma * min(Q1_target, Q2_target)
critic_loss = MSE(Q1, target_q) + MSE(Q2, target_q)
```

In modalità refinement (vedi oltre) l'optimizer del Critic non viene applicato: la loss resta calcolata a fini diagnostici.

Per l'Actor, l'update avviene solo se `global_step ≥ 15000` (warm-up: non ha senso ottimizzare la policy contro un Critic non ancora informativo), ogni `policy_freq = 2` update del Critic, e mai mentre l'Actor è congelato post-rollback:

- componente RL: `actor_loss_td3 = -mean(Q1(s, pi(s)))`;
- BC penalty con masking rigoroso: azione predetta e azione target vengono riportate nello spazio pedali `[0,1]` e confrontate via MSE solo sui campioni con `expert_mask > 0.5`. Sugli stati sporchi generati online l'agente deve poter inventare recuperi, non copiare azioni umane che lì non esistono. Sterzo e freno pesano doppio: `bc_penalty = 2·steer_loss + accel_loss + 2·brake_loss`;
- penalità di mutua esclusione `0.1 * mean(accel * brake)` per disincentivare i pedali premuti insieme;
- coefficiente dinamico del paper TD3+BC: `lambda = 2.5 / mean(|Q1(s, pi(s))|)` (clampato a min `1e-5`), che mantiene la componente RL sulla stessa scala del termine di imitazione qualunque sia la magnitudo corrente dei Q;
- loss totale, con `bc_weight = 1.0` in training normale e `0.3` in refinement:

```text
actor_loss = lambda * (-mean(Q1(s, pi(s)))) + bc_weight * bc_penalty
```

Entrambe le reti usano gradient clipping a norma 1.0. Dopo ogni update dell'Actor, le reti target vengono aggiornate con Polyak averaging:

```text
target = tau * online + (1 - tau) * target        (tau = 0.005)
```

Le reti target rendono lento il bersaglio del Critic: senza, il valore da inseguire cambierebbe con gli stessi pesi in aggiornamento, creando un inseguimento instabile. `tau = 0.005` è abbastanza reattivo da seguire il training e abbastanza lento da filtrare le oscillazioni.

### Ciclo episodico

Ogni episodio: reset con `relaunch=True`, inizializzazione del frame stack (13 copie dello stato iniziale), marcia in prima con `steps_since_shift = 999` (primo cambio consentito subito). Ad ogni step: selezione azione con rumore, conversione pedali e mutua esclusione, calcolo marcia con `compute_gear`, step ambiente, update dell'agente. Le transizioni vengono accumulate e inserite nel buffer online solo a fine episodio (così il `mask` finale è noto); un episodio interrotto da STOP viene scartato.

Record esplorativi salvati al volo durante l'episodio: `td3_expl_best_lap.pth` al miglior tempo su giro completato, `td3_expl_best_dist.pth` alla miglior distanza (sopra 500 m). A fine episodio vengono salvati il checkpoint completo (`td3_checkpoint.pth` + buffer) e l'ultimo Actor (`td3_policy.pth`), e viene scritta una riga di log con reward, step, distanza, loss medie e stato di Critic (`ON`/`OFF`) e Actor (`ON`/`WARM`/`FREEZE`).

SIGINT (Ctrl+C) e SIGTERM non uccidono il processo: impostano `stop_requested`, il loop completa l'episodio, salva un checkpoint completo coerente e poi esce.

### Valutazione deterministica periodica

Ogni 5 episodi, superato il warm-up di 15000 step, l'agente viene valutato con `evaluate=True` (zero rumore, `actor.eval()`), marcia algoritmica e stop anticipato al primo giro valido completato. La distanza è misurata con la stessa logica wrap-aware e filtrata per plausibilità (≤ 3800 m). I record deterministici aggiornano tre checkpoint distinti:

- `td3_det_best_dist_run.pth`: miglior distanza del run corrente;
- `td3_det_best_dist.pth` + sidecar testuale `td3_det_best_dist.txt`: miglior distanza assoluta (con tolleranza `BEST_DIST_EPS = 1.0` m contro le oscillazioni). Il sidecar rende il record persistente anche dopo `--clean` e recuperabile se il checkpoint binario si corrompe;
- `td3_det_best_lap.pth` + sidecar `td3_det_best_lap.txt`: miglior giro valido deterministico, il candidato per la submission.

La valutazione deterministica è separata dal training perché il rumore esplorativo è utile per imparare ma non rappresenta la policy da portare in gara.

## Auto-Refinement E Rollback

Il progetto include una macchina a stati per uscire dai plateau, ispirata a Beeson & Montana (2022): ridurre progressivamente il vincolo BC quando la policy è già buona, con protezioni operative contro collasso e regressioni. I parametri:

| Costante | Valore | Significato |
| --- | ---: | --- |
| `REFINE_WINDOW` | 8 | Valutazioni nella finestra mobile |
| `REFINE_IMPROVE_FRAC` | 1.02 | Miglioramento minimo (+2%) della media per non contare verso il plateau |
| `REFINE_PLATEAU_EVALS` | 4 | Valutazioni consecutive senza miglioramento per decretare il plateau |
| `REFINE_MIN_EP` | 200 | Episodio minimo per attivare l'auto-refinement |
| `REFINE_BC_WEIGHT` | 0.3 | Peso BC durante il refinement |
| `REFINE_BREAKOUT_FRAC` | 1.10 | Superamento del plateau (+10%) per contare un "breakout" |
| `REFINE_GOOD_EVALS_TO_CONSOLIDATE` | 3 | Breakout consecutivi per consolidare |
| `REFINE_NEAR_BEST_MARGIN` | 5.0 m | Vicinanza al record storico per consolidare subito |
| `REFINE_COLLAPSE_FRAC` | 0.6 | Soglia di crollo (60% del riferimento) |
| `REFINE_MAX_ATTEMPTS` | 3 | Tentativi massimi per plateau |
| `REFINE_NEW_PLATEAU_FRAC` | 1.10 | Plateau più alto del +10% → reset del contatore tentativi |

Rilevamento del plateau: il segnale è la media della finestra delle ultime 8 valutazioni, non il singolo best (robusto ai colpi di fortuna). Se la media non supera la migliore media storica di almeno il 2% per 4 valutazioni consecutive, e si è oltre l'episodio 200 con Actor non congelato, il refinement si attiva. Il riferimento del plateau è la mediana della finestra — il "modo buono" di una distribuzione bimodale — non il massimo stocastico.

Durante il refinement: l'aggiornamento del Critic è disattivato (la loss resta solo diagnostica), il peso BC scende a 0.3 e l'Actor cerca di sfruttare meglio una value function fissa. Le uscite possibili:

- consolidamento: 3 breakout consecutivi sopra `plateau × 1.10`, oppure un breakout entro 5 m dal record deterministico assoluto, oppure direttamente un nuovo record assoluto. Il peso BC torna a 1.0, il Critic si riattiva e l'Actor viene temporaneamente congelato (default 30 episodi) per riallineare il Critic alla nuova policy;
- rollback: distanza sotto il 60% del riferimento per 3 valutazioni consecutive → l'Actor viene ripristinato da `td3_det_best_dist.pth`, il tentativo viene contato e il rilevamento del plateau riparte da zero;
- timeout: 8 valutazioni in refinement (≈40 episodi) senza superare il record → uscita automatica con tentativo contato.

Esauriti i 3 tentativi su uno stesso plateau, l'auto-refinement resta disarmata finché non emerge un plateau più alto di almeno il 10% (che azzera il contatore). Al resume, la finestra delle valutazioni viene ripopolata leggendo le righe `[EVAL]` dal log di training (`load_recent_evals_from_log`), così lo stato del plateau sopravvive ai riavvii.

Controlli manuali da CLI: `--refine` arma il refinement da subito (riferimento = mediana recente o record storico); `--no-auto-refine` disattiva solo l'attivazione automatica; `--rollback` forza il ripristino dell'Actor dalla migliore policy disponibile, scendendo in ordine di priorità (`det_best_lap` → `det_best_dist` → `det_best_dist_run` → `expl_best_lap` → `expl_best_dist`), reinizializza l'optimizer dell'Actor e lo congela per `--actor-freeze-episodes` episodi (default 30); `--pretrain_critic`, da usare con `--rollback` in emergenza, ri-allena il Critic offline per 50000 passi sui buffer prima di riprendere.

## Checkpoint E Robustezza

Il salvataggio è progettato per sopravvivere a interruzioni di corrente e Ctrl+C. `safe_save()` implementa un protocollo atomico:

1. scrive l'oggetto su un file temporaneo `.tmp`;
2. forza `fsync` sul file (persistenza fisica, contro i file da 0 byte post-crash);
3. ruota i backup: il `.bak` esistente diventa `.prev`, il file corrente viene copiato in `.bak` (i backup dei file in `train_set/checkpoints/` vengono organizzati nella sottocartella `backups/`);
4. sostituisce atomicamente con `os.replace()`;
5. esegue `fsync` anche sulla directory genitrice, per persistere i metadati del rename.

Lo stesso protocollo è applicato ai sidecar testuali (`safe_write_text`) e ai buffer `.npz` (`safe_save_npz`, con un trucco sull'estensione temporanea `.tmp.npz` perché numpy appende `.npz` automaticamente). `safe_read_float` legge i sidecar provando in cascata file principale e backup.

`save_checkpoint()` scrive prima i replay buffer (l'operazione I/O più onerosa) e poi il dizionario `.pth` con pesi di Actor/Critic e relative reti target, stati degli optimizer, episodio, `global_step` e metriche di record. L'ordine è intenzionale: se il processo muore a metà, l'assenza del `.pth` aggiornato segnala al resume che i nuovi buffer non gli appartengono.

`load_checkpoint()` gestisce il resume in modo difensivo:

- scansiona i candidati (file principale, `.bak`, `.prev`, anche in `backups/`) fino a trovarne uno leggibile e completo;
- valida le distanze memorizzate con il filtro di plausibilità (≤ 3800 m) e in caso di anomalie le recupera dai sidecar testuali;
- se il file contiene solo pesi dell'Actor (es. un record `.pth`), esegue un warm-start dei soli parametri di guida azzerando optimizer e contatori;
- carica i buffer rifiutando file con timestamp di modifica successivo a quello del checkpoint caricato (tolleranza 1 ms): un buffer "più nuovo" appartiene a un salvataggio interrotto prima della scrittura del `.pth`, e usarlo creerebbe uno stato incoerente. In quel caso cerca un backup del buffer allineato; solo come ultima risorsa accetta il buffer più nuovo;
- se nessun checkpoint è valido, recupera almeno il numero di episodio dal log di training (regex su `Episode/Episodio/Ep N`) e i record dai sidecar.

Questa complessità è giustificata dal costo dei run lunghi: perdere un checkpoint o ripartire con pesi e replay non allineati può buttare via ore di simulazione.

I checkpoint principali sono:

| File | Significato |
| --- | --- |
| `bc_policy.pth` | Policy supervisionata iniziale (miglior validation loss) |
| `state_norm.npz` | Statistiche mean/std delle feature 29D |
| `td3_checkpoint.pth` | Stato completo di training TD3+BC (pesi, target, optimizer, contatori, record) |
| `td3_policy.pth` | Ultimo Actor TD3 salvato (fine di ogni episodio) |
| `td3_expl_best_lap.pth` | Miglior giro ottenuto in fase esplorativa (con rumore) |
| `td3_expl_best_dist.pth` | Miglior distanza in fase esplorativa |
| `td3_det_best_dist_run.pth` | Miglior distanza deterministica del run corrente |
| `td3_det_best_dist.pth` (+ `.txt`) | Miglior distanza deterministica assoluta, sopravvive a `--clean` |
| `td3_det_best_lap.pth` (+ `.txt`) | Miglior giro deterministico valido, candidato submission |

`train_rl.sh --clean` cancella checkpoint, buffer e record di run/esplorazione (con tutti i loro backup), ma preserva deliberatamente `td3_det_best_dist.pth` e `td3_det_best_lap.pth`: i record assoluti non vanno persi ripartendo da zero.

## Test Deterministico

`test_agent.py` usa `PolicyActor`, una rete identica all'Actor con due percorsi di inferenza: `forward()` applica le attivazioni BC (tanh sullo sterzo, sigmoid sui pedali) e viene usato con pesi BC; `sample(evaluate=True)` applica tanh su tutti i canali e viene usato con pesi TD3+BC. La modalità determina anche la denormalizzazione: `denormalize_action_bc` clippa soltanto, `denormalize_action_rl` mappa i pedali da `[-1,1]` a `[0,1]` con `(x+1)/2`. La mutua esclusione `accel *= (1-brake)` viene applicata prima della denormalizzazione per la BC (valori già in `[0,1]`) e dopo per la RL.

In assenza di `--weights`, `load_best_weights()` esamina i checkpoint in ordine di priorità decrescente:

1. `td3_det_best_lap.pth` — miglior giro valido deterministico (mostra anche il tempo letto dal sidecar);
2. `td3_det_best_dist.pth` — miglior distanza assoluta;
3. `td3_det_best_dist_run.pth` — miglior distanza del run corrente;
4. `td3_expl_best_lap.pth` — miglior giro esplorativo;
5. `td3_expl_best_dist.pth` — miglior distanza esplorativa;
6. `td3_policy.pth` — ultima policy salvata;
7. `bc_policy.pth` — fallback supervisionato.

Il caricamento filtra le chiavi dello state dict per nome e shape (ignora eventuali pesi del Critic salvati nello stesso file). Il tipo di policy è dedotto dal nome del file (`td3` → RL, `bc` → BC); se non è deducibile, va passato esplicitamente `--kind rl|bc`, perché applicare la denormalizzazione sbagliata renderebbe la guida insensata.

Durante il test:

- il modello è in `eval()` e `cudnn` è in modalità deterministica (`deterministic=True`, `benchmark=False`) per la riproducibilità;
- nessun rumore viene aggiunto; la marcia resta algoritmica via `compute_gear` (con l'acceleratore effettivamente applicato);
- ogni tentativo rilancia TORCS con `relaunch=True` per ripulire lo stato del motore fisico;
- l'ambiente è `early_termination=False` e i criteri di invalidazione sono dello script: fuori pista (`|trackPos| > 1.25`), testacoda (`cos(angle) < 0`) e stallo (velocità in avanti `< 5` km/h per ≥ 50 step consecutivi dopo i primi 500 step);
- il completamento del giro è rilevato dalla variazione di `lastLapTime`;
- per ogni tentativo viene scritta la telemetria completa in `telemetry/telemetry_attempt_N.csv` con colonne `step, dist, speed, trackPos, angle, steer, accel, brake, gear`.

Il loop continua finché non vengono completati `--laps` giri validi (default 3, timeout 15000 step/giro), poi stampa best e media. Questo è il percorso da usare per misurare le prestazioni reali candidate alla competizione: la telemetria per tentativo permette di capire dove la policy perde tempo o stabilità — velocità, freno, sterzo, marcia e `trackPos` mostrano se il limite è una staccata, una curva o una scelta del cambio.

## Script Operativi

- `train_bc.sh`: verifica che esistano giri completi `lap_[0-9]*.h5` in `train_set/laps` (i segmenti sono esclusi dal BC), crea le directory e lancia `behavioral_cloning.py` con 300 epoche e batch 256.
- `train_rl.sh`: rileva la situazione di partenza — resume da `td3_checkpoint.pth`, warm-start dai pesi BC o cold-start — e lancia `td3_bc.py` (default: 1000 episodi, seed 42, 5000 step massimi, configurabili via `TD3_EPISODES`, `TD3_SEED`, `TD3_MAX_STEPS`). Gestisce `--clean` (ripartenza pulita preservando i record assoluti) e inoltra gli altri flag (`--rollback`, `--refine`, `--no-auto-refine`, `--actor-freeze-episodes`, `--pretrain_critic`) allo script Python.
- `stop_training.sh`: termina via `pkill` gli script di training, i processi Python di BC/TD3/test e l'intero stack TORCS/Xvfb. Grazie alla gestione di SIGTERM in `td3_bc.py`, il training si arresta dopo un checkpoint completo.

## Riferimenti Scientifici

I riferimenti qui sotto sono quelli effettivamente collegati alle scelte del codice:

1. Lillicrap et al., "Continuous Control with Deep Reinforcement Learning", 2015. Base concettuale DDPG per controllo continuo actor-critic. https://arxiv.org/abs/1509.02971
2. Fujimoto, van Hoof, Meger, "Addressing Function Approximation Error in Actor-Critic Methods", 2018. Paper TD3: twin critics, delayed policy update, target policy smoothing. https://arxiv.org/abs/1802.09477
3. Fujimoto, Gu, "A Minimalist Approach to Offline Reinforcement Learning", 2021. Paper TD3+BC: termine BC nella policy loss e normalizzazione dei dati. https://arxiv.org/abs/2106.06860
4. Beeson, Montana, "Improving TD3-BC: Relaxed Policy Constraint for Offline Learning and Stable Online Fine-Tuning", 2022. Ispirazione per refinement e riduzione controllata del vincolo BC. https://arxiv.org/abs/2211.11802
5. Bojarski et al., "End to End Learning for Self-Driving Cars", 2016. Riferimento per apprendimento da dimostrazioni umane e augmentation di guida. https://arxiv.org/abs/1604.07316
6. Loiacono, Cardamone, Lanzi, "Simulated Car Racing Championship: Competition Software Manual", 2013. Descrizione del software SCR, sensori e attuatori usati dal client TORCS. https://arxiv.org/abs/1304.1672
