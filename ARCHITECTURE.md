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
| `data_collection.py` | Raccolta HDF5 da pilota umano, con controller/tastiera, TCS e segmentazione delle curve |
| `behavioral_cloning.py` | Addestramento supervisionato della policy iniziale su giri completi |
| `td3_bc.py` | Fine-tuning TD3+BC, replay buffer, checkpoint, evaluation e auto-refinement |
| `test_agent.py` | Inferenza deterministica e confronto dei checkpoint migliori |
| `gearing.py` | Cambio marcia algoritmico condiviso da training RL e test |
| `gym_torcs/gym_torcs.py` | Wrapper ambiente, reward shaping, reset/relaunch TORCS |
| `gym_torcs/snakeoil3_gym.py` | Client UDP SCR a basso livello: parsing sensori e invio azioni |
| `gym_torcs/autostart.sh` | Automazione menu TORCS tramite `xte` |
| `train_bc.sh` | Script operativo per avviare il Behavioral Cloning |
| `train_rl.sh` | Script operativo per avviare o riprendere il TD3+BC |
| `stop_training.sh` | Arresto dei processi di training/test/TORCS/Xvfb |

Le directory `train_set/` e `telemetry/` sono pensate per artefatti locali. La `.gitignore` mantiene versionate solo le sottodirectory tramite `.gitkeep`, mentre dataset, checkpoint e CSV restano fuori dal versionamento.

## Ambiente TORCS E Comunicazione

Il progetto usa TORCS come simulatore fisico e il protocollo SCR tramite UDP. Il file `snakeoil3_gym.py` mantiene la connessione con il server TORCS sulla porta `3001`, invia azioni nel formato SCR e riceve stringhe di telemetria come `speedX`, `track`, `trackPos`, `rpm`, `wheelSpinVel`, `distFromStart` e tempi sul giro.

`gym_torcs.py` è il livello di astrazione usato dal resto del progetto. Quando viene istanziato `TorcsEnv`, il wrapper:

- termina eventuali processi TORCS rimasti attivi, salvo `TORCS_KILL_ALL=0`;
- avvia TORCS con `torcs -nofuel -nodamage`;
- usa Xvfb in headless se `SHOW_GUI` non vale `1`;
- avvia `autostart.sh`, che invia i tasti necessari per entrare in una sessione di pratica;
- espone `reset()`, `step()`, `get_obs()` ed `end()`.

Durante data collection, `data_collection.py` forza `SHOW_GUI=1` perché il pilota deve vedere la pista. Durante training e test, invece, il sistema può girare headless.

L'avvio headless con Xvfb serve a rendere il training più leggero e ripetibile: durante il TD3+BC non serve renderizzare la gara per un umano, quindi si evita di sprecare risorse grafiche. Il rilancio periodico/forzato di TORCS, invece, è una scelta di robustezza: nelle sessioni lunghe il simulatore può accumulare stato sporco, socket bloccati o memory leak; ripartire da un processo pulito riduce la probabilità che il training venga falsato da problemi esterni alla policy.

## Stato Sensoriale

La policy non riceve immagini. Riceve un vettore sensoriale compatto a 29 dimensioni costruito da `flatten_state()`:

Questa scelta privilegia controllo e campionamento rispetto alla percezione visiva. Le immagini richiederebbero una CNN, molti più dati, più GPU e introdurrebbero un problema di visione che non è centrale per il nostro obiettivo: in TORCS i sensori SCR forniscono già geometria pista, velocità e stato meccanico. Usare sensori numerici rende il learning più sample-efficient e permette al TD3+BC di concentrarsi sulle decisioni racing, cioè traiettorie, staccate e recuperi.

| Blocco | Dimensioni | Descrizione |
| --- | ---: | --- |
| `angle` | 1 | Angolo tra asse vettura e asse pista |
| `track` | 19 | Sensori di distanza dal bordo pista, già scalati dal wrapper con `/200` |
| `trackPos` | 1 | Posizione laterale rispetto al centro pista |
| `speedX`, `speedY`, `speedZ` | 3 | Velocità normalizzate dal wrapper rispetto a `default_speed=50` |
| `wheelSpinVel` | 4 | Velocità ruote, riscalate in `flatten_state()` con `/100` |
| `rpm` | 1 | Regime motore, riscalato in `flatten_state()` con `/10000` |

`distFromStart` viene salvata come metadato negli HDF5, ma non entra nello stato della rete. Questa scelta evita che l'agente impari un'associazione rigida tra posizione assoluta e comando. In gara deve guidare dai sensori, non "ricordare" una sequenza di azioni legata al metro del tracciato.

La rete usa frame stacking: lo stato finale ha 87 dimensioni, ottenute concatenando tre frame da 29 dimensioni. I frame sono `t-12`, `t-6` e `t`, con `k=6`, quindi circa 0.24 secondi di storia a 50 Hz. Questo aggiunge informazione temporale senza passare a una rete ricorrente: l'Actor vede non solo dove si trova l'auto, ma anche come ci sta arrivando.

Il frame stacking è stato preferito a LSTM/GRU perché mantiene l'Actor semplice e deterministico. Una rete ricorrente potrebbe modellare dinamiche più lunghe, ma complicherebbe il replay buffer, il resume e la stabilità del Critic. Tre frame distanziati sono un compromesso pragmatico: abbastanza storia per inferire deriva, accelerazione e tendenza dello sterzo, senza aumentare troppo la complessità.

Dopo il Behavioral Cloning viene salvato `train_set/checkpoints/state_norm.npz`, contenente media e deviazione standard delle feature 29D calcolate sul dataset esperto. TD3+BC e test caricano lo stesso file e applicano:

```text
s_norm = (s - mean) / (std + 1e-3)
```

La normalizzazione delle feature è una scelta importante in TD3+BC: riduce scale numeriche sbilanciate e stabilizza sia Actor sia Critic.

## Azioni E Cambio Marcia

TORCS riceve un'azione 4D:

```text
[steer, accel, brake, gear]
```

La rete neurale predice solo i primi tre comandi. La marcia è gestita da `gearing.py`, che usa regole deterministiche basate su velocità, RPM, acceleratore effettivamente applicato e cooldown. Questo riduce lo spazio d'azione e impedisce alla rete di sprecare capacità su una decisione discreta e facilmente descrivibile con soglie.

Nel Behavioral Cloning:

- `steer` usa `tanh`, quindi sta in `[-1, 1]`;
- `accel` e `brake` usano `sigmoid`, quindi stanno in `[0, 1]`.

Nel TD3+BC:

- l'Actor usa `tanh` su tutti e tre i canali;
- `accel` e `brake` vengono mappati da `[-1, 1]` a `[0, 1]` prima dell'invio a TORCS;
- viene applicata mutua esclusione moltiplicativa: `accel = accel * (1 - brake)`.

Le attivazioni sono scelte in base ai vincoli fisici dei comandi: lo sterzo è naturalmente simmetrico intorno a zero, mentre gas e freno sono pedali non negativi. In TD3 usiamo `tanh` su tutti i canali perché l'algoritmo lavora in uno spazio continuo normalizzato e simmetrico; la conversione dei pedali avviene solo al momento dell'interazione con TORCS. La mutua esclusione evita una condizione poco realistica e dannosa, cioè accelerare e frenare insieme, ma lo fa con una formula continua che non crea salti bruschi nella policy.

Quando l'Actor TD3 parte dai pesi BC, `load_bc_weights()` moltiplica per `0.5` i pesi e bias dei canali acceleratore/freno. Il motivo è matematico: `tanh(z / 2)` equivale a `2 * sigmoid(z) - 1`. In questo modo i logits appresi dalla BC vengono trasferiti nel nuovo range TD3 senza rompere il comportamento iniziale.

## Raccolta Dati

`data_collection.py` genera il dataset esperto. Il pilota può usare:

- controller PS5 DualSense: stick sinistro per sterzo, R2 per acceleratore, L2 per freno, quadrato/X per cambio;
- tastiera: WASD per guida e frecce su/giù per cambio.

La raccolta salva:

- `lap_001.h5`, `lap_002.h5`, ... per giri completi validi;
- `lap_seg_*.h5` per segmenti mirati, soprattutto curve difficili;
- log testuali in `train_set/session_logs/giri/`.

Il formato HDF5 è stato scelto perché salva array numerici compressi, attributi e metadati nello stesso file, restando facile da leggere con `h5py`. Per la pipeline è più adatto di CSV o JSON: gli stati e le azioni sono sequenze dense, e HDF5 permette di caricarle velocemente senza parsing testuale fragile.

Ogni HDF5 contiene:

- `states`: sequenza di stati 29D;
- `actions`: sequenza di azioni registrate `[steer, accel, brake, gear]`;
- `dist_from_start`: metadato per analisi e segmentazione;
- attributi come `lap_time`, `num_steps`, `timestamp`.

Il TCS opzionale riduce l'acceleratore se lo spin medio posteriore supera quello anteriore oltre una soglia. Non è parte della policy finale: serve a rendere più pulite le dimostrazioni umane e a ridurre giri scartati durante la raccolta.

La distinzione tra giri completi e segmenti è importante. I giri completi rappresentano la distribuzione globale di guida e sono adatti al Behavioral Cloning. I segmenti, invece, aumentano la densità di dati nelle curve problematiche: usarli direttamente nel BC sbilancerebbe la media delle azioni, ma usarli come expert data nel TD3+BC aiuta il Critic e l'Actor a rivedere proprio gli stati più difficili.

## Behavioral Cloning

Il Behavioral Cloning è implementato in `behavioral_cloning.py`. La rete `PolicyNetwork` ha:

- input 87D;
- quattro blocchi `Linear -> LayerNorm -> ReLU`, ciascuno con 512 neuroni;
- testa lineare `continuous_head` a 3 canali;
- attivazioni finali specifiche per BC: `tanh` sullo sterzo, `sigmoid` su acceleratore e freno.

La rete è una MLP perché l'input è già vettoriale e strutturato: non c'è un'immagine da cui estrarre feature spaziali. Quattro layer da 512 neuroni danno capacità sufficiente per rappresentare curve, staccate e recuperi senza introdurre una rete eccessivamente grande. `LayerNorm` è usata per stabilizzare le attivazioni interne, soprattutto perché le feature derivano da sensori con scale e distribuzioni molto diverse. `ReLU` mantiene il modello semplice, veloce e ben supportato da PyTorch.

Il training usa solo i giri completi `lap_[0-9]*.h5`, escludendo i segmenti `lap_seg_*.h5`. Questa è una scelta pratica: la BC minimizza l'errore medio sull'azione esperta, quindi un dataset pieno di soli segmenti di curva rischierebbe di sbilanciare la policy. I segmenti restano utili nel TD3+BC, dove entrano nel buffer esperto.

La loss supervisionata è una MSE pesata:

- sterzo: peso base `1.0`, boost `3x` quando `|steer| > 0.10`;
- acceleratore: peso `1.0`;
- freno: peso base `5.0`, boost fino a `25x` quando il pilota frena oltre `0.05`.

Il freno viene pesato molto perché nelle dimostrazioni racing è un evento meno frequente ma cruciale: sbagliare una staccata costa più che sbagliare lievemente il gas in rettilineo.

Il training include augmentation ispirata alla guida end-to-end:

- perturbazione laterale di `trackPos`;
- perturbazione angolare;
- correzione coerente dei sensori `track`;
- correzione del target di sterzo;
- riduzione del target acceleratore in stati perturbati;
- simulazione di overspeed in curva, aumentando artificialmente `speedX` e rinforzando freno/rilascio.

L'augmentation risponde al limite classico del Behavioral Cloning: il modello vede soprattutto stati puliti generati dal pilota umano, ma in inferenza può trovarsi fuori traiettoria a causa dei propri piccoli errori. Perturbare posizione, angolo e velocità insegna alla rete una prima strategia di recupero. Non viene applicata a tutti i campioni perché serve conservare anche la traiettoria ideale: metà batch perturbato e metà pulito mantengono equilibrio tra robustezza e fedeltà.

L'output principale è:

- `train_set/checkpoints/bc_policy.pth`;
- `train_set/checkpoints/state_norm.npz`;
- log in `train_set/session_logs/`.

## Fine-Tuning TD3+BC

`td3_bc.py` implementa il cuore del progetto. TD3 è un algoritmo actor-critic per azioni continue: l'Actor sceglie l'azione, il Critic stima il valore dell'azione nello stato corrente. TD3 migliora DDPG usando tre accorgimenti: due Critic indipendenti, aggiornamento ritardato dell'Actor e smoothing dell'azione target.

Iterative Motors usa TD3+BC perché il solo RL da zero sarebbe costoso e instabile in un simulatore racing. La BC fornisce un comportamento iniziale umano; il TD3 cerca traiettorie migliori e recuperi più robusti.

### Actor

L'Actor TD3 ha la stessa forma della rete BC:

- input 87D;
- quattro blocchi fully connected da 512 con LayerNorm e ReLU;
- output 3D con `tanh` su tutti i canali.

Al fresh-start, l'Actor carica `bc_policy.pth`. In resume, invece, i pesi vengono ripresi dal checkpoint TD3 completo.

Mantenere la stessa architettura dell'Actor BC rende il warm-start diretto e affidabile: possiamo trasferire backbone e testa continua senza adattatori intermedi. Questo riduce drasticamente il tempo iniziale del RL, perché la policy parte già da una guida plausibile invece che da azioni casuali.

### Critic

Il Critic è doppio:

- `Q1(s, a)`;
- `Q2(s, a)`.

Ogni rete riceve la concatenazione dello stato 87D e dell'azione 3D, quindi 90 input, passa per due layer hidden da 512 con ReLU e produce un valore scalare. Il target TD usa `min(Q1, Q2)` per ridurre l'overestimation bias.

Il Critic non usa LayerNorm perché valuta coppie stato-azione e viene addestrato continuamente sul replay: una rete più semplice riduce costo computazionale e superfici di instabilità. I due Critic sono invece essenziali: nelle azioni continue una sovrastima del valore può spingere l'Actor verso comandi apparentemente ottimi ma fisicamente pessimi. Prendere il minimo tra Q1 e Q2 rende il target più conservativo.

### Replay Buffer

Il training usa tre sorgenti di esperienza:

| Buffer | Capacità | Origine | Uso |
| --- | ---: | --- | --- |
| Expert | 400000 | HDF5 umani, inclusi segmenti | Ancora BC permanente |
| Online | 1000000 | Episodi generati dall'agente | Esplorazione e correzione off-distribution |
| Elite | 20000 | Episodi autonomi sopra soglia | Self-imitation delle traiettorie migliori |

Ogni batch da 256 transizioni cerca di rispettare:

- 25% expert;
- 15% elite;
- 60% online.

Se online o elite non hanno abbastanza dati, il batch viene compensato con campioni expert. Questo rende il training possibile fin dall'inizio e mantiene sempre una quota di dimostrazione umana nella loss.

Le percentuali 25/15/60 sono empiriche ma hanno una logica precisa: l'expert buffer mantiene l'ancora umana, l'online buffer insegna a gestire stati realmente visitati dall'agente e l'elite buffer preserva traiettorie autonome promettenti. Se l'expert fosse troppo alto, l'Actor resterebbe vicino al pilota umano; se l'online fosse troppo alto, il Critic rischierebbe di inseguire troppo rumore esplorativo; se l'elite fosse assente, le buone scoperte dell'agente verrebbero diluite nella FIFO.

Il done masking è coerente con il significato della transizione:

- `mask = 0.0` per fallimenti terminali o episodi incompleti;
- `mask = 1.0` per transizioni non terminali e giri completati validamente;
- i dati expert caricati dagli HDF5 vengono trattati con `mask = 1.0`, perché una transizione di un giro umano valido non deve essere confusa con un crash nel target di Bellman.

### Reward

La reward base è calcolata in `gym_torcs.py`:

```text
progress = (speedX / 50) * cos(angle)
pos_penalty = -2.0 * max(0, |trackPos| - 1.0)^2
smooth_penalty = -0.05 * |steer_t - steer_t-1|
reward = 1.5 * progress + pos_penalty + smooth_penalty
```

La reward premia avanzamento lungo l'asse della pista, non semplicemente movimento. `cos(angle)` penalizza guida di traverso o testacoda. La penalità su `trackPos` è nulla finché l'auto resta dentro i bordi nominali e cresce quando supera `|trackPos| > 1.0`.

La reward è volutamente minimale. Non imponiamo direttamente una traiettoria, una velocità target o un punto di frenata, perché altrimenti limiteremmo la possibilità di superare il comportamento umano. Il segnale dice cosa conta in gara: avanzare nella direzione giusta, restare in pista e non oscillare inutilmente. Il resto viene lasciato alla policy.

L'ambiente termina l'episodio se:

- `|trackPos| > 1.25`, considerato fuori pista;
- dopo i primi 500 step il progresso in avanti è troppo basso, quindi l'auto è in stallo;
- `cos(angle) < 0`, quindi la vettura è girata rispetto alla pista;
- viene completato un giro valido.

`td3_bc.py` aggiunge:

- `+50` quando il giro è completato validamente;
- `-25` se l'episodio termina in modo incompleto;
- record di distanza basati su `distFromStart`, corretti per il wrap al traguardo su `TRACK_LENGTH_M = 3608.0`.

Prima di aggiornare il Critic, le reward campionate vengono moltiplicate per `0.02`. Questo mantiene le stime Q in un range più stabile.

La scala della reward influenza direttamente la magnitudo dei Q-value. Se i Q diventano troppo grandi, la parte RL della loss dell'Actor può dominare la BC penalty o produrre gradienti instabili. Il fattore `0.02` rende più equilibrato il confronto tra valore stimato e imitazione esperta.

### Aggiornamento TD3+BC

Ad ogni step di simulazione, se il buffer expert contiene abbastanza dati, viene eseguito un update.

Per il Critic:

1. l'Actor target calcola l'azione sul prossimo stato;
2. viene aggiunto rumore gaussiano `0.2`, clippato a `[-0.5, 0.5]`;
3. l'azione target viene clippata in `[-1, 1]`;
4. si calcola il target di Bellman:

```text
target_q = reward_scaled + mask * gamma * min(Q1_target, Q2_target)
```

con `gamma = 0.99`.

Per l'Actor:

- gli update partono solo dopo `15000` step globali, per dare tempo al Critic di stabilizzarsi;
- l'Actor viene aggiornato ogni `policy_freq = 2` update del Critic;
- la componente RL massimizza `Q1(s, pi(s))`;
- la componente BC confronta `pi(s)` con l'azione esperta solo sui campioni con `expert_mask > 0.5`;
- sterzo e freno hanno peso doppio nella BC penalty;
- si aggiunge una piccola penalità contro acceleratore e freno premuti insieme;
- la scala RL usa `lambda = 2.5 / mean(|Q|)`.

La loss dell'Actor è:

```text
actor_loss = lambda * (-mean(Q1(s, pi(s)))) + bc_weight * bc_penalty
```

Nel training normale `bc_weight = 1.0`. In refinement può scendere a `0.3`.

Il warm-up dell'Actor evita di ottimizzare la policy contro un Critic ancora non informativo. `policy_freq = 2` segue l'idea di TD3: aggiornare meno spesso l'Actor permette al Critic di correggere più volte le proprie stime prima che la policy le sfrutti. La BC penalty mascherata solo sui campioni expert impedisce una forzatura sbagliata: sugli stati sporchi generati online l'agente deve poter inventare recuperi, non copiare azioni umane che in quel punto non esistono nel dataset.

Le reti target vengono aggiornate con Polyak averaging:

```text
target = tau * online + (1 - tau) * target
```

con `tau = 0.005`.

Le reti target e il Polyak averaging servono a rendere più lento il bersaglio del Critic. Senza target network, il valore da inseguire cambierebbe con gli stessi pesi che stiamo aggiornando, creando un inseguimento instabile. `tau = 0.005` rende il target abbastanza reattivo da seguire il training, ma abbastanza lento da filtrare oscillazioni.

## Auto-Refinement E Rollback

Il progetto include una macchina a stati per uscire dai plateau. Ogni 5 episodi, dopo i 15000 step di warm-up, l'agente viene valutato deterministicamente. Le ultime 8 distanze valide formano una finestra mobile.

L'auto-refinement si attiva quando:

- la finestra è piena;
- la media recente non migliora di almeno il 2%;
- questo accade per 4 valutazioni;
- si è oltre l'episodio 200;
- l'Actor non è congelato.

Durante refinement:

- il Critic non viene aggiornato;
- la loss Critic resta diagnostica;
- il peso BC scende a `0.3`;
- l'Actor cerca di sfruttare meglio una value function fissa.

La refinement viene consolidata se supera stabilmente il plateau o torna vicino al miglior record deterministico. Viene invece interrotta con rollback se la distanza scende sotto il 60% del riferimento per 3 valutazioni consecutive. Dopo un rollback, l'Actor può essere congelato per alcuni episodi per permettere al Critic di riallinearsi.

Questa logica deriva dall'idea di ridurre progressivamente il vincolo BC quando la policy è già buona, ma con protezioni operative contro collasso e regressioni.

## Checkpoint E Record

Il salvataggio è progettato per sopravvivere a interruzioni:

- `safe_save()` scrive prima su file temporaneo;
- forza `fsync`;
- ruota backup `.bak` e `.prev`;
- usa `os.replace()` per sostituzione atomica;
- salva i replay buffer separatamente in `train_set/checkpoints/buffers/`.

Questa complessità è giustificata dal costo dei run lunghi: perdere un checkpoint o ripartire con pesi e replay non allineati può buttare via ore di simulazione. Scrittura temporanea, `fsync`, backup e controllo dei timestamp riducono il rischio di file corrotti o stati incoerenti dopo Ctrl+C, crash o spegnimenti improvvisi.

I checkpoint principali sono:

| File | Significato |
| --- | --- |
| `bc_policy.pth` | Policy supervisionata iniziale |
| `state_norm.npz` | Statistiche mean/std delle feature |
| `td3_checkpoint.pth` | Stato completo di training TD3+BC |
| `td3_policy.pth` | Ultimo Actor TD3 salvato |
| `td3_expl_best_lap.pth` | Miglior giro ottenuto in fase esplorativa |
| `td3_expl_best_dist.pth` | Miglior distanza in fase esplorativa |
| `td3_det_best_dist_run.pth` | Miglior distanza deterministica del run corrente |
| `td3_det_best_dist.pth` | Miglior distanza deterministica assoluta |
| `td3_det_best_lap.pth` | Miglior giro deterministico valido, candidato submission |

`test_agent.py` usa questa gerarchia per scegliere automaticamente i pesi migliori quando `--weights` non viene specificato.

## Test Deterministico

Il test usa `PolicyActor`, compatibile con pesi BC e TD3+BC. La modalità viene dedotta dal nome del checkpoint:

- file con `td3` sono trattati come RL;
- file con `bc` sono trattati come Behavioral Cloning;
- se il nome non è riconoscibile, bisogna passare `--kind rl` o `--kind bc`.

Durante il test:

- l'Actor è in `eval()`;
- non viene aggiunto rumore;
- la marcia resta algoritmica;
- ogni tentativo rilancia TORCS con `relaunch=True`;
- viene salvata telemetria CSV in `telemetry/`.

Questo è il percorso da usare per misurare le prestazioni reali candidate alla competizione.

La valutazione deterministica è separata dal training perché il rumore esplorativo è utile per imparare, ma non rappresenta la policy da portare in gara. Salvare telemetria per ogni tentativo permette di capire dove la policy perde tempo o stabilità: velocità, freno, sterzo, marcia e `trackPos` mostrano se il limite è una staccata, una curva o una scelta del cambio.

## Riferimenti Scientifici

I riferimenti qui sotto sono quelli effettivamente collegati alle scelte del codice:

1. Lillicrap et al., "Continuous Control with Deep Reinforcement Learning", 2015. Base concettuale DDPG per controllo continuo actor-critic. https://arxiv.org/abs/1509.02971
2. Fujimoto, van Hoof, Meger, "Addressing Function Approximation Error in Actor-Critic Methods", 2018. Paper TD3: twin critics, delayed policy update, target policy smoothing. https://arxiv.org/abs/1802.09477
3. Fujimoto, Gu, "A Minimalist Approach to Offline Reinforcement Learning", 2021. Paper TD3+BC: termine BC nella policy loss e normalizzazione dei dati. https://arxiv.org/abs/2106.06860
4. Beeson, Montana, "Improving TD3-BC: Relaxed Policy Constraint for Offline Learning and Stable Online Fine-Tuning", 2022. Ispirazione per refinement e riduzione controllata del vincolo BC. https://arxiv.org/abs/2211.11802
5. Bojarski et al., "End to End Learning for Self-Driving Cars", 2016. Riferimento per apprendimento da dimostrazioni umane e augmentation di guida. https://arxiv.org/abs/1604.07316
6. Loiacono, Cardamone, Lanzi, "Simulated Car Racing Championship: Competition Software Manual", 2013. Descrizione del software SCR, sensori e attuatori usati dal client TORCS. https://arxiv.org/abs/1304.1672
