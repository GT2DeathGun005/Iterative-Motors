# AIcar: Architettura Ibrida BC-RL (TORCS)

Questo documento descrive in dettaglio l'architettura del modello e le scelte implementative per il fine-tuning tramite TD3+BC (Twin Delayed DDPG con Behavioral Cloning) partendo da un modello addestrato via Behavioral Cloning (BC).

## 1. Actor (La Policy) e Il Passaggio a TD3+BC
A causa di persistenti problemi di *Catastrophic Forgetting* ed *Escalation Entropica* riscontrati con il framework SAC, l'architettura è stata migrata al **TD3+BC** (Twin Delayed DDPG con Behavioral Cloning), progettato appositamente per il fine-tuning offline-to-online da Fujimoto & Gu (2021). 

L'Actor è ora una rete completamente **deterministica**:
- **Rimozione Entropia**: Il campionamento gaussiano è stato rimosso, eliminando il rumore distruttivo dalla policy di base.
- **Target Policy Smoothing**: Durante il training, l'esplorazione è vincolata e sicura tramite l'aggiunta di rumore additivo clippato ($a = \tanh(\mu) + \text{clip}(\epsilon, -0.2, 0.2)$).
- **Delayed Policy Update**: Per mitigare l'Overestimation Bias, la policy e le reti target vengono aggiornate solo ogni 2 step del Critic (`policy_freq = 2`).

*(Nota: la testa `log_std_head` è mantenuta unicamente disconnessa per retro-compatibilità dei pesi negli script di testing).*

### 1.1 Adattamenti Offline-to-Online vs Paper Originale (Fujimoto & Gu, 2021)
La nostra implementazione cattura l'essenza matematica del TD3+BC, ma introduce le seguenti variazioni ingegneristiche per operare la transizione da un dominio puramente *Offline* (usato nel paper) a un dominio *Online* con esplorazione attiva:

- **Target dell'Azione Esperta**: Nel paper originale l'azione target $a_{expert}$ viene campionata dal dataset. Anche noi applichiamo il **masking rigoroso**, sfruttando l'azione empirica registrata in memoria per gli stati dell'Elite Buffer e azzerando la BC Penalty per i campioni esplorativi online. Questo impedisce all'agente di subire il covariate shift su stati OOD.
- **BC weight COSTANTE = 1.0 (niente decay)**: fissiamo $\lambda = 2.5$ in modo che $\alpha = \frac{\lambda}{\frac{1}{N} \sum |Q(s_i, a_i)|}$ mantenga una scala costante per il termine RL, e teniamo il peso della BC penalty **fisso a 1.0**: $\mathcal{L}_{actor} = \alpha \cdot \mathcal{L}_{actor\_td3} + \text{BC\_Penalty}$, cioè *esattamente* il TD3+BC originale $-\lambda Q + (\pi-a)^2$. *(Un decay precedente del peso BC indeboliva l'ancora nella fase fragile post-warm-up → "troppo RL troppo presto" → collasso, cfr. Beeson & Montana 2022 Ablation 1 e Fujimoto & Gu 2021 ablation su α. Il rilassamento del vincolo, se desiderato, va fatto in una FASE separata dopo il training stabile, con il Critic congelato.)*
- **Loss di Imitazione Domain-Specific (Prevenzione della Diluizione)**: Invece del generico MSE su tutto il vettore d'azione, applichiamo una loss pesata (sterzo e freno pesati doppiamente) calcolata **esclusivamente sul sotto-batch di campioni esperti** per evitare la diluizione causata dall'inserimento di campioni online nel batch ibrido. Inoltre, la loss viene sommata direttamente (senza dividere per la somma dei pesi) per allineare l'intensità del gradiente BC con i coefficienti del paper originale, controbilanciando la costante $\lambda = 2.5$. Viene aggiunta una *Mutual Exclusion Penalty* per impedire il blocco simultaneo di freno e acceleratore.
- **Compensazione dell'Attivazione per l'Inizializzazione (BC Weight Scaling)**: Per risolvere la discrepanza tra l'attivazione `Sigmoid` (usata nel BC per acceleratore e freno) e l'attivazione `Tanh` (usata nell'Actor del TD3), i pesi e i bias caricati da `bc_policy.pth` per i canali di accelerazione e freno vengono dimezzati (`0.5`) al caricamento. Questo compensa perfettamente la relazione algebrica $\frac{\tanh(0.5x)+1}{2} = \sigma(x)$, rendendo l'inizializzazione al warm-start matematicamente indistinguibile dal modello BC originale.
- **Mutual Exclusion Fisica Unificata (Formula Moltiplicativa)**: Per ripristinare uno spazio d'azione liscio e differenziabile (evitando discontinuità a gradiente nullo/causal-confusion alla linea di partenza), abbiamo sostituito l'esclusione a soglia rigida con la formula moltiplicativa continua: $\text{accel}_{\text{final}} = \text{accel} \times (1.0 - \text{brake})$. Questa logica è unificata in `td3_bc.py` (training/eval) e `test_agent.py` (BC/RL), garantendo partenze senza stalli.
- **Normalizzazione degli stati (mean-0 / std-1)**: il secondo cambiamento chiave del paper. Le statistiche (media/dev.std delle 29 feature) sono calcolate sul dataset da `behavioral_cloning.py`, salvate in `state_norm.npz` e applicate in modo identico in BC, RL (`td3_bc.flatten_state`/`load_expert_data`) e test, **prima** di passare lo stato alla rete. Migliora sensibilmente la stabilità. *(Nota: la normalizzazione del dataset BC avviene DOPO l'augmentation Bojarski, che lavora in spazio fisico.)*

## 2. Critic (Twin Q-Network)
Il Critic ha il compito di stimare il valore (Q-value) della coppia (Stato, Azione). Poiché il BC non usa una value-function, il Critic deve essere addestrato da zero.
- **Architettura Twin**: Usa due reti Q indipendenti per mitigare l'Overestimation Bias tipico del Q-learning. Si prende il minimo tra le due stime durante l'aggiornamento dell'Actor.
- **Critic Warm-Up Exteso (15.000 step)**: Poiché nel nostro setup Offline-to-Online il Critic viene inizializzato da zero (a differenza del paper dove è pre-addestrato offline), l'Actor viene congelato per i primi `15.000` step. Questo permette al Critic di apprendere una Value Function solida e previene la *Critic Warmup Degradation*, ovvero la distruzione dei pesi BC perfetti da parte di gradienti casuali o sproporzionati inviati da un Critic immaturo.

## 3. Parametri TD3+BC e Il Sistema di Loss (Anti-Drift)
L'integrazione di una BC Penalty in un algoritmo TD3 richiede una calibrazione millimetrica per bilanciare l'imitazione dell'esperto e la massimizzazione del Reward.

- **Equazione Actor Loss (TD3+BC)**: $\mathcal{L}_{actor} = - \alpha \cdot Q(s,a) + \text{BC\_Penalty}(a, a_{expert})$.
- L'Actor viene costretto a massimizzare il Q-Value (derivato dal RL) **senza** abbandonare la traccia dei dati estratti dal Behavioral Cloning.
- **Dynamic Alpha Normalization**: Il coefficiente $\alpha$ viene calcolato dinamicamente come $\frac{\lambda}{\frac{1}{N} \sum |Q|}$. Il parametro $\lambda$ è mantenuto **fisso a 2.5** (come da Fujimoto & Gu, 2021): applicato al termine Q, rende il gradiente RL invariante alla scala di Q (e normalizza implicitamente il LR). Il **peso della BC Penalty è costante = 1.0** (niente decay): è il TD3+BC originale. *(Vedi `td3_bc.py`, blocco `dynamic_alpha`/`bc_weight` — il codice è la fonte di verità.)*

### A. Reward per Singolo Step (Reward da Corsa, minimalista)
A ogni istante `t`, l'agente riceve una ricompensa così calcolata:
`Reward = (Progress * 1.5) + Pos_Penalty - Steer_Smoothness`

Filosofia (allineata al reward grezzo di Fujimoto & Gu, 2021): **massimizzare la velocità**, lasciando l'agente **libero** su staccate e linee, punendo solo l'uscita dal giro valido. Niente penalità che vincolino *come* guidare (es. la vecchia *corner overspeed penalty* è stata RIMOSSA: creava un attrattore "vai piano/fermati" in cui l'Actor collassava a fine warm-up).

- **Progress = (speedX / 50.0) * cos(angle) * 1.5**: incoraggia la velocità in avanti; il `cos(angle)` taglia il punteggio se l'auto è disallineata all'asse pista. È il termine dominante → guida verso giri veloci.
- **Pos_Penalty = -2.0 * max(0, |trackPos| - 1.0)²** (**deadzone**): **zero** entro `|trackPos| < 1.0` (libertà piena in pista), poi una rampa morbida nella fascia dei cordoli `1.0→1.25` come margine prima del limite di giro valido. Oltre `1.25` → terminale (§B).
- **Steer Smoothness = -0.05 * abs(steer - last_steer)**: lieve anti-zigzag, non vincola velocità/staccate.

### B. Penalità Terminali e Validità del Giro
L'episodio termina con **`-10.0`** se: `|trackPos| > 1.25` (taglio curva / muro — è lo **stesso limite usato in raccolta dati**: cordoli consentiti fino a 1.25, oltre = giro NON valido), schianto (danno), stallo, o spin. Un **bonus `+50.0`** premia il completamento di un giro **valido** (TORCS aggiorna `lastLapTime` solo per giri senza tagli/uscite).

> **Perché -10 (non -1000)?** Con `gamma = 0.99` e reward densa, il valore atteso di una guida continua a velocità sostenuta è ~`progress/(1-γ)` ≈ centinaia di punti. Il deterrente reale del crash è la **perdita di tutto quel futuro** (l'episodio finisce); il `-10` è solo un piccolo segnale terminale aggiuntivo, abbastanza per il Critic ma sicuro per i gradienti.

### C. Bilanciamento Matematico (Gamma, Reward Scale, λ)
- **Gamma = 0.99**: valore di riferimento TD3+BC. (Era `0.999`, che inflazionava i Q di ~10× e aumentava l'overestimation. La curva è già visibile nei sensori dello stato, non serve un orizzonte di 20s.)
- **Reward Scale = 0.02**: tiene la magnitudo dei Q in un range sano con gamma=0.99. *(La normalizzazione λ rende comunque l'Actor invariante alla scala: questo parametro influisce solo sul Critic.)*
- **λ normalization (Fujimoto & Gu, 2021)**: $\alpha = \frac{\lambda}{\frac{1}{N}\sum|Q|}$ con $\lambda = 2.5$ fisso, applicata al termine Q. **Normalizza implicitamente anche il learning rate** rispetto alla scala di Q. Il peso della BC penalty è **costante = 1.0** (vedi §1.1/§8): la loss è quindi *esattamente* quella del TD3+BC: $\mathcal{L} = -\lambda Q + (\pi - a)^2$.
- **Bonus Completamento Giro (+50.0)**: Quando l'agente completa un giro, riceve un bonus di `+50.0` reward. Senza questo segnale esplicito, il Critic non distingue "stava andando bene prima del crash" da "ha completato il circuito".

### D. Ambiente Esplorativo (Anti-Stall Relaxed)
Per proteggere l'esplorazione nei primissimi secondi di un episodio, il motore fisico di `TorcsEnv` è stato allentato. L'antistallo originario (che uccideva l'episodio se l'auto non superava i 20 km/h in 3 secondi) è stato portato a **10 secondi e 5 km/h**. Questo permette alla rete, inizialmente incerta a causa del rumore esplorativo, di scoprire i pedali senza subire terminazioni falsamente punitive.

## 4. Replay Buffer, Checkpointing e Caricamento Dati Esperti
- **Masking Corretto**: Il flag `mask=0.0` (terminale) viene salvato nel buffer *esclusivamente* in caso di crash o fallimento. Il superamento del tempo massimo (`max_steps`) o il completamento del giro non alterano il valore di Bellman (mask = 1.0). **I dati expert** provenienti da giri umani completi usano uniformemente `mask=1.0` per tutti i campioni — il completamento del giro NON è un crash.
- **Compressione su Disco**: Per evitare di perdere dati tra i vari run, il buffer online viene salvato come array numpy compresso (`.npz`).
- **Caricamento Dati Esperti (nativo, niente script esterni)**: i giri umani vengono caricati **automaticamente da `train_set/laps`** nel buffer `expert_memory` permanente ad **ogni** avvio di `td3_bc.py` (sia `--clean` che resume). Per aggiungere nuovi dati a training già avviato basta depositare i nuovi `.h5` in `train_set/laps` e rilanciare `./train_rl.sh` (senza `--clean`): il buffer expert li ricaricherà tutti. *(Il vecchio script `inject_expert_buffer.py`, che iniettava i dati nel buffer online, è stato RIMOSSO: con l'architettura a buffer separato — sez. 9 — iniettare nell'online esporrebbe di nuovo i dati umani alla FIFO, vanificando l'ancora permanente.)* Il Critic valuta così istantaneamente i Q-Value delle mosse esperte, forzando l'Actor a imitarle.
- **Update Frequency 1:1**: un aggiornamento del Critic ad ogni step di simulazione (standard TD3), con Delayed Policy Update dell'Actor ogni 2 step. Il buffer è ampio e diversificato (1M online + expert permanente), quindi l'overfitting su transizioni correlate non è un problema; il 1:1 sfrutta al meglio i dati raccolti e velocizza l'apprendimento.

## 5. Memory Safety (TORCS C++ Engine)
L'ambiente TORCS nativo soffre di un grave memory leak interno quando si riavvia la gara via socket (UDP). 

> **Soluzione Relaunch**: Abbiamo bypassato il memory leak a livello di sistema operativo. Passando `relaunch=True` ad ogni episodio, il server TORCS viene ucciso (`pkill -9 torcs`), le porte UDP vengono svuotate, e viene lanciata una nuova istanza pulita all'interno di un server display virtuale isolato (`xvfb-run`). Questo rende l'ambiente **100% memory safe** anche per addestramenti di giorni interi.

## 6. Strategie per Velocizzare l'Addestramento (Fast-Track)
L'addestramento RL puro per il superamento di ostacoli complessi (come curve molto strette) può richiedere ore. Per accelerare massivamente il processo, è consigliato sfruttare la flessibilità dell'architettura ibrida:

1. **Raccogliere nuovi dati mirati**: Usare `data_collection.py` per guidare manualmente e mostrare alla rete come superare il settore in cui si blocca.
2. **Aggiornare il Backbone BC (Scelta Consigliata)**: Ri-addestrare la rete da zero con i nuovi dati tramite `behavioral_cloning.py`. Il BC impiega pochi minuti su GPU. Successivamente, riavviare il TD3+BC (`./train_rl.sh --clean`); il RL convergerà quasi istantaneamente perché partirà da un modello che conosce già la fisica della curva.
3. **Iniezione Offline-to-Online**: In alternativa, la funzione `memory.load_expert_data()` in `td3_bc.py` viene chiamata per caricare le traiettorie umane direttamente nel Replay Buffer (senza resettare i pesi attuali). Il Critic estrarrà dal buffer i campioni perfetti e guiderà l'Actor ad apprendere la nuova manovra.

## 7. Anti-Covariate Shift (Doppio Livello)
Il **Covariate Shift** è il problema fondamentale del Behavioral Cloning: il modello è addestrato su stati esperti (on-policy), ma a test time, piccoli errori si accumulano perché il modello incontra stati mai visti durante il training (off-policy). L'architettura affronta questo problema su **due livelli complementari**:

### Livello 1: Bojarski-Style Augmentation (behavioral_cloning.py)
Durante il training BC, il **50% di ogni mini-batch** (gating casuale per-campione) viene perturbato sinteticamente per simulare stati off-distribution; l'altro 50% resta **pulito** (delta=0). Questo è cruciale: applicando la perturbazione a *tutti* i campioni (come nella versione precedente) la rete non vedeva mai lo stato ideale sulla linea ottimale e perdeva fedeltà di sterzo (steer MAE misurato 0.116). Col gating al 50% — metà guida precisa, metà recupero OOD — lo steer MAE è sceso a **0.062** e la frenata in staccata da 0.86 a 0.92. Le perturbazioni applicate al sotto-batch sono:
- **Perturbazione Laterale**: `trackPos` viene spostato di ±0.4 (40% della larghezza della pista). La rete impara a correggere lo sterzo proporzionalmente allo spostamento (gain = 0.25).
- **Perturbazione Angolare**: `angle` viene perturbato di ±0.08 rad (~4.5°). La rete impara a raddrizzare l'auto quando è disallineata rispetto alla pista (gain = 1.5).
- **Perturbazione dei Sensori**: I 19 sensori di distanza dalla pista vengono ricalcolati geometricamente in base alla nuova posizione/angolo simulata, mantenendo la coerenza fisica.
- **Correzione Throttle**: L'acceleratore viene ridotto proporzionalmente alla perturbazione combinata per insegnare cautela in stati anomali.

### Livello 2: Relaxed Policy Constraint (td3_bc.py)
Il TD3 esplora naturalmente stati off-distribution. Grazie all'implementazione del Masking Rigoroso della BC Penalty, per gli stati online (esplorativi) il peso dell'imitazione viene azzerato. L'Actor è quindi **completamente libero** di imparare correzioni locali (come frenare e raddrizzarsi per evitare il muro) basate esclusivamente sui Q-Value del Critic, senza che nessuna rete BC interferisca tentando di suggerire azioni OOD "allucinate".

### Livello 3: Split dei Dati BC vs RL (giri interi vs segmenti di curva)
Il BC minimizza l'errore **medio** ed è **cieco alla posizione** (lo stato è 29D, senza `distFromStart`): mappa solo *stato sensoriale → azione*. Di conseguenza una manciata di **segmenti concentrati su una sola curva** (es. la Corkscrew, con sterzo medio doppio rispetto al giro) **sbilancia il BC**: la sterzata pesante di quella curva "trabocca" su stati sensorialmente simili altrove (es. la curva 1), facendo uscire di pista l'agente in punti del tutto scollegati. *(Misurato: aggiungendo 18 segmenti Corkscrew al BC, gli eval di warm-up sono crollati da ~400-818m a ~19-188m; rimuovendoli sono tornati a ~470-813m.)*

La soluzione è separare le sorgenti dati per i due stadi:
- **BC** carica **solo i giri interi** (`lap_[0-9]*.h5`) → distribuzione bilanciata dell'intera pista, nessuno sbilanciamento da segmenti.
- **RL expert buffer** carica **tutto** (`lap_*.h5`, giri + segmenti `lap_seg_*.h5`) → i segmenti mirati rinforzano le curve difficili, ma solo come **25% di anchor** dentro un buffer diversificato, e l'RL ha la *value function* (Critic) per usarli senza imitare ciecamente.

La distinzione è automatica via convenzione di naming (il glob `lap_[0-9]*.h5` esclude i `lap_seg_*.h5`): raccogliere nuovi segmenti mirati con `data_collection.py --segment_only` li indirizza da solo al solo RL, senza rischio di avvelenare il BC.

### Corner Emphasis: Oversampling Pesato per Posizione sul Tracciato (behavioral_cloning.py)
Quando un settore specifico del circuito (es. una staccata ad alta velocità) è **sotto-rappresentato** o richiede una manovra molto più precisa del resto del giro, il BC — che minimizza un MSE *medio* — tende a non dargli abbastanza importanza, e l'agente esce di pista sempre nello stesso punto. La soluzione è un **oversampling pesato per posizione**:

- **Identificazione al metro (`distFromStart`)**: una curva non si delimita per *tempo* (impreciso, varia ad ogni giro) ma per **posizione sul tracciato**, cioè un intervallo di `distFromStart` costante. La posizione esatta per ogni step (`_lap_positions`) si ricava, in ordine di preferenza: (1) dal metadato **`dist_from_start`** salvato nel giro stesso dalle nuove raccolte di `data_collection.py`; (2) dal **backup 30D** (`dataset_backup/laps/`, col 29 × `DIST_NORM_DIVISOR`) allineato per indice; (3) fallback a peso uniforme. Le nuove raccolte sono quindi auto-sufficienti (non serve più il backup).
- **Pesatura nella loss (`CORNER_EMPHASIS_ZONES`)**: ogni zona è una tupla `(start_m, end_m, peso)`. I campioni che cadono nell'intervallo ricevono un peso maggiore nella *media pesata* della loss BC. Con peso `1` ovunque la loss coincide esattamente con la media semplice (scala e val-loss invariate).
- **⚠️ DISATTIVATA di default (`CORNER_EMPHASIS_ZONES = []`)**: questo ripeso artificiale, testato, **degradava il comportamento closed-loop** (la policy regrediva: usciva di pista *prima*, a ~236m invece di ~811m). Il bilanciamento curva/resto-pista si ottiene quindi in modo naturale tramite la **quantità di dati reali** raccolti sulla curva (`data_collection --segment_only`), non con un moltiplicatore. L'infrastruttura resta disponibile per riattivazioni mirate.
- **Vincolo architetturale fondamentale**: `distFromStart` è usato **esclusivamente come etichetta** (per pesare i campioni e per le analisi) — **NON** viene mai concatenato al vettore di stato. La rete resta rigorosamente **29D** (la 30ª feature era stata rimossa per train-test mismatch e non viene reintrodotta).

## 8. Multimodal Averaging & Ancora BC Costante (TD3+BC completo)
Il dataset umano originale del Behavioral Cloning (BC) contiene intrinsecamente traiettorie eterogenee (es. stringere in una curva al giro 1, allargare al giro 2). Quando una rete neurale impara da questi dati minimizzando il Mean Squared Error (MSE), tende ad apprendere la **media matematica** delle manovre. In curve complesse, questo porta spesso al **Multimodal Averaging** (un comportamento indeciso).

Per ovviare a questo problema senza far deragliare l'agente (Extrapolation Error), l'architettura usa il **TD3+BC standard con ancora BC costante**:
- **Dynamic Alpha Normalization, BC costante**: il gradiente RL è bilanciato da $\lambda = 2.5$ fisso (normalizzato su $\frac{1}{N}\sum|Q|$); il peso della BC penalty è **costante = 1.0**. L'Actor massimizza il Q restando ancorato ai dati umani per tutto il training (la loss è esattamente $-\lambda Q + (\pi-a)^2$). Niente decay del vincolo durante la fase online: indebolirlo presto causa collasso (Beeson & Montana 2022, Ablation 1).
- **Backbone scongelato + Learning Rate di riferimento `3e-4`**: come nel TD3+BC originale si allena **tutta la rete dell'Actor** (backbone + `continuous_head`), non solo la testa lineare. È sicuro perché l'ancora BC è forte e **costante** (`bc_weight=1.0`), e sblocca capacità di apprendimento prima limitata. La `gear_head` resta congelata ed è ormai **inutilizzata**: la marcia è calcolata da una logica deterministica esterna (vedi §17.1 `gearing.py`), non più predetta dalla rete. Il LR **non si tunara a tentativi**: la normalizzazione $\lambda$ normalizza già il learning rate rispetto alla scala di Q.

## 9. Elite Buffer e Self-Imitation Learning (Episodic Prioritization)
Per mitigare la *Sample Inefficiency* e il *Catastrophic Forgetting* intrinseco nel campionamento casuale uniforme (Uniform Random Sampling), l'architettura sfrutta una strategia di **Self-Imitation Learning** basata su un'architettura a **Doppio Buffer**:
- **Caching Episodico**: Le transizioni non vengono caricate step-by-step, ma raggruppate per episodio.
- **Elite Buffer (Monotonic Threshold)**: Se un episodio supera una soglia di eccellenza, viene clonato in un buffer secondario (`20.000` step). La soglia è legata al record globale assoluto (`best_distance * 0.7`), risultando monotonicamente non decrescente: impedisce alla soglia di abbassarsi per episodi sub-ottimali e previene l'avvelenamento del buffer. *(Moltiplicatore tarato a `0.7`: con `0.9` un singolo record fortunato precoce bloccava la soglia troppo in alto — l'Elite Buffer si riempiva pochissime volte e il Self-Imitation rinforzava solo poche traiettorie irriproducibili, peggiorando la stabilità.)*
- **Iniezione Expert**: I campioni clonati nell'Elite Buffer vengono flaggati con `expert=1.0`. Questo "inganna" la `bc_penalty` dell'Actor, forzando la rete a trattare i propri record come se fossero dimostrazioni umane ottimali, innescando l'auto-imitazione (Self-Imitation Learning).
- **Buffer EXPERT separato e permanente (anti-FIFO)**: i dati umani vivono in un buffer **dedicato** (`expert_memory`, capacità > dataset) che **non viene MAI svuotato dalla logica FIFO** della `deque`. Prima i campioni esperti erano caricati nello stesso buffer online (FIFO): riempito il buffer, i dati umani — essendo i più vecchi — venivano evicted per primi, facendo **sparire l'ancora BC** e rischiando il collasso della policy. Ora l'expert è separato e ricaricato da disco ad ogni avvio, garantendo l'ancora per tutto il training.
- **Hybrid Sampling a 3 vie**: ogni minibatch è composto **25% Expert (umano) + 15% Elite (self-imitation) + 60% Online**. La quota Expert fissa garantisce che la BC penalty abbia sempre campioni umani su cui ancorarsi (stabilità). Se Online/Elite sono ancora scarsi (inizio training), il resto del batch è riempito dall'Expert (sempre pieno) — di fatto il Critic si pre-allena in modalità offline sui dati umani prima dell'esplorazione online.
- **Prevenzione dei Memory Leak**: Il Replay Buffer effettua la copia esplicita degli array numpy (`.copy()`) durante il campionamento dei dati esperti per slegare i riferimenti in memoria dai file HDF5 originari, garantendo l'efficienza della memoria RAM.
- **Isolamento Dati**: Per mantenere pulita la directory dei checkpoint, entrambi i buffer (principale e elite) vengono serializzati in formato `.npz` e memorizzati in una sottocartella dedicata `train_set/checkpoints/buffers/`.

## 10. Prevenzione del Collasso (Masking Rigoroso e Causal Confusion)
Durante l'addestramento ibrido, l'architettura risolve due problematiche critiche intrinseche al Self-Imitation Learning:

1. **Masking Rigoroso per Prevenire il Covariate Shift**: 
   Nel buffer standard (75% del batch esplorativo), i gradienti RL puri possono degenerare se affiancati ad un'imitazione impropria.
   L'agente sfrutta una **Maschera Esperta** (`expert_mask=1.0` per Elite, `0.0` per Online). La BC Penalty calcola l'MSE tra l'azione umana e l'azione deterministica **solo sui campioni esperti**, azzerandosi per quelli online. 
   Questo elimina la necessità di interrogare una rete BC per gli stati OOD, annullando le allucinazioni e rimuovendo i milioni di parametri extra del vecchio *Frozen BC Anchor*.

2. **Terminal State Mimicry (Sgancio Pre-Schianto)**: 
   Quando un episodio record (salvato nell'Elite Buffer) termina con uno schianto, le ultime azioni sono la causa diretta del fallimento. Forzare l'Actor a imitarle (tramite Self-Imitation) indurrebbe una *Causal Confusion*. 
   Il sistema risolve questo paradosso azzerando la maschera di imitazione (`expert=0.0`) negli ultimi 50 step (esattamente 1 secondo a 50Hz) di un record schiantato. In quella "finestra di evasione", l'agente smette di imitare il suo vecchio errore e torna istantaneamente sotto l'influenza del Reinforcement Learning puro, riuscendo così a frenare e a sopravvivere per estendere ulteriormente il record.

## 11. Evaluation Periodica Deterministica
Ogni 5 episodi di training, il sistema esegue automaticamente un **episodio di valutazione deterministica** (`evaluate=True`, zero rumore). Da questo eval si salvano due record distinti: la **distanza** migliore (`td3_best_eval.pth`, e il globale `td3_best_ever.pth`) e — quando l'agente chiude un giro intero — il **tempo sul giro valido** più veloce (`td3_best_eval_laptime.pth`, vedi §14). Questo garantisce che il checkpoint usato per la presentazione video sia sempre la policy migliore *riproducibile* — non quella del miglior episodio esplorativo (che potrebbe essere un outlier fortunato con rumore stocastico).

> **Coerenza Training ↔ Deployment (Marcia Deterministica)**: durante il rollout, la eval e il test (`test_agent.py`) la marcia è calcolata dalla **stessa funzione deterministica** `gearing.compute_gear` (vedi §17.1), con `current_gear` inizializzato a 1 ad ogni episodio. Questo assicura che `td3_best_eval.pth`/`td3_best_eval_laptime.pth` esibiscano in inferenza esattamente la dinamica con cui sono stati selezionati — eliminando il *train/test mismatch* sul gear che altrimenti renderebbe il giro non riproducibile.
>
> **Nota sul determinismo**: la policy è deterministica al bit; l'ambiente TORCS (UDP real-time + relaunch per episodio) è *near-deterministico*. La partenza è identica per griglia/auto, ma il timing reale può introdurre divergenze: la riproducibilità è alta in pratica, non garantita matematicamente.

## 12. Offline RL Warm-Start (Safe Restart)
Durante le lunghe sessioni di RL, il Critic può saturarsi irrimediabilmente di Q-Value negativi a causa della continua esplorazione stocastica.

Per risolvere questo stallo, l'architettura implementa un caricamento **disaccoppiato** tra i pesi neurali e i Replay Buffer:
- I file `.npz` vengono caricati in memoria **indipendentemente** dall'esistenza di un checkpoint valido.
- Questo consente il **Safe Restart**: cancellare i pesi della rete TD3 (`.pth`), ripartendo con un Actor immacolato (clonato dal BC) e un Critic a zero, ma fornendo un Elite Buffer già popolato.

## 13. Guida all'Interpretazione dei Log di Training
Durante l'esecuzione di `td3_bc.py`, l'analisi dei log è fondamentale per comprendere la salute del sistema e il corretto funzionamento delle dinamiche ibride implementate:

- **CriticL sana (~`0.01` - `5.0`)**: con `gamma=0.99` e `reward_scale=0.02` i Q hanno una magnitudo ragionevole e la CriticL si attesta tipicamente tra `0.01` e qualche unità (con picchi isolati quando l'agente scopre tratti di pista inediti). Valori stabilmente nell'ordine delle centinaia indicherebbero esplosione dei gradienti.
- **ActorL non più inchiodato a plateau**: A differenza delle architetture precedenti che esibivano una ActorL fissa a `2.516` durante i crash, il nuovo Masking Rigoroso libera l'Actor dalla componente imitativa durante le fasi off-distribution. Ci aspetteremo di vedere valori di ActorL progressivamente variabili e decrescenti nel lungo periodo.
- **ActorL ≈ -2.5 a regime**: con `bc_weight=1.0` costante, dopo il warm-up il termine RL normalizzato ($-\lambda Q$) domina e l'ActorL si assesta intorno a `-2.5` (= $-\lambda$). È il comportamento atteso del TD3+BC, non un segno di instabilità.
- **Spike della CriticL associato all'Elite Buffer**: Quando l'agente stabilisce una traiettoria record prolungata (es. una corsa da `2.400m`), questa viene iniettata nell'Elite Buffer. Nelle iterazioni successive, il *Self-Imitation Learning* espone il Critic a questa traiettoria inedita e iper-performante: questo spiazza le vecchie credenze del Critic, provocando un leggero picco temporaneo nella CriticL (es. a `0.004`). Subito dopo l'assimilazione, l'ActorL sprofonda per allinearsi al nuovo record e le distanze dell'agente subiscono un forte balzo in avanti.

## 14. Nomenclatura dei Checkpoint Salvati
La pipeline di training genera diversi file di checkpoint per scopi differenti, salvati nella cartella `train_set/checkpoints/`:

- **`td3_best_lap.pth` (Best Lap)**: Rappresenta la policy con il **tempo sul giro più veloce in assoluto** (lap time minimo su un giro completato con successo). Viene salvato quando l'episodio si conclude con `SUCCESS` e il tempo sul giro è inferiore a tutti i precedenti.
- **`td3_best_dist.pth` (Best Distance)**: Rappresenta la policy con la **distanza percorsa più lunga** registrata durante gli episodi di addestramento stocastici (esplorativi) prima di un crash (se superiore a 500m).
- **`td3_best_eval.pth` (Best Eval)**: Rappresenta la policy con la **migliore performance di distanza** ottenuta esclusivamente durante le valutazioni deterministiche (senza rumore di esplorazione). ⚠️ Viene **cancellato da `--clean`**.
- **`td3_best_ever.pth` (Best Ever) + `td3_best_ever.txt`**: La **migliore policy deterministica ASSOLUTA tra tutti i run** per **distanza percorsa**. A differenza di `td3_best_eval.pth`, **sopravvive a `--clean`** (lo script non lo cancella) e viene aggiornato solo se un eval supera il record globale di distanza (memorizzato nel sidecar `.txt`). Serve a non perdere mai una policy ottima per colpa di un restart pulito sfortunato.
- **`td3_best_eval_laptime.pth` (Best Eval Lap-Time) + `td3_best_eval_laptime.txt`**: La policy con il **GIRO VALIDO più VELOCE chiuso in valutazione deterministica**. Mentre `td3_best_ever` ottimizza la *distanza* (proxy della velocità), questo cattura *direttamente* il **tempo sul giro valido** in condizioni deterministiche e riproducibili → è il **candidato diretto per la submission**. Viene salvato quando un eval completa un giro valido più rapido del record (sidecar `.txt` col tempo in secondi), **sopravvive a `--clean`**, ed è la **priorità massima** nell'auto-detect di `test_agent.py`. *(Distinto da `td3_best_lap.pth`, che usa i giri **esplorativi** rumorosi — potenzialmente non riproducibili.)*
- **`td3_policy.pth`**: Pesi correnti della rete Actor salvati alla fine di ogni episodio.
- **`td3_checkpoint.pth`**: Contiene lo stato globale del training (ottimizzatori di Actor e Critic, contatori globali di step, record storici, ecc.) per supportare il ripristino sicuro (`--resume`) senza perdita di avanzamento.

## 15. Stabilizzazione del Critic e Prevenzione della Degradazione (Actor Freezing)
Durante le sessioni di fine-tuning online o post-rollback della policy, l'agente può andare incontro a repentini collassi a causa del disallineamento temporaneo tra la policy dell'Actor e il valore Q stimato dal Critic (che può essere impreciso o calibrato su vecchie dinamiche).

Per prevenire questo fenomeno denominato *Critic Shock*, l'architettura implementa un meccanismo opzionale attivabile all'avvio:
- **Attivazione Manuale (`--rollback`)**: L'operatore può forzare il recupero eseguendo `./train_rl.sh --rollback`. Questo carica i pesi del miglior giro storico (`td3_best_lap.pth`), reimposta l'ottimizzatore dell'Actor e abilita il congelamento.
- **Actor Freezing Temporaneo**: L'Actor viene congelato (`agent.actor_frozen = True`) per un periodo stabilito di 10 episodi dal momento del ripristino.
- **Warm-Up del Critic**: Durante questa finestra di congelamento, solo i parametri del Critic vengono aggiornati sulle nuove traiettorie generate dall'Actor. Questo permette al Critic di "assimilare" e allineare la sua Value Function alla policy ottimale sotto la nuova fisica del sistema (ad esempio la mutual exclusion moltiplicativa).
- **Scongelamento Sicuro**: Trascorsi i 10 episodi di stabilizzazione, l'Actor viene sbloccato (`agent.actor_frozen = False`), riprendendo l'addestramento TD3+BC con gradienti stabili e costruttivi.

## 16. Dynamic Checkpoint Metrics (Zero Hardcoding)
Per garantire la massima generalità su diversi tracciati ed evitare la sovrascrittura accidentale di checkpoint ottimali (ad esempio se un'interruzione di corrente o un crash forzano il riavvio del training):
- **Stato Dinamico**: Le metriche storiche dei record (`best_lap_time`, `best_eval_dist`, `best_distance`) vengono salvate all'interno dello stato del checkpoint `td3_checkpoint.pth`.
- **Rilievo e Fallback Automatico**: All'avvio del training, queste metriche vengono ripristinate dinamicamente dal checkpoint. Se il checkpoint è in formato weights-only (privo di queste chiavi), i record **ripartono puliti** (`best_lap_time=inf`, `best_distance=0`). *(Nota: il vecchio fallback che hardcodava 84.3s/3619m è stato RIMOSSO — corrompeva `elite_threshold` su resume weights-only, congelando l'Elite Buffer.)*

## 17.1 Cambio Marcia Deterministico (`gearing.py`)
La marcia **non** è più predetta dalla rete (la `gear_head`, congelata durante l'RL, produceva *hunting* estremo — fino a ~322 cambi ogni 1000 step, con assurdità come la 1ª a 150 km/h, che destabilizzavano l'intero giro e spezzavano la trazione). È sostituita da `gearing.compute_gear`, una logica **deterministica velocità-primaria** usata in modo identico in training/eval/test.

- **Anti-hunting by design**: il problema classico degli auto-shifter è che in **staccata** il downshift fa *salire* gli rpm → uno shifter rpm-based crede di dover risalire di marcia → oscilla. Qui il **downshift guarda la VELOCITÀ** (monotòna decrescente in frenata), non gli rpm → il picco di rpm è irrilevante. L'**upshift** scatta solo **se sul gas** (`accel > 0.4`) e con rpm alti: durante la frenata (gas≈0) è bloccato anche se gli rpm superano la soglia.
- **Isteresi + cooldown**: soglie di upshift > soglie di downshift, più un lockout di alcuni step dopo ogni cambio → zero jitter al confine.
- **Soglie derivate e validate sui 75 giri umani**: accordo **±1 marcia 99.0%** con la guida umana, **9.7 cambi/1000 step** (umano reale 7.8), **0% rischio fuorigiri** (mai marce troppo basse ad alta velocità). Conferma empirica del design: gli upshift umani avvengono con `accel~1.00` a rpm~19400, i downshift con `brake~1.00`.

## 17.2 Auto-Refinement (Relaxed Policy Constraint a Plateau, Beeson & Montana 2022)
Quando la policy deterministica si **stabilizza in un plateau** sotto il giro completo (tipico: il muro di una curva difficile), il vincolo BC che la àncora ai dati umani diventa un freno. Il *Relaxed Policy Constraint* (Paper 2) lo allenta in una **fase separata** per spingere oltre. Qui è automatizzato con una macchina a stati nel loop di training (`td3_bc.py`):

- **Trigger conservativo e STATISTICO**: la refinement si attiva SOLO quando la **MEDIA delle ultime 8 valutazioni** (la performance *tipica*, non un singolo colpo di fortuna) **smette di salire** (incremento < 2%) per **12 valutazioni** (~60 episodi) E l'episodio ≥ 200. Usare la media (e non il singolo `best`) evita di attivarsi prematuramente quando la performance tipica sta ancora migliorando pur sotto un picco fortunato precoce. *(Attivarla troppo presto → collasso, Ablation 1 del paper: per questo è conservativa.)* Il flag `--refine` la avvia **subito** quando l'operatore sa già di essere in plateau (riferimento rollback fissato dopo i primi eval).
- **Azione**: il **Critic viene congelato** (value function fissa) e il `bc_weight` scende da `1.0` a `0.3`, lasciando l'Actor più libero di raffinarsi verso i Q-Value del Critic.
- **Avviso di recupero**: quando un eval supera il riferimento di plateau di oltre il 10%, viene loggato una-tantum `🚀 PLATEAU SUPERATO` — segnale esplicito che la refinement sta funzionando (la policy ha rotto il muro).
- **🛡️ Rete di sicurezza (auto-rollback)**: se durante la refinement l'eval crolla sotto il **60% del livello recente** (il MAX delle ultime ~6 valutazioni — robusto all'alta varianza bimodale) per **3 valutazioni consecutive**, l'Actor viene **ripristinato da `td3_best_ever.pth`**, il `bc_weight` torna a `1.0` e il Critic si scongela. La policy migliore non si perde MAI (è sempre su disco).
- **Anti-loop**: massimo **3 tentativi** di refinement; oltre, il training prosegue normale. La logica della macchina a stati è validata offline (plateau→attiva, collasso→rollback, ancora-in-salita→non attiva, bimodale→nessun rollback spurio).

## 17. Possibili Miglioramenti Futuri (non bloccanti)
Punti identificati come migliorabili ma **deliberatamente non modificati** per non destabilizzare un sistema funzionante a ridosso della scadenza. Vanno valutati solo dopo aver ottenuto un giro valido.
- **Frame stacking di 3 stati (87D) — probabilmente ridondante**: l'input concatena 3 frame (`t-12, t-6, t` a `k=6`) per dare contesto temporale alla rete. Ma lo stato 29D **contiene già** le velocità (`speedX/Y/Z`, `wheelSpinVel`, `rpm`): le dinamiche istantanee sono quindi già osservabili senza stacking. Il paper TD3+BC originale lavora su single-frame. Lo stacking aggiunge contesto a soli `~0.24s` al costo di **triplicare** la dimensione dell'input (e quindi i parametri del primo layer). **Candidato a semplificazione** (29D single-frame): ridurrebbe i parametri e semplificherebbe la pipeline, ma richiede ri-addestramento BC + ri-validazione, quindi va fatto solo a obiettivo raggiunto e con tempo a disposizione.
