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
La nostra implementazione cattura l'essenza matematica del TD3+BC, ma introduce cinque variazioni ingegneristiche fondamentali per operare la transizione da un dominio puramente *Offline* (usato nel paper) a un dominio *Online* con esplorazione attiva:

- **Target dell'Azione Esperta**: Nel paper originale l'azione target $a_{expert}$ viene campionata dal dataset. Anche noi applichiamo il **masking rigoroso**, sfruttando l'azione empirica registrata in memoria per gli stati dell'Elite Buffer e azzerando la BC Penalty per i campioni esplorativi online. Questo impedisce all'agente di subire il covariate shift su stati OOD.
- **Relaxed Policy Constraint Corretto (Alpha Dinamico e BC Decay)**: Invece dell'Alpha statico proposto in TD3+BC, implementiamo la versione corretta del *Relaxed Policy Constraint* di Beeson & Montana (2022). Fissiamo $\lambda = 2.5$ in modo che $\alpha = \frac{\lambda}{\frac{1}{N} \sum |Q(s_i, a_i)|}$ mantenga una scala costante ed invariante per i gradienti del Critic, e applichiamo un decadimento esponenziale al peso del Behavioral Cloning ($w_{BC}$ che scende da $1.0$ verso un **floor permanente di $0.5$** su 200.000 step, senza mai azzerare l'ancora): $\mathcal{L}_{actor} = \alpha \cdot \mathcal{L}_{actor\_td3} + (w_{BC} \cdot \text{BC\_Penalty})$.
- **Loss di Imitazione Domain-Specific (Prevenzione della Diluizione)**: Invece del generico MSE su tutto il vettore d'azione, applichiamo una loss pesata (sterzo e freno pesati doppiamente) calcolata **esclusivamente sul sotto-batch di campioni esperti** per evitare la diluizione causata dall'inserimento di campioni online nel batch ibrido. Inoltre, la loss viene sommata direttamente (senza dividere per la somma dei pesi) per allineare l'intensità del gradiente BC con i coefficienti del paper originale, controbilanciando la costante $\lambda = 2.5$. Viene aggiunta una *Mutual Exclusion Penalty* per impedire il blocco simultaneo di freno e acceleratore.
- **Compensazione dell'Attivazione per l'Inizializzazione (BC Weight Scaling)**: Per risolvere la discrepanza tra l'attivazione `Sigmoid` (usata nel BC per acceleratore e freno) e l'attivazione `Tanh` (usata nell'Actor del TD3), i pesi e i bias caricati da `bc_policy.pth` per i canali di accelerazione e freno vengono dimezzati (`0.5`) al caricamento. Questo compensa perfettamente la relazione algebrica $\frac{\tanh(0.5x)+1}{2} = \sigma(x)$, rendendo l'inizializzazione al warm-start matematicamente indistinguibile dal modello BC originale.
- **Mutual Exclusion Fisica Unificata (Formula Moltiplicativa)**: Per ripristinare uno spazio d'azione liscio e differenziabile (evitando discontinuità a gradiente nullo/causal-confusion alla linea di partenza), abbiamo sostituito l'esclusione a soglia rigida con la formula moltiplicativa continua: $\text{accel}_{\text{final}} = \text{accel} \times (1.0 - \text{brake})$. Questa logica è unificata in `td3_bc.py` (training/eval) e `test_agent.py` (BC/RL), garantendo partenze senza stalli.

## 2. Critic (Twin Q-Network)
Il Critic ha il compito di stimare il valore (Q-value) della coppia (Stato, Azione). Poiché il BC non usa una value-function, il Critic deve essere addestrato da zero.
- **Architettura Twin**: Usa due reti Q indipendenti per mitigare l'Overestimation Bias tipico del Q-learning. Si prende il minimo tra le due stime durante l'aggiornamento dell'Actor.
- **Critic Warm-Up Exteso (15.000 step)**: Poiché nel nostro setup Offline-to-Online il Critic viene inizializzato da zero (a differenza del paper dove è pre-addestrato offline), l'Actor viene congelato per i primi `15.000` step. Questo permette al Critic di apprendere una Value Function solida e previene la *Critic Warmup Degradation*, ovvero la distruzione dei pesi BC perfetti da parte di gradienti casuali o sproporzionati inviati da un Critic immaturo.

## 3. Parametri TD3+BC e Il Sistema di Loss (Anti-Drift)
L'integrazione di una BC Penalty in un algoritmo TD3 richiede una calibrazione millimetrica per bilanciare l'imitazione dell'esperto e la massimizzazione del Reward.

- **Equazione Actor Loss (TD3+BC)**: $\mathcal{L}_{actor} = - \alpha \cdot Q(s,a) + \text{BC\_Penalty}(a, a_{expert})$.
- L'Actor viene costretto a massimizzare il Q-Value (derivato dal RL) **senza** abbandonare la traccia dei dati estratti dal Behavioral Cloning.
- **Dynamic Alpha Normalization**: Il coefficiente $\alpha$ viene calcolato dinamicamente come $\frac{\lambda}{\frac{1}{N} \sum |Q|}$. Il parametro $\lambda$ è mantenuto **fisso a 2.5** (come da Fujimoto & Gu, 2021): essendo applicato al termine Q, rende il gradiente RL auto-bilanciante e totalmente invariato a eventuali *reward scaling*. Il rilassamento del vincolo imitativo **non** avviene riducendo $\lambda$, ma applicando un decadimento esponenziale al **peso della BC Penalty** ($w_{BC}$ da $1.0$ verso il floor $0.5$): durante i primi 15.000 step di warm-up $w_{BC}=1.0$ (massima fedeltà), poi decade asintoticamente verso $0.5$ sui 200.000 step successivi, **senza mai svanire** (Permanent BC Adherence). *(Vedi `td3_bc.py`, blocco `dynamic_alpha`/`bc_weight` — il codice è la fonte di verità.)*

### A. Reward per Singolo Step (Dense Reward & Soft Shaping)
A ogni istante `t`, l'agente riceve una ricompensa così calcolata:
`Reward = (Progress * 1.5) + Pos_Penalty - Steer_Smoothness - Corner_Overspeed_Penalty`

- **Progress = (speedX / 50.0) * cos(angle) * 1.5**:
  - Incoraggia la velocità (`speedX`): scalato per compensare la presenza continua della penalità di posizione.
  - Penalizza le sbandate (`cos(angle)`): Se la macchina non è perfettamente allineata all'asse della pista, il coseno (es. `cos(60°) = 0.5`) taglia drasticamente il punteggio.
- **Pos_Penalty = -1.0 * (trackPos ** 2)**: 
  - **Soft Constrained**: Una penalità quadratica sulla distanza dal centro. Quando l'auto è al centro (`trackPos ~ 0.1`), la penalità è irrisoria (`-0.01`), fungendo da "deadzone" naturale per le lievi sbandate apprese dal BC. Man mano che l'auto scivola verso l'erba (`trackPos = 0.8`), la penalità cresce esponenzialmente (`-0.64`), fungendo da "muro repulsivo" molto prima del crash. Questo previene il *Reward Hacking* dove l'agente preferiva sbattere piuttosto che sterzare.
- **Steer Smoothness = -0.05 * abs(steer - last_steer)**:
  - Penalizza le variazioni brusche di sterzo, impedendo comportamenti a "zig-zag". Il peso ridotto (`0.05`) incoraggia l'agente a usare lo sterzo per tornare verso il centro della pista senza temere eccessive perdite di punti.
- **Corner Overspeed Penalty = -K · max(0, 0.5 - front)² · (speedX/50)** (con `K = CORNER_OVERSPEED_K = 2.5`):
  - **Anti-understeer in staccata**: attacca il fallimento ricorrente in cui l'agente arriva al tornante troppo veloce (~165 invece di ~130 km/h) e allarga di pista. `front = min(track[8..10])/200` è la distanza frontale normalizzata: quando una curva è vicina (`front < 0.5`, cioè entro ~100m) la penalità cresce **quadraticamente** con la vicinanza e **linearmente** con la velocità, insegnando a *scaricare velocità in ingresso* (frenare) dove serve.
  - **Generale, non hardcoded**: basata sui sensori di pista, non su una posizione specifica del tracciato → funziona su qualsiasi curva/circuito. Su pista libera (`front ≥ 0.5`) la penalità è esattamente `0`.
  - **Coerenza Critic**: la stessa identica formula (con `K` replicato) è applicata sia al reward online (`gym_torcs.py`) sia al reward dei campioni expert iniettati nel buffer (`td3_bc.load_expert_data`), così il Critic riceve un segnale coerente. `K` va tarato durante il training.

### B. Penalità Terminali (Crash e Fuoripista)
Se l'auto esce di pista (`|trackPos| > 1.5`), si schianta, o va in stallo, l'episodio termina (`done=True`) e riceve un **`-10.0`**.

> **Perché -10 e non -1000? La matematica di Bellman**
> Il Critic valuta il Q-Value con un discount factor `gamma = 0.999`. Il valore massimo stimabile per una guida perfetta e infinita è una serie geometrica: `Q_max = 1.0 / (1 - 0.999) = 1000.0`.
> Se l'auto va fuori strada, il flag `done=True` "brucia" del tutto l'aspettativa di vita (+1000.0) e impone il limite terminale di `-10.0`.
> La perdita reale percepita dalla rete neurale per quell'errore è quindi un **differenziale di -1010 punti** su una scala di 1000 (prima del Reward Scaling). 
> Se impostassimo la penalità a `-1000`, la Mean Squared Error del Critic impazzirebbe, causando esplosione dei gradienti e *Catastrophic Forgetting*. La penalità di -10 è letale per l'agente, ma "sicura" per i gradienti.

### C. Bilanciamento Matematico (Gamma, Reward Scale, Alpha)
L'integrazione di una BC Penalty in un algoritmo RL ad alta frequenza (50Hz) richiede una calibrazione millimetrica per evitare che una forza matematica sopprima l'altra.
- **Gamma = 0.999 (Orizzonte Lungo)**: Aumentato dallo standard `0.99` per estendere la visione del Critic a 1000 step (20 secondi). Senza questo orizzonte lungo, l'agente non "vedeva" in tempo le curve ad alta velocità.
- **Reward Scale = 0.002**: L'aumento del Gamma decuplica la magnitudo dei Q-Values. Riducendo lo scaling si compensa l'effetto e si evitano gradienti esplosivi.
- **Dynamic Alpha Normalization con Relaxed Constraint**: Il gradiente RL è bilanciato dalla formula $\alpha = \frac{\lambda}{\frac{1}{N} \sum |Q|}$ con $\lambda = 2.5$ fisso per invarianza allo scale del Critic. Il vincolo imitativo viene allentato applicando un decadimento esponenziale direttamente alla BC Penalty ($w_{BC}$ da $1.0$ verso il floor $0.5$ su 200k step, mai a zero) (Beeson & Montana, 2022).
- **Bonus Completamento Giro (+50.0)**: Quando l'agente completa un giro, riceve un bonus di `+50.0` reward. Senza questo segnale esplicito, il Critic non distingue "stava andando bene prima del crash" da "ha completato il circuito".

### D. Ambiente Esplorativo (Anti-Stall Relaxed)
Per proteggere l'esplorazione nei primissimi secondi di un episodio, il motore fisico di `TorcsEnv` è stato allentato. L'antistallo originario (che uccideva l'episodio se l'auto non superava i 20 km/h in 3 secondi) è stato portato a **10 secondi e 5 km/h**. Questo permette alla rete, inizialmente incerta a causa del rumore esplorativo, di scoprire i pedali senza subire terminazioni falsamente punitive.

## 4. Replay Buffer, Checkpointing e Buffer Injection
- **Masking Corretto**: Il flag `mask=0.0` (terminale) viene salvato nel buffer *esclusivamente* in caso di crash o fallimento. Il superamento del tempo massimo (`max_steps`) o il completamento del giro non alterano il valore di Bellman (mask = 1.0). **I dati expert** provenienti da giri umani completi usano uniformemente `mask=1.0` per tutti i campioni — il completamento del giro NON è un crash.
- **Compressione su Disco**: Per evitare di perdere dati tra i vari run, l'intero buffer viene salvato come array numpy compresso (`.npz`).
- **Expert Buffer Injection**: Il Critic valuterà istantaneamente i Q-Value delle mosse esperte, forzando l'Actor a imitarle.
- **Update Frequency 1:4**: Il rapporto update/data è ridotto a 1:4 (un aggiornamento ogni 4 step di simulazione). Questo previene l'overfitting sulle stesse transizioni e stabilizza i gradienti del Critic, che altrimenti oscillerebbero campionando ripetutamente dati correlati.

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

### Corner Emphasis: Oversampling Pesato per Posizione sul Tracciato (behavioral_cloning.py)
Quando un settore specifico del circuito (es. una staccata ad alta velocità) è **sotto-rappresentato** o richiede una manovra molto più precisa del resto del giro, il BC — che minimizza un MSE *medio* — tende a non dargli abbastanza importanza, e l'agente esce di pista sempre nello stesso punto. La soluzione è un **oversampling pesato per posizione**:

- **Identificazione al metro (`distFromStart`)**: una curva non si delimita per *tempo* (impreciso, varia ad ogni giro) ma per **posizione sul tracciato**, cioè un intervallo di `distFromStart` costante. La posizione esatta per ogni step (`_lap_positions`) si ricava, in ordine di preferenza: (1) dal metadato **`dist_from_start`** salvato nel giro stesso dalle nuove raccolte di `data_collection.py`; (2) dal **backup 30D** (`dataset_backup/laps/`, col 29 × `DIST_NORM_DIVISOR`) allineato per indice; (3) fallback a peso uniforme. Le nuove raccolte sono quindi auto-sufficienti (non serve più il backup).
- **Pesatura nella loss (`CORNER_EMPHASIS_ZONES`)**: ogni zona è una tupla `(start_m, end_m, peso)`. I campioni che cadono nell'intervallo ricevono un peso maggiore nella *media pesata* della loss BC. Con peso `1` ovunque la loss coincide esattamente con la media semplice (scala e val-loss invariate).
- **⚠️ DISATTIVATA di default (`CORNER_EMPHASIS_ZONES = []`)**: questo ripeso artificiale, testato, **degradava il comportamento closed-loop** (la policy regrediva: usciva di pista *prima*, a ~236m invece di ~811m). Il bilanciamento curva/resto-pista si ottiene quindi in modo naturale tramite la **quantità di dati reali** raccolti sulla curva (`data_collection --segment_only`), non con un moltiplicatore. L'infrastruttura resta disponibile per riattivazioni mirate.
- **Vincolo architetturale fondamentale**: `distFromStart` è usato **esclusivamente come etichetta** (per pesare i campioni e per le analisi) — **NON** viene mai concatenato al vettore di stato. La rete resta rigorosamente **29D** (la 30ª feature era stata rimossa per train-test mismatch e non viene reintrodotta).

## 8. Multimodal Averaging & Permanent BC Adherence (Residual RL)
Il dataset umano originale del Behavioral Cloning (BC) contiene intrinsecamente traiettorie eterogenee (es. stringere in una curva al giro 1, allargare al giro 2). Quando una rete neurale impara da questi dati minimizzando il Mean Squared Error (MSE), tende ad apprendere la **media matematica** delle manovre. In curve complesse, questo porta spesso al **Multimodal Averaging** (un comportamento indeciso).

Per ovviare a questo problema senza far deragliare l'agente (Extrapolation Error), l'architettura implementa una strategia di **Residual Reinforcement Learning con Relaxed Constraint**:
- **Dynamic Alpha Normalization con BC Decay**: Il gradiente RL è bilanciato dinamicamente sui Q-value del Critic con $\lambda = 2.5$ fisso. Grazie al decadimento esponenziale del peso $w_{BC}$ della BC Penalty (da $1.0$ verso il floor $0.5$ su 200.000 step), l'Actor parte come "imitatore guidato" e transita verso un'ottimizzazione RL residua sugli stati esplorativi, mantenendo **sempre** un ancoraggio significativo (≥0.5) sui dati expert.
- **Learning Rate Mirato**: L'Actor viene addestrato con un Learning Rate standard di `3e-4`, ma agisce solo ed esclusivamente sul `continuous_head`, lasciando il resto della rete congelato per proteggere i pesi calibrati.

## 9. Elite Buffer e Self-Imitation Learning (Episodic Prioritization)
Per mitigare la *Sample Inefficiency* e il *Catastrophic Forgetting* intrinseco nel campionamento casuale uniforme (Uniform Random Sampling), l'architettura sfrutta una strategia di **Self-Imitation Learning** basata su un'architettura a **Doppio Buffer**:
- **Caching Episodico**: Le transizioni non vengono caricate step-by-step, ma raggruppate per episodio.
- **Elite Buffer (Monotonic Threshold)**: Se un episodio supera una soglia di eccellenza, viene clonato in un buffer secondario (`20.000` step). La soglia è rigorosamente legata al record globale assoluto (`best_distance * 0.9`), risultando monotonicamente non decrescente. Questo impedisce alla soglia di abbassarsi per colpa di episodi sub-ottimali e previene l'inquinamento del buffer con dati scadenti (avvelenamento dell'Elite Buffer).
- **Iniezione Expert**: I campioni clonati nell'Elite Buffer vengono flaggati con `expert=1.0`. Questo "inganna" la `bc_penalty` dell'Actor, forzando la rete a trattare i propri record come se fossero dimostrazioni umane ottimali, innescando l'auto-imitazione (Self-Imitation Learning).
- **Hybrid Sampling Robusto**: Durante il training, il TD3 estrae il 75% del minibatch dal buffer standard e il 25% dall'Elite Buffer (`b2`). Per prevenire eccezioni e crash in caso di ridimensionamento della `batch_size`, il sistema valida dinamicamente che il numero di transizioni nell'Elite Buffer sia superiore o uguale a `b2` prima di procedere con l'estrazione mista.
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
Ogni 5 episodi di training, il sistema esegue automaticamente un **episodio di valutazione deterministica** (`evaluate=True`, zero rumore). Se la distanza percorsa o il tempo sul giro migliorano, il checkpoint viene salvato come `td3_best_eval.pth`. Questo garantisce che il checkpoint usato per la presentazione video sia sempre la policy migliore *riproducibile* — non quella del miglior episodio esplorativo (che potrebbe essere un outlier fortunato con rumore stocastico).

> **Coerenza Training ↔ Deployment (Gear Sequential Constraint)**: durante il rollout, la eval e il test (`test_agent.py`) il cambio marcia è soggetto allo **stesso vincolo sequenziale ±1 per step** (no salti 1→4), con `current_gear` inizializzato a 1 ad ogni episodio. Questo assicura che `td3_best_eval.pth`/`td3_best_lap.pth` esibiscano in inferenza esattamente la dinamica con cui sono stati selezionati — eliminando il *train/test mismatch* sul gear che altrimenti renderebbe il giro non riproducibile.
>
> **Nota sul determinismo**: la policy è deterministica al bit; l'ambiente TORCS (UDP real-time + relaunch per episodio) è *near-deterministico*. La partenza è identica per griglia/auto, ma il timing reale può introdurre divergenze: la riproducibilità è alta in pratica, non garantita matematicamente.

## 12. Offline RL Warm-Start (Safe Restart)
Durante le lunghe sessioni di RL, il Critic può saturarsi irrimediabilmente di Q-Value negativi a causa della continua esplorazione stocastica.

Per risolvere questo stallo, l'architettura implementa un caricamento **disaccoppiato** tra i pesi neurali e i Replay Buffer:
- I file `.npz` vengono caricati in memoria **indipendentemente** dall'esistenza di un checkpoint valido.
- Questo consente il **Safe Restart**: cancellare i pesi della rete TD3 (`.pth`), ripartendo con un Actor immacolato (clonato dal BC) e un Critic a zero, ma fornendo un Elite Buffer già popolato.

## 13. Guida all'Interpretazione dei Log di Training
Durante l'esecuzione di `td3_bc.py`, l'analisi dei log è fondamentale per comprendere la salute del sistema e il corretto funzionamento delle dinamiche ibride implementate:

- **CriticL microscopico (`0.000` - `0.004`)**: L'errore del Critic appare irrisorio a causa del forte *Reward Scaling* (`0.002`). Poiché i Q-Value sono numericamente compressi in partenza, il loro Errore Quadratico Medio (MSE) in fase di apprendimento scende spesso sotto la soglia del millesimo, venendo arrotondato a `0.000` in console. Questo è il segno di un Critic sano: gradienti così piccoli evitano la *Gradient Explosion* e proteggono la rete dal collasso. Quando compare uno `0.001`, significa semplicemente che il Critic sta affinando una precisione estrema.
- **ActorL non più inchiodato a plateau**: A differenza delle architetture precedenti che esibivano una ActorL fissa a `2.516` durante i crash, il nuovo Masking Rigoroso libera l'Actor dalla componente imitativa durante le fasi off-distribution. Ci aspetteremo di vedere valori di ActorL progressivamente variabili e decrescenti nel lungo periodo.
- **Transizione del Peso BC**: $\lambda$ resta fisso a `2.5`; ciò che cambia nel tempo è il peso $w_{BC}$ della BC Penalty, che decade da `1.0` verso il **floor `0.5`** sui 200k step successivi al warm-up (mai a zero). L'impatto della BC penalty cala ma resta un'ancora permanente; la componente RL affina in modo residuo.
- **Spike della CriticL associato all'Elite Buffer**: Quando l'agente stabilisce una traiettoria record prolungata (es. una corsa da `2.400m`), questa viene iniettata nell'Elite Buffer. Nelle iterazioni successive, il *Self-Imitation Learning* espone il Critic a questa traiettoria inedita e iper-performante: questo spiazza le vecchie credenze del Critic, provocando un leggero picco temporaneo nella CriticL (es. a `0.004`). Subito dopo l'assimilazione, l'ActorL sprofonda per allinearsi al nuovo record e le distanze dell'agente subiscono un forte balzo in avanti.

## 14. Nomenclatura dei Checkpoint Salvati
La pipeline di training genera diversi file di checkpoint per scopi differenti, salvati nella cartella `train_set/checkpoints/`:

- **`td3_best_lap.pth` (Best Lap)**: Rappresenta la policy con il **tempo sul giro più veloce in assoluto** (lap time minimo su un giro completato con successo). Viene salvato quando l'episodio si conclude con `SUCCESS` e il tempo sul giro è inferiore a tutti i precedenti.
- **`td3_best_dist.pth` (Best Distance)**: Rappresenta la policy con la **distanza percorsa più lunga** registrata durante gli episodi di addestramento stocastici (esplorativi) prima di un crash (se superiore a 500m).
- **`td3_best_eval.pth` (Best Eval)**: Rappresenta la policy con la **migliore performance di distanza** ottenuta esclusivamente durante le valutazioni deterministiche (senza rumore di esplorazione).
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
- **Rilievo e Fallback Automatico**: All'avvio del training, queste metriche vengono ripristinate dinamicamente. Se il checkpoint caricato non contiene tali chiavi (formato legacy), il sistema effettua un rilevamento automatico della presenza del file `td3_best_lap.pth` per inizializzare coerentemente i record alle performance storiche del circuito di Corkscrew (84.3s / 3619m), prevenendo regressioni.
