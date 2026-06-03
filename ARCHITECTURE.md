# AIcar: Architettura Ibrida BC-RL (TORCS)

Questo documento descrive in dettaglio l'architettura del modello e le scelte implementative per il fine-tuning tramite Soft Actor-Critic (SAC) partendo da un modello addestrato via Behavioral Cloning (BC).

## 1. Actor (La Policy)
L'Actor è una rete neurale ibrida progettata per trattenere la conoscenza dell'esperto (BC) pur permettendo un adattamento dinamico (RL). Riceve in input uno stato "appiattito" a 29 dimensioni.

- **Backbone (Congelato)**: 4 layer lineari da 512 neuroni con `LayerNorm` e `ReLU`. Congelato (`requires_grad=False`) per prevenire il *Latent Shift* e conservare l'estrazione delle feature originali.
- **Gear Head (Congelato)**: Testa discreta a 7 output per la scelta delle marce. Totalmente deterministica e derivata dal BC.
- **Continuous Head (Fine-tuned)**: Testa a 3 output per Sterzo, Acceleratore e Freno. Aggiornata con un micro-LR di `1e-5` per preservare i pesi BC calibrati dal Behavioral Cloning. L'uso di un LR così basso è fondamentale: LR più alti (es. `3e-4`) distruggono i pesi BC in 50-100 episodi, causando il collasso della policy.
- **Log_Std Head (Fine-tuned)**: Nuova testa introdotta per il SAC che definisce la varianza (rumore) delle azioni. Inizializzata con un bias di `-3.0` (varianza bassa per esplorazione sicura). Usa un LR separato di `1e-4` perché è inizializzata da zero e deve convergere più velocemente.

> **Tanh Explosion Prevention**: L'output della rete viene fatto passare per una funzione `tanh`. Poiché la derivata della `tanh` tende a zero agli estremi, la probabilità `log_pi` rischia di esplodere a `+infinito`. Abbiamo implementato un limite matematico (`torch.clamp(log_prob, -20.0, 10.0)`) per garantire la stabilità numerica ad ogni step.

> [!NOTE]
> **Interpretazione dei Log: Perché l'Actor Loss deve essere NEGATIVA?**
> Nel Machine Learning tradizionale (come il Behavioral Cloning), la *Loss* calcola un errore (Mean Squared Error) e l'obiettivo è averla positiva e tendente a zero. In Reinforcement Learning (Actor-Critic), l'obiettivo dell'Actor è *massimizzare* il punteggio (il Q-Value stimato dal Critic). Poiché gli ottimizzatori (PyTorch) lavorano unicamente per *minimizzare*, la funzione di Loss dell'Actor è definita invertendo il segno: `ActorL = BC_Penalty - Q_Value`.
> Quando l'Actor impara a compiere azioni redditizie, il Q-Value previsto diventa fortemente positivo (es. `+100`). Poiché la BC Penalty è già vicina a zero, il termine dominante nell'equazione diventa `-100`. Di conseguenza, nei log di training, una **ActorL che scende sotto lo zero e diventa sempre più negativa è la prova matematica che l'Actor sta imparando a vincere**.
## 2. Critic (Twin Q-Network)
Il Critic ha il compito di stimare il valore (Q-value) della coppia (Stato, Azione). Poiché il BC non usa una value-function, il Critic deve essere addestrato da zero.
- **Architettura Twin**: Usa due reti Q indipendenti per mitigare l'Overestimation Bias tipico del Q-learning. Si prende il minimo tra le due stime durante l'aggiornamento dell'Actor.
- **Critic Warm-Up**: Per i primi `5000` step, viene aggiornato SOLO il Critic. L'Actor non viene toccato. Questo protegge l'Actor dall'essere distrutto da stime casuali di un Critic non ancora addestrato.

## 3. Parametri SAC e Il Sistema di Reward (Anti-Hacking)
Il sistema di ricompensa (Reward Function) è il cuore dell'apprendimento. È stato progettato meticolosamente per evitare il *"Reward Hacking"* (andare troppo piano o fare zig-zag) e per mantenere i gradienti stabili.

### A. Reward per Singolo Step (Dense Reward & Soft Shaping)
A ogni istante `t`, l'agente riceve una ricompensa così calcolata:
`Reward = (Progress * 1.5) + Pos_Penalty - Steer_Smoothness`

- **Progress = (speedX / 50.0) * cos(angle) * 1.5**:
  - Incoraggia la velocità (`speedX`): scalato per compensare la presenza continua della penalità di posizione.
  - Penalizza le sbandate (`cos(angle)`): Se la macchina non è perfettamente allineata all'asse della pista, il coseno (es. `cos(60°) = 0.5`) taglia drasticamente il punteggio.
- **Pos_Penalty = -1.0 * (trackPos ** 2)**: 
  - **Soft Constrained**: Una penalità quadratica sulla distanza dal centro. Quando l'auto è al centro (`trackPos ~ 0.1`), la penalità è irrisoria (`-0.01`), fungendo da "deadzone" naturale per le lievi sbandate apprese dal BC. Man mano che l'auto scivola verso l'erba (`trackPos = 0.8`), la penalità cresce esponenzialmente (`-0.64`), fungendo da "muro repulsivo" molto prima del crash. Questo previene il *Reward Hacking* dove l'agente preferiva sbattere piuttosto che sterzare.
- **Steer Smoothness = -0.05 * abs(steer - last_steer)**:
  - Penalizza le variazioni brusche di sterzo, impedendo comportamenti a "zig-zag". Il peso ridotto (`0.05`) incoraggia l'agente a usare lo sterzo per tornare verso il centro della pista senza temere eccessive perdite di punti.

### B. Penalità Terminali (Crash e Fuoripista)
Se l'auto esce di pista (`|trackPos| > 1.5`), si schianta, o va in stallo, l'episodio termina (`done=True`) e riceve un **`-10.0`**.

> **Perché -10 e non -1000? La matematica di Bellman**
> Il Critic valuta il Q-Value con un discount factor `gamma = 0.99`. Il valore massimo stimabile per una guida perfetta e infinita è una serie geometrica: `Q_max = 1.0 / (1 - 0.99) = 100.0`.
> Se l'auto va fuori strada, il flag `done=True` "brucia" del tutto l'aspettativa di vita (+100.0) e impone il limite terminale di `-10.0`.
> La perdita reale percepita dalla rete neurale per quell'errore è quindi un **differenziale di -110 punti** su una scala di 100. 
> Se impostassimo la penalità a `-1000`, la Mean Squared Error del Critic impazzirebbe (`MSE = 1.2 Milioni`), causando esplosione dei gradienti e *Catastrophic Forgetting*. La penalità di -10 è letale per l'agente, ma "sicura" per i gradienti.

### C. Bilanciamento Matematico (Gamma, Reward Scale, Alpha)
L'integrazione di una BC Penalty in un algoritmo SAC ad alta frequenza (50Hz) richiede una calibrazione millimetrica per evitare che una forza matematica sopprima l'altra.
- **Gamma = 0.999 (Orizzonte Lungo)**: Aumentato dallo standard `0.99` per estendere la visione del Critic a 1000 step (20 secondi). Senza questo orizzonte lungo, l'agente non "vedeva" in tempo le curve ad alta velocità.
- **Reward Scale = 0.002**: L'aumento del Gamma decuplica la magnitudo dei Q-Values. Riducendo lo scaling si compensa l'effetto e si ristabilisce un braccio di ferro equo tra BC e RL.
- **Alpha Fisso = 0.01**: L'alpha è fissato a un valore basso e costante. L'auto-tuning dell'entropia è stato **disattivato** perché in un regime di fine-tuning da BC, l'entropia tende a salire inesorabilmente (nei log: `0.02 → 0.03 → 0.05...`), aggiungendo rumore crescente proprio quando la policy ha bisogno di stabilità. L'esplorazione è gestita nativamente dal campionamento gaussiano minimo del `log_std_head`.
- **BC Weight Fisso = 5.0**: Il peso della BC Penalty è **costante e permanente** — nessun decay esponenziale. L'intero punto dell'architettura è che l'RL fa *correzioni residuali* al BC. Se il vincolo BC decade, la policy diventa puro RL → collasso garantito perché il Critic non è mai sufficientemente accurato su tutti gli stati del circuito.
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
2. **Aggiornare il Backbone BC (Scelta Consigliata)**: Fondere i nuovi dati con `preprocess_dataset.py` e ri-addestrare la rete da zero con `behavioral_cloning.py`. Il BC impiega pochi minuti su GPU. Successivamente, riavviare il SAC (`./train_rl.sh --clean`); il RL convergerà quasi istantaneamente perché partirà da un modello che conosce già la fisica della curva.
3. **Iniezione Offline-to-Online**: In alternativa, scommentare `memory.load_expert_data()` in `sac_rl.py` per caricare le proprie traiettorie umane direttamente nel Replay Buffer del SAC (senza resettare i pesi attuali). Il Critic estrarrà dal buffer i campioni perfetti e guiderà l'Actor ad apprendere la nuova manovra.

## 7. Anti-Covariate Shift (Doppio Livello)
Il **Covariate Shift** è il problema fondamentale del Behavioral Cloning: il modello è addestrato su stati esperti (on-policy), ma a test time, piccoli errori si accumulano perché il modello incontra stati mai visti durante il training (off-policy). L'architettura affronta questo problema su **due livelli complementari**:

### Livello 1: Bojarski-Style Augmentation (behavioral_cloning.py)
Durante il training BC, ogni mini-batch viene perturbato sinteticamente per simulare stati off-distribution:
- **Perturbazione Laterale**: `trackPos` viene spostato di ±0.4 (40% della larghezza della pista). La rete impara a correggere lo sterzo proporzionalmente allo spostamento (gain = 0.25).
- **Perturbazione Angolare**: `angle` viene perturbato di ±0.08 rad (~4.5°). La rete impara a raddrizzare l'auto quando è disallineata rispetto alla pista (gain = 1.5).
- **Perturbazione dei Sensori**: I 19 sensori di distanza dalla pista vengono ricalcolati geometricamente in base alla nuova posizione/angolo simulata, mantenendo la coerenza fisica.
- **Correzione Throttle**: L'acceleratore viene ridotto proporzionalmente alla perturbazione combinata per insegnare cautela in stati anomali.

### Livello 2: Residual RL (sac_rl.py)
Il SAC esplora naturalmente stati off-distribution e impara correzioni locali tramite i Q-Value del Critic. Con il BC Weight fisso a 5.0, le correzioni restano "residuali" — piccoli aggiustamenti alla policy BC senza distruggerla.

## 8. Multimodal Averaging & Permanent BC Adherence (Residual RL)
Il dataset umano originale del Behavioral Cloning (BC) contiene intrinsecamente traiettorie eterogenee (es. stringere in una curva al giro 1, allargare al giro 2). Quando una rete neurale impara da questi dati minimizzando il Mean Squared Error (MSE), tende ad apprendere la **media matematica** delle manovre. In curve complesse, questo porta spesso al **Multimodal Averaging** (un comportamento indeciso).

Per ovviare a questo problema senza far deragliare l'agente (Extrapolation Error), l'architettura implementa una strategia di **Residual Reinforcement Learning**:
- **BC Weight Fisso = 5.0**: Il peso della BC Penalty è costante e permanente. L'Actor non diventa mai un agente RL puro — rimane un "imitatore guidato" che usa i Q-Value del Critic solo come piccole correzioni locali (Residuals). Questo garantisce stabilità anche durante sessioni di training prolungate.
- **Learning Rate Differenziati**: Il `continuous_head` (pesi BC) usa `1e-5`, mentre il `log_std_head` usa `1e-4`. Questa asimmetria protegge i pesi calibrati permettendo alla rete di calibrare l'esplorazione più rapidamente.

## 9. Elite Buffer e Self-Imitation Learning (Episodic Prioritization)
Per mitigare la *Sample Inefficiency* e il *Catastrophic Forgetting* intrinseco nel campionamento casuale uniforme (Uniform Random Sampling), l'architettura sfrutta una strategia di **Self-Imitation Learning** basata su un'architettura a **Doppio Buffer**:
- **Caching Episodico**: Le transizioni non vengono caricate step-by-step, ma raggruppate per episodio.
- **Elite Buffer (Monotonic Threshold)**: Se un episodio supera una soglia di eccellenza, viene clonato in un buffer secondario (`20.000` step). La soglia è rigorosamente legata al record globale assoluto (`best_distance * 0.9`), risultando monotonicamente non decrescente. Questo impedisce alla soglia di abbassarsi per colpa di episodi sub-ottimali e previene l'inquinamento del buffer con dati scadenti (avvelenamento dell'Elite Buffer).
- **Iniezione Expert**: I campioni clonati nell'Elite Buffer vengono flaggati con `expert=1.0`. Questo "inganna" la `bc_penalty` dell'Actor, forzando la rete a trattare i propri record come se fossero dimostrazioni umane ottimali, innescando l'auto-imitazione (Self-Imitation Learning).
- **Hybrid Sampling (Generalization Balance)**: Durante il training, il SAC estrae il 75% del minibatch dal buffer standard e il 25% dall'Elite Buffer. Sebbene in passato si sia tentato un "Extreme Optimism" (85% Elite), questo portava a un forte **overfitting** sui singoli stati esatti dei record. Poiché la rete aggiunge un rumore Gaussiano esplorativo (`std=0.05`), l'auto si troverà sempre in stati "sporchi" leggermente diversi dalla traiettoria perfetta. Il 75% di Standard Buffer (con la BC_Penalty ancorata al maestro umano) è vitale per insegnare all'agente a **generalizzare** e recuperare la traiettoria quando si verifica una deviazione stocastica.
- **Isolamento Dati**: Per mantenere pulita la directory dei checkpoint, entrambi i buffer (principale e elite) vengono serializzati in formato `.npz` e memorizzati in una sottocartella dedicata `train_set/checkpoints/buffers/`.

## 10. Prevenzione del Collasso (Frozen BC Anchor e Causal Confusion)
Durante l'addestramento ibrido, l'architettura risolve due problematiche critiche intrinseche al Self-Imitation Learning:

1. **Frozen BC Anchor (Prevenzione Extrapolation Error)**: 
   Nel buffer standard (75% del batch esplorativo), i gradienti RL puri possono degenerare se il Critic si riempie di Q-Value negativi, portando l'Actor a manovre suicide. 
   L'agente istanzia un **Frozen BC Anchor** (`self.bc_policy`), una copia congelata della rete BC.
   La BC Penalty calcola l'MSE tra l'azione umana e l'azione deterministica `torch.tanh(mean)`, con componente direzionale normalizzata e Mutual Exclusion Penalty bilanciata (Soft Shaping).
   Il peso (`bc_weight = 5.0`) è **fisso e permanente** — nessun decay. Il Learning Rate del Critic è `1e-4` per una discesa simmetrica.

1. **Terminal State Mimicry (Sgancio Pre-Schianto)**: 
   Quando un episodio record (salvato nell'Elite Buffer) termina con uno schianto, le ultime azioni sono la causa diretta del fallimento. Forzare l'Actor a imitarle (tramite Self-Imitation) indurrebbe una *Causal Confusion*. 
   Il sistema risolve questo paradosso azzerando la maschera di imitazione (`expert=0.0`) negli ultimi 50 step (esattamente 1 secondo a 50Hz) di un record schiantato. In quella "finestra di evasione", l'agente smette di imitare il suo vecchio errore e torna istantaneamente sotto l'influenza del Reinforcement Learning puro e del Frozen BC Anchor, riuscendo così a frenare e a sopravvivere per estendere ulteriormente il record.

## 11. Evaluation Periodica Deterministica
Ogni 25 episodi di training, il sistema esegue automaticamente un **episodio di valutazione deterministica** (`evaluate=True`, zero rumore). Se la distanza percorsa o il tempo sul giro migliorano, il checkpoint viene salvato come `sac_best_eval.pth`. Questo garantisce che il checkpoint usato per la presentazione video sia sempre la policy migliore *riprod ucibile* — non quella del miglior episodio esplorativo (che potrebbe essere un outlier fortunato con rumore stocastico).

## 12. Offline RL Warm-Start (Safe Restart)
Durante le lunghe sessioni di RL, il Critic può saturarsi irrimediabilmente di Q-Value negativi a causa della continua esplorazione stocastica.

Per risolvere questo stallo, l'architettura implementa un caricamento **disaccoppiato** tra i pesi neurali e i Replay Buffer:
- I file `.npz` vengono caricati in memoria **indipendentemente** dall'esistenza di un checkpoint valido.
- Questo consente il **Safe Restart**: cancellare i pesi della rete SAC (`.pth`), ripartendo con un Actor immacolato (clonato dal BC) e un Critic a zero, ma fornendo un Elite Buffer già popolato.

## 13. Piano B — TD3+BC (Strategia Alternativa)

> [!NOTE]
> Se il SAC dovesse continuare a manifestare collassi della policy nonostante i fix applicati, la strategia B prevede la migrazione a **TD3+BC** (Fujimoto & Gu, 2021).

**Motivazione:** TD3+BC è stato progettato *specificamente* per il fine-tuning offline-to-online ed è intrinsecamente più stabile del SAC per i seguenti motivi:

1. **Nessuna Entropia**: TD3+BC non usa entropia né campionamento gaussiano durante il training. L'esplorazione è gestita da rumore additivo deterministico, eliminando alla radice il problema dell'escalation entropica.
2. **BC Penalty Nativa**: La loss dell'Actor è definita come `ActorLoss = -Q(s,a) + α · MSE(a, a_BC)`. Il termine BC è strutturale, non un add-on, e bilancia automaticamente RL e imitazione.
3. **Un Solo Iperparametro Critico (α)**: A differenza del SAC (che ha alpha, target_entropy, bc_weight, reward_scale tutti interdipendenti), TD3+BC ha un singolo parametro α che controlla il trade-off BC-RL.
4. **Delayed Policy Update**: L'Actor viene aggiornato solo ogni 2 step del Critic, riducendo ulteriormente il rischio di policy collapse.

**Piano di Migrazione (se necessario):**
- Sostituire la classe `SACAgent` con `TD3BCAgent`
- Rimuovere `log_std_head`, `alpha`, `target_entropy`
- L'Actor usa solo `mean` (deterministico) + rumore Gaussian clip
- Il Critic rimane identico (Twin Q-Network)
- Il Replay Buffer, l'Elite Buffer e il checkpoint system restano invariati
