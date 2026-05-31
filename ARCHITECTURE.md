# AIcar: Architettura Ibrida BC-RL (TORCS)

Questo documento descrive in dettaglio l'architettura del modello e le scelte implementative per il fine-tuning tramite Soft Actor-Critic (SAC) partendo da un modello addestrato via Behavioral Cloning (BC).

## 1. Actor (La Policy)
L'Actor è una rete neurale ibrida progettata per trattenere la conoscenza dell'esperto (BC) pur permettendo un adattamento dinamico (RL). Riceve in input uno stato "appiattito" a 29 dimensioni.

- **Backbone (Congelato)**: 4 layer lineari da 512 neuroni con `LayerNorm` e `ReLU`. Congelato (`requires_grad=False`) per prevenire il *Latent Shift* e conservare l'estrazione delle feature originali.
- **Gear Head (Congelato)**: Testa discreta a 7 output per la scelta delle marce. Totalmente deterministica e derivata dal BC.
- **Continuous Head (Fine-tuned)**: Testa a 3 output per Sterzo, Acceleratore e Freno. Inizialmente l'Actor veniva aggiornato con un Learning Rate microscopico (`1e-5`) per non distruggere i pesi BC. Ora che è stato implementato il BC Penalty Decay, il LR viene ripristinato a uno standard SAC di `3e-4` per consentire l'apprendimento RL puro in tempi ragionevoli. Sfruttiamo un override forzato sui `param_groups` post-resume per impedire a PyTorch di ricaricare il vecchio LR in fase di `load_state_dict`.
- **Log_Std Head (Fine-tuned)**: Nuova testa introdotta per il SAC che definisce la varianza (rumore) delle azioni. Inizializzata con un bias di `-3.0` (varianza bassa per esplorazione sicura).

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
L'integrazione di una BC Penalty in un algoritmo SAC ad alta frequenza (50Hz) richiede una calibrazione millimetrica per evitare che una forza matematica sopprima l'altra (causando il collasso della policy o *Reward Hacking*).
- **Gamma = 0.999 (Orizzonte Lungo)**: Aumentato dallo standard `0.99` per estendere la visione del Critic a 1000 step (20 secondi). Senza questo orizzonte lungo, l'agente non "vedeva" in tempo le curve ad alta velocità.
- **Reward Scale = 0.002**: L'aumento del Gamma decuplica la magnitudo matematica dei Q-Values (portandoli a valori altissimi). Senza scalare le ricompense di un fattore di 10 (da `0.02` a `0.002`), il gradiente del Critic diventava così enorme da ignorare del tutto la BC Penalty (portando l'auto a schiantarsi volontariamente per evitare le lievi penalità di stallo). Riducendo lo scaling, si compensa l'effetto del Gamma e si ristabilisce un braccio di ferro equo tra BC e RL.
- **Auto-Tuning dell'Entropia (Alpha)**: L'entropia non funge solo da esplorazione, ma da fondamentale regolarizzatore nell'equazione di Bellman per prevenire l'Overestimation Bias dei Q-Values. Il parametro duale `alpha` è ora ottimizzato automaticamente per convergere verso una `target_entropy` prefissata. L'esplorazione è gestita intrinsecamente dal campionamento stocastico del SAC (`log_std`), rimuovendo la necessità di rumore manuale artificiale che inquinava le dimostrazioni nell'Elite Buffer.

### D. Ambiente Esplorativo (Anti-Stall Relaxed)
Per proteggere l'esplorazione nei primissimi secondi di un episodio, il motore fisico di `TorcsEnv` è stato allentato. L'antistallo originario (che uccideva l'episodio se l'auto non superava i 20 km/h in 3 secondi) è stato portato a **10 secondi e 5 km/h**. Questo permette alla rete, inizialmente incerta a causa del rumore esplorativo, di scoprire i pedali senza subire terminazioni falsamente punitive.

## 4. Replay Buffer, Checkpointing e Buffer Injection
- **Masking Corretto**: Il flag `done=True` viene salvato nel buffer *esclusivamente* in caso di crash o fallimento. Il superamento del tempo massimo (`max_steps`) o il completamento del giro non alterano il valore di Bellman (mask = 1.0).
- **Compressione su Disco**: Per evitare di perdere dati tra i vari run e mitigare il catastrophic forgetting, l'intero buffer viene salvato come array numpy compresso (`.npz`) parallelamente ai pesi PyTorch (`.pth`).
- **Expert Buffer Injection**: Metodo `memory.load_expert_data()` implementato per risolvere le colli di bottiglia esplorativi (*Sample Inefficiency*). Consente di raccogliere dati umani mirati tramite `data_collection.py` su settori ostici e caricarli *offline-to-online* nel Replay Buffer del SAC. Il Critic valuterà istantaneamente i Q-Value di queste mosse esperte (ricalcolando i reward tramite il Soft Shaping), forzando l'Actor a imitarle in pochissimi step di gradiente, abbattendo le tempistiche da ore a minuti.

## 5. Memory Safety (TORCS C++ Engine)
L'ambiente TORCS nativo soffre di un grave memory leak interno quando si riavvia la gara via socket (UDP). 

> **Soluzione Relaunch**: Abbiamo bypassato il memory leak a livello di sistema operativo. Passando `relaunch=True` ad ogni episodio, il server TORCS viene ucciso (`pkill -9 torcs`), le porte UDP vengono svuotate, e viene lanciata una nuova istanza pulita all'interno di un server display virtuale isolato (`xvfb-run`). Questo rende l'ambiente **100% memory safe** anche per addestramenti di giorni interi.

## 6. Strategie per Velocizzare l'Addestramento (Fast-Track)
L'addestramento RL puro per il superamento di ostacoli complessi (come curve molto strette) può richiedere ore. Per accelerare massivamente il processo, è consigliato sfruttare la flessibilità dell'architettura ibrida:

1. **Raccogliere nuovi dati mirati**: Usare `data_collection.py` per guidare manualmente e mostrare alla rete come superare il settore in cui si blocca.
2. **Aggiornare il Backbone BC (Scelta Consigliata)**: Fondere i nuovi dati con `preprocess_dataset.py` e ri-addestrare la rete da zero con `behavioral_cloning.py`. Il BC impiega pochi minuti su GPU. Successivamente, riavviare il SAC (`./train_rl.sh --clean`); il RL convergerà quasi istantaneamente perché partirà da un modello che conosce già la fisica della curva.
3. **Iniezione Offline-to-Online**: In alternativa, scommentare `memory.load_expert_data()` in `sac_rl.py` per caricare le proprie traiettorie umane direttamente nel Replay Buffer del SAC (senza resettare i pesi attuali). Il Critic estrarrà dal buffer i campioni perfetti e guiderà l'Actor ad apprendere la nuova manovra.

## 7. Multimodal Averaging & Permanent BC Adherence (Residual RL)
Il dataset umano originale del Behavioral Cloning (BC) contiene intrinsecamente traiettorie eterogenee (es. stringere in una curva al giro 1, allargare al giro 2). Quando una rete neurale impara da questi dati minimizzando il Mean Squared Error (MSE), tende ad apprendere la **media matematica** delle manovre. In curve complesse, questo porta spesso al **Multimodal Averaging** (un comportamento indeciso).

Per ovviare a questo problema senza far deragliare l'agente (Extrapolation Error), l'architettura implementa una strategia di **Residual Reinforcement Learning**:
- **Inizio Training (Step < 100.000)**: Il peso della `bc_penalty` è molto alto (`10.0`). L'Actor è costretto a rimanere vicinissimo alla policy BC. Questo agisce come un "salvagente" o "camicia di forza" che impedisce che la policy crolli (Policy Collapse) scontrandosi ai bordi della pista mentre il Critic mappa i Q-Value.
- **Decadimento Asintotico (fino a 500.000 step)**: Il peso decresce linearmente in un orizzonte lunghissimo, dando tempo all'Actor di allontanarsi molto gradualmente dalla traiettoria umana.
- **Hard Minimum (BC Permanente)**: A differenza del SAC tradizionale, **il peso del BC non scende mai a 0.0**, ma si ferma a un minimo di `2.0`. L'Actor non diventa mai un agente RL puro. Rimane un "imitatore guidato" che usa i Q-Value del Critic solo come piccole correzioni locali (Residuals) per ottimizzare la velocità e gestire le curve in cui la media umana fallisce. Questo garantisce stabilità vitale nell'ambiente.

## 8. Elite Buffer e Self-Imitation Learning (Episodic Prioritization)
Per mitigare la *Sample Inefficiency* e il *Catastrophic Forgetting* intrinseco nel campionamento casuale uniforme (Uniform Random Sampling), l'architettura sfrutta una strategia di **Self-Imitation Learning** basata su un'architettura a **Doppio Buffer**:
- **Caching Episodico**: Le transizioni non vengono caricate step-by-step, ma raggruppate per episodio.
- **Elite Buffer (Monotonic Threshold)**: Se un episodio supera una soglia di eccellenza, viene clonato in un buffer secondario (`20.000` step). La soglia è rigorosamente legata al record globale assoluto (`best_distance * 0.9`), risultando monotonicamente non decrescente. Questo impedisce alla soglia di abbassarsi per colpa di episodi sub-ottimali e previene l'inquinamento del buffer con dati scadenti (avvelenamento dell'Elite Buffer).
- **Iniezione Expert**: I campioni clonati nell'Elite Buffer vengono flaggati con `expert=1.0`. Questo "inganna" la `bc_penalty` dell'Actor, forzando la rete a trattare i propri record come se fossero dimostrazioni umane ottimali, innescando l'auto-imitazione (Self-Imitation Learning).
- **Hybrid Sampling (Generalization Balance)**: Durante il training, il SAC estrae il 75% del minibatch dal buffer standard e il 25% dall'Elite Buffer. Sebbene in passato si sia tentato un "Extreme Optimism" (85% Elite), questo portava a un forte **overfitting** sui singoli stati esatti dei record. Poiché la rete aggiunge un rumore Gaussiano esplorativo (`std=0.05`), l'auto si troverà sempre in stati "sporchi" leggermente diversi dalla traiettoria perfetta. Il 75% di Standard Buffer (con la BC_Penalty ancorata al maestro umano) è vitale per insegnare all'agente a **generalizzare** e recuperare la traiettoria quando si verifica una deviazione stocastica.
- **Isolamento Dati**: Per mantenere pulita la directory dei checkpoint, entrambi i buffer (principale e elite) vengono serializzati in formato `.npz` e memorizzati in una sottocartella dedicata `train_set/checkpoints/buffers/`.

## 9. Prevenzione del Collasso (Frozen BC Anchor e Causal Confusion)
Durante l'addestramento ibrido, l'architettura risolve due problematiche critiche intrinseche al Self-Imitation Learning:

1. **Frozen BC Anchor (Prevenzione Extrapolation Error)**: 
   Nel buffer standard (75% del batch esplorativo), i gradienti RL puri possono degenerare se il Critic si riempie di Q-Value negativi (a seguito di molti schianti in esplorazione), portando l'Actor a manovre suicide (es. schiantarsi alla partenza). 
   Per impedirlo, l'agente istanzia un **Frozen BC Anchor** (`self.bc_policy`), ovvero una copia congelata e immutabile della rete neurale al suo stato iniziale (pesi del clone umano `bc_policy.pth`). 
   Durante il training, la BC Penalty calcola deterministicamente l'errore quadratico medio (MSE) isolando la varianza stocastica. La componente direzionale viene **normalizzata** dividendo per la somma dei pesi, prevenendo un'esplosione della magnitudine (Scalar Shock) che distruggerebbe la deviazione standard. Inoltre, viene applicata una **Mutual Exclusion Penalty** bilanciata (Soft Shaping Bayesiano) per scoraggiare l'uso simultaneo dei pedali senza generare gradienti distruttivi. L'Actor è spinto ad aderire all'umano senza subire un panico entropico.
   Il peso di questa penalità (`bc_weight`) decade in modo **Esponenziale Smorzato** (proteggendo totalmente i pesi nei primissimi step per poi scendere dolcemente senza mai annullarsi). Per mitigare ulteriormente i salti dei gradienti asintotici e ancorare il Critic al buffer d'élite, il **Learning Rate del Critic** è stato ridotto a `1e-4` (pari a quello dell'Actor), garantendo una discesa del gradiente fluida e simmetrica.

1. **Terminal State Mimicry (Sgancio Pre-Schianto)**: 
   Quando un episodio record (salvato nell'Elite Buffer) termina con uno schianto, le ultime azioni sono la causa diretta del fallimento. Forzare l'Actor a imitarle (tramite Self-Imitation) indurrebbe una *Causal Confusion*. 
   Il sistema risolve questo paradosso azzerando la maschera di imitazione (`expert=0.0`) negli ultimi 50 step (esattamente 1 secondo a 50Hz) di un record schiantato. In quella "finestra di evasione", l'agente smette di imitare il suo vecchio errore e torna istantaneamente sotto l'influenza del Reinforcement Learning puro e del Frozen BC Anchor, riuscendo così a frenare e a sopravvivere per estendere ulteriormente il record.

## 10. Offline RL Warm-Start (Safe Restart)
Durante le lunghe sessioni di RL, il Critic può saturarsi irrimediabilmente di Q-Value negativi a causa della continua esplorazione stocastica, portando la rete a un punto morto ("Traumatized Critic") dove l'Actor Loss impazzisce.

Per risolvere questo stallo senza perdere le decine di ore di esperienza accumulate, l'architettura implementa un caricamento **disaccoppiato** tra i pesi neurali e i Replay Buffer:
- I file `.npz` (`buffer` ed `elite_buffer`) vengono caricati in memoria **indipendentemente** dall'esistenza di un checkpoint valido (`sac_checkpoint.pth`).
- Questo consente il **Safe Restart**: è possibile cancellare manualmente i pesi della rete SAC (`.pth`), ripartendo con un Actor immacolato (clonato dal BC) e un Critic inizializzato a zero, ma fornendo loro *fin dal primo step* un Elite Buffer già popolato di record da chilometri.
- **Vantaggio**: Il nuovo Critic salta l'intera fase di esplorazione traumatica iniziale, estraendo immediatamente Q-Value ottimali per le sezioni avanzate della pista, permettendo all'Actor di superare agilmente plateau di addestramento irrecuperabili. Durante i primi 5000 step di questo nuovo ciclo ("Critic Warm-Up"), l'Actor viene deliberatamente "congelato" per proteggere i pesi clonati dal BC mentre il Critic assimila l'esperienza offline.
