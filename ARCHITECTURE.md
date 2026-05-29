# AIcar: Architettura Ibrida BC-RL (TORCS)

Questo documento descrive in dettaglio l'architettura del modello e le scelte implementative per il fine-tuning tramite Soft Actor-Critic (SAC) partendo da un modello addestrato via Behavioral Cloning (BC).

## 1. Actor (La Policy)
L'Actor è una rete neurale ibrida progettata per trattenere la conoscenza dell'esperto (BC) pur permettendo un adattamento dinamico (RL). Riceve in input uno stato "appiattito" a 29 dimensioni.

- **Backbone (Congelato)**: 4 layer lineari da 512 neuroni con `LayerNorm` e `ReLU`. Congelato (`requires_grad=False`) per prevenire il *Latent Shift* e conservare l'estrazione delle feature originali.
- **Gear Head (Congelato)**: Testa discreta a 7 output per la scelta delle marce. Totalmente deterministica e derivata dal BC.
- **Continuous Head (Fine-tuned)**: Testa a 3 output per Sterzo, Acceleratore e Freno. Viene aggiornata dal SAC con un Learning Rate estremamente conservativo (`1e-6`) per fare piccoli aggiustamenti senza distruggere i pesi originali.
- **Log_Std Head (Fine-tuned)**: Nuova testa introdotta per il SAC che definisce la varianza (rumore) delle azioni. Inizializzata con un bias di `-3.0` (varianza bassa per esplorazione sicura).

> **Tanh Explosion Prevention**: L'output della rete viene fatto passare per una funzione `tanh`. Poiché la derivata della `tanh` tende a zero agli estremi, la probabilità `log_pi` rischia di esplodere a `+infinito`. Abbiamo implementato un limite matematico (`torch.clamp(log_prob, -20.0, 10.0)`) per garantire la stabilità numerica ad ogni step.

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

- **Auto-Tuning dell'Entropia (Alpha)**: Il parametro Alpha non è più statico. Viene auto-regolato (Auto-Tuning) dinamicamente basandosi su un `target_entropy` di `-3.0` (uguale a `-dim(A)` per le 3 azioni continue). Questo agisce come un "termostato": quando la rete neurale converge verso policy troppo deterministiche (rischiando di bloccarsi in ottimi locali e schiantarsi ripetutamente), il sistema alza automaticamente l'Alpha (aumentando la deviazione standard), costringendo l'agente a esplorare nuove traiettorie. Non appena l'agente trova una via d'uscita e l'incertezza torna a livelli ottimali, l'Alpha si abbassa da solo per massimizzare la precisione e il tempo sul giro.
- **Action Noise nel Rollout**: Oltre all'entropia interna (Alpha), per rompere l'iper-confidenza del Critic viene iniettato un piccolo rumore Gaussiano esterno (`μ=0, σ=0.05`) alle azioni durante il rollout. Questo causa un jitter costante e assicura che due episodi non siano mai identici al 100%. L'inferenza in testing (`evaluate=True`) rimane invece puramente deterministica.

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
