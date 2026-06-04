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
La nostra implementazione cattura l'essenza matematica del TD3+BC, ma introduce tre variazioni ingegneristiche fondamentali per operare la transizione da un dominio puramente *Offline* (usato nel paper) a un dominio *Online* con esplorazione attiva:

- **Target dell'Azione Esperta**: Nel paper originale l'azione target $a_{expert}$ viene campionata dal dataset. Noi usiamo il backbone congelato `self.bc_policy(state)` per inferire il target. Questo è vitale perché, esplorando online, l'agente incontra stati off-distribution che non esistono nel dataset originale.
- **Dynamic Alpha Normalization Invariante**: Applicata la formula originale $\alpha = \frac{2.5}{\frac{1}{N} \sum |Q(s_i, a_i)|}$. Tuttavia, l'Alpha non moltiplica la BC Penalty, ma normalizza il gradiente del RL: $\mathcal{L}_{actor} = -\alpha \cdot Q(s,a) + \text{BC\_Penalty}$. Questa formulazione è matematicamente invariante al nostro `reward_scale` di 0.002, in quanto la divisione per la magnitudo di Q annulla lo scale factor.
- **Loss di Imitazione Domain-Specific**: Invece del generico MSE su tutto il vettore d'azione, applichiamo una funzione che soppesa doppiamente sterzo e freno e aggiunge una *Mutual Exclusion Penalty* per impedire il blocco dei freni in accelerazione.
## 2. Critic (Twin Q-Network)
Il Critic ha il compito di stimare il valore (Q-value) della coppia (Stato, Azione). Poiché il BC non usa una value-function, il Critic deve essere addestrato da zero.
- **Architettura Twin**: Usa due reti Q indipendenti per mitigare l'Overestimation Bias tipico del Q-learning. Si prende il minimo tra le due stime durante l'aggiornamento dell'Actor.
- **Critic Warm-Up Exteso (15.000 step)**: Poiché nel nostro setup Offline-to-Online il Critic viene inizializzato da zero (a differenza del paper dove è pre-addestrato offline), l'Actor viene congelato per i primi `15.000` step. Questo permette al Critic di apprendere una Value Function solida e previene la *Critic Warmup Degradation*, ovvero la distruzione dei pesi BC perfetti da parte di gradienti casuali o sproporzionati inviati da un Critic immaturo.

## 3. Parametri TD3+BC e Il Sistema di Loss (Anti-Drift)
L'integrazione di una BC Penalty in un algoritmo TD3 richiede una calibrazione millimetrica per bilanciare l'imitazione dell'esperto e la massimizzazione del Reward.

- **Equazione Actor Loss (TD3+BC)**: $\mathcal{L}_{actor} = - \alpha \cdot Q(s,a) + \text{BC\_Penalty}(a, a_{expert})$.
- L'Actor viene costretto a massimizzare il Q-Value (derivato dal RL) **senza** abbandonare la traccia dei dati estratti dal Behavioral Cloning.
- **Dynamic Alpha Normalization**: Il coefficiente $\alpha$ viene calcolato dinamicamente come $\frac{0.1}{\frac{1}{N} \sum |Q|}$. Il parametro originale $\lambda=2.5$ è stato ridotto a `0.1` per depotenziare la forza del gradiente RL e proteggere la *BC\_Penalty* (che ha gradienti deboli) durante l'apprendimento esplorativo. Essendo applicato al termine Q, rende il gradiente RL auto-bilanciante e totalmente invariato a eventuali *reward scaling*.

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
> Il Critic valuta il Q-Value con un discount factor `gamma = 0.999`. Il valore massimo stimabile per una guida perfetta e infinita è una serie geometrica: `Q_max = 1.0 / (1 - 0.999) = 1000.0`.
> Se l'auto va fuori strada, il flag `done=True` "brucia" del tutto l'aspettativa di vita (+1000.0) e impone il limite terminale di `-10.0`.
> La perdita reale percepita dalla rete neurale per quell'errore è quindi un **differenziale di -1010 punti** su una scala di 1000 (prima del Reward Scaling). 
> Se impostassimo la penalità a `-1000`, la Mean Squared Error del Critic impazzirebbe, causando esplosione dei gradienti e *Catastrophic Forgetting*. La penalità di -10 è letale per l'agente, ma "sicura" per i gradienti.

### C. Bilanciamento Matematico (Gamma, Reward Scale, Alpha)
L'integrazione di una BC Penalty in un algoritmo RL ad alta frequenza (50Hz) richiede una calibrazione millimetrica per evitare che una forza matematica sopprima l'altra.
- **Gamma = 0.999 (Orizzonte Lungo)**: Aumentato dallo standard `0.99` per estendere la visione del Critic a 1000 step (20 secondi). Senza questo orizzonte lungo, l'agente non "vedeva" in tempo le curve ad alta velocità.
- **Reward Scale = 0.002**: L'aumento del Gamma decuplica la magnitudo dei Q-Values. Riducendo lo scaling si compensa l'effetto e si evitano gradienti esplosivi.
- **Dynamic Alpha Normalization**: Il gradiente RL è bilanciato dalla formula $\alpha = \frac{\lambda}{\frac{1}{N} \sum |Q|}$. Nel paper originale si usa $\lambda = 2.5$, ma noi lo abbiamo ridotto a **`0.1`**. Questa drastica riduzione è essenziale nell'Offline-to-Online fine-tuning per impedire che l'Actor riceva gradienti RL 25 volte più forti della `BC_Penalty`, prevenendo il collasso immediato della policy di imitazione. L'invarianza allo scale è comunque mantenuta.
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
Durante il training BC, ogni mini-batch viene perturbato sinteticamente per simulare stati off-distribution:
- **Perturbazione Laterale**: `trackPos` viene spostato di ±0.4 (40% della larghezza della pista). La rete impara a correggere lo sterzo proporzionalmente allo spostamento (gain = 0.25).
- **Perturbazione Angolare**: `angle` viene perturbato di ±0.08 rad (~4.5°). La rete impara a raddrizzare l'auto quando è disallineata rispetto alla pista (gain = 1.5).
- **Perturbazione dei Sensori**: I 19 sensori di distanza dalla pista vengono ricalcolati geometricamente in base alla nuova posizione/angolo simulata, mantenendo la coerenza fisica.
- **Correzione Throttle**: L'acceleratore viene ridotto proporzionalmente alla perturbazione combinata per insegnare cautela in stati anomali.

### Livello 2: Residual RL (td3_bc.py)
Il TD3 esplora naturalmente stati off-distribution e impara correzioni locali tramite i Q-Value del Critic. Grazie all'Alpha Dinamico, le correzioni restano sempre "residuali" — piccoli aggiustamenti alla policy BC senza mai sfuggire al controllo umano, a prescindere dall'entità dei reward scoperti online.

## 8. Multimodal Averaging & Permanent BC Adherence (Residual RL)
Il dataset umano originale del Behavioral Cloning (BC) contiene intrinsecamente traiettorie eterogenee (es. stringere in una curva al giro 1, allargare al giro 2). Quando una rete neurale impara da questi dati minimizzando il Mean Squared Error (MSE), tende ad apprendere la **media matematica** delle manovre. In curve complesse, questo porta spesso al **Multimodal Averaging** (un comportamento indeciso).

Per ovviare a questo problema senza far deragliare l'agente (Extrapolation Error), l'architettura implementa una strategia di **Residual Reinforcement Learning**:
- **Dynamic Alpha Normalization**: Il peso della BC Penalty si calibra dinamicamente sui Q-value del Critic. L'Actor non diventa mai un agente RL puro — rimane un "imitatore guidato" che usa i Q-Value solo come piccole correzioni (Residuals). Questo previene l'oscuramento della loss imitativa (Catastrophic Forgetting) causato dall'aumento naturale dei Q-value nel training prolungato.
- **Learning Rate Mirato**: L'Actor viene addestrato con un Learning Rate standard di `3e-4`, ma agisce solo ed esclusivamente sul `continuous_head`, lasciando il resto della rete congelato per proteggere i pesi calibrati.

## 9. Elite Buffer e Self-Imitation Learning (Episodic Prioritization)
Per mitigare la *Sample Inefficiency* e il *Catastrophic Forgetting* intrinseco nel campionamento casuale uniforme (Uniform Random Sampling), l'architettura sfrutta una strategia di **Self-Imitation Learning** basata su un'architettura a **Doppio Buffer**:
- **Caching Episodico**: Le transizioni non vengono caricate step-by-step, ma raggruppate per episodio.
- **Elite Buffer (Monotonic Threshold)**: Se un episodio supera una soglia di eccellenza, viene clonato in un buffer secondario (`20.000` step). La soglia è rigorosamente legata al record globale assoluto (`best_distance * 0.9`), risultando monotonicamente non decrescente. Questo impedisce alla soglia di abbassarsi per colpa di episodi sub-ottimali e previene l'inquinamento del buffer con dati scadenti (avvelenamento dell'Elite Buffer).
- **Iniezione Expert**: I campioni clonati nell'Elite Buffer vengono flaggati con `expert=1.0`. Questo "inganna" la `bc_penalty` dell'Actor, forzando la rete a trattare i propri record come se fossero dimostrazioni umane ottimali, innescando l'auto-imitazione (Self-Imitation Learning).
- **Hybrid Sampling (Generalization Balance)**: Durante il training, il TD3 estrae il 75% del minibatch dal buffer standard e il 25% dall'Elite Buffer. Sebbene in passato si sia tentato un "Extreme Optimism" (85% Elite), questo portava a un forte **overfitting** sui singoli stati esatti dei record. Poiché la rete aggiunge un rumore Gaussiano esplorativo, l'auto si troverà sempre in stati "sporchi" leggermente diversi dalla traiettoria perfetta. Il 75% di Standard Buffer (con la BC_Penalty ancorata al maestro umano) è vitale per insegnare all'agente a **generalizzare** e recuperare la traiettoria quando si verifica una deviazione stocastica.
- **Isolamento Dati**: Per mantenere pulita la directory dei checkpoint, entrambi i buffer (principale e elite) vengono serializzati in formato `.npz` e memorizzati in una sottocartella dedicata `train_set/checkpoints/buffers/`.

## 10. Prevenzione del Collasso (Frozen BC Anchor e Causal Confusion)
Durante l'addestramento ibrido, l'architettura risolve due problematiche critiche intrinseche al Self-Imitation Learning:

1. **Frozen BC Anchor (Prevenzione Extrapolation Error)**: 
   Nel buffer standard (75% del batch esplorativo), i gradienti RL puri possono degenerare se il Critic si riempie di Q-Value negativi, portando l'Actor a manovre suicide. 
   L'agente istanzia un **Frozen BC Anchor** (`self.bc_policy`), una copia congelata della rete BC.
   La BC Penalty calcola l'MSE tra l'azione umana e l'azione deterministica `torch.tanh(mean)`, con componente direzionale normalizzata e Mutual Exclusion Penalty bilanciata (Soft Shaping).
   La BC Penalty viene ora calibrata tramite la Normalizzazione Dinamica dell'Alpha, garantendo una regolarizzazione proporzionata e permanente.

2. **Terminal State Mimicry (Sgancio Pre-Schianto)**: 
   Quando un episodio record (salvato nell'Elite Buffer) termina con uno schianto, le ultime azioni sono la causa diretta del fallimento. Forzare l'Actor a imitarle (tramite Self-Imitation) indurrebbe una *Causal Confusion*. 
   Il sistema risolve questo paradosso azzerando la maschera di imitazione (`expert=0.0`) negli ultimi 50 step (esattamente 1 secondo a 50Hz) di un record schiantato. In quella "finestra di evasione", l'agente smette di imitare il suo vecchio errore e torna istantaneamente sotto l'influenza del Reinforcement Learning puro e del Frozen BC Anchor, riuscendo così a frenare e a sopravvivere per estendere ulteriormente il record.

## 11. Evaluation Periodica Deterministica
Ogni 5 episodi di training, il sistema esegue automaticamente un **episodio di valutazione deterministica** (`evaluate=True`, zero rumore). Se la distanza percorsa o il tempo sul giro migliorano, il checkpoint viene salvato come `td3_best_eval.pth`. Questo garantisce che il checkpoint usato per la presentazione video sia sempre la policy migliore *riproducibile* — non quella del miglior episodio esplorativo (che potrebbe essere un outlier fortunato con rumore stocastico).

## 12. Offline RL Warm-Start (Safe Restart)
Durante le lunghe sessioni di RL, il Critic può saturarsi irrimediabilmente di Q-Value negativi a causa della continua esplorazione stocastica.

Per risolvere questo stallo, l'architettura implementa un caricamento **disaccoppiato** tra i pesi neurali e i Replay Buffer:
- I file `.npz` vengono caricati in memoria **indipendentemente** dall'esistenza di un checkpoint valido.
- Questo consente il **Safe Restart**: cancellare i pesi della rete TD3 (`.pth`), ripartendo con un Actor immacolato (clonato dal BC) e un Critic a zero, ma fornendo un Elite Buffer già popolato.

## 13. Guida all'Interpretazione dei Log di Training
Durante l'esecuzione di `td3_bc.py`, l'analisi dei log è fondamentale per comprendere la salute del sistema e il corretto funzionamento delle dinamiche ibride implementate:

- **CriticL microscopico (`0.000` - `0.004`)**: L'errore del Critic appare irrisorio a causa del forte *Reward Scaling* (`0.002`). Poiché i Q-Value sono numericamente compressi in partenza, il loro Errore Quadratico Medio (MSE) in fase di apprendimento scende spesso sotto la soglia del millesimo, venendo arrotondato a `0.000` in console. Questo è il segno di un Critic sano: gradienti così piccoli evitano la *Gradient Explosion* e proteggono la rete dal collasso. Quando compare uno `0.001`, significa semplicemente che il Critic sta affinando una precisione estrema.
- **ActorL positivo in fase di avvio**: Durante i primi episodi, i Q-Value sono piccoli. La formula dell'*Alpha Dinamico* ($2.5 / |Q|$) compensa questa piccolezza generando un peso enorme per la *BC Penalty*. Finché il Critic è "acerbo", l'ActorL si mantiene alta e positiva, forzando l'Actor a ignorare le scorribande stocastiche per ancorarsi rigidamente alla traiettoria umana.
- **ActorL progressivamente negativo**: Con il procedere del training, i Q-Value calcolati dal Critic crescono in magnitudo (l'agente massimizza i punti per la velocità). L'Alpha decresce fisiologicamente per via del denominatore più grande, e la componente RL pura (che minimizza $-Q$) domina l'equazione. Una loss a `-0.600` indica che la rete ha allentato le redini imitative per abbracciare l'ottimizzazione del tempo sul giro.
- **Spike della CriticL associato all'Elite Buffer**: Quando l'agente stabilisce una traiettoria record prolungata (es. una corsa da `2.400m`), questa viene iniettata nell'Elite Buffer. Nelle iterazioni successive, il *Self-Imitation Learning* espone il Critic a questa traiettoria inedita e iper-performante: questo spiazza le vecchie credenze del Critic, provocando un leggero picco temporaneo nella CriticL (es. a `0.004`). Subito dopo l'assimilazione, l'ActorL sprofonda per allinearsi al nuovo record e le distanze dell'agente subiscono un forte balzo in avanti.
