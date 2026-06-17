"""
Behavioral Cloning (Imitation Learning) — TORCS Giro Secco

Addestra una PolicyNetwork su spazi di dimensione continua sulle dimostrazioni umane (HDF5).

Features:
  - Supporto multi-file: accetta sia un singolo .h5 sia una directory di lap_[0-9]*.h5 (solo giri completi)
  - Supporto hardware per CPU e GPU tramite pytorch per un training più veloce su sistemi con GPU 
  - Split del dataset in train e validation set (80/20) con Early Stopping per evitare overfitting
  - Il Cosine LR scheduler modifica dinamicamente il learning rate durante il training per una migliore convergenza del modello.
    Grazie a questa feature il learning rate varia seguendo l'andamento del coseno, se avessimo usato altri tipi di scheduler
    il modello avrebbe potuto convergere più lentamente o non convergere affatto alla fine del training. In questo modo il learning rate è
    molto alto all'inizio, si riduce gradualmente fino a diventare il valore minimo impostato ovvero 1e-6.
  
  - Bojarski-style data augmentation per cercare di mitigare il covariate shift tra training e inferenza, è impostato per perturbare nei seguenti modi:
      - Perturbazione laterale: aggiunge una perturbazione laterale casuale compresa tra -0.4 e +0.4 (40% della trackpos), simulando la posizione della vettura in pista.
      - Perturbazione angolare: aggiunge una perturbazione angolare casuale compresa tra -0.08 e +0.08 radianti (~4.5°), simulando l'angolo della vettura rispetto alla pista.
      In questo modo si insegna alla rete a recuperare da stati fuori distribuzione, migliorando la sua capacità di generalizzazione.
      Il valore delle perturbazioni è stato scelto empiricamente (probabilmente modificarli in valori più appropriati permetterebbe di migliorare le performance del bc),
      ma la scarsità della BC è stata corretta dall'RL con il TD3+BC.

NOTA: I dati HDF5 sono già normalizzati da data_collection.flatten_state():
  - track[19]: /200 (via gym_torcs.make_observaton)
  - speedX/Y/Z: /50 (via gym_torcs.make_observaton, default_speed=50)
  - wheelSpinVel[4]: /100 (via data_collection.flatten_state)
  - rpm: /10000 (via data_collection.flatten_state)
  - distFromStart: Raccolta ma rimossa dal dataset perché non volevamo che la rete imparasse a correlare l'azione con la distanza dal traguardo (causando un potenziale train-test mismatch).


Mapping delle azioni prodotte dalla rete neurale e poi inviate a TORCS:

  Indice Torcs | Azione              | Range        | Funzione di attivazione rete
  ----------------------------------------------------------------------------------
  [0]          | sterzata            | [-1, 1]      | Tanh output (perfetta perché come codominio ha [-1, 1], come le azioni registrate)
  [1]          | accelerazione       | [0, 1]       | Sigmoid output (perfetta perché come codominio ha [0, 1], come le azioni registrate)
  [2]          | freno               | [0, 1]       | Sigmoid output (perfetta perché come codominio ha [0, 1], come le azioni registrate)

NOTA: la marcia non viene predetta dalla rete, nel dataset è presente come dato ma viene ignorato. 
Il cambio è affidato allo script gearing.py che si occupa di selezionare la marcia appropriata in base a giri del motore e velocità.
"""

import os
import glob
import argparse
import numpy as np
import h5py #libreria usata per leggere i file h5
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from datetime import datetime

BATCH_SIZE = 256
LR = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Dataset HDF5

class TorcsHDF5Dataset(Dataset):
    """Estende la classe Dataset di PyTorch per caricare dati da un file HDF5 contenente osservazioni e azioni del pilota.
    Feature principali:
     - Frame stacking temporale di tre frame a intervalli regolari k=6 (equivalenti a 0.24s a 50Hz), al modello quindi vengono forniti gli stati (t-12, t-6, t) e lo stato da 29D passa 87D.
       Questa feature è stata pensata per rendere il modello consapevole del moto della vettura, aiutandolo a prevedere la traiettoria futura.
     - Sanity check sui dati all'inizializzazione:
        - Verifica ci siano i gruppi 'states' e 'actions' nei file HDF5.
        - Verifica l'assenza di NaN e Inf nei dati. È stato deciso di implementare questo check perché la presenza di valori non validi nel dataset provocherebbe l'instabilità o il collasso della policy durante l'addestramento.
    """

    def __init__(self, file_path: str):
        super().__init__()

        # Se nel file path specificato non ci sono file HDF5, solleva un errore di tipo FileNotFoundError
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File dataset non trovato: {file_path}")

        self.file_path = file_path

        # Legge il file h5 in modalità read-only nel path specificato e gli da l'alias h5f
        # La chiusura del file è gestita automaticamente dal costrutto 'with'
        with h5py.File(self.file_path, 'r') as h5f:

            # Verifica la presenza dei gruppi 'states' e 'actions' nel file h5
            if 'states' not in h5f:
                raise KeyError(f"Gruppo 'states' mancante in {file_path}")
            if 'actions' not in h5f:
                raise KeyError(f"Gruppo 'actions' mancante in {file_path}")

            states_np = h5f['states'][:]
            actions_np = h5f['actions'][:]

            # Sanity check sui NAN e INF 
            if np.any(np.isnan(states_np)):
                raise ValueError(f"NaN rilevati in 'states' di {file_path}")
            if np.any(np.isinf(states_np)):
                raise ValueError(f"Inf rilevati in 'states' di {file_path}")
            if np.any(np.isnan(actions_np)):
                raise ValueError(f"NaN rilevati in 'actions' di {file_path}")
            if np.any(np.isinf(actions_np)):
                raise ValueError(f"Inf rilevati in 'actions' di {file_path}")

            # Crea i tensor di torch partendo dagli array numpy appena letti dall'h5
            # I tensor sono delle strutture dati di pytorch, simili agli array, possono avere molte dimensioni
            # e sono ottimizzati per lavorare con la GPU, inoltre tengono traccia di ogni operazione effettuata su di essi
            # consentendo un calcolo automatico dei gradienti (autograd)
            self.states = torch.tensor(states_np, dtype=torch.float32)
            self.actions = torch.tensor(actions_np, dtype=torch.float32)

        # Calcola la lunghezza del dataset
        self.length = self.states.shape[0]

    # Restituisce la lunghezza del dataset
    def __len__(self) -> int:
        return self.length

    # Restituisce il frame-stacking di tre frame (t-12, t-6, t) e la corrispondente azione target
    def __getitem__(self, idx: int):
        k = 6
        idx_t6 = max(0, idx - k)
        idx_t12 = max(0, idx - 2 * k)

        stacked = torch.cat([
            self.states[idx_t12],
            self.states[idx_t6],
            self.states[idx]
        ])
        return stacked, self.actions[idx]


def load_dataset(path: str) -> Dataset:
    """Carica l'intero dataset di giri completi dai file h5 e li unisce in un unico dataset."""
    # Se il path è una directory
    if os.path.isdir(path):
        
        # Cerca tutti i file h5 nel percorso specificato, li ordina e restituisce solo quelli che contengono i giri completi
        # (lap_[numero].h5). Escludendo i segmenti di curve, mancanti quindi di alcune parti. 
        # Questo perché la BC cerca esclusivamente di minimizzare l'errore medio tra l'azione predetta e l'azione del pilota umano
        # e la presenza di segmenti con solo curve e incompleti dell'intero giro sbilancerebbe notevolmente il dataset, causando un 
        # deterioramento delle performance della BC.
        h5_files = sorted(glob.glob(os.path.join(path, "**/lap_[0-9]*.h5"), recursive=True))

        # Se non sono stati trovati file h5, solleva un errore di tipo FileNotFoundError
        if not h5_files:
            raise FileNotFoundError(
                f"Nessun file lap_[0-9]*.h5 (giro intero) trovato in {path} o nelle sue sottocartelle"
            )

        # Informa l'utente di quanti file h5 sono stati trovati e che verranno caricati
        print(f"Trovati {len(h5_files)} file HDF5. Carico l'intero dataset...")
        
        datasets = [] # Lista di dataset, uno per ogni file h5
        total_samples = 0 # Conteggio totale dei campioni
        
        # Per ogni file h5 trovato, crea un dataset e aggiungilo alla lista. In caso di errore, informa l'utente con un warning.
        for f in h5_files:
            try:
                ds = TorcsHDF5Dataset(f)
                datasets.append(ds)
                total_samples += len(ds)
            except Exception as e:
                print(f"Warning: Impossibile leggere {f}: {e}")

        # Check per verificare che almeno un dataset sia stato caricato con successo        
        if not datasets:
            raise ValueError("Nessun dataset valido trovato.")
        # Stampa un riepilogo del dataset caricato    
        print(f"Dataset caricato: {len(datasets)} giri, {total_samples} campioni totali.")
        
        # Restituisce il dataset concatenato e il numero totale di campioni
        return ConcatDataset(datasets), total_samples
    
    # Se il path è un file h5 carica solo quello e restituiscilo
    else: 
        ds = TorcsHDF5Dataset(path)
        return ds, len(ds)


class PolicyNetwork(nn.Module):
    """
    Classe che definisce la rete neurale a 87D stacked di 3 frame da 29D, estende la classe nn.Module
    di pytorch che fornisce molte funzionalità utili per la costruzione di reti neurali.

    La rete ha una hidden size di 512 neuroni per strato, quattro strati nascosti e layer normalization.
    Queste sono state scelte per dare alla rete una grande capacità di apprendimento e per evitare il collasso della policy.

    Riassunto della struttura della rete:
    - Input: 87D (3 frame stacked da 29D ciascuno)
  
    - Quattro strati (profondità 4 della rete) nascosti costituiscono la backbone della rete neurale essi agiscono in sequenza:
        - Linear (87 -> 512) -> Layer Norm -> ReLU
            In questo strato i dati dei sensori vengono elaborati per la prima volta e la rete produce un output a 512 dimensioni
            (ogni neurone processa questi dati e ne ricava un output), questo output viene passato al layer di normalizzazione,
            che si occupa di normalizzare i dati, calcolando media e varianza per ogni feature e facendo in modo che abbiano tutti media 0 e deviazione standard 1 (per evitare la dominanza di alcune feature),
            stabilizzando così il processo di addestramento. L'output passa poi al layer ReLU, che applica la funzione di attivazione ReLU (Rectified Linear Unit),
            che introduce non linearità nel modello. 

            Tale funzione opera nel seguente modo:  
            - Se l'input è positivo, restituisce l'input stesso
            - Se l'input è negativo, restituisce 0

            In questo modo la rete non si attiva sempre, ma solo quando l'input è positivo. Permettendogli di apprendere pattern che non sempre
            usano tutte le feature disponibili.

        - Linear (512 -> 512) -> Layer Norm -> ReLU
            Questo strato è identico al precedente, aumentando la profondità della rete e permettendo di apprendere relazioni più complesse.
        
        - Linear (512 -> 512) -> Layer Norm -> ReLU
            Questo strato è identico ai precedenti, aumentando ulteriormente la profondità della rete.
        
        - Linear (512 -> 512) -> Layer Norm -> ReLU
            Questo strato è identico ai precedenti, aumentando ancora di più la profondità della rete.
  
    - Strato di Output (Continuous Head), restituisce valori in un intervallo continuo: 
        - Linear (512 -> 3D) con funzioni di attivazione:
            - Sterzata (Indice [0]) -> Tanh (codominio [-1, 1])
            - Accelerazione (Indice [1]) -> Sigmoid (codominio [0, 1])
            - Freno (Indice [2]) -> Sigmoid (codominio [0, 1])

    Il numero di layer e di neuroni per layer sono state scelte in base ai riferimenti trovati in letteratura, in particolare sono state prese come 
    base i paper:
        - A minimalist approach to offline reinforcement learning. Fujimoto & Gu, Google Research Brain team(2021)
        - Improving TD3-BC: Relaxed Policy Constraint for Offline Learning and Stable Online Fine-Tuning. Beeson & Montana, University of Warwick, Alan Turing Institute (2022)

    Valori troppo bassi di neuroni o layer porterebbero ad un Underfitting, dove la rete non riuscirebbe a catturare le relazioni complesse tra gli input e gli output, mentre valori troppo alti porterebbero ad un Overfitting, dove la rete imparerebbe a memoria il dataset di training senza generalizzare a nuovi dati.
    """

    def __init__(self, state_dim: int = 87, hidden_size: int = 512):
        super(PolicyNetwork, self).__init__()

        # Backbone della rete neurale (4 strati nascosti)
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )

        # Continuous Head per l'output della rete neurale
        self.continuous_head = nn.Linear(hidden_size, 3)

    # Funzione di forward pass, prende in input lo stato e restituisce l'azione predetta
    # Restituisce di fatto l'output della rete neurale
    def forward(self, state: torch.Tensor):
        # Variabile che contiene le features estratte dal backbone
        features = self.backbone(state)
        #La funzione backbone si occupa di eseguire tutto il processo che poi fornisce in output le feature estratte
        
        
        # Variabile che contiene l'output del continuous head
        cont_out = self.continuous_head(features)
        #Continous head si occupa di fornire in output l'azione predetta 
        
        # Applichiamo le activation function per ogni output
        steer = torch.tanh(cont_out[:, 0:1])          # [-1, 1]
        accel_brake = torch.sigmoid(cont_out[:, 1:3])   # [0, 1]
        
        # Riuniamo l'output del continuous head in un unico tensore di dimensione 3
        continuous = torch.cat([steer, accel_brake], dim=1) # 3D: [steer, accel, brake]

        return continuous


# ──────────────────────────────────────────────────────────────────────
#  Trainer
# ──────────────────────────────────────────────────────────────────────

class BehaviorCloningTrainer:
    """Addestra la PolicyNetwork

    Funzionalità:
        - Loss continua pesata per sterzo, acceleratore e freno:
            - Viene utilizzato l'errore quadratico medio (MSE) pesato per ogni azione.
            - I pesi sono assegnati per dare più importanza ad azioni critiche e meno frequenti:
                - Sterzo: peso base 1.0 (con un boost di 3x in curva per migliorare la traiettoria). La curva viene rilevata se la rotazione dello sterzo è superiore a STEER_CURVE_THRESHOLD ora impostata a 0.10 (10% del massimo della rotazione).
                    Tale iperparametro probabilmente andrà aumentato per migliorare la BC e adattato al circuito. 
                - Acceleratore: peso base 1.0.
                - Freno: peso base 5.0 (con un boost dinamico del 25x quando il pilota frena, per costringere la rete ad apprendere le staccate).

      - Validation split 80/20 con Early Stopping configurabile
            - Il dataset viene diviso in due parti: 80% per il training e 20% per la validazione
            - L'early stopping impedisce l'overfitting fermando l'allenamento quando la validation loss smette di migliorare (evita di sprecare risorse computazionali)
      
      - Cosine Annealing LR scheduler
        - Il Learning Rate (LR) viene ridotto gradualmente durante l'allenamento seguendo una curva coseno
      
      - Bojarski-style Data augmentation con rumore gaussiano strutturato sugli stati
        - Viene aggiunto un rumore gaussiano agli stati per aumentare la robustezza del modello
      
      - Salvataggio dei pesi del modello basato sulla migliore validation loss
        - Durante le varie epoche di allenamento, la rete neurale viene salvata solo qualora il suo punteggio sulla validation loss sia migliore rispetto ai precedenti.
    """

    STEER_CURVE_THRESHOLD = 0.10  # soglia sterzo per curva (nel range [-1,1])
    STEER_BOOST_FACTOR = 3.0      # moltiplicatore dell'errore sullo sterzo in curva
    BRAKE_ACTIVE_THRESHOLD = 0.05  # soglia sopra la quale consideriamo che l'umano stia frenando
    BRAKE_BOOST_FACTOR = 25.0      # moltiplicatore dell'errore sul freno quando attivo

    # I default globali vengono sovrascritti dagli argomenti CLI passati da train_bc.sh/main().

    def __init__(self, model: nn.Module, dataset: Dataset,
                 batch_size: int = BATCH_SIZE, lr: float = LR, device: str = DEVICE,
                 state_mean=None, state_std=None):

        ## Controllo del device e spostamento del modello su GPU se disponibile
        self.device = torch.device(device)
        self.model = model.to(self.device)
        print(f"  Modello spostato su: {self.device}")


        # Preparazione dei parametri di normalizzazione (mean e std), la normalizzazione verrà applicata dopo l'augmentation e prima del forward pass alla rete
        # I parametri verranno salvati in state_norm.npz e riutilizzati se già presenti
        # Tale approccio è stato convalidato da Fujimoto & Gu 2021, che hanno dimostrato che normalizzare ciascuna feature
        # usando le statistiche globali (media e deviazione standard) calcolate preventivamente sull'intero dataset
        # stabilizza l'addestramento dell'offline RL e ne migliora le prestazioni.
        if state_mean is not None:
            self.state_mean = torch.tensor(state_mean, dtype=torch.float32, device=self.device)
            self.state_std = torch.tensor(state_std, dtype=torch.float32, device=self.device)
        else:
            self.state_mean, self.state_std = None, None


        # Ottimizzatore Adam (Adaptive Moment Estimation) con learning rate lr e weight decay 1e-5
        # Adam è uno standard negli addestramenti di reti neurali, decide cose e quanto modificare il peso dei neuroni durante l'apprendimento.
        # weight_decay è un parametro che serve a penalizzare i pesi troppo grandi del modello, evitando l'overfitting.
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=1e-5
        )

        # Split del dataset (con seed fisso per la riproducibilità) in training e validation set (80% training, 20% validation)
        total = len(dataset)
        val_size = max(1, int(total * 0.2))
        train_size = total - val_size
        self.train_dataset, self.val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        # Un DataLoader è un iteratore che permette di scorrere il dataset
        # shuffle=True fa sì che i dati vengano mescolati ad ogni epoca (evita che l'agente impari i dati in ordine)
        # num_workers=2 fa sì che i dati vengano caricati in parallelo (accelera l'addestramento)
        # pin_memory se true fa sì che i dati vengano copiati nella memoria della GPU (accelera l'addestramento)
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=2, pin_memory=(device != "cpu")
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=batch_size,
            shuffle=False, num_workers=2, pin_memory=(device != "cpu")
        )

        self.best_val_loss = float('inf')

        print(f"Dataset split: {train_size} train / {val_size} val")


    # Funzione che calcola l'errore commesso dalla rete rispetto alle azioni del pilota umano.
    # È chiamata combined perché combina più errori (mse per steer, accel, brake) in un solo valore finale
    # pred_continuous: azioni predette dalla rete neurale
    # target_actions: azioni del pilota umano
    def _combined_loss(self, pred_continuous, target_actions):
    
        targets_cont = target_actions[:, 0:3]
        sq_error = (pred_continuous - targets_cont) ** 2

        # Pesi per canale continuo: [steer, accel, brake]
        channel_weights = torch.tensor([1.0, 1.0, 5.0], device=pred_continuous.device)

        # Boost freno dinamico se l'umano frena
        brake_target = targets_cont[:, 2]
        is_braking = (brake_target > self.BRAKE_ACTIVE_THRESHOLD).float()
        brake_boost = 1.0 + (self.BRAKE_BOOST_FACTOR - 1.0) * is_braking

        # Boost sterzo in curva
        steer_target = targets_cont[:, 0].abs()
        is_curve = (steer_target > self.STEER_CURVE_THRESHOLD).float()
        steer_boost = 1.0 + (self.STEER_BOOST_FACTOR - 1.0) * is_curve

        weighted_sq = sq_error * channel_weights.unsqueeze(0)
        weighted_sq[:, 0] = weighted_sq[:, 0] * steer_boost
        weighted_sq[:, 2] = weighted_sq[:, 2] * brake_boost
        return weighted_sq.mean()

    # Funzione che definisce un'epoca di addestramento, ad ogni epoca i pesi della rete vengono modificati per minimizzare l'errore.
    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0

        for states, targets in self.train_loader:
            states = states.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # Reshape temporaneo per applicare l'augmentation su ciascuno dei 3 frame in modo coerente
            batch_size = states.size(0)
            states = states.view(batch_size, 3, 29)

            # Data Augmentation Bojarski-Style
            delta_pos = torch.randn(batch_size, device=states.device) * 0.20
            delta_pos = torch.clamp(delta_pos, -0.40, 0.40)

            # Perturbazione Angolare (angle)
            delta_angle = torch.randn(batch_size, device=states.device) * 0.04
            delta_angle = torch.clamp(delta_angle, -0.08, 0.08)

            # Gating dell'augmentation (50%)
            # Perturbare ogni batch significa che l'agente vedrà ad ogni passo una traiettoria leggermente diversa
            # ma rischia di non vedere la traiettoria corretta e ideale, quindi applichiamo l'augmentation solo su il 50% dei campioni.
            # A scegliere dove applicare l'augmentation e dove no è il termine aug_mask, viene scelto un numero casuale tra 0 e 1,
            # se il numero è inferiore a 0.5 la maschera vale 1 (applichiamo l'augmentation), altrimenti vale 0 (non applichiamo l'augmentation).  
            aug_mask = (torch.rand(batch_size, device=states.device) < 0.5).float()
            delta_pos = delta_pos * aug_mask
            delta_angle = delta_angle * aug_mask

            # Per ognuno dei 3 frame, viene calcolata la perturbazione geometrica coerente con l'augmentation
            # Dato che i frame sono stacked nel tempo ad un certo punto si sovrapporranno e la perturbazione si accumula
            # Questo non è un problema se i parametri dell'augmentation sono ben tarati per il nostro applicativo. 
            for f_idx in range(3):
                frame_states = states[:, f_idx, :]
                
                # Estrazione dei valori dei sensori di pista e angle
                angle = frame_states[:, 0]
                # Moltiplicati per 200 per riportarli nella scala dei metri
                L_0 = frame_states[:, 1] * 200.0  # Sensore -45 gradi
                L_18 = frame_states[:, 19] * 200.0  # Sensore 45 gradi
                
                # Calcolo geometrico dinamico della semi-larghezza della pista
                W_L = L_18 * torch.sin(angle + 0.785398) # 45 gradi = 0.785398 rad
                W_R = L_0 * torch.sin(0.785398 - angle)
                W_half = torch.clamp((W_L + W_R) / 2.0, 4.0, 10.0) # clamping tra 4m e 10m
                
                # Spostamento laterale fisico in metri (scalato del 50% per correzione più leggera)
                dy = delta_pos * W_half * 0.5
                
                # Perturbazione trackPos (indice 20)
                frame_states[:, 20] = frame_states[:, 20] + delta_pos
                
                # Perturbazione angolare (indice 0)
                # L'angle è già in radianti — aggiungiamo la perturbazione direttamente
                frame_states[:, 0] = frame_states[:, 0] + delta_angle
                
                # Perturbazione geometricamente coerente dei 19 sensori track (indici 1:20)
                alpha = torch.tensor([
                    -45.0, -19.0, -12.0, -7.0, -4.0, -2.5, -1.7, -1.0, -0.5, 0.0, 
                    0.5, 1.0, 1.7, 2.5, 4.0, 7.0, 12.0, 19.0, 45.0
                ], device=states.device) * 3.14159265 / 180.0
                
                # Angolo assoluto di ciascun raggio (usa l'angle perturbato)
                perturbed_angle = frame_states[:, 0]
                beta = perturbed_angle.unsqueeze(1) + alpha.unsqueeze(0)
                
                # Perturbazione lineare sui 19 raggi (combinata: laterale + angolare)
                dL = - dy.unsqueeze(1) * torch.sin(beta)
                frame_states[:, 1:20] = torch.clamp(frame_states[:, 1:20] + dL / 200.0, 0.0, 1.0)
                
            # Correzione del target di sterzata per compensare le perturbazioni introdotte:
            #   - Il termine 'delta_pos' (con guadagno 0.25) corregge lo sterzo per far rientrare l'auto verso il centro della pista.
            #   - Il termine 'delta_angle' (con guadagno 1.5) agisce sull'orientamento per riallineare l'auto parallelamente alla mezzeria.
            #   - Il target finale dello sterzo viene infine limitato al range fisico [-1.0, 1.0].
            targets[:, 0] = targets[:, 0] - 0.25 * delta_pos - 1.5 * delta_angle
            targets[:, 0] = torch.clamp(targets[:, 0], -1.0, 1.0)
            
            # Regolazione (parzializzazione) dell'acceleratore in base all'entità della perturbazione:
            # - Calcoliamo una perturbazione combinata sommando i moduli dello spostamento laterale e angolare.
            # - Riduciamo proporzionalmente il target dell'acceleratore per insegnare alla rete a rilasciare il gas
            #   quando il veicolo sbanda o è fuori traiettoria, facilitando il recupero di aderenza.
            # - Infine, limitiamo l'acceleratore nel range fisico [0.0, 1.0].   
            combined_perturbation = delta_pos.abs() + delta_angle.abs() * 5.0
            targets[:, 1] = targets[:, 1] * (1.0 - 0.15 * combined_perturbation)
            targets[:, 1] = torch.clamp(targets[:, 1], 0.0, 1.0)


            # Data Augmentation: Simulazione di velocità eccessiva in curva (Overspeed Recovery):
            # - Condizione: Il veicolo viaggia a velocità elevata (> 90 km/h) ed è in prossimità di una curva
            #   (rilevata tramite sterzata del pilota o riduzione del sensore di distanza frontale).
            # - Operazione: Incrementiamo artificialmente la velocità percepita (speedX), forzando una 
            #   proporzionale riduzione dell'acceleratore e un incremento del freno target.
            # - Scopo: Insegnare preventivamente alla policy a rallentare e frenare prima delle curve 
            #   qualora la velocità di ingresso sia superiore al limite di stabilità dinamica.
            if torch.rand(1).item() < 0.5:
                # Estraiamo speedX (indice 21) dall'ultimo frame (de-normalizzato)
                speedX_latest = states[:, 2, 21] * 50.0
                steer_target_abs = targets[:, 0].abs()
                sensor_front_latest = states[:, 2, 10]  # track_s9 (indice 10, cioè 0 gradi)

                is_speed_critical = (speedX_latest > 90.0) & ((steer_target_abs > 0.10) | (sensor_front_latest < 0.60))

                if is_speed_critical.any():
                    # Genera un incremento del 10% - 30% per i campioni critici
                    speed_factor = 0.10 + 0.20 * torch.rand(batch_size, device=states.device)
                    speed_factor = speed_factor * is_speed_critical.float()

                    # Aumentiamo speedX in tutti e 3 i frame
                    for f_idx in range(3):
                        states[:, f_idx, 21] = states[:, f_idx, 21] * (1.0 + speed_factor)

                    # Riduciamo l'accelerazione target
                    targets[:, 1] = targets[:, 1] * (1.0 - 0.7 * speed_factor)
                    targets[:, 1] = torch.clamp(targets[:, 1], 0.0, 1.0)

                    # Aumentiamo il freno target (insegniamo a frenare correttivamente)
                    targets[:, 2] = targets[:, 2] + 0.8 * speed_factor
                    targets[:, 2] = torch.clamp(targets[:, 2], 0.0, 1.0)

            # Standardizzazione dello stato e flattening finale:
            # - La normalizzazione (z-score) viene applicata in questa fase poiché l'augmentation precedente
            #   deve operare sulle grandezze fisiche reali (metri, radianti, km/h).
            # - Ripristiniamo la dimensionalità piatta (87D) richiesta in input dalla rete neurale.
            if self.state_mean is not None:
                states = (states - self.state_mean) / (self.state_std + 1e-3)
            states = states.view(batch_size, 87)

            self.optimizer.zero_grad() #azzera i gradienti del passo precedente
            pred_cont = self.model(states) #passa il batch alla rete neurale per essere processato
            loss = self._combined_loss(pred_cont, targets) #calcola la loss
            loss.backward() #calcola i gradienti
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0) #clippa i gradienti per evitare esplosione del gradiente se superano 1.0
            self.optimizer.step() #aggiorna i pesi della rete neurale

            total_loss += loss.item() #somma la loss per calcolare la media finale

        return total_loss / len(self.train_loader) #ritorna la loss media sul batch

    #Questo metodo viene chiamato per calcolare la loss sulla validation set. 
    #Viene chiamato alla fine di ogni epoch per valutare le performance del modello. 
    @torch.no_grad()    # non serve calcolare i gradienti per la validation
    def validate(self) -> float:
        self.model.eval() #metti il modello in validation mode
        total_loss = 0.0

        for states, targets in self.val_loader: #cicla sul validation set
            states = states.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # Stessa normalizzazione del training (nessuna augmentation in validazione)
            # Normalizzazione e Forward Pass
            if self.state_mean is not None:
                states = states.view(states.size(0), 3, 29)
                states = (states - self.state_mean) / (self.state_std + 1e-3)
                states = states.view(states.size(0), 87)

            pred_cont = self.model(states) #calcola la predizione del modello
            loss = self._combined_loss(pred_cont, targets) #calcola la loss
            total_loss += loss.item() #somma la loss per calcolare la media finale

        return total_loss / len(self.val_loader) #ritorna la loss media sul batch

    # Metodo che coordina il processo di addestramento del modello in BC
    def train(self, max_epochs: int = 200,
              checkpoint_path: str = "train_set/checkpoints/bc_policy.pth",
              patience: int = 50, log_path: str = None):

        # Log su file per tenere traccia del training
        logf = open(log_path, "a", encoding="utf-8") if log_path else None

        # Funzione di utilità per il logging, stampa la riga passata, va a capo e la salva sul file di log
        def _log(line: str):
            print(line)
            if logf:
                logf.write(line + "\n")
                logf.flush()

        # Inizializzazione del training
        try:
            _log(f"\n  Inizio training Behavioral Cloning su {self.device}...")
            _log(f"  Max epochs: {max_epochs} | Early Stopping patience: {patience}\n")

            # Cosine Annealing LR
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max_epochs, eta_min=1e-6
            )

            patience_counter = 0

            # Ciclo principale di addestramento
            for epoch in range(max_epochs):
                train_loss = self.train_epoch() # Lancia l'epoca di addestramento
                val_loss = self.validate() # Valida il risultato sul validation set
                lr = self.optimizer.param_groups[0]['lr'] # prendiamo il learning rate
                scheduler.step() # aggiorna il learning rate

                improved = ""
                if val_loss < self.best_val_loss: # Se la loss sul validation set è migliore della migliore loss precedente
                    self.best_val_loss = val_loss # Aggiorna la migliore loss precedente
                    torch.save(self.model.state_dict(), checkpoint_path) # Salva il checkpoint del modello
                    improved = " saved" # Aggiunge " saved" alla stringa da stampare
                    patience_counter = 0 # Resetta il contatore di patience
                else:
                    patience_counter += 1

                _log(
                    f"  Epoch {epoch+1:03d}/{max_epochs} | "
                    f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                    f"LR: {lr:.2e}{improved}"
                )

                # Early Stopping: se la loss sul validation set non migliora per 'patience' epoche, interrompe il training
                if patience_counter >= patience:
                    _log(f"\n  Early Stopping: nessun miglioramento per {patience} epoche.")
                    break

            _log(f"\n  Training completato. Best val loss: {self.best_val_loss:.6f}")
            _log(f"  Miglior checkpoint: {checkpoint_path}")
            _log(f"  Fine: {datetime.now().isoformat()}")
        finally: # chiude il file di log quando il training è completato o in caso di errore
            if logf:
                logf.close()


# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────

def main():
    # Definisce la funzione che gestisce gli argomenti da riga di comando e li passa al modello per regolarne il comportamento
    parser = argparse.ArgumentParser(
        description="Behavioral Cloning per l'addestramento dell'agente"
    )
    
    # argomento per modificare la directory dei dati di addestramento 
    parser.add_argument(
        "--dataset", type=str, default="train_set/laps",
        help="Path al dataset HDF5 (file singolo, o directory: il BC carica SOLO i giri interi lap_[0-9]*.h5, i segmenti lap_seg_*.h5 sono esclusi)"
    )
    
    # argomento per modificare il numeor di epoche di addestramento
    parser.add_argument("--epochs", type=int, default=300, help="Max epoche")
    
    # argomento per modificare la batch size
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    
    # argomento per modificare il learning rate
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    
    # argomento per modificare il path di output
    parser.add_argument(
        "--output", type=str, default="train_set/checkpoints/bc_policy.pth",
        help="Path di output per i pesi del modello"
    )

    args = parser.parse_args() #legge gli argomenti da riga di comando

    # Controlla se è disponibile una GPU, altrimenti usa la CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'=' * 64}")
    print(f"  BEHAVIORAL CLONING ")
    print(f"  Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Stride Type: static (k=6, 0.24s)")
    print(f"{'=' * 64}\n")

    # Caricamento dataset
    print("  Caricamento dataset...")
    dataset, total_samples = load_dataset(args.dataset)

    # Rileva le dimensioni del dataset
    sample_state, _sample_action = dataset[0]
    state_dim = sample_state.shape[0]
    print(f"  Dimensioni: state={state_dim}, action_dim=4 (steer, accel, brake, gear registrata)")

    # Assicurati che la directory di output esista
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    if os.path.isdir(args.dataset):
        _h5s = sorted(glob.glob(os.path.join(args.dataset, "**/lap_[0-9]*.h5"), recursive=True))
    else:
        _h5s = [args.dataset]
    _all_states = []
    for _f in _h5s:
        try:
            with h5py.File(_f, 'r') as _h:
                _all_states.append(_h['states'][:])
        except Exception:
            pass
    _all_states = np.concatenate(_all_states, axis=0).astype(np.float32)  # (N, 29)
    state_mean = _all_states.mean(axis=0)
    state_std = _all_states.std(axis=0)
    _norm_path = os.path.join(os.path.dirname(args.output) or ".", "state_norm.npz")
    np.savez(_norm_path, mean=state_mean, std=state_std)
    print(f"  Normalizzazione stati salvata: {_norm_path} (mean/std su {len(_all_states)} stati 29D)")

    # Crea la rete neurale
    model = PolicyNetwork(state_dim=state_dim)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parametri totali: {total_params:,}")

    # Crea il trainer della rete neurale
    trainer = BehaviorCloningTrainer(
        model=model,
        dataset=dataset,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        state_mean=state_mean,
        state_std=state_std
    )

    # Crea il log di sessione se non esiste ad un path prestabilito (train_set/session_logs/)
    try:
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(args.output))), "session_logs")
        os.makedirs(log_dir, exist_ok=True)
    except OSError:
        log_dir = os.path.dirname(os.path.abspath(args.output)) or "."
        os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"bc_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=== BEHAVIORAL CLONING TRAINING LOG ===\n")
        f.write(f"Avvio:        {datetime.now().isoformat()}\n")
        f.write(f"Dataset:      {args.dataset} | Campioni: {total_samples} | Device: {device}\n")
        f.write(f"Iperparam:    epochs={args.epochs} batch={args.batch_size} lr={args.lr} state_dim={state_dim}\n")
        f.write(f"Output:       {args.output}\n")
    print(f"  Log di sessione: {log_path}")

    # Avvia l'addestramento
    trainer.train(max_epochs=args.epochs, checkpoint_path=args.output, patience=100, log_path=log_path)

    print("\n  Addestramento Behavioral Cloning completato.")
    print(f"  Pesi salvati in: {args.output}\n")

if __name__ == "__main__":
    main()
