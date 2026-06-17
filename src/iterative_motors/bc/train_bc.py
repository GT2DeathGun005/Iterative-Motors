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
import sys
import glob
import argparse
import numpy as np
import h5py #libreria usata per leggere i file h5
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from datetime import datetime

# Iterative Motors: rete della BC condivisa dal package.
_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
from iterative_motors.models.networks import PolicyNetwork
from iterative_motors.data.hdf5_dataset import TorcsHDF5Dataset, load_dataset
from iterative_motors.bc.augmentation import AugmentConfig, augment_batch

BATCH_SIZE = 256
LR = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"






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

    # Pesi della loss rivisti (Iterative Motors): il vecchio freno (base 5 × boost 25 = fino a
    # 125× lo sterzo) rendeva la loss quasi un solo regressore di frenata, peggiorando la
    # precisione di sterzo. Ridotti a un picco ~24× (base 3 × boost 8); più enfasi in curva.
    STEER_CURVE_THRESHOLD = 0.07  # soglia sterzo per curva (nel range [-1,1])
    STEER_BOOST_FACTOR = 4.0      # moltiplicatore dell'errore sullo sterzo in curva
    BRAKE_ACTIVE_THRESHOLD = 0.05  # soglia sopra la quale consideriamo che l'umano stia frenando
    BRAKE_BOOST_FACTOR = 8.0       # moltiplicatore dell'errore sul freno quando attivo

    # I default globali vengono sovrascritti dagli argomenti CLI passati da main().

    def __init__(self, model: nn.Module, dataset: Dataset,
                 batch_size: int = BATCH_SIZE, lr: float = LR, device: str = DEVICE,
                 state_mean=None, state_std=None, aug_cfg: AugmentConfig = None):

        ## Controllo del device e spostamento del modello su GPU se disponibile
        self.device = torch.device(device)
        self.model = model.to(self.device)
        # Configurazione della data augmentation Bojarski-style (default = AugmentConfig()).
        self.aug_cfg = aug_cfg or AugmentConfig()
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

        # Pesi per canale continuo: [steer, accel, brake] (freno base ridotto 5 -> 3)
        channel_weights = torch.tensor([1.0, 1.0, 3.0], device=pred_continuous.device)

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

            # Data augmentation Bojarski-style (parametri configurabili in AugmentConfig).
            states, targets = augment_batch(states, targets, self.aug_cfg)

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
        help="Path di output per i pesi del modello (state_norm.npz viene salvato nella stessa cartella)"
    )
    parser.add_argument(
        "--auto_laps", type=str, default=None,
        help="Directory opzionale di giri auto-raccolti dalla TD3 (lap_*.h5) da unire al dataset "
             "umano per l'arricchimento (flywheel dati). La normalizzazione viene ricalcolata sull'unione."
    )

    args = parser.parse_args() #legge gli argomenti da riga di comando
    auto_dirs = [args.auto_laps] if args.auto_laps else []

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

    # Caricamento dataset (umano + eventuali giri auto-raccolti per l'arricchimento)
    print("  Caricamento dataset...")
    dataset, total_samples = load_dataset(args.dataset, extra_dirs=auto_dirs)

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
    # Includi anche i giri auto-raccolti nel calcolo della normalizzazione (coerenza con il dataset).
    for _ad in auto_dirs:
        if _ad and os.path.isdir(_ad):
            _h5s.extend(sorted(glob.glob(os.path.join(_ad, "**/lap_*.h5"), recursive=True)))
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

    # Avvia l'addestramento
    trainer.train(max_epochs=args.epochs, checkpoint_path=args.output, patience=100)

    print("\n  Addestramento Behavioral Cloning completato.")
    print(f"  Pesi salvati in: {args.output}\n")

if __name__ == "__main__":
    main()
