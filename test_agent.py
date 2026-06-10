"""
Script per il testing deterministico delle performance di guida autonoma dell'agente su TORCS.

Questo script ha lo scopo di caricare un modello addestrato (in formato Behavioral Cloning o
TD3+BC), connettersi al simulatore TORCS tramite il wrapper Gym, e far guidare l'agente.
Il ciclo di test viene eseguito in modalità deterministica per valutare le prestazioni reali,
senza esplorazione stocastica, permettendo di quantificare metriche quali tempo sul giro,
fuoripista o stalli.

Meccanismi chiave implementati per garantire un test deterministico ed affidabile:
  1. Stato di Valutazione (model.eval()): mette il modello in modalità inferenza. L'Actor non usa Dropout;
     LayerNorm è deterministica, ma la modalità eval mantiene il percorso coerente con il testing.
  2. Rimozione del Rumore: L'azione è determinata in modalità pura tanh(mean) escludendo
     qualsiasi rumore di esplorazione gaussiana o Ornstein-Uhlenbeck usato in fase di addestramento.
  3. Cambio Marcia Deterministico: La selezione della marcia è delegata interamente al modulo
     'gearing.py', basandosi su velocità in km/h, giri motore (RPM) e livello di acceleratore applicato.
  4. Rilancio Fisico dell'Ambiente: Ad ogni tentativo di giro, l'ambiente TORCS viene resettato con
     relaunch=True per forzare la ricarica completa del simulatore, ripulendo lo stato del motore fisico.

Gerarchia di auto-rilevamento dei pesi (in assenza di argomento esplicito --weights):
  1. td3_det_best_lap.pth      -> Miglior giro valido deterministico registrato (candidato per la submission).
  2. td3_det_best_dist.pth     -> Policy RL deterministica con maggior distanza percorsa cumulativa.
  3. td3_det_best_dist_run.pth -> Miglior checkpoint deterministico della sessione di training corrente.
  4. td3_expl_best_lap.pth     -> Miglior tempo sul giro ottenuto durante le fasi esplorative di RL.
  5. td3_expl_best_dist.pth    -> Massimo record di distanza ottenuto durante le fasi esplorative di RL.
  6. td3_policy.pth            -> Ultima policy salvata al termine dei passi di addestramento RL.
  7. bc_policy.pth             -> Modello iniziale addestrato solo tramite Behavioral Cloning (supervisionato).

Esempi di utilizzo:
  python test_agent.py                                           # Trova ed esegue il miglior checkpoint in automatico
  python test_agent.py --weights train_set/checkpoints/bc_policy.pth  # Forza il caricamento dei pesi BC
  python test_agent.py --laps 5                                  # Imposta il test per 5 giri completi
"""

import os
import sys
import argparse
import csv
import numpy as np
import torch
import torch.nn as nn
from collections import deque

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'gym_torcs')))

from gym_torcs import TorcsEnv
from gearing import compute_gear  # cambio marcia deterministico (anti-hunting), condiviso col training

# Riproducibilità
# Costringe la GPU a effettuare calcoli deterministici (piu lenti)
torch.backends.cudnn.deterministic = True

# Impedisce che la gpu scelga l'algoritmo migliore ma con possibilità di cambiarlo ogni volta (non deterministico)
torch.backends.cudnn.benchmark = False 

# PolicyActor — Rete continua BC/TD3+BC
class PolicyActor(nn.Module):
    """
    Definisce l'architettura neurale dell'agente (l'Actor) condivisa tra BC e TD3+BC.
    
    La rete accetta in ingresso uno stato a 87 dimensioni (composto da 3 frame a 29 dimensioni concatenati)
    e produce 3 comandi continui: sterzo, acceleratore e freno.
    
    L'architettura è composta da:
      - Un backbone di 4 livelli lineari (Fully-Connected) da 512 unità ciascuno.
      - Strati di Layer Normalization (LayerNorm) e attivazioni ReLU dopo ogni livello lineare
        per prevenire l'esplosione o l'annullamento del gradiente.
      - Una testa di uscita lineare ('continuous_head') a 3 dimensioni per mappare i comandi finali.
    """

    def __init__(self, state_dim: int = 87, hidden_size: int = 512):
        super(PolicyActor, self).__init__()

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

        # Testa lineare a 3 canali continui: [sterzo, acceleratore, freno]
        self.continuous_head = nn.Linear(hidden_size, 3)

    def forward(self, state: torch.Tensor):
        """
        Esegue l'inferenza usando l'attivazione classica del Behavioral Cloning (BC).
        
        Questa funzione viene invocata quando l'agente utilizza pesi addestrati tramite BC.
        Come funziona:
          - Lo sterzo (canale 0) viene normalizzato in [-1, 1] tramite una tangente iperbolica (Tanh).
          - L'acceleratore e il freno (canali 1 e 2) vengono mappati in [0, 1] tramite la funzione Sigmoidea.
          
        Args:
            state: Tensore PyTorch dello stato di input 87D (dimensione del batch: [B, 87]).
        Returns:
            Un tensore di dimensione [B, 3] contenente [steer, accel, brake].
        """
        features = self.backbone(state)
        cont_out = self.continuous_head(features)

        # Applica le funzioni di attivazione specifiche della policy BC
        steer = torch.tanh(cont_out[:, 0:1])            # Limitato a [-1, 1] per la sterzata
        accel_brake = torch.sigmoid(cont_out[:, 1:3])   # Limitati a [0, 1] per acceleratore/freno

        continuous = torch.cat([steer, accel_brake], dim=1)
        return continuous

    def sample(self, state: torch.Tensor, evaluate: bool = False):
        """
        Esegue l'inferenza deterministica per la policy RL (TD3+BC).
        
        Questa funzione viene invocata quando l'agente utilizza pesi TD3+BC in modalità deterministica.
        Come funziona:
          - Applica la tangente iperbolica (Tanh) su tutti e tre i canali di output.
          - Questo restituisce un vettore di comandi in [-1, 1] per tutti i controlli (sterzo, acceleratore, freno).
          - In fase di guida, acceleratore e freno verranno riscalati linearmente da [-1, 1] a [0, 1] mediante denormalize_action_rl().
          
        Args:
            state: Tensore PyTorch dello stato di input 87D (dimensione del batch: [B, 87]).
            evaluate: Se True (default in test_agent), disabilita il rumore e restituisce la media deterministica.
        Returns:
            Un tensore di dimensione [B, 3] contenente [steer, accel, brake] in [-1, 1].
        """
        features = self.backbone(state)
        mean = self.continuous_head(features)
        return torch.tanh(mean)



# Utilities di Normalizzazione e Elaborazione dello Stato

# Caricamento delle statistiche di normalizzazione dello stato (mean-0 / std-1).
# Per far convergere correttamente le reti neurali, le osservazioni 29D vengono normalizzate usando
# media e deviazione standard calcolate sull'intero dataset esperto (salvate in 'state_norm.npz').
# Questo garantisce che tutte le feature siano in una scala numerica adatta ad evitare gradienti instabili.
# Questo file DEVE coincidere esattamente con quello caricato durante la fase di addestramento.
_STATE_NORM_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'train_set', 'checkpoints', 'state_norm.npz')
if os.path.exists(_STATE_NORM_PATH):
    _sn = np.load(_STATE_NORM_PATH)
    _STATE_MEAN, _STATE_STD = _sn['mean'].astype(np.float32), _sn['std'].astype(np.float32)
else:
    _STATE_MEAN, _STATE_STD = None, None


def apply_state_norm(s):
    """
    Applica la standardizzazione mean-0/std-1 a un vettore di stato numpy.
    
    Formula applicata: s_norm = (s - media) / (dev_std + epsilon), dove epsilon = 1e-3 previene divisioni per zero.
    Se il file delle statistiche non è presente, restituisce lo stato invariato (fallback).
    
    Args:
        s: Vettore di stato grezzo.
    Returns:
        Il vettore standardizzato float32.
    """
    if _STATE_MEAN is None:
        return s
    return ((s - _STATE_MEAN) / (_STATE_STD + 1e-3)).astype(np.float32)


def flatten_state(state_dict: dict) -> np.ndarray:
    """
    Estrae le grandezze fisiche dal dizionario di TORCS e le appiattisce in un vettore standardizzato a 29 dimensioni.
    
    Come funziona:
      - Estrae i valori scalari o vettoriali (es. track a 19 direzioni) dal dizionario inviato dal wrapper di Gym.
      - Riduce in scala specifiche grandezze per allineare gli intervalli numerici (es. velocità delle ruote divisa per 100.0, RPM diviso per 10000.0).
      - Ordine del vettore risultante (29D):
          - angle (1 dimensione): Angolo tra l'auto e la direzione della pista.
          - track (19 dimensioni): Distanza dal bordo pista in 19 direzioni (già scalata per 200.0 dal wrapper).
          - trackPos (1 dimensione): Posizione trasversale rispetto al centro pista [-1.0, 1.0].
          - speedX, speedY, speedZ (3 dimensioni): Velocità longitudinali, trasversali e verticali (già scalate per 50.0).
          - wheelSpinVel (4 dimensioni): Velocità di rotazione delle 4 ruote riscalata (/100.0).
          - rpm (1 dimensione): Giri motore riscaldati (/10000.0).
      - Infine, applica la normalizzazione mean-0/std-1 tramite apply_state_norm().
      
    Args:
        state_dict: Dizionario contenente le letture dei sensori di bordo di TORCS.
    Returns:
        Un array numpy normalizzato di dimensione (29,).
    """
    def _s(key, default=0.0):
        v = state_dict.get(key, default)
        if isinstance(v, np.ndarray):
            return float(v.flat[0])
        return float(v) if v is not None else default

    def _a(key, size):
        v = state_dict.get(key, None)
        if v is None:
            return np.zeros(size, dtype=np.float32)
        return np.array(v, dtype=np.float32).flatten()[:size]

    try:
        s = np.concatenate([
            [_s('angle')],
            _a('track', 19),              # 19 sensori di distanza dai bordi
            [_s('trackPos')],             # Posizione rispetto al centro della carreggiata
            [_s('speedX')],               # Velocità in avanti
            [_s('speedY')],               # Velocità laterale
            [_s('speedZ')],               # Velocità verticale
            _a('wheelSpinVel', 4) / 100.0,# Velocità angolare delle ruote normalizzata
            [_s('rpm') / 10000.0],        # Giri al minuto del motore normalizzati
        ]).astype(np.float32)
        return apply_state_norm(s)  # standardizzazione coerente col training
    except Exception as e:
        # NON silenziare: uno stato a zero falserebbe completamente l'inferenza e causerebbe incidenti
        print(f"flatten_state fallita (stato a zero): {e}")
        return apply_state_norm(np.zeros(29, dtype=np.float32))


def denormalize_action_bc(cont_action: np.ndarray) -> np.ndarray:
    """
    Converte le azioni predette dalla policy Behavioral Cloning (BC) nel formato nativo richiesto da TORCS.
    
    Come funziona:
      - Riceve le azioni generate dalla rete neurale (sterzo limitato a [-1, 1] tramite Tanh, acceleratore/freno in [0, 1] tramite Sigmoid).
      - Esegue un clipping di sicurezza per assicurarsi che i valori rimangano all'interno dei limiti fisici del gioco.
      - Struttura del vettore di output (4D): [sterzo, acceleratore, freno, marcia] (la marcia è provvisoria e viene sovrascritta dopo).
    """
    env_action = np.zeros(4, dtype=np.float32)
    env_action[0] = np.clip(cont_action[0], -1.0, 1.0)               # sterzo clippato
    env_action[1] = np.clip(cont_action[1], 0.0, 1.0)                # acceleratore (già Sigmoid)
    env_action[2] = np.clip(cont_action[2], 0.0, 1.0)                # freno (già Sigmoid)
    return env_action


def denormalize_action_rl(cont_action: np.ndarray) -> np.ndarray:
    """
    Mappa e denormalizza le azioni predette dalla policy RL (TD3+BC) nel formato nativo richiesto da TORCS.
    
    Come funziona:
      - Riceve le azioni generate dalla rete neurale (tutte normalizzate in [-1, 1] tramite Tanh).
      - Mantiene lo sterzo invariato nell'intervallo [-1, 1].
      - Mappa l'acceleratore e il freno dall'intervallo [-1, 1] all'intervallo [0, 1] tramite la trasformazione
        affine: x_real = (x_tanh + 1.0) / 2.0.
      - Applica clipping di sicurezza e restituisce il vettore [sterzo, accelerazione, freno, marcia].
    """
    env_action = np.zeros(4, dtype=np.float32)
    env_action[0] = np.clip(cont_action[0], -1.0, 1.0)               # sterzo clippato
    env_action[1] = np.clip((cont_action[1] + 1.0) / 2.0, 0.0, 1.0) # acceleratore mappato da [-1, 1] a [0, 1]
    env_action[2] = np.clip((cont_action[2] + 1.0) / 2.0, 0.0, 1.0) # freno mappato da [-1, 1] a [0, 1]
    return env_action



#  Auto-detect e caricamento pesi

def load_best_weights(model, weights_arg, device, kind='auto'):
    """
    Rileva automaticamente e carica i pesi migliori disponibili per l'Actor, identificandone la natura (BC o RL).
    
    Come funziona:
      - Se l'utente specifica un percorso tramite '--weights', viene caricato direttamente quel file.
      - Se '--weights' è None, esamina la cartella dei checkpoint in ordine di importanza decrescente per trovare il file migliore.
      - Carica lo state_dict ed esegue una filtrazione delle chiavi (filtered_state) per caricare solo i pesi compatibili con
        l'architettura corrente dell'Actor, ignorando eventuali pesi del Critic presenti nel file.
      - Rileva se il modello è RL (TD3+BC) o BC in base al nome del file (presenza della stringa 'td3' o 'bc'), oppure
        usando l'argomento esplicito '--kind'. Questo determina la successiva denormalizzazione dell'azione.
        
    Args:
        model: Istanza della classe PolicyActor da caricare.
        weights_arg: Stringa del percorso dei pesi (opzionale).
        device: Dispositivo su cui caricare il modello ('cuda' o 'cpu').
        kind: Stringa di selezione del formato ('auto', 'rl', o 'bc').
    Returns:
        Una tupla (model, is_rl: bool) indicante il modello caricato e se si trata di una policy RL.
    """
    checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'train_set', 'checkpoints')
    td3_det_best_lap_path = os.path.join(checkpoint_dir, 'td3_det_best_lap.pth')
    td3_det_best_dist_path = os.path.join(checkpoint_dir, 'td3_det_best_dist.pth')
    td3_det_best_dist_run_path = os.path.join(checkpoint_dir, 'td3_det_best_dist_run.pth')
    td3_expl_best_lap_path = os.path.join(checkpoint_dir, 'td3_expl_best_lap.pth')
    td3_expl_best_dist_path = os.path.join(checkpoint_dir, 'td3_expl_best_dist.pth')
    td3_path = os.path.join(checkpoint_dir, 'td3_policy.pth')
    bc_path = os.path.join(checkpoint_dir, 'bc_policy.pth')

    # Priorità 1: Percorso esplicito inserito dall'utente
    if weights_arg:
        load_path = weights_arg
    # Priorità 2: Miglior giro valido deterministico (il file ideale per i test)
    elif os.path.exists(td3_det_best_lap_path):
        load_path = td3_det_best_lap_path
        _lt = ''
        _lt_txt = os.path.join(checkpoint_dir, 'td3_det_best_lap.txt')
        if os.path.exists(_lt_txt):
            try:
                with open(_lt_txt) as f: _lt = f' ({float(f.read().strip()):.3f}s)'
            except Exception: pass
        print(f"  Auto-detect: trovato td3_det_best_lap.pth (Miglior GIRO VALIDO deterministico{_lt} — candidato submission!)")
    # Priorità 3: Migliore distanza complessiva (sopravvissuta a pulizie --clean)
    elif os.path.exists(td3_det_best_dist_path):
        load_path = td3_det_best_dist_path
        print(f"  Auto-detect: trovato td3_det_best_dist.pth (Miglior policy ASSOLUTA per distanza, sopravvive ai --clean!)")
    # Priorità 4: Miglior checkpoint della sessione corrente di RL
    elif os.path.exists(td3_det_best_dist_run_path):
        load_path = td3_det_best_dist_run_path
        print(f"  Auto-detect: trovato td3_det_best_dist_run.pth (Miglior checkpoint deterministico TD3+BC!)")
    # Priorità 5: Checkpoint con miglior giro esplorativo
    elif os.path.exists(td3_expl_best_lap_path):
        load_path = td3_expl_best_lap_path
        print(f"  Auto-detect: trovato td3_expl_best_lap.pth (Record sul giro TD3+BC!)")
    # Priorità 6: Checkpoint con miglior distanza esplorativa
    elif os.path.exists(td3_expl_best_dist_path):
        load_path = td3_expl_best_dist_path
        print(f"  Auto-detect: trovato td3_expl_best_dist.pth (Record di distanza TD3+BC!)")
    # Priorità 7: Ultima policy salvata
    elif os.path.exists(td3_path):
        load_path = td3_path
        print(f"  Auto-detect: trovato td3_policy.pth (Ultimo step TD3+BC)")
    # Priorità 8: Pesi di Behavioral Cloning (supervisionato base)
    elif os.path.exists(bc_path):
        load_path = bc_path
        print(f"  Auto-detect: uso bc_policy.pth come ultima priorità supervisionata")
    else:
        print(f"  Nessun file pesi trovato!")
        sys.exit(1)

    if not os.path.exists(load_path):
        print(f"  File pesi non trovato: {load_path}")
        sys.exit(1)

    # Carica in sicurezza i pesi PyTorch sul dispositivo specificato (GPU o CPU)
    try:
        loaded = torch.load(load_path, map_location=device, weights_only=True)
    except Exception:
        loaded = torch.load(load_path, map_location=device, weights_only=False)

    # Estrae lo stato dell'Actor (se salvato all'interno di un dizionario con chiavi addizionali come optim o epoch)
    state_dict = loaded.get('actor', loaded) if isinstance(loaded, dict) else loaded
    if not hasattr(state_dict, 'items'):
        print(f"  File pesi non adatto all'Actor corrente: {load_path}")
        sys.exit(1)
        
    # Filtra ed associa le chiavi per garantire compatibilità con l'Actor del modulo corrente
    model_state = model.state_dict()
    filtered_state = {
        key: value for key, value in state_dict.items()
        if key in model_state and hasattr(value, 'shape') and model_state[key].shape == value.shape
    }
    if not filtered_state:
        print(f"  File pesi non adatto all'Actor corrente: {load_path}")
        sys.exit(1)
        
    model.load_state_dict(filtered_state, strict=False)
    model.eval()

    # Rilevamento automatico o manuale del tipo di policy (RL vs BC)
    if kind == 'rl':
        is_rl = True
    elif kind == 'bc':
        is_rl = False
    else:
        fname = os.path.basename(load_path).lower()
        if 'td3' in fname:
            is_rl = True
        elif 'bc' in fname:
            is_rl = False
        else:
            print("   Tipo pesi non deducibile dal nome file. Usa --kind rl|bc per essere esplicito.")
            sys.exit(1)

    weight_type = "RL (TD3+BC)" if is_rl else "BC"
    print(f"  Pesi [{weight_type}] caricati da: {load_path}")

    return model, is_rl


# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test Agent Autonomo (BC/TD3+BC) — TORCS")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path ai pesi del modello (.pth). Se omesso, auto-detect.")
    parser.add_argument("--laps", type=int, default=3,
                        help="Numero di giri da completare")
    parser.add_argument("--max_steps", type=int, default=15000,
                        help="Max step per giro (timeout)")
    parser.add_argument("--kind", choices=["auto", "rl", "bc"], default="auto",
                        help="Tipo di pesi: 'rl' (tanh→[0,1]) o 'bc' (sigmoid). 'auto' deduce dal nome file.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'=' * 64}")
    print(f"  TEST AGENTE AUTONOMO (BC/TD3+BC) — TORCS")
    print(f"  Device: {device}")
    print(f"  Stride Type: static (k=6, 0.24s)")
    print(f"  Modalità: DETERMINISTICA (evaluate=True, Zero Noise)")
    print(f"{'=' * 64}\n")

    # Carica la rete dell'agente e posiziona i pesi migliori (auto-detect)
    model = PolicyActor().to(device)
    model, is_rl = load_best_weights(model, args.weights, device, kind=args.kind)

    # Seleziona la funzione di denormalizzazione e la descrizione in base alla policy caricata
    denormalize_fn = denormalize_action_rl if is_rl else denormalize_action_bc
    inference_mode = "RL (sample evaluate=True)" if is_rl else "BC (forward diretto)"
    print(f"  Inference mode: {inference_mode}")

    # Inizializzazione del client di connessione al simulatore TORCS
    print("  Inizializzazione TORCS...")
    env = TorcsEnv(early_termination=False)

    lap_times = []
    total_attempts = 0

    try:
        # Il loop continua finché non completiamo il numero richiesto di giri validi
        while len(lap_times) < args.laps:
            total_attempts += 1

            # Rilancio forzato dell'ambiente TORCS ad ogni reset per prevenire accumulo di errori di fisica
            obs = env.reset(relaunch=True)
            initial_state = flatten_state(obs)

            # State Stacking (Fujimoto 2021): la rete neurale necessita di 3 frame temporali concatenati.
            # Il buffer memorizza gli ultimi 13 frame (corrispondenti a 0.24 secondi totali, con k=6).
            # All'avvio, il buffer viene inizializzato replicando lo stato iniziale.
            state_buffer = deque(maxlen=13)
            for _ in range(13):
                state_buffer.append(initial_state)

            # Rilevamento dei tempi sul giro iniziali
            raw = env.client.S.d
            prev_last_lap = float(raw.get('lastLapTime', 0.0))
            if isinstance(prev_last_lap, list): prev_last_lap = prev_last_lap[0]

            lap_completed = False
            lap_time = 0.0
            telemetry_data = []

            # Stato iniziale del cambio deterministico (anti-hunting, prevenzione oscillazioni continue di marcia)
            current_gear = 1
            steps_since_shift = 999
            cur_speed_kmh = float(np.array(obs.get('speedX', 0.0)).flat[0]) * 50.0
            cur_rpm = float(np.array(obs.get('rpm', 0.0)).flat[0])
            stall_low_speed_steps = 0  # Contatore per stalli prolungati a bassa velocità

            print(f"\n  {'─' * 50}")
            print(f"  Tentativo #{total_attempts} (giri completati: {len(lap_times)}/{args.laps})")

            for step in range(1, args.max_steps + 1):
                # Costruiamo il vettore di input 87D:
                # - state_buffer[0]  -> frame t-12 (0.24 secondi fa)
                # - state_buffer[6]  -> frame t-6 (0.12 secondi fa)
                # - state_buffer[12] -> frame t (corrente)
                stacked_state = np.concatenate([
                    state_buffer[0],
                    state_buffer[6],
                    state_buffer[12]
                ])

                # Inferenza deterministica pura, senza aggiunta di rumore di esplorazione
                with torch.no_grad():
                    state_t = torch.FloatTensor(stacked_state).to(device).unsqueeze(0)

                    if is_rl:
                        # Pesi TD3+BC: produce 3 comandi in [-1, 1] tramite tangente iperbolica (Tanh)
                        tanh_action = model.sample(state_t, evaluate=True)
                        cont_action = tanh_action.cpu().numpy()[0]
                    else:
                        # Pesi Behavioral Cloning: produce sterzo in [-1, 1] e gas/freno in [0, 1] (Sigmoid)
                        pred_cont = model(state_t)
                        cont_action = pred_cont.cpu().numpy()[0]

                # Regola di mutua esclusione per l'essere umano (non premiamo mai acceleratore e freno contemporaneamente)
                # Per la policy BC applichiamo la mutua esclusione prima della denormalizzazione poiché i valori sono già in [0,1]
                if not is_rl:
                    cont_action[1] = cont_action[1] * (1.0 - cont_action[2])

                # Converte le uscite continue nel formato compatibile con l'ambiente TORCS
                env_action = denormalize_fn(cont_action)

                # Per la policy RL applichiamo la mutua esclusione dopo la denormalizzazione (quando l'azione è scalata in [0,1])
                if is_rl:
                    env_action[1] = env_action[1] * (1.0 - env_action[2])

                # Calcolo della marcia deterministica tramite gearing.py.
                # Nota: passiamo il comando dell'acceleratore effettivo APPLICATO (env_action[1]) per evitare false cambiate in frenata.
                current_gear, _shifted = compute_gear(cur_speed_kmh, float(env_action[1]), cur_rpm, current_gear, steps_since_shift)
                steps_since_shift = 0 if _shifted else steps_since_shift + 1
                env_action[3] = current_gear

                # Eseguiamo il passo di simulazione fisica in TORCS
                next_obs, _, env_done, _ = env.step(env_action)
                next_state = flatten_state(next_obs)
                cur_speed_kmh = float(np.array(next_obs.get('speedX', 0.0)).flat[0]) * 50.0
                cur_rpm = float(np.array(next_obs.get('rpm', 0.0)).flat[0])

                # Estrazione dati per telemetria locale
                dist_raw = next_obs.get('distFromStart', 0.0)
                if isinstance(dist_raw, np.ndarray): dist_raw = float(dist_raw.flat[0])
                dist_m = dist_raw
                
                # Velocità grezza in km/h. Nota: non estraiamo da next_state poiché è normalizzato (mean/std)
                spd_kmh = float(np.array(next_obs.get('speedX', 0.0)).flat[0]) * 50.0
                track_pos = float(np.array(next_obs.get('trackPos', 0.0)).flat[0])
                angle = float(np.array(next_obs.get('angle', 0.0)).flat[0])

                telemetry_data.append({
                    'step': step,
                    'dist': dist_m,
                    'speed': spd_kmh,
                    'trackPos': track_pos,
                    'angle': angle,
                    'steer': float(env_action[0]),
                    'accel': float(env_action[1]),
                    'brake': float(env_action[2]),
                    'gear': int(env_action[3])
                })

                # Criterio 1 di invalidazione del giro: Fuori pista completo (trackPos > 1.25, tolleranza cordoli)
                if abs(track_pos) > 1.25:
                    print(f"   Fuori pista / giro non valido allo step {step} (trackPos={track_pos:.3f})")
                    break
                
                # Criterio 2 di invalidazione del giro: Testacoda dell'auto (cos dell'angolo negativo rispetto alla pista)
                if np.cos(angle) < 0:
                    print(f"   Spin allo step {step} (angle={angle:.3f})")
                    break

                # Criterio 3 di invalidazione del giro: Stallo del veicolo.
                # Se l'auto procede a meno di 5 km/h per oltre 50 step (~1 secondo di simulazione) dopo i primi 500 step,
                # consideriamo l'auto bloccata o insabbiata e interrompiamo il tentativo.
                fwd_kmh = spd_kmh * float(np.cos(angle))
                if step > 500 and fwd_kmh < 5.0:
                    stall_low_speed_steps += 1
                else:
                    stall_low_speed_steps = 0
                if stall_low_speed_steps >= 50:
                    print(f"   Stallo allo step {step} (vel. avanti {fwd_kmh:.1f} km/h)")
                    break

                # Stampa telemetria intermedia ogni 200 passi per monitoraggio live
                if step % 200 == 0:
                    print(
                        f"    [Passo {step:4d}] posizione pista={track_pos:+.3f} | "
                        f"velocità={spd_kmh:.0f} km/h | sterzo={env_action[0]:+.3f} | "
                        f"acceleratore={env_action[1]:.2f} | freno={env_action[2]:.2f} | "
                        f"marcia={int(env_action[3])}"
                    )

                # Verifica se l'agente ha tagliato il traguardo completando un giro.
                # Confronta il lastLapTime corrente nel simulatore rispetto a quello salvato ad inizio giro.
                current_last_lap = float(raw.get('lastLapTime', 0.0))
                if isinstance(current_last_lap, list): current_last_lap = current_last_lap[0]

                if current_last_lap > 0.0 and abs(current_last_lap - prev_last_lap) > 0.01:
                    lap_completed = True
                    lap_time = current_last_lap
                    break

                # Avanzamento del buffer temporale e aggiornamento stato
                state_buffer.append(next_state)
                obs = next_obs
                if env_done: break

            # Scrittura telemetria su file CSV al termine di ogni tentativo
            telemetry_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'telemetry')
            os.makedirs(telemetry_dir, exist_ok=True)
            csv_path = os.path.join(telemetry_dir, f'telemetry_attempt_{total_attempts}.csv')
            with open(csv_path, 'w', newline='') as f_csv:
                writer = csv.DictWriter(f_csv, fieldnames=['step', 'dist', 'speed', 'trackPos', 'angle', 'steer', 'accel', 'brake', 'gear'])
                writer.writeheader()
                writer.writerows(telemetry_data)
            print(f"  Telemetria del tentativo salvata in: {csv_path}")

            if lap_completed:
                lap_times.append(lap_time)
                print(f"  GIRO COMPLETATO: {lap_time:.3f}s")
            else:
                print(f"  Fallito (step: {step})")
                print("   Tentativo fallito. Prossimo tentativo...")

    except KeyboardInterrupt:
        print(f"\n  Test interrotto.")
    finally:
        env.end()

    # Riepilogo finale delle prestazioni di tutti i tentativi eseguiti
    print(f"\n{'=' * 64}")
    print(f"  RIEPILOGO TEST")
    if lap_times:
        print(f"  Giri completati: {len(lap_times)}/{args.laps}")
        print(f"  Best:  {min(lap_times):.3f}s | Media: {sum(lap_times)/len(lap_times):.3f}s")
        for i, t in enumerate(lap_times):
            print(f"  Giro {i+1}: {t:.3f}s")
    else:
        print(f"  Nessun giro completato su {total_attempts} tentativi.")
    print(f"{'=' * 64}\n")


if __name__ == "__main__":
    main()
