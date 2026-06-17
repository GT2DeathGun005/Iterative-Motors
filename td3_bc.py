"""
Modulo di Fine-Tuning TD3+BC — Twin Delayed DDPG (TD3) integrato con Behavioral Cloning per TORCS.

L'algoritmo TD3 è una variante più robusta del noto DDPG (Deep Deterministic Policy Gradient), introdotto
nel paper "Addressing Function Approximation Error in Actor-Critic Methods" (Fujimoto et al., 2018). 

Tale algoritmo ha come caratteristiche principali:
    - Twin Critic: Introduce rispetto a DDPG due reti indipendenti, Q1 e Q2, il target di bellman viene calcolato come il minimo tra i due, riducendo la probabilità di sovrastima.
    - Delayed Policy Update: L'aggiornamento delle policy di Critic e Actor non avviene contemporaneamente, l'actor viene aggiornato meno frequentemente rispetto al critic (policy_frequency = 2 step) 
    - Target Policy Smoothing: Aggiunge rumore clippato all'azione target calcolata per le reti target del Critic
      per prevenire l'overfitting su picchi stretti della Q-function.

Questo modulo implementa l'algoritmo TD3+BC (Fujimoto & Gu, 2021), un'architettura offline-to-online
cioè il modello viene prima pre addestrato con dati raccolti ma senza interagire con l'ambiente, dopodiché viene inserito nel simulatore TORCS
ibrida ideata per addestrare un agente di guida autonoma sul simulatore TORCS. Lo scopo principale
è ereditare la conoscenza iniziale appresa per imitazione da un pilota umano (Behavioral Cloning - BC)
e raffinarla tramite apprendimento per rinforzo (Reinforcement Learning - RL) senza incorrere
nella degradazione della policy o nel collasso dei gradienti causati da stime del valore (Q-values) errate all'avvio.

ARCHITETTURA E LOGICA DEL SISTEMA

1. Struttura dei Modelli:
   - Actor (Agente di Guida): Eredita il backbone e la testa continua del modello BC pre-addestrato
     (Warm-Start). Durante il training RL, l'intera rete dell'Actor viene aggiornata per ottimizzare
     le traiettorie. I comandi continui generati controllano sterzo, acceleratore e freno, mentre
     il cambio marcia è governato dal modulo esterno 'gearing.py'.
   - Critic (Twin Q-Networks): Due reti identiche e indipendenti, addestrate da zero. Esse
     ricevono in input lo stato 87D dell'ambiente e l'azione 3D scelta, stimando il valore Q atteso.

2. Funzione di Perdita (Loss) dell'Actor TD3+BC (Tali formule sono state prese dal paper Fujimoto & Gu (2021)):
   La loss dell'Actor è definita come:
       Loss = -lambda * Q(s, pi(s)) + BC_Penalty    
   Dove:
     - Q(s, pi(s)) è il valore Q stimato dal Critic Q1.
     - BC_Penalty è l'errore quadratico medio (MSE) tra l'azione predetta e l'azione dell'esperto umano.
     - lambda è un coefficiente di bilanciamento calcolato dinamicamente ad ogni batch come:
       lambda = alpha / mean(|Q(s, pi(s))|), con alpha = 2.5. Questo garantisce che la componente di
       rinforzo (RL) mantenga la stessa scala del termine di imitazione (BC), rendendo il gradiente stabile
       rispetto a forti variazioni delle magnitudini dei Q-values.

3. Mascheramento Rigoroso della BC Penalty (Expert Masking):
   La BC Penalty viene calcolata ed applicata esclusivamente sulla quota di campioni del batch che provengono
   dal dataset esperto (contrassegnati da expert_mask = 1.0). Questo permette al modello di non essere
   penalizzato se prova traiettorie differenti da quelle del pilota umano (esperto) durante la guida online.

4. Campionamento Ibrido a Tre Vie (Three-Way Replay Buffer):
   I batch di addestramento sono composti combinando tre sorgenti di esperienza distinte per massimizzare
   sia la fedeltà all'esperto sia la capacità di recupero dagli errori:
     - 25% Dataset Umano (Expert): Buffer permanente, previene il distacco dalla BC e la degenerazione ad un RL puro.
     - 15% Migliori Run (Elite): Riproduce transizioni provenienti dai tentativi autonomi migliori, incoraggiando l'auto-imitazione(Self-Imitation).
     - 60% Esplorazione Online: Permette all'agente di imparare a gestire stati sporchi fuori traiettoria.

    Tali percentuali sono state scelte empiricamente.

5. Funzione di Ricompensa e Gestione delle Penalità:
   Il codice della reward è contenuto all'interno del file gym_torcs.py, in questo file è richiamata e vengono sommati eventuali malus o bonus.
   
   Ad ogni passo di simulazione (step), l'ambiente calcola una ricompensa multi-obiettivo per guidare l'ottimizzazione dell'agente:
     - Progresso Longitudinale (Velocità): Calcolato come `(progress * 1.5)`. Premia la proiezione della velocità 
       lungo l'asse centrale del tracciato, penalizzando derive trasversali. Il fattore di scala `1.5` incentiva 
       la ricerca di velocità massime nei rettilinei.
     - Penalità di Posizione Stradale: `-2.0 * (max(0.0, |trackPos| - 1.0) ^ 2)`. Una penalità quadratica applicata 
       esclusivamente quando l'auto esce dai bordi dell'asfalto (`|trackPos| > 1.0`). Se l'auto è all'interno 
       delle linee stradali, questo termine è nullo (`0.0`), agendo così come una barriera virtuale morbida.
     - Penalità di Cambio Direzione (Anti-oscillazione): `-0.05 * |steer_change|`. Penalizza le variazioni brusche 
       dello sterzo tra due istanti temporali adiacenti. Evita il comportamento di zigzag nei rettilinei 
       e costringe l'Actor ad apprendere traiettorie di guida fluide.
     - Penalità di Uscita Pista (Off-Track Crash): Se `|trackPos| > 1.25` (taglio curva o urto barriere), l'episodio 
       viene interrotto prematuramente e viene applicata una sanzione di stacco `-base_penalty - (extra_penalty * eccesso)`.
     - Penalità di Stallo e Spin: Se la vettura rimane ferma per più di 10 secondi o compie un testacoda (coseno dell'angolo 
       rispetto al tracciato negativo, `cos(angle) < 0`), l'episodio termina con una penalità fissa di collisione.
     - Bonus di Fine Giro (aggiunto in TD3+BC): `+50.0` se il traguardo viene tagliato regolarmente con successo,
       più un bonus proporzionale al tempo (`+10.0` per ogni secondo sotto il riferimento di 80s) che premia
       direttamente i giri veloci: la sola reward di progresso produce un ritorno per giro quasi costante.
     - Malus Giro Incompleto (aggiunto in TD3+BC): `-25.0` se l'episodio termina prematuramente per sbandata o crash,
       scoraggiando la guida imprudente a favore del completamento del circuito.
"""

import os
import sys
import argparse
import random
import re
import signal
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from datetime import datetime

# Import gym_torcs
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'gym_torcs')))
try:
    from gym_torcs import TorcsEnv
    import snakeoil3_gym as snakeoil3
except ImportError:
    print("Warning: gym_torcs non trovato.")

from gearing import compute_gear  # cambio marcia algoritmico

_PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
_CHECKPOINT_ROOT = os.path.join(_PROJECT_ROOT, 'train_set', 'checkpoints')
_CHECKPOINT_BACKUP_ROOT = os.path.join(_CHECKPOINT_ROOT, 'backups')
LAP_SUCCESS_BONUS = 50.0
INCOMPLETE_LAP_PENALTY = 25.0
TRACK_LENGTH_M = 3608.0
EVAL_DISTANCE_SANITY_LIMIT = 3800.0

# Bonus terminale proporzionale al tempo sul giro: il solo LAP_SUCCESS_BONUS fisso premia
# allo stesso modo un giro da 70s e uno da 85s; questo termine aggiunge un incentivo diretto
# alla riduzione del tempo (10 punti per ogni secondo sotto il riferimento di 80s).
LAP_TIME_BONUS_REF_S = 80.0
LAP_TIME_BONUS_PER_S = 10.0

# Score di valutazione unificato: per giri incompleti coincide con la distanza percorsa,
# per giri completati cresce al diminuire del tempo (score = TRACK_LENGTH_M * T_REF / lap_time).
# Risolve la saturazione della metrica a 3608m quando l'agente completa il giro: senza score,
# la macchina a stati del refinement non vede più alcun gradiente di miglioramento.
# Con T_REF = 90s: giro da 70.0s -> 4639m; 1 secondo di giro vale circa 66m di score.
EVAL_SCORE_T_REF_S = 90.0
EVAL_SCORE_SANITY_LIMIT = 6500.0  # corrisponde a un giro < 50s, fisicamente implausibile

# Relaunch completo di TORCS (kill + riavvio + macro di autostart) solo ogni N episodi:
# costa ~6 secondi reali contro il reset soft (meta-restart in-place) quasi istantaneo.
# Il relaunch periodico mantiene la pulizia dello stato del simulatore (memory leak,
# socket sporchi) senza pagarne il costo a ogni episodio. L'offset 2 lo tiene lontano
# dalle eval (episodi ≡ 4 mod 5), che fanno già relaunch completi per conto loro.
# Se la connessione col server cade, gym_torcs forza comunque il relaunch da solo.
RELAUNCH_EVERY_EPISODES = 5
RELAUNCH_EPISODE_OFFSET = 2

# Annealing del rumore esplorativo: a inizio training serve esplorazione ampia (0.10),
# ma a regime un disturbo cosi' grande sullo sterzo causa crash sistematici in curva veloce;
# per limare gli ultimi decimi servono micro-variazioni di traiettoria (0.04).
EXPL_NOISE_START = 0.10
EXPL_NOISE_END = 0.04
EXPL_NOISE_ANNEAL_EPISODES = 1500

def _is_plausible_eval_dist(value):
    """
    Verifica se una distanza percorsa misurata durante la fase di evaluation è fisicamente plausibile.
    
    Serve a filtrare eventuali anomalie nei log in cui la distanza registrata supera i limiti fisici
    del singolo giro (EVAL_DISTANCE_SANITY_LIMIT = 3800m), prevenendo statistiche inficiate.
    """
    return 0.0 <= float(value) <= EVAL_DISTANCE_SANITY_LIMIT

def _is_plausible_eval_score(value):
    """
    Verifica la plausibilità di uno score di valutazione (distanza o equivalente-tempo).

    A differenza di _is_plausible_eval_dist, ammette valori oltre la lunghezza del tracciato:
    un giro completato in 70s produce uno score di ~4639m. Il limite di 6500m corrisponde
    a un giro sotto i 50 secondi, fisicamente irraggiungibile.
    """
    return 0.0 <= float(value) <= EVAL_SCORE_SANITY_LIMIT

def _eval_score(eval_dist, lap_time=None):
    """
    Converte il risultato di una valutazione deterministica in uno score scalare confrontabile.

    Due regimi:
      - Giro incompleto (lap_time assente): score = distanza percorsa, clampata alla lunghezza pista.
      - Giro completato: score = TRACK_LENGTH_M * (EVAL_SCORE_T_REF_S / lap_time), con floor a
        TRACK_LENGTH_M così un giro completato (anche lento) vale sempre più di uno incompleto.

    Lo score sostituisce la distanza pura in tutta la logica di record e refinement: una volta
    che l'agente completa il giro stabilmente, la distanza satura a 3608m e smette di dare segnale,
    mentre lo score continua a crescere al migliorare del tempo sul giro.
    """
    if lap_time is not None and 30.0 < float(lap_time) < EVAL_SCORE_T_REF_S * 4:
        return TRACK_LENGTH_M * max(1.0, EVAL_SCORE_T_REF_S / float(lap_time))
    return max(0.0, min(float(eval_dist), TRACK_LENGTH_M))

def _track_progress_from_start(start_dist, current_dist):
    """
    Calcola la distanza percorsa lungo il circuito a partire da un punto iniziale specificato.
    
    Gestisce correttamente la logica di wrap-around (ritorno a zero) al passaggio sulla linea del traguardo
    sfruttando la lunghezza totale nota del circuito (TRACK_LENGTH_M = 3608m).
    
    Non usiamo distRaced perché misura la distanza realmente percorsa dal veicolo anche quando sbanda
    o allunga la traiettoria: per il record di giro interessa invece il progresso lungo il tracciato,
    misurato tramite distFromStart e corretto per il wrap al traguardo.
    """
    start = float(start_dist)
    current = float(current_dist)
    progress = current - start
    if progress < 0.0:
        progress += TRACK_LENGTH_M
    return max(0.0, min(progress, TRACK_LENGTH_M))

def _fsync_file(path):
    """
    Forza la scrittura fisica (flush) dei dati dal buffer di memoria del sistema operativo sul disco fisso.
    
    Viene usata dopo le operazioni di scrittura dei checkpoint per assicurare che il file sia memorizzato
    fisicamente e non rimanga in una coda volatile volatile del kernel, evitando file corrotti (da 0 byte)
    in caso di improvviso crash del sistema.
    """
    with open(path, 'rb') as f:
        os.fsync(f.fileno())

def _fsync_dir(path):
    """
    Sincronizza i metadati della directory genitrice su disco tramite la chiamata di sistema fsync.
    
    Questo passaggio è cruciale per garantire la persistenza dell'operazione atomica di sostituzione (os.replace)
    ed evitare perdite di puntatori all'interno del file system in caso di spegnimento anomalo del computer.
    """
    dir_fd = os.open(path or '.', os.O_DIRECTORY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)

def _backup_paths(filepath):
    """
    Determina i percorsi assoluti da destinare ai file di backup del checkpoint (.bak e .prev).
    
    Se il file originale si trova nella cartella principale dei checkpoint, organizza i relativi backup
    in una sottocartella dedicata ('train_set/checkpoints/backups') per mantenere l'albero dei file pulito.
    """
    abs_filepath = os.path.abspath(filepath)
    backup_base = None
    try:
        if os.path.commonpath([abs_filepath, _CHECKPOINT_ROOT]) == _CHECKPOINT_ROOT:
            relative_path = os.path.relpath(abs_filepath, _CHECKPOINT_ROOT)
            if relative_path != 'backups' and not relative_path.startswith('backups' + os.sep):
                backup_base = os.path.join(_CHECKPOINT_BACKUP_ROOT, relative_path)
    except ValueError:
        backup_base = None

    if backup_base is None:
        backup_base = filepath

    return backup_base + ".bak", backup_base + ".prev"

def _rotate_backup(filepath):
    """
    Ruota ciclicamente le copie di backup esistenti per conservare la cronologia recente.
    
    Sposta il file '.bak' (backup precedente) in '.prev' (penultimo backup) e crea una copia
    del checkpoint corrente nominandola '.bak'. L'operazione è resa sicura tramite passaggi temporanei
    e forzature di scrittura fisica (fsync).
    """
    if not os.path.exists(filepath):
        return
    backup_path, previous_path = _backup_paths(filepath)
    backup_directory = os.path.dirname(backup_path) or '.'
    os.makedirs(backup_directory, exist_ok=True)

    if os.path.exists(backup_path):
        os.replace(backup_path, previous_path)

    temp_backup = backup_path + ".tmp"
    shutil.copy2(filepath, temp_backup)
    _fsync_file(temp_backup)
    os.replace(temp_backup, backup_path)
    _fsync_dir(backup_directory)

def _checkpoint_candidates(filepath):
    """
    Restituisce una lista ordinata di percorsi candidati in cui cercare un checkpoint valido.
    
    L'ordine va dal file primario cercato ai vari backup storici (.bak, .prev). Questa ridondanza
    permette all'agente di riprendere l'esecuzione (resume) caricando lo stato coerente più recente
    anche se il file principale si è danneggiato o è stato interrotto a metà scrittura.
    """
    backup_path, previous_path = _backup_paths(filepath)
    candidates = [filepath, backup_path, previous_path, filepath + ".bak", filepath + ".prev"]
    unique_candidates = []
    seen = set()
    for candidate in candidates:
        key = os.path.abspath(candidate)
        if key not in seen:
            unique_candidates.append(candidate)
            seen.add(key)
    return unique_candidates

def safe_save(obj, filepath, keep_backup=True):
    """
    Salva un oggetto PyTorch in modo atomico e sicuro contro le interruzioni di corrente.
    
    Come funziona:
      - Salva l'oggetto su un percorso temporaneo (estensione '.tmp').
      - Esegue fsync per forzare la persistenza fisica.
      - Esegue la rotazione dei backup esistenti (.bak e .prev).
      - Rinomina atomicamente il file temporaneo nel percorso finale usando os.replace.
      - Sincronizza i metadati della directory genitrice.
    """
    directory = os.path.dirname(filepath) or '.'
    os.makedirs(directory, exist_ok=True)
    temp_filepath = filepath + ".tmp"
    torch.save(obj, temp_filepath)
    _fsync_file(temp_filepath)
    if keep_backup:
        _rotate_backup(filepath)
    os.replace(temp_filepath, filepath)
    _fsync_dir(directory)

def safe_write_text(filepath, text, keep_backup=True):
    """
    Scrive una stringa di testo (es. metadati e sidecar di record) in modo atomico e sicuro.
    
    Implementa lo stesso protocollo di scrittura temporanea, sincronizzazione forzata e
    rotazione dei backup usato per i checkpoint binari di PyTorch.
    """
    directory = os.path.dirname(filepath) or '.'
    os.makedirs(directory, exist_ok=True)
    temp_filepath = filepath + ".tmp"
    with open(temp_filepath, 'w', encoding='utf-8') as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    if keep_backup:
        _rotate_backup(filepath)
    os.replace(temp_filepath, filepath)
    _fsync_dir(directory)

def safe_read_float(filepath, default):
    """
    Legge un valore a virgola mobile da un file sidecar testuale, gestendo potenziali errori.
    
    In caso di problemi di lettura o di assenza del file primario, tenta automaticamente di caricare
    il valore dai backup storici (.bak, .prev). Restituisce il valore di default in ultima istanza.
    """
    for candidate in _checkpoint_candidates(filepath):
        if not os.path.exists(candidate):
            continue
        try:
            with open(candidate, 'r', encoding='utf-8') as f:
                return float(f.read().strip())
        except Exception as e:
            print(f"Impossibile leggere valore numerico da {candidate}: {e}")
    return default

def safe_save_npz(buffer_obj, filepath, keep_backup=True):
    """
    Salva il ReplayBuffer in formato binario compresso (.npz) garantendo atomicità.
    
    Utilizza un file temporaneo con estensione '.tmp.npz' per evitare che la libreria numpy
    aggiunga desinenze ridondanti e per non sovrascrivere direttamente il file principale
    prima che sia interamente registrato sul disco fisso.
    """
    if len(buffer_obj.buffer) == 0:
        return
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    # np.savez_compressed appende automaticamente '.npz' se non presente.
    # Per evitarlo, facciamo terminare il file temporaneo con '.tmp.npz'.
    temp_filepath = filepath.replace(".npz", "") + ".tmp.npz"
    buffer_obj.save(temp_filepath)
    if os.path.exists(temp_filepath):
        _fsync_file(temp_filepath)
        if keep_backup:
            _rotate_backup(filepath)
        os.replace(temp_filepath, filepath)
        _fsync_dir(os.path.dirname(filepath) or '.')

_STATE_NORM_PATH = 'train_set/checkpoints/state_norm.npz'

def _load_state_norm():
    """
    Carica i file delle statistiche di normalizzazione (media e deviazione standard) degli stati.
    
    Queste statistiche sono pre-calcolate a partire dal dataset esperto umano per consentire
    una normalizzazione mean-0/std-1 stabile come raccomandato in TD3+BC (Fujimoto & Gu, 2021).
    
    Ritorna:
        Una tupla (mean, std) di array numpy a 32-bit float, o (None, None) se il file non esiste.
    """
    if os.path.exists(_STATE_NORM_PATH):
        d = np.load(_STATE_NORM_PATH)
        return d['mean'].astype(np.float32), d['std'].astype(np.float32)
    return None, None

_STATE_MEAN, _STATE_STD = _load_state_norm()

def apply_state_norm(s):
    """
    Normalizza le feature di stato sensoriali grezze (29D) per centrarle a media 0 e deviazione standard 1.
    
    Formula:
        s_norm = (s - mean) / (std + 1e-3)
    Il termine 1e-3 evita divisioni per zero su sensori statici.
    Se le statistiche non sono caricate, restituisce il vettore grezzo senza modifiche (no-op).
    """
    if _STATE_MEAN is None:
        return s
    return ((s - _STATE_MEAN) / (_STATE_STD + 1e-3)).astype(np.float32)

# ──────────────────────────────────────────────────────────────────────
#  Determinismo
# ──────────────────────────────────────────────────────────────────────
def set_seed(seed=42):
    """
    Imposta i seed di generazione casuale per garantire la riproducibilità degli esperimenti.
    
    Configura i generatori casuali per:
      - Libreria standard python (random)
      - NumPy (np.random)
      - PyTorch CPU e CUDA (torch.manual_seed, torch.cuda.manual_seed)
      - Configurazione backend cuDNN in modalità deterministica
      - Variabile d'ambiente PYTHONHASHSEED
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

#  Replay Buffer
class ReplayBuffer:
    """
    Buffer di memorizzazione delle transizioni per l'addestramento Off-Policy (Replay Buffer).
    
    Questa classe memorizza le esperienze sotto forma di tuple: (stato, azione, reward, stato_successivo, done).
    Gestisce anche un flag parallelo ('expert') per marcare i campioni originati dall'operatore umano (expert=1.0)
    rispetto a quelli collezionati in autonomia dall'agente (expert=0.0). Questo marcatore è fondamentale
    per isolare i campioni su cui calcolare la penalità BC (Behavioral Cloning Penalty).
    """
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        self.expert_masks = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, expert=0.0):
        """
        Inserisce una nuova transizione nel buffer. Se la capacità massima è superata,
        il campione più vecchio viene rimosso (coda circolare FIFO).
        """
        self.buffer.append((state, action, reward, next_state, done))
        self.expert_masks.append(expert)

    def sample(self, batch_size: int):
        """
        Estrae casualmente un batch di transizioni dal buffer.
        
        Ritorna:
            Una tupla di array numpy (state, action, reward, next_state, done, expert_mask).
        """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        expert_masks_batch = [self.expert_masks[i] for i in indices]
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done, np.array(expert_masks_batch, dtype=np.float32)

    def save(self, filepath: str):
        """
        Salva l'intero contenuto del buffer su un file compresso numpy (.npz) per consentire
        il ripristino o il riavvio del training.
        """
        if len(self.buffer) == 0: return
        states, actions, rewards, next_states, dones = zip(*self.buffer)
        np.savez_compressed(filepath,
            states=np.array(states, dtype=np.float32),
            actions=np.array(actions, dtype=np.float32),
            rewards=np.array(rewards, dtype=np.float32),
            next_states=np.array(next_states, dtype=np.float32),
            dones=np.array(dones, dtype=np.float32),
            expert_masks=np.array(list(self.expert_masks), dtype=np.float32))

    def load_expert_data(self, h5_dir_or_file: str, max_samples: int = None, max_lap_time: float = None):
        """
        Carica i dati di guida registrati dall'esperto umano (file .h5) e li inserisce nel buffer.

        Come funziona:
          - Legge i file HDF5 estratti durante la guida manuale.
          - Se max_lap_time è specificato, scarta i file il cui attributo 'lap_time' supera la soglia
            (vale sia per i giri completi sia per i segmenti, che ereditano il tempo del giro padre).
            Questo alza il livello dell'ancora BC: imitare la media di tutti i giri umani tira la policy
            verso il giro medio, mentre per superare il pilota serve imitare solo i suoi giri migliori.
          - Normalizza gli stati fisici 29D grezzi usando la media e deviazione standard pre-calcolate.
          - Applica lo State Stacking (Fujimoto 2021) concatenando t-12 (index i-12), t-6 (index i-6) e t (index i)
            per formare gli stati 87D che la rete si aspetta in input.
          - Mappa l'azione dell'esperto (acceleratore e freno) dall'intervallo [0, 1] (Sigmoid) all'intervallo [-1, 1] (Tanh)
            per renderle coerenti con le uscite della testa continua dell'Actor.
          - Calcola a posteriori il reward associato a ciascuna transizione usando la stessa formula di gym_torcs,
            favorendo il progresso longitudinale e penalizzando le uscite di pista.
          - Salva i campioni marcando il flag expert = 1.0.
        """
        import glob
        import h5py
        import os

        if os.path.isdir(h5_dir_or_file):
            h5_files = sorted(glob.glob(os.path.join(h5_dir_or_file, "**/lap_*.h5"), recursive=True))
        else:
            h5_files = [h5_dir_or_file]

        loaded = 0
        skipped_slow = 0
        for f in h5_files:
            # Check per non superare il numero massimo di campioni
            if max_samples and loaded >= max_samples: break
            try:
                with h5py.File(f, 'r') as h5f:
                    if max_lap_time is not None:
                        file_lap_time = h5f.attrs.get('lap_time', None)
                        if file_lap_time is not None and float(file_lap_time) > max_lap_time:
                            skipped_slow += 1
                            continue
                    states_np = h5f['states'][:]
                    actions_np = h5f['actions'][:]

                states_norm = apply_state_norm(states_np)  # applica la normalizzazione
                length = len(states_np)
                k = 6

                # applica lo state stacking concatenando t-12, t-6 e t
                for i in range(length - 1):
                    if max_samples and loaded >= max_samples: break
                    idx_t6 = max(0, i - k)
                    idx_t12 = max(0, i - 2 * k)
                    next_i = i + 1
                    n_idx_t6 = max(0, next_i - k)
                    n_idx_t12 = max(0, next_i - 2 * k)

                    stacked_state = np.concatenate([states_norm[idx_t12], states_norm[idx_t6], states_norm[i]])
                    next_stacked_state = np.concatenate([states_norm[n_idx_t12], states_norm[n_idx_t6], states_norm[next_i]])

                    cont_action = actions_np[i, 0:3].copy()
                    cont_action[1] = (cont_action[1] * 2.0) - 1.0
                    cont_action[2] = (cont_action[2] * 2.0) - 1.0

                    speedX = states_np[i, 21] * 50.0
                    angle = states_np[i, 0]
                    trackPos = states_np[i, 20]

                    progress = (speedX / 50.0) * np.cos(angle)
                    tp = abs(trackPos)
                    pos_penalty = -2.0 * (max(0.0, tp - 1.0) ** 2)
                    steer_change = cont_action[0] - actions_np[i-1, 0] if i > 0 else 0.0
                    reward = (progress * 1.5) + pos_penalty - (0.05 * abs(steer_change))

                    mask = 1.0 

                    self.push(stacked_state, cont_action, reward, next_stacked_state, mask, expert=1.0)
                    loaded += 1
            except Exception as e:
                print(f"Errore caricando {f}: {e}")

        filtro_msg = ""
        if max_lap_time is not None:
            filtro_msg = f" (filtro lap_time <= {max_lap_time:.1f}s: scartati {skipped_slow} file più lenti)"
        print(f"  [EXPERT INJECTION] Caricati {loaded} campioni esperti nel Replay Buffer.{filtro_msg}")

    def load(self, filepath: str):
        """
        Carica le transizioni compresse salvate in un file .npz nel buffer in memoria.
        """
        if not os.path.exists(filepath): return
        with np.load(filepath) as data:
            states = data['states']
            actions = data['actions']
            rewards = data['rewards']
            next_states = data['next_states']
            dones = data['dones']
            expert_masks_data = data['expert_masks'] if 'expert_masks' in data.files else np.zeros(len(states))
            states, actions, rewards, next_states, dones, expert_masks_data = [
                np.asarray(x) for x in (states, actions, rewards, next_states, dones, expert_masks_data)
            ]
        lengths = {len(states), len(actions), len(rewards), len(next_states), len(dones), len(expert_masks_data)}
        if len(lengths) != 1:
            raise ValueError(f"Replay Buffer non coerente in {filepath}: lunghezze diverse {sorted(lengths)}")

        new_buffer = deque(maxlen=self.buffer.maxlen)
        new_expert_masks = deque(maxlen=self.expert_masks.maxlen)
        for i in range(len(states)):
            new_buffer.append((states[i], actions[i], float(rewards[i]), next_states[i], float(dones[i])))
            new_expert_masks.append(float(expert_masks_data[i]))
        self.buffer = new_buffer
        self.expert_masks = new_expert_masks
        print(f"  Replay Buffer caricato: {len(self.buffer)} transizioni")

    def __len__(self):
        return len(self.buffer)

def flatten_state(state_dict: dict) -> np.ndarray:
    """
    Esegue l'appiattimento e la normalizzazione delle letture sensoriali grezze di TORCS.
    
    Estrae le 29 caratteristiche dello stato (angoli, track, trackPos, speedX/Y/Z, velocità ruote, RPM),
    le normalizza utilizzando media e deviazione standard pre-calcolate, e restituisce il vettore 29D.
    """
    def _s(key, default=0.0):
        v = state_dict.get(key, default)
        if isinstance(v, np.ndarray): return float(v.flat[0])
        return float(v) if v is not None else default
    def _a(key, size):
        v = state_dict.get(key, None)
        if v is None: return np.zeros(size, dtype=np.float32)
        return np.array(v, dtype=np.float32).flatten()[:size]
    try:
        s = np.concatenate([
            [_s('angle')], _a('track', 19), [_s('trackPos'), _s('speedX'), _s('speedY'), _s('speedZ')],
            _a('wheelSpinVel', 4) / 100.0, [_s('rpm') / 10000.0]
        ]).astype(np.float32)
        return apply_state_norm(s)  
    except Exception as e:
        print(f"flatten_state fallita (stato a zero): {e}")
        return apply_state_norm(np.zeros(29, dtype=np.float32))

# ──────────────────────────────────────────────────────────────────────
#  Architettura TD3
# ──────────────────────────────────────────────────────────────────────
class Actor(nn.Module):
    """
    Policy deterministica per la generazione dei comandi di guida (Actor Network).
    
    L'input è uno stato concatenato a 87 dimensioni (3 stack temporali di 29 sensori).
    Usa un backbone a 4 strati lineari fully-connected (512 neuroni ciascuno) con Layer Normalization
    e attivazioni ReLU per estrarre le caratteristiche di guida.
    La testa continua produce 3 uscite continue normalizzate nell'intervallo [-1, 1] tramite Tanh:
      - Uscita 0: Sterzo dell'auto.
      - Uscita 1: Pressione dell'acceleratore.
      - Uscita 2: Pressione del freno.
    """
    def __init__(self, state_dim=87, hidden_size=512):
        super(Actor, self).__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU(),
        )
        self.continuous_head = nn.Linear(hidden_size, 3)  # steer, accel, brake

    def forward(self, state):
        """
        Calcola l'azione deterministica grezza (senza rumore) a partire dallo stato 87D.
        """
        features = self.backbone(state)
        mean = self.continuous_head(features)
        return torch.tanh(mean)

    def sample(self, state, evaluate=False, noise_std=0.1):
        """
        Determina l'azione da eseguire sull'ambiente TORCS a partire dallo stato corrente.

        A seconda della modalità di esecuzione, l'azione può essere esplorativa o deterministica:
          - Training (evaluate = False): Aggiunge un rumore Gaussiano esplorativo con deviazione standard
            noise_std (clippato in [-2*noise_std, +2*noise_std]) all'azione deterministica. La deviazione
            standard viene annealata dal training loop (da EXPL_NOISE_START a EXPL_NOISE_END) perché a fine
            training servono micro-variazioni di traiettoria, non sbandate. L'azione finale viene saturata
            nell'intervallo [-1.0, 1.0].
          - Valutazione (evaluate = True): Restituisce l'azione deterministica pura prodotta dalla rete Actor,
            garantendo una guida stabile, pulita e riproducibile per la fase di submission/test.
        """
        action = self.forward(state)

        # Se non siamo in evaluate, aggiunge rumore all'azione
        if not evaluate:
            noise = torch.randn_like(action) * noise_std
            noise = torch.clamp(noise, -2.0 * noise_std, 2.0 * noise_std)
            action = torch.clamp(action + noise, -1.0, 1.0)

        return action

    def load_bc_weights(self, bc_path):
        """
        Inizializza l'agente caricando i pesi pre-addestrati tramite Behavioral Cloning.
        
        Compensa lo scaling delle uscite per acceleratore e freno moltiplicandone pesi e bias per 0.5.
        Questo è necessario perché la policy BC usava la Sigmoid [0, 1] per gas/freno, mentre TD3+BC
        usa la Tanh [-1, 1], richiedendo una conversione lineare y = 0.5 * x per preservare i valori iniziali.
        """
        if not os.path.exists(bc_path): return
        bc_state = torch.load(bc_path, map_location='cpu', weights_only=True)

        if 'continuous_head.weight' in bc_state:
            bc_state['continuous_head.weight'][1:3] = bc_state['continuous_head.weight'][1:3] * 0.5
        
        if 'continuous_head.bias' in bc_state:
            bc_state['continuous_head.bias'][1:3] = bc_state['continuous_head.bias'][1:3] * 0.5
        
        self.load_state_dict(bc_state, strict=False)
        print(f"Pesi BC caricati con successo da {bc_path} (compensato scaling 0.5 per accel/brake).")

    def load_actor_weights(self, path, device):
        """
        Carica i pesi dell'Actor filtrando solo i parametri adatti alla struttura corrente.
        """
        if not os.path.exists(path): return
        try:
            loaded = torch.load(path, map_location=device, weights_only=True)
        except Exception:
            loaded = torch.load(path, map_location=device, weights_only=False)
        state_dict = loaded.get('actor', loaded) if isinstance(loaded, dict) else loaded
        model_state = self.state_dict()
        filtered_state = {
            k: v for k, v in state_dict.items()
            if k in model_state and hasattr(v, 'shape') and model_state[k].shape == v.shape
        }
        self.load_state_dict(filtered_state, strict=False)

class Critic(nn.Module):
    """
    Twin Critic Network per la stima del valore Q(s, a).
    
    Implementa due reti Q indipendenti (Q1 e Q2) che prendono in input la concatenazione
    dello stato 87D e dell'azione 3D. L'uso di due reti distinte previene l'Overestimation Bias:
    ad ogni passo di ottimizzazione si sceglie il minimo tra le due stime per calcolare il target TD.
    """
    def __init__(self, state_dim=87, action_dim=3, hidden_size=512):
        super(Critic, self).__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
        )

    def forward(self, state, action):
        """
        Ritorna le stime Q1(s,a) e Q2(s,a) come tupla.
        """
        xu = torch.cat([state, action], 1)
        return self.q1(xu), self.q2(xu)

# ──────────────────────────────────────────────────────────────────────
#  TD3+BC Agent
# ──────────────────────────────────────────────────────────────────────
class TD3BCAgent:
    """
    Classe principale dell'agente TD3+BC che coordina l'ottimizzazione e il ciclo di addestramento.
    
    Questa classe gestisce l'interazione tra i modelli neurali dell'Actor e del Twin Critic, controllando i passaggi chiave:
      - Selezione delle azioni (con o senza rumore esplorativo Gaussiano).
      - Ottimizzazione dei Critic tramite la minimizzazione dell'errore di differenza temporale (TD Error), 
        ovvero la discrepanza tra la stima Q corrente e il target calcolato con l'equazione di Bellman.
      - Ottimizzazione dell'Actor minimizzando la loss ibrida RL/BC descritta nella documentazione del modulo.
      - Stabilizzazione del training tramite Polyak Averaging, ovvero l'aggiornamento lento e progressivo delle 
        reti target interpolando i pesi attivi con un tasso controllato dal coefficiente tau (soft update).
      - Gestione degli stati speciali di training (congelamento temporaneo dell'Actor post-rollback e modalità di refinement).
    """
    def __init__(self, device="cuda"):
        self.device = torch.device(device)
        self.gamma = 0.99   # Fattore di sconto temporale per il calcolo del valore Q futuro
        self.tau = 0.005    # Parametro per l'aggiornamento soft Polyak delle reti target
        self.policy_freq = 2 # Frequenza di aggiornamento dell'Actor rispetto al Critic (Delayed Policy Update)
        self.expl_noise = EXPL_NOISE_START  # Dev. standard del rumore esplorativo, annealata dal training loop
        self.bc_alpha = 2.5  # Coefficiente alpha del TD3+BC: piu' alto = piu' peso alla componente RL rispetto alla BC

        # Inizializzazione Actor (online e target)
        self.actor = Actor().to(self.device)
        self.actor_target = Actor().to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Inizializzazione Critic (online e target)
        self.critic = Critic().to(self.device)
        self.critic_target = Critic().to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Ottimizzatori Adam per l'aggiornamento dei parametri neurali
        actor_params = [p for p in self.actor.parameters() if p.requires_grad]
        self.actor_optimizer = optim.Adam(actor_params, lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)

    def select_action(self, state, evaluate=False):
        """
        Seleziona l'azione continua 3D per lo stato corrente.
        
        Se evaluate = True, la scelta è deterministica. Altrimenti, viene aggiunto
        rumore esplorativo per facilitare la ricerca off-policy.
        """
        state_t = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            cont_action = self.actor.sample(state_t, evaluate=evaluate, noise_std=self.expl_noise)
        return cont_action.cpu().numpy()[0]

    def update(self, online_memory, elite_memory, expert_memory, batch_size, global_step):
        """
        Esegue un singolo passo di addestramento per il Critic ed (eventualmente) per l'Actor.
        
        Come funziona:
          1. Campiona un batch ibrido a 3 vie: 25% esperti (pilota umano), 15% elite (migliori prestazioni dell'agente)
             e 60% online (esplorazione corrente). Se online o elite contengono pochi dati, compensa con campioni expert.
          2. Applica una riscalatura delle ricompense (reward_scale = 0.02) per mantenere i valori Q entro un range stabile.
          3. Aggiorna il Critic (Twin Critic):
             - Calcola l'azione target per lo stato successivo aggiungendo rumore clippato (Target Policy Smoothing).
             - Estrae Q1_target(s', a') e Q2_target(s', a') dalle reti target del Critic.
             - Prende il minimo tra le due stime (per evitare sovrastime) e calcola il target di Bellman Q_target = r + gamma * min(Q1, Q2).
             - Esegue la discesa del gradiente minimizzando l'errore quadratico medio (MSE) delle stime correnti Q1 e Q2 rispetto a Q_target.
          4. Aggiorna l'Actor (Delayed Policy Update):
             - Se global_step >= 15000 (warm-up concluso) e global_step è un multiplo di policy_freq (ogni 2 passi del critic):
             - Calcola la componente RL: l'Actor massimizza il valore atteso Q1(s, pi(s)).
             - Isola i campioni del batch contrassegnati come expert (expert_mask > 0.5).
             - Calcola la BC Penalty (MSE tra l'azione predetta e quella dell'esperto umano) unicamente su questi campioni
               per evitare di forzare la policy in stati esplorativi.
             - Applica una penalità di mutua esclusione per disincentivare la pressione simultanea di acceleratore e freno.
             - Calcola il coefficiente dinamico lambda del paper: bc_alpha / mean(|Q(s, pi(s))|).
             - Combina le due loss in: Loss = dynamic_alpha * RL_Loss + BC_Penalty.
             - Esegue il backward dei gradienti sull'Actor applicando il clipping a 1.0.
          5. Aggiorna le reti target tramite Polyak Averaging con parametro tau, ogni policy_freq step
             a prescindere da warm-up e congelamento dell'Actor (come nel TD3 originale).
        """
        # Hybrid Sampling a 3 vie: Expert + Online + Elite
        b_expert = int(batch_size * 0.25)
        b_elite = min(int(batch_size * 0.15), len(elite_memory.buffer))
        b_online = min(batch_size - b_expert - b_elite, len(online_memory.buffer))
        b_expert = batch_size - b_online - b_elite  # il resto dall'expert (sempre disponibile)

        parts = [expert_memory.sample(b_expert)]

        if b_online > 0: parts.append(online_memory.sample(b_online))
        if b_elite > 0:  parts.append(elite_memory.sample(b_elite))

        state_b      = np.concatenate([p[0] for p in parts], axis=0)
        action_b     = np.concatenate([p[1] for p in parts], axis=0)
        reward_b     = np.concatenate([p[2] for p in parts], axis=0)
        next_state_b = np.concatenate([p[3] for p in parts], axis=0)
        mask_b       = np.concatenate([p[4] for p in parts], axis=0)
        expert_mask_b= np.concatenate([p[5] for p in parts], axis=0)

        # Riscalatura della ricompensa per mantenere in un range sano la magnitudo del Critic
        reward_scale = 0.02
        reward_b = reward_b * reward_scale

        state_b = torch.FloatTensor(state_b).to(self.device)
        next_state_b = torch.FloatTensor(next_state_b).to(self.device)
        action_b = torch.FloatTensor(action_b).to(self.device)
        reward_b = torch.FloatTensor(reward_b).to(self.device).unsqueeze(1)
        mask_b = torch.FloatTensor(mask_b).to(self.device).unsqueeze(1)
        expert_mask_b = torch.FloatTensor(expert_mask_b).to(self.device).unsqueeze(1)

        # Aggiornamento del Critic (Bellman equation con Twin Q-Network)
        with torch.no_grad():
            # Target Policy Smoothing (TD3): aggiungiamo rumore clippato per regolarizzare le stime Q
            noise = (torch.randn_like(action_b) * 0.2).clamp(-0.5, 0.5)
            next_action = self.actor_target(next_state_b)
            next_action = (next_action + noise).clamp(-1.0, 1.0)

            q1_next, q2_next = self.critic_target(next_state_b, next_action)
            min_q_next = torch.min(q1_next, q2_next)
            target_q = reward_b + mask_b * self.gamma * min_q_next

        q1, q2 = self.critic(state_b, action_b)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        # In refinement l'aggiornamento del Critic è disattivato:
        # l'Actor si raffina verso una value function fissa con vincolo BC ridotto.
        if not getattr(self, 'refine_mode', False):
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)  # Impedisce gradient explosion
            self.critic_optimizer.step()

        actor_loss_val = 0.0

        # Delayed Policy Update (TD3: ogni 2 step del Critic)
        # Warm-Up di 15000 step per far stabilizzare il Critic prima di aggiornare l'Actor.
        if global_step >= 15000 and global_step % self.policy_freq == 0 and not getattr(self, 'actor_frozen', False):
            pi = self.actor(state_b)
            q1_pi, _ = self.critic(state_b, pi)

            # Componente RL: massimizzazione del Q-Value stimato
            actor_loss_td3 = -q1_pi.mean()

            # BC Penalty (Masking Rigoroso: Solo su sotto-batch Expert)
            det_steer = pi[:, 0]
            det_accel = (pi[:, 1] + 1.0) / 2.0
            det_brake = (pi[:, 2] + 1.0) / 2.0
            target_steer = action_b[:, 0]
            target_accel = (action_b[:, 1] + 1.0) / 2.0
            target_brake = (action_b[:, 2] + 1.0) / 2.0

            expert_mask_flat = expert_mask_b.squeeze(1)
            expert_indices = torch.where(expert_mask_flat > 0.5)[0]

            if len(expert_indices) > 0:
                steer_loss = F.mse_loss(det_steer[expert_indices], target_steer[expert_indices])
                accel_loss = F.mse_loss(det_accel[expert_indices], target_accel[expert_indices])
                brake_loss = F.mse_loss(det_brake[expert_indices], target_brake[expert_indices])
                # Somma i contributi dei tre controlli per la loss BC
                bc_penalty = (steer_loss * 2.0 + accel_loss + brake_loss * 2.0)
            else:
                bc_penalty = torch.tensor(0.0, device=self.device)

            # Penalità per evitare acceleratore e freno premuti contemporaneamente
            mutual_exclusion_penalty = (det_accel * det_brake).mean()
            bc_penalty = bc_penalty + (mutual_exclusion_penalty * 0.1)

            # Normalizzazione λ del TD3+BC (Fujimoto & Gu, 2021).
            # bc_alpha (default 2.5, configurabile con --bc_alpha) regola il rapporto RL/BC:
            # valori piu' alti spostano il bilanciamento verso il RL, utile per superare l'esperto.
            Q_abs_mean = q1_pi.abs().mean().detach().clamp(min=1e-5)
            dynamic_alpha = self.bc_alpha / Q_abs_mean

            # Gestione del refinement (allentamento del vincolo BC su plateau)
            bc_weight = self.refine_bc_weight if getattr(self, 'refine_mode', False) else 1.0

            total_actor_loss = dynamic_alpha * actor_loss_td3 + (bc_weight * bc_penalty)

            self.actor_optimizer.zero_grad()
            total_actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)  # Impedisce gradient explosion
            self.actor_optimizer.step()
            actor_loss_val = total_actor_loss.item()

        # Soft Update (Polyak Averaging, τ=0.005) — eseguito ogni policy_freq step a prescindere
        # da warm-up e congelamento dell'Actor, come nel TD3 originale. Tenerlo dentro il ramo
        # dell'aggiornamento Actor lasciava i target del Critic congelati per decine di migliaia
        # di step (warm-up e post-rollback), facendo divergere stime correnti e target di Bellman.
        if global_step % self.policy_freq == 0:
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return critic_loss.item(), actor_loss_val, 0.0

    def save_checkpoint(self, filepath, episode, global_step, memory, elite_memory=None, best_lap_time=float('inf'), best_eval_dist=0.0, best_distance=0.0):
        """
        Salva lo stato corrente dell'agente e dei replay buffer su disco in modo atomico.
        
        Per garantire l'integrità del checkpoint ed evitare disallineamenti o corruzioni dovuti ad arresti improvvisi:
          1. Crea la cartella 'buffers/' parallela alla cartella del checkpoint.
          2. Salva i Replay Buffer (principale ed elite) in formato .npz. I buffer vengono scritti PRIMA dei pesi,
             poiché rappresentano l'operazione più onerosa in termini di I/O.
          3. Crea un dizionario contenente i pesi di Actor, Critic, le rispettive reti target, gli ottimizzatori,
             il numero dell'episodio, il global_step e le metriche di record (best_lap_time, best_eval_dist, best_distance).
          4. Salva questo dizionario in formato .pth usando safe_save (scrittura temporanea, fsync e backup rotation).
          
        Se il processo viene interrotto a metà, l'assenza del file .pth aggiornato indicherà al resume che i nuovi buffer
        non sono allineati, inducendo il sistema a ignorarli a favore dei backup temporali coerenti.
        """
        checkpoint = {
            'checkpoint_version': 2,
            'saved_at': datetime.now().isoformat(),
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'episode': episode,
            'global_step': global_step,
            'best_lap_time': best_lap_time,
            'best_eval_dist': best_eval_dist,
            'best_distance': best_distance,
        }
        buffer_dir = os.path.join(os.path.dirname(filepath), 'buffers')
        os.makedirs(buffer_dir, exist_ok=True)
        base_name = os.path.basename(filepath).replace('.pth', '')

        safe_save_npz(memory, os.path.join(buffer_dir, f"{base_name}_buffer.npz"))
        if elite_memory:
            safe_save_npz(elite_memory, os.path.join(buffer_dir, f"{base_name}_elite_buffer.npz"))
        safe_save(checkpoint, filepath)

    def load_checkpoint(self, filepath, memory, elite_memory=None):
        """
        Carica un checkpoint precedentemente salvato, ripristinando lo stato dell'agente e dei replay buffer.
        
        Gestione robusta del ripristino (Resume):
          1. Scansiona i percorsi dei candidati (incluso backups/) per trovare un file .pth leggibile.
          2. Se il checkpoint contiene la struttura completa di training, ripristina i pesi dei modelli,
             i target, gli stati degli ottimizzatori e le variabili di avanzamento (episodio, step globali).
          3. Valida le distanze memorizzate (best_eval_dist e best_distance). Se contengono valori anomali
             (fuori dal limite fisso monogiro di 3800m), tenta di ripristinarle leggendo i sidecar testuali
             compilati in parallelo ('td3_det_best_dist.txt').
          4. Se il checkpoint contiene solo pesi dell'Actor (es. modelli estratti da terze parti per testing),
             esegue un warm-start dei soli parametri di guida deterministici, azzerando ottimizzatori e buffer.
          5. Carica i Replay Buffer (principale ed elite) forzando la sincronizzazione temporale:
             - Non carica buffer con timestamp di modifica successivo a quello del checkpoint .pth (con tolleranza 1ms),
               poiché indicherebbe che il buffer appartiene ad un salvataggio successivo interrotto prima di scrivere il .pth.
             - In tal caso, scansiona i backup del buffer per trovarne uno coerente con l'epoca del checkpoint caricato.
          
        Ritorna:
            Una tupla (episode, global_step, best_lap_time, best_eval_dist, best_distance) aggiornata.
        """
        buffer_dir = os.path.join(os.path.dirname(filepath), 'buffers')
        base_name = os.path.basename(filepath).replace('.pth', '')
        buffer_path = os.path.join(buffer_dir, f"{base_name}_buffer.npz")
        elite_buffer_path = os.path.join(buffer_dir, f"{base_name}_elite_buffer.npz")

        def _load_buffer_aligned(buffer_obj, path, label, loaded_checkpoint_path=None):
            """
            Carica il buffer più recente che non sia temporalmente successivo al checkpoint .pth caricato.
            Previene il disallineamento dei dati in caso di interruzioni durante il salvataggio.
            """
            if buffer_obj is None:
                return False
            max_mtime = None
            if loaded_checkpoint_path and os.path.exists(loaded_checkpoint_path):
                max_mtime = os.path.getmtime(loaded_checkpoint_path)

            skipped_newer = []
            for candidate in _checkpoint_candidates(path):
                if not os.path.exists(candidate):
                    continue
                if max_mtime is not None and os.path.getmtime(candidate) > max_mtime + 1e-3:
                    skipped_newer.append(candidate)
                    continue
                try:
                    buffer_obj.load(candidate)
                    if candidate != path:
                        print(f"{label} recuperato dal backup coerente: {candidate}")
                    if skipped_newer:
                        print(f"{label}: ignorati file più nuovi del checkpoint caricato: {skipped_newer}")
                    return True
                except Exception as e:
                    print(f"Impossibile caricare {label} da {candidate}: {e}")

            # Recupero di emergenza: se nessun backup allineato è integro, carica il buffer più nuovo disponibile
            for candidate in skipped_newer:
                try:
                    buffer_obj.load(candidate)
                    print(f"{label}: nessun backup allineato trovato; uso {candidate} (più nuovo del checkpoint).")
                    return True
                except Exception as e:
                    print(f"Impossibile caricare {label} da {candidate}: {e}")
            return False

        loaded_ok = False
        loaded_checkpoint_path = None
        for candidate in _checkpoint_candidates(filepath):
            if not os.path.exists(candidate):
                continue
            try:
                checkpoint = torch.load(candidate, map_location=self.device, weights_only=False)
                if isinstance(checkpoint, dict) and 'actor' in checkpoint:
                    required_keys = ['actor', 'critic', 'critic_target', 'actor_optimizer', 'critic_optimizer', 'episode', 'global_step']
                    missing_keys = [k for k in required_keys if k not in checkpoint]
                    if missing_keys:
                        raise KeyError(f"checkpoint incompleto, chiavi mancanti: {missing_keys}")
                    self.actor.load_state_dict(checkpoint['actor'], strict=False)
                    if 'actor_target' in checkpoint: self.actor_target.load_state_dict(checkpoint['actor_target'], strict=False)
                    self.critic.load_state_dict(checkpoint['critic'])
                    self.critic_target.load_state_dict(checkpoint['critic_target'])
                    self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
                    self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])

                    best_lap_time = checkpoint.get('best_lap_time', float('inf'))
                    best_eval_dist = checkpoint.get('best_eval_dist', 0.0)
                    best_distance = checkpoint.get('best_distance', 0.0)
                    # best_eval_dist è uno SCORE (distanza o equivalente-tempo): la soglia di
                    # plausibilità è quella degli score, non quella della distanza monogiro.
                    if not _is_plausible_eval_score(best_eval_dist):
                        det_best_dist_txt = 'train_set/checkpoints/td3_det_best_dist.txt'
                        sidecar_best_eval_dist = safe_read_float(det_best_dist_txt, 0.0)
                        print(
                            f"best_eval_dist={best_eval_dist:.2f} non plausibile come score di eval; "
                            f"uso sidecar {sidecar_best_eval_dist:.2f}."
                        )
                        best_eval_dist = sidecar_best_eval_dist if _is_plausible_eval_score(sidecar_best_eval_dist) else 0.0
                    if not _is_plausible_eval_dist(best_distance):
                        # best_distance è una distanza fisica di esplorazione: se corrotta, si riparte
                        # dallo score clampato alla lunghezza pista (mai oltre i metri reali percorribili).
                        best_distance = min(best_eval_dist, TRACK_LENGTH_M)
                    episode = checkpoint['episode']
                    global_step = checkpoint['global_step']
                    if candidate != filepath:
                        print(f"Checkpoint principale non usato: recupero da backup {candidate}")
                    print(f"Checkpoint caricato: ripresa dall'Episodio {episode}")
                else:
                    # È un file di soli pesi dell'actor (come td3_expl_best_dist.pth).
                    print(f"{candidate} contiene solo pesi dell'Actor. Inizializzazione degli altri componenti.")
                    self.actor.load_state_dict(checkpoint, strict=False)
                    self.actor_target.load_state_dict(self.actor.state_dict())
                    best_lap_time = float('inf')
                    best_eval_dist = 0.0
                    best_distance = 0.0
                    episode = 0
                    global_step = 0
                loaded_ok = True
                loaded_checkpoint_path = candidate
                break
            except Exception as e:
                print(f"Checkpoint non utilizzabile da {candidate}: {e}")

        if not loaded_ok:
            if not any(os.path.exists(candidate) for candidate in _checkpoint_candidates(filepath)):
                return 0, 0, float('inf'), 0.0, 0.0
            print("Nessun checkpoint completo valido trovato tra principale e backup recenti.")
            print("Tentativo di recupero minimo delle informazioni dal log...")
            log_file = 'train_set/session_logs/td3_training.log'
            last_ep = 0
            if os.path.exists(log_file):
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            match = re.search(r'\b(?:Episode|Episodio|Ep)\s+(\d+)', line)
                            if match:
                                ep_num = int(match.group(1))
                                last_ep = max(last_ep, ep_num)
                except Exception:
                    pass
            print(f"Ripristinato ultimo episodio: {last_ep}. Il training riprenderà dall'episodio {last_ep + 1}.")
            episode = last_ep + 1
            global_step = episode * 1500
            best_lap_time = float('inf')

            best_eval_dist = 0.0
            det_best_dist_txt = 'train_set/checkpoints/td3_det_best_dist.txt'
            best_eval_dist = safe_read_float(det_best_dist_txt, 0.0)
            # Il sidecar contiene uno score: per la distanza fisica di esplorazione va clampato.
            best_distance = min(best_eval_dist, TRACK_LENGTH_M)

        _load_buffer_aligned(memory, buffer_path, "Replay Buffer", loaded_checkpoint_path if loaded_ok else None)
        if elite_memory:
            _load_buffer_aligned(elite_memory, elite_buffer_path, "Elite Buffer", loaded_checkpoint_path if loaded_ok else None)

        return episode, global_step, best_lap_time, best_eval_dist, best_distance

def load_recent_evals_from_log(log_path, max_len=8):
    """
    Analizza il file di log del training per estrarre gli ultimi score di valutazione registrati
    (campo 'Score' nel formato corrente; ricade sulla distanza 'Dist' per le righe storiche).
    
    Questo serve a popolare la finestra di memoria per l'Auto-Refinement all'avvio dell'agente (Resume),
    evitando che lo stato del plateau venga perso o resettato quando si riavvia il processo di training.
    
    Argomenti:
        log_path: Percorso del file di testo td3_training.log.
        max_len: Lunghezza massima della finestra temporale (default: 8 valutazioni).
        
    Ritorna:
        Una lista contenente le ultime max_len distanze valide estratte.
    """
    evals = []
    if os.path.exists(log_path):
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    # Filtra solo le righe che contengono tag di valutazione deterministica [EVAL]
                    if '[EVAL]' in line and 'Result' in line:
                        try:
                            # Preferisce il campo Score (formato nuovo: distanza o equivalente-tempo);
                            # le righe storiche senza Score ricadono sulla sola distanza.
                            match = re.search(r'\bScore\s+([0-9]+(?:\.[0-9]+)?)m', line)
                            if match:
                                eval_score = float(match.group(1))
                                if _is_plausible_eval_score(eval_score):
                                    evals.append(eval_score)
                                continue
                            match = re.search(r'\b(?:Dist|Distanza)\s+([0-9]+(?:\.[0-9]+)?)m', line)
                            if match:
                                eval_dist = float(match.group(1))
                                # Valida il dato con il filtro di plausibilità fisica monogiro
                                if _is_plausible_eval_dist(eval_dist):
                                    evals.append(eval_dist)
                        except Exception:
                            pass
        except Exception as e:
            print(f"Impossibile leggere gli eval recenti dal log: {e}")
    return evals[-max_len:]

def train():
    """
    Funzione principale che coordina il ciclo di addestramento online dell'agente TD3+BC su TORCS.
    
    Flusso di Esecuzione:
      1. Parsing degli argomenti da riga di comando (configurazione di rollback, pretrain del critic,
         congelamento dell'actor e impostazioni di refinement).
      2. Inizializzazione dell'ambiente di simulazione TorcsEnv e dei tre replay buffer
         (Online, Elite, Expert).
      3. Ripristino (Resume) dello stato dell'agente da checkpoint tramite `load_checkpoint`.
      4. Caricamento permanente in memoria del dataset di transizioni esperte umane.
      5. Ciclo principale di simulazione degli episodi:
         - Gestione del frame stacking temporale degli stati sensoriali.
         - Interazione con TORCS per ricavare stato successivo, calcolo delle marce (gearing.py) e reward.
         - Aggiornamento dei pesi dell'agente ad ogni step di simulazione.
         - Memorizzazione e riempimento differenziato dei replay buffer.
      6. Valutazione deterministica periodica dell'agente (ogni 5 episodi) con tracciamento
         dei record assoluti e salvataggio dei pesi d'azione deterministici ottimali per la submission.
      7. Macchina a stati finiti per l'Auto-Refinement (attivazione/disattivazione della refinement,
         riduzione del peso BC, gestione dei rollback e prevenzione del timeout su plateau).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--bc_weights', type=str, default='train_set/checkpoints/bc_policy.pth')
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--max_steps', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--rollback', action='store_true', help="Forza il rollback dell'Actor alla migliore policy deterministica e lo congela temporaneamente")
    parser.add_argument('--actor-freeze-episodes', '--actor_freeze_episodes', dest='actor_freeze_episodes',
                        type=int, default=30,
                        help="Numero di episodi di congelamento Actor dopo --rollback (default: 30; aumenta per dare piu' tempo al Critic)")
    parser.add_argument('--no-auto-refine', '--no_auto_refine', dest='no_auto_refine',
                        action='store_true',
                        help="Disattiva solo la refinement automatica da plateau; --refine manuale resta disponibile")
    parser.add_argument('--refine', action='store_true', help="Avvia subito la refinement: aggiornamento del Critic disattivato, loss Critic solo diagnostica, peso Behavioral Cloning ridotto")
    parser.add_argument('--pretrain_critic', action='store_true', help="Esegue il pre-training offline del Critic per 50k passi in caso di emergenza (da usare con --rollback)")
    parser.add_argument('--expert_max_lap_time', type=float, default=71.0,
                        help="Carica nel buffer expert solo i file con lap_time <= soglia (secondi); "
                             "<= 0 disattiva il filtro e carica tutti i giri (default: 71.0)")
    parser.add_argument('--bc_alpha', type=float, default=2.5,
                        help="Coefficiente alpha del TD3+BC: piu' alto = piu' peso al RL rispetto alla BC "
                             "(default: 2.5 come nel paper; 3.5-5.0 per spingere oltre l'esperto)")
    args = parser.parse_args()
    if args.actor_freeze_episodes < 0:
        parser.error("--actor-freeze-episodes deve essere >= 0")
    actor_freeze_episodes = args.actor_freeze_episodes
    auto_refine_enabled = not args.no_auto_refine

    set_seed(args.seed)

    env = TorcsEnv(early_termination=True)

    # Buffer ONLINE (FIFO) per l'esperienza dell'agente. 1M transizioni:
    # con ~700 step/episodio copre ~1400 episodi senza evizione precoce.
    memory = ReplayBuffer(1000000)
    elite_memory = ReplayBuffer(20000)

    # Buffer EXPERT SEPARATO e PERMANENTE (dati umani): capacità > dataset così non viene
    # MAI svuotato dalla FIFO. Risolve la perdita dell'ancora BC e la rende presente in ogni batch.
    expert_memory = ReplayBuffer(400000)

    # Inizializzazione Agent
    agent = TD3BCAgent()
    agent.bc_alpha = args.bc_alpha

    # Nota: i pesi BC pre-addestrati vengono caricati solo al fresh-start (blocco successivo); in caso di resume, sono ripristinati dal checkpoint.
    checkpoint_path = 'train_set/checkpoints/td3_checkpoint.pth'
    start_episode, global_step, best_lap_time, best_eval_dist, best_distance = agent.load_checkpoint(checkpoint_path, memory, elite_memory)

    # Carica i dati dell'esperto nel buffer permanente sia all'avvio che al riavvio, garantendo l'ancora BC.
    # Il filtro sul lap_time tiene solo i giri migliori del pilota: l'ancora deve puntare al suo best, non alla sua media.
    expert_lap_filter = args.expert_max_lap_time if args.expert_max_lap_time > 0 else None
    expert_memory.load_expert_data('train_set/laps', max_samples=350000, max_lap_time=expert_lap_filter)

    agent.actor_frozen = False

    # Configurazione dell'Auto-Refinement (Beeson & Montana, 2022).
    # Rileva situazioni di stallo (plateau) dei risultati e ottimizza la policy
    # disattivando l'aggiornamento del Critic e riducendo il vincolo BC.
    # Può essere disattivata tramite il flag --no-auto-refine.
    REFINE_BC_WEIGHT = 0.3                 # Peso ridotto per la penalità BC durante il refinement.
    REFINE_MAX_ATTEMPTS = 3                # Numero massimo di tentativi di refinement prima di forzare il training standard.
    REFINE_COLLAPSE_FRAC = 0.6             # Soglia di crollo (60% del valore recente) per innescare un rollback preventivo.
    REFINE_WINDOW = 8                      # Numero di valutazioni storiche considerate per la finestra mobile.
    REFINE_PLATEAU_EVALS = 4               # Numero di valutazioni consecutive senza incremento necessarie per decretare un plateau.
    REFINE_MIN_EP = 200                    # Episodio minimo richiesto per poter attivare l'auto-refinement.
    REFINE_IMPROVE_FRAC = 1.02             # Incremento minimo (+2%) per considerare una valutazione come miglioramento significativo.
    REFINE_BREAKOUT_FRAC = 1.10            # Coefficiente di superamento del plateau (+10%) per considerare un tentativo come "breakout".
    REFINE_NEW_PLATEAU_FRAC = 1.10         # Soglia (+10%) per stabilire un nuovo plateau e resettare i tentativi.
    REFINE_GOOD_EVALS_TO_CONSOLIDATE = 3  # Valutazioni positive consecutive richieste per consolidare e riattivare il Critic.
    REFINE_NEAR_BEST_MARGIN = 5.0          # Margine di vicinanza al record storico (in metri) per indurre il consolidamento immediato.
    BEST_DIST_EPS = 1.0                    # Tolleranza metrica per ignorare oscillazioni minori nei log delle distanze record.

    # Soglie per il regime "giro completato": quando il riferimento supera TRACK_LENGTH_M lo score
    # è in equivalente-tempo (1% di score ≈ 0.7s di giro), quindi le percentuali del regime distanza
    # sarebbero irraggiungibili: +10% equivarrebbe a chiedere 7 secondi di miglioramento sul giro.
    REFINE_LAP_IMPROVE_FRAC = 1.004        # +0.4% di score ≈ 0.3s di giro: miglioramento significativo.
    REFINE_LAP_BREAKOUT_FRAC = 1.01        # +1% di score ≈ 0.7s di giro: breakout dal plateau.
    REFINE_LAP_NEW_PLATEAU_FRAC = 1.01     # +1% di score: nuovo plateau, reset dei tentativi.

    def _refine_frac(reference, dist_frac, lap_frac):
        """
        Seleziona la soglia percentuale corretta in base al regime del riferimento:
        sotto TRACK_LENGTH_M lo score è una distanza (giri incompleti), sopra è in
        equivalente-tempo (giri completati) e richiede soglie molto più fini.
        """
        return lap_frac if reference > TRACK_LENGTH_M else dist_frac
    agent.refine_mode = False
    agent.refine_bc_weight = 1.0
    recent_eval_window = deque(maxlen=REFINE_WINDOW)  # Coda mobile per le ultime valutazioni.

    # Popoliamo la finestra leggendo i dati recenti direttamente dal log
    initial_evals = load_recent_evals_from_log('train_set/session_logs/td3_training.log', REFINE_WINDOW)
    for ev in initial_evals:
        recent_eval_window.append(ev)
    if len(recent_eval_window) > 0:
        print(f"Caricati {len(recent_eval_window)} eval recenti dal log: {list(recent_eval_window)}")
    if auto_refine_enabled:
        print("Auto-refinement automatica: attiva di default.")
    else:
        print("Auto-refinement automatica: disattivata da --no-auto-refine. --refine manuale resta disponibile.")

    refine_best_mean = 0.0                 # Migliore media mobile della finestra di valutazione registrata (segnale di plateau).
    if len(recent_eval_window) >= REFINE_WINDOW:
        refine_best_mean = sum(recent_eval_window) / len(recent_eval_window)

    refine_evals_no_improve = 0            # Numero di valutazioni consecutive senza incremento significativo.
    refine_attempts = 0                    # Numero totale di tentativi di refinement avviati.
    refine_attempt_plateau_ref = 0.0       # Distanza di riferimento del plateau per il tentativo corrente.
    refine_attempt_limit_logged = False    # Stato di logging per il limite massimo di tentativi.
    refine_plateau_level = 0.0             # Livello di plateau memorizzato (0.0 se non ancora stabilito).
    refine_collapse_count = 0              # Contatore dei crolli prestazionali rilevati.
    refine_evals_count = 0                 # Numero di valutazioni effettuate durante la fase di refinement.
    refine_good_eval_count = 0             # Numero di valutazioni positive consecutive post-breakout.
    refine_breakout_logged = False         # Stato di logging per il primo superamento del plateau.

    if getattr(args, 'refine', False):
        # Attivazione manuale immediata del refinement (es. se si rileva uno stallo persistente).
        agent.refine_mode = True
        agent.refine_bc_weight = REFINE_BC_WEIGHT

        # Stima del livello di plateau iniziale:
        # Se la finestra recente contiene dati sufficienti e non si è in fase di rollback,
        # si calcola la mediana. Altrimenti, si usa il miglior record storico deterministico.
        if len(recent_eval_window) >= 4 and not getattr(args, 'rollback', False):
            refine_plateau_level = float(np.median(list(recent_eval_window)))
        else:
            refine_plateau_level = best_eval_dist
            det_best_dist_txt = 'train_set/checkpoints/td3_det_best_dist.txt'
            if refine_plateau_level <= 0.0:
                refine_plateau_level = safe_read_float(det_best_dist_txt, 0.0)

        if refine_plateau_level > 0.0:
            refine_attempt_plateau_ref = refine_plateau_level
            print(f"--refine attivo: refinement attiva da subito "
                  f"(aggiornamento Critic disattivato, peso Behavioral Cloning={REFINE_BC_WEIGHT}). "
                  f"Riferimento plateau={refine_plateau_level:.0f}m. Loss Critic solo diagnostica.")
        else:
            print(f"--refine attivo: refinement attiva da subito "
                  f"(aggiornamento Critic disattivato, peso Behavioral Cloning={REFINE_BC_WEIGHT}). "
                  f"Riferimento plateau impostato dopo i primi eval (mediana recente). Loss Critic solo diagnostica.")

    batch_size = 256

    # Warm-Start: inizializzazione dell'Actor con i pesi BC se si tratta di un nuovo avvio (episodio 0).
    if start_episode == 0:
        agent.actor.load_bc_weights(args.bc_weights)
        agent.actor_target.load_state_dict(agent.actor.state_dict())
    else:
        # Procedura di rollback per il ripristino di emergenza dei pesi dell'Actor.
        # Si cerca di ripristinare il modello partendo dalla migliore policy deterministica disponibile,
        # scendendo in ordine di priorità fino alle policy esplorative se le prime sono assenti.
        if args.rollback:
            rollback_candidates = [
                'train_set/checkpoints/td3_det_best_lap.pth',      # 1) Giro valido deterministico più veloce.
                'train_set/checkpoints/td3_det_best_dist.pth',     # 2) Miglior distanza deterministica assoluta.
                'train_set/checkpoints/td3_det_best_dist_run.pth', # 3) Miglior distanza deterministica del run corrente.
                'train_set/checkpoints/td3_expl_best_lap.pth',     # 4) Miglior giro esplorativo (con rumore attivo).
                'train_set/checkpoints/td3_expl_best_dist.pth',    # 5) Miglior distanza esplorativa (con rumore attivo).
            ]
            best_path = next((p for p in rollback_candidates if os.path.exists(p)), None)
            if best_path:
                print(f"[EMERGENZA] Rollback Actor: caricamento della migliore policy deterministica da {best_path}")
                agent.actor.load_actor_weights(best_path, agent.device)
                agent.actor_target.load_state_dict(agent.actor.state_dict())
                import torch.optim as optim
                agent.actor_optimizer = optim.Adam(
                    [p for p in agent.actor.parameters() if p.requires_grad], lr=3e-4)
                
                # Congelamento temporaneo dell'Actor post-rollback per stabilizzare la convergenza del Critic.
                if actor_freeze_episodes > 0:
                    agent.actor_frozen = True
                    print(f"Actor congelato per {actor_freeze_episodes} episodi: stabilizzazione post-rollback.")
                else:
                    agent.actor_frozen = False
                    print("Congelamento Actor post-rollback disattivato (--actor-freeze-episodes 0).")

                # Pre-addestramento offline opzionale del Critic per sintonizzarne i pesi sulle transizioni del buffer.
                if getattr(args, 'pretrain_critic', False) and (len(memory) > batch_size or len(expert_memory) > batch_size):
                    print("[EMERGENZA] Pre-addestramento del Critic in corso sui dati offline del Replay Buffer (50,000 passi)...")
                    for pretrain_step in range(50000):
                        critic_loss_val, _, _ = agent.update(memory, elite_memory, expert_memory, batch_size, global_step=0)
                        if (pretrain_step + 1) % 10000 == 0:
                            print(f"  [Pre-addestramento] Passo {pretrain_step + 1}/50000 | Loss del Critic: {critic_loss_val:.4f}")
                    print("Pre-addestramento del Critic completato con successo!")
            else:
                print("Rollback richiesto ma nessun checkpoint valido trovato! Avvio ripresa normale.")
        else:
            print("Ripresa regolare dal checkpoint (nessun rollback o congelamento Actor).")

    os.makedirs('train_set/checkpoints', exist_ok=True)
    os.makedirs('train_set/session_logs', exist_ok=True)
    log_file = 'train_set/session_logs/td3_training.log'

    def _control_log(message):
        """
        Stampa un messaggio a terminale e lo scrive contemporaneamente nel file di log.
        """
        print(message)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(message + "\n")

    stop_requested = False

    def _request_stop(signum, frame):
        """
        Gestore dei segnali SIGINT (Ctrl+C) e SIGTERM per l'uscita pulita ed ordinata.

        Imposta la variabile stop_requested a True. L'episodio corrente NON viene
        interrotto: prosegue fino al suo esito naturale (giro completato, crash o
        limite di passi), così i suoi dati restano transizioni valide. Al termine
        dell'episodio il loop salva il checkpoint completo ed esce senza ripartire.
        """
        nonlocal stop_requested
        if not stop_requested:
            stop_requested = True
            print("\nRichiesta di arresto ricevuta: l'episodio corrente termina naturalmente, "
                  "poi checkpoint completo e uscita. Un secondo Ctrl+C forza l'uscita immediata.")
        else:
            print("\nSecondo Ctrl+C: uscita forzata immediata. L'ultimo checkpoint completo "
                  "resta quello salvato a fine dell'episodio precedente.")
            os._exit(130)

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    # L'attesa del server TORCS in snakeoil controlla questo hook: un Ctrl+C dato mentre
    # il client è bloccato su "Waiting for server" abortisce l'attesa con uscita pulita,
    # invece di restare appesi finché il server non compare.
    snakeoil3.abort_check = lambda: stop_requested

    # Registrazione sul file di log dei parametri di avvio selezionati per tracciare la sessione.
    if getattr(args, 'refine', False):
        with open(log_file, 'a', encoding='utf-8') as f:
            initial_ref = f"{refine_plateau_level:.0f}m" if refine_plateau_level > 0.0 else "da impostare"
            f.write(f"AVVIO con --refine: REFINEMENT armata da subito "
                    f"(aggiornamento Critic disattivato, loss Critic solo diagnostica, "
                    f"peso Behavioral Cloning={REFINE_BC_WEIGHT}, riferimento plateau={initial_ref}, "
                    f"episodio iniziale {start_episode})\n")
    if getattr(args, 'rollback', False):
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"AVVIO con --rollback: Actor congelato per {actor_freeze_episodes} episodi "
                    f"(0 = nessun congelamento), auto-refinement automatica="
                    f"{'attiva' if auto_refine_enabled else 'disattivata'}.\n")
    elif not auto_refine_enabled:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write("AVVIO con --no-auto-refine: refinement automatica disattivata; "
                    "--refine manuale resta disponibile.\n")

    batch_size = 256
    elite_threshold = 500.0

    print("Avvio training TD3+BC...")

    for episode in range(start_episode, args.episodes):
        # Annealing lineare del rumore esplorativo: da EXPL_NOISE_START a EXPL_NOISE_END
        # in EXPL_NOISE_ANNEAL_EPISODES episodi. A regime servono micro-variazioni di
        # traiettoria, non sbandate da 0.1 di sterzo a velocità di gara.
        agent.expl_noise = max(
            EXPL_NOISE_END,
            EXPL_NOISE_START - (EXPL_NOISE_START - EXPL_NOISE_END) * episode / EXPL_NOISE_ANNEAL_EPISODES
        )

        # Gestione dello scongelamento dell'Actor dopo la fase di stabilizzazione post-rollback
        if agent.actor_frozen and episode >= start_episode + actor_freeze_episodes:
            agent.actor_frozen = False
            print(f"Actor scongelato dopo {actor_freeze_episodes} episodi: "
                  f"riavvio aggiornamenti Actor con gradienti del Critic stabilizzati.")

        # Reset soft di default; relaunch completo periodico (vedi RELAUNCH_EVERY_EPISODES).
        full_relaunch = (episode % RELAUNCH_EVERY_EPISODES == RELAUNCH_EPISODE_OFFSET)
        try:
            ob = env.reset(relaunch=full_relaunch)
        except snakeoil3.ServerTimeoutError as e:
            if e.aborted:
                _control_log(f"[{datetime.now().strftime('%H:%M:%S')}] STOP richiesto durante "
                             f"l'attesa del server TORCS: uscita pulita (ultimo checkpoint "
                             f"completo: episodio {episode}).")
                break
            raise
        episode_transitions = []

        # Frame Stacking: concatenazione di 3 frame temporali distanziati (t-12, t-6, t)
        # per fornire informazioni sulla dinamica temporale (velocità e accelerazione).
        f_state = flatten_state(ob)
        state_stack = deque([f_state]*13, maxlen=13)
        stacked_state = np.concatenate([state_stack[0], state_stack[6], state_stack[12]])

        episode_reward, step, max_dist = 0, 0, 0.0
        critic_losses, actor_losses = [], []
        termination_reason = "TIMEOUT"
        new_record = False

        # Configurazione iniziale delle marce e delle variabili di telemetria.
        current_gear = 1
        steps_since_shift = 999  # consenti il primo cambio subito
        cur_speed_kmh = float(np.array(ob.get('speedX', 0.0)).flat[0]) * 50.0
        cur_rpm = float(np.array(ob.get('rpm', 0.0)).flat[0])
        # Riferimento del tempo dell'ultimo giro per rilevare il completamento del tracciato.
        prev_last_lap = float(np.array(ob.get('lastLapTime', 0.0)).flat[0])
        torcs_lap_time = float(np.array(ob.get('curLapTime', 0.0)).flat[0])
        completed_lap_time = None
        episode_start_dist = float(np.array(ob.get('distFromStart', 0.0)).flat[0])

        while True:
            # Selezione dell'azione tramite la policy corrente (con rumore esplorativo).
            cont_action = agent.select_action(stacked_state, evaluate=False)

            # Mappatura delle azioni continue: conversione di acceleratore e freno da [-1, 1] a [0, 1].
            env_action = np.zeros(4)
            env_action[0:3] = cont_action

            torcs_action = env_action.copy()
            torcs_action[1] = np.clip((torcs_action[1] + 1.0) / 2.0, 0.0, 1.0)  # accel: [-1,1] → [0,1]
            torcs_action[2] = np.clip((torcs_action[2] + 1.0) / 2.0, 0.0, 1.0)  # brake: [-1,1] → [0,1]
            # Mutual exclusion continua/moltiplicativa per prevenire stalli repentini
            torcs_action[1] = torcs_action[1] * (1.0 - torcs_action[2])

            # Calcolo algoritmico della marcia ottimale in base a velocità, giri al minuto e acceleratore.
            current_gear, _shifted = compute_gear(cur_speed_kmh, torcs_action[1], cur_rpm, current_gear, steps_since_shift)
            steps_since_shift = 0 if _shifted else steps_since_shift + 1
            torcs_action[3] = current_gear
            env_action[3] = current_gear

            next_ob, reward, env_done, info = env.step(torcs_action)
            cur_speed_kmh = float(np.array(next_ob.get('speedX', 0.0)).flat[0]) * 50.0
            cur_rpm = float(np.array(next_ob.get('rpm', 0.0)).flat[0])
            next_f_state = flatten_state(next_ob)
            state_stack.append(next_f_state)

            current_track_pos_m = float(np.array(next_ob.get('distFromStart', 0.0)).flat[0])
            current_dist = _track_progress_from_start(episode_start_dist, current_track_pos_m)
            last_lap_time = float(np.array(next_ob.get('lastLapTime', 0.0)).flat[0])
            torcs_lap_time = float(np.array(next_ob.get('curLapTime', 0.0)).flat[0])
            max_dist = max(max_dist, current_dist)

            lap_completed = bool(info.get('lap_completed', False))
            if not lap_completed:
                lap_completed = last_lap_time > 0.0 and abs(last_lap_time - prev_last_lap) > 0.01 and step > 500

            done = False
            if lap_completed and not info.get('crash', False):
                done, termination_reason = True, "SUCCESS"
                completed_lap_time = last_lap_time
                max_dist = max(max_dist, TRACK_LENGTH_M)
                # Bonus fisso di completamento + bonus proporzionale al tempo: la reward di progresso
                # da sola produce un ritorno per giro quasi costante (la distanza è fissa), quindi
                # senza questo termine un giro da 70s e uno da 85s sarebbero premiati quasi uguale.
                reward += LAP_SUCCESS_BONUS + LAP_TIME_BONUS_PER_S * max(0.0, LAP_TIME_BONUS_REF_S - last_lap_time)
                if last_lap_time < best_lap_time:
                    best_lap_time = last_lap_time
                    new_record = True
                    safe_save(agent.actor.state_dict(), 'train_set/checkpoints/td3_expl_best_lap.pth')

            if info.get('crash', False):
                done, termination_reason = True, "CRASH"


            if max_dist > best_distance and max_dist > 500.0:
                best_distance = max_dist
                safe_save(agent.actor.state_dict(), 'train_set/checkpoints/td3_expl_best_dist.pth')

            next_stacked_state = np.concatenate([state_stack[0], state_stack[6], state_stack[12]])

            time_limit_reached = (step >= args.max_steps)
            episode_finishes_now = done or env_done or time_limit_reached
            incomplete_lap = episode_finishes_now and termination_reason != "SUCCESS"
            if incomplete_lap:
                if termination_reason == "TIMEOUT":
                    termination_reason = "INCOMPLETE"
                reward -= INCOMPLETE_LAP_PENALTY

            mask = 0.0 if incomplete_lap else 1.0
            episode_transitions.append((stacked_state, cont_action, reward, next_stacked_state, mask))

            stacked_state = next_stacked_state
            episode_reward += reward
            step += 1
            global_step += 1

            # Update Frequency 1:1: un aggiornamento ad ogni step di simulazione (standard TD3).
            # L'expert buffer è sempre pieno → si aggiorna fin dal primo step (il Critic si
            # pre-allena sui dati umani, come la fase offline del TD3+BC) e l'ancora BC è garantita.
            if len(expert_memory) > batch_size:
                critic_loss_val, actor_loss_val, _ = agent.update(memory, elite_memory, expert_memory, batch_size, global_step)
                critic_losses.append(critic_loss_val)
                if actor_loss_val != 0.0:
                    actor_losses.append(actor_loss_val)

            if done or env_done or time_limit_reached:
                for t in episode_transitions:
                    memory.push(t[0], t[1], t[2], t[3], t[4], expert=0.0)

                # Elite Buffer Injection:
                # Se la distanza percorsa supera la soglia (70% del record di distanza corrente),
                # le transizioni dell'episodio vengono inserite nell'Elite Buffer per la Self-Imitation.
                # Per evitare di memorizzare comportamenti errati (Causal Confusion), gli ultimi 50 passi
                # prima di un crash non vengono contrassegnati come dati validi per l'imitazione.
                if max_dist >= elite_threshold:
                    n_trans = len(episode_transitions)
                    for i, t in enumerate(episode_transitions):
                        is_danger = (termination_reason == "CRASH") and (i >= n_trans - 50)
                        elite_memory.push(t[0], t[1], t[2], t[3], t[4], expert=0.0 if is_danger else 1.0)
                    elite_threshold = max(500.0, best_distance * 0.7)  # Soglia monotonicamente crescente
                break

        # Rilevamento del tempo sul giro fornito dai sensori di TORCS.
        lap_time = completed_lap_time if completed_lap_time is not None else torcs_lap_time
        time_str = datetime.now().strftime("%H:%M:%S")
        avg_critic_loss = np.mean(critic_losses) if len(critic_losses) > 0 else 0.0
        avg_actor_loss = np.mean(actor_losses) if len(actor_losses) > 0 else 0.0
        critic_status = "OFF" if getattr(agent, 'refine_mode', False) else "ON"
        if global_step < 15000:
            actor_status = "WARM"
        elif getattr(agent, 'actor_frozen', False):
            actor_status = "FREEZE"
        else:
            actor_status = "ON"
        log_msg = (f"[{time_str}] Ep {episode+1:03d} | [{termination_reason}] | "
                   f"Reward: {episode_reward:7.1f} | Steps: {step:4d} | "
                   f"LapTime: {lap_time:5.1f}s | Dist: {int(max_dist):5d}m | "
                   f"CriticL: {avg_critic_loss:.3f} ({critic_status}) | "
                   f"ActorL: {avg_actor_loss:.3f} ({actor_status})")
        if new_record: log_msg += f" | Record"
        print(f" {log_msg}")

        with open(log_file, 'a', encoding='utf-8') as f: f.write(log_msg + "\n")

        agent.save_checkpoint(checkpoint_path, episode + 1, global_step, memory, elite_memory,
                              best_lap_time=best_lap_time,
                              best_eval_dist=best_eval_dist,
                              best_distance=best_distance)
        safe_save(agent.actor.state_dict(), 'train_set/checkpoints/td3_policy.pth')
        if stop_requested:
            _control_log(f"[{datetime.now().strftime('%H:%M:%S')}] STOP richiesto: "
                         f"checkpoint completo salvato all'episodio {episode + 1}; uscita pulita.")
            break

        if (episode + 1) % 5 == 0 and global_step > 15000:
            def _rlog(m):
                """
                Helper specifico per la valutazione (EVAL) che stampa il messaggio a video
                e lo appende al file di log principale.
                """
                print(m)
                with open(log_file, 'a', encoding='utf-8') as f: f.write(m + "\n")
            def _run_deterministic_eval():
                """
                Esegue un singolo episodio di valutazione deterministica (senza rumore).

                Ritorna una tupla (eval_dist, eval_lap_time, eval_reward); eval_lap_time è None
                se il giro non è stato completato validamente.
                """
                eval_ob = env.reset(relaunch=True)
                eval_stack = deque([flatten_state(eval_ob)]*13, maxlen=13)
                eval_stacked = np.concatenate([eval_stack[0], eval_stack[6], eval_stack[12]])
                eval_dist, eval_step, eval_reward = 0.0, 0, 0.0
                eval_lap_completed = False
                eval_lap_time = None
                eval_current_gear = 1  # Utilizzo della marcia algoritmica per la valutazione.
                eval_steps_since_shift = 999
                eval_cur_speed_kmh = float(np.array(eval_ob.get('speedX', 0.0)).flat[0]) * 50.0
                eval_cur_rpm = float(np.array(eval_ob.get('rpm', 0.0)).flat[0])
                # Cronometraggio del miglior giro VALIDO completato in questa valutazione deterministica.
                eval_prev_last_lap = float(np.array(eval_ob.get('lastLapTime', 0.0)).flat[0])
                eval_start_dist = float(np.array(eval_ob.get('distFromStart', 0.0)).flat[0])

                agent.actor.eval()
                while eval_step < args.max_steps:
                    eval_step += 1
                    with torch.no_grad():
                        eval_action = agent.select_action(eval_stacked, evaluate=True)
                    eval_env = np.zeros(4)
                    eval_env[0:3] = eval_action
                    eval_env[1], eval_env[2] = np.clip((eval_env[1]+1)/2, 0, 1), np.clip((eval_env[2]+1)/2, 0, 1)
                    # Mutual exclusion continua/moltiplicativa per EVAL
                    eval_env[1] = eval_env[1] * (1.0 - eval_env[2])
                    # Calcolo della marcia ottimale per la fase di valutazione.
                    eval_current_gear, _esh = compute_gear(eval_cur_speed_kmh, eval_env[1], eval_cur_rpm, eval_current_gear, eval_steps_since_shift)
                    eval_steps_since_shift = 0 if _esh else eval_steps_since_shift + 1
                    eval_env[3] = eval_current_gear

                    eval_ob, eval_r, eval_done, eval_info = env.step(eval_env)
                    eval_cur_speed_kmh = float(np.array(eval_ob.get('speedX', 0.0)).flat[0]) * 50.0
                    eval_cur_rpm = float(np.array(eval_ob.get('rpm', 0.0)).flat[0])
                    eval_reward += eval_r
                    eval_stack.append(flatten_state(eval_ob))
                    eval_stacked = np.concatenate([eval_stack[0], eval_stack[6], eval_stack[12]])
                    current_eval_track_pos_m = float(np.array(eval_ob.get('distFromStart', 0.0)).flat[0])
                    current_eval_dist = _track_progress_from_start(eval_start_dist, current_eval_track_pos_m)
                    eval_dist = max(eval_dist, current_eval_dist)

                    # Arresto anticipato della valutazione al completamento del primo giro valido.
                    eval_last_lap = float(np.array(eval_ob.get('lastLapTime', 0.0)).flat[0])
                    eval_lap_completed = bool(eval_info.get('lap_completed', False))
                    if not eval_lap_completed:
                        eval_lap_completed = eval_last_lap > 0.0 and abs(eval_last_lap - eval_prev_last_lap) > 0.01 and eval_step > 500
                    if eval_lap_completed and not eval_info.get('crash', False):
                        eval_prev_last_lap = eval_last_lap
                        eval_lap_time = eval_last_lap
                        eval_dist = max(eval_dist, TRACK_LENGTH_M)
                        break

                    if eval_info.get('crash', False) or eval_done: break
                agent.actor.train()

                if not _is_plausible_eval_dist(eval_dist):
                    _rlog(
                        f"  [EVAL] distanza {eval_dist:.1f}m non plausibile per un eval monogiro; "
                        "scartata da record/refinement."
                    )
                    eval_dist = 0.0
                return eval_dist, eval_lap_time, eval_reward

            _rlog("\n   [EVAL] Valutazione deterministica...")
            # Run singolo: policy deterministica + simulatore deterministico danno risultati
            # riproducibili al metro (verificato empiricamente: run ripetuti sempre identici,
            # anche sui crash anomali), quindi il best-of-N non aggiunge informazione.
            try:
                eval_dist, eval_lap_time, eval_reward = _run_deterministic_eval()
            except snakeoil3.ServerTimeoutError as e:
                if e.aborted:
                    _control_log(f"[{datetime.now().strftime('%H:%M:%S')}] STOP richiesto durante "
                                 f"l'attesa del server TORCS in eval: uscita pulita (ultimo "
                                 f"checkpoint completo: episodio {episode + 1}).")
                    break
                raise
            eval_score = _eval_score(eval_dist, eval_lap_time)
            eval_best_lap_in_run = eval_lap_time if eval_lap_time is not None else float('inf')

            refine_status = ""
            if getattr(agent, 'refine_mode', False):
                riferimento_plateau = f"{refine_plateau_level:.0f}m" if refine_plateau_level > 0.0 else "none"
                refine_status = (f" | Refine: ON (BC={agent.refine_bc_weight:.1f}, "
                                 f"Critic=OFF, plateau_ref={riferimento_plateau})")
            else:
                refine_status = " | Refine: OFF"

            lap_status = f" | Lap: {eval_lap_time:.3f}s" if eval_lap_time is not None else ""
            eval_msg = (f"[{time_str}]  [EVAL] Result: Dist {int(eval_dist)}m{lap_status} | "
                        f"Score {eval_score:.0f}m | Reward: {eval_reward:.1f}{refine_status}")
            print(f"  {eval_msg}")
            with open(log_file, 'a', encoding='utf-8') as f: f.write(eval_msg + "\n")

            # best_eval_dist contiene lo SCORE (nome mantenuto per compatibilità con i checkpoint):
            # sotto 3608 coincide con la distanza, sopra cresce al migliorare del tempo sul giro.
            if eval_score > best_eval_dist:
                best_eval_dist = eval_score
                safe_save(agent.actor.state_dict(), 'train_set/checkpoints/td3_det_best_dist_run.pth')

            # Salvataggio persistente del record storico di score deterministico assoluto
            # (distanza per giri incompleti, equivalente-tempo per giri completati).
            det_best_dist_pth = 'train_set/checkpoints/td3_det_best_dist.pth'
            det_best_dist_txt = 'train_set/checkpoints/td3_det_best_dist.txt'
            prev_det_best_dist = safe_read_float(det_best_dist_txt, 0.0)
            if eval_score > prev_det_best_dist + BEST_DIST_EPS:
                safe_save(agent.actor.state_dict(), det_best_dist_pth)
                safe_write_text(det_best_dist_txt, f"{eval_score:.2f}")
                msg = (f"  NUOVO MIGLIOR DETERMINISTICO ASSOLUTO: score {int(eval_score)}m "
                       f"(precedente {int(prev_det_best_dist)}m, preservato anche dopo --clean)")
                print(msg)
                with open(log_file, 'a', encoding='utf-8') as f: f.write(msg + "\n")
                if getattr(agent, 'refine_mode', False):
                    agent.refine_mode = False
                    agent.refine_bc_weight = 1.0
                    agent.actor_frozen = actor_freeze_episodes > 0
                    start_episode = episode  # Congela l'Actor per la finestra configurata a partire da ora
                    refine_evals_no_improve = 0
                    refine_collapse_count = 0
                    refine_best_mean = 0.0
                    refine_plateau_level = 0.0
                    refine_attempts = 0
                    refine_attempt_plateau_ref = 0.0
                    refine_attempt_limit_logged = False
                    refine_evals_count = 0
                    refine_good_eval_count = 0
                    recent_eval_window.clear()
                    _rlog("  REFINEMENT CONCLUSA CON SUCCESSO! Nuovo record deterministico rilevato.")
                    if actor_freeze_episodes > 0:
                        _rlog(f"  Rientro in modalità allineamento Critic: "
                              f"Actor congelato per {actor_freeze_episodes} episodi.")
                    else:
                        _rlog("  Rientro in training normale: congelamento Actor disattivato.")

            # Salvataggio persistente del miglior giro deterministico valido (candidato per la submission).
            if eval_best_lap_in_run < float('inf'):
                det_best_lap_pth = 'train_set/checkpoints/td3_det_best_lap.pth'
                det_best_lap_txt = 'train_set/checkpoints/td3_det_best_lap.txt'
                prev_det_best_lap = safe_read_float(det_best_lap_txt, float('inf'))
                if eval_best_lap_in_run < prev_det_best_lap:
                    safe_save(agent.actor.state_dict(), det_best_lap_pth)
                    safe_write_text(det_best_lap_txt, f"{eval_best_lap_in_run:.3f}")
                    msg = f"  NUOVO MIGLIOR GIRO VALIDO (eval deterministica): {eval_best_lap_in_run:.3f}s (preservato anche dopo --clean)"
                    print(msg)
                    with open(log_file, 'a', encoding='utf-8') as f: f.write(msg + "\n")

            # Gestione della macchina a stati dell'Auto-Refinement (attivazione refinement o rollback).
            # Se l'Actor è congelato, le valutazioni vengono ignorate ai fini del calcolo del plateau.
            actor_is_frozen = getattr(agent, 'actor_frozen', False)
            if actor_is_frozen and not agent.refine_mode:
                refine_evals_no_improve = 0
                refine_best_mean = 0.0
                recent_eval_window.clear()
                _rlog("  Auto-refinement sospesa: Actor congelato; eval ignorato per il plateau.")
            else:
                recent_eval_window.append(eval_score)

            if auto_refine_enabled and not agent.refine_mode and not actor_is_frozen:
                # Rilevamento PLATEAU su STATISTICA (non sul singolo best, robusto ai colpi di
                # fortuna): la MEDIA della finestra recente smette di salire. Serve la finestra piena.
                if len(recent_eval_window) >= REFINE_WINDOW:
                    cur_mean = sum(recent_eval_window) / len(recent_eval_window)
                    if cur_mean > refine_best_mean * _refine_frac(refine_best_mean, REFINE_IMPROVE_FRAC, REFINE_LAP_IMPROVE_FRAC):
                        refine_best_mean = cur_mean          # la performance tipica sta ancora salendo
                        refine_evals_no_improve = 0
                    else:
                        refine_evals_no_improve += 1          # tipica ferma → conta verso il plateau
                    if refine_evals_no_improve >= REFINE_PLATEAU_EVALS and (episode + 1) >= REFINE_MIN_EP:
                        # Riferimento plateau = MEDIANA recente (modo 'buono' del bimodale), non il singolo max stocastico.
                        candidate_plateau_level = float(np.median(list(recent_eval_window)))
                        reset_msg = None
                        if refine_attempt_plateau_ref <= 0.0:
                            refine_attempt_plateau_ref = candidate_plateau_level
                        elif candidate_plateau_level > refine_attempt_plateau_ref * _refine_frac(refine_attempt_plateau_ref, REFINE_NEW_PLATEAU_FRAC, REFINE_LAP_NEW_PLATEAU_FRAC):
                            previous_plateau_ref = refine_attempt_plateau_ref
                            refine_attempts = 0
                            refine_attempt_plateau_ref = candidate_plateau_level
                            refine_attempt_limit_logged = False
                            reset_msg = (f"  Nuovo plateau rilevato: riferimento {candidate_plateau_level:.0f}m "
                                         f"> riferimento attuale {previous_plateau_ref:.0f}m "
                                         f"(+{(candidate_plateau_level / previous_plateau_ref - 1.0) * 100:.0f}%). "
                                         "Contatore refinement azzerato per il nuovo regime.")

                        if refine_attempts < REFINE_MAX_ATTEMPTS:
                            agent.refine_mode = True
                            agent.refine_bc_weight = REFINE_BC_WEIGHT
                            refine_plateau_level = candidate_plateau_level
                            refine_collapse_count = 0
                            refine_evals_count = 0
                            refine_good_eval_count = 0
                            refine_breakout_logged = False
                            refine_attempt_limit_logged = False
                            if reset_msg is not None:
                                _rlog(reset_msg)
                            _rlog(f"  AUTO-REFINEMENT ATTIVA (tentativo {refine_attempts+1}/{REFINE_MAX_ATTEMPTS} "
                                  f"sul plateau {refine_attempt_plateau_ref:.0f}m): "
                                  f"media recente in plateau a {cur_mean:.0f}m, aggiornamento Critic disattivato, "
                                  f"loss Critic solo diagnostica, peso Behavioral Cloning→{REFINE_BC_WEIGHT}, "
                                  f"riferimento plateau (mediana)={refine_plateau_level:.0f}m "
                                  f"(max recente={max(recent_eval_window):.0f}m)")
                        elif not refine_attempt_limit_logged:
                            refine_attempt_limit_logged = True
                            _rlog(f"  AUTO-REFINEMENT non riattivata: limite {REFINE_MAX_ATTEMPTS}/{REFINE_MAX_ATTEMPTS} "
                                  f"raggiunto per il plateau {refine_attempt_plateau_ref:.0f}m. "
                                  "Training normale finché non emerge un plateau più alto.")
            elif agent.refine_mode and refine_plateau_level <= 0.0:
                # Determinazione del livello di plateau iniziale per il refinement manuale tramite mediana.
                if len(recent_eval_window) >= 4:
                    refine_plateau_level = float(np.median(list(recent_eval_window)))
                    if refine_attempt_plateau_ref <= 0.0:
                        refine_attempt_plateau_ref = refine_plateau_level
                    refine_evals_count = 0
                    refine_good_eval_count = 0
                    _rlog(f"  refinement: riferimento plateau = {refine_plateau_level:.0f}m "
                          f"(MEDIANA degli ultimi {len(recent_eval_window)} eval, max={max(recent_eval_window):.0f}m)")
            elif agent.refine_mode:
                # In REFINEMENT.
                refine_evals_count += 1
                # Rilevamento e logging del superamento del plateau di riferimento (breakout).
                # Soglia a doppio regime: +10% in regime distanza, +1% (≈0.7s di giro) in regime tempo.
                breakout_frac = _refine_frac(refine_plateau_level, REFINE_BREAKOUT_FRAC, REFINE_LAP_BREAKOUT_FRAC)
                breakout_detected = eval_score > refine_plateau_level * breakout_frac
                if breakout_detected:
                    refine_good_eval_count += 1
                else:
                    refine_good_eval_count = 0
                if not refine_breakout_logged and breakout_detected:
                    refine_breakout_logged = True
                    _rlog(f"  PLATEAU SUPERATO: eval score {eval_score:.0f}m "
                          f"> riferimento {refine_plateau_level:.0f}m (+{(eval_score/refine_plateau_level-1)*100:.1f}%)")
                # Ripristino del training normale (con consolidamento dei pesi ed eventuale congelamento dell'Actor)
                # se il breakout è stabile o vicino al miglior record assoluto.
                near_best_breakout = breakout_detected and prev_det_best_dist > 0.0 and eval_score >= prev_det_best_dist - REFINE_NEAR_BEST_MARGIN
                stable_breakout = refine_good_eval_count >= REFINE_GOOD_EVALS_TO_CONSOLIDATE
                if near_best_breakout or stable_breakout:
                    ref_lvl = refine_plateau_level
                    good_eval_count = refine_good_eval_count
                    agent.refine_mode = False
                    agent.refine_bc_weight = 1.0
                    agent.actor_frozen = actor_freeze_episodes > 0
                    start_episode = episode
                    refine_evals_no_improve = 0
                    refine_collapse_count = 0
                    refine_best_mean = 0.0
                    refine_plateau_level = 0.0
                    refine_attempts = 0
                    refine_attempt_plateau_ref = 0.0
                    refine_attempt_limit_logged = False
                    refine_evals_count = 0
                    refine_good_eval_count = 0
                    recent_eval_window.clear()
                    if near_best_breakout:
                        _rlog(f"  REFINEMENT CONSOLIDATA: breakout vicino al miglior deterministico "
                              f"(eval score {eval_score:.0f}m, best {prev_det_best_dist:.0f}m). "
                              "Peso Behavioral Cloning→1.0, aggiornamento Critic riattivato.")
                    else:
                        _rlog(f"  REFINEMENT CONSOLIDATA: {good_eval_count} eval buone consecutive "
                              f"sopra il riferimento plateau (ultima {eval_score:.0f}m, riferimento {ref_lvl:.0f}m). "
                              "Peso Behavioral Cloning→1.0, aggiornamento Critic riattivato.")
                    if actor_freeze_episodes > 0:
                        _rlog(f"  Rientro in modalità allineamento Critic: "
                              f"Actor congelato per {actor_freeze_episodes} episodi.")
                    else:
                        _rlog("  Rientro in training normale: congelamento Actor disattivato.")
                # Attivazione del rollback preventivo in caso di crollo prestazionale prolungato.
                elif eval_score < refine_plateau_level * REFINE_COLLAPSE_FRAC:
                    refine_collapse_count += 1
                else:
                    refine_collapse_count = 0
                if refine_collapse_count >= 3:
                    ref_lvl = refine_plateau_level
                    if refine_attempt_plateau_ref <= 0.0:
                        refine_attempt_plateau_ref = ref_lvl
                    if os.path.exists(det_best_dist_pth):
                        agent.actor.load_actor_weights(det_best_dist_pth, agent.device)
                        agent.actor_target.load_state_dict(agent.actor.state_dict())
                    agent.refine_mode = False
                    agent.refine_bc_weight = 1.0
                    refine_attempts += 1
                    refine_evals_no_improve = 0
                    refine_collapse_count = 0
                    refine_evals_count = 0
                    refine_good_eval_count = 0
                    refine_best_mean = 0.0          # Reset dei parametri per misurare nuovamente il plateau.
                    refine_plateau_level = 0.0
                    _rlog(f"  REFINEMENT collassata (<{int(REFINE_COLLAPSE_FRAC*100)}% di {ref_lvl:.0f}m) "
                          f"→ ROLLBACK al miglior deterministico (td3_det_best_dist), peso Behavioral Cloning→1.0, "
                          f"aggiornamento Critic riattivato. Tentativi sul plateau {refine_attempt_plateau_ref:.0f}m: "
                          f"{refine_attempts}/{REFINE_MAX_ATTEMPTS}")
                # Uscita per timeout se il refinement si protrae per 40 episodi senza miglioramenti.
                elif refine_evals_count >= 8:
                    ref_lvl = refine_plateau_level
                    if refine_attempt_plateau_ref <= 0.0:
                        refine_attempt_plateau_ref = ref_lvl
                    agent.refine_mode = False
                    agent.refine_bc_weight = 1.0
                    refine_attempts += 1
                    refine_evals_no_improve = 0
                    refine_collapse_count = 0
                    refine_evals_count = 0
                    refine_good_eval_count = 0
                    refine_best_mean = 0.0
                    refine_plateau_level = 0.0
                    _rlog(f"  TIMEOUT REFINEMENT (40 episodi in refinement senza superare il record) "
                          f"→ Uscita automatica, peso Behavioral Cloning→1.0, aggiornamento Critic riattivato. "
                          f"Tentativi sul plateau {refine_attempt_plateau_ref:.0f}m: "
                          f"{refine_attempts}/{REFINE_MAX_ATTEMPTS}")

        if stop_requested:
            _control_log(f"[{datetime.now().strftime('%H:%M:%S')}] STOP richiesto durante/ dopo eval: "
                         f"ultimo checkpoint completo episodio {episode + 1}; uscita pulita.")
            break

    env.end()

if __name__ == '__main__':
    train()
