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
     - Bonus di Fine Giro (aggiunto in TD3+BC): `+50.0` se il traguardo viene tagliato regolarmente con successo.
       Il bonus proporzionale al tempo (`+10.0` per ogni secondo sotto gli 80s) e il bonus di record personale
       sono attivi SOLO in time-attack: in stabilizzazione il completamento è premiato in modo piatto, così un
       giro lento ma pulito vale quanto uno veloce ma rischioso e la policy impara prima a chiudere il giro.
     - Malus Giro Incompleto (aggiunto in TD3+BC): `-25.0` se l'episodio termina prematuramente per sbandata o crash,
       scoraggiando la guida imprudente a favore del completamento del circuito.
     - Penalità di Corridoio (STABILIZZAZIONE): penalità quadratica per-step che scatta già a `|trackPos| > 0.80`,
       cioè PRIMA di uscire dalla superficie di guida, per insegnare un margine di sicurezza dal bordo. Piena in
       stabilizzazione (giri completi affidabili), ridotta in time-attack (serve usare tutta la pista). Vedi
       `reward.margin_penalty`.
     - Reward Telemetrica a Settori (TIME-ATTACK): la pista è divisa in settori; alla chiusura di ognuno si premia
       (o penalizza) l'agente in base a quanto batte il proprio miglior tempo-settore. Segnale denso che indica
       DOVE guadagnare tempo, con log del "giro ideale teorico" (somma dei migliori parziali). Vedi `sector_timer`.
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

# ── Iterative Motors: package (ambiente, utility, reti condivise) ─────────
_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
from iterative_motors.env.gym_torcs import TorcsEnv
from iterative_motors.env import snakeoil3_gym as snakeoil3
from iterative_motors.env.gearing import compute_gear  # cambio marcia algoritmico
from iterative_motors.common.constants import TRACK_LENGTH_M, LAPS_AUTO_DIR, SESSION_LOGS_DIR
from iterative_motors.common.checkpoint import (
    safe_save, safe_write_text, safe_read_float, safe_save_npz,
    _fsync_file, _fsync_dir, _backup_paths, _rotate_backup, _checkpoint_candidates,
)
from iterative_motors.common.state import apply_state_norm, flatten_state_norm as flatten_state
from iterative_motors.models.networks import Actor, Critic
from iterative_motors.data.replay_buffer import ReplayBuffer
from iterative_motors.data.lap_recorder import LapRecorder
from iterative_motors.rl.reward import (
    LAP_SUCCESS_BONUS, INCOMPLETE_LAP_PENALTY, EVAL_DISTANCE_SANITY_LIMIT,
    LAP_TIME_BONUS_REF_S, LAP_TIME_BONUS_PER_S, EVAL_SCORE_T_REF_S, EVAL_SCORE_SANITY_LIMIT,
    _is_plausible_eval_dist, _is_plausible_eval_score, _eval_score, _track_progress_from_start,
    personal_best_bonus, TIME_ATTACK_BC_ALPHA, TIME_ATTACK_NOISE_FLOOR, TIME_ATTACK_ENTRY_S,
    margin_penalty, MARGIN_PENALTY_COEF,
    TA_SECTORS_DEFAULT, TA_SECTOR_REWARD_K, TA_SECTOR_REWARD_CAP,
)
from iterative_motors.rl.agent import TD3BCAgent
from iterative_motors.rl.sector_timer import SectorTimer


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
    parser.add_argument('--trust_region', type=float, default=0.0,
                        help="Peso FISSO della trust region verso le azioni del buffer sui campioni "
                             "non-expert (0 = off; ~0.3 ancora l'Actor al supporto dati e cura il "
                             "collasso della policy quando l'Actor torna attivo)")
    parser.add_argument('--reseed_elite_max_lap_time', type=float, default=0.0,
                        help="Semina UNA TANTUM il buffer elite con i giri auto-registrati "
                             "(train_set/laps_auto) con lap_time <= soglia (0 = off). Rompe la "
                             "starvation dell'elite dando alla self-imitation giri completi da imitare; "
                             "i semi lenti invecchiano ed escono man mano che la policy cattura giri "
                             "più veloci. Passalo SOLO al primo lancio di semina (poi è nel checkpoint).")
    parser.add_argument('--capture_eval_elite', action='store_true',
                        help="Cattura nell'elite i giri COMPLETI dell'eval deterministica (la linea "
                             "pulita/veloce). Default OFF: in fase di stabilizzazione self-imitare la "
                             "linea-rasoio sovra-affila la policy e destabilizza l'esplorazione; tienila "
                             "spenta finché l'esplorazione non è stabile, riattivala nella limatura.")
    args = parser.parse_args()
    if args.actor_freeze_episodes < 0:
        parser.error("--actor-freeze-episodes deve essere >= 0")
    if args.trust_region < 0:
        parser.error("--trust_region deve essere >= 0")
    actor_freeze_episodes = args.actor_freeze_episodes
    auto_refine_enabled = not args.no_auto_refine

    set_seed(args.seed)

    env = TorcsEnv(early_termination=True)

    # Buffer ONLINE (FIFO) per l'esperienza dell'agente. 2M transizioni: orizzonte raddoppiato per
    # trattenere più storia recente prima dell'evizione (richiesta utente: "non dimenticare il vecchio").
    # Le proporzioni del batch restano 25/15/60: la capienza decide solo quanto orizzonte tiene l'online.
    memory = ReplayBuffer(2000000)
    # Buffer ELITE (self-imitation): capienza aumentata 20k→200k per ospitare una collezione VARIA di
    # giri buoni — semina iniziale (--reseed_elite_max_lap_time) + catture dei giri completi (eval e
    # online) — senza che la FIFO evicchi troppo presto. ~110 giri interi di respiro.
    elite_memory = ReplayBuffer(200000)

    # Buffer EXPERT SEPARATO e PERMANENTE (dati umani): capacità > dataset così non viene
    # MAI svuotato dalla FIFO. Risolve la perdita dell'ancora BC e la rende presente in ogni batch.
    expert_memory = ReplayBuffer(400000)

    # Inizializzazione Agent
    agent = TD3BCAgent()
    agent.bc_alpha = args.bc_alpha
    agent.trust_region_weight = args.trust_region
    if args.trust_region > 0.0:
        print(f"Trust region attiva: peso {args.trust_region} verso le azioni del buffer sui campioni "
              f"non-expert (ancora l'Actor al supporto dati, stabilizza la policy deterministica).")

    # Nota: i pesi BC pre-addestrati vengono caricati solo al fresh-start (blocco successivo); in caso di resume, sono ripristinati dal checkpoint.
    checkpoint_path = 'train_set/checkpoints/td3_checkpoint.pth'
    start_episode, global_step, best_lap_time, best_eval_dist, best_distance = agent.load_checkpoint(checkpoint_path, memory, elite_memory)

    # Carica i dati dell'esperto nel buffer permanente sia all'avvio che al riavvio, garantendo l'ancora BC.
    # Il filtro sul lap_time tiene solo i giri migliori del pilota: l'ancora deve puntare al suo best, non alla sua media.
    expert_lap_filter = args.expert_max_lap_time if args.expert_max_lap_time > 0 else None
    expert_memory.load_expert_data('train_set/laps', max_samples=350000, max_lap_time=expert_lap_filter)

    # Semina dell'elite dai giri auto-registrati (train_set/laps_auto) per la self-imitation. Due trigger:
    #  - ESPLICITO: --reseed_elite_max_lap_time S (semina coi giri <= S);
    #  - AUTOMATICO (rete di sicurezza): se dopo il caricamento l'elite è AFFAMATO (< floor) — perché
    #    starved o perché un salvataggio precedente era quasi vuoto — si semina da solo con una soglia di
    #    default. Così la self-imitation non resta MAI a secco tra i restart, e le catture dei giri veloci
    #    (eval/online) col tempo invecchiano ed espellono in FIFO i semi più lenti.
    ELITE_STARVATION_FLOOR = 8000
    DEFAULT_RESEED_LAP_TIME = 74.5
    elite_loaded = len(elite_memory)
    print(f"  [ELITE] {elite_loaded} transizioni caricate dal checkpoint.")
    seed_cut = args.reseed_elite_max_lap_time
    if seed_cut <= 0.0 and elite_loaded < ELITE_STARVATION_FLOOR:
        seed_cut = DEFAULT_RESEED_LAP_TIME
        print(f"  [ELITE] affamato (<{ELITE_STARVATION_FLOOR}): auto-semina di sicurezza dai giri auto <= {seed_cut}s.")
    if seed_cut > 0.0:
        n_before = len(elite_memory)
        elite_memory.load_expert_data(LAPS_AUTO_DIR, max_samples=150000, max_lap_time=seed_cut)
        print(f"  [ELITE SEED] {n_before} -> {len(elite_memory)} transizioni (giri auto <= {seed_cut:.1f}s).")

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
    time_attack = (os.environ.get('IM_TIME_ATTACK', '0') == '1')
    phase_log_file = os.path.join(
        SESSION_LOGS_DIR,
        'time-attack.log' if time_attack else 'td3_training.log',
    )

    # Popoliamo la finestra leggendo i dati recenti direttamente dal log
    initial_evals = load_recent_evals_from_log(phase_log_file, REFINE_WINDOW)
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
    os.makedirs(SESSION_LOGS_DIR, exist_ok=True)
    log_file = os.devnull if os.environ.get('IM_WRAPPER_LOG_ONLY', '0') == '1' else phase_log_file

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
        initial_ref = f"{refine_plateau_level:.0f}m" if refine_plateau_level > 0.0 else "da impostare"
        _control_log(f"AVVIO con --refine: REFINEMENT armata da subito "
                     f"(aggiornamento Critic disattivato, loss Critic solo diagnostica, "
                     f"peso Behavioral Cloning={REFINE_BC_WEIGHT}, riferimento plateau={initial_ref}, "
                     f"episodio iniziale {start_episode})")
    if getattr(args, 'rollback', False):
        _control_log(f"AVVIO con --rollback: Actor congelato per {actor_freeze_episodes} episodi "
                     f"(0 = nessun congelamento), auto-refinement automatica="
                     f"{'attiva' if auto_refine_enabled else 'disattivata'}.")
    elif not auto_refine_enabled:
        _control_log("AVVIO con --no-auto-refine: refinement automatica disattivata; "
                     "--refine manuale resta disponibile.")

    elite_threshold = 500.0

    # ── Fase TIME-ATTACK (opt-in via IM_TIME_ATTACK=1) ────────────────────────
    # Da attivare DOPO aver raccolto abbastanza giri completi e riaddestrato la BC: riduce
    # l'ancoraggio alla BC (alpha più alto) e abbassa il floor del rumore esplorativo per
    # limare i tempi. Il bonus di record personale è invece sempre attivo (vedi blocco SUCCESS).
    noise_floor = TIME_ATTACK_NOISE_FLOOR if time_attack else EXPL_NOISE_END
    if time_attack:
        agent.bc_alpha = TIME_ATTACK_BC_ALPHA
        print(f"[TIME-ATTACK] Fase attiva: bc_alpha={agent.bc_alpha}, noise_floor={noise_floor}. "
              f"L'agente ottimizza il tempo sul giro battendo il proprio record.")

    # Override manuale del rumore esplorativo (IM_EXPL_NOISE): fissa expl_noise a un valore costante,
    # scavalcando sia l'annealing sia il floor. Serve per la fase di STABILIZZAZIONE: forzare un rumore
    # basso su una policy gia' a convergenza (es. resume a episodi bassi) restando in td3 standard, senza
    # passare per il time-attack che alzerebbe anche bc_alpha indebolendo l'ancora BC. <=0 o assente => off.
    expl_noise_override = None
    _noise_override_env = os.environ.get('IM_EXPL_NOISE')
    if _noise_override_env is not None:
        try:
            _v = float(_noise_override_env)
            if _v > 0:
                expl_noise_override = _v
                print(f"[STABILIZZAZIONE] IM_EXPL_NOISE attivo: expl_noise fisso a {_v} "
                      f"(annealing e floor scavalcati).")
            else:
                print(f"IM_EXPL_NOISE={_noise_override_env} <= 0: override ignorato.")
        except ValueError:
            print(f"IM_EXPL_NOISE='{_noise_override_env}' non numerico: override ignorato.")

    # Lap recorder: raccoglie i giri completi e puliti guidati dall'agente in esplorazione
    # e li salva in train_set/laps_auto/ per arricchire il dataset della BC (flywheel dati).
    # Soglia tempo configurabile via IM_RECORD_MAX_LAP_TIME (default 80s); disattivabile con IM_RECORD_LAPS=0.
    lap_recorder = LapRecorder(
        LAPS_AUTO_DIR,
        max_lap_time=float(os.environ.get('IM_RECORD_MAX_LAP_TIME', '80.0')),
        on_track_limit=1.0,
        enabled=(os.environ.get('IM_RECORD_LAPS', '1') != '0'),
    )

    # ── Penalità di corridoio (STABILIZZAZIONE) ───────────────────────────────
    # Spinge la policy a tenersi lontano dal bordo PRIMA di uscire → giri completi affidabili.
    # Piena in stabilizzazione; ridotta al 25% in time-attack (lì serve usare tutta la pista per
    # limare i tempi). Sovrascrivibile via IM_MARGIN_PENALTY (0 = disattiva del tutto).
    margin_coef_default = MARGIN_PENALTY_COEF * (0.25 if time_attack else 1.0)
    try:
        margin_coef = float(os.environ.get('IM_MARGIN_PENALTY', margin_coef_default))
    except ValueError:
        margin_coef = margin_coef_default
    if margin_coef > 0.0:
        print(f"[STABILIZZAZIONE] Penalità di corridoio attiva: coef={margin_coef:.1f} "
              f"(margine dal bordo per completare i giri).")

    # ── Telemetria a settori (TIME-ATTACK) ────────────────────────────────────
    # Solo in time-attack: premia il battere i propri split di settore e logga dove si perde tempo.
    sector_timer = None
    if time_attack:
        sector_timer = SectorTimer(
            TRACK_LENGTH_M,
            n_sectors=int(os.environ.get('IM_TA_SECTORS', str(TA_SECTORS_DEFAULT))),
            sidecar_path='train_set/checkpoints/td3_sector_best.json',
            reward_k=float(os.environ.get('IM_TA_SECTOR_K', str(TA_SECTOR_REWARD_K))),
            reward_cap=float(os.environ.get('IM_TA_SECTOR_CAP', str(TA_SECTOR_REWARD_CAP))),
            log=_control_log,
        )
        print(f"[TIME-ATTACK] Reward a settori attiva: {sector_timer.n} settori, "
              f"k={sector_timer.reward_k}, cap={sector_timer.reward_cap}.")

    print("Avvio training TD3+BC...")

    for episode in range(start_episode, args.episodes):
        # Annealing lineare del rumore esplorativo: da EXPL_NOISE_START al floor (EXPL_NOISE_END)
        # in EXPL_NOISE_ANNEAL_EPISODES episodi. A regime servono micro-variazioni di traiettoria,
        # non sbandate a velocità di gara.
        # In time-attack la policy ha già raggiunto la convergenza (chiude il giro): l'annealing — agganciato al
        # numero ASSOLUTO di episodio — imporrebbe ancora ~0.065 a episodi bassi dopo un resume, cioè
        # rumore da warmup su una policy matura, che la butta fuori alla prima curva veloce. Si va
        # quindi diritti al floor (micro-variazioni attorno alla linea ottima), che è proprio lo scopo
        # della fase di rifinitura dei tempi.
        if expl_noise_override is not None:
            agent.expl_noise = expl_noise_override
        elif time_attack:
            agent.expl_noise = noise_floor
        else:
            agent.expl_noise = max(
                noise_floor,
                EXPL_NOISE_START - (EXPL_NOISE_START - noise_floor) * episode / EXPL_NOISE_ANNEAL_EPISODES
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

        # Lap recorder: nuovo giro, e tracciamento dell'osservazione grezza pre-step.
        lap_recorder.start_episode()
        if sector_timer is not None:
            sector_timer.start_lap(float(np.array(ob.get('distFromStart', 0.0)).flat[0]))
        cur_ob = ob
        # Etichetta di fase (curriculum): time-attack se attiva, altrimenti warmup (Critic
        # non ancora caldo) oppure online. Usata per i metadati del recorder e i log.
        current_phase = "time_attack" if time_attack else ("warmup" if global_step < 15000 else "online")

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

            # Lap recorder: stato grezzo pre-step + azione realmente eseguita su TORCS.
            lap_recorder.record_step(cur_ob, torcs_action)

            next_ob, reward, env_done, info = env.step(torcs_action)
            cur_ob = next_ob
            cur_speed_kmh = float(np.array(next_ob.get('speedX', 0.0)).flat[0]) * 50.0
            cur_rpm = float(np.array(next_ob.get('rpm', 0.0)).flat[0])
            next_f_state = flatten_state(next_ob)
            state_stack.append(next_f_state)

            current_track_pos_m = float(np.array(next_ob.get('distFromStart', 0.0)).flat[0])
            current_dist = _track_progress_from_start(episode_start_dist, current_track_pos_m)
            last_lap_time = float(np.array(next_ob.get('lastLapTime', 0.0)).flat[0])
            torcs_lap_time = float(np.array(next_ob.get('curLapTime', 0.0)).flat[0])
            max_dist = max(max_dist, current_dist)

            # Shaping per-step sulle transizioni ONLINE (la guida effettiva dell'agente):
            #  - corridoio: margine dal bordo (stabilizzazione → giri completi affidabili);
            #  - settori: premio per aver battuto i propri split (solo time-attack).
            if margin_coef > 0.0:
                reward += margin_penalty(
                    float(np.array(next_ob.get('trackPos', 0.0)).flat[0]), coef=margin_coef)
            if sector_timer is not None:
                reward += sector_timer.update(current_track_pos_m, torcs_lap_time)

            lap_completed = bool(info.get('lap_completed', False))
            if not lap_completed:
                lap_completed = last_lap_time > 0.0 and abs(last_lap_time - prev_last_lap) > 0.01 and step > 500

            done = False
            if lap_completed and not info.get('crash', False):
                done, termination_reason = True, "SUCCESS"
                completed_lap_time = last_lap_time
                max_dist = max(max_dist, TRACK_LENGTH_M)
                # STABILIZZAZIONE: il completamento è premiato in modo PIATTO (solo LAP_SUCCESS_BONUS),
                # così un giro lento ma pulito vale quanto un giro veloce ma rischioso → la policy
                # impara prima a chiudere il giro. La pressione sul tempo (bonus proporzionale e
                # record personale) è riservata al TIME-ATTACK, dove serve limare i decimi.
                reward += LAP_SUCCESS_BONUS
                if time_attack:
                    reward += LAP_TIME_BONUS_PER_S * max(0.0, LAP_TIME_BONUS_REF_S - last_lap_time)
                if last_lap_time < best_lap_time:
                    # Bonus di RECORD PERSONALE: solo in time-attack (calcolato sul best PRECEDENTE);
                    # in stabilizzazione si registra comunque il best lap, ma senza premio sul tempo.
                    if time_attack:
                        reward += personal_best_bonus(best_lap_time, last_lap_time)
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

                # Lap recorder: salva il giro solo se completato pulito (gate qualità interno).
                if termination_reason == "SUCCESS":
                    lap_recorder.finish_lap(completed_lap_time, phase=current_phase,
                                            episode=episode, global_step=global_step)
                else:
                    lap_recorder.discard()

                # Time-attack: chiudi il giro a settori (logga dove perde tempo + giro ideale) se
                # completato, altrimenti scarta il parziale conservando i best-settore già migliorati.
                if sector_timer is not None:
                    if termination_reason == "SUCCESS":
                        sector_timer.finish_lap(completed_lap_time)
                    else:
                        sector_timer.discard_lap()
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
                # Stato per lo stop GEOMETRICO di fine giro: posizione sul tracciato e cronometro del giro
                # allo step precedente, per rilevare il riattraversamento del traguardo a prescindere dal
                # sensore lastLapTime (che lagga ~1 tick).
                eval_prev_track_pos_m = eval_start_dist
                eval_prev_cur_lap = float(np.array(eval_ob.get('curLapTime', 0.0)).flat[0])
                eval_transitions = []  # transizioni del giro, per la cattura nell'elite se completato

                agent.actor.eval()
                while eval_step < args.max_steps:
                    eval_step += 1
                    prev_eval_stacked = eval_stacked  # stato corrente PRIMA dello step (per la transizione)
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
                    eval_cur_lap = float(np.array(eval_ob.get('curLapTime', 0.0)).flat[0])
                    # Registra la transizione (per l'eventuale cattura nell'elite a giro completato)
                    eval_transitions.append((prev_eval_stacked, eval_action.copy(), eval_r, eval_stacked, 1.0))

                    # Stop GEOMETRICO di fine giro: indipendente dal sensore lastLapTime (che si aggiorna con
                    # ~1 tick di ritardo, e il clamp della distanza maschererebbe uno sforamento nel 2° giro).
                    # Se l'auto ha coperto >=90% del tracciato e poi il distFromStart "salta indietro" oltre
                    # mezza pista (riattraversamento del traguardo), il giro è completo: ci si ferma SUBITO,
                    # niente 2° giro. Tempo del giro dal sensore se aggiornato, altrimenti dal cronometro del
                    # giro all'ultimo step prima del wrap.
                    crossed_finish = (current_eval_track_pos_m + TRACK_LENGTH_M * 0.5 < eval_prev_track_pos_m)
                    if eval_dist >= TRACK_LENGTH_M * 0.9 and crossed_finish and not eval_info.get('crash', False):
                        eval_lap_completed = True
                        eval_dist = TRACK_LENGTH_M
                        eval_sensor_lap = float(np.array(eval_ob.get('lastLapTime', 0.0)).flat[0])
                        if eval_sensor_lap > 0.0 and abs(eval_sensor_lap - eval_prev_last_lap) > 0.01:
                            eval_lap_time = eval_sensor_lap
                        elif eval_prev_cur_lap > 0.0:
                            eval_lap_time = eval_prev_cur_lap
                        break
                    eval_prev_track_pos_m = current_eval_track_pos_m
                    eval_prev_cur_lap = eval_cur_lap

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

                # Cattura del giro completo dell'eval nell'elite (self-imitation): se il giro è stato chiuso
                # pulito, le sue transizioni deterministiche (la "linea pulita") entrano nell'elite marcate
                # expert=1.0, come le iniezioni online. È la fonte di giri VELOCI per la self-imitation, che i
                # giri auto-registrati (più lenti) non danno; man mano che la policy migliora, queste catture
                # rimpiazzano in FIFO i semi lenti.
                # Gated da --capture_eval_elite (default OFF): in stabilizzazione self-imitare la linea-rasoio
                # dell'eval sovra-affila la policy e destabilizza l'esplorazione; riattivala nella limatura.
                if args.capture_eval_elite and eval_lap_completed and len(eval_transitions) > 0:
                    for (s, a, r, ns, m) in eval_transitions:
                        elite_memory.push(s, a, r, ns, m, expert=1.0)
                    print(f"  [ELITE CAPTURE] Giro eval completo nell'elite: "
                          f"+{len(eval_transitions)} transizioni (totale {len(elite_memory)}).")

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
