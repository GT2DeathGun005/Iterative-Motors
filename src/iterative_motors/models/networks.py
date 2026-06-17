"""Reti neurali condivise di Iterative Motors: backbone, Actor (RL), PolicyNetwork (BC), Critic.

Questo modulo centralizza TUTTE le architetture neurali del progetto, che prima erano
duplicate in tre file diversi (training BC, training TD3+BC, valutazione). Avere un'unica
definizione elimina il rischio di disallineamenti tra le tre fasi e garantisce che i pesi
addestrati in una fase siano caricabili nelle altre.

Tutte le reti che producono comandi di guida condividono lo stesso "corpo" (backbone) e la
stessa testa continua: ciò che cambia è solo la funzione di attivazione delle uscite.

------------------------------------------------------------------------------------------
CONTRATTO DELLE CHIAVI state_dict — NON MODIFICARE SENZA MIGRARE I CHECKPOINT
------------------------------------------------------------------------------------------
I checkpoint storici salvati in ``train_set/checkpoints/`` (l'"archivio intoccabile" dei
record) devono continuare a caricarsi senza conversioni. Per questo il backbone e la testa
sono costruiti ESATTAMENTE come nel codice originale, producendo le seguenti chiavi:

    Actor / PolicyNetwork (stessa struttura):
        backbone.0.{weight,bias}    backbone.1.{weight,bias}      # Linear(87,512) + LayerNorm
        backbone.3.{weight,bias}    backbone.4.{weight,bias}      # Linear(512,512) + LayerNorm
        backbone.6.{weight,bias}    backbone.7.{weight,bias}      # Linear(512,512) + LayerNorm
        backbone.9.{weight,bias}    backbone.10.{weight,bias}     # Linear(512,512) + LayerNorm
        continuous_head.{weight,bias}                              # Linear(512,3)
    Critic:
        q1.0/2/4.{weight,bias}      q2.0/2/4.{weight,bias}

Gli indici "mancanti" del backbone (2, 5, 8, 11) sono i layer ReLU, privi di parametri e
quindi assenti dallo state_dict. Poiché Actor e PolicyNetwork hanno le stesse chiavi, i pesi
della BC e quelli del TD3+BC sono interscambiabili: differiscono solo le attivazioni di uscita
applicate a runtime (vedi sotto).
"""

import os

import torch
import torch.nn as nn

from ..common.constants import STACK_DIM


def make_backbone(state_dim: int = STACK_DIM, hidden_size: int = 512) -> nn.Sequential:
    """Costruisce il backbone condiviso: 4 blocchi {Linear -> LayerNorm -> ReLU}.

    La profondità (4 strati) e la larghezza (512 neuroni) sono state scelte sulla base della
    letteratura sull'offline RL (Fujimoto & Gu 2021; Beeson & Montana 2022): abbastanza capacità
    da catturare relazioni non lineari tra i 19 sensori, la velocità e l'assetto della vettura,
    senza eccedere al punto da memorizzare il dataset (overfitting).

    La Layer Normalization dopo ogni Linear stabilizza l'addestramento mantenendo media 0 e
    varianza 1 per feature, evitando che alcune dimensioni dominino il gradiente; la ReLU
    introduce non linearità attivandosi solo per input positivi.

    Args:
        state_dim: dimensione dell'input (default 87 = 3 frame impilati da 29 sensori).
        hidden_size: neuroni per strato nascosto (default 512).
    Returns:
        Un ``nn.Sequential`` con chiavi ``backbone.{0,1,3,4,6,7,9,10}`` (vedi contratto in alto).
    """
    return nn.Sequential(
        nn.Linear(state_dim, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU(),
        nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU(),
        nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU(),
        nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU(),
    )


def make_q_net(state_dim: int = STACK_DIM, action_dim: int = 3, hidden_size: int = 512) -> nn.Sequential:
    """Costruisce una singola rete Q del Critic: {Linear -> ReLU -> Linear -> ReLU -> Linear(.,1)}.

    Prende in input la concatenazione di stato (87D) e azione (3D) e produce un singolo scalare,
    la stima del valore atteso Q(s, a). Il Critic ne usa due indipendenti (Twin Q) per ridurre
    la sovrastima del valore (vedi ``Critic``).
    """
    return nn.Sequential(
        nn.Linear(state_dim + action_dim, hidden_size), nn.ReLU(),
        nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1),
    )


class Actor(nn.Module):
    """Policy deterministica dell'agente RL (TD3+BC): da stato 87D a comandi continui 3D.

    L'input è lo stato concatenato a 87 dimensioni (3 stack temporali di 29 sensori a t-12,
    t-6, t). Il backbone estrae le caratteristiche di guida; la testa continua produce 3 uscite
    portate in [-1, 1] da una Tanh:

        - uscita 0: sterzo  (negativo = sinistra, positivo = destra)
        - uscita 1: acceleratore (rimappato a [0, 1] dal mapping delle azioni a valle)
        - uscita 2: freno       (rimappato a [0, 1] dal mapping delle azioni a valle)

    A differenza della ``PolicyNetwork`` della BC, qui anche acceleratore e freno usano la Tanh:
    è la rappresentazione su cui il Critic ha imparato a stimare Q, e il mapping a pedali [0, 1]
    avviene solo al momento di inviare il comando a TORCS (vedi ``action_mapping.rl_to_pedals``).
    """

    def __init__(self, state_dim: int = STACK_DIM, hidden_size: int = 512):
        super(Actor, self).__init__()
        self.backbone = make_backbone(state_dim, hidden_size)
        self.continuous_head = nn.Linear(hidden_size, 3)  # steer, accel, brake

    def forward(self, state):
        """Azione deterministica grezza (senza rumore): Tanh applicata a tutti e 3 i canali.

        È la π(s) usata sia in valutazione sia, internamente, nel calcolo della loss dell'Actor
        del TD3+BC (massimizzazione di Q(s, π(s))).
        """
        features = self.backbone(state)
        mean = self.continuous_head(features)
        return torch.tanh(mean)

    def sample(self, state, evaluate=False, noise_std=0.1):
        """Azione da eseguire sull'ambiente, deterministica o con rumore esplorativo.

        - ``evaluate=True``: ritorna l'azione deterministica pura prodotta dalla rete, per una
          guida stabile e riproducibile (valutazione/submission).
        - ``evaluate=False``: aggiunge rumore gaussiano (deviazione standard ``noise_std``,
          clippato a [-2σ, +2σ]) per l'esplorazione off-policy. La σ viene annealata dal training
          loop (da EXPL_NOISE_START a EXPL_NOISE_END, o al floor del time-attack) perché a regime
          servono micro-variazioni di traiettoria, non sbandate. L'azione finale è saturata in [-1, 1].
        """
        action = self.forward(state)
        if not evaluate:
            noise = torch.randn_like(action) * noise_std
            noise = torch.clamp(noise, -2.0 * noise_std, 2.0 * noise_std)
            action = torch.clamp(action + noise, -1.0, 1.0)
        return action

    def load_bc_weights(self, bc_path):
        """Inizializza l'Actor (warm-start) dai pesi pre-addestrati in Behavioral Cloning.

        È il ponte tra le due fasi: l'Actor del TD3+BC eredita backbone e testa continua della BC
        così da partire "sapendo già guidare". Poiché la BC usa la Sigmoid [0, 1] per gas/freno
        mentre l'Actor usa la Tanh [-1, 1], i pesi e i bias dei canali 1:3 della testa vengono
        moltiplicati per 0.5: è la trasformazione lineare y = 0.5·x che preserva l'ordine di
        grandezza delle uscite iniziali nel nuovo intervallo. ``strict=False`` ignora eventuali
        chiavi extra (es. una vecchia ``gear_head`` non più usata).
        """
        if not os.path.exists(bc_path):
            return
        bc_state = torch.load(bc_path, map_location='cpu', weights_only=True)
        if 'continuous_head.weight' in bc_state:
            bc_state['continuous_head.weight'][1:3] = bc_state['continuous_head.weight'][1:3] * 0.5
        if 'continuous_head.bias' in bc_state:
            bc_state['continuous_head.bias'][1:3] = bc_state['continuous_head.bias'][1:3] * 0.5
        self.load_state_dict(bc_state, strict=False)
        print(f"Pesi BC caricati con successo da {bc_path} (compensato scaling 0.5 per accel/brake).")

    def load_actor_weights(self, path, device):
        """Carica i pesi dell'Actor da un file (checkpoint completo o solo-Actor).

        Filtra le chiavi tenendo solo i parametri presenti nel modello corrente con forma
        compatibile: così è possibile caricare anche file che contengono altri componenti
        (es. il Critic) o che provengono da varianti dell'architettura, ignorando il resto.
        """
        if not os.path.exists(path):
            return
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


class PolicyNetwork(nn.Module):
    """Rete della Behavioral Cloning e dell'inferenza in valutazione.

    Stessa identica architettura dell'``Actor`` (quindi stesse chiavi state_dict, pesi
    interscambiabili), ma con due modalità di uscita pensate per coprire entrambi i tipi di
    checkpoint senza dover istanziare classi diverse:

      - ``forward`` (attivazione BC): lo sterzo (canale 0) passa per una Tanh in [-1, 1], mentre
        acceleratore e freno (canali 1, 2) passano per una Sigmoid in [0, 1]. È la forma con cui
        la BC viene addestrata, perché coincide esattamente col codominio delle azioni umane
        registrate (sterzo in [-1, 1], pedali in [0, 1]).
      - ``sample`` (attivazione RL): Tanh su tutti e 3 i canali ([-1, 1]). Serve a valutare un
        checkpoint TD3+BC con la stessa classe usata per la BC.

    In ``test_agent`` la scelta tra ``forward`` e ``sample`` dipende dal tipo di pesi caricati
    (BC oppure RL), rilevato dal nome del file o dall'argomento ``--kind``.
    """

    def __init__(self, state_dim: int = STACK_DIM, hidden_size: int = 512):
        super(PolicyNetwork, self).__init__()
        self.backbone = make_backbone(state_dim, hidden_size)
        self.continuous_head = nn.Linear(hidden_size, 3)

    def forward(self, state):
        """Attivazione BC: Tanh sullo sterzo ([-1, 1]), Sigmoid su acceleratore/freno ([0, 1])."""
        features = self.backbone(state)
        cont_out = self.continuous_head(features)
        steer = torch.tanh(cont_out[:, 0:1])
        accel_brake = torch.sigmoid(cont_out[:, 1:3])
        return torch.cat([steer, accel_brake], dim=1)

    def sample(self, state, evaluate: bool = False):
        """Attivazione RL deterministica: Tanh su tutti e 3 i canali (uscite in [-1, 1])."""
        features = self.backbone(state)
        mean = self.continuous_head(features)
        return torch.tanh(mean)


class Critic(nn.Module):
    """Twin Critic Network per la stima del valore Q(s, a) nel TD3+BC.

    Implementa due reti Q indipendenti (Q1 e Q2), addestrate da zero, che ricevono in input la
    concatenazione dello stato 87D e dell'azione 3D. L'uso di due stime distinte combatte
    l'Overestimation Bias tipico dei metodi actor-critic: nel calcolo del target di Bellman si
    usa il MINIMO tra Q1 e Q2, ottenendo una stima del valore più conservativa e stabile.
    """

    def __init__(self, state_dim: int = STACK_DIM, action_dim: int = 3, hidden_size: int = 512):
        super(Critic, self).__init__()
        self.q1 = make_q_net(state_dim, action_dim, hidden_size)
        self.q2 = make_q_net(state_dim, action_dim, hidden_size)

    def forward(self, state, action):
        """Ritorna la coppia di stime (Q1(s, a), Q2(s, a))."""
        xu = torch.cat([state, action], 1)
        return self.q1(xu), self.q2(xu)
