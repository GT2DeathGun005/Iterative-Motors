"""
Soft Actor-Critic (SAC) con RLPD — Fine-Tuning RL per Giro Secco TORCS

Addestra un agente SAC pre-inizializzato con i pesi del Behavioral Cloning
per battere i tempi umani su singolo giro con partenza da fermo.

Features:
  - RLPD: replay buffer pre-riempito con 71k transizioni umane (reward calcolata)
  - Warm Start: carica backbone + mean_linear dal BC checkpoint
  - BC Regularization: previene catastrophic forgetting dei pesi BC
  - AdaptiveScheduler: λ_bc, σ, cpi_weight, α adattati ogni 50 ep in base alle metriche
  - Actor LR separato (1e-5): aggiornamenti lenti per preservare BC
  - Reward dinamica: basata sui tempi umani reali
  - Terminazione: uscita pista, spin, stallo prolungato
  - Checkpoint periodici in train_set/checkpoints/ (con stato scheduler)
  - GPU-optimized (CUDA)

Action de-normalization (Tanh [-1,1] → env ranges):
  steer = action[0]                              # [-1, 1]
  accel = (action[1] + 1) / 2                    # [0, 1]
  brake = (action[2] + 1) / 2                    # [0, 1]
  gear  = round((action[3] + 1) * 3)             # [0, 6] discretizzato
"""

import os
import sys
import glob
import argparse
import random
import time
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from collections import deque
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'gym_torcs')))

try:
    from gym_torcs import TorcsEnv
except ImportError:
    print("Warning: gym_torcs non trovato.")

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPSILON = 1e-6


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


# ──────────────────────────────────────────────────────────────────────
#  Replay Buffer
# ──────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.base_buffer = deque(maxlen=capacity)
        self.curve_buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, is_curve=False):
        if is_curve:
            self.curve_buffer.append((state, action, reward, next_state, done))
        else:
            self.base_buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        # Campionamento stratificato: 75% base, 25% curve
        if len(self.curve_buffer) == 0:
            batch = random.sample(self.base_buffer, batch_size)
        elif len(self.base_buffer) == 0:
            batch = random.sample(self.curve_buffer, batch_size)
        else:
            curve_batch_size = int(round(batch_size * 0.25))
            curve_batch_size = max(0, min(curve_batch_size, len(self.curve_buffer)))
            base_batch_size = batch_size - curve_batch_size
            
            base_batch_size = max(0, min(base_batch_size, len(self.base_buffer)))
            if base_batch_size + curve_batch_size < batch_size:
                needed = batch_size - (base_batch_size + curve_batch_size)
                if len(self.base_buffer) > base_batch_size:
                    base_batch_size += min(needed, len(self.base_buffer) - base_batch_size)
                elif len(self.curve_buffer) > curve_batch_size:
                    curve_batch_size += min(needed, len(self.curve_buffer) - curve_batch_size)

            batch_base = random.sample(self.base_buffer, base_batch_size)
            batch_curve = random.sample(self.curve_buffer, curve_batch_size)
            batch = batch_base + batch_curve
            random.shuffle(batch)
            
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.base_buffer) + len(self.curve_buffer)


# ──────────────────────────────────────────────────────────────────────
#  Actor (Policy Network)
# ──────────────────────────────────────────────────────────────────────

class Actor(nn.Module):
    """Rete Actor SAC con output Gaussiano (mean, log_std).

    Backbone identico alla PolicyNetwork del BC (layers 0-5 del Sequential):
      0: Linear → 1: LayerNorm → 2: ReLU → 3: Linear → 4: LayerNorm → 5: ReLU
    Poi si biforca in mean_linear e log_std_linear.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 256):
        super(Actor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),    # 0
            nn.LayerNorm(hidden_size),             # 1
            nn.ReLU(),                             # 2
            nn.Linear(hidden_size, hidden_size),   # 3
            nn.LayerNorm(hidden_size),             # 4
            nn.ReLU()                              # 5
        )

        self.mean_linear = nn.Linear(hidden_size, action_dim)
        self.log_std_linear = nn.Linear(hidden_size, action_dim)
        self.apply(weights_init_)

    def forward(self, state):
        x = self.net(state)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t

        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + EPSILON)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, torch.tanh(mean)

    def load_bc_weights(self, bc_model_path: str, device: str = "cpu"):
        """Carica i pesi pre-addestrati dal Behavioral Cloning (Warm Start).

        Mappa il backbone (net.0-5) e l'output layer (net.6 → mean_linear).
        Il log_std_linear viene inizializzato a bassa varianza.
        """
        if not os.path.exists(bc_model_path):
            print(f"  ⚠️  File BC '{bc_model_path}' non trovato. Actor parte da zero.")
            return False

        print(f"  Caricamento pesi BC da: {bc_model_path}")
        bc_state = torch.load(bc_model_path, map_location=device, weights_only=True)

        # Mappatura BC → SAC Actor
        mapping = {
            'net.0.weight': ('net.0.weight', 'Linear layer 1'),
            'net.0.bias':   ('net.0.bias',   None),
            'net.1.weight': ('net.1.weight', 'LayerNorm 1'),
            'net.1.bias':   ('net.1.bias',   None),
            'net.3.weight': ('net.3.weight', 'Linear layer 2'),
            'net.3.bias':   ('net.3.bias',   None),
            'net.4.weight': ('net.4.weight', 'LayerNorm 2'),
            'net.4.bias':   ('net.4.bias',   None),
            'net.6.weight': ('mean_linear.weight', 'Output → Mean'),
            'net.6.bias':   ('mean_linear.bias',   None),
        }

        loaded = 0
        with torch.no_grad():
            for bc_key, (sac_key, label) in mapping.items():
                if bc_key not in bc_state:
                    print(f"    ⚠️  Chiave BC '{bc_key}' non trovata, skip.")
                    continue

                # Naviga nella struttura dell'Actor per trovare il parametro
                parts = sac_key.split('.')
                param = self
                for p in parts[:-1]:
                    param = getattr(param, p) if not p.isdigit() else param[int(p)]
                target = getattr(param, parts[-1])

                if target.shape != bc_state[bc_key].shape:
                    print(f"    ⚠️  Shape mismatch {bc_key}: BC={bc_state[bc_key].shape} vs SAC={target.shape}")
                    continue

                target.copy_(bc_state[bc_key])
                loaded += 1
                if label:
                    print(f"    ✓ {label}: {bc_key} → {sac_key}")

            # Log_std inizializzato molto basso: la policy BC deve essere
            # quasi-deterministica per guidare correttamente fin dal primo episodio.
            # L'esplorazione verrà aumentata gradualmente durante il training.
            nn.init.constant_(self.log_std_linear.weight, 0.0)
            nn.init.constant_(self.log_std_linear.bias, -5.0)

        print(f"  Warm Start completato: {loaded}/10 parametri caricati.")
        print(f"  Log_std inizializzato a -5.0 (quasi-deterministico, preserva BC).")
        return True


# ──────────────────────────────────────────────────────────────────────
#  Critic (Twin Q-Network)
# ──────────────────────────────────────────────────────────────────────

class Critic(nn.Module):
    """Twin Q-Network per Double Q-learning (mitiga sovrastima)."""

    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 256):
        super(Critic, self).__init__()

        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        return self.q1(xu), self.q2(xu)


# ──────────────────────────────────────────────────────────────────────
#  SAC Agent
# ──────────────────────────────────────────────────────────────────────

class SACAgent:
    def __init__(self, state_dim: int, action_dim: int,
                 device: str = "cpu", gamma: float = 0.99,
                 tau: float = 0.005, critic_lr: float = 3e-4,
                 actor_lr: float = 1e-5, bc_lambda: float = 1.0):
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.bc_lambda = bc_lambda  # Coefficiente regolarizzazione BC

        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # LR separato: actor lento (preserva BC), critic veloce
        # IMPORTANTE: log_std_linear è ESCLUSO dall'optimizer.
        # Il SAC entropy loss (alpha * log_pi) spinge log_std verso l'alto,
        # aumentando l'esplorazione e distruggendo la policy BC.
        # Congelando log_std a -5.0 (std≈0.007), l'actor impara solo la media
        # (backbone + mean_linear) con rumore fisso quasi-deterministico.
        actor_params = [p for n, p in self.actor.named_parameters()
                        if 'log_std_linear' not in n]
        self.actor_optimizer = optim.Adam(actor_params, lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Alpha fisso (NO auto-tuning con BC warm start)
        # L'auto-tuning standard forza alpha in alto perché la policy BC
        # è quasi deterministica, il che distrugge i pesi BC.
        self.alpha = 0.02  # Basso: esplorazione conservativa per preservare BC

        # BC model congelato come riferimento (caricato dopo)
        self.bc_model = None

    def load_bc_reference(self, bc_path: str):
        """Carica una copia congelata del BC model per la regularization."""
        from behavioral_cloning import PolicyNetwork
        self.bc_model = PolicyNetwork().to(self.device)
        self.bc_model.load_state_dict(
            torch.load(bc_path, map_location=self.device, weights_only=True)
        )
        self.bc_model.eval()
        for p in self.bc_model.parameters():
            p.requires_grad = False
        print(f"  ✅ BC reference model caricato e congelato per regularization.")

    def select_action(self, state, evaluate=False):
        state_t = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate:
            _, _, action = self.actor.sample(state_t)
        else:
            action, _, _ = self.actor.sample(state_t)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory: ReplayBuffer, batch_size: int,
                          train_episode: int = 0, cpi_weight: float = None):
        """Update Actor e Critic con Advantage-Weighted CPI.

        Args:
            train_episode: numero di episodi dall'inizio della fase TRAIN
                           (esclude la fase FREEZE). Usato per lo schedule CPI.
        """
        state_b, action_b, reward_b, next_state_b, mask_b = memory.sample(batch_size)

        state_b = torch.FloatTensor(state_b).to(self.device)
        next_state_b = torch.FloatTensor(next_state_b).to(self.device)
        action_b = torch.FloatTensor(action_b).to(self.device)
        reward_b = torch.FloatTensor(reward_b).to(self.device).unsqueeze(1)
        mask_b = torch.FloatTensor(mask_b).to(self.device).unsqueeze(1)

        # ── Critic update ──
        with torch.no_grad():
            next_action, next_log_pi, _ = self.actor.sample(next_state_b)
            q1_next, q2_next = self.critic_target(next_state_b, next_action)
            min_q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_pi
            next_q = reward_b + mask_b * self.gamma * min_q_next

        q1, q2 = self.critic(state_b, action_b)
        qf_loss = F.mse_loss(q1, next_q) + F.mse_loss(q2, next_q)

        self.critic_optimizer.zero_grad()
        qf_loss.backward()
        self.critic_optimizer.step()

        # ── Actor update: Advantage-Weighted CPI ──
        # Un SINGOLO forward pass dell'Actor produce sia l'azione per BC
        # sia l'azione per Q-improvement, evitando gradienti conflittuali.

        pi, _, pi_det = self.actor.sample(state_b)

        # 1. BC loss: distanza dall'azione BC (obiettivo principale)
        bc_loss = torch.tensor(0.0, device=self.device)
        if self.bc_model is not None:
            with torch.no_grad():
                bc_actions = self.bc_model(state_b)
            bc_loss = F.mse_loss(pi_det, bc_actions)

        # 2. Advantage-filtered Q-improvement con maschera binaria hard
        q1_pi, q2_pi = self.critic(state_b, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)

        # Q delle azioni nel buffer (baseline) — DETACHED
        with torch.no_grad():
            q1_buf, q2_buf = self.critic(state_b, action_b)
            q_baseline = torch.min(q1_buf, q2_buf)
            # Maschera binaria: 1.0 dove l'Actor è migliore del buffer, 0.0 altrove.
            # Il detach() impedisce ai gradienti di fluire attraverso la maschera.
            adv_mask = (min_q_pi.detach() > q_baseline).float()

        # Applica Q-improvement SOLO dove advantage > 0 (maschera hard)
        # Dove adv_mask=0, il gradiente è zero → il Critic non può degradare l'Actor
        q_scale = max(q_baseline.abs().mean().item(), 1.0)
        q_improvement = -((min_q_pi * adv_mask) / q_scale).mean()

        # CPI schedule: usa il valore dallo scheduler adattivo, o fallback statico
        if cpi_weight is None:
            if train_episode < 50:
                cpi_weight = 0.01   # Stabilizzazione: CPI ultra-conservativo
            elif train_episode < 150:
                cpi_weight = 0.05   # Crescita: il Critic inizia a influenzare
            else:
                cpi_weight = 0.1    # Pieno: il Critic guida il miglioramento

        # FIX #1: bc_lambda ora moltiplica effettivamente la BC loss
        policy_loss = self.bc_lambda * bc_loss + cpi_weight * q_improvement

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        # Soft update target
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        return qf_loss.item(), policy_loss.item(), adv_mask.mean().item()

    def update_critic_only(self, memory: ReplayBuffer, batch_size: int):
        """Aggiorna SOLO il Critic (e il target) — usato per il pre-training offline.

        I gradienti dell'Actor sono categoricamente disabilitati durante questa fase.

        IMPORTANTE: il termine entropia (-alpha * log_pi) è RIMOSSO dal target
        Bellman. Con log_std=-5.0 l'Actor è quasi-deterministico, il che produce
        log_pi≈90. Moltiplicato per alpha=0.02 dà un penalty di ~1.8/step che
        domina la reward (~0.7) e spinge i Q-values a -80 invece del vero ~+68.
        L'entropia è un incentivo all'esplorazione per il loop online,
        non ha senso durante il pre-training offline su dataset fisso.
        """
        state_b, action_b, reward_b, next_state_b, mask_b = memory.sample(batch_size)

        state_b = torch.FloatTensor(state_b).to(self.device)
        next_state_b = torch.FloatTensor(next_state_b).to(self.device)
        action_b = torch.FloatTensor(action_b).to(self.device)
        reward_b = torch.FloatTensor(reward_b).to(self.device).unsqueeze(1)
        mask_b = torch.FloatTensor(mask_b).to(self.device).unsqueeze(1)

        # ── Congela Actor: nessun gradiente deve fluire verso i suoi parametri ──
        for p in self.actor.parameters():
            p.requires_grad = False

        with torch.no_grad():
            # Usa la media deterministica dell'Actor (no rumore di sampling)
            _, _, next_action = self.actor.sample(next_state_b)
            q1_next, q2_next = self.critic_target(next_state_b, next_action)
            # NO entropia: il termine -alpha*log_pi è omesso intenzionalmente
            min_q_next = torch.min(q1_next, q2_next)
            next_q = reward_b + mask_b * self.gamma * min_q_next

        q1, q2 = self.critic(state_b, action_b)
        qf_loss = F.mse_loss(q1, next_q) + F.mse_loss(q2, next_q)

        self.critic_optimizer.zero_grad()
        qf_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # Soft update target
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        # ── Scongela Actor per il training online successivo ──
        for p in self.actor.parameters():
            p.requires_grad = True

        return qf_loss.item()


# ──────────────────────────────────────────────────────────────────────
#  Adaptive Scheduler
# ──────────────────────────────────────────────────────────────────────

class AdaptiveScheduler:
    """Regola λ_bc, σ, cpi_weight, α in base alle metriche di training.

    Ogni `eval_every` episodi, analizza una finestra mobile di metriche
    e adatta i parametri per massimizzare il progresso dell'agente.

    Mastery-Driven Phase Transition:
        Il mastery score (0→1) misura quanto l'agente ha assimilato la pista,
        calcolato da: lap completion rate (50%), survival consistency (30%),
        e lap time convergence (20%). I limiti dei parametri vengono interpolati
        linearmente tra "modalità sopravvivenza" (mastery=0) e "modalità
        time-attack" (mastery=1). La transizione è bidirezionale: se l'agente
        perde padronanza, i vincoli si rilassano automaticamente.
    """

    # ── Limiti dei due regimi per l'interpolazione ──
    # (survival_mode_value, time_attack_mode_value)
    BOUNDS = {
        'max_bc_lambda':  (1.0,  0.4),   # BC domina → BC regolarizza
        'min_sigma':      (0.05, 0.08),  # esplorazione minima → floor strutturale
        'max_cpi_weight': (0.20, 0.08),  # vincolo alto → vincolo rilassato
    }

    def __init__(self, bc_lambda=1.0, sigma=0.1, cpi_weight=0.01,
                 alpha=0.02, window=50, eval_every=50):
        # Parametri adattivi correnti
        self.bc_lambda = bc_lambda
        self.sigma = sigma
        self.cpi_weight = cpi_weight
        self.alpha = alpha

        # Configurazione
        self.window = window
        self.eval_every = eval_every

        # Storico metriche
        self.rewards = deque(maxlen=window * 2)
        self.steps_list = deque(maxlen=window * 2)
        self.critic_losses = deque(maxlen=1000)
        self.adv_hits = deque(maxlen=500)

        # Storico lap completions e tempi (per mastery)
        self.lap_completions = deque(maxlen=window * 2)
        self.lap_times = deque(maxlen=20)

        # Mastery score: 0.0 = nessuna padronanza, 1.0 = time-attack pieno
        self.mastery = 0.0

        # Stima step per giro completo (~70s a 50Hz)
        self.LAP_STEPS_ESTIMATE = 3500

    @staticmethod
    def _lerp(survival_val, attack_val, t):
        """Interpolazione lineare survival → attack."""
        return survival_val + (attack_val - survival_val) * t

    def record_episode(self, reward, steps, lap_completed=False, lap_time=0.0):
        """Registra reward, steps e stato di completamento di un episodio."""
        self.rewards.append(reward)
        self.steps_list.append(steps)
        self.lap_completions.append(1 if lap_completed else 0)
        if lap_completed and lap_time > 0:
            self.lap_times.append(lap_time)

    def record_update(self, critic_loss, adv_hit_rate):
        """Registra metriche di un singolo update Actor+Critic."""
        self.critic_losses.append(critic_loss)
        self.adv_hits.append(adv_hit_rate)

    def should_adapt(self, episode):
        """True se è il momento di rivalutare i parametri."""
        return (episode % self.eval_every == 0
                and len(self.rewards) >= self.window)

    def _compute_mastery(self):
        """Calcola il mastery score (0→1) da 3 componenti pesate.

        Componenti:
          1. Lap Completion Rate (50%): frequenza di completamento giri recenti
          2. Survival Consistency (30%): stabilità della sopravvivenza
          3. Lap Time Convergence (20%): convergenza dei tempi giro

        Returns:
            float: mastery score clamped in [0.0, 1.0]
        """
        w = self.window

        # 1. Lap Completion Rate (peso: 50%)
        if len(self.lap_completions) >= w:
            recent_laps = list(self.lap_completions)[-w:]
            lap_rate = sum(recent_laps) / len(recent_laps)
        elif len(self.lap_completions) > 0:
            laps = list(self.lap_completions)
            lap_rate = sum(laps) / len(laps)
        else:
            lap_rate = 0.0

        # 2. Survival Consistency (peso: 30%)
        #    Quanto l'agente sopravvive stabilmente? (target: 60%+ del giro)
        if len(self.steps_list) >= w:
            mean_survival = np.mean(list(self.steps_list)[-w:]) / self.LAP_STEPS_ESTIMATE
        else:
            mean_survival = 0.0
        survival_consistency = min(1.0, mean_survival / 0.6)

        # 3. Lap Time Convergence (peso: 20%)
        #    I tempi dei giri completati convergono? (bassa varianza = padronanza)
        if len(self.lap_times) >= 3:
            times = list(self.lap_times)
            cv = np.std(times) / (np.mean(times) + 1e-8)  # Coefficiente di variazione
            time_convergence = max(0.0, 1.0 - cv * 10)    # cv < 0.1 → convergenza alta
        else:
            time_convergence = 0.0

        mastery = 0.5 * lap_rate + 0.3 * survival_consistency + 0.2 * time_convergence
        return max(0.0, min(1.0, mastery))

    def adapt(self):
        """Ricalcola tutti i parametri. Ritorna (changes_dict, metrics_dict)."""
        rewards = list(self.rewards)
        steps = list(self.steps_list)
        w = self.window

        # ── Mastery: interpola i limiti tra survival e time-attack ──
        self.mastery = self._compute_mastery()
        max_bc_lambda = self._lerp(*self.BOUNDS['max_bc_lambda'], self.mastery)
        min_sigma = self._lerp(*self.BOUNDS['min_sigma'], self.mastery)
        max_cpi_weight = self._lerp(*self.BOUNDS['max_cpi_weight'], self.mastery)

        # ── Metriche ──
        recent_r = rewards[-w:]
        older_r = rewards[-2*w:-w] if len(rewards) >= 2*w else recent_r
        reward_trend = np.mean(recent_r) - np.mean(older_r)
        survival = np.mean(steps[-w:]) / self.LAP_STEPS_ESTIMATE

        q_stab = 1.0
        if len(self.critic_losses) > 100:
            cl = list(self.critic_losses)[-500:]
            q_stab = np.std(cl) / (np.mean(cl) + 1e-8)

        adv_rate = 0.0
        if len(self.adv_hits) > 50:
            adv_rate = np.mean(list(self.adv_hits)[-200:])

        changes = {}

        # ── λ_bc: decade se reward stabile e agente sopravvive ──
        old_lbc = self.bc_lambda
        # Soglia abbassata da 0.15 a 0.08 per far decadere bc_lambda anche in stallo iniziale a ~400 step (survival ≈ 0.114)
        if reward_trend >= -10.0 and survival > 0.08:
            self.bc_lambda = max(0.1, self.bc_lambda - 0.05)
        elif reward_trend < -50.0:
            # Cap dinamico via mastery: survival(1.0) → time-attack(0.4)
            self.bc_lambda = min(max_bc_lambda, self.bc_lambda + 0.1)
        if abs(self.bc_lambda - old_lbc) > 1e-6:
            changes['λ_bc'] = (old_lbc, self.bc_lambda)

        # ── σ_exploration: aumenta su stagnazione, riduce su progresso ──
        old_sig = self.sigma
        if abs(reward_trend) < 5.0 and survival < 0.3:
            self.sigma = min(0.3, self.sigma + 0.02)
        elif reward_trend > 20.0:
            # Floor dinamico via mastery: survival(0.05) → time-attack(0.08)
            self.sigma = max(min_sigma, self.sigma - 0.02)
        if abs(self.sigma - old_sig) > 1e-6:
            changes['σ'] = (old_sig, self.sigma)

        # ── cpi_weight: moderato dalla stabilità del Critic ──
        old_cpi = self.cpi_weight
        if q_stab < 1.0 and adv_rate > 0.1:
            # Cap dinamico via mastery: survival(0.20) → time-attack(0.08)
            self.cpi_weight = min(max_cpi_weight, self.cpi_weight + 0.01)
        elif q_stab > 3.0:
            self.cpi_weight = max(0.01, self.cpi_weight - 0.02)
        if abs(self.cpi_weight - old_cpi) > 1e-6:
            changes['cpi'] = (old_cpi, self.cpi_weight)

        # ── α: spinta esplorativa quando Actor troppo deterministico ──
        old_a = self.alpha
        if adv_rate < 0.05 and survival < 0.2:
            self.alpha = min(0.1, self.alpha + 0.005)
        elif adv_rate > 0.3:
            self.alpha = max(0.01, self.alpha - 0.005)
        if abs(self.alpha - old_a) > 1e-6:
            changes['α'] = (old_a, self.alpha)

        metrics = {
            'reward_trend': reward_trend,
            'survival': survival,
            'q_stability': q_stab,
            'adv_hit_rate': adv_rate,
            'mastery': self.mastery,
        }

        return changes, metrics

    def state_dict(self):
        """Serializza lo stato per il checkpoint."""
        return {
            'bc_lambda': self.bc_lambda,
            'sigma': self.sigma,
            'cpi_weight': self.cpi_weight,
            'alpha': self.alpha,
            'rewards': list(self.rewards),
            'steps_list': list(self.steps_list),
            'lap_completions': list(self.lap_completions),
            'lap_times': list(self.lap_times),
            'mastery': self.mastery,
        }

    def load_state_dict(self, d):
        """Ripristina lo stato dal checkpoint."""
        self.bc_lambda = d['bc_lambda']
        self.sigma = d['sigma']
        self.cpi_weight = d['cpi_weight']
        self.alpha = d['alpha']
        self.rewards.extend(d.get('rewards', []))
        self.steps_list.extend(d.get('steps_list', []))
        self.lap_completions.extend(d.get('lap_completions', []))
        self.lap_times.extend(d.get('lap_times', []))
        self.mastery = d.get('mastery', 0.0)


# ──────────────────────────────────────────────────────────────────────
#  State Flattening & Action De-normalization
# ──────────────────────────────────────────────────────────────────────

def flatten_state(state_dict: dict) -> np.ndarray:
    """Appiattisce osservazione TORCS → vettore 29D."""
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
        return np.concatenate([
            [_s('angle')],
            _a('track', 19),
            [_s('trackPos'), _s('speedX'), _s('speedY'), _s('speedZ')],
            _a('wheelSpinVel', 4) / 100.0,
            [_s('rpm') / 10000.0],
        ]).astype(np.float32)
    except Exception:
        return np.zeros(29, dtype=np.float32)


def denormalize_action(action: np.ndarray) -> np.ndarray:
    """Converte azione SAC (Tanh [-1,1]) in formato env TORCS."""
    env_action = np.zeros(4, dtype=np.float32)
    env_action[0] = np.clip(action[0], -1.0, 1.0)               # steer
    env_action[1] = np.clip((action[1] + 1.0) / 2.0, 0.0, 1.0)  # accel
    env_action[2] = np.clip((action[2] + 1.0) / 2.0, 0.0, 1.0)  # brake
    gear = int(round((action[3] + 1.0) * 3.0))                   # gear
    env_action[3] = float(max(0, min(6, gear)))
    return env_action


def normalize_action(env_action: np.ndarray) -> np.ndarray:
    """Converte azione env TORCS → formato SAC Tanh [-1,1] (inversa di denormalize)."""
    return np.array([
        env_action[0],                          # steer: già [-1,1]
        env_action[1] * 2.0 - 1.0,              # accel: [0,1] → [-1,1]
        env_action[2] * 2.0 - 1.0,              # brake: [0,1] → [-1,1]
        env_action[3] / 3.0 - 1.0,              # gear: [0,6] → [-1,1]
    ], dtype=np.float32)


def compute_demo_reward(state: np.ndarray, next_state: np.ndarray) -> float:
    """Calcola la reward per una transizione demo usando il vettore stato.

    Usa le stesse componenti di compute_step_reward() ma ricavate
    dal vettore flattened (non dal dict obs):
      state[0]  = angle
      state[20] = trackPos
      state[21] = speedX (÷50, normalizzato dal wrapper)

    Il progresso è approssimato da speedX * dt (dt ≈ 0.02s a 50Hz).
    """
    speed_x = next_state[21]      # Normalizzato (÷50)
    track_pos = next_state[20]
    angle = next_state[0]

    # Progresso approssimato: distanza ≈ velocità × tempo
    # speedX è in unità normalizzate (÷50), dt=0.02s, speed reale = speedX*50 km/h
    # Converti in m/s: speed_real = speedX * 50 / 3.6
    # delta_dist ≈ speed_real * 0.02 ≈ speedX * 50 / 3.6 * 0.02 ≈ speedX * 0.278
    if speed_x >= 0:
        progress = speed_x * 0.278
        speed_bonus = speed_x * 0.05
        reverse_penalty = 0.0
    else:
        progress = 0.0
        speed_bonus = 0.0
        reverse_penalty = -abs(speed_x) * 0.1

    center_penalty = -1.0 * (track_pos ** 2)
    angle_penalty = -1.5 * abs(angle)
    time_penalty = -0.1

    return progress + speed_bonus + center_penalty + angle_penalty + reverse_penalty + time_penalty


def prefill_buffer_from_demos(memory: ReplayBuffer, demo_dir: str):
    """Carica le transizioni dalle demo umane nel replay buffer.

    Ogni coppia (state_t, action_t) → (state_t+1) diventa una transizione.
    Le azioni vengono normalizzate in formato SAC (tanh [-1,1]).
    La reward viene calcolata dalle componenti del vettore stato usando
    la stessa formula della reward online (speed, center, angle, time penalty).
    """
    h5_files = sorted(glob.glob(os.path.join(demo_dir, "**/lap_*.h5"), recursive=True))
    if not h5_files:
        print(f"  ⚠️  Nessun file demo trovato in {demo_dir} o nelle sue sottocartelle")
        return 0

    total = 0
    rewards_sum = 0.0
    for h5_path in h5_files:
        with h5py.File(h5_path, 'r') as h5f:
            states = h5f['states'][:]
            actions = h5f['actions'][:]

        is_curve_snippet = "lap_curve_" in os.path.basename(h5_path)

        n_transitions = len(states) - 1
        for i in range(n_transitions):
            norm_action = normalize_action(actions[i])
            reward = compute_demo_reward(states[i], states[i + 1])
            
            # FIX #5: ultima transizione di ogni giro demo → mask=0.0 (terminale)
            # ECCEZIONE: per gli snippet delle curve, l'episodio non finisce realmente lì,
            # quindi il Critic deve fare bootstrap (mask=1.0) sul next_state.
            is_last = (i == n_transitions - 1)
            mask = 0.0 if (is_last and not is_curve_snippet) else 1.0
            
            memory.push(states[i], norm_action, reward, states[i + 1], mask, is_curve=is_curve_snippet)
            rewards_sum += reward
            total += 1

    avg_reward = rewards_sum / total if total > 0 else 0.0
    print(f"  ✅ Buffer pre-riempito con {total:,} transizioni da {len(h5_files)} giri demo")
    print(f"     -> Base transitions: {len(memory.base_buffer):,} | Curve transitions: {len(memory.curve_buffer):,}")
    print(f"     Reward media demo: {avg_reward:.3f}")
    return total


# ──────────────────────────────────────────────────────────────────────
#  Reward Function — Giro Secco
# ──────────────────────────────────────────────────────────────────────

# ── Tempi umani dai session_logs (riferimenti per la reward) ──
HUMAN_BEST_TIME = 71.038    # Miglior giro umano (lap_017.h5)
HUMAN_WORST_TIME = 77.146   # Peggior giro umano (lap_001.h5)


def compute_step_reward(obs: dict, prev_dist: float, raw_obs: dict) -> tuple:
    """Reward per singolo step. Ritorna (reward, done, dist_raced).

    Componenti:
      + progress: avanzamento sulla pista (Δ distRaced)
      + speed:    bonus velocità longitudinale
      - center:   penalità quadratica per distanza dal centro
      - angle:    penalità per disallineamento
      - offtrack: terminazione + altissima penalità
      - spin:     terminazione se l'auto si gira (cos(angle) < 0)
    """
    speed_x = float(np.array(obs.get('speedX', 0.0)).flat[0])
    track_pos = float(np.array(obs.get('trackPos', 0.0)).flat[0])
    angle = float(np.array(obs.get('angle', 0.0)).flat[0])

    # Distanza percorsa (raw, non normalizzata)
    dist_raced = float(raw_obs.get('distRaced', 0.0))
    if isinstance(dist_raced, list):
        dist_raced = dist_raced[0]
    delta_dist = dist_raced - prev_dist

    # ── Componenti reward ──
    if speed_x < 0:
        # Retromarcia: nessun bonus di progresso, penalità scalata
        progress = 0.0
        speed_bonus = 0.0
        reverse_penalty = -abs(speed_x) * 0.1
    else:
        progress = delta_dist * 1.0
        speed_bonus = speed_x * 0.05
        reverse_penalty = 0.0

    center_penalty = -1.0 * (track_pos ** 2)
    angle_penalty = -1.5 * abs(angle)
    time_penalty = -0.1  # Costo costante per step: rende lo stallo intrinsecamente costoso

    reward = progress + speed_bonus + center_penalty + angle_penalty + reverse_penalty + time_penalty

    # ── Terminazione ──
    done = False

    # Uscita di pista
    if abs(track_pos) > 1.0:
        done = True
        reward = -100.0

    # Spin: l'auto si è girata
    if np.cos(angle) < 0:
        done = True
        reward = -100.0

    return reward, done, dist_raced


def compute_lap_bonus(lap_time: float, best_time: float) -> float:
    """Bonus/penalità per completamento giro basati sui tempi umani.

    Soglie (dai session_logs):
      Best umano:  71.038s (lap_017)
      Worst umano: 77.146s (lap_001)

    Reward:
      - Batte il best umano (< 71.038s):     +1000 base + 200/s di miglioramento
      - Tra best e worst umano:               +500 base
      - Poco più lento del worst (77-82s):    -50  (media penalità)
      - Molto più lento del worst (> 82s):    -100 (alta penalità)
    """
    if lap_time < HUMAN_BEST_TIME:
        # Premio cospicuo: ha battuto il miglior giro umano!
        improvement = HUMAN_BEST_TIME - lap_time
        return 1000.0 + improvement * 200.0
    elif lap_time <= HUMAN_WORST_TIME:
        # Giro nella fascia umana: buono, bonus base
        return 500.0
    elif lap_time <= 82.0:
        # Poco più lento del peggior giro umano: media penalità
        return -50.0
    else:
        # Molto più lento del peggior giro umano: alta penalità
        return -100.0


# ──────────────────────────────────────────────────────────────────────
#  Main Training Loop
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SAC RL Training — TORCS Giro Secco (RLPD)")
    parser.add_argument("--episodes", type=int, default=1000, help="Episodi di training")
    parser.add_argument("--max_steps", type=int, default=10000,
                        help="Max step per episodio (~200s a 50Hz, ~2.8x tempo umano worst)")
    parser.add_argument("--bc_weights", type=str, default="train_set/checkpoints/bc_policy.pth",
                        help="Path ai pesi BC per warm start")
    parser.add_argument("--demo_dir", type=str, default="train_set/laps",
                        help="Directory con i file HDF5 delle demo umane")
    parser.add_argument("--save_dir", type=str, default="train_set/checkpoints",
                        help="Directory per checkpoint")
    parser.add_argument("--target_time", type=float, default=75.0,
                        help="Tempo target umano in secondi (default: 75s)")
    parser.add_argument("--buffer_size", type=int, default=200000,
                        help="Capacità replay buffer")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--warmup_steps", type=int, default=5000,
                        help="Campioni nel buffer prima di iniziare gli update (default: 5000)")
    parser.add_argument("--actor_lr", type=float, default=1e-5,
                        help="Learning rate dell'actor (basso per preservare BC)")
    parser.add_argument("--critic_lr", type=float, default=3e-4,
                        help="Learning rate del critic")
    parser.add_argument("--bc_lambda", type=float, default=0.75,
                        help="Coefficiente regolarizzazione BC (0=disabilitato, default: 0.75)")
    parser.add_argument("--critic_warmup_steps", type=int, default=10000,
                        help="Step di pre-training offline del Critic prima del loop episodi")
    parser.add_argument("--actor_freeze_episodes", type=int, default=5,
                        help="Episodi in cui l'Actor è congelato (solo Critic si aggiorna)")
    parser.add_argument("--bc_decay_episodes", type=int, default=500,
                        help="Episodi su cui decadere bc_lambda linearmente fino a 0.1")
    parser.add_argument("--relaunch_every", type=int, default=20,
                        help="Rilancia TORCS ogni N episodi")
    parser.add_argument("--checkpoint_every", type=int, default=50,
                        help="Salva checkpoint ogni N episodi")
    parser.add_argument("--resume", type=str, default="",
                        help="Path a un checkpoint completo per riprendere il training")
    parser.add_argument("--exploration_sigma", type=float, default=0.1,
                        help="Deviazione standard del rumore Gaussiano aggiunto alle azioni in fase TRAIN (0=disabilitato)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state_dim = 29
    action_dim = 4

    os.makedirs(args.save_dir, exist_ok=True)

    print(f"\n{'=' * 64}")
    print(f"  🏎️  SAC REINFORCEMENT LEARNING (RLPD) — Giro Secco TORCS")
    print(f"  Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Episodi: {args.episodes} | Buffer: {args.buffer_size}")
    print(f"  Actor LR: {args.actor_lr} | Critic LR: {args.critic_lr}")
    print(f"  BC λ: {args.bc_lambda} (adattivo) | Tempo target: {args.target_time:.1f}s")
    print(f"  Exploration σ: {args.exploration_sigma} (adattivo) | Actor freeze: {args.actor_freeze_episodes} ep | Critic warmup: {args.critic_warmup_steps} step")
    if args.resume:
        print(f"  Resume da: {args.resume}")
    print(f"{'=' * 64}\n")

    # ── Agent (con LR separati e BC lambda) ──
    agent = SACAgent(state_dim, action_dim, device,
                     actor_lr=args.actor_lr,
                     critic_lr=args.critic_lr,
                     bc_lambda=args.bc_lambda)

    # ── Resume o Warm Start ──
    start_episode = 1
    best_lap_time = args.target_time
    best_completed_lap_time = float('inf')
    total_updates = 0

    if args.resume and os.path.exists(args.resume):
        print(f"  Ripristino checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        agent.actor.load_state_dict(ckpt['actor'])
        # Forza la correzione di log_std_linear anche su resume da checkpoint precedenti
        with torch.no_grad():
            torch.nn.init.constant_(agent.actor.log_std_linear.weight, 0.0)
            torch.nn.init.constant_(agent.actor.log_std_linear.bias, -5.0)
        agent.critic.load_state_dict(ckpt['critic'])
        agent.critic_target.load_state_dict(ckpt['critic_target'])
        agent.actor_optimizer.load_state_dict(ckpt['actor_optimizer'])
        agent.critic_optimizer.load_state_dict(ckpt['critic_optimizer'])
        start_episode = ckpt.get('episode', 0) + 1
        best_lap_time = ckpt.get('best_lap_time', args.target_time)
        best_completed_lap_time = ckpt.get('best_completed_lap_time', float('inf'))
        if best_completed_lap_time == float('inf') and 'best_lap_time' in ckpt:
            ckpt_best = ckpt['best_lap_time']
            if ckpt_best < args.target_time:
                best_completed_lap_time = ckpt_best
        total_updates = ckpt.get('total_updates', 0)
        # Usa bc_lambda da CLI (non dal checkpoint) per permettere tuning su resume
        ckpt_lambda = ckpt.get('bc_lambda', args.bc_lambda)
        if abs(args.bc_lambda - ckpt_lambda) > 1e-6:
            print(f"  ⚠️  bc_lambda override: checkpoint={ckpt_lambda:.3f} → CLI={args.bc_lambda:.3f}")
        agent.bc_lambda = args.bc_lambda
        print(f"  ✅ Checkpoint ripristinato: ep={start_episode-1}, "
              f"best={best_lap_time:.3f}s, best_completed={best_completed_lap_time:.3f}s, updates={total_updates}, λ_bc={agent.bc_lambda:.3f}")
    else:
        # Warm Start da BC
        agent.actor.load_bc_weights(args.bc_weights, device)

    # ── BC Reference Model (congelato, per regularization) ──
    if args.bc_lambda > 0:
        agent.load_bc_reference(args.bc_weights)

    # ── Replay Buffer ──
    memory = ReplayBuffer(capacity=args.buffer_size)

    # ── Pre-fill buffer con demo umane (RLPD) ──
    print(f"\n  Caricamento demo umane nel replay buffer...")
    prefill_buffer_from_demos(memory, args.demo_dir)

    # ── Offline Critic Pre-training ──
    # Addestra solo il Critic sulle demo prima di iniziare il loop degli episodi.
    # Questo permette al Critic di apprendere una Q-function ragionevole
    # PRIMA che possa influenzare l'Actor, prevenendo catastrophic forgetting.
    # SKIP se stiamo facendo resume: il checkpoint contiene già un Critic addestrato.
    is_resuming = args.resume and os.path.exists(args.resume)
    if args.critic_warmup_steps > 0 and len(memory) > args.batch_size and not is_resuming:
        print(f"\n  🧠 Critic pre-training offline: {args.critic_warmup_steps} step...")
        # Tau più basso durante il pre-training per stabilizzare i target Bellman.
        # Con tau=0.005 e 10000 step, il target network converge completamente
        # al main network (1-(1-0.005)^10000 ≈ 1.0), eliminando la stabilizzazione.
        # tau=0.001 mantiene il target network come ancora stabile.
        original_tau = agent.tau
        agent.tau = 0.001
        for cw_step in range(1, args.critic_warmup_steps + 1):
            cw_loss = agent.update_critic_only(memory, args.batch_size)
            if cw_step % 1000 == 0 or cw_step == 1:
                pct = 100 * cw_step / args.critic_warmup_steps
                print(f"    [{pct:5.1f}%] step {cw_step:6d}/{args.critic_warmup_steps} | critic_loss: {cw_loss:.4f}")
        agent.tau = original_tau
        # Sincronizza il target network al critic addestrato per partire allineati
        agent.critic_target.load_state_dict(agent.critic.state_dict())
        print(f"  ✅ Critic pre-training completato (target sync).\n")
    elif is_resuming:
        print(f"\n  ⏭️  Critic pre-training SKIPPATO (resume da checkpoint, Critic già addestrato).\n")

    # ── Training log ──
    log_dir = os.path.join(os.path.dirname(args.save_dir), "session_logs")
    os.makedirs(log_dir, exist_ok=True)
    if args.resume and os.path.exists(args.resume):
        # Riprendi il log della sessione originale (salvato nel checkpoint)
        log_path = ckpt.get('log_path', '')
        if not log_path or not os.path.exists(log_path):
            # Fallback: trova il log il cui ultimo episodio corrisponde al checkpoint
            existing_logs = glob.glob(os.path.join(log_dir, "sac_training_*.log"))
            target_ep = start_episode - 1  # ultimo episodio completato prima del resume
            matched_log = None
            for lf in existing_logs:
                try:
                    with open(lf, 'r') as flog:
                        lines = [l.strip() for l in flog if l.startswith('ep=')]
                    if lines:
                        last_ep = int(lines[-1].split(',')[0].split('=')[1])
                        if last_ep == target_ep:
                            matched_log = lf
                            break
                except Exception:
                    continue
            if matched_log:
                log_path = matched_log
                print(f"  📄 Resume: appendo al log originale (ep={target_ep}): {os.path.basename(log_path)}")
            else:
                log_path = os.path.join(log_dir, f"sac_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
                print(f"  📄 Nessun log corrispondente trovato, creo: {os.path.basename(log_path)}")
        else:
            print(f"  📄 Resume: appendo al log originale: {os.path.basename(log_path)}")
    else:
        log_path = os.path.join(log_dir, f"sac_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    # ── Ambiente ──
    print("  Inizializzazione TORCS...")
    env = TorcsEnv(vision=False, throttle=True, gear_change=True, early_termination=False)

    try:
        # Contatore episodi nella fase TRAIN (post-freeze), per CPI schedule
        # La freeze boundary è assoluta: ep < (1 + actor_freeze_episodes)
        # Per resume da ep=150 con freeze=50 → counter = max(0, 150 - 51) = 99
        freeze_boundary = 1 + args.actor_freeze_episodes
        train_ep_counter = max(0, start_episode - freeze_boundary)

        # ── Adaptive Scheduler ──
        scheduler = AdaptiveScheduler(
            bc_lambda=agent.bc_lambda,
            sigma=args.exploration_sigma,
            cpi_weight=0.01,
            alpha=agent.alpha,
        )
        if is_resuming and 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])
            print(f"  ✅ Scheduler adattivo ripristinato: λ_bc={scheduler.bc_lambda:.3f}, "
                  f"σ={scheduler.sigma:.3f}, cpi={scheduler.cpi_weight:.3f}, α={scheduler.alpha:.4f}, mastery={scheduler.mastery:.3f}")
        else:
            print(f"  🔧 Scheduler adattivo inizializzato: λ_bc={scheduler.bc_lambda:.3f}, "
                  f"σ={scheduler.sigma:.3f}, eval ogni {scheduler.eval_every} ep, mastery={scheduler.mastery:.3f}")

        for ep in range(start_episode, args.episodes + 1):

            # ── Reset ──
            need_relaunch = (ep == 1) or (ep % args.relaunch_every == 0)
            if ep == 1:
                obs = env.reset(relaunch=True)
            else:
                obs = env.reset(relaunch=need_relaunch)

            state = flatten_state(obs)
            episode_reward = 0.0

            # Raw obs per distRaced e lapTime
            raw = env.client.S.d
            prev_dist = float(raw.get('distRaced', 0.0))
            if isinstance(prev_dist, list):
                prev_dist = prev_dist[0]
            prev_last_lap = float(raw.get('lastLapTime', 0.0))
            if isinstance(prev_last_lap, list):
                prev_last_lap = prev_last_lap[0]

            stall_counter = 0  # Conta step consecutivi a bassa velocità
            lap_completed = False
            ep_lap_time = 0.0

            # Determina la fase di training per questo episodio
            is_freeze = (ep < freeze_boundary)
            if not is_freeze:
                train_ep_counter += 1  # Incrementa contatore fase TRAIN

            for step in range(1, args.max_steps + 1):
                # ── Selezione azione ──
                # In fase FREEZE: azione deterministica (pura BC policy)
                # In fase TRAIN: azione stocastica (esplorazione SAC)
                if is_freeze:
                    action = agent.select_action(state, evaluate=True)
                else:
                    action = agent.select_action(state)
                    # ── Exploration Noise ──
                    # Il log_std è congelato a -5.0 (std≈0.007), quasi deterministico.
                    # Senza rumore esterno, l'agente percorre la stessa traiettoria
                    # ad ogni episodio e non può scoprire come superare le curve
                    # dove la BC policy crasha. Il rumore Gaussiano aggiunge
                    # diversità alle esperienze raccolte senza corrompere i pesi.
                    # Applicato solo a steer/accel/brake (non gear).
                    if scheduler.sigma > 0:
                        noise = np.random.normal(0, scheduler.sigma, size=3)
                        action[0] = np.clip(action[0] + noise[0], -1.0, 1.0)  # steer
                        action[1] = np.clip(action[1] + noise[1], -1.0, 1.0)  # accel
                        action[2] = np.clip(action[2] + noise[2], -1.0, 1.0)  # brake



                # ── De-normalizza e step ──
                env_action = denormalize_action(action)
                if ep <= 2 and (step <= 10 or step % 50 == 0 or step >= 310):
                    print(f"      [DEBUG Step {step:3d}] SpeedX: {state[21]*50:.1f} km/h | TrackPos: {state[20]:.3f} | Angle: {state[0]:.3f} | Action (env): Steer={env_action[0]:.3f}, Accel={env_action[1]:.3f}, Brake={env_action[2]:.3f}, Gear={env_action[3]}")
                next_obs, _, env_done, _ = env.step(env_action)
                next_state = flatten_state(next_obs)

                # ── Reward ──
                raw = env.client.S.d
                reward, custom_done, prev_dist = compute_step_reward(
                    next_obs, prev_dist, raw
                )

                # FIX #4: terminazione per stallo (velocità < 5 km/h per 100+ step)
                # speedX è normalizzato dal wrapper (÷50), 5 km/h = 0.1
                speed_x_raw = float(np.array(next_obs.get('speedX', 0.0)).flat[0])
                if abs(speed_x_raw) < 0.1:  # 0.1 = 5 km/h / 50
                    stall_counter += 1
                else:
                    stall_counter = 0
                if stall_counter >= 100 and not custom_done:
                    custom_done = True
                    reward = -50.0  # Penalità moderata per stallo

                # ── Lap completion ──
                current_last_lap = float(raw.get('lastLapTime', 0.0))
                if isinstance(current_last_lap, list):
                    current_last_lap = current_last_lap[0]
                if current_last_lap > 0.0 and abs(current_last_lap - prev_last_lap) > 0.01:
                    lap_completed = True
                    ep_lap_time = current_last_lap
                    bonus = compute_lap_bonus(ep_lap_time, best_lap_time)
                    reward += bonus

                    # Salvataggio del checkpoint ad ogni episodio con tempo migliore (o primo giro valido)
                    if ep_lap_time < best_completed_lap_time:
                        old_completed_best = best_completed_lap_time
                        best_completed_lap_time = ep_lap_time

                        # Salva checkpoint completo
                        best_ckpt_path = os.path.join(args.save_dir, "sac_checkpoint_best.pth")
                        torch.save({
                            'episode': ep,
                            'actor': agent.actor.state_dict(),
                            'critic': agent.critic.state_dict(),
                            'critic_target': agent.critic_target.state_dict(),
                            'actor_optimizer': agent.actor_optimizer.state_dict(),
                            'critic_optimizer': agent.critic_optimizer.state_dict(),
                            'best_lap_time': best_lap_time,
                            'best_completed_lap_time': best_completed_lap_time,
                            'total_updates': total_updates,
                            'bc_lambda': agent.bc_lambda,
                            'log_path': log_path,
                            'scheduler': scheduler.state_dict(),
                        }, best_ckpt_path)

                        # Salva anche l'actor best (.pth) per test_agent.py
                        best_actor_path = os.path.join(args.save_dir, "sac_actor_best.pth")
                        torch.save(agent.actor.state_dict(), best_actor_path)

                        if old_completed_best == float('inf'):
                            print(f"  🏆 PRIMO GIRO COMPLETATO: {ep_lap_time:.3f}s | Checkpoint completo salvato: {best_ckpt_path}")
                        else:
                            print(f"  🏆 NUOVO RECORD LAPTIME: {ep_lap_time:.3f}s (precedente: {old_completed_best:.3f}s) | Checkpoint completo salvato: {best_ckpt_path}")

                    if ep_lap_time < best_lap_time:
                        old_best = best_lap_time
                        best_lap_time = ep_lap_time
                        print(f"  🏆 NUOVO BEST LAP (target): {ep_lap_time:.3f}s (precedente: {old_best:.3f}s)")

                        # ── Decay λ_bc basato sulla performance (bonus) ──
                        # Override: quando l'agente batte il best, forza λ_bc
                        # in base alla vicinanza al tempo umano migliore.
                        if HUMAN_WORST_TIME > HUMAN_BEST_TIME:
                            progress = (HUMAN_WORST_TIME - best_lap_time) / (HUMAN_WORST_TIME - HUMAN_BEST_TIME)
                            progress = max(0.0, min(1.0, progress))  # clamp [0, 1]
                            new_lambda = max(0.1, 1.0 * (1.0 - 0.9 * progress))
                            scheduler.bc_lambda = new_lambda
                            agent.bc_lambda = new_lambda
                            print(f"    📉 BC λ aggiornato (performance): {new_lambda:.3f} (progress: {progress:.1%})")

                done = custom_done or env_done or lap_completed
                mask = 0.0 if done else 1.0
                memory.push(state, action, reward, next_state, mask)

                state = next_state
                episode_reward += reward

                # ── Update ──
                if len(memory) > max(args.warmup_steps, args.batch_size) and step % 4 == 0:
                    if is_freeze:
                        # Fase FREEZE: solo Critic si aggiorna, Actor intatto
                        agent.update_critic_only(memory, args.batch_size)
                    else:
                        # Fase TRAIN: update completo (Actor + Critic)
                        qf_l, pi_l, adv_r = agent.update_parameters(
                            memory, args.batch_size,
                            train_episode=train_ep_counter,
                            cpi_weight=scheduler.cpi_weight)
                        scheduler.record_update(qf_l, adv_r)
                    total_updates += 1

                if done:
                    break

            # ── Logging episodio ──
            status = "LAP" if lap_completed else "FAIL"
            lap_str = f"{ep_lap_time:.3f}s" if lap_completed else "N/A"
            phase = "FREEZE" if is_freeze else "TRAIN"

            # Registra nel scheduler adattivo
            scheduler.record_episode(episode_reward, step,
                                     lap_completed=lap_completed,
                                     lap_time=ep_lap_time)

            print(
                f"  Ep {ep:4d}/{args.episodes} | {phase} | {status} | "
                f"Reward: {episode_reward:8.1f} | Steps: {step:5d} | "
                f"LapTime: {lap_str} | Best: {best_lap_time:.3f}s | "
                f"Updates: {total_updates} | α: {scheduler.alpha:.4f} | "
                f"λ_bc: {scheduler.bc_lambda:.3f} | σ: {scheduler.sigma:.3f} | "
                f"mastery: {scheduler.mastery:.3f}"
            )

            with open(log_path, 'a') as f:
                f.write(
                    f"ep={ep},reward={episode_reward:.2f},steps={step},"
                    f"lap={status},lap_time={ep_lap_time:.3f},"
                    f"best={best_lap_time:.3f},updates={total_updates},"
                    f"bc_lambda={scheduler.bc_lambda:.3f},sigma={scheduler.sigma:.3f},"
                    f"alpha={scheduler.alpha:.4f},cpi_weight={scheduler.cpi_weight:.3f},"
                    f"mastery={scheduler.mastery:.3f}\n"
                )

            # ── Adattamento parametri ──
            if scheduler.should_adapt(ep):
                changes, metrics = scheduler.adapt()
                # Applica i parametri adattivi all'agente
                agent.bc_lambda = scheduler.bc_lambda
                agent.alpha = scheduler.alpha
                if changes:
                    print(f"    🔧 Adattamento: {changes}")
                    print(f"       Metriche: R_trend={metrics['reward_trend']:+.1f} "
                          f"surv={metrics['survival']:.2f} "
                          f"q_stab={metrics['q_stability']:.2f} "
                          f"adv={metrics['adv_hit_rate']:.2%} "
                          f"mastery={metrics['mastery']:.3f}")

            # ── Checkpoint (salvataggio completo per resume) ──
            if ep % args.checkpoint_every == 0:
                ckpt_path = os.path.join(args.save_dir, f"sac_checkpoint_ep{ep:04d}.pth")
                torch.save({
                    'episode': ep,
                    'actor': agent.actor.state_dict(),
                    'critic': agent.critic.state_dict(),
                    'critic_target': agent.critic_target.state_dict(),
                    'actor_optimizer': agent.actor_optimizer.state_dict(),
                    'critic_optimizer': agent.critic_optimizer.state_dict(),
                    'best_lap_time': best_lap_time,
                    'best_completed_lap_time': best_completed_lap_time,
                    'total_updates': total_updates,
                    'bc_lambda': agent.bc_lambda,
                    'log_path': log_path,
                    'scheduler': scheduler.state_dict(),
                }, ckpt_path)
                print(f"    💾 Checkpoint completo: {ckpt_path}")

    except KeyboardInterrupt:
        print(f"\n\n  🛑 Training interrotto dall'utente all'episodio {ep}.")

    finally:
        # ── Salvataggio finale (checkpoint completo) ──
        final_ckpt = os.path.join(args.save_dir, "sac_checkpoint_latest.pth")
        final_actor = os.path.join(args.save_dir, "sac_actor_final.pth")
        torch.save({
            'episode': ep,
            'actor': agent.actor.state_dict(),
            'critic': agent.critic.state_dict(),
            'critic_target': agent.critic_target.state_dict(),
            'actor_optimizer': agent.actor_optimizer.state_dict(),
            'critic_optimizer': agent.critic_optimizer.state_dict(),
            'best_lap_time': best_lap_time,
            'best_completed_lap_time': best_completed_lap_time,
            'total_updates': total_updates,
            'bc_lambda': agent.bc_lambda,
            'log_path': log_path,
            'scheduler': scheduler.state_dict(),
        }, final_ckpt)
        torch.save(agent.actor.state_dict(), final_actor)
        print(f"\n  Checkpoint finale: {final_ckpt}")
        print(f"  Actor finale: {final_actor}")
        print(f"  Best lap time (target) raggiunto: {best_lap_time:.3f}s")
        print(f"  Best completed lap time raggiunto: {best_completed_lap_time:.3f}s")
        print(f"  Log training: {log_path}")

        env.end()


if __name__ == "__main__":
    main()
