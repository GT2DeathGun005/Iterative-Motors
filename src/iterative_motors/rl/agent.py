"""Agente TD3+BC: Actor + Twin Critic, update ibrida RL/BC, checkpoint con resume robusto.

Implementa Twin Critic, Delayed Policy Update, Target Policy Smoothing e Polyak averaging
(Fujimoto et al. 2018) con la BC Penalty mascherata sui soli campioni esperti e la
normalizzazione λ del TD3+BC (Fujimoto & Gu 2021). Gli attributi ``refine_mode``,
``actor_frozen`` e ``refine_bc_weight`` sono controllati dal training loop (curriculum/refinement).
"""

import os
import re
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from ..models.networks import Actor, Critic
from ..common.constants import TRACK_LENGTH_M
from ..common.checkpoint import safe_save, safe_save_npz, safe_read_float, _checkpoint_candidates
from .reward import _is_plausible_eval_dist, _is_plausible_eval_score

# Rumore esplorativo iniziale (annealato dal training loop da 0.10 a 0.04).
_EXPL_NOISE_START = 0.10


class TD3BCAgent:
    """Agente TD3+BC: coordina i modelli neurali e l'intero ciclo di ottimizzazione.

    Implementa il TD3 (Fujimoto et al. 2018) con il vincolo di Behavioral Cloning del TD3+BC
    (Fujimoto & Gu 2021). Gestisce l'interazione tra Actor e Twin Critic attraverso i passaggi:

      - Selezione delle azioni, con o senza rumore esplorativo gaussiano (``select_action``).
      - Ottimizzazione del Critic minimizzando l'errore di differenza temporale (TD error), cioè
        la discrepanza tra la stima Q corrente e il target di Bellman calcolato con le reti target.
      - Ottimizzazione dell'Actor minimizzando la loss ibrida ``-λ·Q(s,π(s)) + BC_penalty``, dove
        la BC penalty è applicata SOLO ai campioni esperti (mascheramento rigoroso).
      - Stabilizzazione tramite Polyak averaging (soft update delle reti target con tasso τ).
      - Gestione degli stati speciali del curriculum: warm-up del Critic, congelamento temporaneo
        dell'Actor dopo un rollback, e modalità di refinement (vincolo BC ridotto, Critic fermo).

    Gli attributi ``refine_mode``, ``actor_frozen`` e ``refine_bc_weight`` sono pilotati dal
    training loop (refinement/curriculum); qui hanno default robusti.
    """

    def __init__(self, device="cuda"):
        self.device = torch.device(device)
        self.gamma = 0.99   # Fattore di sconto temporale per il calcolo del valore Q futuro
        self.tau = 0.005    # Parametro per l'aggiornamento soft Polyak delle reti target
        self.policy_freq = 2  # Frequenza aggiornamento Actor vs Critic (Delayed Policy Update)
        self.expl_noise = _EXPL_NOISE_START  # std rumore esplorativo, annealata dal training loop
        self.bc_alpha = 2.5  # alpha TD3+BC: piu' alto = piu' peso al RL rispetto alla BC

        # Stato del curriculum/refinement (impostato dal training loop; default robusti).
        self.refine_mode = False
        self.actor_frozen = False
        self.refine_bc_weight = 0.3

        # Inizializzazione Actor (online e target)
        self.actor = Actor().to(self.device)
        self.actor_target = Actor().to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Inizializzazione Critic (online e target)
        self.critic = Critic().to(self.device)
        self.critic_target = Critic().to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Ottimizzatori Adam
        actor_params = [p for p in self.actor.parameters() if p.requires_grad]
        self.actor_optimizer = optim.Adam(actor_params, lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)

    def select_action(self, state, evaluate=False):
        """Azione continua 3D per lo stato corrente (deterministica se evaluate=True)."""
        state_t = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            cont_action = self.actor.sample(state_t, evaluate=evaluate, noise_std=self.expl_noise)
        return cont_action.cpu().numpy()[0]

    def update(self, online_memory, elite_memory, expert_memory, batch_size, global_step):
        """Esegue un singolo passo di addestramento del Critic ed (eventualmente) dell'Actor.

        Passaggi:

          1. Campiona un batch ibrido a 3 vie: 25% esperti (pilota umano), 15% elite (migliori run
             autonome) e 60% online (esplorazione corrente). Se online o elite contengono pochi
             dati, la quota mancante è compensata con campioni esperti (sempre disponibili).
          2. Riscala la ricompensa (``reward_scale = 0.02``) per mantenere i valori Q in un range
             numericamente stabile.
          3. Aggiorna il Critic (Twin Q):
             - calcola l'azione del prossimo stato con Target Policy Smoothing (rumore clippato);
             - estrae Q1_target(s', a') e Q2_target(s', a') dalle reti target;
             - prende il MINIMO tra le due (anti-sovrastima) e forma il target di Bellman
               ``r + γ·mask·min(Q1, Q2)``;
             - minimizza l'MSE delle stime correnti rispetto al target (con gradient clipping).
             In modalità refinement questo aggiornamento è DISATTIVATO: l'Actor si raffina verso una
             value function fissa.
          4. Aggiorna l'Actor (Delayed Policy Update, ogni ``policy_freq`` step) solo dopo il warm-up
             di 15000 step e se non è congelato:
             - componente RL: massimizza Q1(s, π(s));
             - BC penalty: MSE tra azione predetta e azione esperta, calcolata SOLO sui campioni
               con ``expert_mask > 0.5``, più una penalità di mutua esclusione gas/freno;
             - coefficiente dinamico ``λ = bc_alpha / mean(|Q(s, π(s))|)`` che mantiene confrontabili
               la scala del termine RL e di quello BC (Fujimoto & Gu 2021);
             - loss totale ``λ·(-Q) + bc_weight·BC_penalty`` (``bc_weight`` ridotto in refinement).
          5. Soft update (Polyak, τ) delle reti target ogni ``policy_freq`` step, A PRESCINDERE da
             warm-up e congelamento (come nel TD3 originale): tenerlo legato all'update dell'Actor
             lasciava i target del Critic fermi per decine di migliaia di step, facendo divergere
             stime correnti e target.

        Ritorna ``(critic_loss, actor_loss, 0.0)``; ``actor_loss`` è 0 se l'Actor non è stato aggiornato.
        """
        # Hybrid Sampling a 3 vie: Expert + Online + Elite
        b_expert = int(batch_size * 0.25)
        b_elite = min(int(batch_size * 0.15), len(elite_memory.buffer))
        b_online = min(batch_size - b_expert - b_elite, len(online_memory.buffer))
        b_expert = batch_size - b_online - b_elite  # il resto dall'expert (sempre disponibile)

        parts = [expert_memory.sample(b_expert)]

        if b_online > 0:
            parts.append(online_memory.sample(b_online))
        if b_elite > 0:
            parts.append(elite_memory.sample(b_elite))

        state_b = np.concatenate([p[0] for p in parts], axis=0)
        action_b = np.concatenate([p[1] for p in parts], axis=0)
        reward_b = np.concatenate([p[2] for p in parts], axis=0)
        next_state_b = np.concatenate([p[3] for p in parts], axis=0)
        mask_b = np.concatenate([p[4] for p in parts], axis=0)
        expert_mask_b = np.concatenate([p[5] for p in parts], axis=0)

        # Riscalatura della ricompensa per mantenere la magnitudo del Critic in un range sano
        reward_scale = 0.02
        reward_b = reward_b * reward_scale

        state_b = torch.FloatTensor(state_b).to(self.device)
        next_state_b = torch.FloatTensor(next_state_b).to(self.device)
        action_b = torch.FloatTensor(action_b).to(self.device)
        reward_b = torch.FloatTensor(reward_b).to(self.device).unsqueeze(1)
        mask_b = torch.FloatTensor(mask_b).to(self.device).unsqueeze(1)
        expert_mask_b = torch.FloatTensor(expert_mask_b).to(self.device).unsqueeze(1)

        # Aggiornamento del Critic (Bellman con Twin Q-Network)
        with torch.no_grad():
            noise = (torch.randn_like(action_b) * 0.2).clamp(-0.5, 0.5)  # Target Policy Smoothing
            next_action = self.actor_target(next_state_b)
            next_action = (next_action + noise).clamp(-1.0, 1.0)

            q1_next, q2_next = self.critic_target(next_state_b, next_action)
            min_q_next = torch.min(q1_next, q2_next)
            target_q = reward_b + mask_b * self.gamma * min_q_next

        q1, q2 = self.critic(state_b, action_b)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        # In refinement l'aggiornamento del Critic è disattivato (value function fissa, vincolo BC ridotto).
        if not getattr(self, 'refine_mode', False):
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.critic_optimizer.step()

        actor_loss_val = 0.0

        # Delayed Policy Update (ogni 2 step del Critic) dopo il warm-up di 15000 step.
        if global_step >= 15000 and global_step % self.policy_freq == 0 and not getattr(self, 'actor_frozen', False):
            pi = self.actor(state_b)
            q1_pi, _ = self.critic(state_b, pi)

            actor_loss_td3 = -q1_pi.mean()  # Componente RL: massimizza Q1(s, pi(s))

            # BC Penalty (mascheramento rigoroso: solo sotto-batch Expert)
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
                bc_penalty = (steer_loss * 2.0 + accel_loss + brake_loss * 2.0)
            else:
                bc_penalty = torch.tensor(0.0, device=self.device)

            # Penalità per evitare acceleratore e freno premuti insieme
            mutual_exclusion_penalty = (det_accel * det_brake).mean()
            bc_penalty = bc_penalty + (mutual_exclusion_penalty * 0.1)

            # Normalizzazione λ del TD3+BC (Fujimoto & Gu, 2021)
            Q_abs_mean = q1_pi.abs().mean().detach().clamp(min=1e-5)
            dynamic_alpha = self.bc_alpha / Q_abs_mean

            bc_weight = self.refine_bc_weight if getattr(self, 'refine_mode', False) else 1.0

            total_actor_loss = dynamic_alpha * actor_loss_td3 + (bc_weight * bc_penalty)

            self.actor_optimizer.zero_grad()
            total_actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()
            actor_loss_val = total_actor_loss.item()

        # Soft Update (Polyak Averaging, τ) ogni policy_freq step, a prescindere da warm-up/freeze.
        if global_step % self.policy_freq == 0:
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return critic_loss.item(), actor_loss_val, 0.0

    def save_checkpoint(self, filepath, episode, global_step, memory, elite_memory=None,
                        best_lap_time=float('inf'), best_eval_dist=0.0, best_distance=0.0):
        """Salva in modo atomico lo stato completo dell'agente e dei replay buffer.

        Per garantire l'integrità ed evitare disallineamenti in caso di arresto improvviso:
          1. crea la cartella ``buffers/`` accanto al checkpoint;
          2. salva i Replay Buffer (online ed elite) in ``.npz`` PRIMA dei pesi, perché sono
             l'operazione di I/O più onerosa;
          3. costruisce il dizionario con pesi di Actor/Critic, reti target, ottimizzatori,
             episodio, global_step e metriche di record;
          4. lo salva in ``.pth`` con ``safe_save`` (scrittura temporanea, fsync, rotazione backup).

        Se l'interruzione avviene a metà, l'assenza del ``.pth`` aggiornato segnala al resume che i
        buffer nuovi non sono allineati, così verranno ignorati a favore dei backup coerenti.
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
        """Ripristina lo stato dell'agente e dei buffer da un checkpoint, in modo robusto (resume).

        Gestione del ripristino:
          1. scansiona i candidati (incluso ``backups/``) per trovare un ``.pth`` leggibile;
          2. se il file contiene lo stato completo di training, ripristina pesi, reti target,
             ottimizzatori e variabili di avanzamento (episodio, global_step, record);
          3. valida i record memorizzati: se ``best_eval_dist`` (uno score) o ``best_distance``
             risultano implausibili, li recupera dai sidecar testuali (``td3_det_best_dist.txt``);
          4. se il file contiene SOLO i pesi dell'Actor (es. un checkpoint estratto per il test),
             esegue un warm-start dei soli parametri di guida azzerando ottimizzatori e buffer;
          5. carica i Replay Buffer forzando la coerenza temporale: non carica buffer con timestamp
             successivo a quello del ``.pth`` (indicherebbe un salvataggio successivo interrotto),
             ricadendo sui backup allineati; in mancanza, recupero d'emergenza dal più recente.

        Se non trova alcun checkpoint, prova a recuperare l'ultimo episodio dal log di training.

        Ritorna ``(episode, global_step, best_lap_time, best_eval_dist, best_distance)``.
        """
        buffer_dir = os.path.join(os.path.dirname(filepath), 'buffers')
        base_name = os.path.basename(filepath).replace('.pth', '')
        buffer_path = os.path.join(buffer_dir, f"{base_name}_buffer.npz")
        elite_buffer_path = os.path.join(buffer_dir, f"{base_name}_elite_buffer.npz")

        def _load_buffer_aligned(buffer_obj, path, label, loaded_checkpoint_path=None):
            """Carica il buffer più recente non successivo al checkpoint .pth (anti-disallineamento)."""
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

            # Recupero di emergenza: nessun backup allineato integro -> usa il più nuovo disponibile
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
                    if 'actor_target' in checkpoint:
                        self.actor_target.load_state_dict(checkpoint['actor_target'], strict=False)
                    self.critic.load_state_dict(checkpoint['critic'])
                    self.critic_target.load_state_dict(checkpoint['critic_target'])
                    self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
                    self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])

                    best_lap_time = checkpoint.get('best_lap_time', float('inf'))
                    best_eval_dist = checkpoint.get('best_eval_dist', 0.0)
                    best_distance = checkpoint.get('best_distance', 0.0)
                    # best_eval_dist è uno SCORE (distanza o equivalente-tempo): soglia degli score.
                    if not _is_plausible_eval_score(best_eval_dist):
                        det_best_dist_txt = 'train_set/checkpoints/td3_det_best_dist.txt'
                        sidecar_best_eval_dist = safe_read_float(det_best_dist_txt, 0.0)
                        print(
                            f"best_eval_dist={best_eval_dist:.2f} non plausibile come score di eval; "
                            f"uso sidecar {sidecar_best_eval_dist:.2f}."
                        )
                        best_eval_dist = sidecar_best_eval_dist if _is_plausible_eval_score(sidecar_best_eval_dist) else 0.0
                    if not _is_plausible_eval_dist(best_distance):
                        best_distance = min(best_eval_dist, TRACK_LENGTH_M)
                    episode = checkpoint['episode']
                    global_step = checkpoint['global_step']
                    if candidate != filepath:
                        print(f"Checkpoint principale non usato: recupero da backup {candidate}")
                    print(f"Checkpoint caricato: ripresa dall'Episodio {episode}")
                else:
                    # File di soli pesi dell'actor (es. td3_expl_best_dist.pth).
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
            best_distance = min(best_eval_dist, TRACK_LENGTH_M)

        _load_buffer_aligned(memory, buffer_path, "Replay Buffer", loaded_checkpoint_path if loaded_ok else None)
        if elite_memory:
            _load_buffer_aligned(elite_memory, elite_buffer_path, "Elite Buffer", loaded_checkpoint_path if loaded_ok else None)

        return episode, global_step, best_lap_time, best_eval_dist, best_distance
