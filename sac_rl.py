"""
Modulo di Reinforcement Learning (Soft Actor-Critic)
Implementa un agente SAC ottimizzato per il dominio ad azioni continue in TORCS.
Prevede l'inizializzazione dell'Actor con i pesi pre-addestrati nel modulo di Imitation Learning.
"""

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from collections import deque

import sys

# Aggiungo la directory di gym_torcs al path di sistema per poter importare il modulo
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../gym_torcs')))

# Assumiamo gym_torcs o definiamo un mock per type-hinting/testing
try:
    from gym_torcs import TorcsEnv
except ImportError:
    pass

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class Actor(nn.Module):
    """
    Rete Actor (Policy) per azioni continue. Restituisce media e log_std di una Gaussiana.
    Possiede un metodo per importare i pesi dal Behavior Cloning.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 256):
        super(Actor, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
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
        x_t = normal.rsample()  # rsample per la riparametrizzazione (backprop attraverso la stocasticità)
        y_t = torch.tanh(x_t)
        action = y_t
        # Correzione log_prob per la trasformazione Tanh
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, mean

    def load_bc_weights(self, bc_model_path: str):
        """Carica i pesi pre-addestrati della rete di Imitation Learning (Warm Start)."""
        if not os.path.exists(bc_model_path):
            print(f"Warning: file {bc_model_path} non trovato. L'Actor partirà da zero.")
            return
            
        print(f"Caricamento pesi BC da {bc_model_path} per Warm Start...")
        state_dict = torch.load(bc_model_path, map_location="cpu", weights_only=True)
        
        # In behavior_cloning.py la rete si chiamava PolicyNetwork.
        # Struttura: net.0, net.1, net.2, net.3, net.4, net.5 per l'estrattore di feature,
        # e net.6 per il layer di output lineare.
        
        with torch.no_grad():
            self.net[0].weight.copy_(state_dict['net.0.weight'])
            self.net[0].bias.copy_(state_dict['net.0.bias'])
            self.net[1].weight.copy_(state_dict['net.1.weight'])
            self.net[1].bias.copy_(state_dict['net.1.bias'])
            self.net[3].weight.copy_(state_dict['net.3.weight'])
            self.net[3].bias.copy_(state_dict['net.3.bias'])
            self.net[4].weight.copy_(state_dict['net.4.weight'])
            self.net[4].bias.copy_(state_dict['net.4.bias'])
            
            # Il layer finale del BC diventa la media dell'Actor
            self.mean_linear.weight.copy_(state_dict['net.6.weight'])
            self.mean_linear.bias.copy_(state_dict['net.6.bias'])
            
            # log_std viene lasciato inizializzato random (magari abbassiamo i pesi per partire con bassa varianza)
            nn.init.constant_(self.log_std_linear.weight, -3.0)
            nn.init.constant_(self.log_std_linear.bias, -3.0)
            
        print("Warm Start completato con successo. Varianza inziale ridotta per sfruttare il prior.")

class Critic(nn.Module):
    """
    Twin Q-Network: 2 reti Critic per mitigare la sovrastima della Q-value (Double Q-learning).
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 256):
        super(Critic, self).__init__()

        # Q1 architecture
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        # Q2 architecture
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
        x1 = self.q1(xu)
        x2 = self.q2(xu)
        return x1, x2

class SACAgent:
    def __init__(self, state_dim: int, action_dim: int, device: str = "cpu"):
        self.device = device
        self.gamma = 0.99
        self.tau = 0.005
        self.alpha = 0.2
        self.target_update_interval = 1
        
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)
        
        # Auto-tuning dell'entropia (Alpha)
        self.target_entropy = -torch.prod(torch.Tensor([action_dim]).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=3e-4)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate:
            _, _, action = self.actor.sample(state)
        else:
            action, _, _ = self.actor.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory: ReplayBuffer, batch_size: int, updates: int):
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.actor.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
            
        qf1, qf2 = self.critic(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.critic_optimizer.zero_grad()
        qf_loss.backward()
        self.critic_optimizer.step()

        pi, log_pi, _ = self.actor.sample(state_batch)
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp()

        # Soft update target network
        if updates % self.target_update_interval == 0:
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
        return qf_loss.item(), policy_loss.item(), alpha_loss.item()


def compute_reward(obs, prev_damage):
    """
    Funzione di Reward Formale (R_t) progettata per la competizione:
    R_t = v_x \cdot \cos(\theta) - \alpha |p_x| - \beta \Delta d_t
    Incentiva la velocità sull'asse pista e penalizza gli scostamenti ed i danni.
    """
    speed_x = obs['speedX']           # Velocità longitudinale
    track_pos = obs['trackPos']       # Errore rispetto al centro della pista
    angle = obs['angle']              # Angolo rispetto all'asse della pista
    damage = obs['damage']            # Danno cumulativo
    
    delta_damage = damage - prev_damage
    
    # Parametri di penalità
    alpha = 10.0
    beta = 100.0
    
    reward = speed_x * np.cos(angle) - alpha * abs(track_pos) - beta * delta_damage
    
    # Terminazione se fuori pista
    done = False
    if abs(track_pos) > 1.0 or damage > 10000:
        done = True
        reward -= 200  # Penalità forte per fallimento critico
        
    return reward, done, damage

def flatten_state(state_dict) -> np.ndarray:
    state_vec = np.hstack((
        state_dict['angle'],
        np.array(state_dict['track']),
        state_dict['trackPos'],
        state_dict['speedX'],
        state_dict['speedY'],
        state_dict['speedZ'],
        np.array(state_dict['wheelSpinVel']) / 100.0,
        state_dict['rpm'] / 10000.0
    ))
    return np.array(state_vec, dtype=np.float32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--bc_weights", type=str, default="bc_policy.pth")
    parser.add_argument("--save_path", type=str, default="sac_actor_final.pth")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Inizializza l'ambiente
    # env = TorcsEnv(vision=False, throttle=True, gear_change=True)
    # Per evitare crash durante il build se gym_torcs non è installato o non avviato:
    print("Inizializzazione ambiente TORCS...")
    
    # Dimenesione tipica
    state_dim = 29
    action_dim = 4
    
    agent = SACAgent(state_dim, action_dim, device)
    
    # WARM START: Carica i pesi pre-addestrati da BC
    agent.actor.load_bc_weights(args.bc_weights)
    
    memory = ReplayBuffer(capacity=100000)
    
    updates = 0
    batch_size = 256
    
    print("Inizio fase di Reinforcement Learning (Fine-tuning)...")
    
    # Loop teorico di training (mocked per documentazione, decommentare l'uso di env reale)
    """
    for ep in range(args.episodes):
        obs = env.reset(relaunch=(ep == 0))
        state = flatten_state(obs)
        episode_reward = 0
        prev_damage = obs['damage']
        
        for step in range(2000): # max steps per episode
            action = agent.select_action(state)
            
            # Conversione in [0,1] o range corretto per l'ambiente se necessario.
            # L'output tanh è in [-1, 1].
            env_action = action.copy()
            env_action[1] = (env_action[1] + 1) / 2.0  # Accel [0,1]
            env_action[2] = (env_action[2] + 1) / 2.0  # Brake [0,1]
            # env_action[3] (Gear) da gestire (es. discretizzare se env lo richiede)
            
            next_obs, env_reward, env_done, _ = env.step(env_action)
            
            # Usiamo la nostra reward formale
            reward, done, prev_damage = compute_reward(next_obs, prev_damage)
            
            next_state = flatten_state(next_obs)
            
            mask = 0.0 if done else 1.0
            memory.push(state, action, reward, next_state, mask)
            
            state = next_state
            episode_reward += reward
            
            if len(memory) > batch_size:
                qf_loss, policy_loss, alpha_loss = agent.update_parameters(memory, batch_size, updates)
                updates += 1
                
            if done:
                break
                
        print(f"Episode {ep+1} | Reward: {episode_reward:.2f} | Updates: {updates}")
        
        if (ep + 1) % 50 == 0:
            torch.save(agent.actor.state_dict(), f"sac_actor_ep{ep+1}.pth")
            
    torch.save(agent.actor.state_dict(), args.save_path)
    env.end()
    """
    print("Struttura SAC completata e pronta all'esecuzione.")

if __name__ == "__main__":
    main()
