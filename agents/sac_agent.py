import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from agents.policy import TransformerPolicy
from agents.q_network import TwinQNetwork
from agents.replay_buffer import ReplayBuffer

class SACAgent:
    """
    Soft Actor-Critic (SAC) Agent with Transformer-based Policy and Twin Q-Networks.
    """

    def __init__(
        self, state_dim, action_dim, hidden_dim=256, gamma=0.99, tau=0.005, 
        alpha=0.2, lr=3e-4, buffer_size=1_000_000, batch_size=256, max_grad_norm=1.0
    ):
        """
        Initializes a SAC Agent.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm

        # Policy (Actor) and Twin Q-Networks (Critic)
        self.actor = TransformerPolicy(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic = TwinQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = TwinQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=lr)

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(buffer_size, state_dim, action_dim)

    def select_action(self, state, deterministic=False):
        # If we get (obs, info), drop info
        if isinstance(state, tuple):
            state = state[0]
        
        state = np.asarray(state, dtype=np.float32)
        # Now let's handle shape logic
        if state.ndim == 1:
            # single environment, shape (state_dim,)
            # Make it (1, state_dim)
            state = torch.tensor(state, device=self.device).unsqueeze(0)
        elif state.ndim == 2:
            # multiple envs parallel, shape (batch_size, state_dim)
            state = torch.tensor(state, device=self.device)
        elif state.ndim == 3:
            # Maybe it's already (batch_size, 1, state_dim)?
            state = torch.tensor(state, device=self.device)
        else:
            raise ValueError(f"Unexpected state shape: {state.shape}")

        print(f"⚡ shape prior to policy forward: {state.shape}")
        # Now pass directly to policy
        with torch.no_grad():
            action = self.actor(state, deterministic).cpu().numpy()

        # If batch_size=1 => (1, act_dim) => squeeze to (act_dim,)
        # If batch_size>1 => (batch_size, act_dim)
        if action.shape[0] == 1:
            action = action.squeeze(0)
        return action


    def update(self):
        """
        Performs a single SAC update step on the actor and critic networks.
        """
        if len(self.replay_buffer) < self.batch_size:
            return  

        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Ensure inputs to Transformer are (batch_size, seq_len=1, state_dim)
        states_seq = states.unsqueeze(1)
        next_states_seq = next_states.unsqueeze(1)

        # Compute target Q-value using the target critic network
        with torch.no_grad():
            next_actions = self.actor(next_states_seq, deterministic=False).squeeze(1)  # (batch_size, action_dim)
            next_q1, next_q2 = self.critic_target(next_states.squeeze(1), next_actions)
            next_q = torch.min(next_q1, next_q2)  
            target_q = rewards + (1 - dones) * self.gamma * (next_q - self.alpha)

        # Compute current Q-values using the critic network
        q1, q2 = self.critic(states.squeeze(1), actions)
        critic_loss = ((q1 - target_q).pow(2) + (q2 - target_q).pow(2)).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        # Actor update:
        # Get actor actions using states_seq
        actor_actions = self.actor(states_seq, deterministic=False).squeeze(1)  # (batch_size, action_dim)
        # Concatenate state and action along the last dimension
        q_input = torch.cat([states.squeeze(1), actor_actions], dim=-1)
        policy_loss = -self.critic.q1(q_input).mean()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        # Soft update the target critic network
        self._soft_update(self.critic, self.critic_target)


    def _soft_update(self, source_model, target_model):
        with torch.no_grad():
            for target_param, param in zip(target_model.parameters(), source_model.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
