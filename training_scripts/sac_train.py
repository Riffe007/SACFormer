import gymnasium as gym
import torch
import numpy as np
from SACFormer.agents.sac_agent import SACAgent

def train_sac(env_name="HalfCheetah-v4", episodes=1000, batch_size=256):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = SACAgent(state_dim, action_dim)
    returns = []

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0

        for _ in range(1000):  # Max episode length
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)

            agent.replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(agent.replay_buffer.states) > batch_size:
                agent.update(batch_size)

            if done:
                break

        returns.append(total_reward)
        print(f"Episode {episode}: Reward {total_reward}")

if __name__ == "__main__":
    train_sac()
