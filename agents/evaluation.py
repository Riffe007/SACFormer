import gymnasium as gym
import torch
import numpy as np
from SACFormer.agents.sac_agent import SACAgent
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))



def evaluate_sac(env_name="HalfCheetah-v4", model_path="models/sac_halfcheetah.pth", episodes=10):
    """
    Evaluate a trained SAC agent in the given environment.

    Args:
        env_name (str): Name of the Gymnasium environment.
        model_path (str): Path to the trained model.
        episodes (int): Number of episodes to evaluate.
    """
    
    # Initialize environment
    env = gym.make(env_name, render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Load the trained agent
    agent = SACAgent(state_dim, action_dim)
    agent.actor.load_state_dict(torch.load(model_path, map_location=agent.device))
    agent.actor.eval()
    
    total_rewards = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state, deterministic=True)
            next_state, reward, done, _, _ = env.step(action)
            state = next_state
            total_reward += reward
            
        total_rewards.append(total_reward)
        print(f"Episode {episode + 1}: Reward = {total_reward}")
    
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    print(f"\nAverage Reward over {episodes} episodes: {avg_reward:.2f} ± {std_reward:.2f}")
    
    env.close()
    
    return avg_reward, std_reward


if __name__ == "__main__":
    evaluate_sac()
