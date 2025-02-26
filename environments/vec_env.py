import gymnasium as gym
import numpy as np
from gymnasium.vector import SyncVectorEnv
from typing import Callable, List

class VecEnvWrapper:
    """
    Advanced wrapper for vectorized environments to improve performance and usability.
    """
    def __init__(self, env_fns: List[Callable], num_envs: int = 4):
        """
        Initializes a synchronized vectorized environment.

        Args:
            env_fns (List[Callable]): List of functions to create environments.
            num_envs (int): Number of parallel environments to run.
        """
        self.num_envs = num_envs
        self.envs = SyncVectorEnv(env_fns)
    
    def reset(self):
        """Resets all environments and returns initial observations."""
        return self.envs.reset()
    
    def step(self, actions):
        """
        Steps through all environments with given actions.
        
        Args:
            actions (np.ndarray): Actions to take in each environment.
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, dict]: Observations, rewards, dones, and additional info.
        """
        return self.envs.step(actions)
    
    def close(self):
        """Closes all environments."""
        self.envs.close()

    def render(self, mode="human"):
        """Renders the first environment."""
        return self.envs.envs[0].render(mode=mode)


def make_vec_env(env_name: str, num_envs: int = 4, seed: int = None) -> VecEnvWrapper:
    """
    Creates a vectorized environment.
    
    Args:
        env_name (str): Name of the Gymnasium environment.
        num_envs (int): Number of parallel environments.
        seed (int, optional): Random seed for reproducibility.
    
    Returns:
        VecEnvWrapper: Wrapped vectorized environment.
    """
    env_fns = [lambda: gym.make(env_name) for _ in range(num_envs)]
    vec_env = VecEnvWrapper(env_fns, num_envs)
    if seed is not None:
        for i, env in enumerate(vec_env.envs.envs):
            env.reset(seed=seed + i)
            env.action_space.seed(seed + i)  # ✅ Only action spaces support .seed()
            
    return vec_env

if __name__ == "__main__":
    vec_env = make_vec_env("HalfCheetah-v4", num_envs=4, seed=42)
    states = vec_env.reset()
    print("Vectorized environment initialized with state shape:", states.shape)
    vec_env.close()
