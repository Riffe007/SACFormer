import gymnasium as gym
import mujoco
import numpy as np
from typing import Optional

class MujocoEnvWrapper:
    """
    Advanced wrapper for MuJoCo environments with enhanced performance and utility functions.
    """
    def __init__(self, env_name: str, seed: Optional[int] = None, render_mode: Optional[str] = None):
        """
        Initializes a MuJoCo environment.
        
        Args:
            env_name (str): Name of the MuJoCo environment.
            seed (Optional[int]): Random seed for reproducibility.
            render_mode (Optional[str]): Rendering mode, e.g., "human" for visualization.
        """
        self.env = gym.make(env_name, render_mode=render_mode)
        self.seed = seed
        if seed is not None:
            self.env.reset(seed=seed)
            self.env.action_space.seed(seed)  # ✅ Action spaces still support .seed()


    def reset(self):
        """Resets the environment and returns the initial observation."""
        return self.env.reset()
    
    def step(self, action: np.ndarray):
        """
        Takes an action in the environment.
        
        Args:
            action (np.ndarray): Action to be executed.
        
        Returns:
            Tuple[np.ndarray, float, bool, dict]: Observation, reward, done flag, and additional info.
        """
        return self.env.step(action)
    
    def render(self, mode: str = "human"):
        """Renders the environment."""
        return self.env.render(mode=mode)
    
    def close(self):
        """Closes the environment."""
        self.env.close()


def make_mujoco_env(env_name: str, seed: Optional[int] = None, render_mode: Optional[str] = None) -> MujocoEnvWrapper:
    """
    Factory function to create and initialize a MuJoCo environment.
    
    Args:
        env_name (str): Name of the MuJoCo environment.
        seed (Optional[int]): Random seed for reproducibility.
        render_mode (Optional[str]): Rendering mode.
    
    Returns:
        MujocoEnvWrapper: Wrapped MuJoCo environment.
    """
    return MujocoEnvWrapper(env_name, seed, render_mode)

if __name__ == "__main__":
    env = make_mujoco_env("HalfCheetah-v4", seed=42, render_mode="human")
    state, _ = env.reset()
    print("MuJoCo environment initialized with state shape:", state.shape)
    env.close()