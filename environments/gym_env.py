import gymnasium as gym

def make_env(env_name: str, seed: int = None, render_mode: str = None):
    """
    Create and configure a Gymnasium environment.
    
    Args:
        env_name (str): Name of the Gymnasium environment.
        seed (int, optional): Random seed for reproducibility.
        render_mode (str, optional): Rendering mode (e.g., "human" for visualization).
    
    Returns:
        gym.Env: Configured Gymnasium environment.
    """
    
    env = gym.make(env_name, render_mode=render_mode)
    
    if seed is not None:
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    
    return env

if __name__ == "__main__":
    env = make_env("HalfCheetah-v4", seed=42, render_mode="human")
    state, _ = env.reset()
    print("Environment initialized with state shape:", state.shape)
    env.close()
