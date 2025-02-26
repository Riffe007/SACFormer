import torch
import os
from agents.sac_agent import SACAgent

def generate_dummy_model(model_dir: str = "models", model_name: str = "sac_halfcheetah.pth"):
    """
    Generates a dummy SAC model and saves it as a PyTorch checkpoint.

    Args:
        model_dir (str): Directory to save the model.
        model_name (str): Name of the model file.
    """
    # Ensure directory exists
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, model_name)
    
    # Define dummy model architecture based on expected dimensions
    state_dim = 17  # HalfCheetah-v4 observation space
    action_dim = 6  # HalfCheetah-v4 action space
    agent = SACAgent(state_dim, action_dim)
    
    # Save only the actor network
    torch.save(agent.actor.state_dict(), model_path)
    
    print(f"Dummy model saved at {model_path}")

if __name__ == "__main__":
    generate_dummy_model()
