import os
import numpy as np
import torch
import random

class Utils:
    """
    Utility class for handling various helper functions in SAC training.
    """
    @staticmethod
    def set_seed(seed: int):
        """Sets the random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    @staticmethod
    def save_model(model: torch.nn.Module, filepath: str):
        """Saves a PyTorch model to the specified file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(model.state_dict(), filepath)

    @staticmethod
    def load_model(model: torch.nn.Module, filepath: str, device: str = "cpu"):
        """Loads a PyTorch model from the specified file."""
        model.load_state_dict(torch.load(filepath, map_location=device))
        model.eval()
        return model

    @staticmethod
    def compute_discounted_returns(rewards: list, gamma: float):
        """Computes discounted returns for a sequence of rewards."""
        discounted_returns = []
        g = 0
        for reward in reversed(rewards):
            g = reward + gamma * g
            discounted_returns.insert(0, g)
        return discounted_returns

# Usage example
if __name__ == "__main__":
    Utils.set_seed(42)
    print("Random seed set for reproducibility.")
