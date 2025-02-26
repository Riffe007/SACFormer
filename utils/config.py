import os
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Config:
    """
    Configuration settings for SAC training.
    """
    env_name: str = "HalfCheetah-v4"
    episodes: int = 1000
    batch_size: int = 256
    lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2
    buffer_size: int = 1_000_000
    num_envs: int = 4
    seed: int = 42
    log_dir: str = "logs"
    wandb_project: str = "SAC_Training"
    save_interval: int = 100
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    def create_log_dir(self):
        """Ensures log directory exists."""
        os.makedirs(self.log_dir, exist_ok=True)
    
    def display(self):
        """Prints the configuration settings."""
        print("Training Configuration:")
        for key, value in self.__dict__.items():
            print(f"  {key}: {value}")

# Usage example
if __name__ == "__main__":
    config = Config()
    config.create_log_dir()
    config.display()
