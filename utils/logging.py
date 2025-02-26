import os
import logging
import wandb

class Logger:
    """
    Advanced logging utility for SAC training with support for console logging and Weights & Biases.
    """
    def __init__(self, log_dir: str = "logs", wandb_project: str = "SAC_Training", use_wandb: bool = True):
        self.log_dir = log_dir
        self.use_wandb = use_wandb
        
        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)
        
        # Set up logging
        logging.basicConfig(
            filename=os.path.join(log_dir, "training.log"),
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # Initialize wandb if required
        if use_wandb:
            wandb.init(project=wandb_project, dir=log_dir)
    
    def log(self, message: str):
        """Logs a message to both console and file."""
        print(message)
        logging.info(message)
        if self.use_wandb:
            wandb.log({"info": message})
    
    def log_metric(self, metric_name: str, value: float, step: int = None):
        """Logs a metric value to wandb."""
        logging.info(f"{metric_name}: {value}")
        if self.use_wandb:
            wandb.log({metric_name: value}, step=step)
    
    def close(self):
        """Closes wandb logging if enabled."""
        if self.use_wandb:
            wandb.finish()

# Usage example
if __name__ == "__main__":
    logger = Logger()
    logger.log("Training started.")
    logger.log_metric("Episode Reward", 100.5)
    logger.close()
