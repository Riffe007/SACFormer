import ray
from ray import tune
from ray.rllib.algorithms.sac import SACConfig
import gymnasium as gym
import argparse


def train_sac_ray(env_name: str, num_workers: int, num_gpus: int, total_timesteps: int):
    """
    Train SAC using Ray RLlib with distributed training capabilities.
    
    Args:
        env_name (str): Name of the Gymnasium environment.
        num_workers (int): Number of parallel workers for environment simulation.
        num_gpus (int): Number of GPUs to use.
        total_timesteps (int): Total training timesteps.
    """
    ray.init(ignore_reinit_error=True)
    
    config = (
        SACConfig()
        .environment(env=env_name)
        .framework("torch")
        .training(gamma=0.99, lr=3e-4, tau=0.005, train_batch_size=256)
        .resources(num_gpus=num_gpus)
        .rollouts(num_rollout_workers=num_workers)
        .debugging(log_level="INFO")
    )
    
    results = tune.run(
        "SAC",
        config=config.to_dict(),
        stop={"timesteps_total": total_timesteps},
        checkpoint_at_end=True,
    )
    
    ray.shutdown()
    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAC with Ray RLlib.")
    parser.add_argument("--env", type=str, default="HalfCheetah-v4", help="MuJoCo environment name.")
    parser.add_argument("--workers", type=int, default=4, help="Number of environment workers.")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use.")
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="Total training timesteps.")
    
    args = parser.parse_args()
    train_sac_ray(args.env, args.workers, args.gpus, args.timesteps)
