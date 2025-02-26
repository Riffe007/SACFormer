import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.rllib.algorithms.sac import SACConfig
import argparse

def hyperparam_search_sac(env_name: str, num_samples: int, num_workers: int, num_gpus: int, max_timesteps: int):
    """
    Perform hyperparameter tuning for SAC using Ray Tune.
    
    Args:
        env_name (str): Name of the Gymnasium environment.
        num_samples (int): Number of hyperparameter configurations to sample.
        num_workers (int): Number of parallel workers for environment simulation.
        num_gpus (int): Number of GPUs to use.
        max_timesteps (int): Maximum training timesteps.
    """
    ray.init(ignore_reinit_error=True)
    
    config = (
        SACConfig()
        .environment(env=env_name)
        .framework("torch")
        .training(
            gamma=tune.grid_search([0.98, 0.99, 0.995]),
            lr=tune.loguniform(1e-5, 3e-3),
            tau=tune.grid_search([0.005, 0.01]),
            train_batch_size=tune.choice([128, 256, 512])
        )
        .resources(num_gpus=num_gpus)
        .rollouts(num_rollout_workers=num_workers)
        .debugging(log_level="INFO")
    )
    
    scheduler = ASHAScheduler(
        metric="episode_reward_mean",
        mode="max",
        max_t=max_timesteps,
        grace_period=10000,
        reduction_factor=2
    )
    
    tune.run(
        "SAC",
        config=config.to_dict(),
        num_samples=num_samples,
        scheduler=scheduler,
        stop={"timesteps_total": max_timesteps},
        checkpoint_at_end=True,
    )
    
    ray.shutdown()
    print("Hyperparameter search complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter search for SAC with Ray Tune.")
    parser.add_argument("--env", type=str, default="HalfCheetah-v4", help="MuJoCo environment name.")
    parser.add_argument("--samples", type=int, default=10, help="Number of hyperparameter configurations to sample.")
    parser.add_argument("--workers", type=int, default=4, help="Number of environment workers.")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use.")
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="Maximum training timesteps.")
    
    args = parser.parse_args()
    hyperparam_search_sac(args.env, args.samples, args.workers, args.gpus, args.timesteps)