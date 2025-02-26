import argparse
import ray
from ray import tune
from ray.rllib.algorithms.sac import SACConfig

def hyperparam_search_sac(env, samples, workers, gpus, timesteps):
    config = (
        SACConfig()
        .environment(env=env)
        .env_runners(num_envs_per_env_runner=workers)  # Updated parameter name.
        .resources(num_gpus=gpus)
        .training(train_batch_size=256)  # Adjust as needed.
    )
    
    analysis = tune.run(
        "SAC",
        config=config.to_dict(),
        stop={"timesteps_total": timesteps},
        metric="episode_reward_mean",
        mode="max",
        num_samples=samples,
    )
    print("Best config:", analysis.get_best_config(metric="episode_reward_mean", mode="max"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="HalfCheetah-v5", help="Environment name")
    parser.add_argument("--samples", type=int, default=10, help="Number of hyperparameter samples")
    parser.add_argument("--workers", type=int, default=4, help="Number of environments per env runner")
    parser.add_argument("--gpus", type=int, default=0, help="Number of GPUs to allocate")
    parser.add_argument("--timesteps", type=int, default=1000000, help="Total timesteps for training")
    args = parser.parse_args()
    
    ray.init()
    hyperparam_search_sac(args.env, args.samples, args.workers, args.gpus, args.timesteps)
