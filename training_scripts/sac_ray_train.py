import argparse
import ray
from ray.rllib.algorithms.sac import SACConfig

def train_sac_ray(env, workers, gpus, timesteps):
    config = (
        SACConfig()
        .environment(env=env)
        .env_runners(num_envs_per_env_runner=workers)
        .resources(num_gpus=gpus)
        .training(train_batch_size=256)
        .evaluation(
            evaluation_interval=1,
            evaluation_duration=1,
            evaluation_duration_unit="episodes",
            evaluation_config={"explore": False}
        )
    )
    algo = config.build_algo()  # or .build() if you prefer (though build_algo is preferred now)

    for i in range(timesteps):
        result = algo.train()
        # Now, evaluation metrics should be present.
        print(f"Iteration {i}: reward: {result.get('episode_reward_mean', 'N/A')}")
    algo.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="HalfCheetah-v5", help="Environment name")
    parser.add_argument("--workers", type=int, default=4, help="Number of environments per env runner")
    parser.add_argument("--gpus", type=int, default=0, help="Number of GPUs")
    parser.add_argument("--timesteps", type=int, default=1000, help="Total number of training iterations")
    args = parser.parse_args()

    ray.init()
    train_sac_ray(args.env, args.workers, args.gpus, args.timesteps)
