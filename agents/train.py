import argparse
import os

import numpy as np
import torch
import gymnasium as gym
import wandb

from agents.sac_agent import SACAgent
from environments.vec_env import make_vec_env
from environments.mujoco_envs import MujocoEnvWrapper

def main():
    parser = argparse.ArgumentParser(description="Train SAC agent in a MuJoCo environment.")
    parser.add_argument("--env", type=str, default="HalfCheetah-v5", help="MuJoCo environment name.")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--tau", type=float, default=0.005, help="Target update rate.")
    parser.add_argument("--alpha", type=float, default=0.2, help="Entropy coefficient.")
    parser.add_argument("--buffer_size", type=int, default=1_000_000, help="Replay buffer size.")
    parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel environments.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory for logs and models.")
    parser.add_argument("--wandb_project", type=str, default="SAC_Training", help="W&B project name.")
    parser.add_argument("--save_interval", type=int, default=100, help="Save model every N episodes.")
    args = parser.parse_args()

    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.log_dir, exist_ok=True)
    wandb.init(project=args.wandb_project, config=args, name=f"SAC_{args.env}")

    # Create parallel environment
    env = make_vec_env(args.env, num_envs=args.num_envs, seed=args.seed)
    states, _ = env.reset()  # Unpack Gymnasium reset output to get observations only
    state_dim = states.shape[1]  # e.g., (num_envs, state_dim) → state_dim = 17
    action_dim = env.envs.single_action_space.shape[0]

    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=256,
        gamma=args.gamma,
        tau=args.tau,
        alpha=args.alpha,
        lr=args.lr,
        buffer_size=args.buffer_size
    )
    print(f"Initialized SACAgent with state_dim={state_dim}, action_dim={action_dim}")

    # ----------------- Training Loop -----------------
    for episode in range(args.episodes):
        states, _ = env.reset()
        total_rewards = np.zeros(args.num_envs, dtype=np.float32)
        done_flags = np.zeros(args.num_envs, dtype=bool)

        while not done_flags.all():
            actions = agent.select_action(states, deterministic=False)
            next_states, rewards, dones, truncated, info = env.step(actions)
            combined_done = np.logical_or(dones, truncated)

            # Store each env's transition individually to match replay buffer expectations.
            for i in range(args.num_envs):
                agent.replay_buffer.add(
                    states[i],
                    actions[i],
                    rewards[i],
                    next_states[i],
                    combined_done[i]
                )

            states = next_states
            total_rewards += rewards
            done_flags |= combined_done

            # Now call update without passing batch_size; the agent handles it internally.
            if len(agent.replay_buffer.states) > args.batch_size:
                agent.update()

        avg_reward = total_rewards.mean()
        wandb.log({"Episode Reward": avg_reward})
        print(f"Episode {episode + 1}/{args.episodes} - Avg Reward: {avg_reward:.2f}")

        if (episode + 1) % args.save_interval == 0:
            model_path = os.path.join(args.log_dir, f"sac_{args.env}_ep{episode+1}.pth")
            torch.save(agent.actor.state_dict(), model_path)
            print(f"[Checkpoint] Saved model at {model_path}")

    env.close()
    wandb.finish()
    print("Training completed successfully! Now go take a break—your agent already did all the hard work.")

if __name__ == "__main__":
    main()
