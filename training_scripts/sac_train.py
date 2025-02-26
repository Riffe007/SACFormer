import os
import sys
import argparse
import gymnasium as gym

# Add the parent directory to sys.path so we can import our agents
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.sac_agent import SACAgent  # Adjusted import
# (Import other necessary modules here, e.g., torch if needed)

def record_episode(agent, env_name, video_dir, seed=42):
    """
    Records a single episode using Gymnasium's RecordVideo wrapper.
    """
    from gymnasium.wrappers import RecordVideo

    os.makedirs(video_dir, exist_ok=True)
    # Create the environment with render_mode="rgb_array"
    env = gym.make(env_name, render_mode="rgb_array")
    env = RecordVideo(env, video_folder=video_dir, name_prefix="episode")
    state, info = env.reset(seed=seed)
    done = False
    total_reward = 0
    while not done:
        # Use deterministic actions for a stable video.
        action = agent.select_action(state, deterministic=True)
        state, reward, done, truncated, info = env.step(action)
        total_reward += reward
        done = done or truncated
    env.close()
    print(f"Recorded episode video saved in {video_dir}. Total reward: {total_reward}")


def main():
    parser = argparse.ArgumentParser(description="Train SAC agent and record an episode.")
    parser.add_argument("--env", type=str, default="HalfCheetah-v5", help="Environment name.")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes.")
    parser.add_argument("--video_dir", type=str, default="videos", help="Directory to save recorded videos.")
    # Add other hyperparameters as needed.
    args = parser.parse_args()

    # Initialize your SAC agent (ensure your SACAgent's __init__ signature is correct).
    # This example assumes your agent handles its own replay buffer etc.
    # You may need to adjust parameters as per your implementation.
    agent = SACAgent(
        state_dim=17,  # Example value for HalfCheetah
        action_dim=6,
        hidden_dim=256,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        lr=3e-4,
        buffer_size=1_000_000,
        batch_size=256
    )
    print("Initialized SACAgent.")

    # Dummy training loop (replace with your actual training loop)
    for ep in range(args.episodes):
        # Assume your training loop runs here, updates the agent, etc.
        # For example:
        print(f"Training episode {ep+1}/{args.episodes}...")
        # agent.train_one_episode()  <-- your training step
        # (You can log rewards, losses, etc.)
    
    # After training, record an evaluation episode
    print("Recording evaluation episode...")
    record_episode(agent, args.env, args.video_dir)

if __name__ == "__main__":
    main()
