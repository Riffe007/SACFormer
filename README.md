# SACFormer
Soft Actor-Critic (SAC) with Transformers in Gymnasium & MuJoCo

📌 Overview

This project implements a modern Soft Actor-Critic (SAC) reinforcement learning agent powered by Transformers instead of traditional MLPs, running on MuJoCo environments (Gymnasium). It is optimized for multi-GPU training, efficient experience replay, and parallelized execution.

🚀 Features

✅ SAC-Based RL – Uses Soft Actor-Critic for optimal policy learning.✅ Transformer-Based Actor – Replaces MLP with self-attention for better sequential decision-making.✅ MuJoCo + Gymnasium – High-quality physics simulations for better training.✅ Multi-GPU Training – Supports distributed learning with Ray RLlib.✅ Vectorized Environments – Stable-Baselines3 VecEnv for faster rollouts.✅ Prioritized Experience Replay (PER) – Smart sampling for efficient learning.✅ TensorBoard & WandB Integration – Real-time monitoring & hyperparameter tuning.

📂 Directory Structure
```plaintext
SAC-Transformer-RL/
│── agents/                     # SAC Agent & Models
│   ├── policy.py                # Transformer-based Actor-Critic
│   ├── sac_agent.py             # SAC Trainer
│   ├── q_network.py             # Twin Q-Networks
│   ├── replay_buffer.py         # PER Experience Replay
│   ├── train.py                 # Training Script
│   ├── evaluation.py            # Evaluation Script
│
│── environments/                # Environment Setup
│   ├── gym_env.py               # Gymnasium Wrapper
│   ├── vec_env.py               # Parallelized VecEnv
│   ├── mujoco_envs.py           # MuJoCo Integration
│
│── utils/                       # Helper Functions
│   ├── logging.py               # TensorBoard & WandB Integration
│   ├── config.py                # Hyperparameter Storage
│   ├── utils.py                 # Common Utilities
│
│── training_scripts/            # Different Training Variants
│   ├── sac_train.py             # Standard SAC Training
│   ├── sac_ray_train.py         # Multi-GPU Training with Ray
│   ├── sac_hyperparam_search.py # Auto Hyperparameter Tuning
│
│── logs/                        # Training Logs
│
│── models/                      # Saved Models
│   ├── sac_halfcheetah.pth      # SAC Weights
│
│── README.md                    # Documentation
│── requirements.txt              # Dependencies
│── train.py                      # Main Entry Point
```
💾 Installation

pip install torch stable-baselines3[extra] gymnasium mujoco ray rllib numpy wandb

🛠️ Usage

Train SAC with Transformers on HalfCheetah-v4:

python training_scripts/sac_train.py

Run parallel training with Ray RLlib:

python training_scripts/sac_ray_train.py

Perform automated hyperparameter tuning:

python training_scripts/sac_hyperparam_search.py

🔬 Improvements Over Old Repo

Old ARS Project

New SAC-Transformer Project

Augmented Random Search (ARS)

✅ Soft Actor-Critic (SAC)

PyBullet Environments

✅ MuJoCo + Gymnasium

No GPU Support

✅ Multi-GPU (Ray RLlib)

Manual Policy Updates (NumPy)

✅ Transformer-Based Actor-Critic

No Parallelization

✅ Vectorized Environments (VecEnv)

Basic Replay Buffer

✅ Prioritized Experience Replay (PER)

Minimal Logging

✅ TensorBoard + WandB

🔮 Future Plans

🔹 Add Meta-RL support (Memory-Augmented Networks).🔹 Experiment with GATO / Decision Transformer.🔹 Optimize JAX version for TPU acceleration.

💡 Why This is the Best Modern RL Setup

✅ State-of-the-Art RL (SAC + Transformers)✅ High-Quality Physics (MuJoCo)✅ Parallel Training (Ray RLlib)✅ Production-Ready Code

🔥 This framework is built for RL research and deployment. 🚀

