# 🚀 SACFormer: Soft Actor-Critic with Transformers in MuJoCo & Gymnasium
A modern Reinforcement Learning (RL) framework using Soft Actor-Critic (SAC) with Transformer-based policies for MuJoCo physics simulations. Designed for multi-GPU training, efficient experience replay, and scalable parallel execution.

# 📌 Features
    ✅ SAC-Based RL – Uses Soft Actor-Critic for optimal policy learning.
    ✅ Transformer-Based Actor – Replaces MLP with self-attention for better sequential decision-making.
    ✅ MuJoCo + Gymnasium – High-quality physics simulations for realistic training.
    ✅ Multi-GPU Training – Supports distributed RL training via Ray RLlib.
    ✅ Vectorized Environments – Uses Stable-Baselines3 VecEnv for fast rollouts.
    ✅ Prioritized Experience Replay (PER) – Smart sampling for faster, efficient learning.
    ✅ TensorBoard & Weights & Biases (WandB) Integration – Real-time monitoring & logging.

# 💾 Installation Guide
Follow these steps to clone, install dependencies, and run the project.

## Step 1: Clone the Repository
```bash
git clone https://github.com/Riffe007/SACFormer.git
cd SACFormer
```
## Step 2: Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv sac-env
source sac-env/bin/activate  # MacOS/Linux
sac-env\Scripts\activate     # Windows
```
## Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```
Alternatively, install core dependencies manually:

```bash
pip install torch stable-baselines3[extra] gymnasium mujoco ray rllib numpy wandb
```
## Step 4: Verify Installation
Check that dependencies installed correctly:

```bash
python -c "import torch; import gymnasium; import mujoco; import stable_baselines3; print('✅ Installation successful!')"
```
# 🛠️ Usage: Running the Training
Train SAC with Transformers on HalfCheetah
```bash
python training_scripts/sac_train.py
```
Run Parallel Training with Ray RLlib
```bash
python training_scripts/sac_ray_train.py
```
Perform Automated Hyperparameter Tuning
```bash
python training_scripts/sac_hyperparam_search.py
```
# 📂 Directory Structure

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
│── training_scripts/            # Training Variants
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

```markdown
## 🔬 Improvements Over Previous Implementations

| **Old ARS Project**             | **New SAC-Transformer Project**         |
|----------------------------------|-----------------------------------------|
| Augmented Random Search (ARS)   | ✅ Soft Actor-Critic (SAC)              |
| PyBullet Environments           | ✅ MuJoCo + Gymnasium                   |
| No GPU Support                  | ✅ Multi-GPU (Ray RLlib)                |
| Manual Policy Updates (NumPy)   | ✅ Transformer-Based Actor-Critic       |
| No Parallelization              | ✅ Vectorized Environments (VecEnv)     |
| Basic Replay Buffer             | ✅ Prioritized Experience Replay (PER)  |
| Minimal Logging                 | ✅ TensorBoard + WandB                  |
```
## 🔮 Future Plans
## 🚀 Upcoming Enhancements:

    🔹 Meta-RL Support (Memory-Augmented Networks).
    🔹 GATO & Decision Transformer experiments.
    🔹 Optimized JAX version for TPU acceleration.
    🔹 Multi-Agent RL support.
# 💡 Why This is the Best Modern RL Setup
    ✅ State-of-the-Art RL (SAC + Transformers)
    ✅ High-Quality Physics (MuJoCo)
    ✅ Parallel Training (Ray RLlib)
    ✅ Production-Ready Code

# 🔥 This framework is built for RL research and real-world deployment. 🚀

# 🤝 Contributing
Contributions are welcome! If you’d like to improve the repo:

1. Fork the project
2. Create a new branch
3. Commit your changes
4. Push to your branch and submit a PR
# 📜 License
This project is licensed under the MIT License.

# 📩 Contact & Support
For issues, open a GitHub issue or reach out via email:
✉️ techavenger83@gmail.com

