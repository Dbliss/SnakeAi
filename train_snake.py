import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import glob
import re
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import torch as th
import torch.nn as nn

from snake_env import SnakeEnv

# For curriculum learning, we start with a small grid.
GRID_SIZE = 5  # You can later increase this (e.g., to 20) once learning is validated.

# ------------------------------------------
# Define a Residual Block for the CNN.
# ------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return self.relu(out + residual)

# ------------------------------------------
# Custom CNN with Residual Connections and Adaptive Pooling.
# ------------------------------------------
class CustomCNN(BaseFeaturesExtractor):
    """
    A custom CNN for the simplified snake environment.
    Uses residual blocks and adaptive pooling to handle variable grid sizes.
    """
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]  # Should be 6.
        self.initial_conv = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.resblock1 = ResidualBlock(32)
        self.resblock2 = ResidualBlock(32)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))  # Fixed output of (4,4)
        # The flattened output will have 32*4*4 = 512 features.
        self.linear = nn.Linear(32 * 4 * 4, features_dim)
    
    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = self.initial_conv(observations)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.adaptive_pool(x)
        x = th.flatten(x, start_dim=1)
        return self.linear(x)

# ------------------------------------------
# Callback for plotting training progress.
# ------------------------------------------
class PlottingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
    
    def _on_step(self) -> bool:
        return True
    
    def _on_rollout_end(self) -> None:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
    
    def _on_training_end(self) -> None:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.episode_rewards, label="Episode Reward")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Episode Rewards Over Time")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(self.episode_lengths, label="Episode Length", color="orange")
        plt.xlabel("Episode")
        plt.ylabel("Length")
        plt.title("Episode Length Over Time")
        plt.legend()
        plt.tight_layout()
        plt.show()

# ------------------------------------------
# Environment creation.
# ------------------------------------------
def make_env():
    def _init():
        env = SnakeEnv(grid_size=GRID_SIZE)
        return env
    return _init

def train_snake():
    models_dir = "models/snake_ppo"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    n_envs = 8
    env_fns = [make_env() for _ in range(n_envs)]
    raw_vec_env = DummyVecEnv(env_fns)
    vec_env = VecMonitor(raw_vec_env)
    
    # Check environment compliance.
    check_env(env_fns[0](), warn=True)
    
    # Look for saved models.
    model_files = glob.glob(os.path.join(models_dir, "ppo_snake_*_steps.zip"))
    best_model_path = None
    max_steps = 0
    if model_files:
        for path in model_files:
            match = re.search(r"ppo_snake_(\d+)_steps", os.path.basename(path))
            if match:
                steps = int(match.group(1))
                if steps > max_steps:
                    max_steps = steps
                    best_model_path = os.path.join(models_dir, f"ppo_snake_{steps}_steps")
    
    if best_model_path is not None:
        print(f"Resuming training from {best_model_path}.zip with {max_steps} timesteps.")
        model = PPO.load(best_model_path, env=vec_env, device="cuda")
        model.num_timesteps = max_steps
    else:
        print("No valid saved model found. Starting a new model.")
        policy_kwargs = {
            'net_arch': [dict(pi=[512, 256], vf=[512, 256])],
            'features_extractor_class': CustomCNN,
            'features_extractor_kwargs': {'features_dim': 512},
            'normalize_images': False
        }
        model = PPO(
            "CnnPolicy",
            vec_env,
            n_steps=2048,
            batch_size=128,
            n_epochs=10,
            gamma=0.995,
            gae_lambda=0.95,
            ent_coef=0.01,
            learning_rate=3e-4,
            verbose=2,
            device="cuda",
            policy_kwargs=policy_kwargs
        )
    
    total_timesteps = 1_000_000
    callback = PlottingCallback()
    
    print(f"Training for {total_timesteps} timesteps with {n_envs} parallel envs (n_steps=2048)...")
    model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False, callback=callback)
    
    new_model_path = os.path.join(models_dir, f"ppo_snake_{round(model.num_timesteps, -4)}_steps")
    model.save(new_model_path)
    print(f"Model saved: {new_model_path}")
    print("Training complete!")

if __name__ == "__main__":
    train_snake()
