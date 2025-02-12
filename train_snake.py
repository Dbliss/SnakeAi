import os
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

# -----------------------------------
# Custom CNN for (7, 10, 10) input
# -----------------------------------
class CustomCNN(BaseFeaturesExtractor):
    """
    A custom CNN for the 7-channel snake environment:
      Channels: snake, food, dir_up, dir_down, dir_left, dir_right, danger
    The input shape is (7, 10, 10).
    We apply simple conv layers and flatten.
    """
    def __init__(self, observation_space, features_dim=1152):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]  # 7
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        # 10x10 -> after two conv layers (kernel=3, stride=1), final size is (10-2=8), then (8-2=6).
        # shape is (32, 6, 6) => 32*6*6 = 1152
        self._features_dim = 32 * 6 * 6

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.cnn(observations)

# -----------------------------------
# Simple Callback for plotting
# -----------------------------------
class PlottingCallback(BaseCallback):
    """
    A callback that collects episode rewards and lengths throughout training
    and plots them only once at the very end (_on_training_end).
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        """
        Collect episode info from the "infos" list, which VecMonitor adds to the rollout.
        """
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                # The "episode" dict has 'r' (reward) and 'l' (length).
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])

    def _on_training_end(self) -> None:
        """
        Called once at the end of training: show the reward/length plots.
        """
        plt.figure(figsize=(12, 5))

        # Plot rewards
        plt.subplot(1, 2, 1)
        plt.plot(self.episode_rewards, label="Episode Reward")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Episode Rewards Over Time")
        plt.legend()

        # Plot episode lengths
        plt.subplot(1, 2, 2)
        plt.plot(self.episode_lengths, label="Episode Length", color="orange")
        plt.xlabel("Episode")
        plt.ylabel("Length")
        plt.title("Episode Length Over Time")
        plt.legend()

        plt.tight_layout()
        plt.show()

# -----------------------------------
# Make a parallel environment function
# -----------------------------------
def make_env():
    def _init():
        env = SnakeEnv(grid_size=10)
        return env
    return _init

def train_snake():
    models_dir = "models/snake_ppo"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # 1. Create multiple environments (8 in this example)
    n_envs = 8
    env_fns = [make_env() for _ in range(n_envs)]
    # First wrap with DummyVecEnv
    raw_vec_env = DummyVecEnv(env_fns)
    # Then wrap with VecMonitor to track episode rewards and lengths in logs
    vec_env = VecMonitor(raw_vec_env)

    # 2. Check a single env instance
    check_env(env_fns[0](), warn=True)

    # 3. Look for saved models
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

    # 4. Create or load PPO model
    if best_model_path is not None:
        print(f"Resuming training from {best_model_path}.zip with {max_steps} timesteps.")
        model = PPO.load(best_model_path, env=vec_env, device="cuda")
        model.num_timesteps = max_steps
    else:
        print("No valid saved model found. Starting a new model.")
        model = PPO(
            "CnnPolicy",
            vec_env,
            verbose=1,
            device="cuda",
            n_steps=512,  # Lower rollout length for small environment
            policy_kwargs={
                'normalize_images': False,
                'features_extractor_class': CustomCNN,
                'features_extractor_kwargs': {'features_dim': 1152}
            }
        )

    # 5. Train for a single run
    total_timesteps = 500_000
    callback = PlottingCallback()

    print(f"Training for {total_timesteps} timesteps with {n_envs} parallel envs, n_steps=512...")
    model.learn(
        total_timesteps=total_timesteps,
        reset_num_timesteps=False,
        callback=callback
    )

    # 6. Save final model
    new_model_path = os.path.join(models_dir, f"ppo_snake_{round(model.num_timesteps, -4)}_steps")
    model.save(new_model_path)
    print(f"Model saved: {new_model_path}")
    print("Training complete!")

if __name__ == "__main__":
    train_snake()
