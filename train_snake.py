import os
import glob
import re
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from snake_env import SnakeEnv

# Define a custom callback that collects episode rewards and lengths and plots them.
class PlottingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
    def _on_step(self) -> bool:
        return True
    def _on_rollout_end(self) -> None:
        # Extract episode info from the "infos" list (assuming the Monitor wrapper is used).
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
    def _on_training_end(self) -> None:
        # Plot results at the end of training.
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.episode_rewards, label="Reward")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Episode Reward Over Time")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(self.episode_lengths, label="Episode Length", color="orange")
        plt.xlabel("Episode")
        plt.ylabel("Length")
        plt.title("Episode Length Over Time")
        plt.legend()
        plt.tight_layout()
        plt.show()

def train_snake():
    models_dir = "models/snake_ppo"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # Create a larger environment (grid_size=100) for optimal play.
    env = SnakeEnv(grid_size=100)
    check_env(env, warn=True)

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
            model = PPO.load(best_model_path, env=env)
            model.num_timesteps = max_steps
        else:
            print("No valid saved model found. Starting a new model.")
            model = PPO("CnnPolicy", env, verbose=1, policy_kwargs={'normalize_images': False})
    else:
        print("No saved models found. Starting a new model.")
        model = PPO("CnnPolicy", env, verbose=1, policy_kwargs={'normalize_images': False})

    total_timesteps_per_iter = 100_000
    n_iterations = 10
    callback = PlottingCallback()

    for i in range(n_iterations):
        print(f"Training iteration {i+1}/{n_iterations}...")
        model.learn(total_timesteps=total_timesteps_per_iter, reset_num_timesteps=False, callback=callback)
        new_model_path = os.path.join(models_dir, f"ppo_snake_{round(model.num_timesteps, -4)}_steps")
        model.save(new_model_path)
        print(f"Model saved: {new_model_path}")

    print("Training complete!")

if __name__ == "__main__":
    train_snake()
