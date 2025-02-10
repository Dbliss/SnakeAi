import os
import glob
import re
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from snake_env import SnakeEnv

def train_snake():
    models_dir = "models/snake_ppo"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # Create the environment and validate it.
    env = SnakeEnv(grid_size=10)
    check_env(env, warn=True)

    # Look for previously saved models matching the pattern "ppo_snake_*_steps.zip"
    model_files = glob.glob(os.path.join(models_dir, "ppo_snake_*_steps.zip"))
    best_model_path = None
    max_steps = 0

    if model_files:
        for path in model_files:
            # Expect filenames like "ppo_snake_500000_steps.zip"
            match = re.search(r"ppo_snake_(\d+)_steps", os.path.basename(path))
            if match:
                steps = int(match.group(1))
                if steps > max_steps:
                    max_steps = steps
                    best_model_path = os.path.join(models_dir, f"ppo_snake_{steps}_steps")
        if best_model_path is not None:
            print(f"Resuming training from {best_model_path}.zip with {max_steps} timesteps.")
            model = PPO.load(best_model_path, env=env)
            # Manually restore the timestep counter
            model.num_timesteps = max_steps
        else:
            print("No valid saved model found. Starting a new model.")
            model = PPO("MlpPolicy", env, verbose=1)
    else:
        print("No saved models found. Starting a new model.")
        model = PPO("MlpPolicy", env, verbose=1)

    total_timesteps_per_iter = 200_000
    n_iterations = 100

    for i in range(n_iterations):
        print(f"Training iteration {i+1}/{n_iterations}...")
        # Continue training without resetting the internal timestep counter.
        # Note: The actual number of timesteps will be rounded up to a multiple of n_steps.
        model.learn(total_timesteps=total_timesteps_per_iter, reset_num_timesteps=False)
        
        # Save the model using the updated total timesteps.
        new_model_path = os.path.join(models_dir, f"ppo_snake_{model.num_timesteps}_steps")
        model.save(new_model_path)
        print(f"Model saved: {new_model_path}")

    print("Training complete!")

if __name__ == "__main__":
    train_snake()
