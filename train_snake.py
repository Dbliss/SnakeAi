import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from snake_env import SnakeEnv

def train_snake():
    models_dir = "models/snake_ppo"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # Create environment
    env = SnakeEnv(grid_size=10)
    # Check that environment follows the Gymnasium API
    check_env(env, warn=True)

    # Create RL model
    model = PPO("MlpPolicy", env, verbose=1)

    total_timesteps_per_iter = 50_000
    n_iterations = 90

    for i in range(n_iterations):
        print(f"Training iteration {i+1}/{n_iterations}...")
        model.learn(total_timesteps=total_timesteps_per_iter)

        # Save model after each iteration
        model_path = f"{models_dir}/ppo_snake_{(i+1)*total_timesteps_per_iter}_steps"
        model.save(model_path)
        print(f"Model saved: {model_path}")

    print("Training complete!")

if __name__ == "__main__":
    train_snake()
