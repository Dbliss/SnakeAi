import numpy as np
from stable_baselines3 import PPO
from snake_env import SnakeEnv

def evaluate_model(model_path, n_episodes=10):
    env = SnakeEnv(grid_size=10)
    model = PPO.load(model_path, env=env)

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done, truncated = False, False
        total_reward = 0.0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)

            # Convert action from array to int
            action = int(action)  # or action = action.item()

            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

        print(f"Episode {ep+1} reward: {total_reward}")

if __name__ == "__main__":
    evaluate_model("models/snake_ppo/ppo_snake_250000_steps", n_episodes=10)
    evaluate_model("models/snake_ppo/ppo_snake_100000_steps", n_episodes=10)
    evaluate_model("models/snake_ppo/ppo_snake_50000_steps", n_episodes=10)
