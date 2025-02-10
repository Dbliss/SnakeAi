# show_model.py

import pygame
import time
import numpy as np
import random

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

class SnakeEnvPygame(gym.Env):
    """
    A Snake environment similar to snake_env.py but adds a Pygame render() method
    so we can visualize in a window.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, grid_size=10, block_size=30):
        super().__init__()
        self.grid_size = grid_size
        self.block_size = block_size

        # Gym spaces (same as your trained environment)
        self.action_space = spaces.Discrete(4)  # 0=up,1=down,2=left,3=right
        self.observation_space = spaces.Box(
            low=0, high=2, 
            shape=(self.grid_size * self.grid_size,),
            dtype=np.int32
        )

        # Internal state
        self.grid = None
        self.snake = None
        self.direction = None
        self.done = None
        self.current_step = 0
        self.max_steps = 1000

        # Pygame-specific attributes
        self.screen = None
        pygame.init()
        self.width = self.block_size * self.grid_size
        self.height = self.block_size * self.grid_size

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.done = False
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)

        # Place snake in middle
        mid = self.grid_size // 2
        self.snake = [(mid, mid)]
        self.direction = 0  # 0=up
        self.grid[mid, mid] = 1

        self._place_food()

        obs = self._get_observation()
        return obs, {}

    def step(self, action):
        if self.done:
            return self._get_observation(), 0.0, True, False, {}

        self.current_step += 1

        # Disallow 180-degree turns
        opposite = {0: 1, 1: 0, 2: 3, 3: 2}
        if action != opposite[self.direction]:
            self.direction = action

        old_distance = self._distance_to_food()

        # Current head
        head_r, head_c = self.snake[-1]

        # Move the head
        if self.direction == 0:  # up
            head_r -= 1
        elif self.direction == 1:  # down
            head_r += 1
        elif self.direction == 2:  # left
            head_c -= 1
        elif self.direction == 3:  # right
            head_c += 1

        # Check wall collision
        if head_r < 0 or head_r >= self.grid_size or head_c < 0 or head_c >= self.grid_size:
            reward = -1.0
            self.done = True
            return self._get_observation(), reward, True, False, {}

        # Check self-collision
        if self.grid[head_r, head_c] == 1:
            reward = -1.0
            self.done = True
            return self._get_observation(), reward, True, False, {}

        # Move snake
        self.snake.append((head_r, head_c))

        reward = 0.0
        if self.grid[head_r, head_c] == 2:  # ate food
            reward = 1.0
            self.grid[head_r, head_c] = 0
            self._place_food()
        else:
            # remove tail
            tail_r, tail_c = self.snake.pop(0)
            self.grid[tail_r, tail_c] = 0

        # Update grid with new head
        self.grid[head_r, head_c] = 1

        # Distance-based shaping
        new_distance = self._distance_to_food()
        if new_distance < old_distance:
            reward += 0.01
        elif new_distance > old_distance:
            reward -= 0.01

        # Step limit
        truncated = False
        if self.current_step >= self.max_steps:
            truncated = True
            self.done = True

        return self._get_observation(), reward, self.done, truncated, {}

    def render(self):
        """
        Renders the grid using Pygame. Each cell is drawn as a rectangle:
         - Snake in green
         - Food in red
         - Empty in black
        """
        if self.screen is None:
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Snake AI")

        # Handle Pygame events so the window stays responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        # Fill background
        self.screen.fill((0, 0, 0))

        # Draw cells
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                cell_value = self.grid[r, c]
                x = c * self.block_size
                y = r * self.block_size

                if cell_value == 1:  # snake
                    color = (0, 255, 0)  # green
                elif cell_value == 2:  # food
                    color = (255, 0, 0)  # red
                else:
                    color = (0, 0, 0)    # empty

                rect = pygame.Rect(x, y, self.block_size, self.block_size)
                pygame.draw.rect(self.screen, color, rect)

        pygame.display.flip()

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    # --- Helper methods ---
    def _get_observation(self):
        return self.grid.flatten()

    def _place_food(self):
        empty_cells = [(r, c) for r in range(self.grid_size)
                       for c in range(self.grid_size)
                       if self.grid[r, c] == 0]
        if not empty_cells:
            self.done = True
            return
        r, c = random.choice(empty_cells)
        self.grid[r, c] = 2

    def _distance_to_food(self):
        head_r, head_c = self.snake[-1]
        positions = np.argwhere(self.grid == 2)
        if len(positions) == 0:
            return 0
        food_r, food_c = positions[0]
        return abs(head_r - food_r) + abs(head_c - food_c)


def show_trained_model(model_path, n_episodes=10, sleep_time=0.1):
    """
    Loads a trained PPO model, creates the SnakeEnvPygame environment,
    and plays n_episodes while rendering with Pygame.
    """
    env = SnakeEnvPygame(grid_size=10, block_size=30)
    model = PPO.load(model_path)

    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0.0

        while not (done or truncated):
            # Model might return actions as np.array, convert to int
            action, _states = model.predict(obs, deterministic=True)
            action = int(action)

            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward

            env.render()
            time.sleep(sleep_time)  # slow down so you can see

        print(f"Episode {episode+1} reward: {episode_reward}")

    env.close()


if __name__ == "__main__":
    # Example usage:
    model_file = "models/snake_ppo/ppo_snake_4500000_steps.zip"  # or any saved model
    show_trained_model(model_file, n_episodes=10, sleep_time=0.1)
