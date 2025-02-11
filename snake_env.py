import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces

class SnakeEnv(gym.Env):
    """
    Snake Environment with CNN-friendly observations.

    Observation:
      A 3D array of shape (2, grid_size, grid_size) where:
        - Channel 0: 1 if the cell is occupied by the snake, 0 otherwise.
        - Channel 1: 1 if the cell contains food, 0 otherwise.

    Actions:
      0 = up, 1 = down, 2 = left, 3 = right.

    Note: This environment does not include a render() function.
    """
    metadata = {"render_modes": []}

    def __init__(self, grid_size=10):
        super().__init__()
        self.grid_size = grid_size

        # Observation space: channels-first: (2, grid_size, grid_size)
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(2, self.grid_size, self.grid_size),
            dtype=np.float32
        )
        # Four discrete actions.
        self.action_space = spaces.Discrete(4)

        # Internal state variables.
        self.grid = None           # 2D grid: 0 = empty, 1 = snake, 2 = food.
        self.snake = None          # List of (row, col) tuples, tail first, head last.
        self.direction = None      # 0 = up, 1 = down, 2 = left, 3 = right.
        self.done = None           # Episode termination flag.
        self.current_step = 0
        self.max_steps = 300

        self.episode_reward = 0.0
        self.reward_threshold = -10.0

        # Hunger mechanism.
        self.steps_since_last_food = 0
        self.hunger_limit = 5 * self.grid_size

    def reset(self, seed=None, options=None):
        """Reset the environment to its initial state and return the observation."""
        super().reset(seed=seed)
        self.current_step = 0
        self.done = False
        self.episode_reward = 0.0
        self.steps_since_last_food = 0

        # Create an empty grid.
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)

        # Initialize the snake in the center with length 3.
        mid = self.grid_size // 2
        self.snake = [(mid, mid - 1), (mid, mid), (mid, mid + 1)]
        self.direction = 3  # Start moving right.

        # Mark snake cells in the grid.
        for (r, c) in self.snake:
            self.grid[r, c] = 1

        self._place_food()
        return self._get_observation(), {}

    def _place_food(self):
        """Place food randomly in an empty cell. End episode if no empty cell exists."""
        empty_cells = [
            (r, c)
            for r in range(self.grid_size)
            for c in range(self.grid_size)
            if self.grid[r, c] == 0
        ]
        if not empty_cells:
            self.done = True
            return False
        r, c = random.choice(empty_cells)
        self.grid[r, c] = 2

    def _find_food_pos(self):
        """Return the (row, col) position of food if it exists; otherwise, return None."""
        positions = np.argwhere(self.grid == 2)
        if len(positions) == 0:
            return None
        return positions[0][0], positions[0][1]

    def _get_observation(self):
        """
        Return the observation as a 3D array with shape (2, grid_size, grid_size):
          - Channel 0: snake occupancy (1 if cell is snake, else 0).
          - Channel 1: food occupancy (1 if cell has food, else 0).
        """
        obs = np.zeros((self.grid_size, self.grid_size, 2), dtype=np.float32)
        obs[:, :, 0] = (self.grid == 1).astype(np.float32)
        obs[:, :, 1] = (self.grid == 2).astype(np.float32)
        # Transpose from (H, W, C) to (C, H, W)
        obs = obs.transpose(2, 0, 1)
        return obs

    def step(self, action):
        """
        Execute an action, update the environment, and return:
          (observation, step_reward, done, truncated, info)
        """
        if self.done:
            return self._get_observation(), 0.0, True, False, {}

        self.current_step += 1

        # Prevent immediate 180° turns.
        opposite = {0: 1, 1: 0, 2: 3, 3: 2}
        if action != opposite[self.direction]:
            self.direction = action

        step_reward = 0.0
        old_distance = self._distance_to_food()
        head_r, head_c = self.snake[-1]

        # Compute new head position.
        if self.direction == 0:      # up
            new_r, new_c = head_r - 1, head_c
        elif self.direction == 1:    # down
            new_r, new_c = head_r + 1, head_c
        elif self.direction == 2:    # left
            new_r, new_c = head_r, head_c - 1
        elif self.direction == 3:    # right
            new_r, new_c = head_r, head_c + 1

        # Check for collisions.
        out_of_bounds = (new_r < 0 or new_r >= self.grid_size or new_c < 0 or new_c >= self.grid_size)
        hits_self = False
        if not out_of_bounds and self.grid[new_r, new_c] == 1:
            hits_self = True

        if out_of_bounds or hits_self:
            step_reward = -10.0
            self.done = True
            self.episode_reward += step_reward
            return self._get_observation(), step_reward, self.done, False, {}

        # Append new head.
        self.snake.append((new_r, new_c))
        self.steps_since_last_food += 1

        ate_food = False
        if self.grid[new_r, new_c] == 2:
            step_reward += 2.0  # Reward for eating food.
            self.steps_since_last_food = 0
            self.grid[new_r, new_c] = 0  # Remove food.
            ate_food = True
        else:
            # Remove tail if no food was eaten.
            tail_r, tail_c = self.snake.pop(0)
            self.grid[tail_r, tail_c] = 0

        # Mark new head as occupied.
        self.grid[new_r, new_c] = 1

        # If food was eaten, place new food.
        if ate_food:
            result = self._place_food()
            if not result:
                step_reward += 100

        # Small step penalty.
        step_reward -= 0.01

        # Reward shaping: reward if closer to food.
        new_distance = self._distance_to_food()
        if new_distance < old_distance:
            step_reward += 0.1
        elif new_distance > old_distance:
            step_reward -= 0.1

        self.episode_reward += step_reward

        # Hunger: if too many steps pass without food, penalize.
        if self.steps_since_last_food > self.hunger_limit:
            step_reward += self.reward_threshold
        if self.episode_reward < self.reward_threshold:
            self.done = True

        truncated = False
        if self.current_step >= self.max_steps:
            truncated = True
            self.done = True

        return self._get_observation(), step_reward, self.done, truncated, {}

    def _distance_to_food(self):
        """Compute Manhattan distance from the snake's head to the food."""
        head_r, head_c = self.snake[-1]
        food_pos = self._find_food_pos()
        if food_pos is None:
            return 0
        food_r, food_c = food_pos
        return abs(head_r - food_r) + abs(head_c - food_c)
