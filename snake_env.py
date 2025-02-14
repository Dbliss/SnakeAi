import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces

class SnakeEnv(gym.Env):
    """
    Snake Environment with a simplified observation space and advanced reward engineering.

    Observation Space (6 channels):
      - Channel 0: Snake occupancy (1 if snake occupies the cell, else 0)
      - Channel 1: Food occupancy (1 if food is in the cell, else 0)
      - Channel 2: Direction encoding (constant value = self.direction/3 across the grid)
      - Channel 3: Danger map (1 for cells that are dangerous to move into)
      - Channel 4: Head row normalized (entire channel set to head_row/(grid_size-1))
      - Channel 5: Head column normalized (entire channel set to head_col/(grid_size-1))

    Rewards:
      - Collision penalty: -10
      - Food reward: +5
      - Fill-grid bonus: +10
      - Approaching or moving away from food: ±1/dist
      - Move awar from food bonus: -1/dist
      - Survival bonus: +0.01
      - Dynamic hunger penalty: increases with extra steps without food
    """
    metadata = {"render_modes": []}

    def __init__(self, grid_size=5):
        super().__init__()
        self.grid_size = grid_size

        # Observation space: 6 channels as described above.
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(6, self.grid_size, self.grid_size),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)

        self.grid = None
        self.snake = None
        self.direction = None
        self.done = None
        self.current_step = 0
        self.max_steps = grid_size * 20  # You can adjust this as needed.

        self.episode_reward = 0.0
        self.reward_threshold = -10.0

        self.steps_since_last_food = 0
        self.hunger_limit = self.grid_size * self.grid_size - self.grid_size

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.done = False
        self.episode_reward = 0.0
        self.steps_since_last_food = 0

        # Create an empty grid.
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)

        # Initialize the snake in the center (length = 3).
        mid = self.grid_size // 2
        self.snake = [(mid, mid - 1), (mid, mid), (mid, mid + 1)]
        self.direction = 3  # Start moving right.

        for (r, c) in self.snake:
            self.grid[r, c] = 1

        self._place_food()
        return self._get_observation(), {}

    def _place_food(self):
        empty_cells = [(r, c) for r in range(self.grid_size)
                             for c in range(self.grid_size)
                             if self.grid[r, c] == 0]
        if not empty_cells:
            self.done = True
            return True
        r, c = random.choice(empty_cells)
        self.grid[r, c] = 2
        return False

    def _find_food_pos(self):
        positions = np.argwhere(self.grid == 2)
        if len(positions) == 0:
            return None
        return positions[0][0], positions[0][1]

    def _distance_to_food(self):
        head_r, head_c = self.snake[-1]
        food_pos = self._find_food_pos()
        if food_pos is None:
            return 0
        return abs(head_r - food_pos[0]) + abs(head_c - food_pos[1])

    def _get_observation(self):
        obs = np.zeros((self.grid_size, self.grid_size, 6), dtype=np.float32)

        # Channel 0: snake occupancy.
        obs[:, :, 0] = (self.grid == 1).astype(np.float32)
        # Channel 1: food occupancy.
        obs[:, :, 1] = (self.grid == 2).astype(np.float32)
        # Channel 2: direction encoding (normalized: 0 to 1).
        obs[:, :, 2] = self.direction / 3.0
        # Channel 3: danger map. Mark adjacent cells that would cause a collision.
        head_r, head_c = self.snake[-1]
        possible_moves = [
            (head_r - 1, head_c),  # up
            (head_r + 1, head_c),  # down
            (head_r, head_c - 1),  # left
            (head_r, head_c + 1)   # right
        ]
        for (r_next, c_next) in possible_moves:
            collision = (r_next < 0 or r_next >= self.grid_size or
                         c_next < 0 or c_next >= self.grid_size or
                         self.grid[r_next, c_next] == 1)
            if collision:
                # Clamp coordinates if necessary.
                r_mark = max(0, min(r_next, self.grid_size - 1))
                c_mark = max(0, min(c_next, self.grid_size - 1))
                obs[r_mark, c_mark, 3] = 1.0
        # Channel 4: head row normalized.
        obs[:, :, 4] = head_r / (self.grid_size - 1) if self.grid_size > 1 else 0
        # Channel 5: head column normalized.
        obs[:, :, 5] = head_c / (self.grid_size - 1) if self.grid_size > 1 else 0

        # Transpose to (channels, height, width)
        return obs.transpose(2, 0, 1)

    def step(self, action):
        if self.done:
            return self._get_observation(), 0.0, True, False, {}

        self.current_step += 1

        # Prevent 180° turns.
        opposite = {0: 1, 1: 0, 2: 3, 3: 2}
        if action != opposite[self.direction]:
            self.direction = action

        step_reward = 0.0
        old_distance = self._distance_to_food()
        head_r, head_c = self.snake[-1]

        # Determine new head position.
        if self.direction == 0:   # up
            new_r, new_c = head_r - 1, head_c
        elif self.direction == 1: # down
            new_r, new_c = head_r + 1, head_c
        elif self.direction == 2: # left
            new_r, new_c = head_r, head_c - 1
        else:                   # right
            new_r, new_c = head_r, head_c + 1

        out_of_bounds = (new_r < 0 or new_r >= self.grid_size or new_c < 0 or new_c >= self.grid_size)
        hits_self = (not out_of_bounds and self.grid[new_r, new_c] == 1)

        if out_of_bounds or hits_self:
            step_reward = -10.0
            self.done = True
            self.episode_reward += step_reward
            return self._get_observation(), step_reward, self.done, False, {}

        # Move the snake.
        self.snake.append((new_r, new_c))
        self.steps_since_last_food += 1

        ate_food = False
        if self.grid[new_r, new_c] == 2:
            step_reward += 5.0
            self.steps_since_last_food = 0
            self.grid[new_r, new_c] = 0
            ate_food = True
        else:
            tail_r, tail_c = self.snake.pop(0)
            self.grid[tail_r, tail_c] = 0

        self.grid[new_r, new_c] = 1

        if ate_food:
            result = self._place_food()
            if result:
                step_reward += 10.0

        # Progressive reward shaping: reward for moving closer to food.
        new_distance = self._distance_to_food()
        if new_distance < old_distance:
            step_reward += 1 * 1 /  max(1, new_distance)
        elif new_distance > old_distance:
            step_reward -= 1 / max(1, new_distance)

        # Survival bonus.
        step_reward += 0.01

        # Dynamic hunger penalty: penalize extra steps without food.
        if self.steps_since_last_food > self.hunger_limit:
            extra = self.steps_since_last_food - self.hunger_limit
            step_reward -= extra * 0.05

        if self.episode_reward < self.reward_threshold:
            self.done = True
            self.episode_reward += step_reward
            return self._get_observation(), step_reward, self.done, False, {}

        truncated = False
        if self.current_step >= self.max_steps:
            truncated = True
            self.done = True

        self.episode_reward += step_reward
        return self._get_observation(), step_reward, self.done, truncated, {}
