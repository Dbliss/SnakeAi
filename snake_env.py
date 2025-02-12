import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces

class SnakeEnv(gym.Env):
    """
    Snake Environment with CNN-friendly observations and advanced features:
      - Mark "dangerous adjacent cells" in an additional channel
      - 7 channels total:
        (0) snake occupancy
        (1) food occupancy
        (2) direction_up
        (3) direction_down
        (4) direction_left
        (5) direction_right
        (6) dangerous cells (1 = if stepping there next turn would cause collision)
      - Rebalanced rewards:
        collision penalty = -20
        food reward = +5
        step penalty = -0.01
        survival/time bonus = +0.002
      - Additional +10 if snake fills the grid
    """
    metadata = {"render_modes": []}

    def __init__(self, grid_size=10):
        super().__init__()
        self.grid_size = grid_size

        # 7 channels: snake, food, direction_up, direction_down, direction_left, direction_right, dangerous
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(7, self.grid_size, self.grid_size),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)

        # Internal state
        self.grid = None
        self.snake = None
        self.direction = None
        self.done = None
        self.current_step = 0
        self.max_steps = 200

        self.episode_reward = 0.0
        self.reward_threshold = -10.0

        # Hunger mechanism
        self.steps_since_last_food = 0
        self.hunger_limit = self.grid_size * self.grid_size - self.grid_size

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.done = False
        self.episode_reward = 0.0
        self.steps_since_last_food = 0

        # Create empty grid
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)

        # Snake init in center, length=3
        mid = self.grid_size // 2
        self.snake = [(mid, mid - 1), (mid, mid), (mid, mid + 1)]
        self.direction = 3  # start right

        # Mark snake in grid
        for (r, c) in self.snake:
            self.grid[r, c] = 1

        self._place_food()
        return self._get_observation(), {}

    def _place_food(self):
        empty_cells = [
            (r, c) for r in range(self.grid_size)
            for c in range(self.grid_size)
            if self.grid[r, c] == 0
        ]
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
        """
        Return (7, grid_size, grid_size):
          ch0: snake
          ch1: food
          ch2-5: direction one-hot
          ch6: dangerous adjacency (1 if next step in that cell = collision)
        """
        obs = np.zeros((self.grid_size, self.grid_size, 7), dtype=np.float32)

        # ch0 = snake
        obs[:, :, 0] = (self.grid == 1).astype(np.float32)
        # ch1 = food
        obs[:, :, 1] = (self.grid == 2).astype(np.float32)

        # direction one-hot
        dir_one_hot = [0, 0, 0, 0]
        if self.direction in [0,1,2,3]:
            dir_one_hot[self.direction] = 1
        # up=0, down=1, left=2, right=3
        # ch2=up, ch3=down, ch4=left, ch5=right
        obs[:, :, 2] = dir_one_hot[0]
        obs[:, :, 3] = dir_one_hot[1]
        obs[:, :, 4] = dir_one_hot[2]
        obs[:, :, 5] = dir_one_hot[3]

        # Mark dangerous adjacent cells in ch6
        # For each possible next move from the head, if that move is out_of_bounds or hits snake, mark that cell in obs
        head_r, head_c = self.snake[-1]
        possible_moves = [
            (head_r - 1, head_c),  # up
            (head_r + 1, head_c),  # down
            (head_r, head_c - 1),  # left
            (head_r, head_c + 1)   # right
        ]
        for (r_next, c_next) in possible_moves:
            if (r_next < 0 or r_next >= self.grid_size or
                c_next < 0 or c_next >= self.grid_size or
                self.grid[r_next, c_next] == 1):
                # Mark this cell as dangerous if it's inside the grid
                if 0 <= r_next < self.grid_size and 0 <= c_next < self.grid_size:
                    obs[r_next, c_next, 6] = 1.0

        # transpose to (channels, H, W)
        obs = obs.transpose(2, 0, 1)
        return obs

    def step(self, action):
        if self.done:
            return self._get_observation(), 0.0, True, False, {}

        self.current_step += 1

        # Prevent 180 turns
        opposite = {0:1,1:0,2:3,3:2}
        if action != opposite[self.direction]:
            self.direction = action

        step_reward = 0.0
        old_distance = self._distance_to_food()
        head_r, head_c = self.snake[-1]

        # compute new head
        if self.direction == 0:   # up
            new_r, new_c = head_r - 1, head_c
        elif self.direction == 1: # down
            new_r, new_c = head_r + 1, head_c
        elif self.direction == 2: # left
            new_r, new_c = head_r, head_c - 1
        else:                     # right
            new_r, new_c = head_r, head_c + 1

        # check collisions
        out_of_bounds = (new_r<0 or new_r>=self.grid_size or new_c<0 or new_c>=self.grid_size)
        hits_self = False
        if not out_of_bounds and self.grid[new_r, new_c] == 1:
            hits_self = True

        if out_of_bounds or hits_self:
            step_reward = -10.0
            self.done = True
            self.episode_reward += step_reward
            return self._get_observation(), step_reward, self.done, False, {}

        # move head
        self.snake.append((new_r, new_c))
        self.steps_since_last_food += 1

        ate_food = False
        if self.grid[new_r, new_c] == 2:
            step_reward += 1.0
            self.steps_since_last_food = 0
            self.grid[new_r, new_c] = 0
            ate_food = True
        else:
            # remove tail
            tail_r, tail_c = self.snake.pop(0)
            self.grid[tail_r, tail_c] = 0

        self.grid[new_r, new_c] = 1

        if ate_food:
            result = self._place_food()
            if result:
                step_reward += 10.0  # fill-grid bonus

        # distance shaping
        new_distance = self._distance_to_food()
        if new_distance > old_distance:
            step_reward -= 0.01
        elif new_distance < old_distance:
            step_reward += 0.01

        # hunger
        if self.steps_since_last_food > self.hunger_limit:
            step_reward += self.reward_threshold  # -10

        if self.episode_reward < self.reward_threshold:
            self.done = True
            self.episode_reward += step_reward
            return self._get_observation(), step_reward, self.done, False, {}

        truncated = False
        if self.current_step >= self.max_steps:
            truncated = True
            self.done = True

        self.episode_reward += step_reward
        obs = self._get_observation()
        return obs, step_reward, self.done, truncated, {}

