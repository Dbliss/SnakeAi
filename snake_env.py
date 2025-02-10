import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces

class SnakeEnv(gym.Env):
    """
    A simplified Snake environment using Gymnasium.
    Collisions with walls or self are disallowed (the move is ignored).
    The episode ends if total reward drops below a certain threshold
    or we reach max_steps.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, grid_size=10):
        super().__init__()
        self.grid_size = grid_size
        
        # Define action and observation space
        self.action_space = spaces.Discrete(4)  # 0=up,1=down,2=left,3=right
        self.observation_space = spaces.Box(
            low=0,
            high=2,
            shape=(self.grid_size * self.grid_size,),
            dtype=np.int32
        )

        # Internal state
        self.grid = None
        self.snake = None
        self.direction = None

        # Episode management
        self.done = None
        self.current_step = 0
        self.max_steps = 1000

        # Track total reward in an episode
        self.episode_reward = 0.0
        self.reward_threshold = -1.0  # End episode if we dip below this

    def _get_observation(self):
        """Return the flattened grid as the observation."""
        return self.grid.flatten()

    def _place_food(self):
        """Place food in a random empty cell. If no empty cell, end episode."""
        empty_cells = [(r, c) for r in range(self.grid_size)
                       for c in range(self.grid_size)
                       if self.grid[r, c] == 0]
        if not empty_cells:
            # Grid is full, no place for food -> end the episode
            self.done = True
            return

        r, c = random.choice(empty_cells)
        self.grid[r, c] = 2

    def _find_food_pos(self):
        """Return (row, col) of the current food cell, or None if none found."""
        positions = np.argwhere(self.grid == 2)
        if len(positions) == 0:
            return None
        return positions[0][0], positions[0][1]

    def reset(self, seed=None, options=None):
        """
        Reset the environment.
        Returns: observation, info
        """
        super().reset(seed=seed)
        self.current_step = 0
        self.done = False
        self.episode_reward = 0.0

        # Create an empty grid
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)

        # Snake initial position (center)
        mid = self.grid_size // 2
        self.snake = [(mid, mid)]
        self.direction = 0  # 0=up,1=down,2=left,3=right

        # Place snake in grid
        self.grid[mid, mid] = 1

        # Place initial food
        self._place_food()

        return self._get_observation(), {}
    
    def _distance_to_food(self):
        head_r, head_c = self.snake[-1]
        food_pos = self._find_food_pos()
        if food_pos is None:
            # No food found, return 0 or some default
            return 0
        food_r, food_c = food_pos
        # Manhattan distance
        return abs(head_r - food_r) + abs(head_c - food_c)

    def step(self, action):
        """
        Apply action. 
        Disallow immediate 180-degree turns and disallow collisions
        by ignoring that move (or penalizing it).
        End episode if reward dips below self.reward_threshold or
        if we reach max_steps.
        Returns: (obs, reward, done, truncated, info)
        """
        if self.done:
            return self._get_observation(), 0.0, True, False, {}

        self.current_step += 1

        # Opposite directions
        opposite = {0: 1, 1: 0, 2: 3, 3: 2}
        # Update direction if not a 180 turn
        if action != opposite[self.direction]:
            self.direction = action

        # Initial reward for this step
        step_reward = 0.0

        # Distance shaping (before moving)
        old_distance = self._distance_to_food()

        # Calculate potential new head
        head_r, head_c = self.snake[-1]

        if self.direction == 0:   # up
            new_r, new_c = head_r - 1, head_c
        elif self.direction == 1: # down
            new_r, new_c = head_r + 1, head_c
        elif self.direction == 2: # left
            new_r, new_c = head_r, head_c - 1
        else:                     # right
            new_r, new_c = head_r, head_c + 1

        # Check for collision with wall or self
        out_of_bounds = (new_r < 0 or new_r >= self.grid_size or
                         new_c < 0 or new_c >= self.grid_size)
        hits_self = False
        if not out_of_bounds:
            if self.grid[new_r, new_c] == 1:
                hits_self = True

        if out_of_bounds or hits_self:
            # Disallow move: do not change snake position
            # Apply a penalty for trying an illegal move
            step_reward -= 0.05
            new_r, new_c = head_r, head_c  # Snake stays in place
        else:
            # Move snake
            self.snake.append((new_r, new_c))

            # Check if food eaten
            if self.grid[new_r, new_c] == 2:
                step_reward += 1.0
                self.grid[new_r, new_c] = 0
                self._place_food()
            else:
                # Remove tail
                tail_r, tail_c = self.snake.pop(0)
                self.grid[tail_r, tail_c] = 0

        # Update snake's head on the grid
        self.grid[new_r, new_c] = 1

        # Small time-step penalty
        step_reward -= 0.001

        # Distance shaping (after move)
        new_distance = self._distance_to_food()
        if new_distance < old_distance:
            step_reward += 0.1
        elif new_distance > old_distance:
            step_reward -= 0.1

        # Update total episode reward
        self.episode_reward += step_reward

        # Check if we should end because of low total reward
        if self.episode_reward < self.reward_threshold:
            self.done = True

        # Check step limit
        truncated = False
        if self.current_step >= self.max_steps:
            truncated = True
            self.done = True

        obs = self._get_observation()
        info = {}

        return obs, step_reward, self.done, truncated, info

    def render(self):
        """
        Simple console-based render. 
        You can adapt for Pygame if desired.
        """
        for r in range(self.grid_size):
            row_str = ""
            for c in range(self.grid_size):
                if self.grid[r, c] == 0:
                    row_str += ". "
                elif self.grid[r, c] == 1:
                    row_str += "S "
                elif self.grid[r, c] == 2:
                    row_str += "F "
            print(row_str)
        print("-" * (2 * self.grid_size))
