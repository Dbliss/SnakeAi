import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces

class SnakeEnv(gym.Env):
    """
    A Snake environment that includes additional context in its observation:
      - The flattened grid (each cell: 0=empty, 1=snake, 2=food)
      - Relative position of the food to the snake's head (Δrow, Δcol)
      - Distances from the snake's head to each wall (top, bottom, left, right)
      - Current snake direction as a one-hot vector (up, down, left, right)
      
    Note: This environment does not include a render() function.
    """
    metadata = {"render_modes": []}  # No rendering provided

    def __init__(self, grid_size=10):
        super().__init__()
        self.grid_size = grid_size
        
        # Compute sizes:
        grid_obs_size = self.grid_size * self.grid_size
        # Additional features:
        #   relative food position: 2 values
        #   distances to walls: 4 values
        #   current direction as one-hot: 4 values
        extra_features = 2 + 4 + 4  # =10 extra features
        total_obs_size = grid_obs_size + extra_features
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_obs_size,),
            dtype=np.float32
        )

        # Action space: 0=up, 1=down, 2=left, 3=right
        self.action_space = spaces.Discrete(4)
        
        # Internal state variables
        self.grid = None
        self.snake = None
        self.direction = None  # 0=up, 1=down, 2=left, 3=right
        self.done = None
        self.current_step = 0
        self.max_steps = 300

        # Reward and episode management
        self.episode_reward = 0.0
        self.reward_threshold = -10.0  # End episode if total reward falls below this

        # Hunger mechanism: if too many steps pass without food, penalize/terminate
        self.steps_since_last_food = 0
        self.hunger_limit = 2 * self.grid_size

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.done = False
        self.episode_reward = 0.0
        self.steps_since_last_food = 0

        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)

        mid = self.grid_size // 2
        # Starting snake of length 3 (tail, middle, head) placed horizontally
        self.snake = [(mid, mid - 1), (mid, mid), (mid, mid + 1)]
        self.direction = 3  # e.g., 3 for right

        # Mark snake cells in the grid
        for (r, c) in self.snake:
            self.grid[r, c] = 1

        self._place_food()
        return self._get_observation(), {}

    def _place_food(self):
        """
        Place food randomly in one of the empty cells.
        If no empty cell exists, mark the episode as done.
        """
        empty_cells = [
            (r, c)
            for r in range(self.grid_size)
            for c in range(self.grid_size)
            if self.grid[r, c] == 0
        ]
        if not empty_cells:
            self.done = True
            return
        r, c = random.choice(empty_cells)
        self.grid[r, c] = 2

    def _find_food_pos(self):
        """
        Return the position of the food on the grid as a tuple (row, col).
        If no food is present, returns None.
        """
        positions = np.argwhere(self.grid == 2)
        if len(positions) == 0:
            return None
        return positions[0][0], positions[0][1]

    def _get_observation(self):
        """
        Return the observation which combines:
          - The flattened grid (values 0, 1, or 2)
          - Relative food position (Δrow, Δcol)
          - Distances from the snake's head to the walls (top, bottom, left, right)
          - Current direction as a one-hot encoded vector (length 4)
        """
        # Flatten the grid observation
        grid_obs = self.grid.flatten().astype(np.float32)

        # Compute relative food position
        head_r, head_c = self.snake[-1]
        food_pos = self._find_food_pos()
        if food_pos is None:
            rel_food = np.array([0, 0], dtype=np.float32)
        else:
            food_r, food_c = food_pos
            rel_food = np.array([food_r - head_r, food_c - head_c], dtype=np.float32)

        # Compute distances to each wall
        dist_top = head_r
        dist_bottom = self.grid_size - head_r - 1
        dist_left = head_c
        dist_right = self.grid_size - head_c - 1
        wall_dists = np.array([dist_top, dist_bottom, dist_left, dist_right], dtype=np.float32)

        # Encode current direction as one-hot vector (4 values)
        direction_one_hot = np.zeros(4, dtype=np.float32)
        direction_one_hot[self.direction] = 1.0

        extra_features = np.concatenate([rel_food, wall_dists, direction_one_hot])
        full_obs = np.concatenate([grid_obs, extra_features])
        return full_obs

    def step(self, action):
        """
        Apply an action and update the environment state.
        
        Args:
            action (int): 0=up, 1=down, 2=left, 3=right
        
        Returns:
            obs (np.array): The next observation.
            step_reward (float): The reward obtained for this step.
            done (bool): Whether the episode has ended.
            truncated (bool): Whether the episode was truncated.
            info (dict): Additional information.
        """
        if self.done:
            return self._get_observation(), 0.0, True, False, {}

        self.current_step += 1

        # Prevent immediate 180-degree turns.
        opposite = {0: 1, 1: 0, 2: 3, 3: 2}
        if action != opposite[self.direction]:
            self.direction = action

        step_reward = 0.0

        # Compute distance to food before the move (for shaping)
        old_distance = self._distance_to_food()

        # Get current head position
        head_r, head_c = self.snake[-1]

        # Compute new head position based on the current direction
        if self.direction == 0:      # up
            new_r, new_c = head_r - 1, head_c
        elif self.direction == 1:    # down
            new_r, new_c = head_r + 1, head_c
        elif self.direction == 2:    # left
            new_r, new_c = head_r, head_c - 1
        elif self.direction == 3:    # right
            new_r, new_c = head_r, head_c + 1

        # Check for collision with walls
        out_of_bounds = (new_r < 0 or new_r >= self.grid_size or
                         new_c < 0 or new_c >= self.grid_size)
        # Check for collision with self (only if within bounds)
        hits_self = False
        if not out_of_bounds and self.grid[new_r, new_c] == 1:
            hits_self = True

        if out_of_bounds or hits_self:
            # Illegal move: assign a strong penalty and terminate the episode.
            step_reward = -10.0
            self.done = True
            self.episode_reward += step_reward
            return self._get_observation(), step_reward, self.done, False, {}

        # Legal move: update snake's body
        self.snake.append((new_r, new_c))
        self.steps_since_last_food += 1

        # Check if the snake eats food
        if self.grid[new_r, new_c] == 2:
            step_reward += 2.0  # Reward for eating food
            self.steps_since_last_food = 0
            self.grid[new_r, new_c] = 0  # Remove food from the grid
            self._place_food()
        else:
            # Move forward: remove the tail
            tail_r, tail_c = self.snake.pop(0)
            self.grid[tail_r, tail_c] = 0

        # Update grid with the new head position
        self.grid[new_r, new_c] = 1

        # Apply a small time-step penalty
        step_reward -= 0.01

        # Distance-based reward shaping (after move)
        new_distance = self._distance_to_food()
        if new_distance < old_distance:
            step_reward += 0.1
        elif new_distance > old_distance:
            step_reward -= 0.1

        self.episode_reward += step_reward

        # Check hunger: if too many steps have passed without eating food
        if self.steps_since_last_food > self.hunger_limit:
            step_reward += self.reward_threshold

        # Terminate if total reward is below threshold
        if self.episode_reward < self.reward_threshold:
            self.done = True

        truncated = False
        if self.current_step >= self.max_steps:
            truncated = True
            self.done = True

        return self._get_observation(), step_reward, self.done, truncated, {}

    def _distance_to_food(self):
        """
        Calculate the Manhattan distance from the snake's head to the food.
        If no food exists, returns 0.
        """
        head_r, head_c = self.snake[-1]
        food_pos = self._find_food_pos()
        if food_pos is None:
            return 0
        food_r, food_c = food_pos
        return abs(head_r - food_r) + abs(head_c - food_c)
