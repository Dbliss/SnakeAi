import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces

class SnakeEnv(gym.Env):
    """
    Enhanced Snake Environment to prevent the snake from making the same invalid move repeatedly.
    """
    metadata = {"render_modes": []}

    def __init__(self, grid_size=5):
        super().__init__()
        self.grid_size = grid_size

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(6, self.grid_size, self.grid_size), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right

        self.grid = None
        self.snake = None
        self.direction = None
        self.done = None
        self.current_step = 0
        self.episode_reward = 0.0
        self.steps_since_last_food = 0
        self.hunger_limit = self.grid_size * self.grid_size - self.grid_size

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.done = False
        self.episode_reward = 0.0
        self.steps_since_last_food = 0

        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)

        mid = self.grid_size // 2
        self.snake = [(mid, mid - 1), (mid, mid), (mid, mid + 1)]
        self.direction = 3  # Start moving right

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
        """
        Find the position of the food in the grid.
        Returns the coordinates of the food, or None if no food is found.
        """
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.grid[r][c] == 2:  # Assuming '2' marks the food
                    return r, c
        return None
    
    def _distance_to_food(self):
        head_r, head_c = self.snake[-1]
        food_pos = self._find_food_pos()
        if food_pos is None:
            return 0
        return abs(head_r - food_pos[0]) + abs(head_c - food_pos[1])

    def _get_observation(self):
        """
        Update observation with possible move checks to avoid hitting walls or itself.
        """
        obs = np.zeros((self.grid_size, self.grid_size, 6), dtype=np.float32)

        obs[:, :, 0] = (self.grid == 1).astype(np.float32)
        obs[:, :, 1] = (self.grid == 2).astype(np.float32)
        obs[:, :, 2] = self.direction / 3.0

        head_r, head_c = self.snake[-1]
        possible_moves = [
            (head_r - 1, head_c),  # Up
            (head_r + 1, head_c),  # Down
            (head_r, head_c - 1),  # Left
            (head_r, head_c + 1)   # Right
        ]

        for i, (r_next, c_next) in enumerate(possible_moves):
            collision = (r_next < 0 or r_next >= self.grid_size or
                         c_next < 0 or c_next >= self.grid_size or
                         self.grid[r_next, c_next] == 1)
            if collision:
                obs[r_next % self.grid_size, c_next % self.grid_size, 3] = 1.0

        obs[:, :, 4] = head_r / (self.grid_size - 1)
        obs[:, :, 5] = head_c / (self.grid_size - 1)

        return obs.transpose(2, 0, 1)
    
    def compute_free_space(self, start):
        """
        Compute the number of reachable cells (free space) from the start position
        using a flood fill algorithm. The snake's body (value 1) and walls act as obstacles.
        """
        grid_copy = self.grid.copy()
        visited = np.zeros_like(grid_copy, dtype=bool)
        stack = [start]
        free_count = 0

        while stack:
            r, c = stack.pop()
            if not (0 <= r < self.grid_size and 0 <= c < self.grid_size):
                continue
            if visited[r, c]:
                continue
            if grid_copy[r, c] == 1:
                continue  # Obstacle: snake body
            visited[r, c] = True
            free_count += 1
            # Add neighbors
            stack.extend([(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)])
        return free_count

    def step(self, action):
        step_reward = 0.0
        if self.done:
            return self._get_observation(), 0.0, True, False, {}

        self.current_step += 1
        head_r, head_c = self.snake[-1]
        old_distance = self._distance_to_food()
        
        # Compute free space before the move
        free_space_before = self.compute_free_space((head_r, head_c))
        
        # Determine valid moves as before
        valid_moves = {}
        directions = [(head_r - 1, head_c), (head_r + 1, head_c),
                    (head_r, head_c - 1), (head_r, head_c + 1)]
        for idx, (new_r, new_c) in enumerate(directions):
            if 0 <= new_r < self.grid_size and 0 <= new_c < self.grid_size and self.grid[new_r][new_c] != 1:
                valid_moves[idx] = (new_r, new_c)
        
        # If chosen action isn't valid, pick a random valid action.
        if action not in valid_moves:
            if not valid_moves:
                self.done = True
                return self._get_observation(), -10.0, self.done, False, {}
            action = random.choice(list(valid_moves.keys()))
            step_reward -= 1.0
        
        new_r, new_c = valid_moves[action]
        
        # Execute the move
        self.snake.append((new_r, new_c))
        
        # Compute free space after the move
        free_space_after = self.compute_free_space((new_r, new_c))
        
        # Base reward shaping based on distance to food
        new_distance = self._distance_to_food()
        if new_distance < old_distance:
            step_reward += 1.0  # Reward for moving closer to food
        elif new_distance > old_distance:
            step_reward -= 1.0  # Penalty for moving away from food

        # Discourage moves that heavily reduce free space:
        if free_space_after < 0.5 * free_space_before:
            step_reward -= 2.0  # Adjust this value as needed

        # Check for food consumption
        if self.grid[new_r][new_c] == 2:  # Food is eaten
            step_reward += 10.0  # Large reward for eating food
            self.steps_since_last_food = 0
            self.grid[new_r][new_c] = 0
            self._place_food()  # Place new food
            # Do NOT remove the tail; the snake grows naturally.
        else:
            tail_r, tail_c = self.snake.pop(0)
            self.grid[tail_r][tail_c] = 0

        self.grid[new_r][new_c] = 1
        self.direction = action

        # Additional survival/hunger related penalties or bonuses can be applied here.

        return self._get_observation(), step_reward, self.done, False, {}

    def find_valid_move(self):
        """
        Check all possible moves and return a valid one if available, else None.
        """
        head_r, head_c = self.snake[-1]
        directions = [0, 1, 2, 3]  # Up, Down, Left, Right
        valid_moves = []

        for dir in directions:
            delta = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            new_r = head_r + delta[dir][0]
            new_c = head_c + delta[dir][1]
            if not (new_r < 0 or new_r >= self.grid_size or new_c < 0 or new_c >= self.grid_size or self.grid[new_r, new_c] == 1):
                valid_moves.append(dir)

        return random.choice(valid_moves) if valid_moves else None

    def update_snake_position(self, action):
        """
        Update the position of the snake in the grid based on the action.
        """
        head_r, head_c = self.snake[-1]
        delta = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right deltas
        new_r = head_r + delta[action][0]
        new_c = head_c + delta[action][1]

        self.snake.append((new_r, new_c))
        if self.grid[new_r, new_c] == 2:
            # Ate food, do not remove the tail (extend the snake)
            self.grid[new_r, new_c] = 0
            self._place_food()
        else:
            # Move normally by removing the tail
            tail_r, tail_c = self.snake.pop(0)
            self.grid[tail_r, tail_c] = 0

        self.grid[new_r, new_c] = 1  # Update the head position in the grid
        self.direction = action  # Update the current direction based on the action taken
