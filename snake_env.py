import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces

class SnakeEnv(gym.Env):
    """
    Enhanced Snake Environment with improved path planning and space management.
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
        self.hunger_limit = self.grid_size * self.grid_size

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
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.grid[r][c] == 2:
                    return r, c
        return None
    
    def _distance_to_food(self):
        head_r, head_c = self.snake[-1]
        food_pos = self._find_food_pos()
        if food_pos is None:
            return 0
        return abs(head_r - food_pos[0]) + abs(head_c - food_pos[1])

    def _get_observation(self):
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
        grid_copy = self.grid.copy()
        grid_copy[start[0], start[1]] = 0
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
            stack.extend([(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)])
        return free_count - 1

    def step(self, action):
        step_reward = 0.0
        if self.done:
            return self._get_observation(), 0.0, True, False, {}

        self.current_step += 1
        head_r, head_c = self.snake[-1]
        old_distance = self._distance_to_food()
        
        # Compute free space before the move
        free_space_before = self.compute_free_space((head_r, head_c))
        
        # Determine valid moves
        valid_moves = {}
        directions = [(head_r - 1, head_c), (head_r + 1, head_c),
                    (head_r, head_c - 1), (head_r, head_c + 1)]
        for idx, (new_r, new_c) in enumerate(directions):
            if 0 <= new_r < self.grid_size and 0 <= new_c < self.grid_size and self.grid[new_r][new_c] != 1:
                valid_moves[idx] = (new_r, new_c)
        
        # Handle invalid action
        if action not in valid_moves:
            if not valid_moves:
                self.done = True
                return self._get_observation(), -10.0, self.done, False, {}
            action = random.choice(list(valid_moves.keys()))
            step_reward -= 2.0  # Penalty for invalid action
        
        new_r, new_c = valid_moves[action]
        
        # Check if new position is food before updating the grid
        ate_food = self.grid[new_r][new_c] == 2
        
        # Update snake and grid
        self.snake.append((new_r, new_c))
        self.grid[new_r][new_c] = 1  # Immediately mark new head position
        
        if ate_food:
            step_reward += 20.0
            self.steps_since_last_food = 0
            self._place_food()
        else:
            # Remove tail
            tail_r, tail_c = self.snake.pop(0)
            self.grid[tail_r][tail_c] = 0
        
        # Compute free space after the move with updated grid
        free_space_after = self.compute_free_space((new_r, new_c))
        
        # Reward shaping based on distance to food
        new_distance = self._distance_to_food()
        if new_distance < old_distance:
            step_reward += 0.5 / new_distance if new_distance != 0 else 0.5
        elif new_distance > old_distance:
            step_reward -= 0.5 / old_distance if old_distance != 0 else 0.5

        # Penalize reduction in accessible space
        if free_space_after < 0.7 * free_space_before:
            step_reward -= 2.0
        elif free_space_after < 0.3 * free_space_before:
            step_reward -= 4.0

        snake_length = len(self.snake)
        if snake_length >= 24:
            step_reward += 35
            print("HE DID IT!!")

        # Hunger penalty
        self.steps_since_last_food += 1
        if self.steps_since_last_food > self.hunger_limit:
            step_reward -= 15.0
            self.done = True

        self.direction = action
        return self._get_observation(), step_reward, self.done, False, {}

    def find_valid_move(self):
        head_r, head_c = self.snake[-1]
        directions = [0, 1, 2, 3]
        valid_moves = []

        for dir in directions:
            delta = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            new_r = head_r + delta[dir][0]
            new_c = head_c + delta[dir][1]
            if 0 <= new_r < self.grid_size and 0 <= new_c < self.grid_size and self.grid[new_r][new_c] != 1:
                valid_moves.append(dir)

        return random.choice(valid_moves) if valid_moves else None