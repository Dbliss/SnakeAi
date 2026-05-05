import os
import glob
import re
import time
import pygame
from stable_baselines3 import PPO
from snake_env import SnakeEnv

def get_angle_from_direction(direction):
    """
    Given a direction (0=up,1=down,2=left,3=right), return
    the rotation angle (in degrees) assuming the default image
    faces "up" (0=up -> angle=0, 1=down->180, 2=left->90, 3=right->-90).
    """
    mapping = {0: 0, 1: 180, 2: 90, 3: -90}
    return mapping.get(direction, 0)

class SnakeEnvPygame(SnakeEnv):
    """
    A subclass of SnakeEnv that uses images for head, body, tail, and turns.
    Even though our new SnakeEnv uses 6-channel observations internally,
    we still use self.grid (with snake=1, food=2) and self.snake (list of segments)
    to render the game.
    """
    def __init__(self, grid_size=5, block_size=30):
        # Note: grid_size now matches our training environment (e.g., 5x5)
        super().__init__(grid_size=grid_size)
        self.block_size = block_size

        # Instead of initializing the display here, we'll set an offscreen surface later.
        self.width = self.grid_size * self.block_size
        self.height = self.grid_size * self.block_size
        self.screen = None  # offscreen surface will be assigned in show_top_models

        # Load images (assuming they are in the 'images/' folder)
        self.head_images = [
            pygame.image.load("images/head_1.png"),
            pygame.image.load("images/head_2.png")
        ]
        self.body_image = pygame.image.load("images/body.png")
        self.tail_images = [
            pygame.image.load("images/tail_1.png"),
            pygame.image.load("images/tail_2.png")
        ]
        self.turn_image = pygame.image.load("images/turn.png")

        # Scale images to block_size x block_size
        self.head_images = [
            pygame.transform.scale(img, (self.block_size, self.block_size))
            for img in self.head_images
        ]
        self.body_image = pygame.transform.scale(self.body_image, (self.block_size, self.block_size))
        self.tail_images = [
            pygame.transform.scale(img, (self.block_size, self.block_size))
            for img in self.tail_images
        ]
        self.turn_image = pygame.transform.scale(self.turn_image, (self.block_size, self.block_size))

        # A frame counter to alternate head and tail images
        self.frame_count = 0

    def render_offscreen(self):
        """
        Render the current grid onto the offscreen surface (self.screen) and return that surface.
        This version does not flip the display or process events.
        """
        if self.screen is None:
            # Create an offscreen surface instead of using display.set_mode.
            self.screen = pygame.Surface((self.width, self.height))

        self.screen.fill((0, 0, 0))

        # Draw food (cells with value 2)
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.grid[r, c] == 2:
                    rect = pygame.Rect(c * self.block_size, r * self.block_size,
                                         self.block_size, self.block_size)
                    pygame.draw.rect(self.screen, (255, 0, 0), rect)

        # Draw the snake (from tail to head)
        for i, segment in enumerate(self.snake):
            r, c = segment
            x = c * self.block_size
            y = r * self.block_size

            if i == 0:
                # Tail
                tail_img = self.tail_images[self.frame_count % 2]
                if len(self.snake) > 1:
                    next_seg = self.snake[1]
                    direction_vector = (next_seg[0] - r, next_seg[1] - c)
                    tail_angle = self._get_angle_from_vector(direction_vector)
                else:
                    tail_angle = 0
                rotated_tail = pygame.transform.rotate(tail_img, tail_angle)
                self.screen.blit(rotated_tail, (x, y))
            elif i == len(self.snake) - 1:
                # Head
                head_img = self.head_images[self.frame_count % 2]
                angle = get_angle_from_direction(self.direction)
                rotated_head = pygame.transform.rotate(head_img, angle)
                self.screen.blit(rotated_head, (x, y))
            else:
                # Body or turn segment
                prev_seg = self.snake[i - 1]
                next_seg = self.snake[i + 1]
                v1 = (r - prev_seg[0], c - prev_seg[1])
                v2 = (next_seg[0] - r, next_seg[1] - c)
                if v1 == v2:
                    body_angle = self._get_angle_from_vector(v1)
                    rotated_body = pygame.transform.rotate(self.body_image, body_angle)
                    self.screen.blit(rotated_body, (x, y))
                else:
                    turn_angle = self._get_turn_angle(v1, v2)
                    rotated_turn = pygame.transform.rotate(self.turn_image, turn_angle)
                    self.screen.blit(rotated_turn, (x, y))
        self.frame_count += 1
        return self.screen

    def _get_angle_from_vector(self, v):
        dr, dc = v
        if dr == -1 and dc == 0:
            return 0   # up
        elif dr == 1 and dc == 0:
            return 180 # down
        elif dr == 0 and dc == -1:
            return 90  # left
        elif dr == 0 and dc == 1:
            return -90 # right
        return 0

    def _get_turn_angle(self, v1, v2):
        d1 = self._vector_to_dir(v1)
        d2 = self._vector_to_dir(v2)
        turn_mapping = {
            (0, 3): 0,      # up -> right
            (3, 1): 270,    # right -> down
            (1, 2): 180,    # down -> left
            (2, 0): 90,     # left -> up 
            (3, 0): 180,    # right -> up 
            (2, 1): 0,      # left -> down
            (1, 3): 90,     # down -> right
            (0, 2): 270     # up -> left
        }
        return turn_mapping.get((d1, d2), 0)

    def _vector_to_dir(self, v):
        dr, dc = v
        if dr == -1 and dc == 0:
            return 0
        elif dr == 1 and dc == 0:
            return 1
        elif dr == 0 and dc == -1:
            return 2
        elif dr == 0 and dc == 1:
            return 3
        return None

def get_top_model_paths(models_dir="models/snake_ppo", top_n=3):
    """
    Finds all model files, sorts them by the number of timesteps (extracted from filename),
    and returns the paths of the top_n models.
    Assumes filenames like "ppo_snake_{steps}_steps.zip".
    """
    model_files = glob.glob(os.path.join(models_dir, "ppo_snake_*_steps.zip"))
    models = []
    for path in model_files:
        basename = os.path.basename(path)
        match = re.search(r"ppo_snake_(\d+)_steps", basename)
        if match:
            steps = int(match.group(1))
            models.append((steps, path))
    models.sort(key=lambda x: x[0], reverse=True)
    top_models = [path for (_, path) in models[:top_n]]
    return top_models

def show_top_models(model_paths, n_episodes=10, sleep_time=0.1):
    """
    Create a main window that displays the top models (each playing a game) side by side.
    Each model gets its own SnakeEnvPygame with an offscreen surface, and we blit them
    into a single display.
    """
    # Parameters (assuming consistent grid and block size)
    grid_size = 5
    block_size = 30
    env_width = grid_size * block_size
    env_height = grid_size * block_size
    num_models = len(model_paths)
    main_width = env_width * num_models
    main_height = env_height

    # Initialize main Pygame display (one window for all models)
    pygame.init()
    main_screen = pygame.display.set_mode((main_width, main_height))
    pygame.display.set_caption("Top Snake Models")

    # Create an environment and model for each top model
    envs = []
    models = []
    for path in model_paths:
        env = SnakeEnvPygame(grid_size=grid_size, block_size=block_size)
        # Create an offscreen surface for the env
        env.screen = pygame.Surface((env_width, env_height))
        model = PPO.load(path, env=env, device="cuda")
        envs.append(env)
        models.append(model)

    # Run one episode for each model concurrently.
    episodes_done = [False] * num_models
    episode_rewards = [0.0] * num_models
    observations = [env.reset()[0] for env in envs]

    running = True
    clock = pygame.time.Clock()
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        for i in range(num_models):
            if not episodes_done[i]:
                action, _ = models[i].predict(observations[i], deterministic=True)
                # Safely convert action to int, handling 0D or 1D numpy arrays.
                if isinstance(action, (list, tuple)) or (hasattr(action, 'ndim') and action.ndim > 0):
                    action = int(action[0])
                else:
                    action = int(action)
                observations[i], reward, done, truncated, info = envs[i].step(action)
                episode_rewards[i] += reward
                if done:
                    episodes_done[i] = True

        # Render each env offscreen and blit into the main window.
        for i, env in enumerate(envs):
            offscreen = env.render_offscreen()
            main_screen.blit(offscreen, (i * env_width, 0))

        pygame.display.flip()
        clock.tick(10)

        if all(episodes_done):
            running = False

    for i in range(num_models):
        print(f"Model {i+1} reward: {episode_rewards[i]}")

    pygame.quit()

if __name__ == "__main__":
    models_dir = "models/snake_ppo"
    top_model_paths = get_top_model_paths(models_dir, top_n=3)
    if top_model_paths:
        print("Top models found:")
        for path in top_model_paths:
            print(path)
    else:
        print("No models found.")
        exit(1)
    show_top_models(top_model_paths, n_episodes=10, sleep_time=0.2)
