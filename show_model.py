import os
import glob
import re
import time
import math
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
    Even though snake_env.py has 7 channels internally (snake, food, direction, danger, etc.),
    we still track 'self.grid' for snake=1, food=2. We'll render based on that.
    """
    def __init__(self, grid_size=10, block_size=30):
        super().__init__(grid_size=grid_size)
        self.block_size = block_size

        # Pygame-specific attributes
        pygame.init()
        self.width = self.grid_size * self.block_size
        self.height = self.grid_size * self.block_size
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Snake with Images")

        # Load images (assuming they are in 'images/' folder).
        # Scale them to the block_size.
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

        # Scale all images to block_size x block_size
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

        # A frame counter to alternate head_1/head_2 and tail_1/tail_2
        self.frame_count = 0

    def render(self):
        """
        Render the current grid using custom images for each part of the snake.
        We'll use self.grid to find food cells (value=2) and keep using
        self.snake (a list of segments) to render body, head, tail, and turns.
        """
        if self.screen is None:
            self.screen = pygame.display.set_mode((self.width, self.height))

        # Clear the screen.
        self.screen.fill((0, 0, 0))

        # Draw food (value=2 in self.grid).
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.grid[r, c] == 2:
                    # Draw a red rectangle for food
                    rect = pygame.Rect(c * self.block_size, r * self.block_size,
                                       self.block_size, self.block_size)
                    pygame.draw.rect(self.screen, (255, 0, 0), rect)

        # Draw the snake from self.snake (tail->head)
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
                # Body or turn
                prev_seg = self.snake[i - 1]
                next_seg = self.snake[i + 1]
                v1 = (r - prev_seg[0], c - prev_seg[1])
                v2 = (next_seg[0] - r, next_seg[1] - c)

                if v1 == v2:
                    # Straight
                    body_angle = self._get_angle_from_vector(v1)
                    rotated_body = pygame.transform.rotate(self.body_image, body_angle)
                    self.screen.blit(rotated_body, (x, y))
                else:
                    # Turn
                    turn_angle = self._get_turn_angle(v1, v2)
                    rotated_turn = pygame.transform.rotate(self.turn_image, turn_angle)
                    self.screen.blit(rotated_turn, (x, y))

        pygame.display.flip()
        self.frame_count += 1

        # Process pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    # --------------------------------------------------------
    # Helper methods
    # --------------------------------------------------------
    def _get_angle_from_vector(self, v):
        """
        Return angle to rotate an up-facing image so it matches vector v.
        v: (dr, dc) in {(-1,0),(1,0),(0,-1),(0,1)}
        up->0°, down->180°, left->90°, right->-90°.
        """
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
        """
        For turning segments, v1 is direction from prev->current,
        v2 is direction from current->next. We define a minimal dictionary
        for angles. Expand if needed for all turn combos.
        """
        d1 = self._vector_to_dir(v1)
        d2 = self._vector_to_dir(v2)

        turn_mapping = {
            (0,3): 0,      # up->right
            (3,1): 90,     # right->down
            (1,2): 0,      # down->left => might need 180
            (2,0): 90,     # left->up
        }
        return turn_mapping.get((d1,d2), 0)

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

def get_highest_model_path(models_dir="models/snake_ppo"):
    model_files = glob.glob(os.path.join(models_dir, "ppo_snake_*_steps.zip"))
    best_model_path = None
    max_steps = 0
    if model_files:
        for path in model_files:
            match = re.search(r"ppo_snake_(\d+)_steps", os.path.basename(path))
            if match:
                steps = int(match.group(1))
                if steps > max_steps:
                    max_steps = steps
                    best_model_path = os.path.join(models_dir, f"ppo_snake_{steps}_steps")
    return best_model_path, max_steps

def show_trained_model(model_path, n_episodes=10, sleep_time=0.2):
    env = SnakeEnvPygame(grid_size=10, block_size=30)
    model = PPO.load(model_path, env=env, device="cuda")
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            if not isinstance(action, int):
                action = int(action)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            env.render()
            time.sleep(sleep_time)
        
        print(f"Episode {episode+1} reward: {episode_reward}")
    
    env.close()

if __name__ == "__main__":
    models_dir = "models/snake_ppo"
    best_model_path, max_steps = get_highest_model_path(models_dir)
    if best_model_path:
        print(f"Resuming training from {best_model_path}.zip with {max_steps} timesteps.")
        model_file = best_model_path + ".zip"
    else:
        print("No model found.")
        exit(1)

    show_trained_model(model_file, n_episodes=10, sleep_time=0.2)
