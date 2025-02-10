import pygame
import time
import math
import gymnasium as gym
from stable_baselines3 import PPO

# Import the environment with your snake logic from snake_env.py
from snake_env import SnakeEnv

def get_angle_from_vector(v):
    """
    Given a vector v = (dr, dc), return the angle (in degrees) that an image 
    (default facing up) should be rotated so that it points in the direction of v.
    Convention (for straight segments):
      - (-1, 0): up    → 0°
      - (1, 0): down   → 180°
      - (0, -1): left  → 90°
      - (0, 1): right  → -90°
    """
    dr, dc = v
    if dr == 0 and dc == 0:
        return 0
    angle = -math.degrees(math.atan2(dc, -dr))
    return angle

def get_turn_angle(v1, v2):
    """
    Given two vectors v1 and v2 (the direction from the previous segment to the current,
    and from the current to the next segment, respectively), return the rotation angle (in degrees)
    for turn.png. We first convert v1 and v2 into discrete direction codes:
      0: up    (vector (-1, 0))
      1: down  (vector (1, 0))
      2: left  (vector (0, -1))
      3: right (vector (0, 1))
    
    Then we use a mapping dictionary to produce the angle (in degrees clockwise from the default orientation).
    
    The desired mappings (per your corrections) are:
      - (1, 2): down to left   → 180°
      - (1, 3): down to right  → 180°
      - (0, 2): up to left     → 270°
      - (3, 0): right to up    → 90°
      - (2, 1): left to down   → 270°
      - (3, 1): right to down  → 90°
      
    For completeness, we assign:
      - (0, 3): up to right    → 0°
      - (2, 0): left to up     → 90°  (as a mirror of up to left)
    """
    def get_dir(v):
        if v == (-1, 0):
            return 0  # up
        elif v == (1, 0):
            return 1  # down
        elif v == (0, -1):
            return 2  # left
        elif v == (0, 1):
            return 3  # right
        else:
            return None

    d1 = get_dir(v1)
    d2 = get_dir(v2)
    turn_mapping = {
        (0, 3): 0,      # up to right
        (3, 0): 180,     # right to up → 90° clockwise
        (0, 2): 270,    # up to left → 270° clockwise
        (2, 0): 90,     # left to up → 90° clockwise
        (1, 2): 180,    # down to left → 180°
        (1, 3): 90,    # down to right → 180°
        (2, 1): 0,    # left to down → 270°
        (3, 1): 270      # right to down → 90°
    }
    return turn_mapping.get((d1, d2), 0)

class SnakeEnvPygame(SnakeEnv):
    """
    A Snake environment that uses the logic (step, reset, etc.) from snake_env.py
    but adds a Pygame render() method for visualization using images.
    
    New images:
      - head_1.png and head_2.png for the head (alternating)
      - body.png for straight body segments (alternating horizontal flip to simulate movement)
      - turn.png for turning segments (default orientation assumed per get_turn_angle)
      - tail_1.png and tail_2.png for the tail (alternating)
      
    (Make sure snake_env.py has been modified so the starting snake size is 3.)
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, grid_size=10, block_size=30):
        # Initialize the original SnakeEnv from snake_env.py
        super().__init__(grid_size=grid_size)
        self.block_size = block_size

        # Pygame-specific attributes
        self.screen = None
        pygame.init()
        self.width = self.block_size * self.grid_size
        self.height = self.block_size * self.grid_size

        # Load images for snake parts (assumed to be in the "images" folder)
        self.head_images = [
            pygame.image.load("images/head_1.png"),
            pygame.image.load("images/head_2.png")
        ]
        self.body_image = pygame.image.load("images/body.png")
        self.turn_image = pygame.image.load("images/turn.png")
        self.tail_images = [
            pygame.image.load("images/tail_1.png"),
            pygame.image.load("images/tail_2.png")
        ]

        # Scale images to fit the block size
        self.head_images = [
            pygame.transform.scale(img, (self.block_size, self.block_size))
            for img in self.head_images
        ]
        self.body_image = pygame.transform.scale(self.body_image, (self.block_size, self.block_size))
        self.turn_image = pygame.transform.scale(self.turn_image, (self.block_size, self.block_size))
        self.tail_images = [
            pygame.transform.scale(img, (self.block_size, self.block_size))
            for img in self.tail_images
        ]

        # Frame counter for alternating images (for head, tail, and body flipping)
        self.frame_count = 0

    def render(self):
        """
        Render the current state of the environment using Pygame.
        Food cells are drawn as red rectangles.
        The snake is drawn using images:
          - Head: Alternates between head_1.png and head_2.png, rotated based on self.direction.
          - Tail: Alternates between tail_1.png and tail_2.png, rotated based on the vector from tail to its neighbor.
          - Body: For straight segments, uses body.png (with alternating horizontal flip for simulated movement);
                  for turning segments, uses turn.png rotated appropriately via get_turn_angle.
        """
        if self.screen is None:
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Snake AI")

        # Process events to keep the window responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        # Fill the background with black
        self.screen.fill((0, 0, 0))

        # Draw food cells (grid value 2)
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.grid[r, c] == 2:
                    x = c * self.block_size
                    y = r * self.block_size
                    rect = pygame.Rect(x, y, self.block_size, self.block_size)
                    pygame.draw.rect(self.screen, (255, 0, 0), rect)

        # Draw the snake. Assume self.snake is an ordered list from tail (index 0) to head (last index).
        for i, segment in enumerate(self.snake):
            r, c = segment
            x = c * self.block_size
            y = r * self.block_size

            if i == 0:  # Tail segment
                # Determine tail rotation based on the vector from tail to its neighbor.
                next_seg = self.snake[1]
                v = (next_seg[0] - r, next_seg[1] - c)
                angle = get_angle_from_vector(v)
                image = self.tail_images[self.frame_count % 2]
                rotated_image = pygame.transform.rotate(image, angle)
            elif i == len(self.snake) - 1:  # Head segment
                # Use self.direction (from snake_env) to determine head rotation.
                mapping = {0: 0, 1: 180, 2: 90, 3: -90}  # 0: up, 1: down, 2: left, 3: right
                angle = mapping.get(self.direction, 0)
                image = self.head_images[self.frame_count % 2]
                rotated_image = pygame.transform.rotate(image, angle)
            else:
                # Body segment
                prev_seg = self.snake[i - 1]
                next_seg = self.snake[i + 1]
                v1 = (segment[0] - prev_seg[0], segment[1] - prev_seg[1])
                v2 = (next_seg[0] - segment[0], next_seg[1] - segment[1])
                if v1 == v2:
                    # Straight segment: use body.png with alternating horizontal flip for simulated movement.
                    angle = get_angle_from_vector(v1)
                    if self.frame_count % 2 == 0:
                        img = self.body_image
                    else:
                        img = pygame.transform.flip(self.body_image, True, False)
                    rotated_image = pygame.transform.rotate(img, angle)
                else:
                    # Turning segment: use turn.png and get the correct rotation angle.
                    angle = get_turn_angle(v1, v2)
                    rotated_image = pygame.transform.rotate(self.turn_image, angle)

            self.screen.blit(rotated_image, (x, y))

        pygame.display.flip()
        self.frame_count += 1

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None


def show_trained_model(model_path, n_episodes=10, sleep_time=0.1):
    """
    Loads a trained PPO model, creates the SnakeEnvPygame environment,
    and runs n_episodes while rendering with Pygame.
    """
    env = SnakeEnvPygame(grid_size=10, block_size=30)
    model = PPO.load(model_path)

    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0.0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            env.render()
            time.sleep(sleep_time)

        print(f"Episode {episode+1} reward: {episode_reward}")

    env.close()

if __name__ == "__main__":
    model_file = "models/snake_ppo\ppo_snake_21389600_steps"  # Path to your saved model
    show_trained_model(model_file, n_episodes=10, sleep_time=0.25)
