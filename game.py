import pygame
import random
import sys

# Initialize pygame
pygame.init()

# Screen dimensions
WIDTH = 600
HEIGHT = 400

# Colors (R, G, B)
WHITE = (255, 255, 255)
BLACK = (0,   0,   0)
RED   = (255, 0,   0)
GREEN = (0,   255, 0)

# Snake settings
SNAKE_SIZE = 20
SNAKE_SPEED = 10  # Adjust to make the game faster or slower

# Set up display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake Game - WASD Controls")

# Font for displaying score and game over
font = pygame.font.SysFont(None, 36)

def draw_snake(snake_body):
    """
    Draws the snake on the screen based on the list of positions (x, y).
    Each segment of the snake is drawn as a square of size SNAKE_SIZE.
    """
    for (x, y) in snake_body:
        pygame.draw.rect(screen, GREEN, [x, y, SNAKE_SIZE, SNAKE_SIZE])

def draw_food(x, y):
    """
    Draws the food on the screen at coordinates (x, y).
    Represented by a small red square.
    """
    pygame.draw.rect(screen, RED, [x, y, SNAKE_SIZE, SNAKE_SIZE])

def show_text(text, color, x, y):
    """
    Renders the given text on the screen at (x, y).
    """
    text_surface = font.render(text, True, color)
    screen.blit(text_surface, (x, y))

def game_loop():
    """
    Main game loop for the Snake game.
    """
    # Initial snake coordinates (start in the middle of the screen)
    x = WIDTH // 2
    y = HEIGHT // 2

    # Movement offsets for x and y
    x_change = 0
    y_change = 0

    # Snake body list and initial length
    snake_body = []
    snake_length = 1

    # Generate initial food position
    food_x = round(random.randrange(0, WIDTH - SNAKE_SIZE) / SNAKE_SIZE) * SNAKE_SIZE
    food_y = round(random.randrange(0, HEIGHT - SNAKE_SIZE) / SNAKE_SIZE) * SNAKE_SIZE

    clock = pygame.time.Clock()
    game_over = False

    while not game_over:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                # WASD controls
                if event.key == pygame.K_a and x_change == 0:
                    x_change = -SNAKE_SIZE
                    y_change = 0
                elif event.key == pygame.K_d and x_change == 0:
                    x_change = SNAKE_SIZE
                    y_change = 0
                elif event.key == pygame.K_w and y_change == 0:
                    y_change = -SNAKE_SIZE
                    x_change = 0
                elif event.key == pygame.K_s and y_change == 0:
                    y_change = SNAKE_SIZE
                    x_change = 0

        # Update snake position
        x += x_change
        y += y_change

        # Check wall collision
        if x < 0 or x >= WIDTH or y < 0 or y >= HEIGHT:
            game_over = True

        # Fill background
        screen.fill(BLACK)

        # Draw food
        draw_food(food_x, food_y)

        # Add new head to the snake body
        snake_head = [x, y]
        snake_body.append(snake_head)

        # Keep the snake body the correct length
        if len(snake_body) > snake_length:
            del snake_body[0]

        # Check self-collision
        for segment in snake_body[:-1]:
            if segment == snake_head:
                game_over = True

        # Draw the snake
        draw_snake(snake_body)

        # Check if snake ate the food
        if x == food_x and y == food_y:
            # Increase length of the snake
            snake_length += 1
            # Generate new food position
            food_x = round(random.randrange(0, WIDTH - SNAKE_SIZE) / SNAKE_SIZE) * SNAKE_SIZE
            food_y = round(random.randrange(0, HEIGHT - SNAKE_SIZE) / SNAKE_SIZE) * SNAKE_SIZE

        # Display the score
        score_text = f"Score: {snake_length - 1}"
        show_text(score_text, WHITE, 10, 10)

        # Update the screen
        pygame.display.update()

        # Control the speed of the game
        clock.tick(SNAKE_SPEED)

    # Game over screen
    screen.fill(BLACK)
    show_text("Game Over!", RED, WIDTH // 2 - 70, HEIGHT // 2 - 20)
    show_text(f"Final Score: {snake_length - 1}", WHITE, WIDTH // 2 - 100, HEIGHT // 2 + 20)
    pygame.display.update()

    # Pause so the player can see the final screen
    pygame.time.wait(2000)

def main():
    """
    Entry point of the Snake game.
    """
    while True:
        game_loop()

if __name__ == "__main__":
    main()
