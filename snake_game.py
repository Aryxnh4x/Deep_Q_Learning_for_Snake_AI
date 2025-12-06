import pygame
import numpy as np
from collections import deque
import random
import time

# --- Presentation Constants ---
BLOCK_SIZE = 40 
SPEED = 20  # Speed of the game loop
GRID_W = 10
GRID_H = 10

WHITE = (255, 255, 255)
BLACK = (20, 20, 20) 
RED = (200, 0, 0)      # Food
GREEN_LIGHT = (0, 255, 100) # Snake Body
BLUE_LIGHT = (0, 150, 255)  # Snake Head
PURPLE_POISON = (180, 0, 180) # Poison

# Directions (dx, dy)
RIGHT = (1, 0)
LEFT = (-1, 0)
UP = (0, -1)
DOWN = (0, 1)

class SnakeGame:
    # THIS INIT METHOD ACCEPTS ARGUMENTS w AND h
    def __init__(self, w=10, h=10):
        self.w = w
        self.h = h
        self.display = None
        self.clock = None
        self.reset()
        
    def init_pygame(self, window_title="Snake AI"):
        pygame.init()
        self.window_w = self.w * BLOCK_SIZE
        self.window_h = self.h * BLOCK_SIZE
        self.display = pygame.display.set_mode((self.window_w, self.window_h + 40)) 
        pygame.display.set_caption(window_title)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('arial', 18, bold=True) 

    def close_pygame(self):
        pygame.quit()
        
    def reset(self):
        self.snake = deque([(self.w // 2, self.h // 2)])
        self.direction = RIGHT 
        self.score = 0
        self.food = self._place_item('food')
        self.poison = self._place_item('poison')
        self.game_over = False
        self.frame_iteration = 0
        return self._get_state()

    def _place_item(self, item_type):
        occupied = set(self.snake)
        if item_type == 'food' and hasattr(self, 'poison'):
             occupied.add(self.poison)
        elif item_type == 'poison' and hasattr(self, 'food'):
             occupied.add(self.food)

        while True:
            x = random.randint(0, self.w - 1)
            y = random.randint(0, self.h - 1)
            if (x, y) not in occupied:
                return (x, y)

    def step(self, action):
        self.frame_iteration += 1
        reward = -0.01 

        # 1. Update Direction
        dir_x, dir_y = self.direction
        if action == 0: # Turn Left
            self.direction = (dir_y, -dir_x)
        elif action == 2: # Turn Right
            self.direction = (-dir_y, dir_x)
        
        # 2. Move Head
        head_x, head_y = self.snake[0]
        new_head = (head_x + self.direction[0], head_y + self.direction[1])
        
        # 3. Check Death
        if self._is_collision(new_head) or self.frame_iteration > 100 * len(self.snake):
            self.game_over = True
            reward = -100 
            return self._get_state(), reward, self.game_over, self.score
            
        self.snake.appendleft(new_head)

        # 4. Check Interactions
        if new_head == self.food:
            reward = 10 
            self.score += 1
            self.food = self._place_item('food')
            if random.random() < 0.3: 
                self.poison = self._place_item('poison')
        
        elif new_head == self.poison:
            reward = -50 
            self.score -= 2 
            self.poison = self._place_item('poison')
            self.snake.pop() 
            if len(self.snake) == 0: 
                self.game_over = True
                reward = -100
        else:
            self.snake.pop() 

        return self._get_state(), reward, self.game_over, self.score
    
    def _is_collision(self, pt):
        if pt[0] < 0 or pt[0] >= self.w or pt[1] < 0 or pt[1] >= self.h:
            return True
        if pt in list(self.snake)[1:]:
            return True
        return False
        
    def _get_state(self):
        head_x, head_y = self.snake[0]
        dir_x, dir_y = self.direction 
        
        point_s = (head_x + dir_x, head_y + dir_y)
        point_r = (head_x + dir_y, head_y - dir_x)
        point_l = (head_x - dir_y, head_y + dir_x)
        
        def is_danger(point):
            px, py = point
            if px < 0 or px >= self.w or py < 0 or py >= self.h: return 1 
            if (px, py) in list(self.snake)[1:]: return 1 
            return 0

        # Feature Set (16 Dimensions)
        state = [
            # Danger (3)
            is_danger(point_s),
            is_danger(point_r),
            is_danger(point_l),
            
            # Direction (4)
            self.direction == LEFT,
            self.direction == RIGHT,
            self.direction == UP,
            self.direction == DOWN,
            
            # Food Location (4)
            self.food[1] < head_y, 
            self.food[1] > head_y, 
            self.food[0] < head_x, 
            self.food[0] > head_x, 

            # Poison Location (4)
            self.poison[1] < head_y, 
            self.poison[1] > head_y, 
            self.poison[0] < head_x, 
            self.poison[0] > head_x, 
            
            # Tail Proximity (1)
            (abs(self.snake[-1][0] - head_x) + abs(self.snake[-1][1] - head_y)) <= 2
        ]
        
        return np.array(state, dtype=np.float32)
        
    def draw_game(self, title, speed=SPEED):
        if not self.display: return

        self.display.fill(BLACK)
        
        for x in range(0, self.window_w, BLOCK_SIZE):
            pygame.draw.line(self.display, (30, 30, 30), (x, 0), (x, self.window_h))
        for y in range(0, self.window_h, BLOCK_SIZE):
            pygame.draw.line(self.display, (30, 30, 30), (0, y), (self.window_w, y))

        for i, pt in enumerate(self.snake):
            color = BLUE_LIGHT if i == 0 else GREEN_LIGHT 
            pygame.draw.rect(self.display, color, 
                             (pt[0]*BLOCK_SIZE+1, pt[1]*BLOCK_SIZE+1, BLOCK_SIZE-2, BLOCK_SIZE-2), 
                             border_radius=4)
            
        fx, fy = self.food
        pygame.draw.circle(self.display, RED, 
                           (fx*BLOCK_SIZE + BLOCK_SIZE//2, fy*BLOCK_SIZE + BLOCK_SIZE//2), 
                           BLOCK_SIZE//2 - 4)

        px, py = self.poison
        pygame.draw.rect(self.display, PURPLE_POISON, 
                         (px*BLOCK_SIZE + 5, py*BLOCK_SIZE + 5, BLOCK_SIZE-10, BLOCK_SIZE-10), 
                         border_radius=8)
        
        # Draw X on poison
        pygame.draw.line(self.display, BLACK, (px*BLOCK_SIZE+10, py*BLOCK_SIZE+10), (px*BLOCK_SIZE+BLOCK_SIZE-10, py*BLOCK_SIZE+BLOCK_SIZE-10), 3)
        pygame.draw.line(self.display, BLACK, (px*BLOCK_SIZE+10, py*BLOCK_SIZE+BLOCK_SIZE-10), (px*BLOCK_SIZE+BLOCK_SIZE-10, py*BLOCK_SIZE+10), 3)

        text = self.font.render(title, True, WHITE)
        self.display.blit(text, [10, self.window_h + 10])

        pygame.display.flip()
        if self.clock: self.clock.tick(speed)