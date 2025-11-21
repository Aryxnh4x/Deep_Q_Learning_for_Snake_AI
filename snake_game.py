import numpy as np
from collections import deque

class SnakeGame:
    """
    Core Snake environment based on project plan.
    State, Action, and Reward structure strictly adheres to project definition.
    """
    def __init__(self, w=10, h=10):
        self.w = w
        self.h = h
        self.reset()
        
    def reset(self):
        # Initialize game state
        self.snake = deque([(self.w // 2, self.h // 2)])
        self.direction = (0, 1) # Start moving right
        self.score = 0
        self.food = self._place_food()
        self.game_over = False
        self.frame_iteration = 0
        return self._get_state()

    def _place_food(self):
        # Simple food placement, avoids snake body
        while True:
            x = np.random.randint(0, self.w)
            y = np.random.randint(0, self.h)
            if (x, y) not in self.snake:
                return (x, y)

    def _get_state(self):
        """
        Calculates the 11-dimensional binary state vector S as defined in the project:
        [Danger (Left, Right, Straight), Food Direction (4 booleans), Tail Direction (4 booleans)]
        """
        head_x, head_y = self.snake[0]
        # Current direction is normalized (e.g., (0, 1) for right, (-1, 0) for up)
        dir_x, dir_y = self.direction 
        
        # Define 3 check points relative to the head based on current direction
        # Straight: in front of the head
        # Right: 90 deg clockwise to the straight direction
        # Left: 90 deg counter-clockwise to the straight direction
        
        point_straight = (head_x + dir_x, head_y + dir_y)
        point_right = (head_x + dir_y, head_y - dir_x) # Math for 90 deg right turn
        point_left = (head_x - dir_y, head_y + dir_x)  # Math for 90 deg left turn

        # 1. DANGER (3 booleans)
        # Danger is defined as hitting a wall or the snake's own body
        def is_danger(point):
            px, py = point
            if px < 0 or px >= self.w or py < 0 or py >= self.h:
                return 1 # Wall
            if (px, py) in list(self.snake)[1:]:
                return 1 # Body
            return 0

        danger_straight = is_danger(point_straight)
        danger_right = is_danger(point_right)
        danger_left = is_danger(point_left)

        # 2. FOOD DIRECTION (4 booleans)
        food_x, food_y = self.food
        food_up = 1 if food_y < head_y else 0
        food_down = 1 if food_y > head_y else 0
        food_left = 1 if food_x < head_x else 0
        food_right = 1 if food_x > head_x else 0
        
        # 3. TAIL DIRECTION (4 booleans)
        # Tail direction is often less critical for initial simple AIs, 
        # but included here for completeness as per the 11-dim state.
        # For simplicity, we compare tail position to the head (tail_pos - head_pos)
        tail_x, tail_y = self.snake[-1]
        
        tail_up = 1 if tail_y < head_y else 0
        tail_down = 1 if tail_y > head_y else 0
        tail_left = 1 if tail_x < head_x else 0
        tail_right = 1 if tail_x > head_x else 0

        state = [
            # Danger (3)
            danger_straight, danger_right, danger_left, 
            # Food Direction (4)
            food_up, food_down, food_left, food_right,
            # Tail Direction (4)
            tail_up, tail_down, tail_left, tail_right
        ]
        
        # Ensure state is exactly 11 dimensions
        assert len(state) == 11 
        return np.array(state, dtype=np.float32)

    def step(self, action):
        """
        Performs one game step.
        Action (A): [Turn Left, Go Straight, Turn Right]
        """
        self.frame_iteration += 1
        reward = -0.1 # Base negative reward to encourage speed [cite: 28]

        # 1. Update direction based on action (relative turn)
        # direction map: (0, 1): R, (0, -1): L, (1, 0): D, (-1, 0): U
        # [dir_x, dir_y] -> [turn left] -> [dir_y, -dir_x] (90 deg CCW)
        # [dir_x, dir_y] -> [turn right] -> [-dir_y, dir_x] (90 deg CW)
        
        dir_x, dir_y = self.direction
        if action == 0: # Turn Left
            self.direction = (dir_y, -dir_x)
        elif action == 2: # Turn Right
            self.direction = (-dir_y, dir_x)
        # action == 1 means Go Straight, direction remains the same

        # 2. Move snake head
        head_x, head_y = self.snake[0]
        new_head = (head_x + self.direction[0], head_y + self.direction[1])
        self.snake.appendleft(new_head)
        
        # 3. Check for game over (Collision)
        if self._is_collision(new_head):
            self.game_over = True
            reward = -100 # Penalty for death [cite: 27]
            
        # 4. Check if food is eaten
        food_eaten = (new_head == self.food)
        if food_eaten:
            reward = 10 # Reward for eating food [cite: 26]
            self.score += 1
            self.food = self._place_food()
        else:
            # If no food, remove tail
            self.snake.pop()

        # 5. Get new state
        new_state = self._get_state()

        return new_state, reward, self.game_over, self.score
    
    def _is_collision(self, pt):
        # Hits boundary
        if pt[0] < 0 or pt[0] >= self.w or pt[1] < 0 or pt[1] >= self.h:
            return True
        # Hits itself
        if pt in list(self.snake)[1:]:
            return True
        return False

# Example usage:
# game = SnakeGame()
# state = game.reset()
# next_state, reward, done, score = game.step(action=1)
# print(f"Initial State ({len(state)}D): {state}")