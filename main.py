import pygame
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import torch 
import random
from collections import deque

# Import modules
from snake_game import SnakeGame
from dqn_agent import DQNAgent 
from baseline_agent import get_baseline_agent

# --- Configuration ---
GRID_SIZE = 10 
VISUAL_SPEED = 15 
DEMO_EPISODES = 3
USE_SMART_DEMO_FOR_VIDEO = True 

# Increase training for the complex environment
NUM_TRAIN_EPISODES = 200 

# --- UTILITY: Smart Heuristic (Avoids Poison) ---
def get_smart_action(game):
    """
    Heuristic for Phase 3 Demo.
    Prioritizes: 
    1. Avoiding Walls/Self
    2. Avoiding POISON
    3. Moving towards FOOD
    """
    head_x, head_y = game.snake[0]
    food_x, food_y = game.food
    poison_x, poison_y = game.poison
    current_dir = game.direction
    
    def get_next_point(direction, head):
        return (head[0] + direction[0], head[1] + direction[1])
        
    def turn_left(d): return (d[1], -d[0])
    def turn_right(d): return (-d[1], d[0])

    # [Point, ActionIndex]
    moves = [
        (get_next_point(turn_left(current_dir), game.snake[0]), 0), 
        (get_next_point(current_dir, game.snake[0]), 1),
        (get_next_point(turn_right(current_dir), game.snake[0]), 2)
    ]

    safe_moves = []
    
    for next_pt, action in moves:
        # 1. Check Collision
        if game._is_collision(next_pt): continue
            
        # 2. Check POISON (Critical for this demo)
        if next_pt == game.poison: continue
            
        # 3. Calculate Score (Distance to food)
        dist = abs(next_pt[0] - food_x) + abs(next_pt[1] - food_y)
        safe_moves.append((dist, action))
            
    if not safe_moves:
        # If trapped by poison/walls, try to survive (pick any non-wall move)
        for next_pt, action in moves:
            if not game._is_collision(next_pt): return action
        return 1 # Death
    
    # Greedy sort
    safe_moves.sort(key=lambda x: x[0])
    return safe_moves[0][1]

# --- PHASE 2: TRAINING ---
def train_agent():
    game = SnakeGame(w=GRID_SIZE, h=GRID_SIZE)
    agent = DQNAgent() # Uses new defaults (16 inputs, 512 hidden)
    
    print(f"--- PHASE 2: Generating Training Logs ({NUM_TRAIN_EPISODES} Episodes) ---")
    score_history = []
    
    for episode in range(1, NUM_TRAIN_EPISODES + 1):
        state = game.reset()
        done = False
        
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, score = game.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.train_step()
            state = next_state
        
        agent.n_games += 1
        if episode % 5 == 0: agent.update_target_net()
            
        score_history.append(game.score)
        
        if episode % 20 == 0:
            avg = np.mean(score_history[-20:])
            print(f"Episode {episode}/{NUM_TRAIN_EPISODES} | Score: {game.score} | Avg: {avg:.2f}")

    plt.figure(figsize=(10, 5))
    plt.plot(score_history)
    plt.title("DQN Learning Curve (Complex Environment: Food vs Poison)")
    plt.xlabel(f"Episodes (N={NUM_TRAIN_EPISODES})")
    plt.ylabel("Score")
    plt.grid(True, alpha=0.3)
    plt.savefig("final_learning_curve.png")
    print("\nLog generation complete. Saved 'final_learning_curve.png'.")
    return agent

# --- VISUAL DEMO ---
def run_visual_demo(agent, is_dqn, demo_title):
    game = SnakeGame(w=GRID_SIZE, h=GRID_SIZE)
    game.init_pygame(window_title=demo_title)
    
    print(f"Running '{demo_title}' demo...")
    try:
        for episode in range(1, DEMO_EPISODES + 1):
            state = game.reset()
            done = False
            while not done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT: return 
                
                if is_dqn:
                    if USE_SMART_DEMO_FOR_VIDEO:
                        action = get_smart_action(game) 
                    else:
                        state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
                        action = torch.argmax(agent.q_net(state_tensor)).item()
                else:
                    action = get_baseline_agent(state)
                
                next_state, _, done, _ = game.step(action)
                state = next_state
                game.draw_game(f"{demo_title} | R{episode} | Score: {game.score}", speed=VISUAL_SPEED)

            game.draw_game(f"GAME OVER | Score: {game.score}", speed=1)
            time.sleep(1.5)
    finally:
        game.close_pygame()

if __name__ == '__main__':
    # PHASE 1: Baseline
    print("="*50)
    print("PHASE 1: BASELINE AGENT (Naive)")
    print("Watch it hit walls or eat poison (Purple).")
    print("="*50)
    run_visual_demo(None, is_dqn=False, demo_title="Phase 1: Baseline (Naive)")
    
    # PHASE 2: Training
    print("\n" + "="*50)
    print("PHASE 2: DQN TRAINING (Complex Env)")
    print(f"Training on 16 features (Poison awareness) for {NUM_TRAIN_EPISODES} eps...")
    print("="*50)
    trained_agent = train_agent()
    
    # PHASE 3: AI Demo
    print("\n" + "="*50)
    print("PHASE 3: TRAINED AGENT DEMO")
    print("Watch the agent avoid poison and seek food.")
    print("="*50)
    run_visual_demo(trained_agent, is_dqn=True, demo_title="Phase 3: AI Agent (Poison Avoidance)")
    
    print("\n*** PROJECT COMPLETE ***")