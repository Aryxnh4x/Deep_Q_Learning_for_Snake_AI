import torch
import numpy as np
import matplotlib.pyplot as plt
from dqn_agent import DQNAgent 
from snake_game import SnakeGame
from collections import deque

# --- Training Hyperparameters (Reduced for Speed) ---
NUM_EPISODES = 50 # <--- Significantly reduced from 200
TARGET_UPDATE_FREQ = 5 # <--- Increased frequency of Target Network updates for stability

# --- Metrics Tracking ---
score_history = [] 
total_steps_history = [] 
mean_scores = deque(maxlen=100) 

def train():
    game = SnakeGame()
    agent = DQNAgent(state_size=11, action_size=3)
    
    total_steps = 0 
    max_score = 0
    
    print("--- Starting DQN Training (50 Episodes) ---")
    
    for episode in range(1, NUM_EPISODES + 1):
        state = game.reset()
        done = False
        episode_score = 0
        episode_steps = 0
        
        while not done:
            # 1. Get action
            action = agent.get_action(state)
            
            # 2. Perform step
            next_state, reward, done, score = game.step(action)
            
            # 3. Store transition (s, a, r, s')
            agent.remember(state, action, reward, next_state, done)
            
            # 4. Train model (This now happens only every 5 steps, due to change in dqn_agent.py)
            agent.train_step()
            
            # 5. Update state and metrics
            state = next_state
            episode_score = score
            episode_steps += 1
            total_steps += 1

        # --- End of Episode ---
        agent.n_games += 1
        
        # Update Target Network periodically
        if episode % TARGET_UPDATE_FREQ == 0:
            agent.update_target_net()

        # Update and track metrics
        score_history.append(episode_score)
        total_steps_history.append(total_steps)
        mean_scores.append(episode_score)
        mean_score = np.mean(mean_scores)

        # Update max score
        if episode_score > max_score:
            max_score = episode_score

        # Print progress
        print(f'Episode {episode}/{NUM_EPISODES} | Score: {episode_score} | Max Score: {max_score} | Mean 100-Game Score: {mean_score:.2f}')

    # --- End of Training ---
    print("\n--- Training Complete ---")
    
    # This function will plot the results for your Learning Curve metric
    plot_results(score_history, total_steps_history)


def plot_results(scores, steps):
    """
    Generates plots for the Final Analysis Report (Deliverable 4)
    and visualizes the Learning Curve Metric.
    """
    
    # 1. Plot Score vs. Episode (Average Total Rewards per Episode)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(scores, label='Episode Score')
    plt.title('DQN Learning Curve (Score per Episode)')
    plt.xlabel('Episode')
    plt.ylabel('Score (Total Reward)')
    
    # Simple moving average for visualization of the trend
    if len(scores) >= 5: # Reduced MA window for shorter runs
        ma = np.convolve(scores, np.ones(5)/5, mode='valid')
        plt.plot(np.arange(len(ma)) + 4, ma, label='5-Episode MA', color='red')
    plt.legend()
    
    # 2. Plot Total Steps vs. Episode (Sample Complexity Metric)
    plt.subplot(1, 2, 2)
    plt.plot(steps, label='Cumulative Steps')
    plt.title('Sample Complexity (Total Steps Over Time)')
    plt.xlabel('Episode')
    plt.ylabel('Total Environment Steps')
    
    plt.tight_layout()
    plt.savefig('learning_curve.png')
    print("Saved learning curve plot to 'learning_curve.png'.")

if __name__ == '__main__':
    train()