import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import os

MAX_MEMORY = 100_000 
BATCH_SIZE = 1000
LR = 0.001
TRAIN_FREQUENCY = 4

class LinearQNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], output_size)
        )

    def forward(self, x):
        return self.model(x)

class DQNAgent:
    # Defaults state_size to 16 for complex environment
    def __init__(self, state_size=16, action_size=3, hidden_sizes=[512, 256]): 
        self.state_size = state_size 
        self.action_size = action_size 
        
        # --- FIX: Initialize n_games here ---
        self.n_games = 0 
        
        self.gamma = 0.9 
        self.memory = deque(maxlen=MAX_MEMORY) 
        
        self.q_net = LinearQNet(state_size, hidden_sizes, action_size)
        self.target_net = LinearQNet(state_size, hidden_sizes, action_size)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval() 

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=LR)
        self.criterion = nn.MSELoss()
        self.step_counter = 0

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self):
        self.step_counter += 1
        if self.step_counter % TRAIN_FREQUENCY != 0: return
        if len(self.memory) < BATCH_SIZE: return

        mini_sample = random.sample(self.memory, BATCH_SIZE)
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        
        states = torch.tensor(np.array(states), dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float)
        dones = torch.tensor(dones, dtype=torch.bool)

        pred = self.q_net(states) 
        target = pred.clone()
        with torch.no_grad(): 
            max_q_next = torch.max(self.target_net(next_states), dim=1)[0]
            
        for idx in range(len(dones)):
            Q_new = rewards[idx]
            if not dones[idx]:
                Q_new = rewards[idx] + self.gamma * max_q_next[idx]
            
            target[idx][actions[idx].item()] = Q_new
            
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
    
    def update_target_net(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def get_action(self, state):
        # Epsilon decay uses self.n_games, which is now guaranteed to exist
        self.epsilon = max(0.01, 0.9 - self.n_games / 400) 
        
        if random.random() < self.epsilon:
            final_move = random.randint(0, self.action_size - 1)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            prediction = self.q_net(state_tensor)
            final_move = torch.argmax(prediction).item()
        return final_move