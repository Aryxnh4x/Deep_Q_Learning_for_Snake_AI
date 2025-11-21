import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

# Hyperparameters for the Agent
MAX_MEMORY = 100_000 # Experience Replay Buffer Size 
BATCH_SIZE = 1000
LR = 0.001

class LinearQNet(nn.Module):
    """
    The Deep Q-Network (DQN) architecture.
    A simple Multi-Layer Perceptron (MLP) for Q(S,a) approximation.
    Input: 11-dim State Vector [cite: 32]
    Output: 3-dim Q-Values (one for each action) [cite: 34]
    """
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        
        # Build network layers dynamically based on project plan 
        layers = []
        current_size = input_size
        for size in hidden_sizes:
            layers.append(nn.Linear(current_size, size))
            layers.append(nn.ReLU()) # Use ReLU activation
            current_size = size
            
        # Output layer
        layers.append(nn.Linear(current_size, output_size))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class DQNAgent:
    def __init__(self, state_size, action_size, hidden_sizes=[128, 64]):
        self.state_size = state_size # Should be 11
        self.action_size = action_size # Should be 3
        self.n_games = 0
        self.gamma = 0.9 # Discount rate
        self.epsilon = 0 # Will be controlled in main training loop
        self.memory = deque(maxlen=MAX_MEMORY) # Experience Replay Buffer 

        # Q-Network (Predicts Q-values)
        self.q_net = LinearQNet(state_size, hidden_sizes, action_size)
        
        # Target Network (Frozen copy for stability) 
        self.target_net = LinearQNet(state_size, hidden_sizes, action_size)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval() # Set to evaluation mode

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=LR)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        """Store transitions (s, a, r, s') in a replay buffer."""
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self):
        """
        Trains the network on random mini-batches from the buffer 
        to de-correlate sequential data and stabilize learning[cite: 36].
        """
        if len(self.memory) < BATCH_SIZE:
            return

        # Get random mini-batch
        mini_sample = random.sample(self.memory, BATCH_SIZE)
        
        # Unzip the batch for vectorized processing
        states, actions, rewards, next_states, dones = zip(*mini_sample)

        states = torch.tensor(np.array(states), dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float)
        dones = torch.tensor(dones, dtype=torch.bool)

        # 1. Predicted Q values (Q(s, a)) from Q_net
        pred = self.q_net(states)

        # 2. Q_new = r + gamma * max(Q_target(s', a'))
        target = pred.clone()
        for idx in range(len(dones)):
            Q_new = rewards[idx]
            if not dones[idx]:
                # Detach the target net output since we don't want to backpropagate through it
                Q_new = rewards[idx] + self.gamma * torch.max(self.target_net(next_states[idx].unsqueeze(0)))
            
            # Update the Q value for the action that was actually taken
            target[idx][actions[idx].item()] = Q_new
            
        # Backpropagation
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
    
    def update_target_net(self):
        """Update the target network weights from the Q-network."""
        self.target_net.load_state_dict(self.q_net.state_dict())

    def get_action(self, state):
        """Epsilon-greedy action selection."""
        # Simple Epsilon decay strategy (will be refined in main loop)
        self.epsilon = 80 - self.n_games 
        final_move = 0 # Default: Go Straight

        if random.randint(0, 200) < self.epsilon:
            # Exploration
            final_move = random.randint(0, self.action_size - 1)
        else:
            # Exploitation
            state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            prediction = self.q_net(state_tensor)
            final_move = torch.argmax(prediction).item()
            
        return final_move