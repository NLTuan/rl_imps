import gymnasium as gym
import time

import torch
from torch import nn, optim
import torch.nn.functional as F

from collections import deque
import random

from dataclasses import dataclass

@dataclass
class DQNConfig:
    env_id: str
    lr: float
    
    epsilon: float
    gamma: float
    
    hidden: int = 64
    batch_size: int = 32
    total_timesteps: int = 1000
    target_update: int = 10
    buffer_size: int = 1000
    
    
class DeepQNetwork(nn.Module):
    def __init__(self, env, hidden):
        super().__init__()
        self.fc1 = nn.Linear(env.observation_space.shape[0], hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, env.action_space.n)
        
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


config = DQNConfig(
    env_id='CartPole-v1',
    lr=1e-3,
    epsilon=0.1,
    gamma=0.99,
    batch_size=64,
    total_timesteps=10000,
    target_update=10,
    buffer_size=10000
)

env = gym.make(config.env_id, render_mode='human')

q_network = DeepQNetwork(env, hidden=config.hidden)
target_network = DeepQNetwork(env, hidden=config.hidden)
target_network.load_state_dict(q_network.state_dict())

optimizer = optim.Adam(q_network.parameters(), lr=config.lr)

replay_buffer = deque(maxlen=config.buffer_size)

observation, info = env.reset()

print(f"Starting observation: {observation}")
over = False
for i in range(config.total_timesteps):
    # SAMPLING STEP    
    # Epsilon-greedy action selection
    if random.random() < config.epsilon:
        action = env.action_space.sample()
    else:
        action = q_network(torch.tensor(observation, dtype=torch.float32)).argmax().item()
    
    # Perform step in the simulation
    observation, reward, terminated, truncated, info = env.step(action)
    
    # Store transition in replay buffer
    replay_buffer.append((observation, action, reward, terminated or truncated))
    
    if terminated or truncated:
        observation, info = env.reset()
    
    # TRAINING STEP
    batch = random.sample(replay_buffer, min(len(replay_buffer), config.batch_size)) 
    
    y = torch.zeros(len(batch))
    
    for i in range(len(batch)):
        y[i] = batch[i][2] if batch[i][3] else  batch[i][2] + config.gamma * target_network(torch.tensor(batch[i][0], dtype=torch.float32)).max().item()

    print(y.shape)
    observations = torch.tensor([batch[i][0] for i in range(len(batch))], dtype=torch.float32)
    loss = F.mse_loss(target_network(observations).max(dim=1).values, y)
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    if i % config.target_update == 0:
        target_network.load_state_dict(q_network.state_dict())
    
    print(f"Loss: {loss.item()}")
    
print(f"Total reward: {total_reward}")