import gymnasium as gym
import time

import torch
from torch import nn, optim

from collections import deque

from dataclasses import dataclass

@dataclass
class DQNConfig:
    env_id: str
    lr: float
    
    epsilon: float
    gamma: float
    
    hidden: int = 64
    batch_size: int
    total_timesteps: int
    target_update: int
    buffer_size: int
    
    
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

replay_buffer = deque(maxlen=config.buffer_size)

observation, info = env.reset()

print(f"Starting observation: {observation}")

over = False
total_reward = 0
while not over:
    action = env.action_space.sample()
    time.sleep(0.2)
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
    print(f"Observation: {observation}")
    total_reward += reward

    
print(f"Total reward: {total_reward}")