import gymnasium as gym
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import random

from collections import deque

env = gym.make("CartPole-v1", render_mode="human")

class DeepQNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, env.action_space.n)
        )
        
    def forward(self, x):
        x = self.layers(x)
        return F.softmax(x)
        
def lin_schedule(timestep):
    return start_e - (start_e - end_e) * timestep / total_timesteps

def sample_batch(buffer, batch_size):
    batch = torch.stack(random.sample(buffer, batch_size))
    import pdb; pdb.set_trace()
    
    return batch
    
# HYPERPARAMETERS/VARIABLES
start_e = 1
end_e = 0.9

total_timesteps = 1000

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# array to store past transitions
replay_buffer = deque()
replay_buffer_capacity = 600 

q_network = DeepQNetwork().to(device)

observation, info = env.reset(seed=3)

try:
    for step in range(total_timesteps):
        env.render()
            
        eps = lin_schedule(step)
        
        r = 0 if random.random() < eps else 1
        
        # r=0 for random action, r=1 for greedy action
        if r == 0:
            action = env.action_space.sample()
        else:
            out = q_network(torch.tensor(observation).to(device))
            action = torch.argmax(out)
        
        next_obs, reward, terminated, truncated, info = env.step(0)
        
        
        replay_buffer.append((observation, action, reward, next_obs))
        if len(replay_buffer) > replay_buffer_capacity:
            replay_buffer.popleft()
        
        # Model training
        sample = random.sample(replay_buffer, 1)
        
        print(sample)
        
        
        if terminated or truncated:
            print("Episode finished!")
            observation, info = env.reset()
finally:      
    env.close()