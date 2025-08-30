import gymnasium as gym
import time
import numpy as np

import torch
from torch import optim
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.nn.functional as F

import random

from collections import deque

device = 'cuda' if torch.cuda.is_available() else 'cpu'

h_size = 128
class DeepQNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], h_size),
            nn.ReLU(),
            nn.Linear(h_size, h_size),
            nn.ReLU(),
            nn.Linear(h_size, h_size),
            nn.ReLU(),
            nn.Linear(h_size, env.action_space.n)
        )
        

    def forward(self, x):
        return self.layers(x)

    def get_best(self, state):
        # This method is still useful for clean action selection
        with torch.no_grad():
            return torch.argmax(self.forward(state), dim=-1).item()


env = gym.make("CartPole-v1", render_mode="human")

q_network = DeepQNetwork()
q_network.load_state_dict(torch.load("q_network128.pth"))

q_network.eval()

num_episodes = 10
for i in range(num_episodes):
    observation, info = env.reset()
    terminated = False
    truncated = False
    total_reward = 0

    while not terminated and not truncated:
        # Render the environment so you can watch
        env.render()

        # CRITICAL: Use torch.no_grad() to disable gradient calculations.
        # This speeds up inference and reduces memory usage.
        with torch.no_grad():
            state_tensor = torch.tensor(observation, dtype=torch.float32).to(device)
            # No more epsilon-greedy, we always choose the best action
            action = q_network.get_best(state_tensor)

        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Add a small delay to make the visualization easier to watch
        time.sleep(0.01)

    print(f"Episode {i+1}: Finished with a total reward of {total_reward}")

env.close()