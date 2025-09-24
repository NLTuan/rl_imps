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

env = gym.make("LunarLander-v3", render_mode="rgb_array")

h_size=128
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
        x = self.layers(x)
        return x
    
    def get_best(self, state):
        with torch.no_grad():
            return torch.argmax(self.forward(state), dim=-1).item()
        
def lin_schedule(timestep):
    return start_e - (start_e - end_e) * timestep / total_timesteps

def sample_batch(buffer, batch_size):
    batch = torch.stack(random.sample(buffer, batch_size))
    import pdb; pdb.set_trace()
    
    return batch
    
# HYPERPARAMETERS/VARIABLES
batch_size = 128
min_replay_size = 1000

start_e = 1
end_e = 0

gamma = 0.99
tau = 0.001

total_timesteps = 100000
test_timesteps = 1000

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# array to store past transitions
replay_buffer = deque()
replay_buffer_capacity = 50000

q_network = DeepQNetwork().to(device)
target_network = DeepQNetwork().to(device)

copy_steps = 1 # how many steps before the q network weights are copied to the target's weights

observation, info = env.reset(seed=3)

lr = 2.5e-4
opt = optim.Adam(q_network.parameters(), lr=lr)
scheduler = lr_scheduler.CosineAnnealingLR(opt, T_max=total_timesteps)

try:
    reward_sum = 0
    for step in range(total_timesteps + test_timesteps):
        env.render()
            
        eps = lin_schedule(step)
        # print(eps)
        r = 0 if random.random() < eps else 1
        
        # Model sampling
        # r=0 for random action, r=1 for greedy action
        if r == 0:
            action = env.action_space.sample()
        else:
            action = q_network.get_best(torch.tensor(observation))

        next_obs, reward, terminated, truncated, info = env.step(action)
        reward_sum += reward
        
        
        replay_buffer.append((observation, action, reward, next_obs, terminated))
        if len(replay_buffer) > replay_buffer_capacity:
            replay_buffer.popleft()
        
        observation = next_obs

        
        if terminated or truncated :
            print("Episode finished!")
            print(f"Total reward: {reward_sum}")
            observation, info = env.reset()
            reward_sum = 0
        
        # Model training
        if len(replay_buffer) < min_replay_size:
            continue
        
        batch = random.sample(replay_buffer, batch_size)
        obs_samples, action_samples, reward_samples, next_obs_samples, terminated_samples = zip(*batch)
        obs_samples = torch.tensor(np.array(obs_samples), dtype=torch.float32).to(device)
        action_samples = torch.tensor(action_samples, dtype=torch.long).to(device)
        reward_samples = torch.tensor(reward_samples, dtype=torch.float32).to(device)
        next_obs_samples = torch.tensor(np.array(next_obs_samples), dtype=torch.float32).to(device)
        terminated_samples = torch.tensor(terminated_samples, dtype=torch.bool).to(device)
        
        with torch.no_grad():
            y = reward_samples + gamma * target_network(next_obs_samples).max(dim=-1).values * (~terminated_samples)
        

        actions_for_gather = action_samples.unsqueeze(1)
        q_values = q_network(obs_samples)
        predicted_q_values = q_values.gather(1, actions_for_gather).squeeze()
        loss = F.mse_loss(y,predicted_q_values)
        
        # if step > min_replay_size:
        #     print(loss)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(q_network.parameters(), max_norm=1.0)
        
        opt.step()
        opt.zero_grad()
        scheduler.step()
        
        if step % copy_steps == 0:
            for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                target_network_param.data.copy_(
                    tau * q_network_param.data + (1.0 - tau) * target_network_param.data
                )
        

finally:      
    env.close()
    torch.save(q_network.state_dict(), "q_network128.pth")