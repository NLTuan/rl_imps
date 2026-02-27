
import torch
from torch import tensor, optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import gymnasium as gym

class PolicyGradientModel(nn.Module):
    def __init__(self, obs_dim, act_dim, n_hidden=32):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, act_dim)
        
    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return self.fc3(h2)
        

env = gym.make('CartPole-v1', render_mode="rgb_array")

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n
policy = PolicyGradientModel(obs_dim, act_dim)

obs, info = env.reset()

n_steps = 300000

gamma = 0.99

step_count = 0

n_eps_per_batch = 8

lr = 6e-4
opt = optim.AdamW(policy.parameters(), lr=lr)
while step_count < n_steps:

    if step_count >= 295000:
        env = gym.make('CartPole-v1', render_mode="human")
        obs, info = env.reset()
    
    act_log_probs = []
    total_cum_rewards = []
    
    eps_lens = []
    for i in range(n_eps_per_batch):
        terminated, truncated = False, False
        ep_rewards = []

        step_before = step_count
        while not (terminated or truncated) and step_count < n_steps :
            action_dist = Categorical(logits = policy(tensor(obs)))
            action = action_dist.sample()
            act_log_probs.append(action_dist.log_prob(action))
                    
            obs, reward, terminated, truncated, info = env.step(action.item())
            
            if truncated:
                ep_rewards.append(50)
            else:
                ep_rewards.append(reward)

            step_count += 1
        
        eps_lens.append(step_count - step_before)
        
        cum_rewards = ep_rewards[:]
        for j in range(len(ep_rewards)):
            if j != 0:
                cum_rewards[-(j+1)] = gamma * cum_rewards[-j] + ep_rewards[-(j+1)]
        
        total_cum_rewards += cum_rewards
        
        if step_count >= n_steps:
            break
        obs, info = env.reset()

    
    act_log_probs = torch.stack(act_log_probs)
    total_cum_rewards = torch.tensor(total_cum_rewards, dtype=torch.float32)
    
    total_cum_rewards = (total_cum_rewards - total_cum_rewards.mean()) / (total_cum_rewards.std() + 1e-8)
    loss = -(act_log_probs * total_cum_rewards).mean()
    
    loss.backward()
    opt.step()
    opt.zero_grad()
    # import pdb; pdb.set_trace()
    print(loss, end=" ")
    print(torch.tensor(eps_lens, dtype=torch.float32).mean(), end=' ')
    print(torch.tensor(eps_lens, dtype=torch.float32).max())

    
    
    
    
    
    
    