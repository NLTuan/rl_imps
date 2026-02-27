
import torch
from torch import tensor, optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import gymnasium as gym

class PolicyGradientModelWithBaseline():
    def __init__(self, obs_dim, act_dim, n_hidden=32):
        self.policy = nn.Sequential(
            nn.Linear(obs_dim, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, act_dim)
        )
        
        self.baseline = nn.Sequential(
            nn.Linear(obs_dim, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1)
        )
        

env = gym.make('CartPole-v1', render_mode="rgb_array")

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n
policy_with_baseline = PolicyGradientModelWithBaseline(obs_dim, act_dim)
policy = policy_with_baseline.policy
baseline = policy_with_baseline.baseline

obs, info = env.reset()

n_steps = 300000

gamma = 0.99

step_count = 0

n_eps_per_batch = 8

lr = 6e-4
lr_baseline = 6e-4
opt = optim.AdamW(policy.parameters(), lr=lr)
opt_baseline = optim.AdamW(baseline.parameters(), lr=lr_baseline)
while step_count < n_steps:

    if step_count >= 295000:
        env = gym.make('CartPole-v1', render_mode="human")
        obs, info = env.reset()
    
    act_log_probs = []
    total_cum_rewards = []
    observations = []
    
    eps_lens = []
    for i in range(n_eps_per_batch):
        terminated, truncated = False, False
        ep_rewards = []

        step_before = step_count
        while not (terminated or truncated) and step_count < n_steps :
            action_dist = Categorical(logits = policy(tensor(obs)))
            action = action_dist.sample()
            act_log_probs.append(action_dist.log_prob(action))
            observations.append(obs)
                    
            obs, reward, terminated, truncated, info = env.step(action.item())
            
            if truncated:
                ep_rewards.append(100)
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
    
    baseline_values = baseline(torch.tensor(observations, dtype=torch.float32)).squeeze()
    
    total_cum_rewards = (total_cum_rewards - total_cum_rewards.mean()) / (total_cum_rewards.std() + 1e-8)
    loss = -(act_log_probs * (total_cum_rewards - baseline_values.detach())).mean()
    
    loss_baseline = F.mse_loss(baseline_values, total_cum_rewards)
    # import pdb; pdb.set_trace()

    loss.backward()
    opt.step()
    opt.zero_grad()
    
    loss_baseline.backward()
    opt_baseline.step()
    opt_baseline.zero_grad()
    # import pdb; pdb.set_trace()
    print(loss, end=" ")
    print(torch.tensor(eps_lens, dtype=torch.float32).mean(), end=' ')
    print(torch.tensor(eps_lens, dtype=torch.float32).max(), end = ' ')
    print(torch.tensor(eps_lens, dtype=torch.float32).std())

    
    
    
    
    
    
    