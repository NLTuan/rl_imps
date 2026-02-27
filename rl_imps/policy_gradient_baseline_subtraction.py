
import torch
from torch import tensor, optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import gymnasium as gym

class PolicyGradientModelWithBaseline():
    def __init__(self, obs_dim, act_dim, n_hidden=64):
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
        

env_name = "CartPole-v1"
env = gym.make(env_name, render_mode="rgb_array")

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n
policy_with_baseline = PolicyGradientModelWithBaseline(obs_dim, act_dim)
policy = policy_with_baseline.policy
baseline = policy_with_baseline.baseline

obs, info = env.reset()

n_steps = 100000

gamma = 0.99

step_count = 0

n_eps_per_batch = 8

lr = 6e-3
lr_baseline = 6e-4
opt = optim.AdamW(policy.parameters(), lr=lr)
opt_baseline = optim.AdamW(baseline.parameters(), lr=lr_baseline)

viz = False
while step_count < n_steps:
    
    act_log_probs = []
    total_cum_rewards = []
    observations = []
    
    eps_lens = []
    for i in range(n_eps_per_batch):
        terminated, truncated = False, False
        ep_rewards = []

        step_before = step_count
        while not (terminated or truncated) and step_count < n_steps :
            action_dist = Categorical(logits = policy(tensor(obs, dtype=torch.float32)))
            action = action_dist.sample()
            act_log_probs.append(action_dist.log_prob(action))
            observations.append(obs)
                    
            obs, reward, terminated, truncated, info = env.step(action.item())
            
            if truncated:
                ep_rewards.append(100)
            else:
                ep_rewards.append(reward)

            step_count += 1
            
            if step_count >= 95000 and not viz:
                env = gym.make(env_name, render_mode="human")
                obs, info = env.reset()
                viz = True
        
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

    loss_baseline = F.mse_loss(baseline_values, total_cum_rewards)

    total_cum_rewards = (total_cum_rewards - total_cum_rewards.mean()) / (total_cum_rewards.std() + 1e-8)
    loss = -(act_log_probs * (total_cum_rewards - baseline_values.detach())).mean()

    opt.zero_grad()
    loss.backward()
    opt.step()

    opt_baseline.zero_grad()
    loss_baseline.backward()
    opt_baseline.step()

    eps_lens_t = torch.tensor(eps_lens, dtype=torch.float32)
    print(f"policy_loss={loss:.4f}  mean_ep_len={eps_lens_t.mean():.1f}  max_ep_len={eps_lens_t.max():.0f}  std_ep_len={eps_lens_t.std():.1f}")

    
    
    
    
    
    
    