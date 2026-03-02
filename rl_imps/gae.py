
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
lam = 0.99

step_count = 0

n_eps_per_batch = 8

lr = 6e-2
lr_baseline = 6e-4
opt = optim.AdamW(policy.parameters(), lr=lr)
opt_baseline = optim.AdamW(baseline.parameters(), lr=lr_baseline)

viz = False
while step_count < n_steps:
    
    if step_count >= 95000 and not viz:
                env = gym.make(env_name, render_mode="human")
                obs, info = env.reset()
                viz = True
                
    act_log_probs = []
    
    total_cum_rewards = []
    
    predicted_values = torch.tensor([], dtype=torch.float32)
    advantages = torch.tensor([], dtype=torch.float32)
    
    eps_lens = []
    for i in range(n_eps_per_batch):
        terminated, truncated = False, False
        ep_rewards = []
        
        ep_obs = []

        step_before = step_count
        while not (terminated or truncated) and step_count < n_steps :
            action_dist = Categorical(logits = policy(tensor(obs, dtype=torch.float32)))
            action = action_dist.sample()
            act_log_probs.append(action_dist.log_prob(action))
            ep_obs.append(obs)
                    
            obs, reward, terminated, truncated, info = env.step(action.item())
            
            if truncated:
                ep_rewards.append(100)
            else:
                ep_rewards.append(reward)

            step_count += 1
        
        ep_len = step_count - step_before
        eps_lens.append(ep_len)
        
        ep_obs = torch.tensor(ep_obs, dtype=torch.float32)
        
        cum_rewards = ep_rewards[:]
        for i in reversed(range(len(cum_rewards) - 1)):
            cum_rewards[i] = cum_rewards[i + 1] * gamma + ep_rewards[i]
        
        total_cum_rewards += cum_rewards
        
        ep_rewards = torch.tensor(ep_rewards, dtype=torch.float32)
        
        eps_values = baseline(ep_obs).squeeze()
        
        predicted_values = torch.cat((predicted_values, eps_values))
        
        next_values = torch.cat((eps_values[1:], torch.tensor([0.0])))
        
        deltas = gamma * next_values + ep_rewards - eps_values 
    
        eps_advantage = torch.zeros(ep_len)
        gae = 0
        
        for i in reversed(range(ep_len)):
            gae = deltas[i] + gamma * lam * gae
            eps_advantage[i] = gae
        
        advantages = torch.cat((advantages, eps_advantage))
        if step_count >= n_steps:
            break
        obs, info = env.reset()

    
    
    act_log_probs = torch.stack(act_log_probs)
    
    total_cum_rewards = torch.tensor(total_cum_rewards, dtype=torch.float32)
    
    loss_baseline = F.mse_loss(predicted_values, total_cum_rewards)
    
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    loss = -(act_log_probs * advantages.detach()).mean()

    opt.zero_grad()
    loss.backward()
    opt.step()

    opt_baseline.zero_grad()
    loss_baseline.backward()
    opt_baseline.step()

    eps_lens_t = torch.tensor(eps_lens, dtype=torch.float32)
    print(f"policy_loss={loss:.4f}  mean_ep_len={eps_lens_t.mean():.1f}  max_ep_len={eps_lens_t.max():.0f}  std_ep_len={eps_lens_t.std():.1f}")

