import gymnasium as gym
import torch
from torch import nn, optim
import torch.nn.functional as F 
from torch.distributions import Categorical

from collections import deque

env = gym.make("CartPole-v1", render_mode='rgb_array')

n_hidden = 64

n_updates = 2000
n_rollouts_per_update = 32

gamma = 0.99

class LinearPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(env.observation_space.shape[0], n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, env.action_space.n)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
class ValueNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(env.observation_space.shape[0], n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

policy = LinearPolicy()
value_network = ValueNetwork()

lr = 3e-5
opt = optim.AdamW(list(policy.parameters()) + list(value_network.parameters()), lr=lr)

for i in range(n_updates):
    action_log_probs = []
    cum_rewards = deque()
    
    states = []
    episode_rewards = []
    for e in range(n_rollouts_per_update):
        observation, reward = env.reset()

        terminated = False
        truncated = False
        while not terminated and not truncated:
            probs = Categorical(torch.softmax(policy(torch.tensor(observation)), dim = -1))
            action = probs.sample()
            action_log_probs.append(probs.log_prob(action))
            states.append(torch.tensor(observation))
            observation, reward, terminated, truncated, info = env.step(action.item())
            if truncated:
                episode_rewards.append(100)
            else:
                episode_rewards.append(reward)
                        
        episode_returns = deque()
        for i in range(len(episode_rewards)-1, -1, -1):
            if i == len(episode_rewards)-1: 
                episode_returns.appendleft(episode_rewards[i])
                continue
            episode_returns.appendleft(episode_rewards[i] + episode_returns[0] * gamma)
        
        cum_rewards.extend(episode_returns)
        episode_rewards = []
        
    cum_rewards = torch.tensor(cum_rewards)
    action_log_probs = torch.stack(action_log_probs)
    states_tensor = torch.stack(states)
    
    baseline = value_network(states_tensor)
    
    value_loss = F.mse_loss(baseline.squeeze(), cum_rewards)
    advantage = cum_rewards - baseline.squeeze()

    norm_advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-09)
    
    policy_loss = (-action_log_probs * norm_advantage.detach()).sum()

    loss = value_loss + policy_loss
    print(loss)
    loss.backward()
    opt.step()
    opt.zero_grad()


env.close()
torch.save(policy.state_dict(), 'a2c.pth')