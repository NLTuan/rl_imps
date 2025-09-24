import gymnasium as gym
import torch
from torch import nn
import torch.nn.functional as F 
import time

# --- IMPORTANT: The model class must be identical to the one used for training ---
n_hidden = 64
class LinearPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, n_hidden) # Using 4 directly for CartPole state size
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, 2) # Using 2 directly for CartPole action size

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# --- PARAMETERS ---
MODEL_PATH = "policy_grad.pth"
N_EVAL_EPISODES = 5

# --- SETUP ---
# Initialize the environment with rendering
env = gym.make("CartPole-v1", render_mode='human')

# Initialize the policy network
policy = LinearPolicy()

# Load the trained weights
policy.load_state_dict(torch.load(MODEL_PATH, weights_only=False))

# Set the network to evaluation mode. This is important for some layers like Dropout.
policy.eval()

print(f"Loaded trained model from {MODEL_PATH}. Running {N_EVAL_EPISODES} episodes.")

# --- EVALUATION LOOP ---
for episode in range(N_EVAL_EPISODES):
    observation, _ = env.reset()
    total_reward = 0
    terminated = False
    truncated = False

    while not terminated and not truncated:
        # --- Take the BEST action (no exploration) ---
        # We don't need softmax or Categorical for evaluation, just the raw output (logits)
        with torch.no_grad(): # Disable gradient calculation for efficiency
            logits = policy(torch.tensor(observation, dtype=torch.float32))
        
        # Choose the action with the highest logit value
        action = torch.argmax(logits).item()

        # Take a step in the environment
        observation, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward

        # Optional: add a small delay to make it easier to watch
        time.sleep(0.01)

    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

env.close()