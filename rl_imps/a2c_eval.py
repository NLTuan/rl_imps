import gymnasium as gym
import torch
from torch import nn
import torch.nn.functional as F 
import time

# --- IMPORTANT: The Policy Network class must be identical to the one used for training ---
# This class defines the "Actor" part of your A2C agent.
n_hidden = 64
class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Using 4 directly for CartPole's state size (cart position, cart velocity, pole angle, pole angular velocity)
        self.fc1 = nn.Linear(4, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        # Using 2 directly for CartPole's action size (move left, move right)
        self.fc3 = nn.Linear(n_hidden, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# --- PARAMETERS ---
# Path to the model file saved by your training script
MODEL_PATH = "a2c.pth" 
# Number of episodes to run for evaluation
N_EVAL_EPISODES = 10

# --- SETUP ---
# Initialize the environment with rendering so we can watch
env = gym.make("CartPole-v1", render_mode='human')

# Initialize the policy network (the Actor)
policy = PolicyNetwork()

# Load the trained weights from the file
try:
    policy.load_state_dict(torch.load(MODEL_PATH))
except FileNotFoundError:
    print(f"Error: Model file not found at '{MODEL_PATH}'")
    print("Please run the training script first to generate the model file.")
    exit()

# Set the network to evaluation mode. This is important for layers like Dropout or BatchNorm,
# and it's a good practice to always include it during inference.
policy.eval()

print(f"Loaded trained model from {MODEL_PATH}. Running {N_EVAL_EPISODES} episodes.")

# --- EVALUATION LOOP ---
for episode in range(N_EVAL_EPISODES):
    # Reset the environment for a new episode
    observation, _ = env.reset()
    total_reward = 0
    terminated = False
    truncated = False

    while not terminated and not truncated:
        # --- Take the BEST action (no exploration) ---
        
        # Disable gradient calculation for efficiency, as we are not training
        with torch.no_grad():
            # Get the raw output (logits) from the policy network
            logits = policy(torch.tensor(observation, dtype=torch.float32))
        
        # Choose the action with the highest logit value (the most likely action)
        action = torch.argmax(logits).item()

        # Take a step in the environment with the chosen action
        observation, reward, terminated, truncated, info = env.step(action)
        
        # Accumulate the reward
        total_reward += reward

        # Optional: add a small delay to make the rendering easier to watch
        time.sleep(0.01)

    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

# Clean up the environment
env.close()