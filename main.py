import gymnasium as gym

env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0, enable_wind=False, wind_power=15.0, turbulence_power=1.5, render_mode="human")
obs, info = env.reset(seed=42)

for _ in range(1000):
    action = env.action_space.sample()
    
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
        
env.close()
