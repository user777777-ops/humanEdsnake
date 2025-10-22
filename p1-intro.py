import gymnasium

# from stable_baselines3 import A2C

from stable_baselines3 import PPO
import numpy as np

# env=gymnasium.make("LunarLander-v2",render_mode="human")

env = gymnasium.make("LunarLander-v3", render_mode="human")

env.reset()

# print("sample action: ", env.action_space.sample())

# print("observation shape: ", env.observation_space.shape)

# print("observation sample: ", env.observation_space.sample())
# model = A2C("MlpPolicy", env, verbose=1)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

episodes = 10
for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        env.render()
        obs, reward, terminated, truncated, info = env.step((env.action_space.sample()))
        done = terminated or truncated
env.close()


#  16:30 on the https://www.youtube.com/watch?v=XbWhJdQgi7E
