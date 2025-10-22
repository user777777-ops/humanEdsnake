import gymnasium

from stable_baselines3 import PPO, A2C
import numpy as np

env = gymnasium.make("LunarLander-v3", render_mode="human")

env.reset()

models_dir = "models/PPO"
model_path = f"{models_dir}/120000.zip"
# 120000 is the best model for me according to the tensorboard.
model = PPO.load(model_path, env=env)

episodes = 10
for ep in range(episodes):
    obs, info = env.reset()
    done = False
    while not done:
        env.render()
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
env.close()
# 20:58 video https://www.youtube.com/watch?v=dLP-2Y6yu70&list=PLQVvvaa0QuDf0O2DWwLZBfJeYY-JOeZB1&index=2
