import gymnasium
from snakeenv import SnekEnv

# switch every PPO for A2C to try another policy.
from stable_baselines3 import PPO
import numpy as np
import os
import time

models_dir = f"models/{int(time.time())}"
logdir = f"logs/{int(time.time())}"
model_path = "models/1763722025/3840000.zip"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)
env = SnekEnv()
env.reset()
model = PPO.load(model_path, env=env, tensorboard_log=logdir)
TIMESTEPS = 10000
for i in range(100000000000):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}/{TIMESTEPS*i}")

env.close()

# video 3 on minute 35.
