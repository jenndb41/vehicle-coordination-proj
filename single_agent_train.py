import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
from final_env import *
from ray import tune
import pygame
import time
# from ray import tune
# from ray.tune import Stopper

ray.init()
env = MultiAgentCars()

config = PPOConfig().environment(env=MultiAgentCars)

config["lr"] = 0.005
algo = config.build()

max_episodes = 10
max_timesteps = 5000
current_episode = 0
iteration = 0
total_timesteps = 0
while current_episode < max_episodes and total_timesteps < max_timesteps:
    iteration+= 1
    print("Current Iteration:", iteration)
    observation = env.reset()
    done = {"__all__": False}

    while not done["__all__"]:
        results = algo.train()
        done["__all__"] = env.is_done(1) or env.is_done(2)
        total_timesteps += results["timesteps_total"]
        current_episode += 1
        print(pretty_print(results))
