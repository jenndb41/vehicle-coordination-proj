import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
from final_env import *
from ray import tune
import pygame
import time
from ray import tune
from ray.tune import Stopper

ray.init()
env = MultiAgentCars()

config = PPOConfig()\
    .environment(env=MultiAgentCars)\
    .multi_agent(
        policies={
            "policy_1": (
                None, env.observation_space, env.action_space, {"gamma": 0.80}
            ),
            "policy_2": (
                None, env.observation_space, env.action_space, {"gamma": 0.95}
            ),
        },
        policy_mapping_fn = lambda agent_id: f"policy_{agent_id}",
    )

config["lr"] = 0.005
algo = config.build()

max_episodes = 10
max_timesteps = 5000
current_episode = 0
iteration = 0
while current_episode < max_episodes:
    iteration+= 1
    print("Current Iteration:", iteration)
    observation = env.reset()
    done = {"__all__": False}
    timesteps = 0

    while not done["__all__"] and timesteps < max_timesteps:
        results = algo.train()
        done["__all__"] = env.is_done(1) or env.is_done(2)
        print(pretty_print(results))
        timesteps += 1

    current_episode += 1
