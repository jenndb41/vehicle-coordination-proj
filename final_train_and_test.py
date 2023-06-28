import ray
from ray.rllib.algorithms.ppo import PPOConfig
from final_env import *
from ray import tune
import pygame
import time

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
    )\
    .rollouts(num_rollout_workers=2)

algo = config.build()

for i in range(10):
    results = algo.train()
    print(results)
