import ray
from ray.rllib.algorithms.ppo import PPO
from final_train_and_test import *
from final_env import *
import numpy as np
import pygame
import time

checkpoint = algo.save()
print(checkpoint)

restored_trainer = PPO.from_checkpoint(checkpoint)

env = MultiAgentCars()
done = {"__all__": False}
observation = env.reset()
agent_ids = [1, 2]

while not done["__all__"]:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done["__all__"] = True
    actions = {}
    for agent_id in agent_ids:
        actions[agent_id] = restored_trainer.compute_single_action(observation[agent_id], policy_id=f"policy_{agent_id}")
        obs, reward, done, info = env.step(actions)
        env.render()
        time.sleep(0.1)
