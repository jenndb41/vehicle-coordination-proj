import ray
from ray.rllib.algorithms.ppo import PPO
from final_env import *
import pygame
import time


ray.init()
env = MultiAgentCars()
checkpoint_path = "/Users/jennyjung/Downloads/multi_agent_final/algo_ind_agents/checkpoint_000000"

config = {
    "multiagent": {
        "policies": {
            "policy_1": (None, env.observation_space, env.action_space, {}),
            "policy_2": (None, env.observation_space, env.action_space, {}),
        },
        "policy_mapping_fn": lambda agent_id: f"policy_{agent_id}",
    },
    "framework": "torch",
    "num_workers": 1,
}

trainer = PPO(config=config, env=MultiAgentCars)
trainer.restore(checkpoint_path)

done = {"__all__": False}
observation = env.reset()

while not done["__all__"]:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done["__all__"] = True

    actions = {}
    for agent_id in observation.keys():
        action = trainer.compute_single_action(observation[agent_id], policy_id=f"policy_{agent_id}")
        actions[agent_id] = action

    observation, reward, done, _ = env.step(actions)  
    env.render()
    time.sleep(0.1)

pygame.quit()
