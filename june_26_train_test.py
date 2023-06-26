import ray
from ray.rllib.algorithms.ppo import PPOConfig
from final_env import *
from ray import tune
import pygame
import time

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
    ).evaluation(
    "evaluation_config": {
        "env_config": {
            "render_env": True
        }
            ),
    ).build()

print(algo.train())

done = {"__all__": False}
observation = env.reset()

while not done["__all__"]:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done["__all__"] = True

    actions = {}
    for agent_id in observation.keys():
       # policy_id = self.config['multiagent']['policy_mapping_fn'](agent_id)
        action = algo.compute_single_action(observation[agent_id], policy_id=f"policy_{agent_id}")
        actions[agent_id] = action

    observation, reward, done, _ = env.step(actions)
    print(observation)
    print(reward)

    if done["__all__"]:
        break

    env.render()
    time.sleep(0.1)

pygame.quit()
