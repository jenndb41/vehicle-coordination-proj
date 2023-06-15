from ray.rllib.algorithms.ppo import PPOConfig
from ray import air
from ray import tune
import gym
from onecar_environment import *


config = PPOConfig()

#print(config.clip_param)

config.training(lr=tune.grid_search([0.001]), clip_param=0.2) 

config = config.environment(env=CarEnv)

tune.Tuner(
    "PPO",
    run_config=air.RunConfig(stop={"episode_reward_mean": 200}),
    param_space=config.to_dict(),).fit()
