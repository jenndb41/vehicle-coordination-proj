from ray import tune
from ray.rllib.algorithms.ppo import PPO
from independent_agents_env import *

tune.run(
    PPO,
    stop={"episode_reward_mean": 200},
    config={
        "env": MultiAgentEnv,
        "lr": 0.001,
        "clip_param": 0.2,
    },
    num_samples=1
)
