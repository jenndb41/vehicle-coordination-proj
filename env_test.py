import gym
from newcar_duplicate.py import *
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

def main():
    env = CarEnv()
    env = DummyVecEnv(env)
    env = gym.make("env")
    model = PPO("MlpPolicy", env, verbose = 1)

    model.learn(timesteps = 10000)

    model.save("PPO_model")

    model = PPO.load("PPO_model")

    obs = env.reset()
