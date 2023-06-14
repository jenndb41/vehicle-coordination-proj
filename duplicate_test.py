import gym
from duplicate import *
import gym
import stable_baselines3

env = gym.make("env")
model = PPO("MlpPolicy", env, verbose = 1)
model.learn(timesteps = 10000)

obs = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, _, done, _ = self.step(*action)
    self.render()
