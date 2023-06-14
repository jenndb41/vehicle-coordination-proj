import gym
from newcar_duplicate import *
import gym
from stable_baselines3 import PPO

def main():
    env = gym.make('newcar_duplicate')
    print("preparing to train")
    model = PPO("MlpPolicy", env, verbose = 1)

    print("training the agent")

    model.learn(timesteps = 10000)

    model.save("Downloads/ai-car-simulation-master/PPO_model")

    if __name__ == "__main__":
        main()
