from ray.rllib.env.multi_agent_env import MultiAgentEnv
import gym
from gym.spaces import Discrete
import numpy as np
import pygame
import time



WIDTH = 1920
HEIGHT = 1080
CELL_SIZE = 100
GRID_SIZE = (11, 11)

class MultiAgentCars(MultiAgentEnv):
    def __init__(self, seeker = None, *args, **kwargs):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
        self.game_map = pygame.image.load('FINAL_GRID_MAP.png').convert()
        self.action_space = Discrete(4)
        self.observation_space = Discrete(GRID_SIZE[0] * GRID_SIZE[1])
        self.clock = pygame.time.Clock()
        self.agents = {1: (1, 2), 2: (8, 8)}
        self.goal = (4, 4)
        self.info = {1: {'obs': self.agents[1]}, 2: {'obs': self.agents[2]}}
        self.grid = np.zeros(GRID_SIZE, dtype=np.uint8)
        self.punish_states = []
        for row in range(GRID_SIZE[1]):
            for col in range(GRID_SIZE[0]):
                coordinate = (col, row)
                if (
                    (col in range(2, 10) and (row == 0 or row == 9)) or
                    (coordinate == (2, 1) or coordinate == (8, 1)) or
                    (row in range(2, 4) and col in range(1, 10)) or
                    (row in range(2, 9) and (col == 1 or col == 9)) or
                    (coordinate == (2, 9) or coordinate == (8, 9))
                ):
                    continue
                self.punish_states.append(coordinate)


    def reset(self):
        self.agents = {1: (1, 2), 2: (8, 8)}
        return {1: self.get_observation(1), 2: self.get_observation(2)}

    def get_observation(self, agent_id):
        seeker = self.agents[agent_id]
        return 11 * seeker[0] + seeker[1]

    def get_reward(self, agent_id):
        reward = -1 if self.agents[agent_id] in self.punish_states else 0
        for next_agent_id, next_seeker in self.agents.items():
            reward -= 1 if agent_id != next_agent_id and self.agents[agent_id] == next_seeker else 0
        return reward

    def is_done(self, agent_id):
        for next_agent_id, next_seeker in self.agents.items():
            return agent_id != next_agent_id and self.agents[agent_id] == next_seeker

    def step(self, action):
        agent_ids = action.keys()

        for agent_id in agent_ids:
            seeker = self.agents[agent_id]
            if action[agent_id] == 0: #down
                seeker = (seeker[0], min(seeker[1] + 1, 10))
            elif action[agent_id] == 1: #left
                seeker = (max(seeker[0] - 1, 0), seeker[1])
            elif action[agent_id] == 2: #up
                seeker = (seeker[0], max(seeker[1] - 1, 0))
            elif action[agent_id] == 3: #right
                seeker = (min(seeker[0] + 1, 10), seeker[1])
            else:
                raise ValueError("Invalid action")
            self.agents[agent_id] = seeker

        observations = {i: self.get_observation(i) for i in agent_ids}
        rewards = {i: self.get_reward(i) for i in agent_ids}
        done = {i: self.is_done(i) for i in agent_ids}

        done["__all__"] = all(done.values())

        return observations, rewards, done, self.info

    def render(self):
        self.screen.blit(self.game_map, (0, 0))

        for agent_id, position in self.agents.items():
            agent_x, agent_y = position
            pygame.draw.rect(
                self.screen,
                (255, 0, 0),
                (
                    agent_x * (WIDTH // 11),
                    agent_y * (HEIGHT // 11),
                    WIDTH // 11,
                    HEIGHT // 11,
                ),
            )

        pygame.display.flip()
        self.clock.tick(60)


env = MultiAgentCars()
'''
while True:
    obs, rew, done, info = env.step(
        {1: env.action_space.sample(), 2: env.action_space.sample()}
    )
    time.sleep(0.1)
    env.render()
    if any(done.values()):
        break
'''

'''
env = MultiAgentCars()

if __name__ == "__main__":
    env = MultiAgentCars()

    done = {"__all__": False}
    while not done["__all__"]:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done["__all__"] = True

        # Perform your RL algorithm's action selection
        action = {1: env.action_space.sample(), 2: env.action_space.sample()}

        # Step the environment
        observations, rewards, done, _ = env.step(action)

        # Render the environment
        env.render()

    pygame.quit()'''
