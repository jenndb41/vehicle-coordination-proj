from ray.rllib.env import BaseEnv
import gym
import numpy as np
import pygame
import random

WIDTH = 1920
HEIGHT = 1080
CELL_SIZE = 100
GRID_SIZE = (10, 10)

class MultiAgentEnv(gym.Env):
    def __init__(self, config = None):
        super(MultiAgentEnv, self).__init__()

        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)

        self.action_space = {0: 'turn_right', 1: 'turn_left', 2: 'up', 3: 'down'}
        self.observation_space = {i: str(i) for i in range(GRID_SIZE[0] * GRID_SIZE[1])}

        self.clock = pygame.time.Clock()
        self.game_map = pygame.image.load('GRID_MAP_01 copy 2.png').convert()
        self.grid = np.zeros(GRID_SIZE, dtype=np.uint8)

        self.car1 = Car((2, 3))
        self.car2 = Car((7, 7))

def reset(self):
    return {
        'agent1': self.car1.get_grid_cell(),
        'agent2': self.car2.get_grid_cell()
    }

    def step(self, actions):
        for agent, action in actions.items():
            if agent == 'agent1':
                if action == 0:
                    self.car1.turn_right()
                elif action == 1:
                    self.car1.turn_left()
                elif action == 2:
                    self.car1.up()
                else:
                    self.car1.down()
            elif agent == 'agent2':
                if action == 0:
                    self.car2.turn_right()
                elif action == 1:
                    self.car2.turn_left()
                elif action == 2:
                    self.car2.up()
                else:
                    self.car2.down()

        self.car1.update()
        self.car2.update()

        done = not (self.car1.is_alive() and self.car2.is_alive())
        rewards = {
            'agent1': self.car1.get_reward(),
            'agent2': self.car2.get_reward()
        }
        observations = {
            'agent1': self.car1.get_grid_cell(),
            'agent2': self.car2.get_grid_cell()
        }

        return observations, rewards, done, {}


    def render(self):
        self.screen.blit(self.game_map, (0, 0))
        self.car.draw(self.screen)
        #self.car_2.draw(self.screen)
        pygame.display.flip()
        pygame.time.wait(200)
        self.clock.tick(60)

    def observation_space(self, agent):
        return self.observation_space

    def action_space(self, agent):
        return self.action_space

class Car:
    def __init__(self, initial_position):
        self.position = initial_position

    def turn_right(self):
        self.position = (min(self.position[0] + 1, GRID_SIZE[0] - 1), self.position[1])

    def turn_left(self):
        self.position = (max(self.position[0] - 1, 0), self.position[1])

    def up(self):
        self.position = self.position = (self.position[0], min(self.position[1] + 1, GRID_SIZE[1] - 1))

    def down(self):
        self.position = (self.position[0], max(self.position[1] - 1, 0))

    def update(self):
        pass

    def is_alive(self):
        return True

    def get_reward(self):
        return 1

    def get_grid_cell(self):
        return self.position[0] * GRID_SIZE[1] + self.position[1]

    def draw(self, screen):
        cell_x = self.position[0] * CELL_SIZE
        cell_y = self.position[1] * CELL_SIZE
        pygame.draw.rect(screen, (255, 0, 0), (cell_x, cell_y, 140, 80))
'''
class Car_2:
    def __init__(self):
        self.position = (7, 7)

    def turn_right(self):
        self.position = (min(self.position[0] + 1, GRID_SIZE[0] - 1), self.position[1])

    def turn_left(self):
        self.position = (max(self.position[0] - 1, 0), self.position[1])

    def up(self):
        self.position = self.position = (self.position[0], min(self.position[1] + 1, GRID_SIZE[1] - 1))

    def down(self):
        self.position = (self.position[0], max(self.position[1] - 1, 0))

    def update(self):
        pass

    def is_alive(self):
        return True

    def get_reward(self):
        return 1

    def get_grid_cell(self):
        return self.position[0] * GRID_SIZE[1] + self.position[1]

    def draw(self, screen):
        cell_x = self.position[0] * CELL_SIZE
        cell_y = self.position[1] * CELL_SIZE
        pygame.draw.rect(screen, (0, 0, 0), (cell_x, cell_y, 140, 80))

env = CarEnv()
obs = env.render()

for _ in range(1000):
    action_first = random.choice(list(env.action_space.keys()))
    action_second = random.choice(list(env.action_space.values()))
    obs, reward, done, _ = env.step(action_first, action_second)
    env.render()

'''
