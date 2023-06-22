from ray import tune
from ray.rllib.algorithms.ppo import PPO
import math
import time
import gym
import numpy as np
import pygame
import random

WIDTH = 1920
HEIGHT = 1080
CELL_SIZE = 100
GRID_SIZE = (10, 10)

class MultiAgentEnv(gym.Env):
    def __init__(self, config=None):
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
        self.car1.draw(self.screen)
        self.car2.draw(self.screen)
        pygame.display.flip()
        pygame.time.wait(200)
        self.clock.tick(60)

    def observation_space(self):
        return self.observation_space

    def action_space(self):
        return self.action_space

class Car:
    def __init__(self, initial_position):
        self.position = initial_position

    def turn_right(self):
        self.position = (min(self.position[0] + 1, GRID_SIZE[0] - 1), self.position[1])

    def turn_left(self):
        self.position = (max(self.position[0] - 1, 0), self.position[1])

    def up(self):
        self.position = (self.position[0], min(self.position[1] + 1, GRID_SIZE[1] - 1))

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

def boundary_penalty(x, y, road_corners):
    if road_corners["xmin"] <= x <= road_corners["xmax"] and y in (range(2, 3) or range(7, 8)):
        return 0
    elif road_corners["xmin"] <= x <= road_corners["xmax"] and (x, y) in bypass_pos:
        return 0
    elif road_corners["ymin"] <= y <= road_corners["ymax"] and x == (0 or 9):
        return 0
    else:
        return -1


def training_function(config):
    x1, y1 = config["car1_x"], config["car1_y"]
    x2, y2 = config["car2_x"], config["car2_y"]
    score = objective(x1, y1, x2, y2)
    tune.report(score=score)

def objective(x1, y1, x2, y2):
    distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    score = 0
    if distance > 0:
        score += 1
    else:
        score -= 1
    score += boundary_penalty(x1, y1, road_corners) + boundary_penalty(x2, y2, road_corners)

    return score

road_corners = {
    "xmin": 1,
    "xmax": 9,
    "ymin": 2,
    "ymax": 8
}

bypass_pos = [(2, 1), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (8, 1), (2, 9), (2, 10), (3, 10), (4, 10), (5, 10), (6, 10), (7, 10), (8, 10), (8, 9)]

config = {
    "car1_x": tune.grid_search(list(range(11))),
    "car1_y": tune.grid_search(list(range(11))),
    "car2_x": tune.grid_search(list(range(11))),
    "car2_y": tune.grid_search(list(range(11)))
}

result = tune.run(
    training_function,
    config=config,
    num_samples=1,
    verbose=1,
    stop={"training_iteration": 1}
)

best_config = result.get_best_config(metric="score", mode="max")
print(best_config)

trainer = PPO(config=best_config)

env = MultiAgentEnv()

for _ in range(5):
    observations = env.reset()
    done = False
    while not done:
        actions = {}
        for agent, observation in observations.items():
            action = trainer.compute_action(observation)
            actions[agent] = action
        next_observations, rewards, done, _ = env.step(actions)
        trainer.learn(
            observations,
            actions,
            rewards,
            next_observations,
            done
        )
        observations = next_observations
