import gym
import numpy as np
from gym import spaces
import pygame
from gym.utils import seeding

from pettingzoo import AECEnv
from pettingzoo.butterfly.pistonball.manual_policy import ManualPolicy
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn

__all__ = ["ManualPolicy", "env", "parallel_env", "raw_env"]


def env(**kwargs):
    env = CarEnv(**kwargs)
    return env


parallel_env = parallel_wrapper_fn(env)


class CarEnv(AECEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "midnight_oil",
        "is_parallelizable": True,
        "has_manual_policy": True,
    }

    def __init__(self, cars_n = 2):
        pygame.init()
        self.cars_n=cars_n
        self.screen_width=1920
        self.screen_height=1080
        self.cell_size=100
        self.grid_size=(10, 10)
        self.car_height=80
        self.car_width=140

        self.agents = ["car_" + str(r) for r in range(self.cars_n)]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.cars_n))))
        self._agent_selector = agent_selector(self.agents)
        self.observation_spaces = dict(zip(self.agents, [spaces.Discrete(self.grid_size[0] * self.grid_size[1]) for _ in range(self.cars_n)]))
        self.action_spaces = dict(zip(self.agents, [spaces.Discrete(4) for _ in range(self.cars_n)]))
        self.state_space = spaces.Discrete(self.grid_size[0] * self.grid_size[1])
        self.position = (0, 0)

        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.game_map = pygame.image.load('GRID_MAP_01 copy.png').convert()

        self.carList = []
        self.carRewards = []

        self.terminate = False
        self.truncate = False

        self.has_reset = False
        self.closed = False
        self._seed()

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def observe(self, agent):
        obs = self.get_grid_cell()
        return obs

    def enable_render(self):
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.screen.blit(self.game_map, (0, 0))
        pygame.display.flip()
        pygame.time.wait(200)
        self.clock = pygame.time.Clock()

        colors = [(255, 0, 0), (0, 0, 0)]
        self.agent_colors = dict(zip(self.agents, colors))

        for car, agent in zip(self.carList, self.agents):
            cell_x = self.position[0] * self.cell_size
            cell_y = self.position[1] * self.cell_size
            pygame.draw.rect(self.screen, self.agent_colors[agent], (cell_x, cell_y, self.car_width, self.car_height))

    def render(self, mode = "human"):
        if mode == "human":
            self.enable_render()
            self.clock.tick(60)
        elif mode == "rgb_array":
            # Return the current screen frame as an RGB array
            return pygame.surfarray.array3d(self.screen)

    def close(self):
        if not self.closed:
            self.closed = True
            if self.renderOn:
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
                self.renderOn = False
                pygame.event.pump()
                pygame.display.quit()

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._seed(seed)
            self.screen.blit(self.game_map, (0, 0))
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()

        self.has_reset = True
        self.terminate = False
        self.truncate = False
        self.rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self._cumulative_rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self.terminations = dict(zip(self.agents, [False for _ in self.agents]))
        self.truncations = dict(zip(self.agents, [False for _ in self.agents]))
        self.infos = dict(zip(self.agents, [{} for _ in self.agents]))

        self.agents = self.possible_agents[:]

        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()

        self.has_reset = True
        self.terminate = False
        self.truncate = False
        self.rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self._cumulative_rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self.terminations = dict(zip(self.agents, [False for _ in self.agents]))
        self.truncations = dict(zip(self.agents, [False for _ in self.agents]))
        self.infos = dict(zip(self.agents, [{} for _ in self.agents]))

    def get_grid_cell(self):
        return self.position[0] * self.grid_size[1] + self.position[1]

    def step(self, action):
        agent = self.agent_selection
        if action == 0:
            self.position = (min(self.position[0] + 1, self.grid_size[0] - 1), self.position[1])  # Go one cell unit up
        elif action == 1:
            self.position = (max(self.position[0] - 1, 0), self.position[1])  # Go one cell unit down
        elif action == 2:
            self.position = (self.position[0], min(self.position[1] + 1, self.grid_size[1] - 1))  # Go one cell unit right
        elif action == 3:
            self.position = (self.position[0], max(self.position[1] - 1, 0))  # Go one cell unit left

        self.agent_selection = self._agent_selector.next()
        return self.observe(agent), self._cumulative_rewards[agent], self.dones[agent], self.infos[agent]
