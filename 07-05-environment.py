from ray.rllib.env.multi_agent_env import MultiAgentEnv
import gym
from gym.spaces import Discrete
import numpy as np
import pygame
import time



WIDTH = 1920
HEIGHT = 1080
CELL_SIZE = 30
GRID_SIZE = (11, 11)

class MultiAgentCars(MultiAgentEnv):
    def __init__(self, seeker = None, *args, **kwargs):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
        self.game_map = pygame.image.load('FINAL_GRID_MAP.png').convert()
        self.action_space = Discrete(5)
        self.observation_space = Discrete(GRID_SIZE[0] * GRID_SIZE[1])
        self.clock = pygame.time.Clock()
        self.agents = {1: (5, 2), 2: (5, 8)}
        self.info = {1: {'obs': self.agents[1]}, 2: {'obs': self.agents[2]}}
        self.grid = np.zeros(GRID_SIZE, dtype=np.uint8)
        self.punish_states = []
        self.up_down = [(2, 1), (8, 1), (2, 9), (8, 9), (1, 4), (1, 5), (1, 6), (9, 4), (9, 5), (9, 6)]
        self.up_down_right = [(1, 3), (1, 7)]
        self.up_down_left = [(9, 3), (9, 7)]
        self.up_right_left = [(3, 8), (4, 8), (5, 8), (6, 8), (7, 8), (2, 3), (3, 3), (4, 3), (5, 3), (6, 3), (7, 3), (8, 3)]
        self.up_right = [(1, 8), (2, 10)]
        self.up_left = [(9, 8), (8, 10)]
        self.down_right_left = [(2, 7), (3, 7), (4, 7), (5, 7), (6, 7), (7, 7), (8, 7), (3, 2), (4, 2), (5, 2), (6, 2), (7, 2)]
        self.left_right = [(3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (3, 10), (4, 10), (5, 10), (6, 10), (7, 10)]
        self.down_right = [(1, 2), (2, 0)]
        self.down_left = [(9, 2), (8, 0)]
        self.upper_bypass = [(2, 0), (2, 1), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (8, 1)]
        self.lower_bypass = [(2, 9), (2, 10), (3, 10), (4, 10), (5, 10), (6, 10), (7, 10), (8, 10), (8, 9)]
        self.cumulative_rewards = {1: 0, 2: 0}
        self.env_timestep = 0

        self.last_actions = {}
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
        self.env_timestep = 0
        self.agents = {1: (5, 2), 2: (5, 8)}
        return {1: self.get_observation(1), 2: self.get_observation(2)}

    def get_observation(self, agent_id):
        seeker = self.agents[agent_id]
        return 11 * seeker[0] + seeker[1]

    def get_reward(self, agent_id):
        self.cumulative_rewards[agent_id] = -20 if self.agents[agent_id] in self.punish_states else 0
        # agent 1 is going counterclockwise
        if agent_id == 1:
            if 2 <= self.agents[agent_id][0] <= 9 and 2 <= self.agents[agent_id][1] <= 3: # reward left/ col 2-9
                if self.agents[agent_id] == [self.agents[agent_id][0] - 1, 2] or self.agents[agent_id] == [self.agents[agent_id][0] - 1, 3]:
                    self.cumulative_rewards[agent_id] +=10
            elif 7 <= self.agents[agent_id][1] <= 8 and 1 <= self.agents[agent_id][0] <= 8: #reward right/ col 1-8
                if self.agents[agent_id] == [self.agents[agent_id][0] + 1, 7] or self.agents[agent_id] == [self.agents[agent_id][0] + 1, 8]:
                    self.cumulative_rewards[agent_id] +=10
            elif 3 <= self.agents[agent_id][1] <= 8 and self.agents[agent_id][0] == 9: #reward up/ col 9
                if self.agents[agent_id] == [9, self.agents[agent_id][1] - 1]:
                    self.cumulative_rewards[agent_id] +=10
            elif 2 <= self.agents[agent_id][1] <= 7 and self.agents[agent_id][0] == 1: #reward down/ col 1
                if self.agents[agent_id] == [1, self.agents[agent_id][1] + 1]:
                    self.cumulative_rewards[agent_id] +=10
        # agent 2 is going clockwise
        elif agent_id == 2:
            if 2 <= self.agents[agent_id][0] <= 9 and 7 <= self.agents[agent_id][1] <= 8: #reward left/ cols 2-9
                if self.agents[agent_id] == [self.agents[agent_id][0] - 1, 7] or self.agents[agent_id] == [self.agents[agent_id][0] - 1, 8]:
                    self.cumulative_rewards[agent_id] +=10
            elif 1 <= self.agents[agent_id][0] <= 8 and 2 <= self.agents[agent_id][1] <= 3: #reward right/ cols 1-8
                if self.agents[agent_id] == [self.agents[agent_id][0] + 1, 2] or self.agents[agent_id] == [self.agents[agent_id][0] + 1, 3]:
                    self.cumulative_rewards[agent_id] +=10
            elif 2 <= self.agents[agent_id][1] <= 7 and self.agents[agent_id][0] == 9: #reward down/ col 9
                if self.agents[agent_id] == [9, self.agents[agent_id][1] + 1]:
                    self.cumulative_rewards[agent_id] +=10
            elif 3 <= self.agents[agent_id][1] <= 8 and self.agents[agent_id][0] == 1: #reward up/ col 1
                if self.agents[agent_id] == [1, self.agents[agent_id][1] - 1]:
                    self.cumulative_rewards[agent_id] +=10

        # collision avoidance
        if agent_id == 1:
            right_coord = [self.agents[agent_id][0] + 1, self.agents[agent_id][1]]
            left_coord = [self.agents[agent_id][0] - 1, self.agents[agent_id][1]]
            above_coord = [self.agents[agent_id][0], self.agents[agent_id][1] - 1]
            below_coord = [self.agents[agent_id][0], self.agents[agent_id][1] + 1]
            upper_right = [self.agents[agent_id][0] + 1, self.agents[agent_id][1] - 1]
            lower_right = [self.agents[agent_id][0] + 1, self.agents[agent_id][1] + 1]
            upper_left = [self.agents[agent_id][0] - 1, self.agents[agent_id][1] - 1]
            lower_left = [self.agents[agent_id][0] - 1, self.agents[agent_id][1] + 1]

            if right_coord in self.agents.values() and right_coord != self.agents[agent_id]:
                if 2 <= self.agents[agent_id][1] <= 3:
                    if self.agents[agent_id][0] == 1:
                        if self.agents[agent_id] == [self.agents[agent_id][0], self.agents[agent_id][1] + 1]:
                            self.cumulative_rewards[agent_id] +=30
                    elif 2 <= self.agents[agent_id][0] <= 8:
                        if self.agents[agent_id] == [self.agents[agent_id][0] - 1, self.agents[agent_id][1]]:
                            self.cumulative_rewards[agent_id] +=30
                elif 7 <= self.agents[agent_id][1] <= 8:
                    if self.agents[agent_id][1] == 8:
                        if self.agents[agent_id] == [self.agents[agent_id][0], self.agents[agent_id][1] - 1]:
                            self.cumulative_rewards[agent_id] +=30
                    elif self.agents[agent_id][1] == 7:
                        if self.agents[agent_id] == [self.agents[agent_id][0], self.agents[agent_id][1] + 1]:
                            self.cumulative_rewards[agent_id] +=30
            elif lower_right in self.agents.values() and lower_right != self.agents[agent_id]:
                if self.agents[agent_id][1] == 2 and 2 <= self.agents[agent_id][0] <= 8:
                    if self.agents[agent_id] == [self.agents[agent_id][0] - 1, self.agents[agent_id][1]]:
                        self.cumulative_rewards[agent_id] +=30 #checked
                elif self.agents[agent_id][1] == 2 and self.agents[agent_id][0] == 1:
                    if self.agents[agent_id] == [self.agents[agent_id][0], self.agents[agent_id][1] + 1]:
                        self.cumulative_rewards[agent_id] +=30 #checked
                elif self.agents[agent_id][1] == 7 and 1 <= self.agents[agent_id][0] <= 8:
                    if self.agents[agent_id] == [self.agents[agent_id][0] + 1, self.agents[agent_id][1]]:
                        self.cumulative_rewards[agent_id] +=30 #checked & done
            elif upper_right in self.agents.values() and upper_right != self.agents[agent_id]:
                if self.agents[agent_id][1] == 3 and 2 <= self.agents[agent_id][0] <= 8:
                    if self.agents[agent_id] == [self.agents[agent_id][0] - 1, self.agents[agent_id][1]]:
                        self.cumulative_rewards[agent_id] +=30 #checked
                elif 3 <= self.agents[agent_id][1] <= 7 and self.agents[agent_id][0] == 1:
                    if self.agents[agent_id] == [self.agents[agent_id][0], self.agents[agent_id][1] + 1]:
                        self.cumulative_rewards[agent_id] +=30 #checked
                elif self.agents[agent_id][1] == 8 and 1 <= self.agents[agent_id][0] <= 8:
                    if self.agents[agent_id] == [self.agents[agent_id][0] + 1, self.agents[agent_id][1]]:
                        self.cumulative_rewards[agent_id] +=30 #checked & done
            elif left_coord in self.agents.values() and left_coord != self.agents[agent_id]:
                if 2 <= self.agents[agent_id][1] <= 3:
                    if self.agents[agent_id] == [self.agents[agent_id][0], self.agents[agent_id][1]]:
                        self.cumulative_rewards[agent_id] +=30
            elif upper_left in self.agents.values() and upper_left != self.agents[agent_id]:
                if self.agents[agent_id][1] == 3 and 2 <= self.agents[agent_id][0] <= 9:
                    if self.agents[agent_id] == [self.agents[agent_id][0] - 1, self.agents[agent_id][1]]:
                        self.cumulative_rewards[agent_id] +=30 #checked
                elif self.agents[agent_id][1] == 8 and self.agents[agent_id][0] == 9:
                    if self.agents[agent_id] == [self.agents[agent_id][0], self.agents[agent_id][1] - 1]:
                        self.cumulative_rewards[agent_id] +=30 #checked
                elif 1 <= self.agents[agent_id][0] <= 8 and self.agents[agent_id][1] == 8:
                    if self.agents[agent_id] == [self.agents[agent_id][0] + 1, self.agents[agent_id][1]]:
                        self.cumulative_rewards[agent_id] +=30 #checked & done
            elif lower_left in self.agents.values() and lower_left != self.agents[agent_id]:
                if self.agents[agent_id][1] == 2 and 2 <= self.agents[agent_id][0] <= 9:
                    if self.agents[agent_id] == [self.agents[agent_id][0] - 1, self.agents[agent_id][1]]:
                        self.cumulative_rewards[agent_id] +=30 #checked
                elif self.agents[agent_id][1] == 7 and 2 <= self.agents[agent_id][0] <= 8:
                    if self.agents[agent_id] == [self.agents[agent_id][0] + 1, self.agents[agent_id][1]]:
                        self.cumulative_rewards[agent_id] +=30 #checked
                elif self.agents[agent_id][1] == 7 and self.agents[agent_id][0] == 1:
                    if self.agents[agent_id] == [self.agents[agent_id][0], self.agents[agent_id][1] - 1]:
                        self.cumulative_rewards[agent_id] +=30 #checked & done
            elif above_coord in self.agents.values() and above_coord != self.agents[agent_id]:
                if self.agents[agent_id][1] == 3 and 2 <= self.agents[agent_id][0] <= 9:
                    if self.agents[agent_id] == [self.agents[agent_id][0] - 1, self.agents[agent_id][1]]:
                        self.cumulative_rewards[agent_id] +=30
                elif self.agents[agent_id][0] == 1 and 3 <= self.agents[agent_id][1] <= 7:
                    if self.agents[agent_id] == [self.agents[agent_id][0], self.agents[agent_id][1] + 1]:
                        self.cumulative_rewards[agent_id] +=30
                elif self.agents[agent_id][1] == 8 and 1 <= self.agents[agent_id][0] <= 8:
                    if self.agents[agent_id] == [self.agents[agent_id][0] + 1, self.agents[agent_id][1]]:
                        self.cumulative_rewards[agent_id] +=30
                elif self.agents[agent_id][1] == 8 and self.agents[agent_id][0] == 9:
                    if self.agents[agent_id] == [self.agents[agent_id][0], self.agents[agent_id][1]]:
                        self.cumulative_rewards[agent_id] +=30
            elif below_coord in self.agents.values() and below_coord != self.agents[agent_id]:
                if 2 <= self.agents[agent_id][0] <= 9 and self.agents[agent_id][1] == 2:
                    if self.agents[agent_id][0] == [self.agents[agent_id][0] - 1, self.agent_id[agent_id][1]]:
                        self.cumulative_rewards[agent_id] += 30
                elif self.agents[agent_id][0] == 1 and self.agents[agent_id][1] == 2:
                    if self.agents[agent_id] == [self.agents[agent_id][0], self.agents[agent_id][1]]:
                        self.cumulative_rewards[agent_id] += 30
                elif 1 <= self.agents[agent_id][0] <= 8 and self.agents[agent_id][1] == 7:
                    if self.agents[agent_id] == [self.agents[agent_id][0] + 1, self.agents[agent_id][1]]:
                        self.cumulative_rewards[agent_id] += 30
                elif self.agents[agent_id][0] == 9 and 3 <= self.agents[agent_id][1] <= 7:
                    if self.agents[agent_id] == [self.agents[agent_id][0], self.agents[agent_id][1] - 1]:
                        self.cumulative_rewards[agent_id] += 30
            elif self.agents[agent_id] in self.upper_bypass:
                if below_coord in self.agents.values() and below_coord != self.agents[agent_id]:
                    if self.agents[agent_id] == [self.agents[agent_id][0], self.agents[agent_id][1]]:
                        self.cumulative_rewards[agent_id] += 30
                elif above_coord in self.agents.values() and above_coord != self.agents[agent_id]:
                    if self.agents[agent_id] == [self.agents[agent_id][0], self.agents[agent_id][1] + 1]:
                        self.cumulative_rewards[agent_id] += 30
                elif right_coord in self.agents.values() and right_coord != self.agents[agent_id]:
                    if 3 <= self.agents[agent_id][0] <= 7:
                        if self.agents[agent_id] == [self.agents[agent_id][0] - 1, self.agents[agent_id][1]]:
                            self.cumulative_rewards[agent_id] += 30
                    elif self.agents[agent_id][0] == 2:
                        if self.agents[agent_id] == [self.agents[agent_id][0], self.agents[agent_id][1] + 1]:
                            self.cumulative_rewards[agent_id] += 30
                elif left_coord in self.agents.values() and left_coord != self.agents[agent_id]:
                    if 3 <= self.agents[agent_id][0] <= 7:
                        if self.agents[agent_id] == [self.agents[agent_id][0] + 1, self.agents[agent_id][1]]:
                            self.cumulative_rewards[agent_id] += 30
                    elif self.agents[agent_id][0] == 8:
                        if self.agents[agent_id] == [self.agents[agent_id][0], self.agents[agent_id][1] + 1]:
                            self.cumulative_rewards[agent_id] += 30
            elif self.agents[agent_id] in self.lower_bypass:
                if below_coord in self.agents.values() and below_coord != self.agents[agent_id]:
                    if self.agents[agent_id] == [self.agents[agent_id][0], self.agents[agent_id][1] - 1]:
                        self.cumulative_rewards[agent_id] += 30
                elif above_coord in self.agents.values() and above_coord != self.agents[agent_id]:
                    if self.agents[agent_id] == [self.agents[agent_id][0], self.agents[agent_id][1]]:
                        self.cumulative_rewards[agent_id] += 30
                elif right_coord in self.agents.values() and right_coord != self.agents[agent_id]:
                    if 3 <= self.agents[agent_id][0] <= 7:
                        if self.agents[agent_id] == [self.agents[agent_id][0] - 1, self.agents[agent_id][1]]:
                            self.cumulative_rewards[agent_id] += 30
                    elif self.agents[agent_id][0] == 2:
                        if self.agents[agent_id] == [self.agents[agent_id][0], self.agents[agent_id][1] - 1]:
                            self.cumulative_rewards[agent_id] += 30
                elif left_coord in self.agents.values() and left_coord != self.agents[agent_id]:
                    if 3 <= self.agents[agent_id][0] <= 7:
                        if self.agents[agent_id] == [self.agents[agent_id][0] + 1, self.agents[agent_id][1]]:
                            self.cumulative_rewards[agent_id] += 30
                    elif self.agents[agent_id][0] == 8:
                        if self.agents[agent_id] == [self.agents[agent_id][0], self.agents[agent_id][1] - 1]:
                            self.cumulative_rewards[agent_id] += 30
                elif lower_right in self.agents.values() and lower_right != self.agents[agent_id]:
                    if self.agents[agent_id] == [self.agents[agent_id][0], self.agents[agent_id][1] - 1]:
                        self.cumulative_rewards[agent_id] += 30
                elif lower_left in self.agents.values() and lower_left != self.agents[agent_id]:
                    if self.agents[agent_id] == [self.agents[agent_id][0], self.agents[agent_id][1] - 1]:
                        self.cumulative_rewards[agent_id] += 30
                elif upper_right in self.agents.values() and upper_right != self.agents[agent_id]:
                    if self.agents[agent_id] == [self.agents[agent_id][0], self.agents[agent_id][1] + 1]:
                        self.cumulative_rewards[agent_id] += 30
                elif upper_left in self.agents.values() and upper_left != self.agents[agent_id]:
                    if self.agents[agent_id] == [self.agents[agent_id][0], self.agents[agent_id][1]]:
                        self.cumulative_rewards[agent_id] += 30

        if agent_id == 2:
            right_coord = [self.agents[agent_id][0] + 1, self.agents[agent_id][1]]
            left_coord = [self.agents[agent_id][0] - 1, self.agents[agent_id][1]]
            above_coord = [self.agents[agent_id][0], self.agents[agent_id][1] - 1]
            below_coord = [self.agents[agent_id][0], self.agents[agent_id][1] + 1]
            upper_right = [self.agents[agent_id][0] + 1, self.agents[agent_id][1] - 1]
            lower_right = [self.agents[agent_id][0] + 1, self.agents[agent_id][1] + 1]
            upper_left = [self.agents[agent_id][0] - 1, self.agents[agent_id][1] - 1]
            lower_left = [self.agents[agent_id][0] - 1, self.agents[agent_id][1] + 1]

            if left_coord in self.agents.values() and left_coord != self.agents[agent_id]:
                if 2 <= self.agents[agent_id][1] <= 3:
                    if 2 <= self.agents[agent_id][0] <= 8:
                        if self.agents[agent_id] == [self.agents[agent_id][0] + 1, self.agents[agent_id][1]]:
                            self.cumulative_rewards[agent_id] += 30
                    elif self.agents[agent_id][0] == 9:
                        if self.agents[agent_id] == [self.agents[agent_id][0], self.agents[agent_id][1] + 1]:
                            self.cumulative_rewards[agent_id] += 30
                elif 7 <= self.agents[agent_id][1] <= 8:
                    if self.agents[agent_id] == [self.agents[agent_id][0], self.agents[agent_id][1]]:
                        self.cumulative_rewards[agent_id] += 30
            elif right_coord in self.agents.values() and right_coord != self.agents[agent_id]:
                if self.agents[agent_id][1] == 3 and 1 <= self.agents[agent_id][0] <= 8:
                    if self.agents[agent_id] == [self.agents[agent_id][0], self.agents[agent_id][1] - 1]:
                        self.cumulative_rewards[agent_id] += 30
                elif self.agents[agent_id][1] == 2 and 1 <= self.agents[agent_id][0] <= 8:
                    if self.agents[agent_id] == [self.agents[agent_id][0], self.agents[agent_id][1] + 1]:
                        self.cumulative_rewards[agent_id] += 30
            elif upper_right in self.agents.values() and upper_right != self.agents[agent_id]:
                if self.agents[agent_id][1] == 3 and 1 <= self.agents[agent_id][0] <= 8:
                    if self.agents[agent_id] == [self.agents[agent_id][0] + 1, self.agents[agent_id][1]]:
                        self.cumulative_rewards[agent_id] +=30 #checked
                elif self.agents[agent_id][1] == 8 and self.agents[agent_id][0] == 1:
                    if self.agents[agent_id] == [self.agents[agent_id][0], self.agents[agent_id][1] - 1]:
                        self.cumulative_rewards[agent_id] +=30 #checked
                elif self.agents[agent_id][1] == 8 and 2 <= self.agents[agent_id][0] <= 8:
                    if self.agents[agent_id] == [self.agents[agent_id][0] - 1, self.agents[agent_id][1]]:
                        self.cumulative_rewards[agent_id] +=30 #checked & done
            elif lower_right in self.agents.values() and lower_left != self.agents[agent_id]:
                if self.agents[agent_id][1] == 2 and 2 <= self.agents[agent_id][0] <= 8:
                    if self.agents[agent_id] == [self.agents[agent_id][0] + 1, self.agents[agent_id][1]]:
                        self.cumulative_rewards[agent_id] +=30 #checked
                elif self.agents[agent_id][1] == 7 and 2 <= self.agents[agent_id][0] <= 9:
                    if self.agents[agent_id] == [self.agents[agent_id][0] - 1, self.agents[agent_id][1]]:
                        self.cumulative_rewards[agent_id] +=30 #checked
                elif self.agents[agent_id][1] == 2 and self.agents[agent_id][0] == 9:
                    if self.agents[agent_id] == [self.agents[agent_id][0], self.agents[agent_id][1] + 1]:
                        self.cumulative_rewards[agent_id] +=30 #checked & done
            elif upper_left in self.agents.values() and upper_left != self.agents[agent_id]:
                if self.agents[agent_id][1] == 3 and 2 <= self.agents[agent_id][0] <= 8:
                    if self.agents[agent_id] == [self.agents[agent_id][0] + 1, self.agents[agent_id][1]]:
                        self.cumulative_rewards[agent_id] +=30 #checked
                elif self.agents[agent_id][1] == 8 and 2 <= self.agents[agent_id][0] <= 9:
                    if self.agents[agent_id] == [self.agents[agent_id][0] - 1, self.agents[agent_id][1]]:
                        self.cumulative_rewards[agent_id] +=30 #checked
                elif self.agents[agent_id][1] == 3 and self.agents[agent_id][0] == 9:
                    if self.agents[agent_id] == [self.agents[agent_id][0], self.agents[agent_id][1] + 1]:
                        self.cumulative_rewards[agent_id] +=30 #checked & done
            elif lower_left in self.agents.values() and lower_left != self.agents[agent_id]:
                if self.agents[agent_id][1] == 3 and 1 <= self.agents[agent_id][0] <= 8:
                    if self.agents[agent_id] == [self.agents[agent_id][0] + 1, self.agents[agent_id][1]]:
                        self.cumulative_rewards[agent_id] +=30 #checked
                elif self.agents[agent_id][1] == 8 and 2 <= self.agents[agent_id][0] <= 8:
                    if self.agents[agent_id] == [self.agents[agent_id][0] - 1, self.agents[agent_id][1]]:
                        self.cumulative_rewards[agent_id] +=30 #checked
                elif self.agents[agent_id][1] == 8 and self.agents[agent_id][0] == 1:
                    if self.agents[agent_id] == [self.agents[agent_id][0], self.agents[agent_id][1] - 1]:
                        self.cumulative_rewards[agent_id] +=30 #checked & done
            elif below_coord in self.agents.values() and below_coord != self.agents[agent_id]:
                if self.agents[agent_id][1] == 2 and 1 <= self.agents[agent_id][0] <= 8:
                    if self.agents[agent_id] == [self.agents[agent_id][0] + 1, self.agents[agent_id][1]]:
                        self.cumulative_rewards[agent_id] += 30
                elif self.agents[agent_id][1] == 2 and self.agents[agent_id][0] == 9:
                    if self.agents[agent_id] == [self.agents[agent_id][0], self.agents[agent][1]]:
                        self.cumulative_rewards[agent_id] += 30
                elif self.agents[agent_id][0] == 1 and 3 <= self.agents[agent_id][1] <= 7:
                    if self.agents[agent_id] == [self.agents[agent_id][0], self.agents[agent][1]]:
                        self.cumulative_rewards[agent_id] += 30
                elif self.agents[agent_id][1] == 7 and 2 <= self.agents[agent_id][0] <= 9:
                    if self.agents[agent_id] == [self.agents[agent_id][0] - 1, self.agents[agent][1]]:
                        self.cumulative_rewards[agent_id] += 30
            elif above_coord in self.agents.values() and above_coord != self.agents[agent_id]:
                if self.agents[agent_id][1] == 3 and 1 <= self.agents[agent_id][0] <= 8:
                    if self.agents[agent_id] == [self.agents[agent_id][0] + 1, self.agents[agent][1]]:
                        self.cumulative_rewards[agent_id] += 30
                elif self.agents[agent_id][0] == 1 and 4 <= self.agents[agent_id][1] <= 7:
                    if self.agents[agent_id] == [self.agents[agent_id][0], self.agents[agent][1] + 1]:
                        self.cumulative_rewards[agent_id] += 30
                elif self.agents[agent_id][1] == 8 and 1 <= self.agents[agent_id][0] <= 8:
                    if self.agents[agent_id] == [self.agents[agent_id][0] + 1, self.agents[agent][1]]:
                        self.cumulative_rewards[agent_id] += 30
                elif self.agents[agent_id][1] == 8 and self.agents[agent_id][0] == 9:
                    if self.agents[agent_id] == [self.agents[agent_id][0], self.agents[agent][1] + 1]:
                        self.cumulative_rewards[agent_id] += 30
            elif self.agents[agent_id] in self.upper_bypass:
                if below_coord in self.agents.values() and below_coord != self.agents[agent_id]:
                    if self.agents[agent_id] == [self.agents[agent_id][0], self.agents[agent_id][1]]:
                        self.cumulative_rewards[agent_id] += 30
                elif above_coord in self.agents.values() and above_coord != self.agents[agent_id]:
                    if self.agents[agent_id] == [self.agents[agent_id][0], self.agents[agent_id][1] + 1]:
                        self.cumulative_rewards[agent_id] += 30
                elif right_coord in self.agents.values() and right_coord != self.agents[agent_id]:
                    if 3 <= self.agents[agent_id][0] <= 7:
                        if self.agents[agent_id] == [self.agents[agent_id][0] - 1, self.agents[agent_id][1]]:
                            self.cumulative_rewards[agent_id] += 30
                    elif self.agents[agent_id][0] == 2:
                        if self.agents[agent_id] == [self.agents[agent_id][0], self.agents[agent_id][1] + 1]:
                            self.cumulative_rewards[agent_id] += 30
                elif left_coord in self.agents.values() and left_coord != self.agents[agent_id]:
                    if 3 <= self.agents[agent_id][0] <= 7:
                        if self.agents[agent_id] == [self.agents[agent_id][0] + 1, self.agents[agent_id][1]]:
                            self.cumulative_rewards[agent_id] += 30
                    elif self.agents[agent_id][0] == 8:
                        if self.agents[agent_id] == [self.agents[agent_id][0], self.agents[agent_id][1] + 1]:
                            self.cumulative_rewards[agent_id] += 30
            elif self.agents[agent_id] in self.lower_bypass:
                if below_coord in self.agents.values() and below_coord != self.agents[agent_id]:
                    if self.agents[agent_id] == [self.agents[agent_id][0], self.agents[agent_id][1] - 1]:
                        self.cumulative_rewards[agent_id] += 30
                elif above_coord in self.agents.values() and above_coord != self.agents[agent_id]:
                    if self.agents[agent_id] == [self.agents[agent_id][0], self.agents[agent_id][1]]:
                        self.cumulative_rewards[agent_id] += 30
                elif right_coord in self.agents.values() and right_coord != self.agents[agent_id]:
                    if 3 <= self.agents[agent_id][0] <= 7:
                        if self.agents[agent_id] == [self.agents[agent_id][0] - 1, self.agents[agent_id][1]]:
                            self.cumulative_rewards[agent_id] += 30
                    elif self.agents[agent_id][0] == 2:
                        if self.agents[agent_id] == [self.agents[agent_id][0], self.agents[agent_id][1] - 1]:
                            self.cumulative_rewards[agent_id] += 30
                elif left_coord in self.agents.values() and left_coord != self.agents[agent_id]:
                    if 3 <= self.agents[agent_id][0] <= 7:
                        if self.agents[agent_id] == [self.agents[agent_id][0] + 1, self.agents[agent_id][1]]:
                            self.cumulative_rewards[agent_id] += 30
                    elif self.agents[agent_id][0] == 8:
                        if self.agents[agent_id] == [self.agents[agent_id][0], self.agents[agent_id][1] - 1]:
                            self.cumulative_rewards[agent_id] += 30
                elif lower_right in self.agents.values() and lower_right != self.agents[agent_id]:
                    if self.agents[agent_id] == [self.agents[agent_id][0], self.agents[agent_id][1] - 1]:
                        self.cumulative_rewards[agent_id] += 30
                elif lower_left in self.agents.values() and lower_left != self.agents[agent_id]:
                    if self.agents[agent_id] == [self.agents[agent_id][0], self.agents[agent_id][1] - 1]:
                        self.cumulative_rewards[agent_id] += 30
                elif upper_right in self.agents.values() and upper_right != self.agents[agent_id]:
                    if self.agents[agent_id] == [self.agents[agent_id][0], self.agents[agent_id][1] + 1]:
                        self.cumulative_rewards[agent_id] += 30
                elif upper_left in self.agents.values() and upper_left != self.agents[agent_id]:
                    if self.agents[agent_id] == [self.agents[agent_id][0], self.agents[agent_id][1]]:
                        self.cumulative_rewards[agent_id] += 30

        return self.cumulative_rewards[agent_id]

    def is_done(self, agent_id):
        for next_agent_id, next_seeker in self.agents.items():
            if (agent_id != next_agent_id and self.agents[agent_id] == next_seeker) or self.env_timestep >= 5000:
                return True
            return False

    def step(self, action):
        agent_ids = action.keys()
        for agent_id in agent_ids:
            seeker = self.agents[agent_id]
            if seeker in self.up_down or (4 <= seeker[1] <= 6 and (seeker[0] == 1 or seeker[0] == 9)) or seeker == (2, 1) or seeker == (8, 1) or seeker == (2, 9) or seeker == (8, 9):
                if action[agent_id] == 0 or action[agent_id] == 1: #down
                    seeker = (seeker[0], min(seeker[1] + 1, 10))
                elif action[agent_id] == 2 or action[agent_id] == 3: #up
                    seeker = (seeker[0], max(seeker[1] - 1, 0))
                elif action[agent_id] == 4: #stop
                    seeker = (seeker[0], seeker[1])
            elif seeker in self.up_down_right or seeker == (1, 3) or seeker == (1, 7):
                if action[agent_id] == 0 or action[agent_id] == 1: #down
                    seeker = (seeker[0], min(seeker[1] + 1, 10))
                elif action[agent_id] == 2: #up
                    seeker = (seeker[0], max(seeker[1] - 1, 0))
                elif action[agent_id] == 3: #right
                    seeker = (min(seeker[0] + 1, 10), seeker[1])
                elif action[agent_id] == 4: #stop
                    seeker = (seeker[0], seeker[1])
            elif seeker in self.up_down_left or seeker == (9, 3) or seeker == (9, 7):
                if action[agent_id] == 0: #down
                    seeker = (seeker[0], min(seeker[1] + 1, 10))
                elif action[agent_id] == 1 or action[agent_id] == 2: #left
                    seeker = (max(seeker[0] - 1, 0), seeker[1])
                elif action[agent_id] == 3: #up
                    seeker = (seeker[0], max(seeker[1] - 1, 0))
                elif action[agent_id] == 4: #stop
                    seeker = (seeker[0], seeker[1])
            elif seeker in self.up_right_left or (3 <= seeker[0] <= 7 and seeker[1] == 8) or (2 <= seeker[0] <= 8 and seeker[1] == 3):
                if action[agent_id] == 0 or action[agent_id] == 1: #left
                    seeker = (max(seeker[0] - 1, 0), seeker[1])
                elif action[agent_id] == 2: #up
                    seeker = (seeker[0], max(seeker[1] - 1, 0))
                elif action[agent_id] == 3: #right
                    seeker = (min(seeker[0] + 1, 10), seeker[1])
                elif action[agent_id] == 4: #stop
                    seeker = (seeker[0], seeker[1])
            elif seeker in self.up_right or seeker == (1, 8) or seeker == (2, 10):
                if action[agent_id] == 0 or action[agent_id] == 1: #up
                    seeker = (seeker[0], max(seeker[1] - 1, 0))
                elif action[agent_id] == 2 or action[agent_id] == 3: #right
                    seeker = (min(seeker[0] + 1, 10), seeker[1])
                elif action[agent_id] == 4: #stop
                    seeker = (seeker[0], seeker[1])
            elif seeker in self.up_left or seeker == (9, 8) or seeker == (8, 10):
                if action[agent_id] == 0 or action[agent_id] == 1: #left
                    seeker = (max(seeker[0] - 1, 0), seeker[1])
                elif action[agent_id] == 2 or action[agent_id] == 3: #up
                    seeker = (seeker[0], max(seeker[1] - 1, 0))
                elif action[agent_id] == 4: #stop
                    seeker = (seeker[0], seeker[1])
            elif seeker in self.down_right_left or (2 <= seeker[0] <= 8 and seeker[1] == 7) or (3 <= seeker[0] <= 7 and seeker[1] == 2):
                if action[agent_id] == 0 or action[agent_id]: #down
                    seeker = (seeker[0], min(seeker[1] + 1, 10))
                elif action[agent_id] == 1 or action[agent_id] == 2: #right
                    seeker = (min(seeker[0] + 1, 10), seeker[1])
                elif action[agent_id] == 3: #left
                    seeker = (max(seeker[0] - 1, 0), seeker[1])
                elif action[agent_id] == 4: #stop
                    seeker = (seeker[0], seeker[1])
            elif seeker in self.left_right or (3 <= seeker[0] <= 7 and (seeker[1] == 0 or seeker[1] == 10)):
                if action[agent_id] == 0 or action[agent_id] == 1: #left
                    seeker = (max(seeker[0] - 1, 0), seeker[1])
                elif action[agent_id] == 2 or action[agent_id] == 3: #right
                    seeker = (min(seeker[0] + 1, 10), seeker[1])
                elif action[agent_id] == 4: #stop
                    seeker = (seeker[0], seeker[1])
            elif seeker in self.down_right or seeker == (1, 2) or seeker == (2, 0):
                if action[agent_id] == 0 or action[agent_id] == 1: #down
                    seeker = (seeker[0], min(seeker[1] + 1, 10))
                elif action[agent_id] == 2 or action[agent_id] == 3: #right
                    seeker = (min(seeker[0] + 1, 10), seeker[1])
                elif action[agent_id] == 4: #stop
                    seeker = (seeker[0], seeker[1])
            elif seeker in self.down_left or seeker == (8, 0) or seeker == (9, 2):
                if action[agent_id] == 0 or action[agent_id] == 1: #down
                    seeker = (seeker[0], min(seeker[1] + 1, 10))
                elif action[agent_id] == 2 or action[agent_id] == 3: #left
                    seeker = (max(seeker[0] - 1, 0), seeker[1])
                elif action[agent_id] == 4: #stop
                    seeker = (seeker[0], seeker[1])
            else:
                if action[agent_id] == 0: #down
                    seeker = (seeker[0], min(seeker[1] + 1, 10))
                elif action[agent_id] == 1: #left
                    seeker = (max(seeker[0] - 1, 0), seeker[1])
                elif action[agent_id] == 2: #up
                    seeker = (seeker[0], max(seeker[1] - 1, 0))
                elif action[agent_id] == 3: #right
                    seeker = (min(seeker[0] + 1, 10), seeker[1])
                elif action[agent_id] == 4: #stop
                    seeker = (seeker[0], seeker[1])
                else:
                    raise ValueError("Invalid action")
            self.agents[agent_id] = seeker
            self.last_actions[agent_id] = action[agent_id]

        self.env_timestep += 1
        observations = {i: self.get_observation(i) for i in agent_ids}
        rewards = {i: self.get_reward(i) for i in agent_ids}
        done = {i: self.is_done(i) for i in agent_ids}

        done["__all__"] = all(done.values())

        return observations, rewards, done, self.info

    def render(self):
        self.screen.blit(self.game_map, (0, 0))

        colors = [(255, 0, 0), (0, 0, 0)]

        for agent_id, position in self.agents.items():
            agent_x, agent_y = position
            color = colors[agent_id - 1]
            pygame.draw.rect(
                self.screen,
                color,
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
