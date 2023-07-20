from ray.rllib.env.multi_agent_env import MultiAgentEnv
import gym
from gym.spaces import Discrete, Box
import numpy as np
import pygamea
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
        self.agents = {1: (5, 2), 2: (5, 7)}
        self.info = {1: {'obs': self.agents[1]}, 2: {'obs': self.agents[2]}}
        self.grid = np.zeros(GRID_SIZE, dtype=np.uint8)

        self.up_down = [(2, 1), (2, 9), (1, 4), (1, 5), (1, 6)] #agent_1 reward 1 #agent_2 #doublechecked
        self.up_down_extra = [(8, 1), (8, 9), (9, 4), (9, 5), (9, 6)] #doublechecked

        self.up_right_left = [(2, 3), (3, 3), (4, 3), (5, 3), (6, 3), (7, 3), (8, 3)] #doublechecked
        self.up_right_left_extra = [(3, 8), (4, 8), (5, 8), (6, 8), (7, 8)] #doublechecked

        self.down_right = [(1, 2), (2, 0)] #doublechecked

        self.down_left = [(9, 2), (8, 0)] #doublechecked

        self.up_right = [(1, 8), (2, 10)]

        self.up_left = [(9, 8), (8, 10)] #doublechecked

        self.down_right_left = [(3, 2), (4, 2), (5, 2), (6, 2), (7, 2)] #doublechecked
        self.down_right_left_extra = [(2, 7), (3, 7), (4, 7), (5, 7), (6, 7), (7, 7), (8, 7)] #doublechecked

        self.left_right = [(3, 0), (4, 0), (5, 0), (6, 0), (7, 0)]
        self.left_right_extra =  [(3, 10), (4, 10), (5, 10), (6, 10), (7, 10)]

        self.up_down_right = (1, 3) #doublechecked
        self.up_down_right_extra = (1, 7) #doublechecked

        self.up_down_left = (9, 3) #doublechecked
        self.up_down_left_extra = (9, 7) #doublechecked

        self.all = [(2, 2), (8, 2)]
        self.all_extra = [(2, 8), (8, 8)]

        self.upper_bypass = [(2, 0), (2, 1), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (8, 1)]
        self.lower_bypass = [(2, 9), (2, 10), (3, 10), (4, 10), (5, 10), (6, 10), (7, 10), (8, 10), (8, 9)]
        self.cumulative_rewards = {1: 0, 2: 0}
        self.env_timestep = 0

        self.last_actions = {}


    def reset(self):
        self.env_timestep = 0
        self.agents = {1: (5, 2), 2: (5, 7)}
        return {1: self.get_observation(1), 2: self.get_observation(2)}

    def get_observation(self, agent_id):
        seeker = self.agents[agent_id]
        return 11 * seeker[0] + seeker[1]

    def get_reward(self, agent_id):
        x, y = self.agents[agent_id]
        self.cumulative_rewards[agent_id] = 0

        agents_list = list(self.agents.items())
        num_agents = len(agents_list)

        for i in range(num_agents):
            next_agent_id, next_seeker = agents_list[i]
            if agent_id != next_agent_id and self.agents[agent_id] == next_seeker:
                self.cumulative_rewards[agent_id] -= 20

        if agent_id == 1:
            right_coord = [self.agents[agent_id][0] + 1, self.agents[agent_id][1]]
            left_coord = [self.agents[agent_id][0] - 1, self.agents[agent_id][1]]
            above_coord = [self.agents[agent_id][0], self.agents[agent_id][1] - 1]
            below_coord = [self.agents[agent_id][0], self.agents[agent_id][1] + 1]

            if right_coord not in self.agents.values() or left_coord not in self.agents.values() or above_coord not in self.agents.values() or below_coord not in self.agents.values():
                if self.last_actions[agent_id] == 1 or self.last_actions[agent_id] == 3:
                    self.cumulative_rewards[agent_id] += 20
                else:
                    self.cumulative_rewards[agent_id] -= 30

        elif agent_id == 2:
            right_coord = [self.agents[agent_id][0] + 1, self.agents[agent_id][1]]
            left_coord = [self.agents[agent_id][0] - 1, self.agents[agent_id][1]]
            above_coord = [self.agents[agent_id][0], self.agents[agent_id][1] - 1]
            below_coord = [self.agents[agent_id][0], self.agents[agent_id][1] + 1]

            if right_coord not in self.agents.values() or left_coord not in self.agents.values() or above_coord not in self.agents.values() or below_coord not in self.agents.values():
                if self.last_actions[agent_id] == 2 or self.last_actions[agent_id] == 0:
                    self.cumulative_rewards[agent_id] += 20
                else:
                    self.cumulative_rewards[agent_id] -= 30

        return self.cumulative_rewards[agent_id]


    def is_done(self, agent_id):
        for next_agent_id, next_seeker in self.agents.items():
            if (agent_id != next_agent_id and self.agents[agent_id] == next_seeker) or self.env_timestep >= 10000:
                return True
        return False


    def step(self, action):
        agent_ids = action.keys()
        for agent_id in agent_ids:
            seeker = self.agents[agent_id]
            if seeker in self.up_down: #DONE
                if action[agent_id] == 0 or action[agent_id] == 2:
                    seeker = (seeker[0], max(seeker[1] - 1, 0)) #UP
                elif action[agent_id] == 1 or action[agent_id] == 3:
                    seeker = (seeker[0], min(seeker[1] + 1, 10)) #DOWN
                elif action[agent_id] == 4: #STOP
                    seeker = (seeker[0], seeker[1])
            elif seeker in self.up_down_extra: #DONE
                if action[agent_id] == 1 or action[agent_id] == 3:
                    seeker = (seeker[0], max(seeker[1] - 1, 0)) #UP
                elif action[agent_id] == 0 or action[agent_id] == 2:
                    seeker = (seeker[0], min(seeker[1] + 1, 10)) #DOWN
                elif action[agent_id] == 4:
                    seeker = (seeker[0], seeker[1]) #STOP
            elif seeker in self.up_right_left:
                if action[agent_id] == 0 or action[agent_id] == 3:
                    seeker = (seeker[0], max(seeker[1] - 1, 0)) #UP
                elif action[agent_id] == 1:
                    seeker = (max(seeker[0] - 1, 0), seeker[1]) #LEFT
                elif action[agent_id] == 2:
                    seeker = (min(seeker[0] + 1, 10), seeker[1]) #RIGHT
                elif action[agent_id] == 4:
                    seeker = (seeker[0], seeker[1]) #STOP
            elif seeker in self.up_right_left_extra:
                if action[agent_id] == 0:
                    seeker = (max(seeker[0] - 1, 0), seeker[1]) #LEFT
                elif action[agent_id] ==  or action[agent_id] == 3:
                    seeker = (seeker[0], max(seeker[1] - 1, 0)) #UP
                elif action[agent_id] == 1:
                    seeker = (min(seeker[0] + 1, 10), seeker[1]) #RIGHT
                elif action[agent_id] == 4:
                    seeker = (seeker[0], seeker[1]) #STOP
            elif seeker in self.down_right: #DONE
                if action[agent_id] == 0 or action[agent_id] == 2:
                    seeker = (min(seeker[0] + 1, 10), seeker[1]) #RIGHT
                elif action[agent_id] == 1 or action[agent_id] == 3: #DOWN
                    seeker = (seeker[0], min(seeker[1] + 1, 10))
                elif action[agent_id] == 4:
                    seeker = (seeker[0], seeker[1])
            elif seeker in self.down_left:
                if action[agent_id] == 2: #DOWN
                    seeker = (seeker[0], min(seeker[1] + 1, 10))
                elif action[agent_id] == 1:
                    seeker = (min(seeker[0] - 1, 10), seeker[1]) #LEFT
                elif action[agent_id] == 0 and action[agent_id] == 4 or action[agent_id] == 3:
                    seeker = (seeker[0], seeker[1])
            elif seeker in self.up_right:
                if action[agent_id] == 2: #up
                    seeker = (seeker[0], max(seeker[1] - 1, 0))
                elif action[agent_id] == 1: #right
                    seeker = (min(seeker[0] + 1, 10), seeker[1])
                elif action[agent_id] == 4 or action[agent_id] == 3 or action[agent_id] == 0: #stop
                    seeker = (seeker[0], seeker[1])
            elif seeker in self.up_left:
                if action[agent_id] == 2: #left
                    seeker = (max(seeker[0] - 1, 0), seeker[1])
                elif action[agent_id] == 1: #up
                    seeker = (seeker[0], max(seeker[1] - 1, 0))
                elif action[agent_id] == 4: #stop
                    seeker = (seeker[0], seeker[1])
            elif seeker in self.down_right_left: #DONE
                if action[agent_id] == 0:
                    seeker = (min(seeker[0] + 1, 10), seeker[1]) #RIGHT
                elif action[agent_id] == 1:
                    seeker = (min(seeker[0] - 1, 10), seeker[1]) #LEFT
                elif action[agent_id] == 2:
                    seeker = (min(seeker[0] + 1, 10), seeker[1]) #RIGHT
                if action[agent_id] == 3:
                    seeker = (seeker[0], min(seeker[1] + 1, 10)) #DOWN
                elif action[agent_id] == 4:
                    seeker = (seeker[0], seeker[1]) #STOP
            elif seeker in self.down_right_left_extra: #DONE
                if action[agent_id] == 0:
                    seeker = (seeker[0], min(seeker[1] + 1, 10)) #DOWN
                if action[agent_id] == 1:
                    seeker = (min(seeker[0] + 1, 10), seeker[1]) #RIGHT
                elif action[agent_id] == 2:
                    seeker = (min(seeker[0] - 1, 10), seeker[1]) #LEFT
                elif action[agent_id] == 3:
                    seeker = (seeker[0], min(seeker[1] + 1, 10)) #DOWN
                elif action[agent_id] == 4:
                    seeker = (seeker[0], seeker[1]) #STOP
            elif seeker in self.left_right:
                if action[agent_id] == 2:
                    seeker = (min(seeker[0] + 1, 10), seeker[1]) #RIGHT
                elif action[agent_id] == 1:
                    seeker = (max(seeker[0] - 1, 0), seeker[1]) #LEFT
                elif action[agent_id] == 4 or action[agent_id] == 0:
                    seeker = (seeker[0], seeker[1]) #STOP
            elif seeker in self.left_right_extra:
                if action[agent_id] == 1:
                    seeker = (min(seeker[0] + 1, 10), seeker[1]) #RIGHT
                elif action[agent_id] == 2:
                    seeker = (max(seeker[0] - 1, 0), seeker[1]) #LEFT
                elif action[agent_id] == 4 or action[agent_id] == 0:
                    seeker = (seeker[0], seeker[1]) #STOP
            elif seeker == self.up_down_right: #DONE
                if action[agent_id] == 1 or action[agent_id] == 3: #down
                    seeker = (seeker[0], min(seeker[1] + 1, 10))
                elif action[agent_id] == 0: #up
                    seeker = (seeker[0], max(seeker[1] - 1, 0))
                elif action[agent_id] == 2: #right
                    seeker = (min(seeker[0] + 1, 10), seeker[1])
                elif action[agent_id] == 4: #stop
                    seeker = (seeker[0], seeker[1])
            elif seeker == self.up_down_right_extra: #DONE
                if action[agent_id] == 3: #down
                    seeker = (seeker[0], min(seeker[1] + 1, 10))
                elif action[agent_id] == 0 or action[agent_id] == 2: #up
                    seeker = (seeker[0], max(seeker[1] - 1, 0))
                elif action[agent_id] == 1: #right
                    seeker = (min(seeker[0] + 1, 10), seeker[1])
                elif action[agent_id] == 4: #stop
                    seeker = (seeker[0], seeker[1])
            elif seeker == self.up_down_left:
                if action[agent_id] == 2: #down
                    seeker = (seeker[0], min(seeker[1] + 1, 10))
                elif action[agent_id] == 0 or action[agent_id] == 3: #up
                    seeker = (seeker[0], max(seeker[1] - 1, 0))
                elif action[agent_id] == 1: #left
                    seeker = (min(seeker[0] - 1, 10), seeker[1])
                elif action[agent_id] == 4: #stop
                    seeker = (seeker[0], seeker[1])
            elif seeker == self.up_down_left_extra:
                if action[agent_id] == 0 or action[agent_id] == 3: #down
                    seeker = (seeker[0], min(seeker[1] + 1, 10))
                elif action[agent_id] == 1: #up
                    seeker = (seeker[0], max(seeker[1] - 1, 0))
                elif action[agent_id] == 2: #left
                    seeker = (min(seeker[0] - 1, 10), seeker[1])
                elif action[agent_id] == 4: #stop
                    seeker = (seeker[0], seeker[1])
            elif seeker in self.all:
                if action[agent_id] == 0: #down
                    seeker = (seeker[0], min(seeker[1] + 1, 10))
                elif action[agent_id] == 1: #left
                    seeker = (max(seeker[0] - 1, 0), seeker[1])
                elif action[agent_id] == 3: #up
                    seeker = (seeker[0], max(seeker[1] - 1, 0))
                elif action[agent_id] == 2: #right
                    seeker = (min(seeker[0] + 1, 10), seeker[1])
                elif action[agent_id] == 4: #stop
                    seeker = (seeker[0], seeker[1])
            elif seeker in self.all_extra:
                if action[agent_id] == 0: #down
                    seeker = (seeker[0], min(seeker[1] + 1, 10))
                elif action[agent_id] == 2: #left
                    seeker = (max(seeker[0] - 1, 0), seeker[1])
                elif action[agent_id] == 3: #up
                    seeker = (seeker[0], max(seeker[1] - 1, 0))
                elif action[agent_id] == 1: #right
                    seeker = (min(seeker[0] + 1, 10), seeker[1])
                elif action[agent_id] == 4: #stop
                    seeker = (seeker[0], seeker[1])

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
