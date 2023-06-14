import gym
import numpy as np
import pygame

WIDTH = 1920
HEIGHT = 1080
CELL_SIZE = 100
GRID_SIZE = (10, 10)

class CarEnv(gym.Env):
    def __init__(self):
        super(CarEnv, self).__init__()

        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)

        self.action_space = gym.spaces.Discrete(4)
        self.action_space_2 = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Discrete(GRID_SIZE[0] * GRID_SIZE[1])

        self.clock = pygame.time.Clock()
        self.game_map = pygame.image.load('GRID_MAP_01.png').convert()
        self.grid = np.zeros(GRID_SIZE, dtype=np.uint8)

        self.car = Car()
        self.car_2 = Car2()

    def step(self, action, action_car_2):
        if action == 0:
            self.car.turn_right()
        elif action == 1:
            self.car.turn_left()
        elif action == 2:
            self.car.up()
        else:
            self.car.down()  # Stop

        if action_car_2 == 0:
            self.car_2.turn_right()
        elif action_car_2 == 1:
            self.car_2.turn_left()
        elif action_car_2 == 2:
            self.car_2.up()
        else:
            self.car_2.down()  # Stop

        self.car.update()
        self.car_2.update()
        done = not self.car.is_alive() or not self.car_2.is_alive()
        reward = self.car.get_reward() + self.car_2.get_reward()
        obs = [self.car.get_grid_cell(), self.car_2.get_grid_cell()]

        reward = 0
        x, y = self.car.position[0], self.car.position[1]
        if self.game_map.get_at((int(x), int(y))) != (0, 255, 0):
            reward -= self.car.get_reward()

        if self.game_map.get_at((int(x), int(y))) == (0, 255, 0):
            if 2 <= self.car.position[1] <= 8:
                if self.car.position[0] == 1:
                    if 4 <= self.car.position[1] <= 8:
                        if self.car.up():
                            reward += self.car.get_reward()
                        else:
                            reward -= self.car.get_reward()
                    elif self.car.position[1] == 3:
                        if self.car.turn_right() or self.car.up():
                            reward += self.car.get_reward()
                        else:
                            reward -= self.car.get_reward()
                    elif self.car.position[1] == 2:
                        if self.car.turn_right():
                            reward += self.car.get_reward()
                        else:
                            reward -= self.car.get_reward()
                elif self.car.position[0] == 9:
                    if 2 <= self.car.position[1] <= 6:
                        if self.car.down():
                            reward += self.car.get_reward()
                        else:
                            reward -= self.car.get_reward()
                    elif self.car.position[1] == 7:
                        if self.car.down() or self.car.turn_left():
                            reward += self.car.get_reward()
                        else:
                            reward -= self.car.get_reward()
                    elif self.car.position[1] == 8:
                        if self.car.turn_left():
                            reward += self.car.get_reward()
                        else:
                            reward -= self.car.get_reward()
            elif 2 <= self.car.position[0] <= 8:
                if self.car.position[1] == 2:
                    if self.car.turn_right() or self.car.down():
                        reward += self.car.get_reward()
                    else:
                        reward -= self.car.get_reward()
                elif self.car.position[1] == 3:
                    if self.car.turn_right() or self.car.up():
                        reward += self.car.get_reward()
                    else:
                        reward -= self.car.get_reward()
                elif self.car.position[1] == 7:
                    if self.car.turn_left() or self.car.down():
                        reward += self.car.get_reward()
                    else:
                        reward -= self.car.get_reward()
                elif self.car.position[1] == 8:
                    if self.car.turn_left() or self.car.up():
                        reward += self.car.get_reward()
                    else:
                        reward -= self.car.get_reward()
        else:
            reward -= self.car.get_reward()

        pygame.event.pump()

        reward_car_2 = 0
        x_2, y_2 = self.car_2.position[0], self.car_2.position[1]

        if self.game_map.get_at((int(x_2), int(y_2))) == (0, 255, 0):
            if 2 <= self.car_2.position[1] <= 8:
                if self.car_2.position[0] == 1:
                    if 2 <= self.car_2.position[1] <= 6:
                        if self.car_2.down():
                            reward_car_2 += self.car_2.get_reward()
                        else:
                            reward_car_2 -= self.car_2.get_reward()
                    elif self.car_2.position[1] == 7:
                        if self.car_2.turn_right() or self.car_2.down():
                            reward_car_2 += self.car_2.get_reward()
                        else:
                            reward_car_2 -= self.car_2.get_reward()
                    elif self.car_2.position[1] == 8:
                        if self.car_2.turn_right():
                            reward_car_2 += self.car_2.get_reward()
                        else:
                            reward_car_2 -= self.car_2.get_reward()
                elif self.car_2.position[0] == 9:
                    if 4 <= self.car_2.position[1] <= 8:
                        if self.car.up():
                            reward_car_2 += self.car_2.get_reward()
                        else:
                            reward_car_2 -= self.car_2.get_reward()
                    elif self.car_2.position[1] == 3:
                        if self.car_2.up() or self.car_2.turn_left():
                            reward_car_2 += self.car_2.get_reward()
                        else:
                            reward_car_2 -= self.car_2.get_reward()
                    elif self.car_2.position[1] == 2:
                        if self.car_2.down() or self.car_2.turn_left():
                            reward_car_2 += self.car_2.get_reward()
                        else:
                            reward_car_2 -= self.car_2.get_reward()
            elif 2 <= self.car_2.position[0] <= 8:
                if self.car_2.position[1] == 2:
                    if self.car_2.turn_left() or self.car_2.down():
                        reward_car_2 += self.car_2.get_reward()
                    else:
                        reward_car_2 -= self.car_2.get_reward()
                elif self.car_2.position[1] == 3:
                    if self.car_2.turn_left() or self.car_2.up():
                        reward_car_2  += self.car_2.get_reward()
                    else:
                        reward_car_2 -= self.car_2.get_reward()
                elif self.car_2.position[1] == 7:
                    if self.car_2.turn_right() or self.car_2.down():
                        reward_car_2 += self.car_2.get_reward()
                    else:
                        reward_car_2 -= self.car_2.get_reward()
                elif self.car_2.position[1] == 8:
                    if self.car_2.turn_right() or self.car_2.up():
                        reward_car_2 += self.car_2.get_reward()
                    else:
                        reward_car_2 -= self.car_2.get_reward()
        else:
            reward_car_2 -= self.car_2.get_reward()

        reward += reward_car_2
        return obs, reward, done, {}

    def reset(self):
        self.car = Car()
        self.car_2 = Car2()
        return [self.car.get_grid_cell(), self.car_2.get_grid_cell()]

    def render(self, mode='human'):
        self.screen.blit(self.game_map, (2, 3))
        self.car.draw(self.screen)
        self.car_2.draw(self.screen)
        pygame.display.flip()
        pygame.time.wait(200)
        self.clock.tick(60)

    def close(self):
        pygame.quit()

    def train(self, total_timesteps=100000):
        env = DummyVecEnv([lambda: self])
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=total_timesteps)

    def run(self, model):
        obs = self.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, _, done, _ = self.step(*action)
            self.render()

class Car:
    def __init__(self):
        self.position = (2, 3)

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

class Car2:
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

'''env = CarEnv()
obs = env.reset()'''

done = False
'''for _ in range(1000):
    action = env.action_space.sample()
    action_car_2 = env.action_space_2.sample()
    obs, reward, done, _ = env.step(action, action_car_2)

    env.render()

env.close()'''
