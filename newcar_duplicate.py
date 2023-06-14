import gym
import numpy as np
import pygame
import stable_baselines3

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
        self.observation_space = gym.spaces.Discrete(GRID_SIZE[0] * GRID_SIZE[1])

        self.action_space_2 = gym.spaces.Discrete(4)
        self.observation_space_2 = gym.spaces.Discrete(GRID_SIZE[0] * GRID_SIZE[1])

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
            self.car_2.down()

        self.car.update()
        self.car_2.update()
        done = not self.car.is_alive()
        reward = self.car.get_reward()
        obs = self.car.get_grid_cell()

        done_car_2 = not self.car_2.is_alive()
        reward_car_2 = self.car_2.get_reward()
        obs_car_2 = self.car_2.get_grid_cell()

        pygame.event.pump()

        done_list = [done, done_car_2]
        obs_list = [obs, obs_car_2]
        reward_list = [reward, reward_car_2]
        return done_list, obs_list, reward_list, {}

    def reset(self):
        self.car = Car()
        return self.car.get_grid_cell()

        self.car_2 = Car()
        return self.car_2.get_grid_cell()

    def render(self, mode='human'):
        self.screen.blit(self.game_map, (2, 3))
        self.car.draw(self.screen)
        self.car_2.draw(self.screen)
        pygame.display.flip()
        pygame.time.wait(200)
        self.clock.tick(60)

    def close(self):
        pygame.quit()

    def train(self):
        # assign variables
        reward = 0
        x, y = self.car.position[0], self.car.position[1]
        reward_car_2 = 0
        x_2, y_2 = self.car_2.position[0], self.car_2.position[1]
        cars = []
        cars = cars.append(self.car)
        cars = cars.append(self.car_2)

        # punish cars for touching white area
        if self.game_map.get_at((int(x), int(y))) == (255, 255, 255):
            reward -= self.car.get_reward()

        # punish cars for colliding/being in same position
        if self.car.position == self.car_2.position:
            reward -= self.car.get_reward()
            reward_car_2 -= self.car_2.get_reward()

        # define bypass area
        upper_bypass_area = [(2, 1)] + [(x, 0) for x in range(2, 8)] + [(8, 1)]
        lower_bypass_area = [(2, 9)] + [(x, 9) for x in range(2, 8)] + [(8, 9)]

        # checking distance of car from two openings to upper bypasses
        distance_to_bypass1 = abs(self.car.position[0] - 2) + abs(self.car.position[1] - 1)
        distance_to_bypass2 = abs(self.car.position[0] - 8) + abs(self.car.position[1] - 1)

        # checking distance of car from two openings to lower bypasses
        distance_to_bypass1 = abs(self.car.position[0] - 2) + abs(self.car.position[1] - 9)
        distance_to_bypass2 = abs(self.car.position[0] - 8) + abs(self.car.position[1] - 9)

        # checking distance between cars
        distance_between_cars = abs(self.car.position[0] - self.car_2.position[0]) + abs(self.car.position[1] - self.car_2.position[1])

        # rewarding either car for going to upper bypass if too close
        if self.game_map.get_at((int(x), int(y))) == (0, 255, 0) and distance_between_cars <= 2 and distance_to_bypass1 <= 2:
            if self.car.position[1] < self.car_2.position[1]:
                if self.car.up() and any((int(x), int(y)) == coord for coord in upper_bypass_area):
                    reward += self.car.get_reward()
                else:
                    if self.car_2.up() and any((int(x_2), int(y_2)) == coord for coord in upper_bypass_area):
                        reward_car_2 += self.car.get_reward()
        elif self.game_map.get_at((int(x), int(y))) == (0, 255, 0) and distance_between_cars <= 2 and distance_to_bypass2 <= 2:
            if self.car.position[1] < self.car_2.position[1]:
                if self.car.up() and any((int(x), int(y)) == coord for coord in upper_bypass_area):
                    reward += self.car.get_reward()
                else:
                    if self.car_2.up() and any((int(x_2), int(y_2)) == coord for coord in upper_bypass_area):
                        reward_car_2 += self.car.get_reward()

        # rewarding either car for going to lower bypass if too close
        if self.game_map.get_at((int(x), int(y))) == (0, 255, 0) and distance_between_cars <= 2 and distance_to_bypass3 <= 2:
            if self.car.position[1] > self.car_2.position[1]:
                if self.car.down() and any((int(x), int(y)) == coord for coord in lower_bypass_area):
                    reward += self.car.get_reward()
                else:
                    if self.car_2.down() and any((int(x_2), int(y_2)) == coord for coord in lower_bypass_area):
                        reward_car_2 += self.car.get_reward()
        elif self.game_map.get_at((int(x), int(y))) == (0, 255, 0) and distance_between_cars <= 2 and distance_to_bypass4 <= 2:
            if self.car.position[1] > self.car_2.position[1]:
                if self.car.down() and any((int(x), int(y)) == coord for coord in lower_bypass_area):
                    reward += self.car.get_reward()
                else:
                    if self.car_2.down() and any((int(x_2), int(y_2)) == coord for coord in lower_bypass_area):
                        reward_car_2 += self.car.get_reward()
        # avoiding collision when cars are not near bypass (horizontal section only)
        elif distance_between_cars <= 2 and distance_to_bypass3 > 2 or distance_between_cars <= 2 and distance_to_bypass4 > 2:
            for i in cars:
                if self.game_map.get_at((int(cars[i].position[0]), int(cars[i].position[1]))) == (0, 255, 0):
                    if 3 <= cars[i].position[0] <= 7 and self.game_map.get_at((int(cars[i].position[0]), int(cars[i].position[1] + 1))) == (0, 255, 0):
                        # might add to above if statement "and if cars[i].position[1] < cars[i + 1].position[1]""
                        while abs(cars[i + 1].position[0] - cars[i].position[0]) <= 2:
                            if cars[i].position == cars[i].position:
                                reward += cars[i].get_reward()
                            elif cars[i + 1].position[1] + 1 == cars[i].position[1]:
                                reward += cars[i + 1].get_reward()
                    elif 3 <= cars[i].position[0] <= 7 and self.game_map.get_at((int(cars[i].position[0]), int(cars[i].position[1] - 1))) == (0, 255, 0):
                        # might add to above if statement "and if cars[i].position[1] > cars[i + 1].position[1]""
                        while abs(cars[i + 1].position[0] - cars[i].position[0]) <= 2:
                            if cars[i].position == cars[i].position:
                                reward += cars[i].get_reward()
                            elif cars[i + 1].position[1] - 1 == cars[i].position[1]:
                                reward += cars[i + 1].get_reward()

       # checking that red car goes clockwise
        if self.game_map.get_at((int(x), int(y))) != (255, 255, 255):
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

       # checking that black car goes counterclockwise
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

        return reward, reward_car_2

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
obs = env.reset()
'''
done = False
'''for _ in range(1000):
    action = env.action_space.sample()
    action_car_2 = env.action_space_2.sample()
    obs, reward, done, _ = env.step(action, action_car_2)

    env.render()

env.close()
'''
