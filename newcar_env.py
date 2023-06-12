import math
import gym
from gym import spaces
import numpy as np
import pygame
import neat
import random


WIDTH = 1920
HEIGHT = 1080
CAR_SIZE_X = 60
CAR_SIZE_Y = 60
BORDER_COLOR = (255, 255, 255)
BYPASS_COLOR = (0, 0, 255)

class CarEnv(gym.Env):
    def __init__(self):
        super(CarEnv, self).__init__()

        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)

        self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.Discrete(5 * 5)

        self.clock = pygame.time.Clock()
        self.game_map = pygame.image.load('final-grid-map.png').convert()
        self.car = Car()
        self.non_agent = NonAgent()

        self.config_path = "./config.txt"
        self.config = neat.config.Config(neat.DefaultGenome,
                                         neat.DefaultReproduction,
                                         neat.DefaultSpeciesSet,
                                         neat.DefaultStagnation,
                                         self.config_path)

        self.population = neat.Population(self.config)
        self.population.add_reporter(neat.StdOutReporter(True))
        self.stats = neat.StatisticsReporter()
        self.population.add_reporter(self.stats)

        self.current_generation = 0

    def step(self, action):
        if action == 0:
            self.car.speed -= 5
            self.car.angle += 90  # Left
        elif action == 1:
            self.car.speed += 2
            self.bypass()
        elif action == 2:
            self.car.speed -= 5
            self.car.angle -= 90  # Right
        else:
            self.car.speed = 0  # Stop

        self.car.update(self.game_map)
        self.non_agent.update(self.game_map)
        done = not self.car.is_alive()
        reward = self.car.get_reward()
        obs = self.car.get_data()

        pygame.event.pump()  # Process Pygame events

        return obs, reward, done, {}

    def reset(self):
        self.car = Car()
        self.current_generation += 1
        self.population.run(self.run_simulation, 1)
        return self.car.get_data()

    def render(self, mode='human'):
        self.screen.blit(self.game_map, (0, 0))
        self.car.draw(self.screen)
        self.non_agent.draw(self.screen)
        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        pygame.quit()

    def run_simulation(self, genomes, config):
        nets = []
        cars = []

        for _, g in genomes:
            net = neat.nn.FeedForwardNetwork.create(g, config)
            nets.append(net)
            g.fitness = 0
            cars.append(Car())

        counter = 0

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit(0)

            for i, car in enumerate(cars):
                output = nets[i].activate(car.get_data())
                choice = output.index(max(output))
                if choice == 0:
                    self.car.speed -= 5
                    self.car.angle += 90  # Left
                elif choice == 1:
                    self.car.speed += 2
                    self.bypass()
                elif choice == 2:
                    self.car.speed -= 5
                    self.car.angle -= 90  # Right
                else:
                    self.car.speed = 0

            still_alive = 0
            for i, car in enumerate(cars):
                if car.is_alive():
                    still_alive += 0
                    car.update(self.game_map)
                    genomes[i][1].fitness += 0

            if still_alive == 0:
                break

            counter += 1
            if counter == 30 * 40:
                break

            for i, car in enumerate(cars):
                if car.position[0] in range(349, 523) or car.position[0] in range(1220, 1394) and self.non_agent.position[1] - car.position[1] < 100 and self.non_agent.position[0] - car.position[0] < 100:
                    if self.bypass():
                        genomes[i][1].fitness += car.get_reward()

            self.render()

    def bypass(self):
        x, y = self.car.position[0], self.car.position[1]
        position_color = self.game_map.get_at((int(x), int(y)))
        return position_color == BYPASS_COLOR

# Custom Car class

class Car:
    def __init__(self):
        self.sprite = pygame.image.load('car.png').convert() # Convert Speeds Up A Lot
        self.sprite = pygame.transform.scale(self.sprite, (CAR_SIZE_X, CAR_SIZE_Y))
        self.rotated_sprite = self.sprite

        self.position = [269, 773]
        self.angle = 0
        self.speed = 20

        self.speed_set = False

        self.center = [self.position[0] + CAR_SIZE_X / 2, self.position[1] + CAR_SIZE_Y / 2] # Calculate Center

        self.radars = []
        self.drawing_radars = []

        self.alive = True

        self.distance = 0
        self.time = 0


    def draw(self, screen):
        screen.blit(self.rotated_sprite, self.position)
        self.draw_radar(screen)

    def draw_radar(self, screen):
        for radar in self.radars:
            position = radar[0]
            pygame.draw.line(screen, (255, 255, 0), self.center, position, 1)
            pygame.draw.circle(screen, (255, 255, 0), position, 5)

    def check_collision(self, game_map):
        self.alive = True
        for point in self.corners:
            if game_map.get_at((int(point[0]), int(point[1]))) == BORDER_COLOR:
                self.alive = False
                break

    def check_radar(self, degree, game_map):
        length = 0
        x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
        y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)


        while length < 300:
            length = length + 1
            x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
            y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

        dist = int(math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2)))
        self.radars.append([(x, y), dist])

    def update(self, game_map):
        if not self.speed_set:
            self.speed = 20
            self.speed_set = True

        self.rotated_sprite = self.rotate_center(self.sprite, self.angle)
        self.position[0] += math.cos(math.radians(360 - self.angle)) * self.speed
        self.position[0] = max(self.position[0], 20)
        self.position[0] = min(self.position[0], WIDTH - 120)

        self.distance += self.speed
        self.time += 1

        self.position[1] += math.sin(math.radians(360 - self.angle)) * self.speed
        self.position[1] = max(self.position[1], 20)
        self.position[1] = min(self.position[1], HEIGHT - 120)

        self.center = [int(self.position[0]) + CAR_SIZE_X / 2, int(self.position[1]) + CAR_SIZE_Y / 2]

        length = 0.5 * CAR_SIZE_X
        left_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 30))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 30))) * length]
        right_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 150))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 150))) * length]
        left_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 210))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 210))) * length]
        right_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 330))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 330))) * length]
        self.corners = [left_top, right_top, left_bottom, right_bottom]

        self.check_collision(game_map)
        self.radars.clear()

        for d in range(-90, 120, 45):
            self.check_radar(d, game_map)

    def get_data(self):
        radars = self.radars
        return_values = [0, 0, 0, 0, 0]
        for i, radar in enumerate(radars):
            return_values[i] = int(radar[1] / 30)

        return return_values

    def is_alive(self):
        return self.alive

    def get_reward(self):
        return self.distance / (CAR_SIZE_X / 2)

    def rotate_center(self, image, angle):
        rectangle = image.get_rect()
        rotated_image = pygame.transform.rotate(image, angle)
        rotated_rectangle = rectangle.copy()
        rotated_rectangle.center = rotated_image.get_rect().center
        rotated_image = rotated_image.subsurface(rotated_rectangle).copy()
        return rotated_image

class NonAgent:
    def __init__(self):
        pygame.init()
        self.sprite = pygame.image.load('non_agent_car.png').convert()
        self.sprite = pygame.transform.scale(self.sprite, (CAR_SIZE_X, CAR_SIZE_Y))
        self.rotated_sprite = self.sprite

        self.position = [773, 269]
        self.angle = 0
        self.speed = 20

        self.center = [self.position[0] + CAR_SIZE_X / 2, self.position[1] + CAR_SIZE_Y / 2]

        self.radars = []
        self.drawing_radars = []
        self.speed_set = False
        self.distance = 0
        self.time = 0

    def draw(self, screen):
        screen.blit(self.rotated_sprite, self.position)
        self.draw_radar(screen)

    def draw_radar(self, screen):
        for radar in self.radars:
            position = radar[0]
            pygame.draw.line(screen, (255, 255, 0), self.center, position, 1)
            pygame.draw.circle(screen, (255, 255, 0), position, 5)

    def check_collision(self, game_map):
        self.alive = True
        for point in self.corners:
            if game_map.get_at((int(point[0]), int(point[1]))) == BORDER_COLOR:
                self.alive = False
                break


    def update(self, game_map):
        if not self.speed_set:
            self.speed = 20
            self.speed_set = True

        self.rotated_sprite = self.rotate_center(self.sprite, self.angle)
        self.position[0] += math.cos(math.radians(360 - self.angle)) * self.speed
        self.position[0] = max(self.position[0], 20)
        self.position[0] = min(self.position[0], WIDTH - 120)

        self.distance += self.speed
        self.time += 1

        self.position[1] += math.sin(math.radians(360 - self.angle)) * self.speed
        self.position[1] = max(self.position[1], 20)
        self.position[1] = min(self.position[1], WIDTH - 120)

        self.center = [int(self.position[0]) + CAR_SIZE_X / 2, int(self.position[1]) + CAR_SIZE_Y / 2]

        length = 0.5 * CAR_SIZE_X
        left_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 30))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 30))) * length]
        right_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 150))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 150))) * length]
        left_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 210))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 210))) * length]
        right_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 330))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 330))) * length]
        self.corners = [left_top, right_top, left_bottom, right_bottom]

        if self.position[0] in range(1513, 1745):
            if self.position[1] in range(215, 389):
                self.angle = -90
                self.rotate_center(self.sprite, self.angle)
                self.position[0] = random.randint(1513 + 30, 1745 - 30)
            elif self.position[1] in range(612, 839):
                self.angle = -180
                self.rotate_center(self.sprite, self.angle)
                self.position[1] = random.randint(612 + 30, 839 - 30)
        elif self.position[0] in range(174, 405):
            if self.position[1] in range(612, 839):
                self.angle = 90
                self.rotate_center(self.sprite, self.angle)
                self.position[0] = random.randint(174 + 30, 405 - 30)
            elif self.position[1] in range(215, 405):
                self.angle = 360
                self.rotate_center(self.sprite, self.angle)
                self.position[1] = random.randint(215 + 30, 389 - 30)

    def rotate_center(self, image, angle):
        rectangle = image.get_rect()
        rotated_image = pygame.transform.rotate(image, angle)
        rotated_rectangle = rectangle.copy()
        rotated_rectangle.center = rotated_image.get_rect().center
        rotated_image = rotated_image.subsurface(rotated_rectangle).copy()
        return rotated_image

env = CarEnv()
obs = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, done, _ = env.step(action)
    env.render()

env.close()
