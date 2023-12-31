from .snake import Snake
from .apple import Apple
import pygame
import random 
# import gymnasium as gym
import numpy as np
import torch

import gymnasium.utils as utils 

import gym

class SnakeEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}
    def __init__(self, config = {}, **kwargs):
        utils.EzPickle.__init__(self, config, **kwargs)
        self.screen_width = config.get('screen_width', 300)
        self.screen_height = config.get('screen_height', 300)
        self.block_size =  config.get('block_size', 20)
        self.snake = Snake(self.screen_width, self.screen_height, self.block_size)
        self.apple = self.generate_apple()
        self.max_steps = config.get('max_steps', 100)*self.screen_width * self.screen_height // self.block_size**2
        self.render_mode = config.get('render_mode',"rgb_array")
        if(self.render_mode == "human"):
            pygame.init()
            self.screen =  pygame.display.set_mode((self.screen_width, self.screen_height))

        self.observation_space = gym.spaces.Box(low = 0.0 , high = 1.0 , shape =  ((1+ self.screen_width* self.screen_height // (self.block_size**2)  )*2,) , dtype = np.float32)
        self.action_space = gym.spaces.Discrete(4)
        self.seed = config.get('seed', lambda x: random.randint(0, 2**32 - 1))
    def normalized_distance(self, a, b):   
        disx = abs(a[0] - b[0])/self.screen_width
        disy = abs(a[1] - b[1])/self.screen_height
        distance = np.sqrt(disx**2 + disy**2) / np.sqrt(2) 
        return distance
    

    def compute_reward(self):
        rewards = {
            "apple": 1 if self.snake.head == self.apple.position else 0,
            "death": -1 if self.snake.head in self.snake.body[:-1] or not self.in_grid_bounds(self.snake.head) else 0,
            "getting_closer": 1.0 if self.latest_distance  > self.normalized_distance(self.snake.head, self.apple.position) else -2.0,
            "normalized_distance": 1.0-(self.normalized_distance(self.snake.head, self.apple.position))**0.25
        }
        self.reward = rewards['death'] + rewards["apple"]  + (rewards["normalized_distance"] if self.latest_distance  > self.normalized_distance(self.snake.head, self.apple.position) else -1.1)/10.0
        


    def reset(self, seed=None):
        # This function resets the game state and returns the initial observation
        # of the game state.
        # Initialize the snake and apple
        #set seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.snake = Snake(self.screen_width, self.screen_height, self.block_size)
        start = (np.random.randint(1,self.screen_width//self.block_size - 1 ) , np.random.randint(1,self.screen_height//self.block_size - 1 ))

        self.snake.head = (start[0] * self.block_size, start[1] * self.block_size)
        self.snake.body = [self.snake.head]
        self.latest_distance = 1 
        self.snake.direction = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
        self.snake.grow = False
        self.apple = self.generate_apple()
        self.score = 0
        self.done = False
        self.steps = 0

        # Return the initial observation of the game state
        return self._get_obs()


    
    def step(self, action):
        # if self.done:
        #     self.reset()
        #     self.step(action)

        # Change snake direction
        self.snake.change_direction(action)
        # Move snake
        self.snake.move()
        self.compute_reward()
        self.steps += 1

        if self.snake.head == self.apple.position:
            self.score += 1
            self.snake.grow = True
            self.apple = self.generate_apple()

        # Check if snake collides with wall
        if self.snake.head[0] < 0 or self.snake.head[0] >= self.screen_width or self.snake.head[1] < 0 or self.snake.head[1] >= self.screen_height:
            self.done = True

        # Check if snake collides with body
        if self.snake.head in self.snake.body[:-1]:
            self.done = True    

        if self.steps > self.max_steps:
            self.done = True

            
        infos = {
            "episode":{
                "reward": self.reward,
                "score": 1 if self.snake.grow else 0,
                "distance_to_goal": self.latest_distance,
            }
            
        }
        return self._get_obs(),self.reward, self.done, infos
    
    # Make a random apple
    def generate_apple(self):
        # Make a random x and y location
        x = random.randint(0, (self.screen_width - self.block_size) // self.block_size) * self.block_size
        y = random.randint(0, (self.screen_height - self.block_size) // self.block_size) * self.block_size
        # Make an apple with those x and y values

        # Check if the apple is in the snake's body
        # If it is, generate a new apple
        while (x,y) in self.snake.body or (x,y) == self.snake.head:
            x = random.randint(0, (self.screen_width - self.block_size) // self.block_size) * self.block_size
            y = random.randint(0, (self.screen_height - self.block_size) // self.block_size) * self.block_size
        return Apple(x, y, self.block_size)
        
    def render(self, mode="rgb_array"):
        if mode == "rgb_array":
            image = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
            # Make the image white
            image[:, :, :] = [255, 255, 255]
            # Red for the apple
            image[self.apple.position[1] : self.apple.position[1] + self.block_size, self.apple.position[0] : self.apple.position[0] + self.block_size, :] = [255, 0, 0]

            # Green for the snake
            for pos in self.snake.body:
                image[pos[1] : pos[1] + self.block_size, pos[0] : pos[0] + self.block_size, :] = [0, 255, 0]
            # Blue for the head
            image[self.snake.head[1] : self.snake.head[1] + self.block_size, self.snake.head[0] : self.snake.head[0] + self.block_size, :] = [0, 0, 255]
            return image
        else:
            # Fill the screen with white background
            self.screen.fill((255, 255, 255))
            # Draw the snake on the screen
            self.snake.draw(self.screen)
            # Draw the apple on the screen
            self.apple.draw(self.screen)
            # Update the screen to show the changes
            pygame.display.update()  # Missing in the original code


    def in_grid_bounds(self, pos):
        return 0 <= pos[0] < self.screen_width and 0 <= pos[1] < self.screen_height

    
    def _get_obs(self):
        obs = torch.zeros((1+ self.screen_width* self.screen_height // (self.block_size**2)  )*2, dtype=torch.float32)
        obs[0] = self.apple.position[0]/ self.screen_width
        obs[1] = self.apple.position[1]/ self.screen_height
        for i in range(len(self.snake.body)-1,-1,-1):
            obs[2*i+2] = self.snake.body[i][0]/ self.screen_width
            obs[2*i+3] = self.snake.body[i][1]/ self.screen_height
        return obs[:self.observation_space.shape[0]]
    
