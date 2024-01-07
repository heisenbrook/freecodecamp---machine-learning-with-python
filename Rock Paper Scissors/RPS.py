# The example function below keeps track of the opponent's history and plays whatever the opponent played two plays ago. It is not a very good player so you will need to change the code to pass the challenge.
from typing import Any, SupportsFloat
import random
import numpy as np
import gymnasium as gym 
from gymnasium import spaces

#create the env

class RPS(gym.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(3,)
        self.observation_space = spaces.Box(low=np.array(0), high=np.array(3))
        self.state = random.randint(0,2)
        
    def step(self, action):
        if (self.state == 1) and (action == 0):
            reward = 1
            self.state = action
        elif (self.state == 2) and (action == 1):
            reward = 1
            self.state = action
        elif (self.state == 0) and (action == 2):
            reward = 1
            self.state = action
        else:
            reward = 0
        done = True
        
        return self.state, reward, done, {}
    
    def reset(self, seed=None, options=dict()):
        super().reset(seed=seed)
        self.state = random.randint(0,2)
        return self.state, options
 
#register the env   

gym.register(id='RPS-v0',
             entry_point=RPS
             )

env = gym.make('RPS-v0')
obs, info = env.reset(seed=123)


def player(prev_play, opponent_history=[]):
    
    match prev_play:
        case 'R':
            prev_play = 0
        case 'P':
            prev_play = 1
        case 'S':
            prev_play = 2
    
    guess, _, done, obs  = env.step(prev_play)
    
    match guess:
        case 0:
            guess = 'R'
        case 1:
            guess = 'P'
        case 3:
            guess = 'S'
            
    
    return guess
