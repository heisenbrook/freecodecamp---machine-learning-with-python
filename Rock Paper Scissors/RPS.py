import random
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete


#create the env

class RPS_Q_mod(gym.Env):
    def __init__(self):
        self.action_space = Discrete(3, start=0)
        self.observation_space = Discrete(9, start=0)
        self.reward = int()
        
    def reset(self, options=None, seed=None):
        self.state = options
        observation = self._get_obs()
        info = self._get_info()
        return observation, info
    
    def step(self, action):
        if ((self.state == 2.0) and (action == 0.0)) or ((self.state == 0.0) and (action == 1.0)) or ((self.state == 1.0) and (action == 2.0)):
            self.reward = 1
            terminated = True
        elif self.state == action:
            self.reward = -1
            terminated = True
        else:
            self.reward = -1
            terminated = True
        
        observation = action
        info = self._get_info()
        
        return observation, self.reward, terminated, False, info
    
    def _get_obs(self):
        return self.state
    
    def _get_info(self):
        return {"history": self.state}

#register the env   

gym.register(id='RPS-Q-mod-v0',
             entry_point=RPS_Q_mod
             )

env = gym.make('RPS-Q-mod-v0')
state_size = env.observation_space.n
action_size = env.action_space.n
Q = np.zeros((state_size, action_size))
STATES = {(0, 0): 0,
          (0, 1): 1,
          (0, 2): 2,
          (1, 0): 3,
          (1, 1): 4,
          (1, 2): 5,
          (2, 0): 6,
          (2, 1): 7,
          (2, 2): 8}
old_action = 0
alpha = 0.7
gamma = 0.01
epsilon = 0.82
discount = 0.99

def update_q_table(state, action, reward, new_state):
    old = Q[state][action]
    next_max = max(Q[new_state])
    Q[state][action] = (1 - alpha) * old + alpha * (reward + gamma * next_max) 
    
def sel_e_greedy_action(q, eps):
    if np.random.rand()  < eps:
        return env.action_space.sample()
    else:
        return np.argmax(q + np.random.randn(1, env.action_space.n)*(1/(num_episodes+1)))
        
num_episodes = 0

#main

def player(prev_play, opponent_history = [0]):
    global num_episodes
    global Q, STATES
    global epsilon, old_action
    
    num_episodes += 1
    
    match prev_play:
        case 'R':
            prev_play = 0
        case 'P':
            prev_play = 1
        case 'S':
            prev_play = 2
        case '':
            prev_play = random.choice(np.arange(3)) 
    
    guess, _ = env.reset(options=prev_play)
    opponent_history.append(guess)
    old_state = STATES[(opponent_history[num_episodes-1], old_action)] 
    
    new_action = sel_e_greedy_action(Q[old_state][:], epsilon)
    new_guess, reward, done, _, info = env.step(new_action)
    new_state = STATES[(opponent_history[num_episodes], new_guess)]
    update_q_table(old_state, new_action, reward, new_state)
    old_action = new_action

    guess = np.argmax(Q[new_state][:])
    
    epsilon = max(0.01, discount*epsilon)
    
    if num_episodes % 1000 == 0:
        Q = np.zeros((state_size, action_size))
        epsilon = 0.82

        
    match guess:
        case 0:
            guess = 'R'
        case 1:
            guess = 'P'
        case 2:
            guess = 'S'
    
    return guess
