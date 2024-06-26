# The example function below keeps track of the opponent's history and plays whatever the opponent played two plays ago. It is not a very good player so you will need to change the code to pass the challenge.
#from typing import Any, SupportsFloat
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import tensorflow as tf
from keras import layers, optimizers, losses, Sequential
from collections import deque, namedtuple
from itertools import islice

#create the env

class RPS(gym.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(3, start=0)
        self.observation_space = spaces.Box(low=0.0, high=3.0, shape=(1,), dtype=np.float32)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array(self.np_random.integers(0,3), dtype=np.float32)
        observation = self._get_obs()
        info = self._get_info()
        return observation, info
    
    def step(self, action):
        if (self.state == 1.0) and (action == 0.0):
            terminated = True
        elif (self.state == 2.0) and (action == 1.0):
            terminated = True
        elif (self.state == 0.0) and (action == 2.0):
            terminated = True
        else:
            terminated = False
        reward = 1 if terminated else 0
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, reward, terminated, False, info
    
    def _get_obs(self):
        return np.expand_dims(self.state, axis=0)
    
    def _get_info(self):
        return {"history": self.state}

#register the env   

gym.register(id='RPS-v0',
             entry_point=RPS
             )

env = gym.make('RPS-v0')
obs, info = env.reset(seed=None)
state_size = env.observation_space.shape
action_size = env.action_space.n


#Creation of model for DQN


main_nn = Sequential([
    layers.Input(shape=state_size),
    layers.Dense(units=64, activation='relu'),
    layers.Dense(units=64, activation='relu'),
    layers.Dense(units=64, activation='relu'),
    layers.Dense(units=64, activation='relu'),
    layers.Dense(units=action_size, activation='linear')
    ])
target_nn = Sequential([
    layers.Input(shape=state_size),
    layers.Dense(units=64, activation='relu'),
    layers.Dense(units=64, activation='relu'),
    layers.Dense(units=64, activation='relu'),
    layers.Dense(units=64, activation='relu'),
    layers.Dense(units=action_size, activation='linear')
    ])

optimizer = optimizers.Adam(1e-4)
mse = losses.MeanSquaredError()

experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
num_episodes = 0
epsilon = 0.1
batch_size = 32
num_step_up = 4
discount = 0.995
buffer = deque(maxlen=10000)
target_nn.set_weights(main_nn.get_weights())

#compute loss

def compute_loss(experiences, gamma, main_nn, target_nn, mse=mse):
   
    states, actions, rewards, next_states, done_vals = experiences
    
    max_qsa = tf.reduce_max(target_nn(next_states), axis=-1)
    y_targets = rewards + ((1-done_vals)*gamma*max_qsa)
    
    q_values = main_nn(states)
    q_values = tf.gather_nd(q_values, tf.stack([tf.range(q_values.shape[0]),
                                                tf.cast(actions, tf.int32)], axis=1))
        
    loss = mse(y_targets,q_values) 
    
    return loss


#train step

@tf.function
def agent(experiences, gamma):
    with tf.GradientTape() as tape:
        loss = compute_loss(experiences, gamma, main_nn, target_nn)
    gradients = tape.gradient(loss, main_nn.trainable_variables)
    optimizer.apply_gradients(zip(gradients, main_nn.trainable_variables))
    for target_weights, q_net_weights in zip(target_nn.weights, main_nn.weights):
        target_weights.assign(1e-3 * q_net_weights + (1.0 - 1e-3) * target_weights)   
    
#Selection of e greedy policy

def sel_e_greedy(state, eps):
    result = random.random()  
    if result < eps:
        return np.expand_dims(np.array(random.choice(np.arange(3)), dtype=np.float32), axis=0)
    else:
        return np.expand_dims(np.argmax(state[0]), axis=0)
        
#get experiences from mini batch

def get_experiences(memory_buffer, start, end):
    experiences = list(islice(memory_buffer, start, end))
    #experiences = random.sample(memory_buffer, batch_size)
    for e in experiences:
        if e is not None:
            states = tf.convert_to_tensor(e.state, dtype=tf.float32)
            actions = tf.convert_to_tensor(np.array(e.action, dtype=np.float32), dtype=tf.float32)
            rewards = tf.convert_to_tensor(np.array(e.reward, dtype=np.float32), dtype=tf.float32)
            next_states = tf.convert_to_tensor(e.next_state, dtype=tf.float32)
            done_vals = tf.convert_to_tensor(np.array(e.done, dtype=np.uint8).astype(np.float32), dtype=tf.float32)
    return (states, actions, rewards, next_states, done_vals)


#main

def player(prev_play, opponent_history=dict()):
    global num_episodes
    global epsilon
    num_episodes += 1
    guess, opponent_history = env.reset(seed=None)
    
    match prev_play:
        case 'R':
            prev_play = np.array([0], dtype=np.float32)
        case 'P':
            prev_play = np.array([1], dtype=np.float32)
        case 'S':
            prev_play = np.array([2], dtype=np.float32)
        case '':
            prev_play = np.array(random.choice(np.arange(3)), dtype=np.float32)
    
    prev_play_dim = np.expand_dims(prev_play, axis=0)
    prev_play_qn = main_nn(prev_play_dim)
    action = sel_e_greedy(prev_play_qn, epsilon)
    guess, reward, terminated, _, info  = env.step(action)
    opponent_history.update(info)
    
    buffer.append(experience(prev_play, action, reward, guess, terminated))
    
    if (num_episodes % num_step_up == 0) and (len(buffer) >= batch_size):
        start = num_episodes - batch_size
        end = num_episodes 
        experiences = get_experiences(buffer, start, end)
        agent(experiences, discount)
            
    epsilon = max(0.01, discount*epsilon)
    
    if num_episodes == 1000:
        epsilon = 0.1
        num_episodes = 0
        buffer.clear()
        
    match guess:
        case 0:
            guess = 'R'
        case 1:
            guess = 'P'
        case 2:
            guess = 'S'
    
    return guess
    

    
    
