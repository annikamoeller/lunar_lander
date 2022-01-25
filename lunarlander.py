import tensorflow as tf
import gym
import os
import random

import numpy as np
import scipy

import pandas as pd
import matplotlib.pyplot as plt
from collections import deque

from ReplayBuffer import ReplayBuffer
from Experience import Experience
from Utils import Logger, AverageRewardTracker, plot, train_loop, render_game
from DDQN import DDQN

env = gym.make("LunarLander-v2")

state_space = env.observation_space.shape[0] #states
action_space = env.action_space.n # actions
learning_rate = 0.001
gamma = 0.99
epsilon = 0.5
min_epsilon = 0.01
decay_rate = 0.995 # per episode
buffer_maxlen = 200000
reg_factor = 0.001

batch_size = 128
training_start = 256 # which step to start training
target_update_freq = 1000
max_episodes = 20
max_steps = 20
train_freq = 4
backup_freq = 100

agent = DDQN(state_space, action_space, learning_rate, 
  gamma, epsilon, min_epsilon, decay_rate, buffer_maxlen, reg_factor)

model, progress_data = train_loop(agent, env, batch_size, training_start,
  target_update_freq, max_episodes, max_steps, train_freq, backup_freq)

plot(progress_data)
