import tensorflow as tf
import gym
import os
import random
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2

import numpy as np
import scipy

import pandas as pd
import matplotlib.pyplot as plt
from collections import deque

env = gym.make("LunarLander-v2")
state_space = env.observation_space.shape[0] #states
action_space = env.action_space.n # actions

class Experience():
  def __init__(self, state, action, reward, next_state, done):
    self.state = state
    self.action = action
    self.reward = reward
    self.next_state = next_state
    self.done = done
  
  def get_experience(self):
    return state, action, reward, next_state, done

class ReplayBuffer():
    def __init__(self, maxlen):
        self.buffer = deque(maxlen=maxlen)
        
    def add(self, experience):
        self.buffer.append(experience)
        
    def sample(self, batch_size):
        sample_size = min(len(self.buffer), batch_size)
        samples = random.choices(self.buffer, k=sample_size)
        return samples

    def length(self):
        return len(self.buffer)

class DDQN():
  def __init__(self, state_space, action_space, learning_rate, gamma, epsilon, 
    min_epsilon, decay_rate, replay_buffer_maxlen, reg_factor, weights, load_from_previous=False):
    self.state_space = state_space
    self.action_space = action_space
    self.buffer = ReplayBuffer(replay_buffer_maxlen)
    self.learning_rate = learning_rate
    self.gamma = gamma
    self.epsilon = epsilon
    self.min_epsilon = min_epsilon
    self.decay_rate = decay_rate
    self.reg_factor = reg_factor
    if load_from_previous:
      self.model = self.build_model()
      self.target_model = self.build_model()
    else:
      self.model = load_model(weights)
      self.target_model = load_model(weights)

  def build_model(self):

    model = Sequential([
    Dense(64, input_dim=self.state_space, activation="relu", kernel_regularizer=l2(self.reg_factor)),
    Dense(64, activation="relu", kernel_regularizer=l2(self.reg_factor)),
    Dense(64, activation="relu", kernel_regularizer=l2(self.reg_factor)),
    Dense(self.action_space, activation='linear', kernel_regularizer=l2(self.reg_factor))
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
    model.compile(loss="mean_squared_error", optimizer=optimizer)
    return model

  def update_target_weights(self):
    self.target_model.set_weights(self.model.get_weights())

  def select_action(self, state):
    if np.random.rand() <= self.epsilon:
      return random.randrange(self.action_space)
    state = np.reshape(state, [1, state_space]) # reshape for .predict()
    q_vals = self.model.predict(state)
    return np.argmax(q_vals[0])

  def train(self, inputs, targets):
    batch_size = len(targets)
    inputs = np.array(inputs).reshape(batch_size, self.state_space)
    targets = np.array(targets)
    self.model.fit(inputs, targets, epochs=1, batch_size=batch_size, verbose=0)

  def calculate_inputs_and_targets(self, experiences):
    inputs = []
    targets = []

    states = []
    next_states = []

    for experience in experiences:
      states.append(experience.state)
      next_states.append(experience.next_state)

    states = np.array(states) # array of states 
    next_states = np.array(next_states) # array of next states 

    # do predictions in batch outside loop so .predict doesn't have to be called repeatedly
    q_values_states = self.model.predict(states) # predict Q(s, a)
    q_values_next_states_local = self.model.predict(next_states) # predict Q(s', a') 
    q_values_next_states_target = self.target_model.predict(next_states) # predict Q(s', a') using target network

    for index, experience in enumerate(experiences):
      inputs.append(experience.state)
      q_values_local = q_values_next_states_local[index]
      q_values_target = q_values_next_states_target[index] 
      
      best_action_index = np.argmax(q_values_local) # index of max from model
      best_action_q_value = q_values_target[best_action_index] # value from target using index from model
      
      if experience.done:  
        target_val = experience.reward 
      else: 
        target_val = experience.reward + self.gamma * best_action_q_value
      
      target_vector = q_values_states[index]
      target_vector[experience.action] = target_val
      #print(f"target vector {target_vector}")
      targets.append(target_vector)
    return inputs, targets
    
  def backup_model(self):
      backup_file = f"model_{episode}.h5"
      print(f"Backing up model to {backup_file}")
      self.model.save(backup_file)

  def epsilon_decay(self):
      if self.epsilon > self.min_epsilon:
        self.epsilon *= self.decay_rate

class AverageRewardTracker():
  current_index = 0

  def __init__(self, episodes_to_avg=100):
    self.episodes_to_avg = episodes_to_avg
    self.tracker = deque(maxlen=episodes_to_avg)

  def add(self, reward):
    self.tracker.append(reward)

  def get_average(self):
    return np.average(self.tracker)

class FileLogger():

  def __init__(self, file_name='progress.log'):
    self.file_name = file_name
    self.clean_progress_file()

  def log(self, episode, steps, reward, average_reward):
    f = open(self.file_name, 'a+')
    f.write(f"{episode};{steps};{reward};{average_reward}\n")
    f.close()

  def clean_progress_file(self):
    if os.path.exists(self.file_name):
      os.remove(self.file_name)
    f = open(self.file_name, 'a+')
    f.write("episode;steps;reward;average\n")
    f.close()

buffer_maxlen = 200000
learning_rate = 0.001
batch_size = 128
training_start = 256 # which step to start training
max_episodes = 1000
max_steps = 1000
target_update_freq = 1000
backup_freq = 100
epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.995 # per episode
gamma = 0.99
train_freq = 4
reg_factor = 0.001

agent = DDQN(state_space, action_space, learning_rate, gamma, epsilon, min_epsilon, decay_rate, buffer_maxlen, reg_factor)
avg_reward_tracker = AverageRewardTracker(100) 
file_logger = FileLogger()

for episode in range(max_episodes): # training loop
  #print(f"Episode {episode} with epsilon = {agent.epsilon}")
  episode_reward = 0 
  state = env.reset() # vector of 8

  total_reward = 0 # reward tracker

  for step in range(1, max_steps + 1): # limit number of steps
    action = agent.select_action(state) # get action 
    next_state, reward, done, info = env.step(action) # next step
    total_reward += reward # increment reward

    if step == max_steps: # stop at max steps 
      print(f"Episode reached the maximum number of steps. {max_steps}")
      done = True

    experience = Experience(state, action, reward, next_state, done) # create new experience object
    agent.buffer.add(experience) # add experience to buffer

    state = next_state # update state

    if step % target_update_freq == 0: # update target weights every x steps 
      print("Updating target model")
      agent.update_target_weights()
    
    if (agent.buffer.length() >= training_start) & (step % train_freq == 0): # train agent every x steps
      batch = agent.buffer.sample(batch_size)
      inputs, targets = agent.calculate_inputs_and_targets(batch)
      agent.train(inputs, targets)

    if done: # stop if this action results in goal reached
      break
  
  avg_reward_tracker.add(total_reward)
  average = avg_reward_tracker.get_average()

  print(f"EPISODE {episode} finished in {step} steps, " )
  print(f"epsilon {agent.epsilon}, reward {total_reward}. ")
  print(f"Average reward over last 100: {average} \n")
  file_logger.log(episode, step, total_reward, average)
  if episode != 0 and episode % backup_freq == 0: # back up model every x steps 
    agent.backup_model()
  
  agent.epsilon_decay()

data = pd.read_csv(file_logger.file_name, sep=';')
plt.figure(figsize=(20,10))
plt.plot(data['average'])
plt.plot(data['reward'])
plt.title('Reward')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend(['Average reward', 'Reward'], loc='upper right')
plt.savefig('reward_plot.png')