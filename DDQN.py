from ReplayBuffer import ReplayBuffer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2

import numpy as np
import random

class DDQN():
  def __init__(self, state_space, action_space, learning_rate, gamma, epsilon, min_epsilon, decay_rate, 
  replay_buffer_maxlen, reg_factor):
    self.state_space = state_space
    self.action_space = action_space
    self.buffer = ReplayBuffer(replay_buffer_maxlen)
    self.learning_rate = learning_rate
    self.gamma = gamma
    self.epsilon = epsilon
    self.min_epsilon = min_epsilon
    self.decay_rate = decay_rate
    self.reg_factor = reg_factor
    self.model = self.build_model()
    self.target_model = self.build_model()

  def load_from_weights(self, weights_path):
    del self.model
    del self.target_model
    self.model = load_model(weights_path)
    self.target_model = load_model(weights_path)  

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
    state = np.reshape(state, [1, self.state_space]) # reshape for .predict()
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

  def epsilon_decay(self):
      if self.epsilon > self.min_epsilon:
        self.epsilon *= self.decay_rate