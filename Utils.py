import numpy as np
from collections import deque 
import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import gym

class Logger():

  def __init__(self, file_name='training_progress.log'):
    self.file_name = file_name
    self.clean_progress_file()

  def log(self, episode, steps, reward, average_reward):
    f = open(self.file_name, 'a')
    f.write(f"{episode};{steps};{reward};{average_reward}\n")
    f.close()

  def clean_progress_file(self):
    if os.path.exists(self.file_name):
      os.remove(self.file_name)
    f = open(self.file_name, 'a')
    f.write("episode;steps;reward;average\n")
    f.close()

class AverageRewardTracker():
  current_index = 0

  def __init__(self, episodes_to_avg=100):
    self.episodes_to_avg = episodes_to_avg
    self.tracker = deque(maxlen=episodes_to_avg)

  def add(self, reward):
    self.tracker.append(reward)

  def get_average(self):
    return np.average(self.tracker)

def backup_model(model, episode):
    backup_file = f"model_{episode}.h5"
    print(f"Backing up model to {backup_file}")
    model.save(backup_file)

def plot(logger):
    data = pd.read_csv(logger.file_name, sep=';')
    plt.figure(figsize=(20,10))
    plt.plot(data['average'])
    plt.plot(data['reward'])
    plt.title('Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend(['Average reward', 'Reward'], loc='upper right')
    plt.savefig('reward_plot.png')

def render_game(model_filename):
  env = gym.make("LunarLander-v2")
  trained_model = load_model(model_filename)

  evaluation_max_episodes = 10
  evaluation_max_steps = 1000

  def get_q_values(model, state):
      state = np.array(state)
      state = np.reshape(state, [1, 8])
      print(state)
      return model.predict(state)

  def select_best_action(q_values):
      return np.argmax(q_values)

  rewards = []
  for episode in range(evaluation_max_episodes):
      state = env.reset()

      episode_reward = 0

      for step in range(evaluation_max_steps):
          env.render()
          q_values = get_q_values(trained_model, state)
          action = select_best_action(q_values)
          next_state, reward, done, info = env.step(action)

          episode_reward += reward

          if step == evaluation_max_steps:
              print(f"Episode reached the maximum number of steps. {evaluation_max_steps}")
              done = True

          state = next_state

          if done:
              break

      print(f"episode {episode} finished in {step} steps with reward {episode_reward}.")
      rewards.append(episode_reward)

  print(rewards)
  print("Average reward: " + np.average(rewards))
