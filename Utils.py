import numpy as np
from collections import deque 
import os
import pandas as pd
import matplotlib.pyplot as plt

class Logger():
  def __init__(self, file_name='metrics/training_progress.log'):
    self.file_name = file_name
    self.reset_progress_file()

  def log(self, episode, steps, reward, average_reward):
    f = open(self.file_name, 'a')
    f.write(f"{episode};{steps};{reward};{average_reward}\n")
    f.close()

  def reset_progress_file(self):
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
    backup_file = f"checkpoints/model_{episode}.h5"
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
    plt.savefig('metrics/reward_plot.png')

