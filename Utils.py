import numpy as np
from collections import deque 
import os
import pandas as pd
import matplotlib.pyplot as plt

class Logger():
  def __init__(self, network_type):
    self.file_name = f'metrics_{network_type}/training_progress.log'
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

def backup_model(model, episode, network_type):
    backup_file = f"checkpoints_{network_type}/model_{episode}.h5"
    print(f"Backing up model to {backup_file}")
    model.save(backup_file)

def plot(logger, network_type):
  data = pd.read_csv(logger.file_name, sep=';')
  plt.figure(figsize=(11,10))
  plt.plot(data['average'])
  plt.plot(data['reward'])
  plt.title('Reward per training episode', fontsize=22)
  plt.xlabel('Episode', fontsize=18)
  plt.ylabel('Reward', fontsize=18)
  plt.xticks(fontsize=18)
  plt.yticks(fontsize=18)
  plt.legend(['Average reward', 'Reward'], loc='upper left', fontsize=18)
  plt.savefig(f'metrics_{network_type}/reward_plot.png')
