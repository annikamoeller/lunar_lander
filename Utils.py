import numpy as np
from collections import deque 
import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import gym
from Experience import Experience

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


def train_loop(agent, env, batch_size, training_start, target_update_freq, 
                max_episodes, max_steps, train_freq, backup_freq):

  # agent.load_from_weights('model_200.h5')

  avg_reward_tracker = AverageRewardTracker(100) 
  logger = Logger()

  for episode in range(max_episodes): # training loop
    state = env.reset() # vector of 8

    episode_reward = 0 # reward tracker

    for step in range(max_steps): # limit number of steps
      action = agent.select_action(state) # get action 
      next_state, reward, done, info = env.step(action) # next step
      episode_reward += reward # increment reward

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
    
    avg_reward_tracker.add(episode_reward)
    average = avg_reward_tracker.get_average()

    print(f"EPISODE {episode} finished in {step} steps, " )
    print(f"epsilon {agent.epsilon}, reward {episode_reward}. ")
    print(f"Average reward over last 100: {average} \n")
    logger.log(episode, step, episode_reward, average)
    if episode != 0 and episode % backup_freq == 0: # back up model every x steps 
      backup_model(agent.model, episode)
    
    agent.epsilon_decay()
  return agent.model, logger


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
      return model.predict(input)[0]

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