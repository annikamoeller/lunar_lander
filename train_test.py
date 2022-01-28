import gym
from Experience import Experience
from Utils import Logger, AverageRewardTracker, backup_model, plot
from DDQN import DDQN
import numpy as np
from tensorflow.keras.models import load_model

# training function
def train_lander(agent, env, batch_size, training_start, target_update_freq, 
  max_episodes, max_steps, train_freq, backup_freq):
  
  step_counter = 0
  avg_reward_tracker = AverageRewardTracker(100) 
  logger = Logger()

  for episode in range(max_episodes): # training loop
    state = env.reset() # vector of 8

    episode_reward = 0 # reward tracker

    for step in range(max_steps): # limit number of step
      step_counter += 1
      action = agent.select_action(state) # get action 
      next_state, reward, done, info = env.step(action) # next step
      episode_reward += reward # increment reward

      if step == max_steps: # stop at max steps 
        print(f"Episode reached the maximum number of steps. {max_steps}")
        done = True

      experience = Experience(state, action, reward, next_state, done) # create new experience object
      agent.buffer.add(experience) # add experience to buffer

      state = next_state # update state

      if step_counter % target_update_freq == 0: # update target weights every x steps 
        print("Updating target model step: ", step)
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

  plot(logger)

# testing function
def test_lander(model_filename, max_episodes, max_steps, render=False):
  env = gym.make("LunarLander-v2")
  trained_model = load_model(model_filename)

  def get_q_values(model, state):
      state = np.array(state)
      state = np.reshape(state, [1, 8])
      return model.predict(state)

  def select_best_action(q_values):
      return np.argmax(q_values)

  rewards = []
  for episode in range(max_episodes):
      state = env.reset()

      episode_reward = 0

      for step in range(1, max_steps+1):
          if render:
            env.render()

          q_values = get_q_values(trained_model, state)
          action = select_best_action(q_values)
          next_state, reward, done, info = env.step(action)

          episode_reward += reward

          if step == max_steps:
              print(f"Episode reached the maximum number of steps. {max_steps}")
              done = True

          state = next_state

          if done:
              break

      print(f"episode {episode} finished in {step} steps with reward {episode_reward}.")
      rewards.append(episode_reward)

  print("Average reward: ", np.average(rewards))