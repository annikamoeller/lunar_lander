import numpy as np
import gym
from tensorflow.keras.models import load_model

def test_lander(model_filename, render=False):
  env = gym.make("LunarLander-v2")
  trained_model = load_model(model_filename)

  evaluation_max_episodes = 10
  evaluation_max_steps = 1000

  def get_q_values(model, state):
      state = np.array(state)
      state = np.reshape(state, [1, 8])
      return model.predict(state)

  def select_best_action(q_values):
      return np.argmax(q_values)

  rewards = []
  for episode in range(evaluation_max_episodes):
      state = env.reset()

      episode_reward = 0

      for step in range(1, evaluation_max_steps+1):
          if render:
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
  print("Average reward: ", np.average(rewards))

test_lander('checkpoints/model_600.h5', False)
