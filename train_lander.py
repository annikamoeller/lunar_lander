import gym
from Experience import Experience
from Utils import Logger, AverageRewardTracker, backup_model, plot
from DDQN import DDQN

env = gym.make("LunarLander-v2")

state_space = env.observation_space.shape[0] #states
action_space = env.action_space.n # actions
learning_rate = 0.001
gamma = 0.99
epsilon = 0.95
min_epsilon = 0.01
decay_rate = 0.995 # per episode
buffer_maxlen = 200000
reg_factor = 0.001

batch_size = 128
training_start = 256 # which step to start training
target_update_freq = 1000
max_episodes = 1000
max_steps = 500
train_freq = 4
backup_freq = 100
step_counter = 0

agent = DDQN(state_space, action_space, learning_rate, 
  gamma, epsilon, min_epsilon, decay_rate, buffer_maxlen, reg_factor)

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
