from train_test import train_lander, test_lander
from params import agent_params, train_params, test_params
from DDQN import DDQN
import argparse
import gym

# parse command line arguments
parser = argparse.ArgumentParser(add_help=True)
parser.add_argument('--run_type', type=str, required=True, help = 'Type "test" or "train"')
args = parser.parse_args()
run_type = args.run_type 

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

model_path = 'checkpoints/model_900.h5'
test_max_episodes = 100
test_max_steps = 500
render_lander = False

agent = DDQN(state_space, action_space, learning_rate, 
    gamma, epsilon, min_epsilon, decay_rate, buffer_maxlen, reg_factor)

if run_type == "train":

    train_lander(agent, env, batch_size, training_start, 
        target_update_freq, max_episodes, max_steps, train_freq, backup_freq)

if run_type == "test":
    test_lander(model_path, test_max_episodes, test_max_steps, render_lander)