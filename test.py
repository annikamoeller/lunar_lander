import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('metrics_ddqn/training_progress.log', sep=';')
plt.figure(figsize=(11,10))
plt.plot(data['average'])
plt.plot(data['reward'])
plt.title('Reward per training episode', fontsize=22)
plt.xlabel('Episode', fontsize=18)
plt.ylabel('Reward', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(['Average reward', 'Reward'], loc='upper left', fontsize=18)
plt.savefig('metrics_ddqn/reward_plot_small.png')