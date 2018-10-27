#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 09:18:37 2018

@author: iraghu
"""

import gym
# !pip install box2d
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
# %matplotlib inline
from dqn_agent import DQNAgent
import pandas as pd
import sys
import time

# !python -m pip install pyvirtualdisplay
# from pyvirtualdisplay import Display
# display = Display(visible=0, size=(1400, 900))
# display.start()

is_ipython = 'inline' in plt.get_backend()
if is_ipython:
	from IPython import display

plt.ion()

env = gym.make('LunarLander-v2')
env.seed(0)
print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space.n)

df = pd.DataFrame(columns=['Episode', 'Steps', 'Reward', 'Epsilon', 'Alpha', 'Gamma', 'Tau'])
plot_data = [['Episode', 'Steps', 'Reward', 'Epsilon', 'Alpha', 'Gamma']]

agent = DQNAgent(state_size=8, action_size=4, seed=0, alpha=0.0001, gamma=0.99, tau=0.001)


def train(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, alpha=0.0001, gamma=0.99,
		  tau=0.001):
	"""Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
	scores = []  # list containing scores from each episode
	scores_window = deque(maxlen=100)  # last 100 scores
	eps = eps_start  # initialize epsilon
	column_labels = ['Episode', 'Steps', 'Reward', 'Epsilon', 'Alpha', 'Gamma']

	time1 = time.time()
	for i_episode in range(1, n_episodes + 1):
		state = env.reset()
		score = 0
		for t in range(max_t):
			action = agent.act(state, eps)
			next_state, reward, done, _ = env.step(action)
			agent.step(state, action, reward, next_state, done)
			state = next_state
			score += reward
			if done:
				break
		scores_window.append(score)  # save most recent score
		scores.append(score)  # save most recent score

		df.loc[len(df)] = [i_episode, t, score, eps, alpha, gamma, tau]
		eps = max(eps_end, eps_decay * eps)  # decrease epsilon

		print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
		if i_episode % 100 == 0:
			# time_1 = time.time()-time1
			print('\rEpisode {}\tAverage Score: {:.2f}\tTime Taken: {:.2f}'.format(i_episode, np.mean(scores_window),
																				   time.time() - time1))
		if np.mean(scores_window) >= 300.0:
			print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
																						 np.mean(scores_window)))
			torch.save(agent.qnetwork_local.state_dict(), 'checkpoint_Dueling_DDQN.pth')
			break
	return scores

scores = train(2000, 1000, 0.99, 0.01, 0.995, 5e-5, 0.99, agent.tau)
scores = train(2000, 1000, 0.99, 0.01, 0.995, 0.00001, 0.99, agent.tau)
scores = train(2000, 1000, 0.99, 0.01, 0.995, 0.0001, 0.99, agent.tau)

# df = pd.concat(plot_data, ignore_index=True)

df.to_csv('df_alphas_v1.csv', index=False)

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Average Reward per Episode')
plt.xlabel('Episode #')
plt.show()

from dqn_agent import Agent

# agent = Agent(state_size=8, action_size=4, seed=0)
## load the weights from file
# agent.qnetwork_local.load_state_dict(torch.load('checkpoint_Dueling_DDQN.pth', map_location=lambda storage, loc: storage))
#
# for i in range(3):
#    state = env.reset()
#    img = plt.imshow(env.render(mode='rgb_array'))
#    for j in range(200):
#        action = agent.act(state)
#        img.set_data(env.render(mode='rgb_array'))
#        plt.axis('off')
#        display.display(plt.gcf())
#        display.clear_output(wait=True)
#        state, reward, done, _ = env.step(action)
#        if done:
#            break

env.close()
