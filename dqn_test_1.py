import numpy as np
import gym
from collections import deque
import pandas as pd
import time
from gym import wrappers
import tensorflow as tf
from os import path
import random

import warnings # This ignore all the warning messages that are normally printed during the training because of skiimage
warnings.filterwarnings('ignore')

# Hyperparameters
gamma = 0.99				# reward discount factor
h1 = 512					# hidden layer 1 size
h2 = 512					# hidden layer 2 size
h3 = 512					# hidden layer 3 size
alpha = 0.0001				# learning rate
lr_decay = 1			# learning rate decay (per episode)
l2_reg = 0.0001			# L2 regularization factor
dropout = 0				# dropout rate (0 = no dropout)
num_episodes = 500		# number of episodes
max_steps_ep = 10000	# default max number of steps per episode, Lunar Lander has a hard coded limit of 1000
slow_target_burnin = 1000		# number of steps where slow target weights are tied to current network weights
update_slow_target_every = 100	# number of steps to use slow target as target before updating it to latest weights
train_every = 1			# number of steps to run the policy (and collect experience) before updating network weights
replay_memory_capacity = int(1e6)	# capacity of experience replay memory
minibatch_size = 1000	# size of minibatch from experience replay memory for updates
epsilon_start = 0.99	# probability of random action at start
epsilon_end = 0.05		# minimum probability of random action after linear decay period
epsilon_decay_length = 100000		# number of steps over which to linearly decay epsilon
epsilon_decay_exp = 0.97	# exponential decay rate after reaching epsilon_end (per episode)

# Environment
env = gym.make("LunarLander-v2")
state_dim = np.prod(np.array(env.observation_space.shape)) 	# Get total number of dimensions in state
n_actions = env.action_space.n 								# Assuming discrete action space

# set seeds to 0
env.seed(0)
np.random.seed(0)


## Tensorflow parameters
tf.reset_default_graph()
tf.set_random_seed(0)

# Create Placeholders
state_ph = tf.placeholder(dtype=tf.float32, shape=[None,state_dim]) # input to Q network
next_state_ph = tf.placeholder(dtype=tf.float32, shape=[None,state_dim]) # input to slow target network
action_ph = tf.placeholder(dtype=tf.int32, shape=[None]) # action indices (indices of Q network output)
reward_ph = tf.placeholder(dtype=tf.float32, shape=[None]) # rewards (go into target computation)
is_not_terminal_ph = tf.placeholder(dtype=tf.float32, shape=[None]) # indicators (go into target computation)
is_training_ph = tf.placeholder(dtype=tf.bool, shape=()) # for dropout

# episode counter
episodes = tf.Variable(0.0, trainable=False, name='episodes')
episode_inc_op = episodes.assign_add(1)

# will use this to initialize both Q network and slowly-changing target network with same structure
# TODO Check if there are better ways to generate networks like conv2d
def generate_network(s, trainable, reuse):
	hidden = tf.layers.dense(s, h1, activation = tf.nn.relu, trainable = trainable, name = 'dense', reuse = reuse)
	hidden_drop = tf.layers.dropout(hidden, rate = dropout, training = trainable & is_training_ph)
	hidden_2 = tf.layers.dense(hidden_drop, h2, activation = tf.nn.relu, trainable = trainable, name = 'dense_1', reuse = reuse)
	hidden_drop_2 = tf.layers.dropout(hidden_2, rate = dropout, training = trainable & is_training_ph)
	hidden_3 = tf.layers.dense(hidden_drop_2, h3, activation = tf.nn.relu, trainable = trainable, name = 'dense_2', reuse = reuse)
	hidden_drop_3 = tf.layers.dropout(hidden_3, rate = dropout, training = trainable & is_training_ph)
	action_values = tf.squeeze(tf.layers.dense(hidden_drop_3, n_actions, trainable = trainable, name = 'dense_3', reuse = reuse))
	return action_values

with tf.variable_scope('q_network') as scope:
	# Q network applied to state_ph
	q_action_values = generate_network(state_ph, trainable = True, reuse = False)
	# Q network applied to next_state_ph (for double Q learning)
	q_action_values_next = tf.stop_gradient(generate_network(next_state_ph, trainable = False, reuse = True))

with tf.variable_scope('slow_target_network', reuse=False):
	# use stop_gradient to treat the output values as constant targets when doing backprop
	slow_target_action_values = tf.stop_gradient(generate_network(next_state_ph, trainable = False, reuse = False))

# isolate vars for each network
q_network_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='q_network')
slow_target_network_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='slow_target_network')

# update values for slowly-changing target network to match current critic network
update_slow_target_ops = []
for i, slow_target_var in enumerate(slow_target_network_vars):
	update_slow_target_op = slow_target_var.assign(q_network_vars[i])
	update_slow_target_ops.append(update_slow_target_op)

update_slow_target_op = tf.group(*update_slow_target_ops, name='update_slow_target')

targets = reward_ph + is_not_terminal_ph * gamma * \
	tf.gather_nd(slow_target_action_values, tf.stack((tf.range(minibatch_size), tf.cast(tf.argmax(q_action_values_next, axis=1), tf.int32)), axis=1))

# Estimated Q values for (s,a) from experience replay
estim_taken_action_vales = tf.gather_nd(q_action_values, tf.stack((tf.range(minibatch_size), action_ph), axis=1))

# Loss function (with regularization) # TODO Need to be changed
loss = tf.reduce_mean(tf.square(targets - estim_taken_action_vales))
for var in q_network_vars:
	if not 'bias' in var.name:
		loss += l2_reg * 0.5 * tf.nn.l2_loss(var)

# Create optimizer # TODO Updating this mmay increase performance
train_op = tf.train.AdamOptimizer(alpha).minimize(loss)

# initialize session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Start Training

total_steps = 0
experience = deque(maxlen=replay_memory_capacity)

epsilon = epsilon_start
epsilon_linear_step = (epsilon_start-epsilon_end)/epsilon_decay_length
df = pd.DataFrame(columns=['Episode', 'Reward', 'Steps', 'Epsilon', 'Gamma', 'Alpha'])

for ep in range(num_episodes):

	total_reward = 0
	steps_in_ep = 0

	# Initial state
	observation = env.reset()
	# env.render() stopped rendering performance issues

	for t in range(max_steps_ep):

		# choose action according to epsilon-greedy policy wrt Q
		if np.random.random() < epsilon:
			action = np.random.randint(n_actions)
		else:
			q_s = sess.run(q_action_values,
				feed_dict = {state_ph: observation[None], is_training_ph: False})
			action = np.argmax(q_s)

		# take step
		next_observation, reward, done, _info = env.step(action)
		# env.render()
		total_reward += reward

		# add this to experience replay buffer
		experience.append((observation, action, reward, next_observation,
			# is next_observation a terminal state?
			0.0 if done else 1.0))

		# update the slow target's weights to match the latest q network if it's time to do so
		if total_steps%update_slow_target_every == 0:
			_ = sess.run(update_slow_target_op)

		# update network weights to fit a minibatch of experience
		if total_steps%train_every == 0 and len(experience) >= minibatch_size:

			# grab N (s,a,r,s') tuples from experience
			minibatch = random.sample(experience, minibatch_size)

			# do a train_op with all the inputs required
			_ = sess.run(train_op,
				feed_dict = {
					state_ph: np.asarray([elem[0] for elem in minibatch]),
					action_ph: np.asarray([elem[1] for elem in minibatch]),
					reward_ph: np.asarray([elem[2] for elem in minibatch]),
					next_state_ph: np.asarray([elem[3] for elem in minibatch]),
					is_not_terminal_ph: np.asarray([elem[4] for elem in minibatch]),
					is_training_ph: True})

		observation = next_observation
		total_steps += 1
		steps_in_ep += 1

		# linearly decay epsilon from epsilon_start to epsilon_end over epsilon_decay_length steps
		if total_steps < epsilon_decay_length:
			epsilon -= epsilon_linear_step
		# then exponentially decay it every episode
		elif done:
			epsilon *= epsilon_decay_exp

		if done:
			# Increment episode counter
			_ = sess.run(episode_inc_op)
			break

	df.append(pd.DataFrame([ep, total_reward, steps_in_ep, epsilon, gamma, alpha]))
	# if(ep%100==0):
	print('Episode %2i, Reward: %7.3f, Steps: %i, Next eps: %7.3f'%(ep,total_reward,steps_in_ep, epsilon))

# Finalize and upload results

df.to_csv('TF_Test1.csv',index=False)
env.close()

