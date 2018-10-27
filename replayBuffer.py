import random
from memory import Memory
from tuple import Tuple
import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer(object):
	def __init__(self, action_size, buffer_size, batch_size, seed):
		self.action_size = action_size
		self.replay_memory = Memory(max_size=buffer_size)
		self.batch_size = batch_size
		self.seed = random.seed(seed)
		self.batch_size = batch_size

	def fill_replay_buffer(self, current_state, action, reward, next_state, done):
		tuple = Tuple(current_state, action, reward, next_state, done)
		self.replay_memory.add(tuple)

	#
	def get_sample_replay_buffer(self):
		"""Get a random sample batch of experiences from stored memory"""
		samples = self.replay_memory.sample(self.batch_size)

		states = torch.from_numpy(np.vstack([s.current_state for s in samples if s is not None])).float().to(device)
		actions = torch.from_numpy(np.vstack([s.action for s in samples if s is not None])).long().to(device)
		rewards = torch.from_numpy(np.vstack([s.reward for s in samples if s is not None])).float().to(device)
		next_states = torch.from_numpy(np.vstack([s.next_state for s in samples if s is not None])).float().to(device)
		dones = torch.from_numpy(np.vstack([s.done for s in samples if s is not None]).astype(np.uint8)).float().to(
			device)

		return states, actions, rewards, next_states, dones

	def __len__(self):
		"""Return the current size of internal memory."""
		return self.replay_memory.tuple_length()
