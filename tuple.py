__author__ = 'hkc'


class Tuple(object):

	def __init__(self, current_state, action, reward, next_state, done):
		self.current_state = current_state
		self.action = action
		self.reward = reward
		self.next_state = next_state
		self.done = done

	@property
	def __repr__(self):
		return str(self.current_state) + str(self.action) + str(self.reward) + str(self.next_state) + str(self.done)
