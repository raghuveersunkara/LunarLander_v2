__author__ = 'hkc'

class Tuple(object):
    def __init__(self, action, obs, reward, info, done):
        self.action = action
        self.obs = obs
        self.reward = reward
        self.info = info
        self.done = done

    def __repr__(self):
        return (str(self.action) + str(self.obs) + str(self.reward) + str(self.info) + str(self.done))