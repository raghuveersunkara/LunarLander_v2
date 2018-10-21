import gym
from memory import Memory
from tuple import Tuple


class ReplayBuffer(object):
    def __init__(self, max_size, envName='LunarLander-v2', preTrainLen=100, episodeMaxRunSteps=100):
        self.memory = Memory(max_size)
        self.envName = envName
        self.preTrainLen = preTrainLen
        self.episodeMaxRunSteps = episodeMaxRunSteps

    def fillReplayBuffer(self):
        env = gym.make(self.envName)
        for episode in range(self.preTrainLen):
            observation = env.reset()
            for t in range(self.episodeMaxRunSteps):
                env.render()
                action = env.action_space.sample()
                observation, reward, done, info = env.step(action)
                tuple = Tuple(action, observation, reward, info, done)
                self.memory.add(tuple)
                if done:
                    break
        return self.memory


if __name__ == "__main__":
    replayBuffer = ReplayBuffer(100, preTrainLen=1)
    memo = replayBuffer.fillReplayBuffer()
    print(memo.sample(4))
