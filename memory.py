import random

class Memory():
    def __init__(self, max_size):
        self.max_size = max_size
        self.tuples = []

    def add(self, tup):
        if len(self.tuples) >= self.max_size:
            self.tuples.pop()
        self.tuples.append(tup)

    def sample(self, batch_size):
        buffer_size = len(self.tuples)
        batch_size = min(batch_size, buffer_size)
        index = random.sample(range(buffer_size), batch_size)
        return [self.tuples[i] for i in index]

    def tuple_length(self):
        return len(self.tuples)
