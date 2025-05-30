import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size, sequential=False):
        n = len(self.buffer)
        batch_size = min(batch_size, n)
        if sequential and n >= batch_size:
            idx = random.randint(0, n - batch_size)
            batch = [self.buffer[i] for i in range(idx, idx + batch_size)]
        else:
            batch = random.sample(self.buffer, batch_size)
        return zip(*batch)

    def __len__(self):
        return len(self.buffer)
    
    def sample_state_action(self, batch_size):
        batch_size = min(batch_size, len(self.buffer))
        batch = random.sample(self.buffer, batch_size)
        states, actions, *_ = zip(*batch)
        return states, actions

