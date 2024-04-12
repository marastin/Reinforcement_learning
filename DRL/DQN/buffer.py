import random
from collections import namedtuple

# Named tuple for storing transitions
Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'terminated', 'truncated'])

class ReplayBuffer:
    """Replay buffer to store transitions"""

    def __init__(self, capacity):
        """Initialize replay buffer"""
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Add a transition to the replay buffer"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Sample a batch of transitions from the replay buffer"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """Return the current size of the replay buffer"""
        return len(self.memory)