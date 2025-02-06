import numpy as np
from collections import deque
import random

class ReplayBufferLoader:
    def __init__(self, buffer_capacity=10000):
        self.buffer = deque(maxlen=buffer_capacity)
    
    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.append( (state, action, reward, next_state, done) )
    
    def sample_batch(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        return zip(*transitions)