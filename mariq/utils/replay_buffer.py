from collections import deque
import numpy as np
import random

class ReplayBuffer(object):
    def __init__(self, maxlen):
        self.buffer = deque(maxlen=maxlen)

    def __len__(self):
        return len(self.buffer)

    def append(self, state, action, reward, next_state, done):
        _state = np.expand_dims(state, 0)
        _next_state = np.expand_dims(next_state, 0)

        self.buffer.append(
            (_state, action, reward, _next_state, done)
        )

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(
            *random.sample(self.buffer, batch_size)
        )
        
        _state = np.concatenate(state)
        _next_state = np.concatenate(next_state)

        return _state, action, reward, _next_state, done
    