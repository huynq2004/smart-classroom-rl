import random
from collections import deque
import numpy as np

class ReplayBuffer:
    """
    Bộ nhớ lưu kinh nghiệm cho DQN/BDQ.
    Lưu tuple: (state, action_tuple, reward, next_state, done)
    """
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s_next, done):
        self.buffer.append((np.array(s, dtype=np.float32),
                            np.array(a, dtype=np.int64),
                            float(r),
                            np.array(s_next, dtype=np.float32),
                            float(done)))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_next, done = map(np.array, zip(*batch))
        return s, a, r, s_next, done

    def __len__(self):
        return len(self.buffer)
