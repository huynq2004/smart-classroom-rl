import random
from collections import deque
import numpy as np

from collections import namedtuple

Transition = namedtuple("Transition", ("s", "a", "r", "s2", "done"))

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        """
        Args:
            capacity: số phần tử tối đa
            alpha   : mức độ ưu tiên (0 → uniform, 1 → theo p_i)
        """
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.pos = 0
        self.buffer = []
        self.priorities = np.zeros((self.capacity,), dtype=np.float32)

    def __len__(self):
        return len(self.buffer)

    def add(self, s, a, r, s2, done, priority=None):
        # Thêm hoặc ghi đè vòng tròn
        t = Transition(s, a, r, s2, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(t)
        else:
            self.buffer[self.pos] = t

        # p_i ban đầu = max priority hiện có (nếu trống thì 1.0)
        if priority is None:
            p = self.priorities.max() if self.pos > 0 else 1.0
        else:
            p = float(priority)
        self.priorities[self.pos] = p + 1e-6  # tránh 0
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        """
        Trả về:
            samples: list[Transition]
            indices: chỉ số trong buffer để cập nhật lại priority
            weights: IS weights (numpy float32) [B]
        """
        assert len(self.buffer) > 0, "Buffer rỗng."
        prios = self.priorities[:len(self.buffer)]
        probs = prios ** self.alpha
        probs /= probs.sum()

        idxs = np.random.choice(len(self.buffer), size=batch_size, p=probs)
        samples = [self.buffer[i] for i in idxs]

        total = len(self.buffer)
        weights = (total * probs[idxs]) ** (-beta)
        weights /= weights.max() + 1e-8

        return samples, idxs, weights.astype(np.float32)

    def update_priorities(self, indices, new_priorities):
        for i, p in zip(indices, new_priorities):
            self.priorities[i] = float(p) + 1e-6
            
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
