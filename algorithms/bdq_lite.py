import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from algorithms.common.networks import BDQNet
from algorithms.common.replay_buffer import ReplayBuffer
from algorithms.common.utils import soft_update, epsilon_decay, compute_q_sum

class BDQLiteAgent:
    def __init__(self, state_dim, action_dims, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = BDQNet(state_dim, action_dims).to(self.device)
        self.target_net = BDQNet(state_dim, action_dims).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=config["lr"])
        self.buffer = ReplayBuffer(config["buffer_size"])

        self.gamma = config["gamma"]
        self.batch_size = config["batch_size"]
        self.tau = config.get("tau", 0.01)

        self.eps = config["epsilon_start"]
        self.eps_end = config["epsilon_end"]
        self.eps_decay = config["epsilon_decay"]

        self.action_dims = action_dims

    def select_action(self, state, greedy: bool=False):
        """
        Epsilon-greedy theo từng nhánh. Khi greedy=True, luôn argmax.
        """
        if (not greedy) and (np.random.rand() < self.eps):
            return [np.random.randint(0, n) for n in self.action_dims]
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            Q_heads = self.q_net(state_t)
        actions = [Q.argmax(1).item() for Q in Q_heads]
        return actions

    def update(self):
        if len(self.buffer) < self.batch_size:
            return {}

        s, a, r, s_next, done = self.buffer.sample(self.batch_size)
        s      = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        s_next = torch.as_tensor(s_next, dtype=torch.float32, device=self.device)
        a      = torch.as_tensor(a, dtype=torch.long, device=self.device)
        r      = torch.as_tensor(r, dtype=torch.float32, device=self.device)
        done   = torch.as_tensor(done, dtype=torch.float32, device=self.device)

        # Q(s,a)
        Q_heads = self.q_net(s)
        Q_pred = compute_q_sum(Q_heads, a)  # [B]

        # Double: a* từ online, giá trị từ target
        with torch.no_grad():
            Q_heads_next_online = self.q_net(s_next)
            a_star = torch.stack([Q.argmax(1) for Q in Q_heads_next_online], dim=1)  # [B, n_heads]
            Q_heads_next_target = self.target_net(s_next)
            Q_tgt_sum = compute_q_sum(Q_heads_next_target, a_star)
            y = r + self.gamma * (1.0 - done) * Q_tgt_sum

        loss = nn.MSELoss()(Q_pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        soft_update(self.target_net, self.q_net, self.tau)
        self.eps = epsilon_decay(self.eps, self.eps_end, self.eps_decay)

        return {"loss": float(loss.item()), "eps": float(self.eps)}

    def train_episode(self, env):
        s, _ = env.reset()
        done = False
        total_r = 0.0
        while not done:
            a = self.select_action(s)
            s_next, r, terminated, truncated, info = env.step(a)
            done = bool(terminated or truncated)
            self.buffer.push(s, a, r, s_next, done)
            stats = self.update()
            s = s_next
            total_r += float(r)
        return total_r
