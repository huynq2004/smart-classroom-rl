import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from algorithms.common.replay_buffer import ReplayBuffer
from algorithms.common.utils import soft_update, epsilon_decay, compute_q_sum

class MultiHeadNet(nn.Module):
    """
    Trunk chung + simple heads (không dueling).
    """
    def __init__(self, state_dim, action_dims):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.heads = nn.ModuleList([nn.Linear(256, n) for n in action_dims])

    def forward(self, state):
        """
        state: [B, state_dim]
        returns: list of tensors [B, n_i] for each head
        """
        z = self.trunk(state)
        return [h(z) for h in self.heads]


class MultiHeadDQNAgent:
    def __init__(self, state_dim, action_dims, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dims = list(action_dims)
        self.n_branches = len(self.action_dims)

        self.q_net = MultiHeadNet(state_dim, self.action_dims).to(self.device)
        self.target_net = MultiHeadNet(state_dim, self.action_dims).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=config["lr"])
        self.buffer = ReplayBuffer(config["buffer_size"])

        self.gamma = config["gamma"]
        self.batch_size = config["batch_size"]
        self.tau = config.get("tau", 0.01)

        self.eps = config["epsilon_start"]
        self.eps_end = config["epsilon_end"]
        self.eps_decay = config["epsilon_decay"]

    def select_action(self, state, greedy=False):
        """epsilon-greedy per-branch"""
        if (not greedy) and (np.random.rand() < self.eps):
            return [int(np.random.randint(0, n)) for n in self.action_dims]

        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            Q_heads = self.q_net(state_t)
        actions = [int(Q.argmax(dim=1).item()) for Q in Q_heads]
        return actions

    def update(self):
        if len(self.buffer) < self.batch_size:
            return {}

        s_batch, a_batch, r_batch, s_next_batch, done_batch = self.buffer.sample(self.batch_size)
        B = len(s_batch)

        s      = torch.as_tensor(s_batch, dtype=torch.float32, device=self.device)
        s_next = torch.as_tensor(s_next_batch, dtype=torch.float32, device=self.device)
        a      = torch.as_tensor(a_batch, dtype=torch.long, device=self.device)   # [B, n_branches]
        r      = torch.as_tensor(r_batch, dtype=torch.float32, device=self.device)
        done   = torch.as_tensor(done_batch, dtype=torch.float32, device=self.device)

        # Q_pred(s,a) via online net
        Q_heads = self.q_net(s)
        Q_pred = compute_q_sum(Q_heads, a)    # [B]

        # Double DQN target
        with torch.no_grad():
            Q_heads_next_online = self.q_net(s_next)
            a_star = torch.stack([Q.argmax(dim=1) for Q in Q_heads_next_online], dim=1)  # [B, n_branches]
            Q_heads_next_target = self.target_net(s_next)
            Q_target_sum = compute_q_sum(Q_heads_next_target, a_star)  # [B]
            y = r + self.gamma * (1.0 - done) * Q_target_sum

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
