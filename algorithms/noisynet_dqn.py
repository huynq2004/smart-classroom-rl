# algorithms/noisynet_dqn.py
import torch, numpy as np
import torch.nn as nn
import torch.optim as optim
from algorithms.common.networks import NoisyMultiHeadNet
from algorithms.common.replay_buffer import ReplayBuffer
from algorithms.common.utils import compute_q_sum, soft_update

class NoisyDQNAgent:
    """
    Noisy Multi-Head DQN agent:
    - Uses NoisyMultiHeadNet for exploration (no epsilon by default)
    - Double Q update: online selects a* (with noise resampled), target network evaluates
    """
    def __init__(self, state_dim, action_dims, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dims = action_dims

        self.q_net = NoisyMultiHeadNet(state_dim, action_dims, hidden=config.get("hidden",256)).to(self.device)
        self.target_net = NoisyMultiHeadNet(state_dim, action_dims, hidden=config.get("hidden",256)).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=config["lr"])
        self.buffer = ReplayBuffer(config["buffer_size"])

        self.gamma = config["gamma"]
        self.batch_size = config["batch_size"]
        self.update_every = config.get("update_every", 1)
        self.tau = config.get("tau", 0.01)
        self.step_cnt = 0

    def select_action(self, state, greedy=False):
        """
        Select action by passing through noisy network (it samples noise in forward).
        If greedy=True, we still use current noisy network but can set eval() to disable noise (optional).
        """
        s_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.q_net.train()  # ensure noise used
        Q_heads = self.q_net(s_t)
        actions = [int(Q.argmax(1).item()) for Q in Q_heads]
        return actions

    def push(self, s,a,r,s_next,done):
        self.buffer.push(s, a, r, s_next, done)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return {}

        s, a, r, s_next, done = self.buffer.sample(self.batch_size)
        s      = torch.tensor(s, dtype=torch.float32, device=self.device)
        s_next = torch.tensor(s_next, dtype=torch.float32, device=self.device)
        a      = torch.tensor(a, dtype=torch.long, device=self.device)
        r      = torch.tensor(r, dtype=torch.float32, device=self.device)
        done   = torch.tensor(done, dtype=torch.float32, device=self.device)

        # resample noise before computing online argmax (Fortunato)
        self.q_net.train()
        Q_heads_next_online = self.q_net(s_next)
        a_star = torch.stack([Q.argmax(1) for Q in Q_heads_next_online], dim=1)  # [B, n_heads]

        # resample noise for target network too (target has its own noise)
        self.target_net.train()
        Q_heads_next_target = self.target_net(s_next)
        Q_target_sum = compute_q_sum(Q_heads_next_target, a_star)  # [B]

        y = r + self.gamma * (1.0 - done) * Q_target_sum

        # compute Q_pred (use q_net with noise sampled)
        self.q_net.train()
        Q_heads = self.q_net(s)
        Q_pred = compute_q_sum(Q_heads, a)

        loss = nn.MSELoss()(Q_pred, y.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # soft target update
        soft_update(self.target_net, self.q_net, self.tau)
        return {"loss": float(loss.item())}

    def train_episode(self, env):
        s, _ = env.reset()
        done = False
        tot = 0.0
        while not done:
            # Noise is sampled in forward passes; we can call reset_noise if we specifically want to re-sample
            action = self.select_action(s)
            s2, r, term, trunc, info = env.step(action)
            done = bool(term or trunc)
            self.push(s, action, r, s2, done)
            stats = self.update()
            s = s2
            tot += float(r)
        return tot
