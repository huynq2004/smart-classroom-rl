# algorithms/bdq_lite.py
# BDQ-Lite Agent -- đúng theo paper & mã giả:
# - Mạng BDQ dùng Eq.1 (Value chung + Advantage theo nhánh).
# - Chính sách ε-greedy (multi-discrete).
# - Warm-up W bước → update mỗi bước sau đó.
# - TD target theo Eq.6: y = r + γ * mean_d Q^-_d(s', argmax_a Q_d(s', a))
# - Loss theo Eq.7: MSE trung bình theo nhánh (có IS weights từ PER).
# - PER priority theo Eq.8: p_i = sum_d |TD_d|.
# - Hard target update mỗi 'target_update_steps'.
# - Rescale gradient 1/(N+1) cho trunk giúp ổn định khi nhiều nhánh.

import torch
import torch.nn as nn
import numpy as np

from algorithms.common.networks import BDQNet
from algorithms.common.replay_buffer import PrioritizedReplayBuffer
from algorithms.common.utils import (
    gather_per_branch, mean_over_branches,
    hard_update, epsilon_linear
)


class BDQLiteAgent:
    def __init__(self, state_dim, action_dims, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ======= Hyper-params
        self.gamma   = float(config.get("gamma", 0.99))
        self.lr      = float(config.get("lr", 1e-4))
        self.batch   = int(config.get("batch_size", 64))
        self.bufsize = int(config.get("buffer_size", 200000))

        # epsilon schedule
        self.eps_start = float(config.get("epsilon_start", 0.2))
        self.eps_end   = float(config.get("epsilon_end", 0.02))
        self.eps_decay = int(config.get("epsilon_decay", 100000))
        self.eps       = self.eps_start

        # PER
        self.per_alpha       = float(config.get("per_alpha", 0.6))
        self.per_beta_start  = float(config.get("per_beta_start", 0.4))
        self.per_beta_frames = int(config.get("per_beta_frames", 100000))

        # schedules
        self.warmup_steps        = int(config.get("warmup_steps", 1000))
        self.target_update_steps = int(config.get("target_update_steps", 1000))
        self.grad_clip_norm      = float(config.get("grad_clip_norm", 10.0))

        # ======= Networks
        self.q_net = BDQNet(state_dim, action_dims).to(self.device)
        self.t_net = BDQNet(state_dim, action_dims).to(self.device)
        hard_update(self.t_net, self.q_net)

        self.optim = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)

        # ======= Replay (PER)
        self.buffer = PrioritizedReplayBuffer(self.bufsize, alpha=self.per_alpha)

        # ======= Book-keeping
        self.n_branches = len(action_dims)
        self.total_steps = 0

    # ---------- Action selection ----------
    def select_action(self, s, greedy: bool = False):
        """
        Args:
            s: np.ndarray state
            greedy: True → dùng argmax (eval), False → ε-greedy (train)
        Returns:
            list[int] độ dài N (mỗi phần tử là chỉ số hành động của nhánh d)
        """
        if (not greedy) and (np.random.rand() < self.eps):
            return [np.random.randint(0, n) for n in self.q_net.action_dims]

        with torch.no_grad():
            s_t = torch.tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)
            Q_heads = self.q_net(s_t)                       # list of [1, n_a]
            acts = [int(Qd.argmax(dim=1).item()) for Qd in Q_heads]
        return acts

    # ---------- Train one episode ----------
    def train_episode(self, env):
        s, _ = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            # 1) chọn hành động và thực thi
            a = self.select_action(s, greedy=False)
            s2, r, term, trunc, _ = env.step(a)
            done = bool(term or trunc)
            ep_reward += float(r)

            # 2) lưu vào PER
            self.buffer.add(s, a, float(r), s2, done, priority=None)

            # 3) update (sau warm-up)
            self.update()

            # 4) bước & epsilon
            self.total_steps += 1
            self.eps = epsilon_linear(self.eps_start, self.eps_end, self.eps_decay, self.total_steps)

            # 5) chuyển trạng thái
            s = s2

        return ep_reward

    # ---------- Update parameters ----------
    def update(self):
        if self.total_steps < self.warmup_steps:
            return
        if len(self.buffer) < self.batch:
            return

        # beta schedule cho PER
        frac = min(1.0, self.total_steps / float(self.per_beta_frames))
        beta = self.per_beta_start + (1.0 - self.per_beta_start) * frac

        # 1) Lấy mẫu từ PER
        samples, indices, weights = self.buffer.sample(self.batch, beta=beta)

        s  = torch.tensor(np.stack([t.s for t in samples]), dtype=torch.float32, device=self.device)
        a  = torch.tensor(np.stack([t.a for t in samples]), dtype=torch.long,     device=self.device)   # [B, N]
        r  = torch.tensor([t.r for t in samples],           dtype=torch.float32, device=self.device)    # [B]
        s2 = torch.tensor(np.stack([t.s2 for t in samples]),dtype=torch.float32, device=self.device)
        done = torch.tensor([t.done for t in samples],      dtype=torch.float32, device=self.device)    # [B]
        w_is = torch.tensor(weights, dtype=torch.float32, device=self.device)    # [B]

        # 2) Q_d(s, a_d) hiện tại (online)  -- Eq.1 trong mạng
        Q_heads = self.q_net(s)                           # list of [B, n_a]
        Q_taken = gather_per_branch(Q_heads, a)           # list length N, mỗi phần tử [B]

        # 3) Double selection ở s' & đánh giá bằng target -- Eq.6 (mean theo nhánh)
        with torch.no_grad():
            Qn_online = self.q_net(s2)                    # online @ s'
            a_star = [Qd.argmax(dim=1) for Qd in Qn_online]       # list of [B]
            Qn_targ  = self.t_net(s2)                     # target @ s'
            Q_eval = [Qn_targ[d].gather(1, a_star[d].unsqueeze(1)).squeeze(1)
                      for d in range(self.n_branches)]    # list of [B]
            y = r + self.gamma * (1.0 - done) * mean_over_branches(Q_eval)  # [B]

        # 4) TD errors & Loss -- Eq.7 (MSE trung bình theo nhánh) + IS weights
        per_branch_losses = []
        per_branch_tds = []
        for d in range(self.n_branches):
            td_d = y - Q_taken[d]                         # [B]
            per_branch_tds.append(td_d.detach())
            per_branch_losses.append(td_d.pow(2))         # [B]

        loss = torch.stack(per_branch_losses, dim=1).mean(dim=1)   # mean over branches
        loss = (loss * w_is).mean()                                 # mean over batch

        # 5) backward + rescale 1/(N+1) + clip
        self.optim.zero_grad()
        loss.backward()

        scale = 1.0 / (self.n_branches + 1.0)
        for p in self.q_net.trunk.parameters():
            if p.grad is not None:
                p.grad.mul_(scale)

        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip_norm)
        self.optim.step()

        # 6) Cập nhật priority theo Eq.8: p_i = sum_d |TD_d|
        with torch.no_grad():
            abs_sum = torch.stack([td.abs() for td in per_branch_tds], dim=1).sum(dim=1)  # [B]
            new_p = (abs_sum + 1e-6).cpu().numpy()
            self.buffer.update_priorities(indices, new_p)

        # 7) Hard target update mỗi target_update_steps
        if (self.total_steps % self.target_update_steps) == 0:
            hard_update(self.t_net, self.q_net)
