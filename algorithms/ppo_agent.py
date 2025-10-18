# algorithms/ppo_agent.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dims, hidden=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        # MultiDiscrete: 1 head cho mỗi thành phần hành động
        self.actor_heads = nn.ModuleList([nn.Linear(hidden, n) for n in action_dims])
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        z = self.shared(x)
        logits = [head(z) for head in self.actor_heads]  # list([B, n_i], ...)
        value  = self.critic(z).squeeze(-1)              # [B]
        return logits, value


class PPOAgent:
    def __init__(self, state_dim, action_dims, cfg):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = ActorCritic(state_dim, action_dims, hidden=cfg.get("hidden", 256)).to(self.device)

        self.opt = optim.Adam(self.net.parameters(), lr=cfg["lr"])
        self.gamma = cfg.get("gamma", 0.99)
        self.lam = cfg.get("gae_lambda", 0.95)
        self.clip_eps = cfg.get("clip_eps", 0.2)
        self.epochs = cfg.get("epochs", 4)
        self.minibatch_size = cfg.get("minibatch_size", 64)
        self.ent_coef = cfg.get("ent_coef", 0.0)
        self.value_coef = cfg.get("value_coef", 0.5)
        self.max_grad_norm = cfg.get("max_grad_norm", 0.5)

    @torch.no_grad()
    def act(self, state):
        """
        Trả về:
          actions: list[int]  (1 phần tử/ head)
          logp_sum: float     (tổng log-prob qua các head)
          value: float        (V(s))
        """
        self.net.eval()
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, value = self.net(s)
        actions, logp_sum = [], 0.0
        for logit in logits:
            dist = Categorical(logits=logit)
            a = dist.sample()
            actions.append(int(a.item()))
            logp_sum += float(dist.log_prob(a).item())
        v = float(value.squeeze(0).item())
        self.net.train()
        return actions, logp_sum, v

    def compute_gae(self, rewards, values, dones):
        """
        Episode-based GAE:
          rewards: list[float], length T
          values : list[float], length T+1  (đÃ bao gồm V(s_T) cuối)
          dones  : list[bool],   length T
        Trả: (advs, returns) có length T
        """
        rewards = np.asarray(rewards, dtype=np.float32)
        values  = np.asarray(values,  dtype=np.float32)   # T+1
        dones   = np.asarray(dones,   dtype=np.float32)   # T
        T = len(rewards)

        advs = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma * values[t+1] * (1.0 - dones[t]) - values[t]
            gae   = delta + self.gamma * self.lam * (1.0 - dones[t]) * gae
            advs[t] = gae
        returns = advs + values[:-1]
        return advs, returns

    def update(self, rollout):
        """
        rollout (concat từ nhiều episode):
          states  : [M, state_dim] float32
          actions : [M, n_heads]   int64
          logps   : [M]            float32 (log-prob cũ)
          returns : [M]
          advs    : [M]
        """
        states  = torch.tensor(rollout["states"],  dtype=torch.float32, device=self.device)
        actions = torch.tensor(rollout["actions"], dtype=torch.long,    device=self.device)
        old_lp  = torch.tensor(rollout["logps"],   dtype=torch.float32, device=self.device)
        returns = torch.tensor(rollout["returns"], dtype=torch.float32, device=self.device)
        advs    = torch.tensor(rollout["advs"],    dtype=torch.float32, device=self.device)

        # Chuẩn hoá advantage
        advs = (advs - advs.mean()) / (advs.std(unbiased=False) + 1e-8)

        n = states.size(0)
        idxs = np.arange(n)
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        batches = 0

        for _ in range(self.epochs):
            np.random.shuffle(idxs)
            for start in range(0, n, self.minibatch_size):
                b = idxs[start:start+self.minibatch_size]
                sb, ab, rb, advb, oldlpb = states[b], actions[b], returns[b], advs[b], old_lp[b]

                logits, vals = self.net(sb)
                # log-prob mới + entropy (tổng theo heads)
                logps_parts = []
                ent_parts   = []
                for logit in logits:
                    dist = Categorical(logits=logit)
                    logps_parts.append(dist.log_prob(ab[:, len(ent_parts)]))
                    ent_parts.append(dist.entropy())  # [B]

                logps_new = torch.stack(logps_parts, dim=1).sum(1)      # [B]
                entropy   = torch.stack(ent_parts,   dim=1).sum(1).mean()# scalar

                ratio = (logps_new - oldlpb).exp()
                surr1 = ratio * advb
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advb
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss  = (rb - vals).pow(2).mean()
                loss = policy_loss + self.value_coef * value_loss - self.ent_coef * entropy

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.opt.step()

                stats["policy_loss"] += float(policy_loss.item())
                stats["value_loss"]  += float(value_loss.item())
                stats["entropy"]     += float(entropy.item())
                batches += 1

        if batches > 0:
            for k in stats:
                stats[k] /= batches
        return stats
