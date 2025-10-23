import torch
import torch.nn as nn
from .noisy_layers import NoisyLinear

import torch
import torch.nn as nn


class BDQNet(nn.Module):
    def __init__(self, state_dim, action_dims, hidden_trunk=(256, 256), hidden_head=128):
        """
        Args:
            state_dim   : int, số chiều trạng thái
            action_dims : list[int], số lựa chọn cho từng nhánh hành động (N nhánh)
            hidden_trunk: tuple layer cho trunk
            hidden_head : độ rộng head (Value/Advantage)
        """
        super().__init__()
        self.action_dims = list(action_dims)
        self.n_branches = len(self.action_dims)

        # Trunk chung
        layers, in_f = [], state_dim
        for h in hidden_trunk:
            layers += [nn.Linear(in_f, h), nn.ReLU()]
            in_f = h
        self.trunk = nn.Sequential(*layers)

        # 1 Value head chung
        self.value_head = nn.Sequential(
            nn.Linear(in_f, hidden_head), nn.ReLU(),
            nn.Linear(hidden_head, 1)
        )

        # N Advantage-heads (mỗi nhánh 1 head)
        self.adv_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_f, hidden_head), nn.ReLU(),
                nn.Linear(hidden_head, n_a)     # trả về A_d(s, ·)
            )
            for n_a in self.action_dims
        ])

    def forward(self, s):
        """
            s: Tensor [B, state_dim]
        Returns:
            list Q_heads: chiều dài = N; mỗi phần tử Tensor [B, n_a]
        """
        f = self.trunk(s)                   # [B, F]
        V = self.value_head(f)              # [B, 1]

        Q_heads = []
        for head in self.adv_heads:
            A = head(f)                     # [B, n_a]
            A = A - A.mean(dim=1, keepdim=True)   # Eq.1 (trừ trung bình advantage theo nhánh)
            Q_heads.append(V + A)           # [B, n_a]
        return Q_heads
    
class NoisyMultiHeadNet(nn.Module):
    """
    Trunk chung (Noisy) + per-branch output heads (Noisy).
    Produces Q-values per branch (for multi-discrete action).
    """
    def __init__(self, state_dim, action_dims, hidden=256):
        super().__init__()
        # trunk with standard linear + Noisy final
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        # final noisy layer for shared features (optional)
        self.noisy_out = NoisyLinear(hidden, hidden)

        self.heads = nn.ModuleList([NoisyLinear(hidden, n) for n in action_dims])

        self.relu = nn.ReLU()

    def forward(self, x):
        z = self.relu(self.fc1(x))
        z = self.relu(self.fc2(z))
        z = self.relu(self.noisy_out(z))
        Qs = [head(z) for head in self.heads]  # each [B, n_i]
        return Qs

    def reset_noise(self):
        # reset noise for all Noisy layers
        self.noisy_out.reset_noise()
        for h in self.heads:
            h.reset_noise()