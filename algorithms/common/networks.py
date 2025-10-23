import torch
import torch.nn as nn
from .noisy_layers import NoisyLinear
class BDQNet(nn.Module):
    """
    BDQ-lite Network: trunk chung + dueling heads cho từng nhánh hành động.
    """
    def __init__(self, state_dim, action_dims):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU()
        )
        # Mỗi nhánh: 1 Value-head + 1 Advantage-head
        self.V_heads = nn.ModuleList([nn.Linear(256, 1) for _ in action_dims])
        self.A_heads = nn.ModuleList([nn.Linear(256, n) for n in action_dims])

    def forward(self, state):
        """
        state: FloatTensor [B, state_dim]
        return: list Q_heads, mỗi phần tử kích thước [B, n_i]
        """
        z = self.trunk(state)
        Q_heads = []
        for Vh, Ah in zip(self.V_heads, self.A_heads):
            V = Vh(z)                              # [B,1]
            A = Ah(z)                              # [B,n_i]
            A_norm = A - A.mean(dim=1, keepdim=True)
            Q_heads.append(V + A_norm)             # [B,n_i]
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