import torch
import torch.nn as nn

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
