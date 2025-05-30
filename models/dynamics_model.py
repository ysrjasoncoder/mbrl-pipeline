import torch.nn as nn
import torch.nn.functional as F
import torch

class DynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head_ds = nn.Linear(hidden_dim, state_dim)
        self.head_r  = nn.Linear(hidden_dim, 1)

    def forward(self, s, a_onehot):
        x = torch.cat([s, a_onehot], dim=-1)
        h = self.net(x)
        ds = self.head_ds(h)
        r  = self.head_r(h).squeeze(-1)
        return ds, r
