"""Shared MLP backbone for all methods.

Same architecture as jump-models repo: 4 hidden layers of width 512 + Swish.
"""

import torch
import torch.nn as nn
from torch import Tensor


class Swish(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(x)


class VelocityMLP(nn.Module):
    """Predicts velocity u(x, t) -> R^2.

    Input: [x (2d), t (1d)] -> 3d
    Output: velocity (2d)
    """

    def __init__(self, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim), Swish(),
            nn.Linear(hidden_dim, hidden_dim), Swish(),
            nn.Linear(hidden_dim, hidden_dim), Swish(),
            nn.Linear(hidden_dim, hidden_dim), Swish(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        t_in = t.reshape(-1, 1).expand(x.shape[0], 1)
        return self.net(torch.cat([x, t_in], dim=1))


class ScoreMLP(nn.Module):
    """Predicts score s(x, t) -> R^2.

    Same architecture as VelocityMLP but separate weights.
    Used by Stochastic Interpolants (SDE mode) and DSBM.
    """

    def __init__(self, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim), Swish(),
            nn.Linear(hidden_dim, hidden_dim), Swish(),
            nn.Linear(hidden_dim, hidden_dim), Swish(),
            nn.Linear(hidden_dim, hidden_dim), Swish(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        t_in = t.reshape(-1, 1).expand(x.shape[0], 1)
        return self.net(torch.cat([x, t_in], dim=1))


class ForwardBackwardMLP(nn.Module):
    """Predicts both forward and backward drift for DSBM / Neural Flow Diffusion.

    Two heads sharing a backbone, or two separate networks.
    Here we use two separate networks for simplicity.
    """

    def __init__(self, hidden_dim: int = 512):
        super().__init__()
        self.forward_net = VelocityMLP(hidden_dim)
        self.backward_net = VelocityMLP(hidden_dim)

    def forward_drift(self, x: Tensor, t: Tensor) -> Tensor:
        return self.forward_net(x, t)

    def backward_drift(self, x: Tensor, t: Tensor) -> Tensor:
        return self.backward_net(x, t)
