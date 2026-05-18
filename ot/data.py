"""Shared 2D toy data generators.

Same checkerboard as the jump-models repo for direct comparison.
"""

import torch
from torch import Tensor


def checkerboard(batch_size: int, device: str = "cpu") -> Tensor:
    """2D checkerboard in [-4.5, 4.5]^2 (same as jump-models repo)."""
    x1 = torch.rand(batch_size, device=device) * 4 - 2
    x2_ = (
        torch.rand(batch_size, device=device)
        - torch.randint(high=2, size=(batch_size,), device=device) * 2
    )
    x2 = x2_ + (torch.floor(x1) % 2)
    return torch.cat([x1[:, None], x2[:, None]], dim=1) / 0.45


def gaussian_8modes(batch_size: int, device: str = "cpu") -> Tensor:
    """8 Gaussians arranged in a circle (classic OT benchmark)."""
    n_per_mode = batch_size // 8
    centers = []
    for i in range(8):
        angle = 2 * 3.14159 * i / 8
        centers.append(torch.tensor([4.0 * angle.__cos__(), 4.0 * angle.__sin__()]))
    centers = torch.stack(centers, dim=0).to(device)  # [8, 2]

    # Sample from each mode
    mode_idx = torch.randint(0, 8, (batch_size,), device=device)
    samples = centers[mode_idx] + 0.3 * torch.randn(batch_size, 2, device=device)
    return samples


def moons(batch_size: int, device: str = "cpu") -> Tensor:
    """Two interleaving half-circles."""
    n = batch_size
    t = torch.rand(n, device=device) * 3.14159
    x = torch.cat(
        [
            torch.cat([torch.cos(t[:n//2]), torch.cos(t[n//2:]) + 0.5], dim=0)[:, None],
            torch.cat([torch.sin(t[:n//2]), -torch.sin(t[n//2:]) + 0.5], dim=0)[:, None],
        ],
        dim=1,
    )
    x = x * 3.0 + 0.2 * torch.randn_like(x)
    return x
