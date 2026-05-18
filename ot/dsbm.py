"""Diffusion Schrödinger Bridge Matching (DSBM) — De Bortoli et al.

Reference: "Diffusion Schrödinger Bridge Matching" (arXiv:2303.16852)

Key idea: Solve the Schrödinger bridge problem
    P* = argmin { KL(P|Q) : P_0 = pi_0, P_T = pi_1 }
via iterative alternating Markovian projections.

The bridge matching loss uses Brownian bridge interpolation:
    z_t = t*z_1 + (1-t)*z_0 + sigma*sqrt(t*(1-t))*noise

IMPORTANT: The network takes (x_t, t) as input — NOT (x_t, t, z_0, z_1).
The endpoints z_0/z_1 are used to compute the LOSS TARGET (conditional
drift), but the network learns the MARGINAL drift (averaged over couplings).
This is the same principle as flow matching: per-pair targets, but the
network learns the average.

Adapted from: https://github.com/yuyang-shi/dsbm-pytorch (DSBM-Gaussian.py)

This toy implementation keeps the Brownian-bridge matching loss from the
official code and follows the same alternating outer-loop semantics:
each newly trained model regenerates the endpoint coupling used by the
next direction.
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


# ============================================================================
# Score network — takes (x_t, t) only, learns MARGINAL drift
# ============================================================================

class DSBMScoreNet(nn.Module):
    """MLP for marginal drift prediction: (x, t) -> drift.

    Does NOT receive endpoint conditioning — it learns the marginal
    drift E[u_t(x|z_0, z_1) | x_t = x] by regression.
    """

    def __init__(self, d: int = 2, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, d),
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        if t.dim() == 0 or t.dim() == 1:
            t = t.reshape(-1, 1).expand(x.shape[0], 1)
        return self.net(torch.cat([x, t], dim=-1))


# ============================================================================
# Bridge matching loss
# ============================================================================

def bridge_matching_loss(
    net: DSBMScoreNet,
    z0: Tensor,
    z1: Tensor,
    sigma: float = 1.0,
    direction: str = "f",
    eps: float = 1e-3,
) -> Tensor:
    """Bridge matching loss for one direction.

    Endpoints z0/z1 are used to compute the TARGET drift (conditional),
    but the NETWORK only sees (z_t, t) and learns the marginal drift.

    Interpolation: z_t = t*z1 + (1-t)*z0 + sigma*sqrt(t*(1-t))*noise
    Forward target: (z1 - z_t) / (1 - t)
    Backward target: (z0 - z_t) / t
    """
    batch_size = z0.shape[0]
    device = z0.device

    t = torch.rand(batch_size, 1, device=device) * (1.0 - 2 * eps) + eps

    # Brownian bridge interpolation
    z_t = t * z1 + (1.0 - t) * z0
    noise = torch.randn_like(z_t)
    z_t = z_t + sigma * torch.sqrt(t * (1.0 - t)) * noise

    # Target drift (uses endpoints — conditional drift)
    if direction == "f":
        target = (z1 - z_t) / (1.0 - t + 1e-8)
    elif direction == "b":
        target = (z0 - z_t) / (t + 1e-8)
    else:
        raise ValueError(f"Unknown direction: {direction}")

    # Predict (network only sees z_t and t — learns marginal drift)
    pred = net(z_t, t)

    return (pred - target).pow(2).mean()


# ============================================================================
# Sampling (Euler-Maruyama)
# ============================================================================

@torch.no_grad()
def dsbm_sample_sde(
    net: DSBMScoreNet,
    x_start: Tensor,
    n_steps: int = 100,
    sigma: float = 1.0,
    direction: str = "f",
    return_trajectory: bool = False,
) -> Tensor:
    """Sample from the learned bridge via Euler-Maruyama."""
    dt = 1.0 / n_steps
    x = x_start.clone()
    traj = [x.cpu()] if return_trajectory else None

    for i in range(n_steps):
        if direction == "f":
            t_val = i * dt
        elif direction == "b":
            t_val = 1.0 - i * dt
        else:
            raise ValueError(f"Unknown direction: {direction}")

        t = torch.full((x.shape[0], 1), t_val, device=x.device)
        drift = net(x, t)

        noise = torch.randn_like(x) if i < n_steps - 1 else torch.zeros_like(x)
        x = x + drift * dt + sigma * math.sqrt(dt) * noise

        if return_trajectory:
            traj.append(x.cpu())

    if return_trajectory:
        return torch.stack(traj, dim=0)
    return x


# ============================================================================
# Alternating DSBM training loop
# ============================================================================

def _train_direction(
    net: DSBMScoreNet,
    optimizer,
    z0: Tensor,
    z1: Tensor,
    sigma: float,
    direction: str,
    inner_iters: int,
    batch_size: int,
    device: str,
    verbose: bool,
    log_prefix: str,
) -> None:
    """Train one direction on a fixed endpoint coupling."""
    n = z0.shape[0]
    log_every = max(1, inner_iters // 2)
    net.train()

    for i in range(inner_iters):
        idx = torch.randint(0, n, (batch_size,))
        batch_z0 = z0[idx].to(device)
        batch_z1 = z1[idx].to(device)

        loss = bridge_matching_loss(
            net, batch_z0, batch_z1, sigma=sigma, direction=direction
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose and (i + 1) % log_every == 0:
            print(f"    [{log_prefix}] iter {i+1}/{inner_iters} loss={loss.item():.4f}")


@torch.no_grad()
def _sample_in_batches(
    net: DSBMScoreNet,
    x_start: Tensor,
    sigma: float,
    direction: str,
    batch_size: int,
    device: str,
    n_steps: int = 100,
) -> Tensor:
    """Sample an entire dataset without materializing it on device at once."""
    outputs = []
    net.eval()

    for start in range(0, len(x_start), batch_size):
        end = min(start + batch_size, len(x_start))
        batch = x_start[start:end].to(device)
        outputs.append(
            dsbm_sample_sde(
                net,
                batch,
                n_steps=n_steps,
                sigma=sigma,
                direction=direction,
            ).cpu()
        )

    return torch.cat(outputs, dim=0)


def _initial_independent_coupling(data_x0: Tensor, data_x1: Tensor) -> Tuple[Tensor, Tensor]:
    """Create the IMF-style initial independent coupling."""
    perm = torch.randperm(data_x1.shape[0])
    return data_x0.clone(), data_x1[perm].clone()


def train_dsbm(
    data_x0: Tensor,
    data_x1: Tensor,
    sigma: float = 1.0,
    n_ipf: int = 5,
    inner_iters: int = 5000,
    batch_size: int = 2048,
    lr: float = 1e-3,
    d: int = 2,
    hidden: int = 256,
    device: str = "cpu",
    verbose: bool = True,
) -> Tuple[DSBMScoreNet, DSBMScoreNet]:
    """Train the toy DSBM model with alternating forward/backward updates.

    The first backward step uses an independent endpoint coupling, matching the
    IMF-style initialization in the official implementation. After that, each
    newly trained model regenerates the coupling for the opposite direction:

      backward on (z0, z1_init)
        -> use backward model to map original x1 to a synthetic z0
      forward on (synthetic z0, original x1)
        -> use forward model to map original x0 to a synthetic z1
      backward on (original x0, synthetic z1)
        -> ...

    This keeps the outer loop faithful to DSBM's alternating projection
    semantics instead of repeatedly updating the coupling with the same
    direction.
    """
    if n_ipf < 1:
        raise ValueError("n_ipf must be >= 1")
    if data_x0.shape[0] != data_x1.shape[0]:
        raise ValueError("data_x0 and data_x1 must have the same number of samples")

    data_x0 = data_x0.detach().cpu()
    data_x1 = data_x1.detach().cpu()

    net_fwd = DSBMScoreNet(d, hidden).to(device)
    net_bwd = DSBMScoreNet(d, hidden).to(device)
    opt_fwd = torch.optim.Adam(net_fwd.parameters(), lr=lr)
    opt_bwd = torch.optim.Adam(net_bwd.parameters(), lr=lr)

    # Initial DSBM-IMF coupling: source samples paired independently with target.
    pair_z0, pair_z1 = _initial_independent_coupling(data_x0, data_x1)

    for ipf_iter in range(n_ipf):
        if verbose:
            print(f"\n  IPF iteration {ipf_iter + 1}/{n_ipf}")

        # --- Train backward on the current coupling ---
        _train_direction(
            net_bwd,
            opt_bwd,
            pair_z0,
            pair_z1,
            sigma=sigma,
            direction="b",
            inner_iters=inner_iters,
            batch_size=batch_size,
            device=device,
            verbose=verbose,
            log_prefix="bwd",
        )

        # Latest backward model generates the forward-training coupling.
        pair_z0 = _sample_in_batches(
            net_bwd,
            data_x1,
            sigma=sigma,
            direction="b",
            batch_size=batch_size,
            device=device,
        )
        pair_z1 = data_x1.clone()

        # --- Train forward on the refreshed coupling ---
        _train_direction(
            net_fwd,
            opt_fwd,
            pair_z0,
            pair_z1,
            sigma=sigma,
            direction="f",
            inner_iters=inner_iters,
            batch_size=batch_size,
            device=device,
            verbose=verbose,
            log_prefix="fwd",
        )

        # Latest forward model generates the next backward-training coupling.
        pair_z0 = data_x0.clone()
        pair_z1 = _sample_in_batches(
            net_fwd,
            data_x0,
            sigma=sigma,
            direction="f",
            batch_size=batch_size,
            device=device,
        )

    return net_fwd, net_bwd
