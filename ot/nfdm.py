"""Neural Flow Diffusion Models (NFDM) — Bartosh, Vetrov, Naesseth, NeurIPS 2024.

Reference: "Neural flow diffusion models: Learnable forward process for
improved diffusion modelling" (arXiv:2404.12940)

Key idea: instead of a FIXED forward process (like standard diffusion),
learn the forward process itself. The forward transform is:

    z_t = mu(x, t) + sigma(x, t) * eps

where mu and sigma are neural networks. The loss matches
the forward and reverse SDE drifts.

Faithfully ported from:
    https://github.com/GrigoryBartosh/neural_diffusion/blob/main/nfdm_from_scratch.ipynb

The CRITICAL implementation detail is using torch.autograd.functional.jvp()
to compute dz/dt (time derivatives of the affine transform), NOT
torch.autograd.grad(z.sum(), t) which incorrectly sums over the batch.
"""

import math
from typing import Callable, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


# ============================================================================
# Helper: JVP-based time derivative (THE KEY to making NFDM work)
# ============================================================================

def _jvp(f: Callable, x: Tensor, v: Tensor):
    """Wrapper around torch.autograd.functional.jvp with graph creation."""
    return torch.autograd.functional.jvp(
        f, x, v, create_graph=torch.is_grad_enabled()
    )


def _t_dir(f: Callable, t: Tensor):
    """Compute df/dt via JVP with direction v = ones_like(t).

    Returns ((outputs...), (tangents...)).
    """
    return _jvp(f, t, torch.ones_like(t))


def score_based_sde_drift(dz: Tensor, score: Tensor, g2: Tensor) -> Tensor:
    """SDE drift: f(z, t) = dz/dt - 0.5 * g^2 * score."""
    return dz - 0.5 * g2 * score


# ============================================================================
# Base MLP (matches original: 4 layers, 64 hidden, SELU)
# ============================================================================

class Net(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.SELU(),
            nn.Linear(64, 64), nn.SELU(),
            nn.Linear(64, 64), nn.SELU(),
            nn.Linear(64, 64), nn.SELU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# ============================================================================
# Affine flow: learnable forward process z = mu(x,t) + sigma(x,t) * eps
# ============================================================================

class AffineNeural(nn.Module):
    """Learnable affine params (mu, sigma) with boundary conditions.

    Boundary conditions (from the notebook):
        t=0: mu(x,0) = x,       sigma(x,0) = 0.01  (tight around data)
        t=1: mu(x,1) = 0,       sigma(x,1) = 1      (*)

    (*) Both mu and sigma have factor t*(1-t) that vanishes at boundaries.
    The log-sigma uses log(0.01) at t=0 as the base scale.
    """

    def __init__(self, d: int):
        super().__init__()
        self.net = Net(d + 1, 2 * d)

    def forward(self, x: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        x_t = torch.cat([x, t], dim=1)
        m_ls = self.net(x_t)
        m, ls = m_ls.chunk(2, dim=1)

        # Boundary-respecting parameterization
        m = (1 - t) * x + t * (1 - t) * m
        ls = (1 - t) * np.log(0.01) + t * (1 - t) * ls

        return m, torch.exp(ls)


class AffineTransform(nn.Module):
    """Forward transform z = mu(x,t) + sigma(x,t)*eps.

    Uses JVP to compute dz/dt and score correctly.
    """

    def __init__(self, flow: AffineNeural):
        super().__init__()
        self.flow = flow

    def get_t_dir(self, x: Tensor, t: Tensor):
        """Compute (mu, sigma) and their time derivatives (dmu/dt, dsigma/dt).

        Uses torch.autograd.functional.jvp to correctly compute per-sample
        time derivatives without summing over the batch.
        """
        def f(t_in):
            return self.flow(x, t_in)

        return _t_dir(f, t)

    def forward(self, eps: Tensor, t: Tensor, x: Tensor):
        """Forward: data x -> latent z.

        Returns:
            z: transformed state
            dz: dz/dt (time derivative of z)
            score: d log p(z|x) / dz = -eps / sigma
        """
        (m, s), (dm, ds) = self.get_t_dir(x, t)

        z = m + s * eps
        dz = dm + ds * eps
        score = -eps / s

        return z, dz, score

    def inverse(self, z: Tensor, t: Tensor, x: Tensor):
        """Inverse: given z and predicted x, compute eps and reverse drift.

        Returns:
            eps: reconstructed noise
            dz: dz/dt using predicted x
            score: score using predicted x
        """
        (m, s), (dm, ds) = self.get_t_dir(x, t)

        eps = (z - m) / s
        dz = dm + ds / s * (z - m)
        score = (m - z) / s ** 2

        return eps, dz, score


# ============================================================================
# Predictor: z -> x_hat
# ============================================================================

class Predictor(nn.Module):
    """Predicts data x from latent z at time t.

    Boundary conditions:
        t=0: x_hat = z  (at t=0, z ≈ x so identity is correct)
        t=1: x_hat = 1.01 * net(z, t)  (pure network prediction)
    """

    def __init__(self, d: int):
        super().__init__()
        self.net = Net(d + 1, d)

    def forward(self, z: Tensor, t: Tensor) -> Tensor:
        z_t = torch.cat([z, t], dim=1)
        x = self.net(z_t)
        # Boundary-respecting: at t=0 → z, at t=1 → net output
        x = (1 - t) * z + (t + 0.01) * x
        return x


# ============================================================================
# Learnable volatility g(t)
# ============================================================================

class VolatilityNeural(nn.Module):
    """Learnable volatility g(t) = softplus(net(t)).

    No explicit boundary conditions — softplus ensures positivity.
    """

    def __init__(self):
        super().__init__()
        self.net = Net(1, 1)
        self.sp = nn.Softplus()

    def forward(self, t: Tensor) -> Tensor:
        return self.sp(self.net(t))


# ============================================================================
# Full NFDM model
# ============================================================================

class NeuralDiffusion(nn.Module):
    """Neural Flow Diffusion Model.

    Loss = 0.5 * ||f_drift - r_drift||^2 / g^2, summed over dimensions.

    Components:
        transform: learnable forward process (AffineTransform)
        pred: data predictor from latent (Predictor)
        vol: learnable volatility (VolatilityNeural)
    """

    def __init__(self, d: int = 2):
        super().__init__()
        self.transform = AffineTransform(AffineNeural(d))
        self.pred = Predictor(d)
        self.vol = VolatilityNeural()

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """Compute NFDM drift-matching loss.

        Args:
            x: data [B, d].
            t: time [B, 1].

        Returns:
            Per-sample loss [B].
        """
        eps = torch.randn_like(x)

        # Forward: x -> z
        z, f_dz, f_score = self.transform(eps, t, x)

        # Reverse: z -> predicted x
        x_ = self.pred(z, t)
        _, r_dz, r_score = self.transform.inverse(z, t, x_)

        # Volatility
        g2 = self.vol(t) ** 2

        # SDE drifts
        f_drift = score_based_sde_drift(f_dz, f_score, g2)
        r_drift = score_based_sde_drift(r_dz, r_score, g2)

        # Drift matching loss
        loss = 0.5 * (f_drift - r_drift) ** 2 / g2
        return loss.sum(dim=1)


# ============================================================================
# Sampling (reverse-time SDE / ODE)
# ============================================================================

def _solve_sde(
    sde_fn: Callable,
    z: Tensor,
    ts: float,
    tf: float,
    n_steps: int,
) -> Tuple[Tensor, list]:
    """Euler-Maruyama SDE solver.

    Args:
        sde_fn: callable (z, t) -> (drift, diffusion_coeff)
        z: initial state [B, d].
        ts: start time.
        tf: final time.
        n_steps: number of steps.

    Returns:
        (z_final, trajectory_list)
    """
    t_steps = torch.linspace(ts, tf, n_steps + 1, device=z.device)
    dt = (tf - ts) / n_steps
    sqrt_abs_dt = abs(dt) ** 0.5

    traj = [z.detach().cpu()]

    for t_val in t_steps[:-1]:
        t = t_val.expand(z.shape[0], 1)

        f, g = sde_fn(z, t)
        w = torch.randn_like(z)
        z = z + f * dt + g * w * sqrt_abs_dt

        traj.append(z.detach().cpu())

    return z, traj


@torch.no_grad()
def nfdm_sample(
    model: NeuralDiffusion,
    n_samples: int,
    n_steps: int = 300,
    device: str = "cpu",
    return_trajectory: bool = False,
    seed: int = 42,
) -> Tensor:
    """Sample via reverse-time SDE: t=1 (noise) -> t=0 (data).

    The reverse SDE drift uses the predictor and inverse transform.
    """
    torch.manual_seed(seed)
    z = torch.randn(n_samples, 2, device=device)

    def sde_fn(z_in, t_in):
        x_ = model.pred(z_in, t_in)
        _, dz, score = model.transform.inverse(z_in, t_in, x_)
        g = model.vol(t_in)
        g2 = g ** 2
        drift = score_based_sde_drift(dz, score, g2)
        return drift, g

    z_final, traj = _solve_sde(sde_fn, z, ts=1.0, tf=0.0, n_steps=n_steps)

    if return_trajectory:
        return torch.stack(traj, dim=0)
    return z_final.cpu()
