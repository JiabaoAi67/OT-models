"""Stochastic Interpolants — Albergo, Boffi, Vanden-Eijnden 2023.

References:
  - "Building Normalizing Flows with Stochastic Interpolants" (arXiv:2209.15571)
  - "Stochastic Interpolants: A Unifying Framework for Flows and Diffusions" (arXiv:2303.08797)

Key idea: define a path x_t = alpha(t) x_0 + beta(t) x_1 + gamma(t) z
where z ~ N(0, I). This induces a time-dependent density p_t(x).

The framework learns:
  - velocity b(x, t) for ODE sampling: dx/dt = b(x, t)
  - score s(x, t) for SDE sampling: dx = [b + eps*s] dt + sqrt(2*eps) dW

Training uses antithetic sampling for variance reduction.
"""

import math
import torch
from torch import Tensor


# ============================================================================
# Interpolant schedules
# ============================================================================

def alpha(t: Tensor) -> Tensor:
    """Coefficient for x_0 in the interpolant. alpha(0)=1, alpha(1)=0."""
    return 1.0 - t


def beta(t: Tensor) -> Tensor:
    """Coefficient for x_1 in the interpolant. beta(0)=0, beta(1)=1."""
    return t


def gamma(t: Tensor) -> Tensor:
    """Noise scale. gamma(0)=gamma(1)=0, positive in between.

    Brownian bridge schedule: gamma(t) = sqrt(t * (1 - t)).
    """
    return torch.sqrt(t * (1.0 - t) + 1e-8)


def dalpha_dt(t: Tensor) -> Tensor:
    """Time derivative of alpha."""
    return torch.full_like(t, -1.0)


def dbeta_dt(t: Tensor) -> Tensor:
    """Time derivative of beta."""
    return torch.full_like(t, 1.0)


def dgamma_dt(t: Tensor) -> Tensor:
    """Time derivative of gamma."""
    g = gamma(t)
    return (1.0 - 2.0 * t) / (2.0 * g + 1e-8)


# ============================================================================
# Interpolant computation
# ============================================================================

def compute_xt(
    t: Tensor, x_0: Tensor, x_1: Tensor, z: Tensor
) -> Tensor:
    """Compute the interpolant x_t = alpha(t)*x_0 + beta(t)*x_1 + gamma(t)*z.

    Args:
        t: time [B].
        x_0: noise samples [B, 2].
        x_1: data samples [B, 2].
        z: Gaussian noise [B, 2].

    Returns:
        x_t: [B, 2].
    """
    t_ = t[:, None]  # [B, 1]
    return alpha(t_) * x_0 + beta(t_) * x_1 + gamma(t_) * z


def compute_antithetic_xts(
    t: Tensor, x_0: Tensor, x_1: Tensor, z: Tensor
):
    """Compute antithetic pair (x_t+, x_t-) for variance reduction.

    x_t+ = alpha(t)*x_0 + beta(t)*x_1 + gamma(t)*z
    x_t- = alpha(t)*x_0 + beta(t)*x_1 - gamma(t)*z

    Returns:
        (x_t_plus, x_t_minus, z)
    """
    t_ = t[:, None]
    base = alpha(t_) * x_0 + beta(t_) * x_1
    noise = gamma(t_) * z
    return base + noise, base - noise, z


# ============================================================================
# Loss functions
# ============================================================================

def si_velocity_loss(model, x_0: Tensor, x_1: Tensor) -> Tensor:
    """Velocity (b) loss with antithetic sampling.

    The target velocity is:
        b_target = dalpha/dt * x_0 + dbeta/dt * x_1 + dgamma/dt * z

    We use antithetic pairs for variance reduction.
    """
    B = x_0.shape[0]
    t = torch.rand(B, device=x_0.device).clamp(1e-4, 1.0 - 1e-4)
    z = torch.randn_like(x_0)

    x_tp, x_tm, z = compute_antithetic_xts(t, x_0, x_1, z)

    t_ = t[:, None]
    # x_t+ uses +z, x_t- uses -z → targets must differ in the dgamma/dt*z term
    base_target = dalpha_dt(t_) * x_0 + dbeta_dt(t_) * x_1
    dg_z = dgamma_dt(t_) * z
    b_target_p = base_target + dg_z   # target for x_t+ (uses +z)
    b_target_m = base_target - dg_z   # target for x_t- (uses -z)

    # Evaluate model on both antithetic samples
    b_pred_p = model(x_tp, t)
    b_pred_m = model(x_tm, t)

    loss_p = (b_pred_p - b_target_p).pow(2).mean()
    loss_m = (b_pred_m - b_target_m).pow(2).mean()

    return 0.5 * (loss_p + loss_m)


def si_score_loss(score_model, x_0: Tensor, x_1: Tensor) -> Tensor:
    """Score (s) loss with antithetic sampling.

    The target for the score involves z/gamma(t):
        s_target = -z / gamma(t)

    We use antithetic pairs: for x_t+, target = -z/gamma;
    for x_t-, target = +z/gamma.
    """
    B = x_0.shape[0]
    t = torch.rand(B, device=x_0.device).clamp(1e-4, 1.0 - 1e-4)
    z = torch.randn_like(x_0)

    x_tp, x_tm, z = compute_antithetic_xts(t, x_0, x_1, z)

    t_ = t[:, None]
    g = gamma(t_)
    s_target_p = -z / (g + 1e-8)
    s_target_m = z / (g + 1e-8)

    s_pred_p = score_model(x_tp, t)
    s_pred_m = score_model(x_tm, t)

    loss_p = (s_pred_p - s_target_p).pow(2).mean()
    loss_m = (s_pred_m - s_target_m).pow(2).mean()

    return 0.5 * (loss_p + loss_m)


# ============================================================================
# Sampling
# ============================================================================

@torch.no_grad()
def si_ode_sample(
    velocity_model,
    n_samples: int,
    n_steps: int = 100,
    device: str = "cpu",
    return_trajectory: bool = False,
    seed: int = 42,
) -> Tensor:
    """Sample via ODE (deterministic): dx/dt = b(x, t).

    Same as flow matching Euler, but the learned velocity accounts for
    the stochastic interpolant structure (including the gamma noise term).
    """
    torch.manual_seed(seed)
    x_t = torch.randn(n_samples, 2, device=device)
    dt = 1.0 / n_steps

    traj = [x_t.cpu()] if return_trajectory else None

    for i in range(n_steps):
        t = torch.full((n_samples,), i * dt, device=device)
        b = velocity_model(x_t, t)
        x_t = x_t + dt * b

        if return_trajectory:
            traj.append(x_t.cpu())

    if return_trajectory:
        return torch.stack(traj, dim=0)
    return x_t


@torch.no_grad()
def si_sde_sample(
    velocity_model,
    score_model,
    n_samples: int,
    n_steps: int = 100,
    eps: float = 0.5,
    device: str = "cpu",
    return_trajectory: bool = False,
    seed: int = 42,
) -> Tensor:
    """Sample via SDE (stochastic): dx = [b + eps*s] dt + sqrt(2*eps) dW.

    Args:
        velocity_model: trained velocity network b(x, t).
        score_model: trained score network s(x, t).
        eps: diffusion coefficient (0 = pure ODE, larger = more stochastic).
        n_steps: number of Euler-Maruyama steps.
    """
    torch.manual_seed(seed)
    x_t = torch.randn(n_samples, 2, device=device)
    dt = 1.0 / n_steps
    sqrt_2eps_dt = math.sqrt(2.0 * eps * dt)

    traj = [x_t.cpu()] if return_trajectory else None

    for i in range(n_steps):
        t_val = i * dt
        t = torch.full((n_samples,), t_val, device=device)

        b = velocity_model(x_t, t)
        s = score_model(x_t, t)

        drift = b + eps * s
        noise = sqrt_2eps_dt * torch.randn_like(x_t)

        x_t = x_t + dt * drift + noise

        if return_trajectory:
            traj.append(x_t.cpu())

    if return_trajectory:
        return torch.stack(traj, dim=0)
    return x_t
