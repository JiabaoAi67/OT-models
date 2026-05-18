"""Flow Matching (CondOT) — baseline.

Reference: Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023.

The simplest OT-based generative model:
  - Forward path: x_t = (1-t) x_0 + t x_1
  - Velocity: u_t = x_1 - x_0 (constant per pair)
  - Loss: ||model(x_t, t) - u_t||^2
  - Sampling: ODE dx/dt = model(x, t) from t=0 to t=1
"""

import torch
from torch import Tensor


def flow_matching_loss(model, x_0: Tensor, x_1: Tensor) -> Tensor:
    """Compute CondOT flow matching loss.

    Args:
        model: VelocityMLP that predicts u(x_t, t).
        x_0: noise samples [B, 2].
        x_1: data samples [B, 2].

    Returns:
        Scalar loss.
    """
    t = torch.rand(x_0.shape[0], device=x_0.device)
    t_expand = t[:, None]  # [B, 1]

    # CondOT interpolation
    x_t = (1.0 - t_expand) * x_0 + t_expand * x_1

    # Target velocity (constant along the path)
    u_t = x_1 - x_0

    # Model prediction
    u_pred = model(x_t, t)

    return (u_pred - u_t).pow(2).mean()


@torch.no_grad()
def flow_matching_sample(
    model,
    n_samples: int,
    n_steps: int = 100,
    device: str = "cpu",
    return_trajectory: bool = False,
    seed: int = 42,
) -> Tensor:
    """Sample via Euler ODE integration.

    Args:
        model: trained VelocityMLP.
        n_samples: number of samples.
        n_steps: number of Euler steps.
        device: torch device.
        return_trajectory: if True, return [n_steps+1, B, 2].
        seed: random seed.

    Returns:
        Final samples [B, 2] or trajectory [n_steps+1, B, 2].
    """
    torch.manual_seed(seed)
    x_t = torch.randn(n_samples, 2, device=device)
    dt = 1.0 / n_steps

    traj = [x_t.cpu()] if return_trajectory else None

    for i in range(n_steps):
        t = torch.full((n_samples,), i * dt, device=device)
        u = model(x_t, t)
        x_t = x_t + dt * u

        if return_trajectory:
            traj.append(x_t.cpu())

    if return_trajectory:
        return torch.stack(traj, dim=0)
    return x_t
