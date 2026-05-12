"""
Diffusion-specific functions: noise schedule, forward sampling, reverse sampling.

These are shared between the training and sampling scripts.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class DiffusionSchedule:
    """Pre-computed quantities for a linear beta noise schedule."""

    T: int
    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_cumprod: torch.Tensor
    sqrt_alphas_cumprod: torch.Tensor
    sqrt_one_minus_alphas_cumprod: torch.Tensor

    def to(self, device):
        """Move all tensors to a device. Returns self for chaining."""
        for name in [
            "betas",
            "alphas",
            "alphas_cumprod",
            "sqrt_alphas_cumprod",
            "sqrt_one_minus_alphas_cumprod",
        ]:
            setattr(self, name, getattr(self, name).to(device))
        return self


def make_schedule(
    T: int = 200,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
) -> DiffusionSchedule:
    """Construct a linear beta schedule and the derived quantities used everywhere."""
    betas = torch.linspace(beta_start, beta_end, T)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return DiffusionSchedule(
        T=T,
        betas=betas,
        alphas=alphas,
        alphas_cumprod=alphas_cumprod,
        sqrt_alphas_cumprod=torch.sqrt(alphas_cumprod),
        sqrt_one_minus_alphas_cumprod=torch.sqrt(1.0 - alphas_cumprod),
    )


def q_sample(
    schedule: DiffusionSchedule,
    x0: torch.Tensor,
    t: torch.Tensor,
    noise: torch.Tensor = None,
) -> torch.Tensor:
    """
    Forward process: sample x_t given x_0 in closed form.

        x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise
    """
    if noise is None:
        noise = torch.randn_like(x0)
    sqrt_alpha = schedule.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha = schedule.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
    return sqrt_alpha * x0 + sqrt_one_minus_alpha * noise


@torch.no_grad()
def p_sample(
    model: nn.Module,
    schedule: DiffusionSchedule,
    x_t: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """One reverse-process step: x_t -> x_{t-1}."""
    eps_pred = model(x_t, t)

    beta_t = schedule.betas[t].view(-1, 1, 1, 1)
    alpha_t = schedule.alphas[t].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_bar_t = schedule.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

    mean = (1.0 / torch.sqrt(alpha_t)) * (
        x_t - (beta_t / sqrt_one_minus_alpha_bar_t) * eps_pred
    )

    # No noise on the final t=0 step
    if (t == 0).all():
        return mean

    sigma_t = torch.sqrt(beta_t)
    z = torch.randn_like(x_t)
    return mean + sigma_t * z


@torch.no_grad()
def sample(
    model: nn.Module,
    schedule: DiffusionSchedule,
    n_samples: int = 16,
    image_size: int = 32,
    channels: int = 1,
    device: str = "cuda",
) -> torch.Tensor:
    """Generate samples by reversing the diffusion process from pure Gaussian noise."""
    model.eval()
    x = torch.randn(n_samples, channels, image_size, image_size, device=device)
    for t_val in reversed(range(schedule.T)):
        t = torch.full((n_samples,), t_val, device=device, dtype=torch.long)
        x = p_sample(model, schedule, x, t)
    return x
