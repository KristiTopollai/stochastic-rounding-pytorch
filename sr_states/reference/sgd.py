"""Functional SGDM reference updates."""

from __future__ import annotations

import torch

from .rounding import stochastic_round_bf16


def sgdm_step(
    parameter: torch.Tensor,
    gradient: torch.Tensor,
    momentum: torch.Tensor,
    *,
    lr: float,
    momentum_factor: float,
    state_dtype: torch.dtype = torch.bfloat16,
    rounding: str = "stochastic",
    seed: int = 0,
    offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one SGDM update, computing all arithmetic in FP32."""

    if rounding not in {"stochastic", "nearest"}:
        raise ValueError("rounding must be 'stochastic' or 'nearest'")
    if state_dtype not in {torch.bfloat16, torch.float32}:
        raise ValueError("state_dtype must be torch.bfloat16 or torch.float32")

    p32 = parameter.float()
    u32 = momentum.float().mul(momentum_factor).add(gradient.float())
    updated_parameter = p32.add(u32, alpha=-lr).to(parameter.dtype)
    if state_dtype == torch.float32:
        stored_momentum = u32
    elif rounding == "stochastic":
        stored_momentum = stochastic_round_bf16(u32, seed=seed, offset=offset)
    else:
        stored_momentum = u32.to(torch.bfloat16)
    return updated_parameter, stored_momentum


@torch.no_grad()
def sgdm_step_(parameter, gradient, momentum, **kwargs) -> None:
    """Apply the reference update in place, including state write-back."""
    updated_p, updated_m = sgdm_step(parameter, gradient, momentum, **kwargs)
    parameter.copy_(updated_p)
    momentum.copy_(updated_m)
