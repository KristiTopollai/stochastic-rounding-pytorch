"""Functional AdamW reference updates."""

from __future__ import annotations

import torch

from .rounding import stochastic_round_bf16


def adamw_step(
    parameter: torch.Tensor,
    gradient: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    *,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    step: int,
    state_dtype: torch.dtype = torch.bfloat16,
    rounding: str = "stochastic",
    seed: int = 0,
    offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return one AdamW update, computing all optimizer arithmetic in FP32."""

    if step < 1:
        raise ValueError("step must be at least 1")
    if rounding not in {"stochastic", "nearest"}:
        raise ValueError("rounding must be 'stochastic' or 'nearest'")
    if state_dtype not in {torch.bfloat16, torch.float32}:
        raise ValueError("state_dtype must be torch.bfloat16 or torch.float32")

    p32 = parameter.float()
    g32 = gradient.float()
    m32 = exp_avg.float().mul(beta1).add(g32, alpha=1.0 - beta1)
    v32 = exp_avg_sq.float().mul(beta2).addcmul(g32, g32, value=1.0 - beta2)

    bias_correction1 = 1.0 - beta1**step
    bias_correction2 = 1.0 - beta2**step
    denominator = v32.sqrt().div_(bias_correction2**0.5).add_(eps)
    updated_parameter = p32.mul(1.0 - lr * weight_decay)
    updated_parameter.addcdiv_(m32, denominator, value=-lr / bias_correction1)
    updated_parameter = updated_parameter.to(parameter.dtype)

    if state_dtype == torch.float32:
        stored_m, stored_v = m32, v32
    elif rounding == "stochastic":
        stored_m = stochastic_round_bf16(m32, seed=seed, offset=offset)
        stored_v = stochastic_round_bf16(v32, seed=seed, offset=offset + m32.numel())
    else:
        stored_m, stored_v = m32.to(torch.bfloat16), v32.to(torch.bfloat16)
    return updated_parameter, stored_m, stored_v


@torch.no_grad()
def adamw_step_(parameter, gradient, exp_avg, exp_avg_sq, **kwargs) -> None:
    """Apply the reference update in place, including state write-back."""
    updated_p, updated_m, updated_v = adamw_step(parameter, gradient, exp_avg, exp_avg_sq, **kwargs)
    parameter.copy_(updated_p)
    exp_avg.copy_(updated_m)
    exp_avg_sq.copy_(updated_v)
