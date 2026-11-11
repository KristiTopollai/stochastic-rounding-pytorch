"""Functional SGDM reference updates."""

from __future__ import annotations

import torch

from sr_states.formats import resolve_formats
from sr_states.rounding import cast


def sgdm_step(
    parameter: torch.Tensor,
    gradient: torch.Tensor,
    momentum: torch.Tensor,
    *,
    lr: float,
    momentum_factor: float,
    state_dtype: torch.dtype | None = None,
    formats: tuple | dict | None = None,
    rounding: str | None = None,
    seed: int = 0,
    offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one SGDM update, computing all arithmetic in FP32."""

    resolved = resolve_formats(
        ("momentum_buffer",), formats, state_dtype=state_dtype, rounding=rounding
    )

    p32 = parameter.float()
    u32 = momentum.float().mul(momentum_factor).add(gradient.float())
    updated_parameter = p32.add(u32, alpha=-lr).to(parameter.dtype)
    stored_momentum = cast(
        u32, dtype=resolved[0].dtype, rounding=resolved[0].rounding, seed=seed, offset=offset
    )
    return updated_parameter, stored_momentum


@torch.no_grad()
def sgdm_step_(parameter, gradient, momentum, **kwargs) -> None:
    """Apply the reference update in place, including state write-back."""
    updated_p, updated_m = sgdm_step(parameter, gradient, momentum, **kwargs)
    parameter.copy_(updated_p)
    momentum.copy_(updated_m)
