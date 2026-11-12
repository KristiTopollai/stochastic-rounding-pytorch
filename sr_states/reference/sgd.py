"""Functional SGDM reference updates."""

from __future__ import annotations

import torch

from sr_states._hash import state_key
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
    """Return one FP32 SGDM update.

    The momentum state derives its counter base from the update key ``seed``.
    ``offset`` is the flat starting index of this shard.
    """
    return _sgdm_step(
        parameter,
        gradient,
        momentum,
        lr=lr,
        momentum_factor=momentum_factor,
        state_dtype=state_dtype,
        formats=formats,
        rounding=rounding,
        state_seeds=(state_key(seed, 0),),
        offset=offset,
    )


def _sgdm_step(
    parameter,
    gradient,
    momentum,
    *,
    lr,
    momentum_factor,
    state_seeds,
    state_dtype=None,
    formats=None,
    rounding=None,
    offset=0,
):

    resolved = resolve_formats(
        ("momentum_buffer",), formats, state_dtype=state_dtype, rounding=rounding
    )

    p32 = parameter.float()
    u32 = momentum.float().mul(momentum_factor).add(gradient.float())
    updated_parameter = p32.add(u32, alpha=-lr).to(parameter.dtype)
    stored_momentum = cast(
        u32,
        dtype=resolved[0].dtype,
        rounding=resolved[0].rounding,
        seed=state_seeds[0],
        offset=offset,
    )
    return updated_parameter, stored_momentum


@torch.no_grad()
def sgdm_step_(parameter, gradient, momentum, *, seed=0, **kwargs) -> None:
    """Apply the reference update in place, including state write-back."""
    _sgdm_step_(parameter, gradient, momentum, state_seeds=(state_key(seed, 0),), **kwargs)


def _sgdm_step_(parameter, gradient, momentum, **kwargs) -> None:
    updated_p, updated_m = _sgdm_step(parameter, gradient, momentum, **kwargs)
    parameter.copy_(updated_p)
    momentum.copy_(updated_m)
