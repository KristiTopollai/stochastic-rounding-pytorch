"""Functional AdamW reference updates."""

from __future__ import annotations

import torch

from sr_states._hash import state_key
from sr_states.formats import resolve_formats
from sr_states.rounding import cast


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
    state_dtype: torch.dtype | None = None,
    formats: tuple | dict | None = None,
    rounding: str | None = None,
    seed: int = 0,
    offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return one FP32 AdamW update.

    Each moment derives its counter base from the update key ``seed``.
    ``offset`` is the flat starting index of this shard.
    """
    return _adamw_step(
        parameter,
        gradient,
        exp_avg,
        exp_avg_sq,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        weight_decay=weight_decay,
        step=step,
        state_dtype=state_dtype,
        formats=formats,
        rounding=rounding,
        state_seeds=(state_key(seed, 0), state_key(seed, 1)),
        offset=offset,
    )


def _adamw_step(
    parameter,
    gradient,
    exp_avg,
    exp_avg_sq,
    *,
    lr,
    beta1,
    beta2,
    eps,
    weight_decay,
    step,
    state_seeds,
    state_dtype=None,
    formats=None,
    rounding=None,
    offset=0,
):

    if step < 1:
        raise ValueError("step must be at least 1")
    resolved = resolve_formats(
        ("exp_avg", "exp_avg_sq"), formats, state_dtype=state_dtype, rounding=rounding
    )

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

    stored_m = cast(
        m32,
        dtype=resolved[0].dtype,
        rounding=resolved[0].rounding,
        seed=state_seeds[0],
        offset=offset,
    )
    stored_v = cast(
        v32,
        dtype=resolved[1].dtype,
        rounding=resolved[1].rounding,
        seed=state_seeds[1],
        offset=offset,
    )
    return updated_parameter, stored_m, stored_v


@torch.no_grad()
def adamw_step_(parameter, gradient, exp_avg, exp_avg_sq, *, seed=0, **kwargs) -> None:
    """Apply the reference update in place, including state write-back."""
    _adamw_step_(
        parameter,
        gradient,
        exp_avg,
        exp_avg_sq,
        state_seeds=(state_key(seed, 0), state_key(seed, 1)),
        **kwargs,
    )


def _adamw_step_(parameter, gradient, exp_avg, exp_avg_sq, **kwargs) -> None:
    updated_p, updated_m, updated_v = _adamw_step(
        parameter, gradient, exp_avg, exp_avg_sq, **kwargs
    )
    parameter.copy_(updated_p)
    exp_avg.copy_(updated_m)
    exp_avg_sq.copy_(updated_v)
