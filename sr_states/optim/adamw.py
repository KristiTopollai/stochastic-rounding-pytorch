"""PyTorch AdamW with per-state storage formats."""

from __future__ import annotations

import math

from sr_states.reference.adamw import adamw_step_

from ._state import StateOptimizer


def validate_adamw(group):
    for name in ("lr", "eps", "weight_decay"):
        if not math.isfinite(group[name]) or group[name] < 0:
            raise ValueError(f"invalid {name}: {group[name]}")
    betas = group["betas"]
    if len(betas) != 2 or any(not 0 <= beta < 1 for beta in betas):
        raise ValueError(f"invalid beta values: {betas}")


class AdamWReferenceSR(StateOptimizer):
    """AdamW with FP32 math and a configurable format for each moment."""

    _state_names = ("exp_avg", "exp_avg_sq")
    _uses_step = True
    _update = staticmethod(adamw_step_.__wrapped__)

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        *,
        state_format=None,
        state_dtype=None,
        rounding=None,
        seed=0,
    ):
        super().__init__(
            params,
            {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay},
            state_format=state_format,
            state_dtype=state_dtype,
            rounding=rounding,
            seed=seed,
        )

    def _validate_group(self, group):
        validate_adamw(group)

    def _step_parameter(self, parameter, state, group, formats, step, offset):
        beta1, beta2 = group["betas"]
        self._update(
            parameter,
            parameter.grad,
            state["exp_avg"],
            state["exp_avg_sq"],
            lr=group["lr"],
            beta1=beta1,
            beta2=beta2,
            eps=group["eps"],
            weight_decay=group["weight_decay"],
            step=step,
            formats=formats,
            seed=group["sr_seed"],
            offset=offset,
        )


class AdamWReferenceRN(AdamWReferenceSR):
    """Deterministic round-to-nearest AdamW reference (BF16 by default).

    Accepts the same storage options as ``AdamWReferenceSR``; they select only
    each state's dtype, and every state uses nearest rounding.
    """

    _forced_rounding = "nearest"
