"""PyTorch momentum SGD with configurable state storage."""

from __future__ import annotations

import math

from sr_states.reference.sgd import sgdm_step_

from ._state import StateOptimizer


def validate_sgdm(group):
    if not math.isfinite(group["lr"]) or group["lr"] < 0:
        raise ValueError(f"invalid learning rate: {group['lr']}")
    if not 0 <= group["momentum"] < 1:
        raise ValueError(f"invalid momentum value: {group['momentum']}")


class SGDMReferenceSR(StateOptimizer):
    """Momentum SGD with FP32 math; BF16 stochastic storage by default."""

    _state_names = ("momentum_buffer",)
    _uses_step = False
    _update = staticmethod(sgdm_step_.__wrapped__)

    def __init__(
        self,
        params,
        lr,
        momentum=0.9,
        *,
        state_format=None,
        state_dtype=None,
        rounding=None,
        seed=0,
    ):
        super().__init__(
            params,
            {"lr": lr, "momentum": momentum},
            state_format=state_format,
            state_dtype=state_dtype,
            rounding=rounding,
            seed=seed,
        )

    def _validate_group(self, group):
        validate_sgdm(group)

    def _step_parameter(self, parameter, state, group, formats, step, offset):
        self._update(
            parameter,
            parameter.grad,
            state["momentum_buffer"],
            lr=group["lr"],
            momentum_factor=group["momentum"],
            formats=formats,
            seed=group["sr_seed"],
            offset=offset,
        )


class SGDMReferenceRN(SGDMReferenceSR):
    """Deterministic round-to-nearest momentum SGD reference (BF16 by default).

    Accepts the same storage options as ``SGDMReferenceSR``; they select only
    the state's dtype, and the state uses nearest rounding.
    """

    _forced_rounding = "nearest"
