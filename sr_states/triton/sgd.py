"""Fused Triton momentum-SGD with stochastic BF16 state write-back."""

from __future__ import annotations

from collections.abc import Iterable

import torch

from sr_states.optim._state import StateOptimizer

from ._common import require_triton, tl, triton
from ._optimizer import nonnegative, validate_options, validate_tensors

if triton is not None:
    from .sr_cast import _stochastic_bf16_value

    @triton.jit
    def _fused_sgdm_kernel(
        parameter_ptr,
        gradient_ptr,
        momentum_ptr,
        n_elements,
        learning_rate,
        momentum_factor,
        seed,
        logical_offset,
        STOCHASTIC: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        # Storage may be BF16, but the optimizer recurrence is evaluated in
        # FP32 before either result is written back to its destination dtype.
        parameter = tl.load(parameter_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        gradient = tl.load(gradient_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        momentum = tl.load(momentum_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        updated_momentum = momentum_factor * momentum + gradient
        updated_parameter = parameter - learning_rate * updated_momentum

        if STOCHASTIC:
            # Logical indexing keeps the random choice independent of the
            # Triton launch partition and consistent across parameter tensors.
            counter = (
                offsets.to(tl.uint32)
                + tl.cast(seed, tl.uint32)
                + tl.cast(logical_offset, tl.uint32)
            )
            stored_momentum = _stochastic_bf16_value(updated_momentum, counter)
        else:
            stored_momentum = updated_momentum

        tl.store(parameter_ptr + offsets, updated_parameter, mask=mask)
        tl.store(momentum_ptr + offsets, stored_momentum, mask=mask)


@torch.no_grad()
def triton_sgdm_step_(
    parameter: torch.Tensor,
    gradient: torch.Tensor,
    momentum: torch.Tensor,
    *,
    lr: float,
    momentum_factor: float,
    seed: int,
    offset: int = 0,
    rounding: str = "stochastic",
    block_size: int = 256,
) -> None:
    """Apply one fused momentum-SGD update in place.

    ``parameter``, ``gradient``, and ``momentum`` must be same-shaped,
    contiguous CUDA tensors on one device. Parameter and gradient storage may
    be FP32 or BF16; momentum storage may be FP32 or BF16. Arithmetic is performed in FP32.

    The function mutates ``parameter`` and ``momentum`` and returns ``None``.
    ``offset`` is the logical starting index for counter-based randomness.
    """

    require_triton()
    validate_options(rounding, block_size, momentum.dtype)
    nonnegative("learning rate", lr)
    if not 0 <= momentum_factor < 1:
        raise ValueError(f"invalid momentum value: {momentum_factor}")
    validate_tensors(parameter, gradient, momentum)
    if parameter.numel() == 0:
        return

    grid = (triton.cdiv(parameter.numel(), block_size),)
    with torch.cuda.device(parameter.device):
        _fused_sgdm_kernel[grid](
            parameter,
            gradient,
            momentum,
            parameter.numel(),
            lr,
            momentum_factor,
            seed=int(seed) & 0xFFFFFFFF,
            logical_offset=int(offset) & 0xFFFFFFFF,
            STOCHASTIC=rounding == "stochastic" and momentum.dtype == torch.bfloat16,
            BLOCK_SIZE=block_size,
            num_warps=4,
        )


class SGDMTriton(StateOptimizer):
    """Momentum SGD backed by the fused Triton BF16-state kernel.

    The seed and logical RNG offset live in each parameter group, so the
    standard optimizer ``state_dict`` contains everything needed to reproduce
    the next stochastic update after checkpoint restore.
    """

    _state_names = ("momentum_buffer",)

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float,
        momentum: float = 0.9,
        *,
        state_dtype: torch.dtype = torch.bfloat16,
        rounding: str = "stochastic",
        seed: int = 0,
        block_size: int = 256,
    ) -> None:
        nonnegative("learning rate", lr)
        if not 0 <= momentum < 1:
            raise ValueError(f"invalid momentum value: {momentum}")
        validate_options(rounding, block_size, state_dtype)

        defaults = {
            "lr": lr,
            "momentum": momentum,
            "state_dtype": state_dtype,
            "rounding": rounding,
            "sr_seed": int(seed),
            "sr_offset": 0,
            "block_size": block_size,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        offset = self._next_offset()
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.grad.is_sparse:
                    raise RuntimeError("SGDMTriton does not support sparse gradients")

                state = self.state[parameter]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(
                        parameter,
                        dtype=group.get("state_dtype", torch.bfloat16),
                        memory_format=torch.preserve_format,
                    )
                triton_sgdm_step_(
                    parameter,
                    parameter.grad,
                    state["momentum_buffer"],
                    lr=group["lr"],
                    momentum_factor=group["momentum"],
                    seed=group["sr_seed"],
                    offset=offset,
                    rounding=group["rounding"],
                    block_size=group["block_size"],
                )
                offset += parameter.numel()
                group["sr_offset"] = offset
        return loss
