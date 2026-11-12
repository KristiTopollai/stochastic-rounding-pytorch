"""Fused Triton momentum SGD with configurable state storage."""

from __future__ import annotations

import torch

from sr_states.formats import StateFormat, resolve_formats
from sr_states.optim.sgd import SGDMReferenceSR
from sr_states.rounding import get_triton_rounding

from ._common import require_triton, tl, triton, validate_fp8_device
from ._optimizer import nonnegative, validate_options, validate_tensors

if triton is not None:
    from .sr_cast import _u32_hash

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
        ROUND: tl.constexpr,
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

        counter = (
            offsets.to(tl.uint32)
            + _u32_hash(tl.cast(seed, tl.uint32))
            + tl.cast(logical_offset, tl.uint32)
        )
        stored_momentum = ROUND(updated_momentum, counter, momentum_ptr.dtype.element_ty)

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
    rounding: str | None = None,
    formats: tuple | dict | None = None,
    block_size: int = 256,
) -> None:
    """Apply one fused momentum-SGD update in place.

    ``parameter``, ``gradient``, and ``momentum`` must be same-shaped,
    contiguous CUDA tensors on one device. Parameter and gradient storage may
    be FP32 or BF16; momentum uses any registered Triton storage format. Arithmetic is performed in FP32.

    The function mutates ``parameter`` and ``momentum`` and returns ``None``.
    The momentum state derives its counter base from the update key ``seed``.
    ``offset`` is the flat starting index of this shard.
    """

    require_triton()
    if formats is None:
        formats = (StateFormat(momentum.dtype, "stochastic" if rounding is None else rounding),)
        rounding = None
    resolved = resolve_formats(("momentum_buffer",), formats, rounding=rounding, backend="triton")
    fmt = resolved[0]
    validate_options(fmt.rounding, block_size, fmt.dtype)
    if momentum.dtype != fmt.dtype:
        raise ValueError("state dtype does not match its state format")
    nonnegative("learning rate", lr)
    if not 0 <= momentum_factor < 1:
        raise ValueError(f"invalid momentum value: {momentum_factor}")
    validate_tensors(parameter, gradient, momentum)
    validate_fp8_device(parameter.device, *(fmt.dtype for fmt in resolved))
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
            ROUND=get_triton_rounding(fmt.dtype, fmt.rounding),
            BLOCK_SIZE=block_size,
            num_warps=4,
        )


class SGDMTriton(SGDMReferenceSR):
    """Fused momentum SGD with configurable state storage and checkpointed RNG."""

    _rounding_backend = "triton"

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
        block_size=256,
    ):
        self._initial_block_size = block_size
        super().__init__(
            params,
            lr=lr,
            momentum=momentum,
            state_format=state_format,
            state_dtype=state_dtype,
            rounding=rounding,
            seed=seed,
        )
        self.defaults["block_size"] = block_size

    def _validate_group(self, group):
        super()._validate_group(group)
        validate_options(
            "nearest", group.get("block_size", self._initial_block_size), torch.float32
        )

    def add_param_group(self, param_group):
        super().add_param_group(
            {"block_size": self.defaults.get("block_size", self._initial_block_size), **param_group}
        )

    def _step_parameter(self, parameter, state, group, formats, step, seed, offset):
        triton_sgdm_step_(
            parameter,
            parameter.grad,
            state["momentum_buffer"],
            lr=group["lr"],
            momentum_factor=group["momentum"],
            formats=formats,
            seed=seed,
            offset=offset,
            block_size=group["block_size"],
        )
