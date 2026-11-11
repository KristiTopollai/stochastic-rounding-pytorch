"""Allocating Triton casts using the same rounding registry as fused optimizers."""

import torch

from sr_states.rounding import get_triton_rounding, validate_rounding

from ._common import require_triton, tl, triton, validate_fp8_device
from .sr_cast import CAST_BLOCK_SIZES, CAST_NUM_WARPS

if triton is not None:

    @triton.jit
    def _cast_kernel(
        input_ptr,
        output_ptr,
        n_elements,
        seed,
        logical_offset,
        ROUND: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        indices = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = indices < n_elements
        value = tl.load(input_ptr + indices, mask=mask, other=0.0).to(tl.float32)
        counter = (
            indices.to(tl.uint32) + tl.cast(seed, tl.uint32) + tl.cast(logical_offset, tl.uint32)
        )
        rounded = ROUND(value, counter, output_ptr.dtype.element_ty)
        tl.store(output_ptr + indices, rounded, mask=mask)


def cast(x, *, dtype, rounding="stochastic", seed=0, offset=0, block_size=256, num_warps=4):
    """Cast contiguous CUDA FP32 input with a registered Triton rounding routine."""
    require_triton()
    validate_rounding(dtype, rounding, backend="triton")
    if not x.is_cuda:
        raise ValueError("Triton cast requires a CUDA tensor")
    if x.dtype != torch.float32:
        raise TypeError(f"expected torch.float32 input, got {x.dtype}")
    if not x.is_contiguous():
        raise ValueError("Triton cast requires a contiguous input")
    if block_size not in CAST_BLOCK_SIZES:
        raise ValueError(f"block_size must be one of {CAST_BLOCK_SIZES}")
    if num_warps not in CAST_NUM_WARPS:
        raise ValueError(f"num_warps must be one of {CAST_NUM_WARPS}")
    validate_fp8_device(x.device, dtype)
    output = torch.empty_like(x, dtype=dtype)
    if x.numel():
        with torch.cuda.device(x.device):
            _cast_kernel[(triton.cdiv(x.numel(), block_size),)](
                x,
                output,
                x.numel(),
                int(seed) & 0xFFFFFFFF,
                int(offset) & 0xFFFFFFFF,
                ROUND=get_triton_rounding(dtype, rounding),
                BLOCK_SIZE=block_size,
                num_warps=num_warps,
            )
    return output
