"""FP32-to-BF16 stochastic rounding in a single Triton kernel."""

from __future__ import annotations

import torch

from ._common import require_triton, tl, triton


def is_triton_available() -> bool:
    """Return whether both Triton and a CUDA device are available."""

    return triton is not None and torch.cuda.is_available()


if triton is not None:

    @triton.jit
    def _u32_hash(counter):
        value = counter ^ (counter >> 16)
        value *= 0x7FEB352D
        value ^= value >> 15
        value *= 0x846CA68B
        return value ^ (value >> 16)

    @triton.jit
    def _stochastic_bf16_value(value, counter):
        bits = tl.cast(value, tl.uint32, bitcast=True)
        random_low = _u32_hash(counter) & 0xFFFF
        truncated = bits & 0xFFFF0000
        rounded = (bits + random_low) & 0xFFFF0000

        # Truncating a NaN whose payload only occupies discarded bits would
        # otherwise turn it into infinity. Force a retained quiet-NaN bit.
        special = (bits & 0x7F800000) == 0x7F800000
        nan = special & ((bits & 0x007FFFFF) != 0)
        preserved_special = tl.where(nan, truncated | 0x00400000, truncated)
        rounded = tl.where(special, preserved_special, rounded)
        return tl.cast(rounded, tl.float32, bitcast=True)

    @triton.jit
    def _sr_cast_bf16_kernel(
        input_ptr,
        output_ptr,
        n_elements,
        seed,
        logical_offset,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        value = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        counter = (
            offsets.to(tl.uint32) + tl.cast(seed, tl.uint32) + tl.cast(logical_offset, tl.uint32)
        )
        rounded = _stochastic_bf16_value(value, counter)
        tl.store(output_ptr + offsets, rounded, mask=mask)


def sr_cast_bf16(
    x: torch.Tensor,
    *,
    seed: int,
    offset: int = 0,
    block_size: int = 256,
) -> torch.Tensor:
    """Stochastically cast a contiguous CUDA FP32 tensor to BF16.

    Randomness is indexed by global logical element offset. Changing the Triton
    block size therefore changes execution partitioning without changing the
    random number assigned to an element.
    """

    require_triton()
    if not x.is_cuda:
        raise ValueError("sr_cast_bf16 requires a CUDA tensor")
    if x.dtype != torch.float32:
        raise TypeError(f"expected torch.float32 input, got {x.dtype}")
    if not x.is_contiguous():
        raise ValueError("sr_cast_bf16 requires a contiguous input")
    if block_size not in {128, 256, 512, 1024}:
        raise ValueError("block_size must be one of 128, 256, 512, or 1024")

    output = torch.empty_like(x, dtype=torch.bfloat16)
    if x.numel() == 0:
        return output

    grid = (triton.cdiv(x.numel(), block_size),)
    with torch.cuda.device(x.device):
        _sr_cast_bf16_kernel[grid](
            x,
            output,
            x.numel(),
            seed=int(seed) & 0xFFFFFFFF,
            logical_offset=int(offset) & 0xFFFFFFFF,
            BLOCK_SIZE=block_size,
        )
    return output
