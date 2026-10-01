"""Allocating FP32-to-BF16 casts with stochastic or nearest-even rounding."""

from __future__ import annotations

import torch

from ._common import require_triton, tl, triton

CAST_BLOCK_SIZES = (128, 256, 512, 1024, 2048, 4096)
CAST_NUM_WARPS = (1, 2, 4, 8)
CAST_STORE_MODES = ("convert", "bits")


def is_triton_available() -> bool:
    """Return whether both Triton and a CUDA device are available."""

    return triton is not None and torch.cuda.is_available()


if triton is not None:

    @triton.jit
    def _u32_hash(counter):
        # A small integer avalanche hash gives each logical element its own
        # deterministic pseudo-random word. This is stateless and deliberately
        # mirrors the arithmetic used by the PyTorch reference implementation.
        value = counter ^ (counter >> 16)
        value *= 0x7FEB352D
        value ^= value >> 15
        value *= 0x846CA68B
        return value ^ (value >> 16)

    @triton.jit
    def _stochastic_bf16_bits(value, counter):
        # An FP32 value lies between two adjacent BF16 values according to its
        # low 16 bits. Adding a uniform 16-bit integer before truncation rounds
        # upward with probability low_bits / 2**16 and downward otherwise.
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
        return rounded

    @triton.jit
    def _stochastic_bf16_value(value, counter):
        return tl.cast(_stochastic_bf16_bits(value, counter), tl.float32, bitcast=True)

    @triton.jit
    def _sr_cast_bf16_kernel(
        input_ptr,
        output_ptr,
        n_elements,
        seed,
        logical_offset,
        BLOCK_SIZE: tl.constexpr,
        STORE_BITS: tl.constexpr,
        STOCHASTIC: tl.constexpr = True,
    ):
        # The mask makes the final program safe when the tensor length is not a
        # multiple of BLOCK_SIZE.
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        value = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        if STOCHASTIC:
            # Logical indices preserve the random stream across launch shapes.
            counter = (
                offsets.to(tl.uint32)
                + tl.cast(seed, tl.uint32)
                + tl.cast(logical_offset, tl.uint32)
            )
            if STORE_BITS:
                packed = (_stochastic_bf16_bits(value, counter) >> 16).to(tl.uint16)
                rounded = tl.cast(packed, tl.bfloat16, bitcast=True)
            else:
                rounded = _stochastic_bf16_value(value, counter)
        else:
            rounded = tl.cast(value, tl.bfloat16, fp_downcast_rounding="rtne")
        tl.store(output_ptr + offsets, rounded, mask=mask)


def sr_cast_bf16(
    x: torch.Tensor,
    *,
    seed: int,
    offset: int = 0,
    block_size: int = 256,
    num_warps: int = 4,
    store_mode: str = "convert",
) -> torch.Tensor:
    """Stochastically cast a contiguous CUDA FP32 tensor to BF16.

    Randomness is indexed by global logical element offset. Changing the Triton
    block size therefore changes execution partitioning without changing the
    random number assigned to an element. ``store_mode="bits"`` writes the
    selected BF16 representation directly; the default retains the FP32 cast.
    """

    return _cast_bf16(
        x,
        seed=seed,
        offset=offset,
        block_size=block_size,
        num_warps=num_warps,
        store_mode=store_mode,
        stochastic=True,
    )


def rn_cast_bf16(x: torch.Tensor, *, block_size: int = 256, num_warps: int = 4) -> torch.Tensor:
    """Round to nearest, ties to even, using the SR allocation and launch path."""

    return _cast_bf16(
        x,
        seed=0,
        offset=0,
        block_size=block_size,
        num_warps=num_warps,
        store_mode="convert",
        stochastic=False,
    )


def _cast_bf16(x, *, seed, offset, block_size, num_warps, store_mode, stochastic):
    require_triton()
    if not x.is_cuda:
        raise ValueError("BF16 cast requires a CUDA tensor")
    if x.dtype != torch.float32:
        raise TypeError(f"expected torch.float32 input, got {x.dtype}")
    if not x.is_contiguous():
        raise ValueError("BF16 cast requires a contiguous input")
    if block_size not in CAST_BLOCK_SIZES:
        raise ValueError(f"block_size must be one of {CAST_BLOCK_SIZES}")
    if num_warps not in CAST_NUM_WARPS:
        raise ValueError(f"num_warps must be one of {CAST_NUM_WARPS}")
    if store_mode not in CAST_STORE_MODES:
        raise ValueError(f"store_mode must be one of {CAST_STORE_MODES}")

    # Both store paths allocate fresh BF16 output on every call.
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
            STORE_BITS=store_mode == "bits",
            STOCHASTIC=stochastic,
            num_warps=num_warps,
        )
    return output
