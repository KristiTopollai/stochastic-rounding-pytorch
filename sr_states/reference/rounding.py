"""Readable FP32-to-BF16 stochastic rounding in ordinary PyTorch operations.

The implementation uses the structure of IEEE-754 binary32 and bfloat16. For
finite inputs, the low 16 bits of the FP32 representation encode exactly the
position between the two adjacent BF16 values within a bin. Adding a uniform
16-bit integer and clearing the low bits therefore selects the two neighbors
with the probabilities required for unbiased stochastic rounding.
"""

from __future__ import annotations

import torch

_U32_MASK = 0xFFFFFFFF
_BF16_MASK = 0xFFFF0000
_EXPONENT_MASK = 0x7F800000
_MANTISSA_MASK = 0x007FFFFF
_BF16_QUIET_NAN_BIT = 0x00400000
_HASH_MUL_1 = 0x7FEB352D
_HASH_MUL_2 = 0x846CA68B


def _require_fp32(x: torch.Tensor) -> None:
    if x.dtype != torch.float32:
        raise TypeError(f"expected torch.float32 input, got {x.dtype}")


def _u32_hash(indices: torch.Tensor, seed: int) -> torch.Tensor:
    """A stateless 32-bit integer hash shared with the Triton kernels.

    Arithmetic is performed in int64 with a mask after each multiplication.
    This avoids signed-int32 overflow differences across PyTorch backends while
    retaining the modulo-2**32 semantics of the Triton uint32 implementation.
    """

    value = (indices.to(torch.int64) + (int(seed) & _U32_MASK)) & _U32_MASK
    value = value ^ (value >> 16)
    value = (value * _HASH_MUL_1) & _U32_MASK
    value = value ^ (value >> 15)
    value = (value * _HASH_MUL_2) & _U32_MASK
    return (value ^ (value >> 16)) & _U32_MASK


def _fp32_to_u32(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous().view(torch.int32).to(torch.int64) & _U32_MASK


def _u32_to_fp32(bits: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    return bits.to(torch.int32).view(torch.float32).reshape(shape)


def stochastic_round_bf16(
    x: torch.Tensor,
    *,
    seed: int,
    offset: int = 0,
) -> torch.Tensor:
    """Stochastically round an FP32 tensor to BF16 with stateless RNG.

    Args:
        x: FP32 input tensor. Non-contiguous inputs are accepted.
        seed: Counter-hash seed. Only its low 32 bits are used.
        offset: Logical starting element index. Combining ``seed`` and
            ``offset`` makes random choices reproducible and independent of
            launch/block shape.

    Returns:
        A BF16 tensor with the same shape and device as ``x``.

    Exceptional values are handled deliberately: infinities retain their sign,
    NaNs remain NaNs (their low payload bits may be discarded), and signed zero
    is preserved. Exact BF16 values are unchanged because their discarded bits
    are zero.
    """

    _require_fp32(x)
    flat = x.contiguous().reshape(-1)
    bits = _fp32_to_u32(flat)
    indices = torch.arange(flat.numel(), dtype=torch.int64, device=x.device)
    random_low = _u32_hash(indices + int(offset), seed) & 0xFFFF

    truncated = bits & _BF16_MASK
    rounded = (bits + random_low) & _BF16_MASK
    special = (bits & _EXPONENT_MASK) == _EXPONENT_MASK
    nan = special & ((bits & _MANTISSA_MASK) != 0)
    preserved_special = torch.where(nan, truncated | _BF16_QUIET_NAN_BIT, truncated)
    rounded = torch.where(special, preserved_special, rounded)
    return _u32_to_fp32(rounded, x.shape).to(torch.bfloat16)


def bf16_neighbors(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the ordered BF16 neighbors bracketing each finite FP32 input.

    Exact BF16 inputs return the same value twice. For NaN and infinities both
    outputs are the ordinary BF16 conversion because an ordered bracket is not
    defined for those values.
    """

    _require_fp32(x)
    bits = _fp32_to_u32(x)
    truncated_bits = bits & _BF16_MASK
    truncated = _u32_to_fp32(truncated_bits, x.shape)
    exact_or_special = ((bits & 0xFFFF) == 0) | ((bits & _EXPONENT_MASK) == _EXPONENT_MASK)
    negative = (bits & 0x80000000) != 0

    # Clearing the low bits moves toward zero. The adjacent value away from
    # zero is obtained by incrementing the encoded BF16 word for either sign.
    away_bits = (truncated_bits + 0x00010000) & _U32_MASK
    away = _u32_to_fp32(away_bits, x.shape)
    lower = torch.where(negative, away, truncated)
    upper = torch.where(negative, truncated, away)
    converted = x.to(torch.bfloat16).to(torch.float32)
    lower = torch.where(exact_or_special, converted, lower)
    upper = torch.where(exact_or_special, converted, upper)
    return lower.to(torch.bfloat16), upper.to(torch.bfloat16)
