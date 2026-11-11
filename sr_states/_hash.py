"""Counter hash shared by elementwise stochastic rounding rules."""

import torch

_U32_MASK = 0xFFFFFFFF
_HASH_MUL_1 = 0x7FEB352D
_HASH_MUL_2 = 0x846CA68B


def _u32_hash(indices: torch.Tensor, seed: int) -> torch.Tensor:
    """A stateless 32-bit integer hash shared with the Triton kernels.

    Arithmetic is performed in int64 with a mask after each multiplication.
    This avoids signed-int32 overflow differences across PyTorch backends while
    retaining the modulo-2**32 semantics of the Triton uint32 implementation.
    """

    # Explicitly type scalar counters: under dynamic compilation Python integer
    # arguments become SymInts, whose scalar arithmetic can otherwise be lowered
    # as FP32 by Inductor before these bitwise operations.
    seed_tensor = torch.scalar_tensor(
        int(seed) % (1 << 32), dtype=torch.int64, device=indices.device
    )
    value = (indices.to(torch.int64) + seed_tensor) & _U32_MASK
    value = value ^ (value >> 16)
    value = (value * _HASH_MUL_1) & _U32_MASK
    value = value ^ (value >> 15)
    value = (value * _HASH_MUL_2) & _U32_MASK
    return (value ^ (value >> 16)) & _U32_MASK
