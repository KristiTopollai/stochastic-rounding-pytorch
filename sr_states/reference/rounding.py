"""Reference casting API; format implementations live outside optimizer code."""

import torch

from sr_states._bf16 import bf16_neighbors
from sr_states._minifloat import minifloat_neighbors
from sr_states.rounding import cast

__all__ = ["bf16_neighbors", "cast", "minifloat_neighbors", "stochastic_round_bf16"]


def stochastic_round_bf16(x, *, seed, offset=0):
    """Compatibility wrapper for ``cast(x, dtype=torch.bfloat16, ...)``."""
    return cast(x, dtype=torch.bfloat16, rounding="stochastic", seed=seed, offset=offset)
