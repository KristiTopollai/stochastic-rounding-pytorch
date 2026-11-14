"""Compilation entry points kept separate from the eager numerical reference.

These helpers intentionally compile the same functional code used by the eager
reference. That controls for algorithmic differences when measuring what
Inductor can fuse.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from sr_states._hash import state_key
from sr_states.reference.adamw import _adamw_step, _adamw_step_
from sr_states.reference.rounding import stochastic_round_bf16
from sr_states.reference.sgd import _sgdm_step, _sgdm_step_
from sr_states.rounding import cast


def _compile(function: Callable[..., Any], **kwargs) -> Callable[..., Any]:
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile requires PyTorch 2.0 or newer")
    return torch.compile(function, **kwargs)


def _compile_update(function, state_count, **kwargs):
    compiled = _compile(function, **kwargs)

    def update(*args, seed=0, **kwargs):
        # PyTorch 2.5 cannot trace bitwise operations on dynamic integer arguments.
        state_seeds = tuple(state_key(seed, index) for index in range(state_count))
        return compiled(*args, state_seeds=state_seeds, **kwargs)

    return update


def compile_sr_cast(**kwargs) -> Callable[..., torch.Tensor]:
    """Compile the reference stochastic BF16 cast with ``torch.compile``."""

    return _compile(stochastic_round_bf16, **kwargs)


def compile_cast(**kwargs) -> Callable[..., torch.Tensor]:
    """Compile the generic cast, including registered reference rounding routines."""
    return _compile(cast, **kwargs)


def compile_sgdm_step(**kwargs) -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
    """Compile the functional SGDM reference."""

    return _compile_update(_sgdm_step, 1, **kwargs)


def compile_adamw_step(
    **kwargs,
) -> Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Compile the functional AdamW reference."""

    return _compile_update(_adamw_step, 2, **kwargs)


def compile_sgdm_step_(**kwargs) -> Callable[..., None]:
    """Compile the complete in-place SGDM update."""
    return torch.no_grad()(_compile_update(_sgdm_step_, 1, **kwargs))


def compile_adamw_step_(**kwargs) -> Callable[..., None]:
    """Compile the complete in-place AdamW update."""
    return torch.no_grad()(_compile_update(_adamw_step_, 2, **kwargs))
