"""Compilation entry points kept separate from the eager numerical reference.

These helpers intentionally compile the same functional code used by Stage A.
That controls for algorithmic differences when measuring what Inductor can fuse.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from sr_states.reference.adamw import adamw_step
from sr_states.reference.rounding import stochastic_round_bf16
from sr_states.reference.sgd import sgdm_step


def _compile(function: Callable[..., Any], **kwargs) -> Callable[..., Any]:
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile requires PyTorch 2.0 or newer")
    return torch.compile(function, **kwargs)


def compile_sr_cast(**kwargs) -> Callable[..., torch.Tensor]:
    """Compile the Stage A stochastic BF16 cast with ``torch.compile``."""

    return _compile(stochastic_round_bf16, **kwargs)


def compile_sgdm_step(**kwargs) -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
    """Compile the functional SGDM reference."""

    return _compile(sgdm_step, **kwargs)


def compile_adamw_step(
    **kwargs,
) -> Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Compile the functional AdamW reference."""

    return _compile(adamw_step, **kwargs)
