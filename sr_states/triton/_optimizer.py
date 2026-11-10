"""Validation shared by fused state updates."""

import itertools
import math

import torch

from sr_states.rounding import validate_rounding

BLOCK_SIZES = (128, 256, 512, 1024)


def validate_options(rounding, block_size, state_dtype):
    validate_rounding(state_dtype, rounding, backend="triton")
    if block_size not in BLOCK_SIZES:
        raise ValueError(f"block_size must be one of {BLOCK_SIZES}")


def nonnegative(name, value):
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"invalid {name}: {value}")


def validate_tensors(parameter, gradient, *states):
    tensors = (parameter, gradient, *states)
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("fused state updates require CUDA tensors")
    if any(tensor.device != parameter.device for tensor in tensors):
        raise ValueError("all tensors must be on the same CUDA device")
    if any(tensor.shape != parameter.shape for tensor in tensors):
        raise ValueError("all tensors must have identical shapes")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("fused state updates require contiguous tensors")
    if parameter.dtype not in {torch.float32, torch.bfloat16} or gradient.dtype not in {
        torch.float32,
        torch.bfloat16,
    }:
        raise TypeError("parameter and gradient must be FP32 or BF16")
    if any(state.dtype != states[0].dtype for state in states):
        raise TypeError("state tensors must have the same dtype")
    # Overlapping writable tensors would introduce inter-program races.
    spans = sorted((tensor.data_ptr(), tensor.data_ptr() + tensor.nbytes) for tensor in tensors)
    if any(end > start for (_, end), (start, _) in itertools.pairwise(spans)):
        raise ValueError("parameter, gradient, and state must not overlap")
