"""Validation shared by fused state updates."""

import math

import torch

from sr_states.rounding import validate_rounding

BLOCK_SIZES = (128, 256, 512, 1024)
_PARAMETER_DTYPES = (torch.float32, torch.bfloat16)


def validate_options(rounding, block_size, state_dtype):
    validate_rounding(state_dtype, rounding, backend="triton")
    if block_size not in BLOCK_SIZES:
        raise ValueError(f"block_size must be one of {BLOCK_SIZES}")


def nonnegative(name, value):
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"invalid {name}: {value}")


def validate_tensor_layout(tensors):
    parameter, gradient, *_ = tensors
    device, shape = parameter.device, parameter.shape
    for tensor in tensors:
        if tensor.device != device:
            raise ValueError("all tensors must be on the same CUDA device")
        if tensor.shape != shape:
            raise ValueError("all tensors must have identical shapes")
        if not tensor.is_contiguous():
            raise ValueError("fused state updates require contiguous tensors")
    if parameter.dtype not in _PARAMETER_DTYPES or gradient.dtype not in _PARAMETER_DTYPES:
        raise TypeError("parameter and gradient must be FP32 or BF16")


def validate_no_overlap(spans):
    # Sweep byte spans on a single device. Two gradients can share storage, but
    # anything written by the optimizer must be disjoint from every other span.
    max_end = max_write_end = 0
    for start, end, write in sorted(spans):
        if start < (max_end if write else max_write_end):
            raise ValueError("parameter, gradient, and state must not overlap optimizer writes")
        max_end = max(max_end, end)
        if write:
            max_write_end = max(max_write_end, end)


def validate_tensors(parameter, gradient, *states):
    tensors = (parameter, gradient, *states)
    if not parameter.is_cuda:
        raise ValueError("fused state updates require CUDA tensors")
    validate_tensor_layout(tensors)
    if parameter.numel():
        pointers = (tensor.data_ptr() for tensor in tensors)
        validate_no_overlap(
            (ptr, ptr + tensor.nbytes, role != 1)
            for role, (ptr, tensor) in enumerate(zip(pointers, tensors))
        )
