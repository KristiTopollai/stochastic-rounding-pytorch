"""Benchmark complete in-place SGDM state updates."""

import torch

from benchmarks._utils import nonnegative_int, positive_int
from benchmarks.bench_states import logical_bytes_per_element, main

__all__ = ["nonnegative_int", "positive_int"]


def _logical_bytes_per_element(parameter_dtype, state_dtype=torch.bfloat16):
    return logical_bytes_per_element("sgdm", parameter_dtype, state_dtype)


if __name__ == "__main__":
    main(optimizer="sgdm")
