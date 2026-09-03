import argparse

import pytest
import torch

from benchmarks.bench_sgdm import _logical_bytes_per_element, nonnegative_int, positive_int


@pytest.mark.parametrize(("dtype", "expected"), [(torch.bfloat16, 10), (torch.float32, 16)])
def test_sgdm_logical_bytes_per_element(dtype, expected):
    assert _logical_bytes_per_element(dtype) == expected


@pytest.mark.parametrize("value", ["1", "1024"])
def test_positive_int_accepts_positive_values(value):
    assert positive_int(value) == int(value)


@pytest.mark.parametrize("value", ["0", "-1"])
def test_positive_int_rejects_nonpositive_values(value):
    with pytest.raises(argparse.ArgumentTypeError, match="positive"):
        positive_int(value)


def test_nonnegative_int_accepts_zero():
    assert nonnegative_int("0") == 0


def test_nonnegative_int_rejects_negative_values():
    with pytest.raises(argparse.ArgumentTypeError, match="nonnegative"):
        nonnegative_int("-1")
