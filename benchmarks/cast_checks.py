"""Bit-level checks shared by the cast tuning run and GPU tests."""

from __future__ import annotations

import torch

from sr_states.reference import stochastic_round_bf16


def cast_cases():
    generator = torch.Generator().manual_seed(91)
    for n in (0, 1, 127, 128, 129, 255, 257, 513, 4097, 100_003):
        bits = torch.randint(0, 2**32, (n,), generator=generator, dtype=torch.int64)
        yield bits.to(torch.int32).view(torch.float32), 7, 2**32 - 13
    # Every exact BF16 encoding, including signs, subnormals and NaNs.
    exact = (torch.arange(65536, dtype=torch.int64) << 16).to(torch.int32)
    yield exact.view(torch.float32).reshape(256, 256), 0, 0
    boundaries = torch.tensor(
        [
            0,
            1,
            0x7FFF,
            0x8000,
            0xFFFF,
            0x10000,
            0x7FFFFF,
            0x800000,
            0x3F807FFF,
            0x3F808000,
            0x3F808001,
            0x7F7FFFFF,
            0x7F800000,
            0x7F800001,
            0x7FC00000,
        ],
        dtype=torch.int64,
    )
    bits = torch.cat((boundaries, boundaries | 0x80000000)).to(torch.int32)
    for seed, offset in ((2**32 + 9, -17), (991, 2**32 - 31)):
        yield bits.view(torch.float32).repeat(19), seed, offset


def assert_cast_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.dtype == torch.bfloat16
    assert actual.shape == expected.shape and actual.device == expected.device
    nan = torch.isnan(expected)
    assert torch.equal(torch.isnan(actual), nan)
    # NaN payloads are not part of the cast API. All other bits are, including -0.
    assert torch.equal(actual.view(torch.int16)[~nan], expected.view(torch.int16)[~nan])


def prepare_cases(device):
    return [
        (x.to(device), stochastic_round_bf16(x, seed=seed, offset=offset).to(device), seed, offset)
        for x, seed, offset in cast_cases()
    ]


def prepare_rn_cases(device):
    inputs = [x for x, _, _ in cast_cases()]
    # Both sides of every positive finite midpoint, including even/odd ties,
    # subnormals and the overflow boundary; repeat with the sign bit set.
    high = torch.arange(0x7F80, dtype=torch.int64) << 16
    bits = (high[:, None] + torch.tensor([0x7FFF, 0x8000, 0x8001])).flatten()
    bits = torch.cat((bits, bits | 0x80000000)).to(torch.int32)
    inputs.append(bits.view(torch.float32))
    return [(x.to(device), x.to(torch.bfloat16).to(device), 0, 0) for x in inputs]


def validate_cast(function, cases) -> None:
    for x, expected, seed, offset in cases:
        original = x.view(torch.int32).clone()
        first = function(x, seed=seed, offset=offset)
        second = function(x, seed=seed, offset=offset)
        assert_cast_equal(first, expected)
        assert_cast_equal(second, expected)
        assert torch.equal(x.view(torch.int32), original), "cast modified its input"
        if x.numel():
            assert first.data_ptr() != second.data_ptr(), "cast reused a live output buffer"
            assert first.data_ptr() != x.data_ptr(), "cast output aliases its input"
