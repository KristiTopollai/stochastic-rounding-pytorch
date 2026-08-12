import math

import pytest
import torch

from sr_states.reference.rounding import bf16_neighbors, stochastic_round_bf16


def test_output_is_one_of_the_two_bf16_neighbors():
    generator = torch.Generator().manual_seed(123)
    x = torch.randn(20_000, generator=generator, dtype=torch.float32) * 100
    output = stochastic_round_bf16(x, seed=7)
    lower, upper = bf16_neighbors(x)
    assert torch.all((output == lower) | (output == upper))


def test_exact_values_and_exceptional_values_are_preserved():
    values = torch.tensor([0.0, -0.0, 1.5, -2.0, math.inf, -math.inf, math.nan])
    output = stochastic_round_bf16(values, seed=999)
    assert torch.equal(output[:6], values[:6].to(torch.bfloat16))
    assert torch.signbit(output[1])
    assert torch.isnan(output[6])


def test_nan_with_payload_only_in_discarded_bits_stays_nan():
    low_payload_nan = torch.tensor([0x7F800001], dtype=torch.int32).view(torch.float32)
    output = stochastic_round_bf16(low_payload_nan, seed=1)
    assert torch.isnan(output).all()


def test_smallest_fp32_subnormal_rounds_to_a_valid_bf16_neighbor():
    tiny = torch.nextafter(torch.tensor(0.0), torch.tensor(1.0)).reshape(1)
    output = stochastic_round_bf16(tiny, seed=3)
    lower, upper = bf16_neighbors(tiny)
    assert torch.all((output == lower) | (output == upper))


def test_fp32_subnormal_grid_and_bf16_midpoint_boundaries():
    raw_bits = torch.tensor(
        [
            0x00000001,
            0x00008000,
            0x0000FFFF,
            0x00010000,
            0x3F807FFF,
            0x3F808000,
            0x3F808001,
        ],
        dtype=torch.int32,
    )
    values = raw_bits.view(torch.float32)
    output = stochastic_round_bf16(values, seed=867)
    lower, upper = bf16_neighbors(values)
    assert torch.all((output == lower) | (output == upper))


def test_rejects_non_fp32_input():
    with pytest.raises(TypeError, match="float32"):
        stochastic_round_bf16(torch.ones(4, dtype=torch.float64), seed=0)
