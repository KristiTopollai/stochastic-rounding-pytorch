import math

import pytest
import torch

from sr_states.reference.rounding import stochastic_round_bf16
from sr_states.triton import is_triton_available, sr_cast_bf16

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not is_triton_available(), reason="requires Triton and CUDA"),
]


@pytest.mark.parametrize("block_size", [128, 256, 512, 1024])
def test_triton_cast_matches_reference_across_block_sizes(block_size):
    x = torch.linspace(-10, 10, 100_003, device="cuda", dtype=torch.float32)
    expected = stochastic_round_bf16(x, seed=991, offset=17)
    actual = sr_cast_bf16(x, seed=991, offset=17, block_size=block_size)
    assert torch.equal(actual, expected)


def test_triton_cast_preserves_exceptional_values():
    ordinary = torch.tensor(
        [0.0, -0.0, 1.5, -2.0, math.inf, -math.inf, math.nan],
        device="cuda",
        dtype=torch.float32,
    )
    low_payload_nan = torch.tensor([0x7F800001], device="cuda", dtype=torch.int32).view(
        torch.float32
    )
    x = torch.cat([ordinary, low_payload_nan])
    output = sr_cast_bf16(x, seed=31)

    assert torch.equal(output[:6], ordinary[:6].to(torch.bfloat16))
    assert torch.signbit(output[1])
    assert torch.isnan(output[6:]).all()


def test_triton_cast_accepts_empty_tensor():
    x = torch.empty(0, device="cuda", dtype=torch.float32)
    output = sr_cast_bf16(x, seed=1)
    assert output.dtype == torch.bfloat16
    assert output.shape == x.shape
