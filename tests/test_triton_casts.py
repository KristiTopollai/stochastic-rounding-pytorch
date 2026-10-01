import pytest
import torch

from benchmarks.cast_checks import prepare_cases, prepare_rn_cases, validate_cast
from sr_states.triton import is_triton_available, rn_cast_bf16, sr_cast_bf16
from sr_states.triton.sr_cast import CAST_BLOCK_SIZES, CAST_NUM_WARPS, CAST_STORE_MODES

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not is_triton_available(), reason="requires Triton and CUDA"),
]


@pytest.fixture(scope="module")
def cases():
    return prepare_cases("cuda")


@pytest.fixture(scope="module")
def rn_cases():
    return prepare_rn_cases("cuda")


@pytest.mark.parametrize("block_size", CAST_BLOCK_SIZES)
@pytest.mark.parametrize("num_warps", CAST_NUM_WARPS)
def test_nearest_cast_matches_native_and_cpu(rn_cases, block_size, num_warps):
    validate_cast(lambda x, **_: x.to(torch.bfloat16), rn_cases)
    validate_cast(
        lambda x, **_: rn_cast_bf16(x, block_size=block_size, num_warps=num_warps), rn_cases
    )


def test_nearest_cast_rejects_invalid_inputs():
    with pytest.raises(TypeError, match="float32"):
        rn_cast_bf16(torch.ones(4, device="cuda", dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="contiguous"):
        rn_cast_bf16(torch.ones(8, device="cuda")[::2])
    with pytest.raises(ValueError, match="block_size"):
        rn_cast_bf16(torch.ones(4, device="cuda"), block_size=3)
    with pytest.raises(ValueError, match="num_warps"):
        rn_cast_bf16(torch.ones(4, device="cuda"), num_warps=3)


@pytest.mark.parametrize("block_size", CAST_BLOCK_SIZES)
@pytest.mark.parametrize("num_warps", CAST_NUM_WARPS)
@pytest.mark.parametrize("store_mode", CAST_STORE_MODES)
def test_launch_and_store_variants_preserve_cast_contract(cases, block_size, num_warps, store_mode):
    validate_cast(
        lambda x, **kwargs: sr_cast_bf16(
            x, block_size=block_size, num_warps=num_warps, store_mode=store_mode, **kwargs
        ),
        cases,
    )


@pytest.mark.parametrize("store_mode", CAST_STORE_MODES)
def test_variants_preserve_partitioned_rng(store_mode):
    x = torch.linspace(-3, 7, 10003, device="cuda")
    seed, offset, split = 77, 2**32 - 19, 4101
    whole = sr_cast_bf16(x, seed=seed, offset=offset, store_mode=store_mode)
    parts = torch.cat(
        [
            sr_cast_bf16(
                x[:split],
                seed=seed,
                offset=offset,
                block_size=128,
                num_warps=1,
                store_mode=store_mode,
            ),
            sr_cast_bf16(
                x[split:],
                seed=seed,
                offset=offset + split,
                block_size=2048,
                num_warps=8,
                store_mode=store_mode,
            ),
        ]
    )
    assert torch.equal(whole.view(torch.int16), parts.view(torch.int16))


@pytest.mark.parametrize("kwargs", [{"block_size": 3}, {"num_warps": 3}, {"store_mode": "invalid"}])
def test_invalid_launch_options_are_rejected(kwargs):
    with pytest.raises(ValueError):
        sr_cast_bf16(torch.ones(4, device="cuda"), seed=7, **kwargs)
