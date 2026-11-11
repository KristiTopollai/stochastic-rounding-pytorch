"""Numerical tests use enumerated target encodings as an independent oracle."""

from fractions import Fraction

import pytest
import torch

from sr_states import StateFormat, cast
from sr_states.compiled import compile_cast
from sr_states.reference.rounding import minifloat_neighbors

DTYPES = [torch.float8_e4m3fn, torch.float8_e5m2, torch.float16]
MODES = ["nearest", "stochastic"]


def all_values(dtype):
    if dtype == torch.float16:
        return torch.arange(65536, dtype=torch.int32).to(torch.int16).view(dtype)
    return torch.arange(256, dtype=torch.int32).to(torch.uint8).view(dtype)


def positive_grid(dtype):
    values = all_values(dtype).float()
    return values[torch.isfinite(values) & ~torch.signbit(values)].sort().values


def boundary_inputs(dtype):
    grid = positive_grid(dtype)
    mid = (grid[:-1] + grid[1:]) * 0.5
    up = torch.nextafter(mid, torch.full_like(mid, float("inf")))
    down = torch.nextafter(mid, torch.zeros_like(mid))
    positive = torch.cat(
        (
            grid,
            mid,
            up,
            down,
            torch.tensor(
                [
                    float(torch.finfo(dtype).max),
                    float(torch.finfo(dtype).max) + 16,
                    2.0**-149,
                    2.0**-126,
                    1e-30,
                    1e30,
                    float("inf"),
                    float("nan"),
                ]
            ),
        )
    )
    return torch.cat((positive, -positive))


def oracle_neighbors(x, dtype):
    grid = positive_grid(dtype)
    a = x.abs()
    saturate = dtype != torch.float16
    if saturate:
        a = a.clamp_max(grid[-1].item())
    index = torch.searchsorted(grid, a)
    upper = grid[index.clamp_max(grid.numel() - 1)]
    exact = upper == a
    lower = torch.where(exact, upper, grid[(index - 1).clamp_min(0).clamp_max(grid.numel() - 1)])
    if not saturate:
        upper = torch.where(a > grid[-1], float("inf"), upper)
        lower = torch.where(torch.isinf(a), float("inf"), lower)
    lo = torch.where(torch.signbit(x), -upper, lower)
    hi = torch.where(torch.signbit(x), -lower, upper)
    lo = torch.copysign(lo.abs(), x)
    hi = torch.copysign(hi.abs(), x)
    lo = torch.where(torch.isnan(x), x, lo)
    hi = torch.where(torch.isnan(x), x, hi)
    return lo, hi


def assert_bits_or_nan(actual, expected):
    a, e = actual.float(), expected.float()
    assert torch.equal(torch.isnan(a), torch.isnan(e))
    keep = ~torch.isnan(a)
    # FP8 advanced indexing is unavailable in older PyTorch: compare FP32 bits.
    assert torch.equal(a[keep].view(torch.int32), e[keep].view(torch.int32))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mode", MODES)
def test_every_finite_encoding_is_fixed(dtype, mode):
    values = all_values(dtype).float()
    values = values[torch.isfinite(values)]
    result = cast(values, dtype=dtype, rounding=mode, seed=913, offset=2**32 - 17)
    assert_bits_or_nan(result, values.to(dtype))


@pytest.mark.parametrize("dtype", DTYPES)
def test_neighbors_agree_with_enumerated_grid(dtype):
    x = boundary_inputs(dtype)
    lower, upper = minifloat_neighbors(x, dtype=dtype)
    expected_lower, expected_upper = oracle_neighbors(x, dtype)
    assert_bits_or_nan(lower, expected_lower.to(dtype))
    assert_bits_or_nan(upper, expected_upper.to(dtype))


@pytest.mark.parametrize("dtype", DTYPES)
def test_sr_brackets_and_rn_ties_boundary_sweep(dtype):
    x = boundary_inputs(dtype)
    generator = torch.Generator().manual_seed(71)
    random_bits = torch.randint(-(2**31), 2**31, (10003,), generator=generator, dtype=torch.int32)
    x = torch.cat((x, random_bits.view(torch.float32)))
    lower, upper = oracle_neighbors(x, dtype)
    result = cast(x, dtype=dtype, seed=2**32 + 9).float()
    finite = ~torch.isnan(x)
    assert torch.all((result[finite] == lower[finite]) | (result[finite] == upper[finite]))
    assert torch.equal(torch.signbit(result[finite]), torch.signbit(x[finite]))
    assert torch.equal(torch.isnan(result), torch.isnan(x))
    expected = (
        x.to(dtype)
        if dtype == torch.float16
        else x.clamp(-torch.finfo(dtype).max, torch.finfo(dtype).max).to(dtype)
    )
    assert_bits_or_nan(cast(x, dtype=dtype, rounding="nearest"), expected)
    # Independent tie-to-even check against enumerated positive encodings.
    grid = positive_grid(dtype)
    mid = (grid[:-1] + grid[1:]) * 0.5
    expected_mid = torch.where(torch.arange(mid.numel()) % 2 == 0, grid[:-1], grid[1:])
    assert_bits_or_nan(cast(mid, dtype=dtype, rounding="nearest"), expected_mid.to(dtype))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mode", MODES)
def test_exceptional_values_and_signed_zero(dtype, mode):
    maximum = torch.finfo(dtype).max
    tiny = positive_grid(dtype)[1].item() * 2**-25
    x = torch.tensor(
        [
            0.0,
            -0.0,
            tiny,
            -tiny,
            maximum * 2,
            -maximum * 2,
            float("inf"),
            -float("inf"),
            float("nan"),
        ]
    )
    result = cast(x, dtype=dtype, rounding=mode).float()
    assert torch.equal(result[:4].view(torch.int32), x[:4].mul(0).view(torch.int32))
    expected = float("inf") if dtype == torch.float16 else maximum
    assert torch.equal(result[4:8], torch.tensor([expected, -expected, expected, -expected]))
    assert torch.isnan(result[-1])


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("region", ["normal", "subnormal", "below_subnormal"])
@pytest.mark.statistical
def test_round_up_probability(dtype, region):
    grid = positive_grid(dtype)
    step = grid[1].item()
    if region == "normal":
        lo, hi = 1.0, grid[torch.searchsorted(grid, torch.tensor(1.0)) + 1].item()
    elif region == "subnormal":
        lo, hi = step * 2, step * 3
    else:
        lo, hi = 0.0, step
    value = torch.tensor(lo + (hi - lo) * 0.3).item()
    probability = Fraction.from_float(value - lo) / Fraction.from_float(hi - lo)
    r = int(probability * 2**24)
    p = r / 2**24
    count = 200003
    result = cast(torch.full((count,), value), dtype=dtype, seed=93, offset=2**32 - 41).float()
    frequency = (result == hi).double().mean().item()
    assert abs(frequency - p) < 6 * (p * (1 - p) / count) ** 0.5
    assert torch.all((result == lo) | (result == hi))
    negative = cast(torch.full((count,), -value), dtype=dtype, seed=93, offset=2**32 - 41).float()
    assert torch.equal(negative, -result)


@pytest.mark.parametrize("dtype", DTYPES)
def test_probability_resolution_and_strict_threshold(dtype, monkeypatch):
    import sr_states._minifloat as module

    step = positive_grid(dtype)[1].item()
    # Counter hashes encode k=0,1,2,3. R=1,2,3 must round up exactly k<R.
    monkeypatch.setattr(
        module,
        "_u32_hash",
        lambda indices, seed: torch.arange(4, device=indices.device, dtype=torch.int64) << 8,
    )
    for r in (0, 1, 2, 3):
        x = torch.full((4,), step * (r + 0.5) * 2**-24)
        result = cast(x, dtype=dtype).float()
        assert torch.equal(result == step, torch.arange(4) < r)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mode", MODES)
def test_compile_cast_and_stream_partition(dtype, mode):
    torch._dynamo.reset()
    x = boundary_inputs(dtype)[::17].reshape(-1)
    options = {"dtype": dtype, "rounding": mode, "seed": 2**40 + 113, "offset": 2**33 - 19}
    eager = cast(x, **options)
    compiled = compile_cast(backend="eager", fullgraph=True)
    assert_bits_or_nan(compiled(x, **options), eager)
    assert_bits_or_nan(cast(x, **options), eager)
    split = x.numel() // 2
    actual = cast(x[split:], **{**options, "offset": options["offset"] + split})
    assert_bits_or_nan(actual, eager.float()[split:].to(dtype))
    assert StateFormat(dtype).to_dict() == {"dtype": dtype, "rounding": "stochastic"}


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mode", MODES)
def test_shapes_layouts_and_no_input_mutation(dtype, mode):
    for x in (torch.empty(0, 3), torch.tensor(-0.0), torch.linspace(-1, 1, 35).reshape(5, 7).T):
        original = x.clone()
        actual = cast(x, dtype=dtype, rounding=mode, seed=7)
        assert actual.shape == x.shape
        assert actual.device == x.device
        assert torch.equal(
            x.contiguous().reshape(-1).view(torch.uint8),
            original.contiguous().reshape(-1).view(torch.uint8),
        )
        assert_bits_or_nan(actual, cast(x.contiguous(), dtype=dtype, rounding=mode, seed=7))


@pytest.mark.parametrize("dtype", DTYPES)
def test_sr_matches_exact_rational_probabilities(dtype):
    from sr_states._hash import _u32_hash

    generator = torch.Generator().manual_seed(904)
    raw = torch.randint(0, 2**31 - 1, (2003,), dtype=torch.int32, generator=generator)
    x = raw.view(torch.float32)
    x = x[torch.isfinite(x) & (x <= torch.finfo(dtype).max)]
    lower, upper = oracle_neighbors(x, dtype)
    threshold = []
    for value, lo, hi in zip(x.tolist(), lower.tolist(), upper.tolist()):
        threshold.append(
            0
            if lo == hi
            else int(
                (Fraction.from_float(value) - Fraction.from_float(lo))
                / (Fraction.from_float(hi) - Fraction.from_float(lo))
                * 2**24
            )
        )
    seed, offset = 71, 2**32 - 91
    k = _u32_hash(torch.arange(x.numel(), dtype=torch.int64) + offset, seed) >> 8
    expected = torch.where(k < torch.tensor(threshold), upper, lower)
    for sign in (1.0, -1.0):
        actual = cast(x * sign, dtype=dtype, seed=seed, offset=offset)
        assert_bits_or_nan(actual, (expected * sign).to(dtype))
