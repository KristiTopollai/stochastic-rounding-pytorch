import torch

from sr_states.reference.rounding import stochastic_round_bf16


def test_seed_and_offset_reproduce_exact_output():
    x = torch.linspace(-10, 10, 10_003, dtype=torch.float32)
    first = stochastic_round_bf16(x, seed=1234, offset=77)
    second = stochastic_round_bf16(x, seed=1234, offset=77)
    assert torch.equal(first, second)


def test_logical_offsets_make_partitioning_invariant():
    x = torch.linspace(-3, 7, 10_003, dtype=torch.float32)
    whole = stochastic_round_bf16(x, seed=99, offset=1000)
    split = 4096
    partitioned = torch.cat(
        [
            stochastic_round_bf16(x[:split], seed=99, offset=1000),
            stochastic_round_bf16(x[split:], seed=99, offset=1000 + split),
        ]
    )
    assert torch.equal(whole, partitioned)


def test_different_seed_changes_non_exact_choices():
    x = torch.full((4096,), 1.0 + 2.0**-9, dtype=torch.float32)
    assert not torch.equal(stochastic_round_bf16(x, seed=1), stochastic_round_bf16(x, seed=2))


def test_counter_periodicity_for_large_python_integers():
    x = torch.linspace(-1, 1, 1003)
    expected = stochastic_round_bf16(x, seed=7, offset=31)
    for periods in (-(2**60), -1, 1, 2**60):
        actual = stochastic_round_bf16(x, seed=7 + periods * 2**32, offset=31 + periods * 2**32)
        assert torch.equal(actual, expected)
