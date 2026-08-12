import math

import pytest
import torch

from sr_states.reference.rounding import stochastic_round_bf16


@pytest.mark.statistical
def test_empirical_up_probability_and_mean():
    n = 200_000
    lower = 1.0
    ulp = 2.0**-7
    probability_up = 0.25
    x_value = lower + probability_up * ulp
    x = torch.full((n,), x_value, dtype=torch.float32)
    output = stochastic_round_bf16(x, seed=2026).float()

    empirical_probability = (output > lower).float().mean().item()
    standard_error = math.sqrt(probability_up * (1 - probability_up) / n)
    assert abs(empirical_probability - probability_up) < 6 * standard_error

    # The same Bernoulli confidence bound translated from probability to value.
    mean_error_bound = 6 * standard_error * ulp
    assert abs(output.mean().item() - x_value) < mean_error_bound


@pytest.mark.statistical
def test_distribution_stress_is_unbiased_in_aggregate():
    exponents = torch.arange(-20, 21, dtype=torch.float32)
    base = torch.pow(2.0, exponents)
    inputs = (base[:, None] * (1.0 + torch.linspace(0.01, 0.99, 1024)[None, :])).flatten()
    rounded = stochastic_round_bf16(inputs, seed=91).float()
    normalized_error = (rounded - inputs) / inputs.abs()
    # This is a broad bias detector, not a claim of independent normal errors.
    assert normalized_error.mean().abs().item() < 1e-4
