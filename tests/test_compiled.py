import pytest
import torch

from sr_states.compiled import compile_sgdm_step, compile_sr_cast
from sr_states.reference.rounding import stochastic_round_bf16
from sr_states.reference.sgd import sgdm_step


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile unavailable")
def test_compiled_sr_matches_eager_frontend():
    x = torch.linspace(-2, 2, 1024)
    compiled = compile_sr_cast(backend="eager", fullgraph=True)
    assert torch.equal(compiled(x, seed=7, offset=9), stochastic_round_bf16(x, seed=7, offset=9))


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile unavailable")
def test_compiled_sgdm_matches_eager_frontend():
    p = torch.linspace(-1, 1, 1024)
    g = torch.cos(p)
    u = torch.zeros_like(p, dtype=torch.bfloat16)
    compiled = compile_sgdm_step(backend="eager", fullgraph=True)
    kwargs = {"lr": 0.1, "momentum_factor": 0.9, "seed": 12, "offset": 4}
    expected = sgdm_step(p, g, u, **kwargs)
    actual = compiled(p, g, u, **kwargs)
    torch.testing.assert_close(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])
