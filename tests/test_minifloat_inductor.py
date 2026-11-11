"""Check actual CPU code generation in Linux CI, alongside eager tracing tests."""

import sys

import pytest
import torch
from test_minifloat import DTYPES, MODES, assert_bits_or_nan, boundary_inputs

from sr_states import cast
from sr_states.compiled import compile_cast


@pytest.mark.skipif(sys.platform != "linux", reason="Linux CI exercises CPU Inductor")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mode", MODES)
def test_inductor_minifloat_cast(dtype, mode):
    torch._dynamo.reset()
    x = boundary_inputs(dtype)[::17].contiguous()
    options = {"dtype": dtype, "rounding": mode, "seed": 2**32 + 11, "offset": 2**32 - 13}
    compiled = compile_cast(fullgraph=True)
    assert_bits_or_nan(compiled(x, **options), cast(x, **options))
