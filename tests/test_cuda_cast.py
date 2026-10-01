import pytest

from benchmarks.cast_checks import prepare_cases, validate_cast
from sr_states.compiled import compile_sr_cast
from sr_states.triton import is_triton_available

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not is_triton_available(), reason="requires Triton and CUDA"),
]


def test_cuda_compiled_cast_matches_reference():
    compiled = compile_sr_cast(fullgraph=True, dynamic=True, options={"triton.cudagraphs": False})
    validate_cast(compiled, prepare_cases("cuda"))
