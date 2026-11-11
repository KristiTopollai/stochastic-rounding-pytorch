"""Execution checks for the H200 validation run; compilation alone is separate."""

import copy
import importlib.util
import os
from pathlib import Path

import pytest
import torch
from test_fp8_storage import check_small_gradient_training
from test_minifloat import DTYPES, MODES, assert_bits_or_nan, boundary_inputs

from sr_states import AdamWReferenceSR, SGDMReferenceSR, StateFormat, cast
from sr_states.compiled import compile_cast
from sr_states.triton import AdamWTriton, SGDMTriton
from sr_states.triton import cast as triton_cast

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]


def require_format(dtype):
    pytest.importorskip("triton")
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2) and torch.cuda.get_device_capability() < (
        8,
        9,
    ):
        pytest.skip("requires sm_89 FP8 conversions")


def test_bf16_second_moment_preserves_small_gradient_training_on_gpu():
    require_format(torch.float8_e4m3fn)
    check_small_gradient_training(AdamWTriton, "cuda")


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mode", MODES)
def test_triton_cast_matches_reference_bitwise(dtype, mode):
    require_format(dtype)
    generator = torch.Generator().manual_seed(541)
    random_bits = torch.randint(-(2**31), 2**31, (10007,), generator=generator, dtype=torch.int32)
    x = torch.cat((boundary_inputs(dtype), random_bits.view(torch.float32))).cuda()
    expected = cast(x, dtype=dtype, rounding=mode, seed=2**32 - 7, offset=2**32 - 93)
    for block in (128, 256, 1024):
        actual = triton_cast(
            x, dtype=dtype, rounding=mode, seed=2**32 - 7, offset=2**32 - 93, block_size=block
        )
        assert_bits_or_nan(actual, expected)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mode", MODES)
def test_inductor_cast_matches_cpu_reference_bitwise(dtype, mode):
    require_format(dtype)
    torch._dynamo.reset()
    try:
        generator = torch.Generator().manual_seed(541)
        random_bits = torch.randint(
            -(2**31), 2**31, (10007,), generator=generator, dtype=torch.int32
        )
        x = torch.cat((boundary_inputs(dtype), random_bits.view(torch.float32)))
        options = {"dtype": dtype, "rounding": mode, "seed": 2**32 - 7, "offset": 2**32 - 93}
        expected = cast(x, **options).cuda()
        compiled = compile_cast(fullgraph=True, dynamic=True, options={"triton.cudagraphs": False})
        assert_bits_or_nan(compiled(x.cuda(), **options), expected)
    finally:
        torch._dynamo.reset()


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_rn_matches_native_saturating_conversion(dtype):
    require_format(dtype)
    import triton

    from sr_states.triton.cast import _cast_kernel
    from sr_states.triton.rounding import nearest

    x = boundary_inputs(dtype).cuda()
    output = torch.empty_like(x, dtype=dtype)
    _cast_kernel[(triton.cdiv(x.numel(), 256),)](
        x, output, x.numel(), 0, 0, ROUND=nearest, BLOCK_SIZE=256, num_warps=4
    )
    assert_bits_or_nan(triton_cast(x, dtype=dtype, rounding="nearest"), output)


@pytest.mark.parametrize("kind", ["adamw", "sgdm"])
@pytest.mark.parametrize("mode", MODES)
def test_mixed_states_match_reference_and_resume(kind, mode):
    require_format(torch.float8_e4m3fn)
    p = torch.nn.Parameter(torch.linspace(-1, 1, 1027, device="cuda"))
    q = torch.nn.Parameter(p.detach().clone())
    formats = (
        {
            "exp_avg": StateFormat(torch.float8_e4m3fn, mode),
            "exp_avg_sq": StateFormat(torch.float8_e5m2, mode),
        }
        if kind == "adamw"
        else StateFormat(torch.float8_e4m3fn, mode)
    )
    kwargs = (
        {"betas": (0.5, 0.5), "eps": 0.125, "weight_decay": 0.0}
        if kind == "adamw"
        else {"momentum": 0.5}
    )
    reference_cls, fused_cls = (
        (AdamWReferenceSR, AdamWTriton) if kind == "adamw" else (SGDMReferenceSR, SGDMTriton)
    )
    reference = reference_cls([p], lr=2**-10, state_format=formats, seed=27, **kwargs)
    fused = fused_cls([q], lr=2**-10, state_format=formats, seed=27, **kwargs)
    r = None
    resumed = None
    for step in range(8):
        # Dyadic gradients/betas make moment arithmetic exact, isolating storage
        # parity from legitimate FMA differences in the general recurrence.
        grad = ((torch.arange(1027, device="cuda") + step) % 15 - 7).float() * 2**-5
        p.grad = q.grad = grad
        reference.step()
        fused.step()
        torch.testing.assert_close(q, p, rtol=1e-5, atol=1e-6)
        for name in reference._state_names:
            assert_bits_or_nan(fused.state[q][name], reference.state[p][name])
        if resumed is not None:
            r.grad = grad
            resumed.step()
            assert torch.equal(q, r)
            for name in reference._state_names:
                assert_bits_or_nan(fused.state[q][name], resumed.state[r][name])
        if step == 3:
            r = torch.nn.Parameter(q.detach().clone())
            resumed = fused_cls([r], **({"lr": 0.1} if kind == "sgdm" else {}))
            resumed.load_state_dict(copy.deepcopy(fused.state_dict()))


def test_gpu_golden_v01_regression():
    path = os.environ.get("SR_STATES_GPU_GOLDEN")
    if not path:
        pytest.skip("set SR_STATES_GPU_GOLDEN to a same-environment baseline capture")
    golden = torch.load(path, weights_only=True)
    assert golden["commit"].startswith("393c527")
    assert golden["device"] == "cuda"
    assert golden["torch"] == str(torch.__version__)
    spec = importlib.util.spec_from_file_location(
        "record_gpu_golden", Path(__file__).parents[1] / "scripts/record_optimizer_golden.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    actual = module.collect("cuda")
    assert actual.keys() == golden["cases"].keys()
    for key, expected in golden["cases"].items():
        for x, y in zip(actual[key]["parameters"], expected["parameters"]):
            assert torch.equal(x, y), key
        for x, y in zip(actual[key]["states"], expected["states"]):
            for name in x:
                assert (
                    torch.equal(x[name], y[name])
                    if isinstance(x[name], torch.Tensor)
                    else x[name] == y[name]
                ), (key, name)
        assert actual[key]["offsets"] == expected["offsets"]


@pytest.mark.parametrize("cls", [AdamWTriton, SGDMTriton])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, *DTYPES])
@pytest.mark.parametrize("mode", MODES)
def test_triton_shorthand_single_and_dict_are_identical(cls, dtype, mode):
    require_format(dtype)
    fmt = StateFormat(dtype, mode)
    params = [torch.nn.Parameter(torch.linspace(-1, 1, 257, device="cuda")) for _ in range(3)]
    args = [
        {"state_dtype": dtype, "rounding": mode},
        {"state_format": fmt},
        {"state_format": dict.fromkeys(cls._state_names, fmt)},
    ]
    opts = [cls([p], lr=0.001, seed=17, **arg) for p, arg in zip(params, args)]
    for step in range(3):
        for p, opt in zip(params, opts):
            p.grad = torch.sin(torch.arange(257, device="cuda") + step)
            opt.step()
        for p, opt in zip(params[1:], opts[1:]):
            assert torch.equal(p, params[0])
            for name in cls._state_names:
                assert_bits_or_nan(opt.state[p][name], opts[0].state[params[0]][name])
