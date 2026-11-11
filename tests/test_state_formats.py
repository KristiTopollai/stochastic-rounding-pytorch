"""Storage API, compatibility, and a subclass that supplies only FP32 math."""

import copy
import importlib.util
import io
import platform
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
import torch

from sr_states import (
    AdamWReferenceRN,
    AdamWReferenceSR,
    SGDMReferenceRN,
    SGDMReferenceSR,
    StateFormat,
    StateOptimizer,
)
from sr_states.compiled import AdamWCompiledSR, SGDMCompiledSR
from sr_states.triton import AdamWTriton, SGDMTriton

CLASSES = [
    AdamWReferenceSR,
    SGDMReferenceSR,
    AdamWCompiledSR,
    SGDMCompiledSR,
    AdamWTriton,
    SGDMTriton,
]


def options(cls):
    return {"compile_kwargs": {"backend": "eager"}} if "Compiled" in cls.__name__ else {}


def bits(x):
    return x.contiguous().view(torch.uint8)


def test_golden_v01_outputs(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "record_golden", Path(__file__).parents[1] / "scripts/record_optimizer_golden.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    golden = torch.load(Path(__file__).parent / "golden/optimizer_v01.pt", weights_only=True)
    assert golden["commit"].startswith("393c527")
    # Arithmetic and transcendental kernels can differ across torch versions
    # and CPU architectures. Re-run the frozen baseline in THIS environment;
    # never weaken the byte comparison to a numerical tolerance.
    with zipfile.ZipFile(Path(__file__).parent / "golden/baseline_393c527.zip") as archive:
        archive.extractall(tmp_path)
    script_dir = tmp_path / "scripts"
    script_dir.mkdir()
    script = script_dir / "record_optimizer_golden.py"
    shutil.copyfile(Path(module.__file__), script)
    output = tmp_path / "baseline.pt"
    process = subprocess.run(
        [sys.executable, str(script), "--output", str(output), "--source-commit", golden["commit"]],
        check=False,
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stderr
    same_environment = torch.load(output, weights_only=True)
    actual = module.collect()
    if (
        str(torch.__version__) != golden["torch"]
        or platform.system() != "Darwin"
        or platform.machine() != "arm64"
    ):
        golden = same_environment
    else:
        # In the capture environment retain the original committed fixture too.
        for key in golden["cases"]:
            for x, y in zip(
                same_environment["cases"][key]["parameters"], golden["cases"][key]["parameters"]
            ):
                assert torch.equal(x, y), key

    assert actual.keys() == golden["cases"].keys()
    for key, expected in golden["cases"].items():
        for x, y in zip(actual[key]["parameters"], expected["parameters"]):
            assert torch.equal(x, y), key
        for x, y in zip(actual[key]["states"], expected["states"]):
            assert x.keys() == y.keys()
            for name in x:
                assert (
                    torch.equal(x[name], y[name])
                    if isinstance(x[name], torch.Tensor)
                    else x[name] == y[name]
                ), (key, name)
        assert actual[key]["offsets"] == expected["offsets"]


@pytest.mark.parametrize("cls", CLASSES)
def test_format_validation_and_group_overrides(cls):
    p, q = [torch.nn.Parameter(torch.ones(3)) for _ in range(2)]
    fmt = StateFormat(torch.float32, "nearest")
    for extra in ({"state_dtype": torch.float32}, {"rounding": "nearest"}):
        with pytest.raises(ValueError, match="combined"):
            cls([p], lr=0.1, state_format=fmt, **extra, **options(cls))
    with pytest.raises(TypeError, match="StateFormat"):
        cls(
            [{"params": [p], "state_format": (fmt,) * len(cls._state_names)}],
            lr=0.1,
            **options(cls),
        )
    with pytest.raises(ValueError, match="keys"):
        cls([p], lr=0.1, state_format={"bad": fmt}, **options(cls))
    opt = cls([{"params": [p], "state_format": fmt}], lr=0.1, **options(cls))
    assert all(f["dtype"] == torch.float32 for f in opt.param_groups[0]["state_format"].values())
    opt.add_param_group({"params": [q], "rounding": "nearest"})
    assert all(
        f == StateFormat(torch.bfloat16, "nearest").to_dict()
        for f in opt.param_groups[1]["state_format"].values()
    )
    with pytest.raises(ValueError, match="registered"):
        opt.add_param_group(
            {
                "params": [torch.nn.Parameter(torch.ones(1))],
                "state_format": StateFormat(torch.bfloat16, "missing"),
            }
        )
    assert len(opt.param_groups) == 2
    with pytest.raises(ValueError):
        opt.add_param_group({"params": [torch.nn.Parameter(torch.ones(1))], "lr": -1})
    assert len(opt.param_groups) == 2


@pytest.mark.parametrize(
    "cls", [AdamWReferenceSR, SGDMReferenceSR, AdamWCompiledSR, SGDMCompiledSR]
)
def test_shorthand_matches_single_and_dict_formats(cls):
    torch._dynamo.reset()
    parameters = [torch.nn.Parameter(torch.linspace(-1, 1, 17)) for _ in range(3)]
    fmt = StateFormat(torch.bfloat16)
    args = [
        {"state_dtype": fmt.dtype, "rounding": fmt.rounding},
        {"state_format": fmt},
        {"state_format": dict.fromkeys(cls._state_names, fmt)},
    ]
    opts = [cls([p], lr=0.01, seed=19, **a, **options(cls)) for p, a in zip(parameters, args)]
    for step in range(3):
        for opt, p in zip(opts, parameters):
            p.grad = torch.sin(torch.arange(17) + step)
            opt.step()
        for p, opt in zip(parameters[1:], opts[1:]):
            assert torch.equal(p, parameters[0])
            for name in cls._state_names:
                assert torch.equal(
                    bits(opt.state[p][name]), bits(opts[0].state[parameters[0]][name])
                )


@pytest.mark.parametrize("compiled", [False, True])
def test_mixed_formats_roundtrip_and_legacy_load(compiled):
    torch._dynamo.reset()
    cls = AdamWCompiledSR if compiled else AdamWReferenceSR
    p = torch.nn.Parameter(torch.ones(9, dtype=torch.bfloat16))
    opt = cls(
        [p],
        lr=0.01,
        state_format={
            "exp_avg": StateFormat(torch.bfloat16),
            "exp_avg_sq": StateFormat(torch.float32, "nearest"),
        },
        seed=7,
        **options(cls),
    )
    p.grad = torch.linspace(-1, 1, 9).bfloat16()
    opt.step()
    stream = io.BytesIO()
    torch.save(opt.state_dict(), stream)
    stream.seek(0)
    checkpoint = torch.load(stream, weights_only=True)
    q = torch.nn.Parameter(p.detach().clone())
    resumed = cls([q], lr=0.3, **options(cls))
    resumed.load_state_dict(checkpoint)
    for name in cls._state_names:
        assert torch.equal(bits(opt.state[p][name]), bits(resumed.state[q][name]))
    for step in range(3):
        p.grad = q.grad = torch.sin(torch.arange(9) + step).bfloat16()
        opt.step()
        resumed.step()
        assert torch.equal(p, q)
    old = copy.deepcopy(checkpoint)
    old["param_groups"][0].pop("state_format")
    old["param_groups"][0].update(state_dtype=torch.float32, rounding="nearest")
    resumed.load_state_dict(old)
    assert all(resumed.state[q][name].dtype == torch.float32 for name in cls._state_names)
    assert "state_format" not in old["param_groups"][0]


class ThreeState(StateOptimizer):
    _state_names = ("a", "b", "c")

    def __init__(self, params, **kwargs):
        super().__init__(params, {"lr": 0.01}, **kwargs)

    def _compute(self, p32, g32, states32, group, step):
        a, b, c = states32
        a = a * 0.8 + g32
        b = b * 0.5 + g32.square()
        c = c + g32 * 0.1
        return p32 - group["lr"] * (a + b + c), (a, b, c)


@pytest.mark.parametrize("mixed", [False, True])
def test_three_state_optimizer_needs_no_storage_code(mixed):
    formats = (
        {
            "a": StateFormat(torch.bfloat16),
            "b": StateFormat(torch.float32),
            "c": StateFormat(torch.bfloat16, "nearest"),
        }
        if mixed
        else StateFormat(torch.bfloat16)
    )
    p = torch.nn.Parameter(torch.ones(13))
    opt = ThreeState([p], state_format=formats, seed=71)
    p.grad = torch.linspace(-1, 1, 13)
    opt.step()
    q = torch.nn.Parameter(p.detach().clone())
    resumed = ThreeState([q])
    resumed.load_state_dict(copy.deepcopy(opt.state_dict()))
    for step in range(3):
        p.grad = q.grad = torch.sin(torch.arange(13) + step)
        opt.step()
        resumed.step()
        assert torch.equal(p, q)
        for name in opt._state_names:
            assert torch.equal(bits(opt.state[p][name]), bits(resumed.state[q][name]))
    assert opt._next_offset() == 4 * 3 * 13


def test_state_format_serialization():
    fmt = StateFormat(torch.bfloat16, "nearest")
    assert StateFormat.from_dict(fmt.to_dict()) == fmt
    fmt.validate("reference")
    fmt.validate("compiled")
    fmt.validate("triton")
    with pytest.raises(ValueError, match="backend"):
        fmt.validate("missing")
    for invalid in (
        {"dtype": torch.bfloat16},
        {"dtype": torch.bfloat16, "rounding": "nearest", "extra": 1},
    ):
        with pytest.raises(ValueError):
            StateFormat.from_dict(invalid)


@pytest.mark.parametrize(
    "rn,sr", [(AdamWReferenceRN, AdamWReferenceSR), (SGDMReferenceRN, SGDMReferenceSR)]
)
def test_rn_classes_accept_state_format_and_force_nearest(rn, sr):
    dtypes = (torch.float8_e4m3fn, torch.float8_e5m2)
    per_state = {name: StateFormat(dtype) for name, dtype in zip(rn._state_names, dtypes)}
    nearest = {name: StateFormat(dtype, "nearest") for name, dtype in zip(rn._state_names, dtypes)}
    p, q = (torch.nn.Parameter(torch.linspace(-1, 1, 17)) for _ in range(2))
    opt = rn([p], lr=0.01, state_format=per_state, seed=5)
    expected = sr([q], lr=0.01, state_format=nearest, seed=5)
    assert opt.param_groups[0]["state_format"] == expected.param_groups[0]["state_format"]
    for step in range(3):
        p.grad = q.grad = torch.sin(torch.arange(17) + step)
        opt.step()
        expected.step()
        assert torch.equal(p, q)
        for name in rn._state_names:
            assert torch.equal(bits(opt.state[p][name]), bits(expected.state[q][name]))

    # Group overrides and checkpoints select only each state's dtype.
    for override in ({"state_format": StateFormat(torch.bfloat16)}, {"rounding": "stochastic"}):
        opt.add_param_group({"params": [torch.nn.Parameter(torch.ones(2))], **override})
        assert all(
            f["rounding"] == "nearest" for f in opt.param_groups[-1]["state_format"].values()
        )
    checkpoint = copy.deepcopy(expected.state_dict())
    for fmt in checkpoint["param_groups"][0]["state_format"].values():
        fmt["rounding"] = "stochastic"
    resumed = rn([torch.nn.Parameter(q.detach().clone())], lr=0.01)
    resumed.load_state_dict(checkpoint)
    assert resumed.param_groups[0]["state_format"] == expected.param_groups[0]["state_format"]

    with pytest.raises(ValueError, match="combined"):
        rn([p], lr=0.01, state_format=per_state, rounding="nearest")
