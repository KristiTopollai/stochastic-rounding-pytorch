import copy
import importlib
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

import sr_states.rounding as rounding_module
from sr_states import AdamWReferenceSR, SGDMReferenceSR, cast, register_rounding
from sr_states.compiled import AdamWCompiledSR, SGDMCompiledSR, compile_cast
from sr_states.rounding import validate_rounding
from sr_states.triton import AdamWTriton, SGDMTriton


def toward_zero(x, *, dtype, seed, offset):
    bits = x.contiguous().view(torch.int32) & -65536
    return bits.view(torch.float32).to(dtype)


def nearest(x, *, dtype, seed, offset):
    return x.to(dtype)


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch):
    monkeypatch.setattr(rounding_module, "_ROUTINES", rounding_module._ROUTINES.copy())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mode", ["nearest", "stochastic"])
def test_builtin_cast_and_compiled_interface(dtype, mode):
    x = torch.tensor([1.003, -1.006, 0.0, -0.0, float("inf"), float("nan")])
    options = {"dtype": dtype, "rounding": mode, "seed": 44, "offset": 97}
    actual = cast(x, **options)
    compiled = compile_cast(backend="eager", fullgraph=True)
    assert torch.equal(actual.view(torch.uint8), compiled(x, **options).view(torch.uint8))
    if dtype == torch.float32 or mode == "nearest":
        assert torch.equal(actual.view(torch.uint8), x.to(dtype).view(torch.uint8))


@pytest.mark.parametrize(
    "cls", [SGDMReferenceSR, AdamWReferenceSR, SGDMCompiledSR, AdamWCompiledSR]
)
def test_custom_rounding_changes_state_only_and_resumes(cls):
    register_rounding("toward_zero", torch.bfloat16, reference=toward_zero)
    options = {"compile_kwargs": {"backend": "eager"}} if "Compiled" in cls.__name__ else {}
    p = torch.nn.Parameter(torch.ones(17))
    q = torch.nn.Parameter(p.detach().clone())
    custom = cls([p], lr=0.01, rounding="toward_zero", **options)
    baseline = cls([q], lr=0.01, state_dtype=torch.float32, **options)
    p.grad = q.grad = torch.linspace(-1.987, 1.987, 17)
    custom.step()
    baseline.step()
    assert torch.equal(p, q)  # This step still uses unrounded FP32 proposals.
    for name in custom._state_names:
        expected = toward_zero(baseline.state[q][name], dtype=torch.bfloat16, seed=0, offset=0)
        assert torch.equal(custom.state[p][name], expected)

    restored_p = torch.nn.Parameter(p.detach().clone())
    restored = cls([restored_p], lr=0.7, **options)
    checkpoint = copy.deepcopy(custom.state_dict())
    assert checkpoint["param_groups"][0]["rounding"] == "toward_zero"
    restored.load_state_dict(checkpoint)
    for step in range(3):
        p.grad = restored_p.grad = torch.sin(torch.arange(17) + step)
        custom.step()
        restored.step()
        assert torch.equal(p, restored_p)
        for name in custom._state_names:
            assert torch.equal(custom.state[p][name], restored.state[restored_p][name])


@pytest.mark.parametrize("cls", [SGDMReferenceSR, AdamWReferenceSR])
def test_register_new_storage_dtype_without_optimizer_changes(cls):
    register_rounding("nearest", torch.float16, reference=nearest)
    p = torch.nn.Parameter(torch.ones(17))
    optimizer = cls([p], lr=0.01, state_dtype=torch.float16, rounding="nearest")
    p.grad = torch.linspace(-0.9, 0.9, 17)
    optimizer.step()
    assert all(optimizer.state[p][name].dtype == torch.float16 for name in optimizer._state_names)


@pytest.mark.parametrize("cls", [SGDMTriton, AdamWTriton])
def test_triton_rejects_reference_only_routine(cls):
    register_rounding("toward_zero", torch.bfloat16, reference=toward_zero)
    with pytest.raises(ValueError, match="no Triton implementation"):
        cls([torch.nn.Parameter(torch.ones(1))], lr=0.1, rounding="toward_zero")


@pytest.mark.parametrize("cls", [SGDMReferenceSR, AdamWReferenceSR, SGDMTriton, AdamWTriton])
def test_group_overrides_and_checkpoint_require_registration(cls):
    p = torch.nn.Parameter(torch.ones(1))
    with pytest.raises(ValueError, match="rounding"):
        cls([{"params": [p], "rounding": "missing"}], lr=0.1)
    optimizer = cls([p], lr=0.1)
    with pytest.raises(ValueError, match="rounding"):
        optimizer.add_param_group(
            {"params": [torch.nn.Parameter(torch.ones(1))], "rounding": "missing"}
        )
    assert len(optimizer.param_groups) == 1
    checkpoint = copy.deepcopy(optimizer.state_dict())
    checkpoint["param_groups"][0]["rounding"] = "missing"
    with pytest.raises(ValueError, match="rounding"):
        optimizer.load_state_dict(checkpoint)
    assert optimizer.param_groups[0]["rounding"] == "stochastic"


def test_registration_validation_and_no_fp8_builtins():
    with pytest.raises(ValueError, match="already registered"):
        register_rounding("nearest", torch.bfloat16, reference=nearest)
    with pytest.raises(ValueError, match="nonempty"):
        register_rounding("", torch.bfloat16, reference=nearest)
    with pytest.raises(ValueError, match="floating-point"):
        register_rounding("rule", torch.int32, reference=nearest)
    with pytest.raises(TypeError, match="callable"):
        register_rounding("rule", torch.bfloat16, reference=None)
    for dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        with pytest.raises(ValueError, match="registered"):
            cast(torch.ones(1), dtype=dtype)
    with pytest.raises(ValueError, match="backend"):
        validate_rounding(torch.bfloat16, "nearest", backend="unknown")


def test_bad_custom_cast_contract_fails_explicitly():
    register_rounding("bad_dtype", torch.bfloat16, reference=lambda x, **kw: x)
    register_rounding("bad_shape", torch.bfloat16, reference=lambda x, **kw: x[:1].bfloat16())
    register_rounding("not_tensor", torch.bfloat16, reference=lambda x, **kw: None)
    for name in ("bad_dtype", "bad_shape"):
        with pytest.raises(ValueError, match="requested dtype"):
            cast(torch.ones(2), dtype=torch.bfloat16, rounding=name)
    with pytest.raises(TypeError, match="return a tensor"):
        cast(torch.ones(2), dtype=torch.bfloat16, rounding="not_tensor")


@pytest.mark.parametrize("kind,cls", [("sgd", SGDMTriton), ("adamw", AdamWTriton)])
def test_triton_host_dispatch_passes_registered_callback(monkeypatch, kind, cls):
    # Exercise the real host wrapper/optimizer on CPU; only the GPU launch is mocked.
    # The Linux compilation test separately verifies real JIT callback specialization.
    def jit_callback(value, counter, dtype):
        raise AssertionError("the mocked launch must not execute the callback")

    register_rounding("custom", torch.bfloat16, reference=toward_zero, triton=jit_callback)
    module = importlib.import_module(f"sr_states.triton.{kind}")
    launches = []

    class Kernel:
        def __getitem__(self, grid):
            return lambda *args, **kwargs: launches.append((args, kwargs))

    kernel_name = "_fused_sgdm_kernel" if kind == "sgd" else "_fused_adamw_kernel"
    monkeypatch.setattr(module, kernel_name, Kernel(), raising=False)
    monkeypatch.setattr(module, "require_triton", lambda: None)
    monkeypatch.setattr(module, "validate_tensors", lambda *args: None)
    monkeypatch.setattr(module, "triton", SimpleNamespace(cdiv=lambda n, b: (n + b - 1) // b))
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    p = torch.nn.Parameter(torch.ones(17))
    optimizer = cls([p], lr=0.01, rounding="custom", seed=19)
    p.grad = torch.ones_like(p)
    optimizer.step()
    optimizer.step()
    assert len(launches) == 2
    assert all(kwargs["ROUND"] is jit_callback for _, kwargs in launches)
    assert optimizer.param_groups[0]["sr_offset"] == 2 * len(optimizer._state_names) * p.numel()
