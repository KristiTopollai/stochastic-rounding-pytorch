"""FP8 mixed-state optimizer integration and pre-launch capability validation."""

import copy
import importlib
import io
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from test_state_formats import ThreeState, bits

from sr_states import AdamWReferenceSR, StateFormat
from sr_states.compiled import AdamWCompiledSR
from sr_states.triton import AdamWTriton, SGDMTriton
from sr_states.triton._common import validate_fp8_device

MIXED = {
    "exp_avg": StateFormat(torch.float8_e4m3fn),
    "exp_avg_sq": StateFormat(torch.float8_e5m2, "nearest"),
}


def check_small_gradient_training(cls, device):
    """Exercise the README's FP8 momentum/BF16 second-moment configuration."""
    torch._dynamo.reset()
    try:
        p = torch.nn.Parameter(torch.zeros(1024, device=device))
        kwargs = {"compile_kwargs": {"backend": "eager"}} if cls == AdamWCompiledSR else {}
        optimizer = cls(
            [p],
            lr=3e-4,
            state_format={
                "exp_avg": StateFormat(torch.float8_e4m3fn),
                "exp_avg_sq": StateFormat(torch.bfloat16),
            },
            seed=7,
            **kwargs,
        )
        for _ in range(10):
            p.grad = torch.full_like(p, 1e-3)
            optimizer.step()
            second_moment = optimizer.state[p]["exp_avg_sq"]
            assert second_moment.dtype == torch.bfloat16
            assert torch.all(second_moment.float() > 0)
            assert torch.all(torch.isfinite(p))
        # Catch denominator collapse, rather than requiring FP8 first moments
        # to follow the same trajectory as full-precision state storage.
        assert p.abs().max() < 0.05
    finally:
        torch._dynamo.reset()


@pytest.mark.parametrize("cls", [AdamWReferenceSR, AdamWCompiledSR])
def test_bf16_second_moment_preserves_small_gradient_training(cls):
    check_small_gradient_training(cls, "cpu")


@pytest.mark.parametrize("cls", [AdamWReferenceSR, AdamWCompiledSR])
def test_mixed_fp8_checkpoint_and_resume(cls):
    torch._dynamo.reset()
    kwargs = {"compile_kwargs": {"backend": "eager"}} if cls == AdamWCompiledSR else {}
    p = torch.nn.Parameter(torch.linspace(-1, 1, 31).bfloat16())
    opt = cls([p], lr=0.001, state_format=MIXED, seed=194, **kwargs)
    for step in range(3):
        p.grad = torch.sin(torch.arange(31) + step).bfloat16()
        opt.step()
    assert opt.state[p]["exp_avg"].nbytes + opt.state[p]["exp_avg_sq"].nbytes == 62
    stream = io.BytesIO()
    torch.save(opt.state_dict(), stream)
    stream.seek(0)
    checkpoint = torch.load(stream, weights_only=True)
    q = torch.nn.Parameter(p.detach().clone())
    resumed = cls([q], **kwargs)
    resumed.load_state_dict(checkpoint)
    for name, fmt in MIXED.items():
        assert resumed.state[q][name].dtype == fmt.dtype
        assert torch.equal(bits(opt.state[p][name]), bits(resumed.state[q][name]))
    for step in range(4):
        p.grad = q.grad = torch.sin(torch.arange(31) * 0.1 + step).bfloat16()
        opt.step()
        resumed.step()
        assert torch.equal(p, q)
        for name in MIXED:
            assert torch.equal(bits(opt.state[p][name]), bits(resumed.state[q][name]))


@pytest.mark.parametrize("mixed", [False, True])
def test_toy_optimizer_fp8_formats(mixed):
    formats = (
        {
            "a": StateFormat(torch.float8_e4m3fn),
            "b": StateFormat(torch.float8_e5m2),
            "c": StateFormat(torch.bfloat16),
        }
        if mixed
        else StateFormat(torch.float8_e4m3fn)
    )
    p = torch.nn.Parameter(torch.ones(17))
    opt = ThreeState([p], state_format=formats, seed=18)
    p.grad = torch.linspace(-1, 1, 17)
    opt.step()
    q = torch.nn.Parameter(p.detach().clone())
    resumed = ThreeState([q])
    resumed.load_state_dict(copy.deepcopy(opt.state_dict()))
    for _ in range(3):
        p.grad = q.grad = torch.ones(17) * 0.13
        opt.step()
        resumed.step()
        assert torch.equal(p, q)
        for name in opt._state_names:
            assert torch.equal(bits(opt.state[p][name]), bits(resumed.state[q][name]))
    assert opt._next_offset() == 4 * 3 * 17


@pytest.mark.parametrize("capability", [(8, 0), (8, 6), (8, 9), (9, 0), (10, 0)])
def test_fp8_capability_check(monkeypatch, capability):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: capability)
    if capability < (8, 9):
        with pytest.raises(ValueError, match="sm_89"):
            validate_fp8_device("cuda:0", torch.float8_e4m3fn, torch.float8_e5m2)
    else:
        validate_fp8_device("cuda:0", torch.float8_e4m3fn, torch.float8_e5m2)
    validate_fp8_device("cpu", torch.float16, torch.bfloat16, torch.float32)


@pytest.mark.parametrize("kind,cls", [("adamw", AdamWTriton), ("sgd", SGDMTriton)])
def test_each_state_dispatches_its_own_callback(monkeypatch, kind, cls):
    module = importlib.import_module(f"sr_states.triton.{kind}")
    seen = []
    callbacks = {}

    def resolve(dtype, rounding):
        callbacks.setdefault((dtype, rounding), object())
        return callbacks[dtype, rounding]

    class Kernel:
        def __getitem__(self, grid):
            return lambda *args, **kwargs: seen.append(kwargs)

    monkeypatch.setattr(
        module,
        "_fused_adamw_kernel" if kind == "adamw" else "_fused_sgdm_kernel",
        Kernel(),
        raising=False,
    )
    monkeypatch.setattr(module, "require_triton", lambda: None)
    monkeypatch.setattr(module, "validate_tensors", lambda *args: None)
    monkeypatch.setattr(module, "validate_fp8_device", lambda *args: None)
    monkeypatch.setattr(module, "get_triton_rounding", resolve)
    monkeypatch.setattr(module, "triton", SimpleNamespace(cdiv=lambda n, b: (n + b - 1) // b))
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    fmt = MIXED if kind == "adamw" else StateFormat(torch.float8_e4m3fn)
    p = torch.nn.Parameter(torch.ones(17))
    opt = cls([p], lr=0.01, state_format=fmt, seed=3)
    p.grad = torch.ones_like(p)
    opt.step()
    opt.step()
    assert opt._next_offset() == 2 * len(opt._state_names) * p.numel()
    if kind == "adamw":
        assert all(
            item["ROUND_M"] is callbacks[torch.float8_e4m3fn, "stochastic"]
            and item["ROUND_V"] is callbacks[torch.float8_e5m2, "nearest"]
            for item in seen
        )
    else:
        assert all(item["ROUND"] is callbacks[torch.float8_e4m3fn, "stochastic"] for item in seen)
