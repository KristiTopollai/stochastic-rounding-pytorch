"""Compile real Triton callback specializations on Linux, even without a GPU."""

import pytest
import torch

triton = pytest.importorskip("triton")
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

import sr_states.rounding as rounding_module
from sr_states import register_rounding
from sr_states.rounding import get_triton_rounding
from sr_states.triton.adamw import _fused_adamw_kernel
from sr_states.triton.cast import _cast_kernel
from sr_states.triton.rounding import nearest as triton_nearest
from sr_states.triton.sgd import _fused_sgdm_kernel


@triton.jit
def toward_zero(value, counter, dtype: tl.constexpr):
    bits = value.to(tl.uint32, bitcast=True) & 0xFFFF0000
    return bits.to(tl.float32, bitcast=True).to(dtype)


def reference_toward_zero(x, *, dtype, seed, offset):
    return (x.contiguous().view(torch.int32) & -65536).view(torch.float32).to(dtype)


@pytest.fixture
def custom_rounding(monkeypatch):
    monkeypatch.setattr(rounding_module, "_ROUTINES", rounding_module._ROUTINES.copy())
    register_rounding(
        "toward_zero", torch.bfloat16, reference=reference_toward_zero, triton=toward_zero
    )
    register_rounding(
        "nearest", torch.float16, reference=lambda x, **kw: x.to(kw["dtype"]), triton=triton_nearest
    )


@pytest.mark.parametrize("kind", ["cast", "sgdm", "adamw"])
@pytest.mark.parametrize(
    "dtype,mode",
    [
        (torch.float32, "nearest"),
        (torch.float32, "stochastic"),
        (torch.bfloat16, "nearest"),
        (torch.bfloat16, "stochastic"),
        (torch.bfloat16, "toward_zero"),
        (torch.float16, "nearest"),
    ],
)
def test_rounding_callback_compiles_without_gpu(kind, dtype, mode, custom_rounding):
    kernels = {"cast": _cast_kernel, "sgdm": _fused_sgdm_kernel, "adamw": _fused_adamw_kernel}
    kernel = kernels[kind]
    state_type = {torch.float32: "fp32", torch.bfloat16: "bf16", torch.float16: "fp16"}[dtype]
    states = {"output_ptr", "momentum_ptr", "exp_avg_ptr", "exp_avg_sq_ptr"}
    signature = {}
    constants = {"ROUND": get_triton_rounding(dtype, mode), "BLOCK_SIZE": 256}
    for name in kernel.arg_names:
        if name in constants:
            continue
        if name.endswith("_ptr"):
            signature[name] = "*" + (state_type if name in states else "fp32")
        elif name == "n_elements":
            signature[name] = "i32"
        elif name in {"seed", "logical_offset"}:
            signature[name] = "u32"
        else:
            signature[name] = "fp32"
    source = ASTSource(kernel, signature, constants)
    compiled = triton.compile(source, target=GPUTarget("cuda", 90, 32), options={"num_warps": 4})
    assert compiled.asm["ptx"]
    assert compiled.asm["cubin"]


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("kind", ["sgdm", "adamw"])
def test_custom_triton_rounding_matches_reference_on_gpu(kind, custom_rounding):
    from sr_states import AdamWReferenceSR, SGDMReferenceSR
    from sr_states.triton import AdamWTriton, SGDMTriton

    reference_cls = SGDMReferenceSR if kind == "sgdm" else AdamWReferenceSR
    triton_cls = SGDMTriton if kind == "sgdm" else AdamWTriton
    p = torch.nn.Parameter(torch.linspace(-1, 1, 1027, device="cuda"))
    q = torch.nn.Parameter(p.detach().clone())
    reference = reference_cls([p], lr=0.01, rounding="toward_zero")
    fused = triton_cls([q], lr=0.01, rounding="toward_zero")
    for step in range(4):
        p.grad = q.grad = torch.sin(torch.arange(1027, device="cuda") + step)
        reference.step()
        fused.step()
        torch.testing.assert_close(q, p, rtol=1e-4, atol=1e-5)
        for name in reference._state_names:
            torch.testing.assert_close(
                fused.state[q][name], reference.state[p][name], rtol=8e-3, atol=1e-6
            )
