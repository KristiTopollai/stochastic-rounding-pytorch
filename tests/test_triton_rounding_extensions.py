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
from sr_states.triton.adamw import _fused_adamw_kernel, _multi_adamw_kernel
from sr_states.triton.cast import _cast_kernel
from sr_states.triton.rounding import nearest as triton_nearest
from sr_states.triton.sgd import _fused_sgdm_kernel, _multi_sgdm_kernel


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
        "custom_nearest",
        torch.float16,
        reference=lambda x, **kw: x.to(kw["dtype"]),
        triton=triton_nearest,
    )


@pytest.mark.parametrize("kind", ["cast", "sgdm", "adamw", "multi_sgdm", "multi_adamw"])
@pytest.mark.parametrize(
    "dtype,mode",
    [
        (torch.float32, "nearest"),
        (torch.float32, "stochastic"),
        (torch.bfloat16, "nearest"),
        (torch.bfloat16, "stochastic"),
        (torch.bfloat16, "toward_zero"),
        (torch.float16, "custom_nearest"),
        *(
            (dtype, mode)
            for dtype in (torch.float16, torch.float8_e4m3fn, torch.float8_e5m2)
            for mode in ("nearest", "stochastic")
        ),
    ],
)
def test_rounding_callback_compiles_without_gpu(kind, dtype, mode, custom_rounding):
    kernels = {
        "cast": _cast_kernel,
        "sgdm": _fused_sgdm_kernel,
        "adamw": _fused_adamw_kernel,
        "multi_sgdm": _multi_sgdm_kernel,
        "multi_adamw": _multi_adamw_kernel,
    }
    kernel = kernels[kind]
    state_type = {
        torch.float32: "fp32",
        torch.bfloat16: "bf16",
        torch.float16: "fp16",
        torch.float8_e4m3fn: "fp8e4nv",
        torch.float8_e5m2: "fp8e5",
    }[dtype]
    states = {"output_ptr", "momentum_ptr", "exp_avg_ptr", "exp_avg_sq_ptr"}
    signature = {}
    constants = {"BLOCK_SIZE": 256}
    if kind.startswith("multi"):
        constants["CHUNK_TILES"] = 1
    for name in ("ROUND_M", "ROUND_V") if kind.endswith("adamw") else ("ROUND",):
        constants[name] = get_triton_rounding(dtype, mode)
    for name in kernel.arg_names:
        if name in constants:
            continue
        if name in {"meta_ptr", "pointers_ptr", "chunks_ptr"}:
            signature[name] = "*i32" if name == "chunks_ptr" else "*i64"
        elif name.endswith(("_ptr", "_witness")):
            role = name.replace("_witness", "_ptr")
            signature[name] = "*" + (state_type if role in states else "fp32")
        elif name in {"n_elements", "seed_lo", "seed_hi", "step_lo", "step_hi"}:
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


@pytest.mark.parametrize("mode", ["nearest", "stochastic"])
@pytest.mark.parametrize("chunk_tiles", [None, 1, 2])
def test_mixed_fp8_adamw_compiles_without_gpu(mode, chunk_tiles):
    kernel = _fused_adamw_kernel if chunk_tiles is None else _multi_adamw_kernel
    constants = {
        "ROUND_M": get_triton_rounding(torch.float8_e4m3fn, mode),
        "ROUND_V": get_triton_rounding(torch.float8_e5m2, mode),
        "BLOCK_SIZE": 256,
    }
    if chunk_tiles is not None:
        constants["CHUNK_TILES"] = chunk_tiles
    signature = {}
    for name in kernel.arg_names:
        if name in constants:
            continue
        if name in {"meta_ptr", "pointers_ptr", "chunks_ptr"}:
            signature[name] = "*i32" if name == "chunks_ptr" else "*i64"
        elif name.endswith(("_ptr", "_witness")):
            signature[name] = {"exp_avg_ptr": "*fp8e4nv", "exp_avg_sq_ptr": "*fp8e5"}.get(
                name.replace("_witness", "_ptr"), "*fp32"
            )
        elif name in {"n_elements", "seed_lo", "seed_hi", "step_lo", "step_hi"}:
            signature[name] = "i32"
        elif name in {"seed", "logical_offset"}:
            signature[name] = "u32"
        else:
            signature[name] = "fp32"
    result = triton.compile(
        ASTSource(kernel, signature, constants),
        target=GPUTarget("cuda", 90, 32),
        options={"num_warps": 4},
    )
    assert result.asm["ptx"]
    assert result.asm["cubin"]


@triton.jit(do_not_specialize=["seed_lo", "seed_hi", "step_lo", "step_hi"])
def _pointer_loop_spike(
    pointers, witness, sizes, seed_lo, seed_hi, step_lo, step_hi, BLOCK: tl.constexpr
):
    ptr = tl.load(pointers).to(witness.dtype)
    numel = tl.load(sizes)
    value = seed_lo.to(tl.uint32) ^ seed_hi.to(tl.uint32)
    value ^= step_lo.to(tl.uint32) ^ step_hi.to(tl.uint32)
    for start in range(0, numel, BLOCK):
        indices = start + tl.arange(0, BLOCK)
        tl.store(ptr + indices, value, indices < numel)


def test_pointer_loop_spike_compiles_without_gpu():
    kernel = _pointer_loop_spike
    assert all(p.do_not_specialize for p in kernel.params if p.name.startswith(("seed", "step")))
    signature = {
        "pointers": "*i64",
        "witness": "*fp32",
        "sizes": "*i64",
        **dict.fromkeys(("seed_lo", "seed_hi", "step_lo", "step_hi"), "i32"),
    }
    result = triton.compile(
        ASTSource(kernel, signature, {"BLOCK": 128}),
        target=GPUTarget("cuda", 90, 32),
        options={"num_warps": 4},
    )
    assert result.asm["ptx"] and result.asm["cubin"]
