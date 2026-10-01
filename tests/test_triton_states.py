import copy

import pytest
import torch

from benchmarks.bench_states import POLICIES, build_case, prepare_inputs, update_options
from sr_states.compiled import compile_adamw_step_, compile_sgdm_step_
from sr_states.reference.adamw import adamw_step_
from sr_states.reference.rounding import stochastic_round_bf16
from sr_states.reference.sgd import sgdm_step_
from sr_states.triton import (
    AdamWTriton,
    SGDMTriton,
    is_triton_available,
    triton_adamw_step_,
    triton_sgdm_step_,
)

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not is_triton_available(), reason="requires Triton and CUDA"),
]


def step_function(kind):
    return triton_adamw_step_ if kind == "adamw" else triton_sgdm_step_


@pytest.mark.parametrize("kind", ["sgdm", "adamw"])
@pytest.mark.parametrize("parameter_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("policy", list(POLICIES))
@pytest.mark.parametrize("n", [0, 1, 1003])
def test_fused_update_matches_reference(kind, parameter_dtype, policy, n):
    inputs = prepare_inputs(kind, n, count=1, dtype=parameter_dtype, device="cuda")
    case = build_case(inputs, optimizer=kind, backend="triton", policy=policy, step=1)
    case.validate()


@pytest.mark.parametrize("kind", ["sgdm", "adamw"])
@pytest.mark.parametrize("parameter_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("rounding", ["stochastic", "nearest"])
def test_multistep_rounds_only_state_writeback(kind, parameter_dtype, rounding):
    tensors = list(prepare_inputs(kind, 4099, count=1, dtype=parameter_dtype, device="cuda")[0])
    tensors[2:] = [value.to(torch.bfloat16) for value in tensors[2:]]
    function = step_function(kind)
    reference = adamw_step_ if kind == "adamw" else sgdm_step_
    for step in range(1, 13):
        tensors[1].copy_(torch.sin(torch.arange(4099, device="cuda") * 0.001 + step))
        raw = [tensor.clone() if i < 2 else tensor.float() for i, tensor in enumerate(tensors)]
        expected = [tensor.clone() for tensor in raw]
        offset = 2**32 - 200 + step * (len(tensors) - 2) * 4099
        options = {**update_options(kind, seed=97, step=step), "offset": offset}
        function(*raw, rounding="nearest", **options)
        reference(*expected, state_dtype=torch.float32, rounding="nearest", **options)
        for actual, target in zip(raw, expected):
            torch.testing.assert_close(
                actual, target, rtol=8e-3 if actual.dtype == torch.bfloat16 else 1e-5, atol=1e-6
            )
        function(*tensors, rounding=rounding, **options)
        # Parameter writes must use the full FP32 intermediates in both paths.
        torch.testing.assert_close(tensors[0], raw[0], rtol=0, atol=0)
        for i in range(2, len(tensors)):
            target = (
                stochastic_round_bf16(raw[i], seed=97, offset=offset + (i - 2) * 4099)
                if rounding == "stochastic"
                else raw[i].to(torch.bfloat16)
            )
            assert torch.equal(tensors[i], target)


@pytest.mark.parametrize("kind", ["sgdm", "adamw"])
@pytest.mark.parametrize("policy", list(POLICIES))
def test_state_kernel_block_invariance(kind, policy):
    inputs = prepare_inputs(kind, 4099, count=1, dtype=torch.float32, device="cuda")
    outputs = []
    for block_size in (128, 256, 512, 1024):
        case = build_case(
            inputs, optimizer=kind, backend="triton", policy=policy, block_size=block_size
        )
        case.run()
        outputs.append(case.buffers[0])
    for tensors in outputs[1:]:
        for tensor, expected in zip(tensors, outputs[0]):
            assert torch.equal(tensor, expected)


@pytest.mark.parametrize("optimizer_class", [SGDMTriton, AdamWTriton])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("parameter_dtype", [torch.float32, torch.bfloat16])
def test_gpu_checkpoint_resumes_identically(optimizer_class, state_dtype, parameter_dtype):
    initial = torch.linspace(-1, 1, 1003, device="cuda", dtype=parameter_dtype)
    p = torch.nn.Parameter(initial.clone())
    optimizer = optimizer_class([p], lr=0.01, state_dtype=state_dtype, seed=77)
    p.grad = torch.cos(initial)
    optimizer.step()
    q = torch.nn.Parameter(p.detach().clone())
    resumed = optimizer_class([q], lr=0.1, seed=12)
    resumed.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    for step in range(1, 6):
        p.grad = torch.sin(initial * step)
        q.grad = p.grad.clone()
        optimizer.step()
        resumed.step()
        assert torch.equal(p, q)
        for name in optimizer._state_names:
            assert optimizer.state[p][name].dtype == state_dtype
            assert torch.equal(optimizer.state[p][name], resumed.state[q][name])


@pytest.mark.parametrize("kind", ["sgdm", "adamw"])
def test_inductor_complete_updates(kind):
    compiler = compile_adamw_step_ if kind == "adamw" else compile_sgdm_step_
    compiled = compiler(fullgraph=True, dynamic=True, options={"triton.cudagraphs": False})
    for n in (257, 1003):
        inputs = prepare_inputs(kind, n, count=2, dtype=torch.float32, device="cuda")
        for policy in POLICIES:
            build_case(
                inputs, optimizer=kind, backend="compiled", policy=policy, compiled=compiled
            ).validate()


@pytest.mark.parametrize("kind", ["sgdm", "adamw"])
def test_state_update_on_nondefault_stream(kind):
    inputs = prepare_inputs(kind, 1003, count=1, dtype=torch.float32, device="cuda")
    case = build_case(inputs, optimizer=kind, backend="triton", policy="bf16_sr")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        case.validate()
    torch.cuda.current_stream().wait_stream(stream)


@pytest.mark.parametrize("kind", ["sgdm", "adamw"])
def test_state_validation_and_disjoint_buffers(kind):
    tensors = list(prepare_inputs(kind, 257, count=1, dtype=torch.float32, device="cuda")[0])
    function = step_function(kind)
    options = update_options(kind, seed=0, step=1)
    with pytest.raises(ValueError, match="overlap"):
        function(tensors[0], tensors[0], *tensors[2:], **options)
    with pytest.raises(ValueError, match="identical shapes"):
        function(tensors[0][:-1], *tensors[1:], **options)
    strided = [torch.zeros(514, device="cuda")[::2] for _ in tensors]
    with pytest.raises(ValueError, match="contiguous"):
        function(*strided, **options)
    if kind == "adamw":
        with pytest.raises(ValueError, match="positive integer"):
            function(*tensors, **{**options, "step": 0})
        with pytest.raises(TypeError, match="same dtype"):
            function(*tensors[:-1], tensors[-1].bfloat16(), **options)


@pytest.mark.parametrize("kind", ["sgdm", "adamw"])
def test_fp32_optimizer_matches_pytorch(kind):
    p = torch.nn.Parameter(torch.linspace(-1, 1, 1003, device="cuda"))
    q = torch.nn.Parameter(p.detach().clone())
    if kind == "adamw":
        options = {"lr": 0.01, "betas": (0.8, 0.95), "eps": 1e-6, "weight_decay": 0.1}
        optimizer = AdamWTriton([p], state_dtype=torch.float32, **options)
        native = torch.optim.AdamW([q], foreach=False, **options)
    else:
        options = {"lr": 0.01, "momentum": 0.8}
        optimizer = SGDMTriton([p], state_dtype=torch.float32, **options)
        native = torch.optim.SGD([q], foreach=False, **options)
    for step in range(1, 21):
        p.grad = torch.sin(torch.arange(1003, device="cuda") * 0.001 + step)
        q.grad = p.grad.clone()
        optimizer.step()
        native.step()
        torch.testing.assert_close(p, q, rtol=1e-5, atol=1e-6)
        for name in optimizer._state_names:
            torch.testing.assert_close(
                optimizer.state[p][name], native.state[q][name], rtol=1e-5, atol=1e-6
            )


@pytest.mark.parametrize("betas", [(0.0, 0.0), (0.9, 0.999)])
def test_adamw_zero_gradient_and_decay(betas):
    p = torch.linspace(-1, 1, 1003, device="cuda")
    expected = p * (1.0 - 0.01 * 0.1)
    g = torch.zeros_like(p)
    m = torch.zeros_like(p, dtype=torch.bfloat16)
    v = torch.zeros_like(m)
    triton_adamw_step_(
        p,
        g,
        m,
        v,
        lr=0.01,
        beta1=betas[0],
        beta2=betas[1],
        eps=1e-8,
        weight_decay=0.1,
        step=1,
        seed=7,
    )
    torch.testing.assert_close(p, expected, rtol=0, atol=0)
    assert torch.count_nonzero(m) == 0
    assert torch.count_nonzero(v) == 0


@pytest.mark.parametrize("kind", ["sgdm", "adamw"])
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_state_update_uses_tensor_device(kind):
    original_device = torch.cuda.current_device()
    other_device = (original_device + 1) % torch.cuda.device_count()
    inputs = prepare_inputs(kind, 1003, count=1, dtype=torch.float32, device=f"cuda:{other_device}")
    build_case(inputs, optimizer=kind, backend="triton", policy="bf16_sr").validate()
    assert torch.cuda.current_device() == original_device
