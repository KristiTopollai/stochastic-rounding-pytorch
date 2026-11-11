import pytest
import torch

from benchmarks.bench_states import (
    POLICIES,
    build_case,
    logical_bytes_per_element,
    parse_args,
    prepare_inputs,
)
from sr_states.compiled import compile_adamw_step_, compile_sgdm_step_
from sr_states.rounding import cast


def fused_sgdm_for_validation(parameter, gradient, momentum, **options):
    # These small dyadic inputs and the FP32 coefficient fit exactly in FP64;
    # one final FP32 conversion reproduces a single FP32 multiply-add rounding.
    factor = float(torch.tensor(options["momentum_factor"], dtype=torch.float32))
    proposal = (momentum.float().double() * factor + gradient.float().double()).float()
    parameter.copy_(parameter.float().add(proposal, alpha=-options["lr"]).to(parameter.dtype))
    momentum.copy_(
        cast(
            proposal,
            dtype=momentum.dtype,
            rounding=options["rounding"],
            seed=options["seed"],
            offset=options["offset"],
        )
    )


def test_fp8_benchmark_validates_fused_subnormal_boundary():
    inputs = [(torch.ones(2), torch.tensor([-8, 8]) / 512, torch.tensor([15, -15]) / 512)]
    case = build_case(inputs, optimizer="sgdm", backend="triton", policy="fp8e4_nr")
    case.function = fused_sgdm_for_validation
    case.run()
    assert torch.equal(case.buffers[0][2].float(), torch.tensor([5, -5]) / 512)
    expected = [t.clone() for t in case.initial[0]]
    case.reference(*expected, **case.options, state_dtype=case.state_dtype, rounding="nearest")
    assert torch.equal(expected[2].float(), torch.tensor([6, -6]) / 512)
    case.validate()


@pytest.mark.parametrize("policy", ["fp8e4_nr", "fp8e4_sr", "fp8e5_nr", "fp8e5_sr"])
def test_fp8_benchmark_rejects_wrong_storage_with_correct_recurrence(policy):
    inputs = [(torch.ones(1), torch.ones(1), torch.zeros(1))]
    case = build_case(inputs, optimizer="sgdm", backend="triton", policy=policy)

    def wrong_storage(parameter, gradient, momentum, **options):
        fused_sgdm_for_validation(parameter, gradient, momentum, **options)
        if momentum.dtype != torch.float32:
            momentum.copy_((momentum.float() + torch.finfo(momentum.dtype).eps).to(momentum.dtype))

    case.function = wrong_storage
    with pytest.raises(AssertionError):
        case.validate()


@pytest.mark.parametrize("policy", ["fp8e4_nr", "fp8e4_sr", "fp8e5_nr", "fp8e5_sr"])
def test_fp8_benchmark_rejects_wrong_fp32_recurrence(policy):
    inputs = [(torch.ones(1), torch.ones(1), torch.zeros(1))]
    case = build_case(inputs, optimizer="sgdm", backend="triton", policy=policy)

    def wrong_recurrence(parameter, gradient, momentum, **options):
        fused_sgdm_for_validation(parameter, gradient, momentum, **options)
        momentum.copy_((momentum.float() + 0.0625).to(momentum.dtype))

    case.function = wrong_recurrence
    with pytest.raises(AssertionError):
        case.validate()


@pytest.mark.parametrize("optimizer", ["sgdm", "adamw"])
@pytest.mark.parametrize("policy", list(POLICIES))
@pytest.mark.parametrize("backend", ["eager", "compiled"])
def test_benchmark_updates_in_place_and_resets(optimizer, policy, backend):
    torch._dynamo.reset()
    inputs = prepare_inputs(optimizer, 37, count=2, dtype=torch.float32, device="cpu")
    compiler = compile_adamw_step_ if optimizer == "adamw" else compile_sgdm_step_
    compiled = compiler(backend="eager", fullgraph=True, dynamic=True)
    case = build_case(
        inputs, optimizer=optimizer, backend=backend, policy=policy, compiled=compiled
    )
    pointers = [[tensor.data_ptr() for tensor in row] for row in case.buffers]
    case.validate()
    case.run()
    assert any(
        not torch.equal(row[0], initial[0]) for row, initial in zip(case.buffers, case.initial)
    )
    for actual, original in zip(case.buffers, pointers):
        assert [tensor.data_ptr() for tensor in actual] == original
    case.reset()
    for actual, initial in zip(case.buffers, case.initial):
        for tensor, source in zip(actual, initial):
            assert torch.equal(tensor.view(torch.uint8), source.view(torch.uint8))


@pytest.mark.parametrize(
    ("kind", "state_dtype", "expected"),
    [
        ("sgdm", torch.bfloat16, 16),
        ("sgdm", torch.float32, 20),
        ("adamw", torch.bfloat16, 20),
        ("adamw", torch.float32, 28),
    ],
)
def test_state_benchmark_traffic(kind, state_dtype, expected):
    assert logical_bytes_per_element(kind, torch.float32, state_dtype) == expected


@pytest.mark.parametrize("option", ["--sizes", "--tensors", "--step", "--repetitions"])
def test_state_benchmark_rejects_zero(option):
    with pytest.raises(SystemExit):
        parse_args(["--optimizer", "adamw", option, "0"])


def test_state_resets_stay_outside_timed_events(monkeypatch):
    from benchmarks._utils import cuda_samples_ms

    calls = []

    class Event:
        count = 0

        def __init__(self, **kwargs):
            self.name = "start" if Event.count == 0 else "end"
            Event.count += 1

        def record(self):
            calls.append(self.name)

        def synchronize(self):
            pass

        def elapsed_time(self, other):
            return 1.0

    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    cuda_samples_ms(
        lambda: calls.append("update"),
        warmup=1,
        repetitions=2,
        before_call=lambda: calls.append("reset"),
    )
    assert calls == [
        "reset",
        "update",
        "reset",
        "update",
        "start",
        "end",
        "reset",
        "start",
        "update",
        "end",
        "reset",
        "start",
        "update",
        "end",
    ]


@pytest.mark.parametrize("kind", ["sgdm", "adamw"])
def test_compiled_inplace_update_accepts_leaf_parameters(kind):
    compiler = compile_adamw_step_ if kind == "adamw" else compile_sgdm_step_
    compiled = compiler(backend="eager", fullgraph=True, dynamic=True)
    inputs = prepare_inputs(kind, 17, count=1, dtype=torch.float32, device="cpu")
    case = build_case(
        inputs, optimizer=kind, backend="compiled", policy="bf16_sr", compiled=compiled
    )
    tensors = list(case.buffers[0])
    tensors[0] = torch.nn.Parameter(tensors[0])
    for offset in (0, 31, 2**32 + 31):
        compiled(*tensors, **case.options, offset=offset, state_dtype=case.state_dtype)
    assert tensors[0].grad is None
