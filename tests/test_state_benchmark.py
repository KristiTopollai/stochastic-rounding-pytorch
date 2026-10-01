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


@pytest.mark.parametrize("optimizer", ["sgdm", "adamw"])
@pytest.mark.parametrize("policy", list(POLICIES))
@pytest.mark.parametrize("backend", ["eager", "compiled"])
def test_benchmark_updates_in_place_and_resets(optimizer, policy, backend):
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
            assert torch.equal(tensor, source)


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
