import copy
import itertools
import json

import pytest
import torch

from benchmarks import bench_optimizer_steps as bench
from sr_states import AdamWReferenceSR, SGDMReferenceSR
from sr_states.triton import is_triton_available


def problem(name, device="cpu"):
    if name == "quadratic":
        return bench.Quadratic(37, 3, device=device, seed=7)
    return bench.MLPRegression(8, 2, 4, device=device, seed=7)


@pytest.mark.parametrize("name", ["quadratic", "mlp"])
def test_synthetic_problems_reproduce_gradients_and_reset(name):
    first, second = problem(name), problem(name)
    first.prepare_gradients()
    second.prepare_gradients()
    gradients = [p.grad.clone() for p in first.parameters]
    for p, q in zip(first.parameters, second.parameters):
        assert torch.equal(p, q)
        assert torch.equal(p.grad, q.grad)
        assert p.dtype == p.grad.dtype == torch.float32
        assert torch.isfinite(p.grad).all()
    if name == "quadratic":
        assert [p.numel() for p in first.parameters] == [13, 12, 12]
        objective = sum(
            (h * (p - t).square()).sum() / 2
            for p, t, h in zip(first.parameters, first.targets, first.curvatures)
        )
        expected = torch.autograd.grad(objective, first.parameters)
        for actual, reference in zip(gradients, expected):
            torch.testing.assert_close(actual, reference)
    first.prepare_gradients()
    for p, expected in zip(first.parameters, gradients):
        assert torch.equal(p.grad, expected)  # No gradient accumulation across steps.
    with torch.no_grad():
        for p in first.parameters:
            p.add_(1)
    first.reset()
    for p, q in zip(first.parameters, second.parameters):
        assert torch.equal(p, q)
        assert p.grad is None


@pytest.mark.parametrize("kind", ["adamw", "sgdm"])
@pytest.mark.parametrize("method", ["native_nr", "eager_sr", "compiled_sr"])
def test_full_optimizer_path_advances_and_restarts(kind, method):
    if method == "compiled_sr":
        torch._dynamo.reset()
    case = bench.build_case(problem("mlp"), kind, method, compile_backend="eager")
    assert case.optimizer.param_groups[0]["rounding"] == (
        "nearest" if method == "native_nr" else "stochastic"
    )
    bench.validate_case(case)
    assert not case.optimizer.state
    for p, initial in zip(case.problem.parameters, case.problem.initial):
        assert torch.equal(p, initial)
    pointers = [p.data_ptr() for p in case.problem.parameters]
    for _ in range(4):
        case.advance()
    assert case.optimizer._next_offset() == 4 * case.problem.elements * len(
        case.optimizer._state_names
    )
    for p in case.problem.parameters:
        for name in case.optimizer._state_names:
            assert case.optimizer.state[p][name].dtype == torch.bfloat16
        if kind == "adamw":
            assert case.optimizer.state[p]["step"] == 4
    assert pointers == [p.data_ptr() for p in case.problem.parameters]
    if method == "compiled_sr":
        count = case.compilation_count()
        assert count > 0
        # Replay the complete warmed trajectory with fresh state buffers.
        case.restart()
        for _ in range(4):
            case.advance()
        assert case.compilation_count() == count


@pytest.mark.parametrize("kind", ["adamw", "sgdm"])
def test_compiled_optimizer_multistep_checkpoint_and_skipped_gradient(kind):
    torch._dynamo.reset()
    case = bench.build_case(problem("quadratic"), kind, "compiled_sr", compile_backend="eager")
    reference_class = AdamWReferenceSR if kind == "adamw" else SGDMReferenceSR
    parameters = [torch.nn.Parameter(p.detach().clone()) for p in case.problem.parameters]
    reference = reference_class(parameters, **bench.optimizer_options(kind, 7))
    for step in range(12):
        for index, (p, q) in enumerate(zip(case.problem.parameters, parameters)):
            p.grad = None if index == 1 and step % 2 == 0 else torch.sin(p.detach() + step)
            q.grad = None if p.grad is None else p.grad.clone()
        case.optimizer.step()
        reference.step()
        for p, q in zip(case.problem.parameters, parameters):
            assert torch.equal(p, q)
            for name in case.optimizer._state_names:
                if name in reference.state[q]:
                    assert torch.equal(case.optimizer.state[p][name], reference.state[q][name])
        if step == 5:
            restored = bench.build_case(case.problem, kind, "compiled_sr", compile_backend="eager")
            restored.optimizer.load_state_dict(copy.deepcopy(case.optimizer.state_dict()))
            case = restored
    assert case.optimizer._next_offset() == reference._next_offset()
    if kind == "adamw":
        assert [case.optimizer.state[p]["step"] for p in case.problem.parameters] == [12, 6, 12]


def fake_cuda(monkeypatch, calls):
    class Event:
        names = iter(("start", "end"))

        def __init__(self, **kwargs):
            self.name = next(self.names)

        def record(self):
            calls.append(self.name)

        def synchronize(self):
            calls.append("wait")

        def elapsed_time(self, end):
            return 0.1

    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: calls.append("sync"))
    clock = itertools.count(0, 100_000)
    monkeypatch.setattr(bench.time, "perf_counter_ns", lambda: next(clock))


@pytest.mark.parametrize("kind", ["adamw", "sgdm"])
def test_timing_excludes_gradients_and_warmup_but_calls_full_step(monkeypatch, kind):
    calls = []
    fake_cuda(monkeypatch, calls)
    case = bench.build_case(problem("quadratic"), kind, "eager_sr")
    prepare, step = case.problem.prepare_gradients, case.optimizer.step

    def gradients():
        calls.append("gradient")
        prepare()

    def update():
        calls.append("step")
        step()

    monkeypatch.setattr(case.problem, "prepare_gradients", gradients)
    monkeypatch.setattr(case.optimizer, "step", update)
    result = bench.measure_steps(case, warmup=2, repetitions=3)
    assert calls == (
        ["gradient", "step"] * 5  # Complete untimed rehearsal.
        + ["sync"]
        + ["gradient", "step"] * 2
        + ["sync", "start", "end", "wait"]
        + ["gradient", "sync", "start", "step", "end", "wait"] * 3
    )
    assert result == {"wall": [0.2] * 3, "cuda": [0.1] * 3, "submit": [0.1] * 3}
    assert case.optimizer._next_offset() == 5 * case.problem.elements * len(
        case.optimizer._state_names
    )
    if kind == "adamw":
        assert all(case.optimizer.state[p]["step"] == 5 for p in case.problem.parameters)


def test_timed_recompilation_rejects_measurement(monkeypatch):
    fake_cuda(monkeypatch, [])
    case = bench.build_case(problem("quadratic"), "sgdm", "eager_sr")
    case.compilation = bench.CompilationCounter("eager")
    step = case.optimizer.step

    def update():
        step()
        case.compilation.count += 1

    monkeypatch.setattr(case.optimizer, "step", update)
    with pytest.raises(RuntimeError, match="recompiled during timing"):
        bench.measure_steps(case, warmup=1, repetitions=2)


def test_speedup_direction_and_single_sample_summary():
    rows = [
        {"method": method, "median_ms": value, "cuda_median_ms": value * 0.5}
        for method, value in zip(bench.METHODS, [10, 2, 20, 5, 4])
    ]
    bench.add_speedups(rows)
    assert rows[-1]["speedup_vs_native_nr"] == 2.5
    assert rows[-1]["speedup_vs_triton_nr"] == 0.5
    assert rows[-1]["cuda_speedup_vs_native_nr"] == 2.5
    summary = bench.summarize([2.5])
    assert summary["median_ms"] == summary["p10_ms"] == summary["p90_ms"] == 2.5
    assert json.loads(summary["samples_ms"]) == [2.5]


@pytest.mark.parametrize("option", ["--sizes", "--tensors", "--warmup", "--repetitions", "--width"])
def test_rejects_nonpositive_workloads(option):
    with pytest.raises(SystemExit):
        bench.parse_args([option, "0"])


def test_quadratic_sizes_are_total_elements_and_cannot_be_smaller_than_tensor_count():
    args = bench.parse_args(["--problems", "quadratic", "--sizes", "100", "--tensors", "1", "4"])
    assert list(bench.workloads(args)) == [
        ("quadratic-100-1", {"elements": 100, "tensors": 1}),
        ("quadratic-100-4", {"elements": 100, "tensors": 4}),
    ]
    with pytest.raises(SystemExit):
        bench.parse_args(["--sizes", "2", "--tensors", "3"])


@pytest.mark.cuda
@pytest.mark.skipif(not is_triton_available(), reason="CUDA and Triton required")
@pytest.mark.parametrize("kind", ["adamw", "sgdm"])
@pytest.mark.parametrize("method", bench.METHODS)
@pytest.mark.parametrize("name", ["quadratic", "mlp"])
def test_gpu_synthetic_full_steps(kind, method, name):
    if method == "compiled_sr":
        torch._dynamo.reset()
    case = bench.build_case(problem(name, "cuda"), kind, method)
    bench.validate_case(case)
    samples = bench.measure_steps(case, warmup=2, repetitions=3)
    assert all(len(values) == 3 and min(values) > 0 for values in samples.values())
    assert case.optimizer._next_offset() == 5 * case.problem.elements * len(
        case.optimizer._state_names
    )
    args = bench.parse_args([])
    row = bench.make_row(case, samples, workload=name, args=args)
    assert row["state_bytes"] == 2 * case.problem.elements * len(case.optimizer._state_names)
    assert row["timed_compiled_graphs"] == 0
