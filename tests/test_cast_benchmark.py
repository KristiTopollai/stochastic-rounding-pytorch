import pytest
import torch

from benchmarks import bench_sr_cast
from benchmarks._utils import cuda_samples_ms
from benchmarks.cast_checks import prepare_cases, prepare_rn_cases, validate_cast
from sr_states.reference import stochastic_round_bf16


def test_benchmark_covers_all_five_methods_with_matching_launches(monkeypatch):
    calls = []

    def cast(x, **kwargs):
        calls.append(kwargs)
        return x.to(torch.bfloat16)

    monkeypatch.setattr(bench_sr_cast, "sr_cast_bf16", cast)
    monkeypatch.setattr(bench_sr_cast, "rn_cast_bf16", cast)
    x = torch.tensor([0.0, -0.0, 1.5])
    methods = bench_sr_cast.build_methods(x, compiled=stochastic_round_bf16, seed=9, block_size=256)
    assert set(methods) == {"bf16_rn", "triton_rn", "pytorch_sr", "compiled_sr", "triton_sr"}
    for function in methods.values():
        assert torch.equal(function().view(torch.int16), x.to(torch.bfloat16).view(torch.int16))
    assert calls == [{"block_size": 256}, {"seed": 9, "block_size": 256}]


def test_correctness_gate_cases_on_cpu():
    validate_cast(stochastic_round_bf16, prepare_cases("cpu"))
    validate_cast(lambda x, **_: x.to(torch.bfloat16), prepare_rn_cases("cpu"))


@pytest.mark.parametrize("flag", ["--sizes", "--repetitions"])
def test_benchmark_rejects_zero_samples_or_size(flag):
    with pytest.raises(SystemExit):
        bench_sr_cast.parse_args([flag, "0"])


def test_zero_warmup_still_initializes_before_timing(monkeypatch):
    initialized = False

    def function():
        nonlocal initialized
        initialized = True

    class Event:
        def __init__(self, **kwargs):
            pass

        def record(self):
            assert initialized, "initial compilation must precede event timing"

        def synchronize(self):
            pass

        def elapsed_time(self, other):
            return 1.0

    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    assert cuda_samples_ms(function, warmup=0, repetitions=2) == [1.0, 1.0]
