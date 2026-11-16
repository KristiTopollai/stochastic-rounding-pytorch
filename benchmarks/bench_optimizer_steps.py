"""Full optimizer.step() timing on synthetic problems and model-shaped tensors."""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import platform
import random
import statistics
import sys
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from benchmarks._utils import positive_int, write_rows
from sr_states import AdamWReferenceRN, AdamWReferenceSR, SGDMReferenceRN, SGDMReferenceSR
from sr_states.compiled import AdamWCompiledSR, SGDMCompiledSR
from sr_states.formats import StateFormat
from sr_states.triton import AdamWTriton, SGDMTriton, is_triton_available

METHODS = ("native_nr", "triton_nr", "eager_sr", "compiled_sr", "triton_sr")
BATCH_METHODS = ("torch_fused", "triton_fp32", "triton_sr", "triton_fp8")
ALL_METHODS = (*METHODS, *(name for name in BATCH_METHODS if name not in METHODS))
FP8 = (torch.float8_e4m3fn, torch.float8_e5m2)

# GPT-2 small and Llama 3.2 1B use tied token embeddings / output projection.
MODEL_SHAPES = {
    "gpt2": (
        (50257, 768),
        (1024, 768),
        *(
            (768,),
            (768,),
            (768, 2304),
            (2304,),
            (768, 768),
            (768,),
            (768,),
            (768,),
            (768, 3072),
            (3072,),
            (3072, 768),
            (768,),
        )
        * 12,
        (768,),
        (768,),
    ),
    "llama32-1b": (
        (128256, 2048),
        *(
            (2048,),
            (2048, 2048),
            (512, 2048),
            (512, 2048),
            (2048, 2048),
            (2048,),
            (8192, 2048),
            (8192, 2048),
            (2048, 8192),
        )
        * 16,
        (2048,),
    ),
}


class Problem:
    parameters: list[torch.nn.Parameter]

    def remember_initial(self):
        self.initial = [p.detach().clone() for p in self.parameters]

    @torch.no_grad()
    def reset(self):
        for parameter, initial in zip(self.parameters, self.initial):
            parameter.copy_(initial)
            parameter.grad = None

    @property
    def elements(self):
        return sum(p.numel() for p in self.parameters)


class Quadratic(Problem):
    """f(p) = sum_i h_i (p_i - target_i)^2 / 2, with h_i in [0.1, 1]."""

    def __init__(self, elements, tensors, *, device, seed):
        if tensors > elements:
            raise ValueError("tensor count cannot exceed the number of elements")
        generator = torch.Generator(device=device).manual_seed(seed)
        sizes = [elements // tensors + (i < elements % tensors) for i in range(tensors)]
        self.parameters = [
            torch.nn.Parameter(torch.randn(n, device=device, generator=generator)) for n in sizes
        ]
        self.targets = [
            torch.randn(p.shape, device=device, generator=generator) for p in self.parameters
        ]
        self.curvatures = [
            0.1 + 0.9 * torch.rand(p.shape, device=device, generator=generator)
            for p in self.parameters
        ]
        self.remember_initial()

    @torch.no_grad()
    def prepare_gradients(self):
        for p, target, curvature in zip(self.parameters, self.targets, self.curvatures):
            if p.grad is None:
                p.grad = torch.empty_like(p)
            p.grad.copy_(curvature * (p - target))


class MLPRegression(Problem):
    """A tanh MLP fitted by mean squared error to a fixed synthetic batch."""

    def __init__(self, width, depth, batch_size, *, device, seed, input_dim=32, output_dim=16):
        generator = torch.Generator(device=device).manual_seed(seed)
        dimensions = [input_dim, *([width] * depth), output_dim]
        self.parameters = []
        for fan_in, fan_out in itertools.pairwise(dimensions):
            weight = torch.randn(fan_out, fan_in, generator=generator, device=device) / math.sqrt(
                fan_in
            )
            self.parameters.extend(
                [
                    torch.nn.Parameter(weight),
                    torch.nn.Parameter(torch.zeros(fan_out, device=device)),
                ]
            )
        self.inputs = torch.randn(batch_size, input_dim, generator=generator, device=device)
        teacher = torch.randn(
            input_dim, output_dim, generator=generator, device=device
        ) / math.sqrt(input_dim)
        self.targets = torch.tanh(self.inputs @ teacher)
        self.remember_initial()

    def loss(self):
        output = self.inputs
        for index in range(0, len(self.parameters), 2):
            output = F.linear(output, self.parameters[index], self.parameters[index + 1])
            if index + 2 < len(self.parameters):
                output = torch.tanh(output)
        return F.mse_loss(output, self.targets)

    def prepare_gradients(self):
        for parameter in self.parameters:
            if parameter.grad is not None:
                parameter.grad.zero_()
        self.loss().backward()


class ModelShapes(Problem):
    """Time optimizer traffic on FP32 tensors without allocating a model or activations."""

    def __init__(self, shapes, *, device, seed):
        self.seed = seed
        self.device = device
        self.parameters = [
            torch.nn.Parameter(torch.empty(shape, device=device)) for shape in shapes
        ]
        self.reset()

    @torch.no_grad()
    def reset(self):
        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        for p in self.parameters:
            p.normal_(std=0.02, generator=generator)
            p.grad = None

    def prepare_gradients(self):
        if self.parameters[0].grad is not None:
            return
        generator = torch.Generator(device=self.device).manual_seed(self.seed + 1)
        for p in self.parameters:
            p.grad = torch.empty_like(p).normal_(std=0.25, generator=generator)


class CompilationCounter:
    """Count actual graph compilations, including any accidental timed compilation."""

    def __init__(self, backend="inductor"):
        self.backend = backend
        self.count = 0

    def __call__(self, graph, inputs):
        self.count += 1
        if self.backend == "eager":
            return graph.forward
        from torch._dynamo import lookup_backend
        from torch._inductor import config

        with config.patch({"triton.cudagraphs": False}):
            return lookup_backend(self.backend)(graph, inputs)


def optimizer_options(kind, seed):
    if kind == "adamw":
        return {"lr": 1e-3, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 1e-2, "seed": seed}
    return {"lr": 1e-2, "momentum": 0.9, "seed": seed}


@dataclass
class OptimizerCase:
    problem: Problem
    optimizer: torch.optim.Optimizer
    kind: str
    method: str
    compilation: CompilationCounter | None

    def restart(self):
        self.problem.reset()
        self.optimizer.state.clear()
        for group in self.optimizer.param_groups:
            if "sr_step" in group:
                group["sr_step"] = 0

    def advance(self):
        self.problem.prepare_gradients()
        self.optimizer.step()

    @property
    def timing_only(self):
        return self.method == "torch_fused"

    @property
    def state_names(self):
        if self.timing_only:
            return ("exp_avg", "exp_avg_sq") if self.kind == "adamw" else ("momentum_buffer",)
        return self.optimizer._state_names

    def compilation_count(self):
        return self.compilation.count if self.compilation is not None else 0


def build_case(problem, kind, method, *, seed=7, block_size=256, compile_backend="inductor"):
    if kind not in {"adamw", "sgdm"} or method not in ALL_METHODS:
        raise ValueError("unknown optimizer or method")
    adamw = kind == "adamw"
    triton_cls = AdamWTriton if adamw else SGDMTriton
    kwargs = optimizer_options(kind, seed)
    tracker = None
    if method == "torch_fused":
        kwargs.pop("seed")
        cls = torch.optim.AdamW if adamw else torch.optim.SGD
        kwargs["fused"] = True
    else:
        kwargs["state_dtype"] = torch.float32 if method == "triton_fp32" else torch.bfloat16
        kwargs["rounding"] = (
            "nearest" if method.endswith("nr") or method == "triton_fp32" else "stochastic"
        )
        if method.startswith("triton"):
            cls = triton_cls
            kwargs["block_size"] = block_size
            if method == "triton_fp8":
                del kwargs["state_dtype"], kwargs["rounding"]
                kwargs["state_format"] = (
                    {
                        "exp_avg": StateFormat(torch.float8_e4m3fn),
                        "exp_avg_sq": StateFormat(torch.bfloat16),
                    }
                    if adamw
                    else StateFormat(torch.float8_e4m3fn)
                )
        elif method == "compiled_sr":
            cls = AdamWCompiledSR if adamw else SGDMCompiledSR
            tracker = CompilationCounter(compile_backend)
            kwargs["compile_kwargs"] = {"backend": tracker}
        elif method == "native_nr":
            cls = AdamWReferenceRN if adamw else SGDMReferenceRN
        else:
            cls = AdamWReferenceSR if adamw else SGDMReferenceSR
    optimizer = cls(problem.parameters, **kwargs)
    return OptimizerCase(problem, optimizer, kind, method, tracker)


def validate_case(case, *, steps=3):
    """Compare each step to eager FP32 math from the same current state and gradient."""
    if case.timing_only:
        return
    reference_class = AdamWReferenceSR if case.kind == "adamw" else SGDMReferenceSR
    for _ in range(steps):
        case.problem.prepare_gradients()
        expected_parameters = [
            torch.nn.Parameter(p.detach().clone()) for p in case.problem.parameters
        ]
        for expected, actual in zip(expected_parameters, case.problem.parameters):
            expected.grad = actual.grad.detach().clone()
        reference = reference_class(expected_parameters, **optimizer_options(case.kind, 0))
        reference.load_state_dict(copy.deepcopy(case.optimizer.state_dict()))
        reference.step()
        case.optimizer.step()
        for actual, expected in zip(case.problem.parameters, expected_parameters):
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            assert actual.dtype == actual.grad.dtype == torch.float32
            for name in case.state_names:
                value = case.optimizer.state[actual][name]
                ref = reference.state[expected][name]
                assert value.dtype == ref.dtype
                if value.dtype in FP8:
                    torch.testing.assert_close(value.float(), ref.float(), rtol=0.125, atol=2**-9)
                else:
                    rtol = 1e-5 if value.dtype == torch.float32 else 8e-3
                    torch.testing.assert_close(value, ref, rtol=rtol, atol=1e-6)
            if case.kind == "adamw":
                assert case.optimizer.state[actual]["step"] == reference.state[expected]["step"]
        assert case.optimizer._rng_step() == reference._rng_step()
    case.restart()


def measure_steps(case, *, warmup, repetitions):
    if warmup < 1 or repetitions < 1:
        raise ValueError("warmup and repetitions must be positive")
    # Rehearse the complete trajectory to warm scalar/shape specializations,
    # including Triton's signed/unsigned counter signatures near wraparound.
    case.restart()
    for _ in range(warmup + repetitions):
        case.advance()
    torch.cuda.synchronize()
    case.restart()
    for _ in range(warmup):
        case.advance()
    torch.cuda.synchronize()
    compilations = case.compilation_count()

    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    end.record()
    end.synchronize()
    wall, cuda, submit = [], [], []
    for _ in range(repetitions):
        case.problem.prepare_gradients()
        torch.cuda.synchronize()
        start.record()
        beginning = time.perf_counter_ns()
        case.optimizer.step()
        returned = time.perf_counter_ns()
        end.record()
        end.synchronize()
        completed = time.perf_counter_ns()
        if case.compilation_count() != compilations:
            raise RuntimeError("compiled SR recompiled during timing; measurement discarded")
        wall.append((completed - beginning) / 1e6)
        submit.append((returned - beginning) / 1e6)
        cuda.append(start.elapsed_time(end))
    if any(
        not math.isfinite(value) or value <= 0
        for samples in (wall, cuda, submit)
        for value in samples
    ):
        raise RuntimeError("nonpositive or nonfinite optimizer timing")
    for parameter in case.problem.parameters:
        if not torch.isfinite(parameter).all():
            raise RuntimeError("synthetic optimization produced nonfinite parameters")
        for name in case.state_names:
            value = case.optimizer.state[parameter][name]
            if not torch.isfinite(value.float() if value.dtype in FP8 else value).all():
                raise RuntimeError("synthetic optimization produced nonfinite states")
    return {"wall": wall, "cuda": cuda, "submit": submit}


def summarize(samples):
    quantiles = statistics.quantiles(samples, n=10, method="inclusive") if len(samples) > 1 else []
    return {
        "median_ms": statistics.median(samples),
        "p10_ms": quantiles[0] if quantiles else samples[0],
        "p90_ms": quantiles[-1] if quantiles else samples[0],
        "samples_ms": json.dumps(samples),
    }


def make_row(case, samples, *, workload, args):
    import triton

    state = case.optimizer.state[case.problem.parameters[0]]
    dtype_names = {
        torch.float32: "fp32",
        torch.bfloat16: "bf16",
        torch.float8_e4m3fn: "fp8_e4m3",
    }
    group = case.optimizer.param_groups[0]
    hyperparameters = optimizer_options(case.kind, args.seed)
    if case.timing_only:
        del hyperparameters["seed"]
    row = {
        "optimizer": case.kind,
        "problem": workload,
        "method": case.method,
        "implementation": type(case.optimizer).__name__,
        "elements": case.problem.elements,
        "tensors": len(case.problem.parameters),
        "shapes": json.dumps([list(p.shape) for p in case.problem.parameters]),
        "parameter_dtype": "fp32",
        "state_dtype": "+".join(
            dict.fromkeys(dtype_names[state[name].dtype] for name in case.state_names)
        ),
        "rounding": ""
        if case.timing_only
        else next(iter(group["state_format"].values()))["rounding"],
        "timing": "optimizer_step_synchronized_wall",
        **summarize(samples["wall"]),
        **{f"cuda_{key}": value for key, value in summarize(samples["cuda"]).items()},
        **{f"submit_{key}": value for key, value in summarize(samples["submit"]).items()},
        "speedup_vs_native_nr": "",
        "speedup_vs_triton_nr": "",
        "cuda_speedup_vs_native_nr": "",
        "cuda_speedup_vs_triton_nr": "",
        "speedup_vs_torch_fused": "",
        "cuda_speedup_vs_torch_fused": "",
        "seed": args.seed,
        "order_seed": args.order_seed,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "rehearsal_steps": args.warmup + args.repetitions,
        "first_timed_step": args.warmup + 1,
        "last_timed_step": args.warmup + args.repetitions,
        "final_rng_step": "" if case.timing_only else case.optimizer._rng_step(),
        "state_bytes": sum(
            state[name].numel() * state[name].element_size()
            for state in case.optimizer.state.values()
            for name in case.state_names
        ),
        "hyperparameters": json.dumps(hyperparameters),
        "block_size": args.block_size if case.method.startswith("triton") else "",
        "compiled_graphs": case.compilation_count(),
        "timed_compiled_graphs": 0,
        "correctness": "timing_only" if case.timing_only else "passed",
        "gpu": torch.cuda.get_device_name(),
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
        "python_version": platform.python_version(),
    }
    return row


def add_speedups(rows):
    for prefix in ("", "cuda_"):
        times = {row["method"]: row[f"{prefix}median_ms"] for row in rows}
        for row in rows:
            for baseline in ("native_nr", "triton_nr", "torch_fused"):
                if baseline in times:
                    row[f"{prefix}speedup_vs_{baseline}"] = (
                        times[baseline] / row[f"{prefix}median_ms"]
                    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--optimizers", nargs="+", choices=("adamw", "sgdm"), default=["adamw", "sgdm"]
    )
    parser.add_argument(
        "--problems",
        nargs="+",
        choices=("quadratic", "mlp", *MODEL_SHAPES),
        default=["quadratic", "mlp"],
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=ALL_METHODS,
        help="defaults to the original methods on quadratic/MLP, or batching comparisons on model shapes",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=positive_int,
        default=[1048576],
        help="total quadratic parameters",
    )
    parser.add_argument(
        "--tensors",
        nargs="+",
        type=positive_int,
        default=[1, 32],
        help="quadratic parameter tensors",
    )
    parser.add_argument("--width", type=positive_int, default=512)
    parser.add_argument("--depth", type=positive_int, default=4)
    parser.add_argument("--batch-size", type=positive_int, default=64)
    parser.add_argument("--warmup", type=positive_int, default=20)
    parser.add_argument("--repetitions", type=positive_int, default=100)
    parser.add_argument("--block-size", type=int, choices=(128, 256, 512, 1024), default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--order-seed", type=int, default=0)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    if "quadratic" in args.problems and max(args.tensors) > min(args.sizes):
        parser.error("each quadratic size must be at least the largest tensor count")
    return args


def workloads(args):
    if "quadratic" in args.problems:
        for elements in args.sizes:
            for tensors in args.tensors:
                yield f"quadratic-{elements}-{tensors}", {"elements": elements, "tensors": tensors}
    if "mlp" in args.problems:
        yield (
            f"mlp-{args.width}-{args.depth}-{args.batch_size}",
            {
                "width": args.width,
                "depth": args.depth,
                "batch_size": args.batch_size,
            },
        )
    for name, shapes in MODEL_SHAPES.items():
        if name in args.problems:
            yield name, {"shapes": shapes}


def main(argv=None):
    args = parse_args(argv)
    if not is_triton_available():
        raise SystemExit("This benchmark requires an NVIDIA CUDA device and Triton")
    order = random.Random(args.order_seed)
    rows = []
    for workload, configuration in workloads(args):
        if workload in MODEL_SHAPES:
            problem_class, defaults = ModelShapes, BATCH_METHODS
        else:
            problem_class = Quadratic if workload.startswith("quadratic") else MLPRegression
            defaults = METHODS
        for kind in args.optimizers:
            methods = list(args.methods or defaults)
            if torch.cuda.get_device_capability() < (8, 9) and "triton_fp8" in methods:
                print(f"{workload}: skipping triton_fp8 (requires sm_89)", file=sys.stderr)
                methods.remove("triton_fp8")
            order.shuffle(methods)
            group_rows = []
            for method in methods:
                if method == "compiled_sr":
                    # Measurements are independent compilation workloads.
                    torch._dynamo.reset()
                problem = problem_class(**configuration, device="cuda", seed=args.seed)
                case = build_case(problem, kind, method, seed=args.seed, block_size=args.block_size)
                print(f"{workload}: {kind} {method}", file=sys.stderr, flush=True)
                validate_case(case)
                samples = measure_steps(case, warmup=args.warmup, repetitions=args.repetitions)
                row = make_row(case, samples, workload=workload, args=args)
                group_rows.append(row)
                rows.append(row)
                print(
                    f"  step {row['median_ms'] * 1000:.2f} us; CUDA events {row['cuda_median_ms'] * 1000:.2f} us",
                    file=sys.stderr,
                    flush=True,
                )
                if args.output:
                    write_rows(rows, args.output)
                del case, problem
            add_speedups(group_rows)
            if args.output:
                write_rows(rows, args.output)
    if not args.output:
        write_rows(rows, None)


if __name__ == "__main__":
    main()
