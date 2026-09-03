from __future__ import annotations

import argparse
from collections.abc import Callable

import torch

from benchmarks._utils import cuda_median_ms, result_row, write_rows
from sr_states.compiled import compile_sgdm_step
from sr_states.reference.sgd import sgdm_step
from sr_states.triton import is_triton_available, triton_sgdm_step_


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark eager, compiled, and fused Triton momentum-SGD updates."
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=positive_int,
        default=[2**k for k in range(10, 28, 2)],
    )
    parser.add_argument("--warmup", type=nonnegative_int, default=20)
    parser.add_argument("--repetitions", type=positive_int, default=100)
    parser.add_argument("--block-size", type=int, choices=(128, 256, 512, 1024), default=256)
    parser.add_argument("--parameter-dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output")
    return parser.parse_args()


def _logical_bytes_per_element(parameter_dtype: torch.dtype) -> int:
    parameter_bytes = torch.empty((), dtype=parameter_dtype).element_size()
    momentum_bytes = torch.empty((), dtype=torch.bfloat16).element_size()

    # Read parameter, gradient, and momentum; write parameter and momentum.
    return 3 * parameter_bytes + 2 * momentum_bytes


def _method_inputs(
    parameter: torch.Tensor,
    gradient: torch.Tensor,
    momentum: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Give each benchmark method independent mutable state."""

    return parameter.clone(), gradient.clone(), momentum.clone()


def _build_methods(
    parameter: torch.Tensor,
    gradient: torch.Tensor,
    momentum: torch.Tensor,
    *,
    compiled_step: Callable,
    seed: int,
    block_size: int,
) -> dict[str, Callable[[], object]]:
    common = {
        "lr": 1e-3,
        "momentum_factor": 0.9,
        "seed": seed,
        "offset": 0,
    }
    methods: dict[str, Callable[[], object]] = {}

    for rounding in ("nearest", "stochastic"):
        eager_p, eager_g, eager_m = _method_inputs(parameter, gradient, momentum)
        methods[f"pytorch_eager_{rounding}"] = (
            lambda p=eager_p, g=eager_g, m=eager_m, mode=rounding: sgdm_step(
                p,
                g,
                m,
                rounding=mode,
                **common,
            )
        )

        compiled_p, compiled_g, compiled_m = _method_inputs(parameter, gradient, momentum)
        methods[f"pytorch_compiled_{rounding}"] = (
            lambda p=compiled_p, g=compiled_g, m=compiled_m, mode=rounding: compiled_step(
                p,
                g,
                m,
                rounding=mode,
                **common,
            )
        )

        if is_triton_available():
            triton_p, triton_g, triton_m = _method_inputs(parameter, gradient, momentum)
            methods[f"triton_fused_{rounding}"] = (
                lambda p=triton_p, g=triton_g, m=triton_m, mode=rounding: triton_sgdm_step_(
                    p,
                    g,
                    m,
                    rounding=mode,
                    block_size=block_size,
                    **common,
                )
            )

    return methods


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("This benchmark requires an NVIDIA CUDA device")

    parameter_dtype = torch.bfloat16 if args.parameter_dtype == "bf16" else torch.float32
    logical_bytes = _logical_bytes_per_element(parameter_dtype)
    compiled_step = compile_sgdm_step(fullgraph=True)
    torch.manual_seed(args.seed)

    rows = []
    for n in args.sizes:
        parameter = torch.randn(n, device="cuda", dtype=parameter_dtype)
        gradient = torch.randn_like(parameter)
        momentum = torch.randn(n, device="cuda", dtype=torch.bfloat16)
        methods = _build_methods(
            parameter,
            gradient,
            momentum,
            compiled_step=compiled_step,
            seed=args.seed,
            block_size=args.block_size,
        )

        for name, function in methods.items():
            milliseconds = cuda_median_ms(
                function,
                warmup=args.warmup,
                repetitions=args.repetitions,
            )
            row = result_row(name, n, milliseconds, bytes_per_element=logical_bytes)
            row["parameter_dtype"] = args.parameter_dtype
            row["block_size"] = args.block_size if name.startswith("triton") else ""
            rows.append(row)

    write_rows(rows, args.output)


if __name__ == "__main__":
    main()
