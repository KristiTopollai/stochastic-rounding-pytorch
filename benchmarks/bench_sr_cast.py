from __future__ import annotations

import argparse
import random
import sys
from functools import partial

import torch

from benchmarks._utils import (
    cuda_samples_ms,
    nonnegative_int,
    positive_int,
    timing_row,
    write_rows,
)
from benchmarks.cast_checks import prepare_cases, prepare_rn_cases, validate_cast
from sr_states.compiled import compile_sr_cast
from sr_states.reference import stochastic_round_bf16
from sr_states.triton import is_triton_available, rn_cast_bf16, sr_cast_bf16


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sizes", nargs="+", type=positive_int, default=[2**k for k in range(10, 25, 2)]
    )
    parser.add_argument("--warmup", type=nonnegative_int, default=20)
    parser.add_argument("--repetitions", type=positive_int, default=100)
    parser.add_argument("--block-size", type=int, choices=(128, 256, 512, 1024), default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--order-seed", type=int, default=0)
    parser.add_argument("--output")
    return parser.parse_args(argv)


def build_methods(x, *, compiled, seed, block_size):
    return {
        "bf16_rn": lambda: x.to(torch.bfloat16),
        "triton_rn": partial(rn_cast_bf16, x, block_size=block_size),
        "pytorch_sr": partial(stochastic_round_bf16, x, seed=seed),
        "compiled_sr": partial(compiled, x, seed=seed, offset=0),
        "triton_sr": partial(sr_cast_bf16, x, seed=seed, block_size=block_size),
    }


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("This benchmark requires an NVIDIA CUDA device")
    if not is_triton_available():
        raise SystemExit("This benchmark requires Triton")

    compiled = compile_sr_cast(fullgraph=True, dynamic=True, options={"triton.cudagraphs": False})
    cases, rn_cases = prepare_cases("cuda"), prepare_rn_cases("cuda")
    for function in (
        stochastic_round_bf16,
        compiled,
        partial(sr_cast_bf16, block_size=args.block_size),
    ):
        validate_cast(function, cases)
    validate_cast(lambda x, **_: x.to(torch.bfloat16), rn_cases)
    validate_cast(lambda x, **_: rn_cast_bf16(x, block_size=args.block_size), rn_cases)
    del cases, rn_cases
    print("SR and NR correctness checks passed", file=sys.stderr, flush=True)
    torch.manual_seed(args.seed)
    order_rng = random.Random(args.order_seed)
    rows = []
    for n in args.sizes:
        x = torch.randn(n, device="cuda", dtype=torch.float32)
        methods = build_methods(x, compiled=compiled, seed=args.seed, block_size=args.block_size)
        ordered = list(methods.items())
        order_rng.shuffle(ordered)
        for name, function in ordered:
            print(f"cast: n={n} {name}", file=sys.stderr, flush=True)
            samples = cuda_samples_ms(function, warmup=args.warmup, repetitions=args.repetitions)
            row = timing_row(name, n, samples, bytes_per_element=6)
            row.update(
                seed=args.seed,
                order_seed=args.order_seed,
                warmup=args.warmup,
                repetitions=args.repetitions,
            )
            row["block_size"] = args.block_size if name.startswith("triton") else ""
            rows.append(row)
            if args.output:
                write_rows(rows, args.output)
    if not args.output:
        write_rows(rows, None)


if __name__ == "__main__":
    main()
