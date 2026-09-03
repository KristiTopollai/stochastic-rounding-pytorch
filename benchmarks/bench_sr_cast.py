from __future__ import annotations

import argparse

import torch

from benchmarks._utils import cuda_median_ms, result_row, write_rows
from sr_states.compiled import compile_sr_cast
from sr_states.reference import stochastic_round_bf16
from sr_states.triton import is_triton_available, sr_cast_bf16


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[2**k for k in range(10, 28, 2)])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--output")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("This benchmark requires an NVIDIA CUDA device")
    if not is_triton_available():
        raise SystemExit("This benchmark requires Triton")

    compiled = compile_sr_cast(fullgraph=True)
    rows = []
    for n in args.sizes:
        x = torch.randn(n, device="cuda", dtype=torch.float32)
        methods = {
            "bf16_rn": lambda x=x: x.to(torch.bfloat16),
            "pytorch_sr": lambda x=x: stochastic_round_bf16(x, seed=7),
            "compiled_sr": lambda x=x: compiled(x, seed=7, offset=0),
            "triton_sr": lambda x=x: sr_cast_bf16(x, seed=7, block_size=args.block_size),
        }
        for name, function in methods.items():
            milliseconds = cuda_median_ms(
                function, warmup=args.warmup, repetitions=args.repetitions
            )
            rows.append(result_row(name, n, milliseconds, bytes_per_element=6))
    write_rows(rows, args.output)


if __name__ == "__main__":
    main()
