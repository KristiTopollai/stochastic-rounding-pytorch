"""Matched in-place optimizer updates with BF16-SR, BF16-NR, and FP32 states."""

from __future__ import annotations

import argparse
import platform
import random
import sys
from dataclasses import dataclass

import torch

from benchmarks._utils import cuda_samples_ms, nonnegative_int, positive_int, timing_row, write_rows
from sr_states.compiled import compile_adamw_step_, compile_sgdm_step_
from sr_states.reference.adamw import adamw_step_
from sr_states.reference.sgd import sgdm_step_
from sr_states.triton import is_triton_available, triton_adamw_step_, triton_sgdm_step_

POLICIES = {
    "bf16_sr": (torch.bfloat16, "stochastic"),
    "bf16_nr": (torch.bfloat16, "nearest"),
    "fp32": (torch.float32, "nearest"),
}


def parse_args(argv=None, *, optimizer=None):
    parser = argparse.ArgumentParser(description=__doc__)
    if optimizer is None:
        parser.add_argument("--optimizer", choices=("sgdm", "adamw"), required=True)
    else:
        parser.set_defaults(optimizer=optimizer)
    parser.add_argument(
        "--sizes", nargs="+", type=positive_int, default=[1024, 65536, 1048576, 16777216]
    )
    parser.add_argument(
        "--tensors", type=positive_int, default=1, help="number of tensors of each size"
    )
    parser.add_argument("--parameter-dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--warmup", type=nonnegative_int, default=20)
    parser.add_argument("--repetitions", type=positive_int, default=100)
    parser.add_argument("--block-size", type=int, choices=(128, 256, 512, 1024), default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--order-seed", type=int, default=0)
    parser.add_argument(
        "--step", type=positive_int, default=100, help="fixed AdamW bias-correction step"
    )
    parser.add_argument("--output")
    return parser.parse_args(argv)


def logical_bytes_per_element(optimizer, parameter_dtype, state_dtype):
    states = 2 if optimizer == "adamw" else 1
    return (
        3 * torch.empty((), dtype=parameter_dtype).element_size()
        + 2 * states * torch.empty((), dtype=state_dtype).element_size()
    )


def update_options(optimizer, *, seed, step=100):
    if optimizer == "sgdm":
        return {"lr": 1e-3, "momentum_factor": 0.9, "seed": seed}
    return {
        "lr": 1e-3,
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8,
        "weight_decay": 1e-2,
        "step": step,
        "seed": seed,
    }


def prepare_inputs(optimizer, n, *, count, dtype, device):
    inputs = []
    for _ in range(count):
        parameter = torch.randn(n, device=device, dtype=dtype)
        gradient = torch.randn_like(parameter) * 0.1
        momentum = torch.randn(n, device=device) * 0.01
        tensors = [parameter, gradient, momentum]
        if optimizer == "adamw":
            tensors.append(torch.rand(n, device=device) * 0.01 + 1e-4)
        inputs.append(tuple(tensors))
    return inputs


@dataclass
class UpdateCase:
    initial: list
    buffers: list
    function: object
    reference: object
    options: dict
    state_dtype: torch.dtype
    rounding: str
    triton: bool
    block_size: int

    @torch.no_grad()
    def reset(self):
        for tensors, initial in zip(self.buffers, self.initial):
            for target, source in zip(tensors, initial):
                target.copy_(source)

    @torch.no_grad()
    def run(self):
        offset = 0
        for tensors in self.buffers:
            kwargs = {**self.options, "rounding": self.rounding, "offset": offset}
            if self.triton:
                kwargs["block_size"] = self.block_size
            else:
                kwargs["state_dtype"] = self.state_dtype
            self.function(*tensors, **kwargs)
            offset += (len(tensors) - 2) * tensors[0].numel()

    @torch.no_grad()
    def validate(self):
        self.reset()
        self.run()
        offset = 0
        for actual, initial in zip(self.buffers, self.initial):
            expected = [tensor.clone() for tensor in initial]
            self.reference(
                *expected,
                **self.options,
                rounding=self.rounding,
                state_dtype=self.state_dtype,
                offset=offset,
            )
            for index, (result, target) in enumerate(zip(actual, expected)):
                if index == 1:
                    torch.testing.assert_close(result, target, rtol=0, atol=0)
                else:
                    tolerance = 8e-3 if result.dtype == torch.bfloat16 else 1e-5
                    torch.testing.assert_close(result, target, rtol=tolerance, atol=1e-6)
            offset += (len(actual) - 2) * actual[0].numel()
        self.reset()


def build_case(
    inputs, *, optimizer, backend, policy, compiled=None, seed=7, step=100, block_size=256
):
    state_dtype, rounding = POLICIES[policy]
    reference = adamw_step_ if optimizer == "adamw" else sgdm_step_
    if backend == "triton":
        function = triton_adamw_step_ if optimizer == "adamw" else triton_sgdm_step_
    elif backend == "compiled":
        function = compiled
    else:
        function = reference
    initial = [
        tuple(
            tensor.clone() if i < 2 else tensor.to(state_dtype).clone()
            for i, tensor in enumerate(row)
        )
        for row in inputs
    ]
    buffers = [tuple(tensor.clone() for tensor in row) for row in initial]
    return UpdateCase(
        initial,
        buffers,
        function,
        reference,
        update_options(optimizer, seed=seed, step=step),
        state_dtype,
        rounding,
        backend == "triton",
        block_size,
    )


def main(argv=None, *, optimizer=None):
    args = parse_args(argv, optimizer=optimizer)
    if not is_triton_available():
        raise SystemExit("This benchmark requires an NVIDIA CUDA device and Triton")
    import triton

    compile_function = compile_adamw_step_ if args.optimizer == "adamw" else compile_sgdm_step_
    compiled = compile_function(fullgraph=True, dynamic=True, options={"triton.cudagraphs": False})
    dtype = torch.float32 if args.parameter_dtype == "fp32" else torch.bfloat16
    torch.manual_seed(args.seed)
    order = random.Random(args.order_seed)
    rows = []
    for n in args.sizes:
        inputs = prepare_inputs(args.optimizer, n, count=args.tensors, dtype=dtype, device="cuda")
        methods = [
            (backend, policy) for backend in ("eager", "compiled", "triton") for policy in POLICIES
        ]
        order.shuffle(methods)
        for backend, policy in methods:
            case = build_case(
                inputs,
                optimizer=args.optimizer,
                backend=backend,
                policy=policy,
                compiled=compiled,
                seed=args.seed,
                step=args.step,
                block_size=args.block_size,
            )
            case.validate()
            print(
                f"{args.optimizer}: {args.tensors} x {n} {backend}_{policy}",
                file=sys.stderr,
                flush=True,
            )
            samples = cuda_samples_ms(
                case.run,
                warmup=args.warmup,
                repetitions=args.repetitions,
                before_call=case.reset,
            )
            row = timing_row(
                f"{backend}_{policy}",
                n * args.tensors,
                samples,
                logical_bytes_per_element(args.optimizer, dtype, case.state_dtype),
            )
            row.update(
                optimizer=args.optimizer,
                tensor_elements=n,
                tensors=args.tensors,
                parameter_dtype=args.parameter_dtype,
                state_dtype=str(case.state_dtype),
                rounding=case.rounding,
                step=args.step if args.optimizer == "adamw" else "",
                block_size=args.block_size if backend == "triton" else "",
                seed=args.seed,
                order_seed=args.order_seed,
                warmup=args.warmup,
                repetitions=args.repetitions,
                correctness="passed",
                reset="outside_timing",
                gpu=torch.cuda.get_device_name(),
                torch_version=torch.__version__,
                triton_version=triton.__version__,
                python_version=platform.python_version(),
            )
            rows.append(row)
            if args.output:
                write_rows(rows, args.output)
            del case
    if not args.output:
        write_rows(rows, None)


if __name__ == "__main__":
    main()
