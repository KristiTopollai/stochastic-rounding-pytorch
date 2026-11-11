"""Record the pre-refactor optimizer contract; run from public main 393c527.

CPU: python scripts/record_optimizer_golden.py --output tests/golden/optimizer_v01.pt
GPU: python scripts/record_optimizer_golden.py --device cuda --output GPU_OUTPUT.pt
The GPU output must be captured from the baseline checkout, before the refactor.
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from sr_states import AdamWReferenceSR, SGDMReferenceSR
from sr_states.compiled import AdamWCompiledSR, SGDMCompiledSR


def collect(device="cpu"):
    classes = [SGDMReferenceSR, AdamWReferenceSR, SGDMCompiledSR, AdamWCompiledSR]
    if device == "cuda":
        from sr_states.triton import AdamWTriton, SGDMTriton

        classes += [SGDMTriton, AdamWTriton]
    cases = {}
    for cls in classes:
        for policy, dtype, rounding in [
            ("fp32", torch.float32, "nearest"),
            ("bf16_rn", torch.bfloat16, "nearest"),
            ("bf16_sr", torch.bfloat16, "stochastic"),
        ]:
            for pdtype in (torch.float32, torch.bfloat16):
                torch._dynamo.reset()
                params = [
                    torch.nn.Parameter(torch.linspace(-1, 1, n, device=device).to(pdtype))
                    for n in (17, 9, 5)
                ]
                options = (
                    {"compile_kwargs": {"backend": "eager"}} if "Compiled" in cls.__name__ else {}
                )
                optimizer = cls(
                    [{"params": params[:2]}, {"params": params[2:], "lr": 0.03}],
                    lr=0.01,
                    state_dtype=dtype,
                    rounding=rounding,
                    seed=2**32 - 13,
                    **options,
                )
                for group in optimizer.param_groups:
                    group["sr_offset"] = 2**32 - 23
                for step in range(7):
                    for i, p in enumerate(params):
                        p.grad = (
                            None
                            if (step + i) % 4 == 0
                            else torch.sin(
                                torch.arange(p.numel(), device=device).float() * 0.137 + step + i
                            ).to(pdtype)
                        )
                    optimizer.step()
                key = f"{cls.__name__}/{policy}/{pdtype}"
                cases[key] = {
                    "parameters": [p.detach().contiguous().view(torch.uint8).cpu() for p in params],
                    "states": [
                        {
                            name: value.contiguous().view(torch.uint8).cpu()
                            if isinstance(value, torch.Tensor)
                            else value
                            for name, value in optimizer.state[p].items()
                        }
                        for p in params
                    ],
                    "offsets": [g["sr_offset"] for g in optimizer.param_groups],
                }
    return cases


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", default=None)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite golden file: {args.output}")
    torch.save(
        {
            "commit": args.source_commit
            or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "torch": str(torch.__version__),
            "device": args.device,
            "cases": collect(args.device),
        },
        args.output,
    )
