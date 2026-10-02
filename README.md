# SR-States

Stochastic rounding for BF16 optimizer-state storage. SGDM stores its momentum
buffer in BF16; AdamW stores its first and second moments in BF16. Updates use
FP32 arithmetic and round the states only when saving them for the next step.
These buffers use half the memory of FP32 states. Parameter and gradient storage
are unchanged.

## Install

Python 3.10+ and PyTorch 2.5+. Triton requires Linux, an NVIDIA GPU, and
CUDA-enabled PyTorch. From a clone of this repository:

```bash
pip install -e '.[triton]'
```

## Usage

```python
import torch
from sr_states.triton import AdamWTriton, SGDMTriton

p = torch.nn.Parameter(torch.randn(1024, device="cuda"))
optimizer = AdamWTriton([p], lr=1e-3, seed=7)
# Momentum SGD: SGDMTriton([p], lr=1e-2, momentum=0.9, seed=7)
p.square().mean().backward()
optimizer.step()
```

Both optimizers default to stochastic BF16 states. `rounding="nearest"` selects
nearest-even (NR) rounding; `state_dtype=torch.float32` selects FP32 states.
Parameter updates use the FP32 moment intermediates before rounding. Checkpoints
preserve state precision, step counts, and RNG position.

Triton accepts contiguous CUDA FP32/BF16 parameters and dense gradients, with one
fused kernel per parameter tensor. SGDM implements classical momentum; AdamW
includes bias correction and decoupled weight decay. Nesterov, dampening, SGDM
weight decay, sparse gradients, and AMSGrad are not supported.

Eager optimizers are in `sr_states.optim`; compiled SR optimizers are in
`sr_states.compiled`. Standalone `sr_cast_bf16(x, seed=7, offset=0)` and
`rn_cast_bf16(x)` are in `sr_states.triton` and allocate BF16 outputs from contiguous
CUDA FP32 inputs. SR uses a reproducible 32-bit counter hash, independent of block
size. Optimizers advance counters across states and steps; counters wrap modulo
2³². Values beyond the finite BF16 range can round to infinity.

## Tests and benchmarks

```bash
pip install -e '.[dev]'
python -m pytest
python -m benchmarks.bench_optimizer_steps --output results/raw/optimizer-steps.csv
python -m benchmarks.bench_sgdm --output results/raw/sgdm.csv
python -m benchmarks.bench_adamw --output results/raw/adamw.csv
python -m benchmarks.bench_sr_cast --output results/raw/cast.csv
```

`bash scripts/run_state_validation.sh` runs the tests and three repeats of the
optimizer benchmarks, saving samples and environment details. The H200 run
passed **299 tests**; two tests requiring a second GPU were skipped.

`bench_optimizer_steps` times full `optimizer.step()` calls on diagonal quadratics
and MLP regression, with FP32 parameters/gradients and BF16 states. Compiled SR
compiles each tensor update; bookkeeping remains in Python. Gradients,
initialization, and compilation are excluded; steps and RNG counters advance normally.

CSVs contain synchronized wall, CUDA-event, and CPU-submission latency, plus
paired speedups. Wall latency includes the event wait; CUDA-event latency includes
host submission gaps. CUDA graphs are disabled.
Quadratic `--sizes` counts total parameters split across `--tensors` (default 1
and 32). In the fixed-input `bench_sgdm`/`bench_adamw` microbenchmarks, `--sizes`
is elements per tensor; inputs, counters, and step numbers reset outside timing.

## H200 results

PyTorch 2.11.0+cu128, Triton 3.6.0. Times are medians of three process medians
(100 samples each); speedups and overheads are medians of within-repeat ratios.

Full optimizer steps, one tensor with 16,777,216 FP32 parameters and BF16 states:

| Method | AdamW (µs, wall) | SGDM (µs, wall) |
| --- | ---: | ---: |
| Matched Triton NR | 145.51 | 122.31 |
| Eager SR | 5,033.19 | 2,494.40 |
| Compiled SR | 381.07 | 147.20 |
| Triton SR | 145.90 | 122.94 |

Triton SR is **2.61× / 1.20× faster than compiled SR** for AdamW / SGDM, with
**0.27% / 0.25%** measured overhead versus matched Triton NR. Across all ten
workloads, median SR overhead versus NR ranges from −0.13% to +1.83%; these differences
are not a precise isolated RNG cost. [Full results and reproduction](results/h200-states-20261002/README.md).
These are synthetic optimizer timings, not training-throughput or convergence results.

Standalone casts, 16,777,216 elements, allocating output on each call:

| Method | Time (µs, CUDA events) |
| --- | ---: |
| Native PyTorch NR | 32.22 |
| Matched Triton NR | 61.97 |
| Eager PyTorch SR | 2,236.56 |
| Compiled PyTorch SR | 120.94 |
| Triton SR | 62.42 |

The simple Triton SR cast is **35.87× faster than eager SR**, **1.94× faster than
compiled SR**, with **0.70%** measured overhead versus matched Triton NR.
[Recorded cast samples](results/h200-simple-cast.json). Both Triton casts use
block size 256 and four warps.
