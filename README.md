# SR-States

FP32-to-BF16 stochastic rounding in PyTorch and Triton. The library provides
an eager reference, a `torch.compile` reference, and a single-pass Triton cast,
plus Triton round-to-nearest (NR, ties-to-even) for comparison.

## Install

Python 3.10+ and PyTorch 2.4+. The Triton backend requires Linux, an NVIDIA GPU,
and CUDA-enabled PyTorch.

```bash
pip install -e '.[triton]'
```

## Usage

```python
import torch
from sr_states.reference import stochastic_round_bf16
from sr_states.compiled import compile_sr_cast
from sr_states.triton import sr_cast_bf16, rn_cast_bf16

x = torch.randn(1_048_576, device="cuda", dtype=torch.float32)
compiled_cast = compile_sr_cast(fullgraph=True, dynamic=True)

y_eager = stochastic_round_bf16(x, seed=7)
y_compiled = compiled_cast(x, seed=7, offset=0)
y_triton = sr_cast_bf16(x, seed=7)
y_nr = rn_cast_bf16(x)
```

The [Triton casts](sr_states/triton/sr_cast.py) accept contiguous FP32 tensors
and allocate fresh BF16 outputs. SR uses a stateless counter hash keyed by
`seed` and `offset`; changing the block size preserves the random sequence.
NR uses the same allocation and launch path without RNG. NaNs, infinities,
signed zero, and exact BF16 values are supported.

## Tests and benchmarks

```bash
pip install -e '.[dev]'
python -m pytest
python -m benchmarks.bench_sr_cast --sizes 1048576 16777216 \
  --output results/raw/cast.csv
```

Tests cover rounding statistics, bit patterns, NR ties, counter wraparound,
and Triton/reference agreement. GPU tests skip when CUDA/Triton is unavailable.
The benchmark requires a GPU and checks all five methods before timing:
eager SR, compiled SR, Triton SR, Triton NR, and native PyTorch NR.

Each call allocates output, with RNG inside SR and compilation outside timing.
The CSV includes raw CUDA-event samples. These measure warm operator calls,
including host submission gaps; CUDA graphs are disabled.

## H200 results

16,777,216 elements; simple Triton cast with block size 256 and four warps.
PyTorch 2.11.0+cu128, Triton 3.6.0. [Recorded samples](results/h200-simple-cast.json).

| Method | Time (µs) |
| --- | ---: |
| Eager PyTorch SR | 2,236.56 |
| Compiled PyTorch SR | 120.94 |
| Simple Triton SR | 62.42 |
| Matched Triton NR | 61.97 |
| Native PyTorch NR | 32.22 |

Simple Triton SR is **35.87× faster than eager SR** and **1.94× faster than
compiled SR**, with **0.70% overhead over matched Triton NR** in this run.
Times are medians of three process medians, with 100 samples per measurement;
speedups and overheads use paired process ratios.
