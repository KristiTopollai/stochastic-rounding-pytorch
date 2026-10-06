# Stochastic Rounding for PyTorch

FP32-to-BF16 stochastic rounding, with standalone casts and BF16 optimizer-state
storage in PyTorch and Triton.

All optimizer backends use a shared rounding registry. Only FP32 and BF16 are
enabled by default; adding a future format or rounding rule does not require
editing the SGDM or AdamW update equations.

## Install

Python 3.10+ and PyTorch 2.5+. Triton requires Linux, an NVIDIA GPU, and
CUDA-enabled PyTorch. From a clone of this repository:

```bash
pip install -e '.[triton]'
```

## Usage

### Casting

The common interface selects storage dtype and rounding independently:

```python
import torch
from sr_states import cast
from sr_states.compiled import compile_cast

x = torch.tensor([1.003, -1.006], dtype=torch.float32)
y = cast(x, dtype=torch.bfloat16, rounding="stochastic", seed=7)
y_nearest = cast(x, dtype=torch.bfloat16, rounding="nearest")
compiled_cast = compile_cast(fullgraph=True)
y_compiled = compiled_cast(x, dtype=torch.bfloat16, rounding="stochastic", seed=7)
```

`sr_states.triton.cast` accepts the same dtype, rounding, seed, and offset options
for contiguous CUDA inputs, plus block-size and warp-count options. It uses the
same registered JIT callback as the fused optimizer kernels. The older BF16-only
cast helpers remain compatibility entry points.

```python
import torch
from sr_states.triton import sr_cast_bf16, rn_cast_bf16

x = torch.randn(1024, device="cuda", dtype=torch.float32)
y_sr = sr_cast_bf16(x, seed=7, offset=0)
y_nr = rn_cast_bf16(x)  # Nearest-even rounding
```

Both casts accept contiguous CUDA FP32 inputs and return new BF16 tensors with
the same shape and device, leaving the input unchanged. SR uses a reproducible
32-bit counter hash: reusing a seed and offset repeats the rounding choices.
For successive casts, advance `offset` by the previous input's `numel()`.
The sequence is independent of block size; counters wrap modulo 2³².
Values beyond the finite BF16 range can round to infinity.

The eager CPU/CUDA equivalent is `sr_states.reference.stochastic_round_bf16(x,
seed=7, offset=0)`. `sr_states.compiled.compile_sr_cast()` returns its compiled version.

### Optimizer states

SGDM stores its momentum buffer in BF16; AdamW stores its first and second moments
in BF16. These buffers use half the memory of FP32 states. Parameter and gradient
storage are unchanged.

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
preserve state precision, step counts, and RNG position. Optimizers advance the
rounding counters automatically across states and steps.

Triton accepts contiguous CUDA FP32/BF16 parameters and dense gradients, with one
fused kernel per parameter tensor. SGDM implements classical momentum; AdamW
includes bias correction and decoupled weight decay. Nesterov, dampening, SGDM
weight decay, sparse gradients, and AMSGrad are not supported.

Eager optimizers are in `sr_states.optim`; compiled SR optimizers are in
`sr_states.compiled`.

### Adding a rounding routine

Register a stable name and a storage dtype, then select them through the existing
`rounding` and `state_dtype` optimizer arguments. For example, a BF16 truncation
rule can be added without modifying any optimizer:

```python
from sr_states import AdamWReferenceSR, register_rounding


def toward_zero(x, *, dtype, seed, offset):
    bits = x.contiguous().view(torch.int32) & -65536
    return bits.view(torch.float32).to(dtype)


register_rounding("toward_zero", torch.bfloat16, reference=toward_zero)
p = torch.nn.Parameter(torch.ones(1024))
optimizer = AdamWReferenceSR([p], rounding="toward_zero", state_dtype=torch.bfloat16)
```

The reference routine receives FP32 proposals and must return the requested dtype
with unchanged shape/device, without mutating the input. Compiled optimizers trace
the same routine, so it must support `torch.compile` when used in that backend.
The parameter update uses the unrounded FP32 proposals, as before.

To use the rule in fused Triton optimizers, include a JIT implementation in its
registration. Register it once, with both implementations, instead of first
performing the reference-only registration above:

```python
import triton
import triton.language as tl


@triton.jit
def toward_zero_triton(value, counter, dtype: tl.constexpr):
    bits = value.to(tl.uint32, bitcast=True) & 0xFFFF0000
    return bits.to(tl.float32, bitcast=True).to(dtype)


register_rounding("toward_zero", torch.bfloat16, reference=toward_zero, triton=toward_zero_triton)
```

The JIT routine takes an FP32 block, uint32 counters (`seed + offset + index`,
modulo 2³²), and a destination `tl.dtype`. It must return already-rounded values
for storage. AdamW gives its two moments distinct consecutive counter ranges.
The callback is a compile-time kernel argument; it does not add a separate GPU
launch. A missing Triton implementation raises an error rather than substituting
another rounding rule.

Registrations are process-local and cannot overwrite existing entries. Register
custom routines before constructing/compiling optimizers and before restoring
their checkpoints. Checkpoints store the rule name, dtype, seed, and offset, not
callables. Parameter-group overrides and checkpoint rules are validated against
the selected backend. Existing checkpoints and built-in behavior are preserved.

This extension point handles elementwise, unscaled casts to floating-point
storage. A future scaled or blockwise quantizer with additional state will need
an explicit metadata/storage extension. No FP8 routine is included in this change.

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

Linux CI also compiles the standalone cast and both fused optimizer kernels for
Hopper with built-in and custom callbacks, without a GPU driver. These checks
validate JIT integration; GPU execution tests remain separate and require CUDA.

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
