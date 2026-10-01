# SR-States

Stochastic rounding for BF16 optimizer-state storage. SGDM keeps its momentum
buffer in BF16; AdamW keeps its first and second moments in BF16. Updates use
FP32 arithmetic, and the moments are rounded only when saved for the next step.
Parameter and gradient storage are unchanged. These state tensors use half the
memory of FP32 states.

The library includes eager PyTorch references, `torch.compile` references, and
fused Triton updates, plus standalone SR and nearest-even (NR) casts.

## Install

Python 3.10+ and PyTorch 2.4+. Triton requires Linux, an NVIDIA GPU, and
CUDA-enabled PyTorch.

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
BF16 NR; `state_dtype=torch.float32` selects full-precision states. Parameter
updates use the FP32 moment intermediates before state rounding. Checkpoints
preserve state precision, step counts, and RNG position.

Triton accepts contiguous CUDA FP32/BF16 parameters and dense gradients. SGDM
implements classical momentum without weight decay, dampening, or Nesterov.
AdamW supports bias correction, epsilon, and decoupled weight decay; AMSGrad is
not implemented. Each parameter tensor uses one fused kernel launch.

Standalone casts are available as `sr_cast_bf16(x, seed=7, offset=0)` and
`rn_cast_bf16(x)` from `sr_states.triton`, for contiguous CUDA FP32 inputs.
They allocate BF16 outputs. SR uses a stateless 32-bit counter hash; identical
seed/offset pairs repeat the rounding choices. Optimizers advance the counter
across state tensors and steps. Changing the block size preserves the sequence.

## Tests and benchmarks

```bash
pip install -e '.[dev]'
python -m pytest
python -m benchmarks.bench_sgdm --output results/raw/sgdm.csv
python -m benchmarks.bench_adamw --output results/raw/adamw.csv
python -m benchmarks.bench_sr_cast --output results/raw/cast.csv
```

GPU tests cover multi-step state updates, state-only rounding, checkpoint
restoration, block sizes, and compiled references; they skip without CUDA/Triton.
`bash scripts/run_state_validation.sh` runs the suite and three benchmark repeats
for single tensors and groups of 32 tensors, saving logs and environment details.

Optimizer benchmarks compare eager, compiled, and Triton updates with BF16 SR,
BF16 NR, and FP32 states. Every method updates preallocated buffers in place;
compiled paths include state write-back. Inputs, RNG offsets, and AdamW's step
are reset outside timing for each sample. `--tensors` selects the tensor count;
`--sizes` gives elements per tensor. These are fixed-input update microbenchmarks.
Compilation is excluded, CUDA graphs are disabled, and raw CUDA-event samples
include host submission gaps. Optimizer GPU results are pending.

## Standalone cast results

H200, 16,777,216 elements, simple Triton cast (block 256, four warps).
PyTorch 2.11.0+cu128, Triton 3.6.0. [Recorded samples](results/h200-simple-cast.json).

| Method | Time (µs) |
| --- | ---: |
| Eager PyTorch SR | 2,236.56 |
| Compiled PyTorch SR | 120.94 |
| Triton SR | 62.42 |
| Matched Triton NR | 61.97 |
| Native PyTorch NR | 32.22 |

SR is **35.87× faster than eager SR**, **1.94× faster than compiled SR**, and has
**0.70% overhead over matched Triton NR** in this cast benchmark. Each call
allocates output and includes RNG for SR. Times are medians of three process
medians (100 samples each); ratios use paired processes. These are cast timings,
not optimizer timings.
