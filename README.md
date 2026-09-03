# SR-States

SR-States is an experimental implementation of stochastic rounding for
low-precision optimizer-state storage. I am using it to study whether momentum
and moment buffers can be stored in BF16 without systematically erasing small,
persistent updates.

Optimizer arithmetic is evaluated in FP32. State is converted only when it is
written back to storage:

```text
BF16 state -> FP32 optimizer update -> stochastic BF16 write-back
```

The repository provides:

- readable PyTorch implementations of stochastic FP32-to-BF16 rounding;
- SGDM and AdamW references with selectable FP32 or BF16 state;
- deterministic round-to-nearest baselines using the same update equations;
- functional update paths compatible with `torch.compile`;
- single-pass Triton stochastic-cast and fused momentum-SGD kernels;
- numerical, statistical, parity, and checkpoint-reproducibility tests.

## Why stochastic rounding?

For adjacent BF16 values $a \leq x \leq b$,

$$
Q_{\mathrm{SR}}(x) =
\begin{cases}
a & \text{with probability } (b-x)/(b-a), \\
b & \text{with probability } (x-a)/(b-a).
\end{cases}
$$

Away from exceptional floating-point cases,
$\mathbb{E}[Q_{\mathrm{SR}}(x)] = x$. A deterministic BF16 cast can repeatedly
discard an update smaller than one ULP. Stochastic rounding gives that update a
chance to change the stored value and preserves it in expectation.

This is a one-step numerical property, not a guarantee that an entire training
trajectory is unbiased. Training behavior still needs to be measured.

## Installation

The reference implementation requires Python 3.10+ and PyTorch 2.4+.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .
```

On a supported Linux system with an NVIDIA GPU, install the optional Triton
dependency with:

```bash
pip install -e '.[triton]'
```

## Optimizer usage

```python
import torch

from sr_states import AdamWReferenceSR

optimizer = AdamWReferenceSR(
    model.parameters(),
    lr=3e-4,
    state_dtype=torch.bfloat16,
    seed=2026,
)
```

The first- and second-moment updates are evaluated in FP32. Only their stored
representations are reduced to BF16. `SGDMReferenceSR` exposes the same state
precision policy for momentum SGD.

For controlled comparisons, use `state_dtype=torch.float32` to remove state
quantization or select `AdamWReferenceRN` / `SGDMReferenceRN` for deterministic
BF16 round-to-nearest storage.

## Casting APIs

The PyTorch reference accepts tensors on any PyTorch device and also accepts
non-contiguous inputs:

```python
from sr_states.reference import stochastic_round_bf16

state_bf16 = stochastic_round_bf16(state_fp32, seed=2026, offset=0)
```

The optional Triton path currently accepts contiguous CUDA FP32 tensors:

```python
from sr_states.triton import sr_cast_bf16

state_bf16 = sr_cast_bf16(state_fp32_cuda, seed=2026, offset=0)
```

Both paths use the same logical counter scheme. Once GPU validation is
available, equal `seed` and `offset` values are expected to produce bitwise
matching BF16 outputs.

The fused optimizer path updates the parameter and BF16 momentum buffer in one
Triton program:

```python
from sr_states.triton import SGDMTriton

optimizer = SGDMTriton(
    model.parameters(),
    lr=1e-2,
    momentum=0.9,
    rounding="stochastic",
    seed=2026,
)
```

This path currently requires contiguous CUDA parameters and gradients.

## Design notes

The cast works directly on the IEEE-754 representation of each FP32 input. Its
low 16 bits locate the input within a BF16 interval. A deterministic 32-bit
counter hash supplies a uniform 16-bit threshold, after which the discarded
bits are cleared in one operation.

NaNs, signed infinities, and signed zero are handled explicitly. Exact BF16
inputs remain unchanged. The implementation is stateless: random choices depend
on the seed and logical element index, not on thread-block shape or call order.

## Reproducibility

Random choices are generated from a stateless 32-bit counter hash keyed by a
seed and logical element offset. This makes the output independent of how a
tensor is partitioned and allows optimizer checkpoints to resume from the same
random position.

The generator is designed for reproducible numerical experiments. It is not
Philox-compatible and should not be treated as a general-purpose or
cryptographic random-number generator.

## Tests

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
pytest
```

CUDA-specific tests are marked separately and skip automatically when CUDA or
Triton is unavailable:

```bash
pytest -m cuda
```

The suite covers BF16 neighbor membership, empirical rounding probabilities,
sample-mean behavior, exceptional floating-point values, seed reproducibility,
partition invariance, optimizer parity, state memory, checkpoint restore, and
Triton/reference equivalence.

## Benchmark

`benchmarks/bench_sr_cast.py` compares native BF16 round-to-nearest, eager
PyTorch stochastic rounding, the compiled reference, and the Triton cast:

```bash
python -m benchmarks.bench_sr_cast \
  --sizes 1048576 4194304 16777216 \
  --warmup 20 \
  --repetitions 100 \
  --output results/sr_cast.csv
```

The benchmark reports median latency, elements per second, and effective
bandwidth. No performance numbers are included until they can be collected on
an available NVIDIA GPU.

The momentum-SGD benchmark compares the eager reference, `torch.compile`, and
the fused Triton update using independent input state for each method:

```bash
python -m benchmarks.bench_sgdm \
  --sizes 1048576 4194304 16777216 \
  --parameter-dtype bf16 \
  --block-size 256 \
  --output results/sgdm.csv
```

Reported bandwidth is logical algorithmic traffic: parameter, gradient, and
momentum reads plus parameter and momentum writes. It should be used to compare
implementations within the same dtype configuration, rather than as a direct
measurement of every byte moved by eager intermediate tensors.

## Limitations

- BF16 and FP32 optimizer state only;
- no sparse gradients, AMSGrad, differentiable optimizer, or distributed state;
- the Triton path has not yet been validated on an NVIDIA GPU;
- fused Triton optimizer support currently covers momentum SGD but not AdamW;
- statistical unbiasedness of one cast does not imply an unbiased optimization
  trajectory.
