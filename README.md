# SR-States

I am exploring whether optimizer states can be stored in BF16 without losing
small, persistent updates. The optimizer math stays in FP32, while momentum
states are stochastically rounded when they are written back to BF16.

The repository currently includes:

- a bit-level FP32-to-BF16 stochastic-rounding implementation in PyTorch;
- SGDM and AdamW with BF16 or FP32 state storage;
- deterministic BF16 round-to-nearest variants for comparison;
- functional update paths that can be passed to `torch.compile`;
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

## Usage

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
representations are reduced to BF16.

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

The tests cover BF16 neighbor membership, empirical rounding probabilities,
sample-mean behavior, exceptional floating-point values, seed reproducibility,
partition invariance, optimizer parity, state memory, and checkpoint restore.

The package requires Python 3.10+ and PyTorch 2.4+.

## Limitations

- BF16 and FP32 optimizer state only;
- no sparse gradients, AMSGrad, differentiable optimizer, or distributed state;
- no GPU performance results are reported yet.
