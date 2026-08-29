# SR-States

**Stochastic-rounding optimizers for low-precision state storage.**

SR-States is a reference-to-GPU implementation study of optimizer states that
are computed in FP32, stored in BF16, and stochastically rounded on writeback.
The repository deliberately preserves each implementation stage so numerical
correctness and performance changes can be compared rather than assumed.

The current public milestone contains:

- **Stage A:** readable PyTorch stochastic rounding, SGDM, and AdamW references;
- **Stage B:** `torch.compile` entry points for the same functional updates;
- deterministic BF16 round-to-nearest and FP32-state control variants;
- statistical, edge-case, parity, and checkpoint-reproducibility tests.

Triton kernels are intentionally reserved for the next development milestone.

## Example

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

Optimizer arithmetic is performed in FP32. With BF16 state, the updated first
and second moments are stochastically rounded before storage.

## Why stochastic rounding?

For adjacent representable values $a \leq x \leq b$, stochastic rounding uses

$$
Q_{\mathrm{SR}}(x) =
\begin{cases}
a & \text{with probability } (b-x)/(b-a), \\
b & \text{with probability } (x-a)/(b-a).
\end{cases}
$$

Away from exceptional floating-point cases,
$\mathbb{E}[Q_{\mathrm{SR}}(x)] = x$. This prevents deterministic rounding from
systematically discarding persistent optimizer-state changes smaller than one
BF16 ULP. Unbiased one-step rounding does **not** imply an unbiased training
trajectory; training-quality claims require experiments.

The implementation uses a stateless 32-bit counter hash keyed by seed and
logical element offset. That makes random choices reproducible across tensor
partitioning and optimizer checkpoint restoration.

## Install and verify

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
pytest
```

The package requires Python 3.10+ and PyTorch 2.4+.

## Repository structure

```text
sr_states/reference/   readable rounding and functional optimizer updates
sr_states/optim/       PyTorch Optimizer wrappers
sr_states/compiled/    torch.compile baselines
tests/                 numerical, parity, and reproducibility tests
```

## Current limitations

- BF16 and FP32 optimizer state only;
- no sparse gradients, AMSGrad, differentiable optimizer, or distributed state;
- the counter hash is reproducible but not Philox-compatible or cryptographic;
- no GPU performance claim is made at this milestone.

## Roadmap

The next stages add a Triton BF16 stochastic-rounding cast primitive and fused
Triton momentum SGD. Fused Triton AdamW, CUDA custom operators, profiling, and
end-to-end transformer experiments follow after those foundations are verified.

