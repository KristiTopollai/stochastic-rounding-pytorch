# Stochastic Rounding for PyTorch

Low-precision optimizer states and standalone casts, with PyTorch, `torch.compile`,
and fused Triton backends. Supports FP32, BF16, FP16, and FP8 E4M3/E5M2 storage.

## Install

Requires Python 3.10+ and PyTorch 2.5+. From a clone:

```bash
pip install -e .              # PyTorch and compiled backends
pip install -e '.[triton]'    # Triton: Linux + NVIDIA GPU + CUDA-enabled PyTorch
```

## Optimizers

```python
import torch
from sr_states.triton import AdamWTriton

p = torch.nn.Parameter(torch.randn(1024, device="cuda"))
optimizer = AdamWTriton([p], lr=3e-4, seed=7)

loss = p.square().mean()
loss.backward()
optimizer.step()
optimizer.zero_grad(set_to_none=True)
```

Use `model.parameters()` in place of `[p]` for a model. By default, states use
BF16 stochastic rounding; optimizer arithmetic uses FP32.

| Backend / import module | AdamW | Momentum SGD |
| --- | --- | --- |
| PyTorch: `sr_states` | `AdamWReferenceSR` | `SGDMReferenceSR` |
| Compiled: `sr_states.compiled` | `AdamWCompiledSR` | `SGDMCompiledSR` |
| Triton: `sr_states.triton` | `AdamWTriton` | `SGDMTriton` |

Triton expects contiguous CUDA FP32/BF16 parameters and dense gradients.

### Choose state storage

Set `state_dtype=torch.float16` to change storage and `rounding="nearest"` for
deterministic rounding. To experiment with FP8 first-moment storage while keeping
the second moment in BF16:

```python
from sr_states import StateFormat

optimizer = AdamWTriton(
    [p],
    lr=3e-4,
    state_format={
        "exp_avg": StateFormat(torch.float8_e4m3fn),
        "exp_avg_sq": StateFormat(torch.bfloat16),
    },
    seed=7,
)
```

A single `state_format=StateFormat(torch.float16)` applies to every state.
Use either `state_format` or the `state_dtype`/`rounding` shorthand.
Triton FP8 requires **sm_89+**. BF16 remains the default for all states.


## Standalone casts

```python
import torch
from sr_states import cast

x = torch.tensor([1.003, -1.006], dtype=torch.float32)
y = cast(x, dtype=torch.bfloat16, seed=7)
y_nearest = cast(x, dtype=torch.bfloat16, rounding="nearest")
```

For repeated stochastic casts, advance `offset` by the previous input's `numel()`.
Use `sr_states.triton.cast` for contiguous CUDA inputs or
`sr_states.compiled.compile_cast()` for a compiled cast.

### Cast policy

FP8 E4M3 (`torch.float8_e4m3fn`) and E5M2 (`torch.float8_e5m2`) casts saturate
finite overflow and infinities to signed maximum finite values, preserve NaN,
and preserve signed zero. Their smallest positive subnormals are `2**-9` and
`2**-16`, respectively; values below those scales may round to zero. No tensor
scaling or block scaling is applied. FP16 and BF16 retain IEEE overflow/Inf
behavior. `rounding="nearest"` means round to nearest, ties to even.

FP16/FP8 stochastic rounding uses a 24-bit probability grid. Away from overflow
and saturation, probability quantization introduces a magnitude bias toward zero
smaller than one target spacing times `2**-24`. BF16 retains its original
counter-based stochastic stream; FP32 built-ins are identity casts.

## Extend storage and optimizer math

Register an elementwise storage rule without changing optimizer implementations:

```python
from sr_states import AdamWReferenceSR, StateFormat, register_rounding


def my_nearest(x, *, dtype, seed, offset):
    return x.to(dtype)


register_rounding("my_nearest", torch.bfloat16, reference=my_nearest)
optimizer = AdamWReferenceSR([p], state_format=StateFormat(torch.bfloat16, "my_nearest"))
```

The reference callable receives FP32 input and must return the requested dtype
with the same shape and device, without mutating the input. Compiled optimizers
trace this callable, so it must support `torch.compile`. For fused Triton use,
also pass `triton=` with a `@triton.jit` callback accepting
`(value, counter, dtype: tl.constexpr)` and returning the stored value.
Registrations are process-local and cannot replace existing names. Register
custom rules before constructing optimizers, compiling, or loading checkpoints.

To add optimizer math, subclass [StateOptimizer](sr_states/optim/_state.py),
declare `_state_names`, and implement
`_compute(p32, g32, states32, group, step)` returning the updated FP32 parameter
and a tuple of FP32 state proposals. The base handles state allocation and
rounding, parameter groups, counters, missing gradients, and checkpoints. See the
[three-state example](tests/test_state_formats.py). Fused backends can override
`_step_parameter`. Scaled or blockwise quantizers need an additional storage and
metadata interface; `register_rounding` currently covers unscaled elementwise casts.

## Tests

```bash
pip install -e '.[dev]'
python -m pytest
```

On a CUDA machine with Triton installed, capture the baseline and run GPU checks:

```bash
bash scripts/run_state_validation.sh results/raw/fp8-validation --tests-only
```
