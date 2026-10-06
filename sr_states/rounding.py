"""Named, backend-aware rounding routines shared by casts and every optimizer.

Only FP32 and BF16 routines are registered by default. Registrations are process
local: import a custom routine's registration module before loading its checkpoints.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from ._bf16 import stochastic_round_bf16


@dataclass(frozen=True)
class _Routine:
    reference: Callable
    # Built-ins use a lazy attribute name so CPU users need not import Triton.
    triton: Callable | str | None = None


_ROUTINES: dict[tuple[torch.dtype, str], _Routine] = {}


def register_rounding(
    name: str,
    dtype: torch.dtype,
    *,
    reference: Callable,
    triton: Callable | None = None,
) -> None:
    """Register a rounding rule for one storage dtype without changing optimizers.

    ``reference(x, *, dtype, seed, offset)`` receives FP32 values and returns a
    tensor of the requested dtype with the same shape and device. It must not
    mutate inputs. Compiled optimizers trace this same callable, which must be
    compatible with torch.compile when used there.

    Optional ``triton(value, counter, dtype)`` must be a @triton.jit function.
    It receives an FP32 block, uint32 logical counters (seed + offset + index),
    and the destination tl.dtype as a constexpr. Return already-rounded values
    for tl.store. Each element owns one counter; implementations needing more
    random words must derive them within that element's stream.

    Register before constructing optimizers or compiling updates. Names and
    dtypes are stored in checkpoints; callables are not. Replacing an existing
    registration is forbidden to prevent silently changing resumed computations.
    This interface covers elementwise, unscaled storage casts. Stateful/block
    quantizers needing additional metadata require a broader storage interface.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("rounding name must be a nonempty string")
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise ValueError("dtype must be a floating-point torch.dtype")
    if not callable(reference) or (triton is not None and not callable(triton)):
        raise TypeError("rounding implementations must be callable")
    key = (dtype, name)
    if key in _ROUTINES:
        raise ValueError(f"rounding {name!r} is already registered for {dtype}")
    _ROUTINES[key] = _Routine(reference, triton)


def validate_rounding(dtype: torch.dtype, rounding: str, *, backend: str = "reference") -> None:
    """Validate a dtype/rule/backend combination without loading GPU dependencies."""
    if backend not in {"reference", "triton"}:
        raise ValueError(f"unknown rounding backend: {backend!r}")
    if not isinstance(rounding, str) or (dtype, rounding) not in _ROUTINES:
        raise ValueError(f"no rounding {rounding!r} registered for state_dtype={dtype}")
    if backend == "triton" and _ROUTINES[dtype, rounding].triton is None:
        raise ValueError(f"rounding {rounding!r} for {dtype} has no Triton implementation")


def get_triton_rounding(dtype: torch.dtype, rounding: str):
    """Resolve a registered JIT function on the host, before kernel compilation."""
    validate_rounding(dtype, rounding, backend="triton")
    implementation = _ROUTINES[dtype, rounding].triton
    if isinstance(implementation, str):
        from .triton import rounding as builtins

        return getattr(builtins, implementation)
    return implementation


def cast(
    x: torch.Tensor,
    *,
    dtype: torch.dtype,
    rounding: str = "stochastic",
    seed: int = 0,
    offset: int = 0,
) -> torch.Tensor:
    """Apply a registered rounding routine to FP32 input, preserving shape/device.

    BF16 stochastic and nearest-even casts allocate outputs; FP32 built-ins
    are no-ops. BF16 exceptional-value and RNG behavior is unchanged. No scaling
    is performed. Custom routines define their own numerical rounding behavior.
    """
    if x.dtype != torch.float32:
        raise TypeError(f"expected torch.float32 input, got {x.dtype}")
    validate_rounding(dtype, rounding)
    result = _ROUTINES[dtype, rounding].reference(x, dtype=dtype, seed=seed, offset=offset)
    if not isinstance(result, torch.Tensor):
        raise TypeError("rounding routine must return a tensor")
    if result.dtype != dtype or result.shape != x.shape or result.device != x.device:
        raise ValueError(
            "rounding routine must preserve shape/device and return the requested dtype"
        )
    return result


def _nearest(x, *, dtype, seed, offset):
    return x.to(dtype)


def _stochastic_bf16(x, *, dtype, seed, offset):
    return stochastic_round_bf16(x, seed=seed, offset=offset)


_ROUTINES[torch.float32, "nearest"] = _Routine(_nearest, "identity")
_ROUTINES[torch.float32, "stochastic"] = _Routine(_nearest, "identity")
_ROUTINES[torch.bfloat16, "nearest"] = _Routine(_nearest, "nearest")
_ROUTINES[torch.bfloat16, "stochastic"] = _Routine(_stochastic_bf16, "stochastic_bf16")
