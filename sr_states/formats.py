"""Elementwise optimizer-state storage formats, independent of optimizer math."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

import torch

from .rounding import validate_rounding


@dataclass(frozen=True)
class StateFormat:
    """One state's storage dtype and registered rounding rule (unscaled)."""

    dtype: torch.dtype
    rounding: str = "stochastic"

    def validate(self, backend: str = "reference") -> None:
        # Compiled storage traces the reference rule.
        validate_rounding(
            self.dtype, self.rounding, backend="reference" if backend == "compiled" else backend
        )

    def to_dict(self) -> dict:
        return {"dtype": self.dtype, "rounding": self.rounding}

    @classmethod
    def from_dict(cls, d: dict) -> StateFormat:
        if not isinstance(d, Mapping) or set(d) != {"dtype", "rounding"}:
            raise ValueError("a state format must contain exactly dtype and rounding")
        return cls(**d)


def force_rounding(formats, rounding):
    """Keep each state's dtype but use ``rounding``; ``None`` leaves formats unchanged."""
    if rounding is None:
        return tuple(formats)
    return tuple(replace(fmt, rounding=rounding) for fmt in formats)


def resolve_formats(
    names,
    formats=None,
    *,
    state_dtype=None,
    rounding=None,
    forced_rounding=None,
    backend="reference",
):
    """Normalize public formats or legacy shorthands to an ordered tuple.

    ``forced_rounding`` replaces every state's rule before validation, for
    optimizers whose class fixes the rule (for example round-to-nearest variants).
    """
    if formats is not None:
        if state_dtype is not None or rounding is not None:
            raise ValueError("state_format/formats cannot be combined with state_dtype or rounding")
        if isinstance(formats, StateFormat):
            result = (formats,) * len(names)
        elif isinstance(formats, Mapping):
            if set(formats) != set(names):
                raise ValueError(f"state_format keys must be exactly {names}")
            result = tuple(formats[name] for name in names)
        elif isinstance(formats, tuple) and len(formats) == len(names):
            result = formats
        else:
            raise TypeError("state_format must be a StateFormat or a dict keyed by state name")
    else:
        result = (
            StateFormat(
                torch.bfloat16 if state_dtype is None else state_dtype,
                "stochastic" if rounding is None else rounding,
            ),
        ) * len(names)
    for fmt in result:
        if not isinstance(fmt, StateFormat):
            raise TypeError("each state format must be a StateFormat")
    result = force_rounding(result, forced_rounding)
    for fmt in result:
        fmt.validate(backend)
    return result
