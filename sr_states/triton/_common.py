"""Shared Triton availability checks."""

from __future__ import annotations

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised on non-Triton platforms
    triton = None
    tl = None


def require_triton() -> None:
    if triton is None:
        raise RuntimeError(
            "Triton is not installed. Install sr-states[triton] on a supported Linux/NVIDIA system."
        )
