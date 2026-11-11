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


def validate_fp8_device(device, *dtypes):
    """Reject unsupported native FP8 conversion before Triton can abort."""
    import torch

    if any(dtype in (torch.float8_e4m3fn, torch.float8_e5m2) for dtype in dtypes):
        major, minor = torch.cuda.get_device_capability(device)
        if major * 10 + minor < 89:
            raise ValueError("FP8 Triton storage requires an NVIDIA GPU with sm_89 or newer")
