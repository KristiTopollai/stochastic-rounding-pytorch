"""Shared Triton availability checks."""

from __future__ import annotations

from functools import cache

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


@cache
def _device_capability(device):
    import torch

    return torch.cuda.get_device_capability(device)


def validate_fp8_device(device, *dtypes):
    """Reject unsupported native FP8 conversion before Triton can abort."""
    import torch

    if any(dtype in (torch.float8_e4m3fn, torch.float8_e5m2) for dtype in dtypes):
        major, minor = _device_capability(device)
        if major * 10 + minor < 89:
            raise ValueError("FP8 Triton storage requires an NVIDIA GPU with sm_89 or newer")


def validate_update_device(device, *dtypes):
    if device.type != "cuda":
        raise ValueError("fused state updates require CUDA tensors")
    validate_fp8_device(device, *dtypes)


if triton is not None:

    @triton.jit
    def _chunk_info(meta, chunks, BLOCK_SIZE: tl.constexpr, CHUNK_TILES: tl.constexpr):
        chunk = tl.program_id(0).to(tl.int64)
        row = tl.load(chunks + chunk).to(tl.int64)
        numel = tl.load(meta + 4 * row)
        first_chunk = tl.load(meta + 4 * row + 1)
        param_index = tl.load(meta + 4 * row + 2)
        shard_start = tl.load(meta + 4 * row + 3)
        start = (chunk - first_chunk) * (BLOCK_SIZE * CHUNK_TILES)
        if CHUNK_TILES == 1:
            tiles = 1
        else:
            tiles = tl.minimum(CHUNK_TILES, tl.cdiv(numel - start, BLOCK_SIZE))
        return row, numel, param_index, shard_start, start, tiles
