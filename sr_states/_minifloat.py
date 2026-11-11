"""Unscaled E4M3/E5M2/FP16 rounding with a 24-bit SR probability grid."""

from __future__ import annotations

import torch

from ._hash import _u32_hash

# Mantissa bits, minimum normal exponent, maximum finite magnitude, saturate.
_FORMATS = {
    torch.float8_e4m3fn: (3, -6, 448.0, True),
    torch.float8_e5m2: (2, -14, 57344.0, True),
    torch.float16: (10, -14, 65504.0, False),
}


def _grid(a, mantissa, emin):
    bits = a.contiguous().view(torch.int32)
    exponent = ((bits >> 23) & 0xFF) - 127
    exponent = exponent.clamp_min(emin)
    # Powers of two are exact FP32 normals for all supported formats/inputs.
    ulp = ((exponent - mantissa + 127) << 23).view(torch.float32)
    reciprocal = ((127 - exponent + mantissa) << 23).view(torch.float32)
    scaled = a * reciprocal
    q = torch.floor(scaled)
    return q, scaled - q, ulp


def nearest_minifloat(x, *, dtype, seed, offset):
    mantissa, emin, maximum, saturate = _FORMATS[dtype]
    if not saturate:
        return x.to(dtype)
    # Round to an exact FP8 grid point before casting. Inductor can otherwise
    # lower the cast through FP16 and lose the side of an FP8 midpoint.
    a = x.abs().clamp_max(maximum)
    safe = torch.where(torch.isnan(a), torch.zeros_like(a), a)
    q, remainder, ulp = _grid(safe, mantissa, emin)
    up = (remainder > 0.5) | ((remainder == 0.5) & ((q.to(torch.int32) & 1) != 0))
    rounded = (q + up.to(torch.float32)) * ulp
    rounded = torch.copysign(rounded, x)
    rounded = torch.where(torch.isnan(x), x, rounded)
    return rounded.to(dtype)


def stochastic_minifloat(x, *, dtype, seed, offset):
    """SR of the magnitude, then restore the sign (including a zero result).

    R=floor(frac(a/ulp)*2**24); a uniform uint32 hash selects the upper grid
    point iff (hash >> 8) < R. Thus exact probabilities are unbiased whenever
    R is integral before flooring. Otherwise the magnitude bias is toward zero
    by less than ulp*2**-24. Saturation and FP16 overflow are excluded from that
    unbiasedness statement. NaN/Inf are selected explicitly after arithmetic.
    """
    mantissa, emin, maximum, saturate = _FORMATS[dtype]
    a = x.abs()
    if saturate:
        a = a.clamp_max(maximum)
    # Avoid inf-inf and NaN in the remainder; select original specials below.
    safe = torch.where(torch.isfinite(a), a, torch.zeros_like(a))
    q, remainder, ulp = _grid(safe, mantissa, emin)
    indices = torch.arange(x.numel(), dtype=torch.int64, device=x.device).reshape(x.shape)
    offset_tensor = torch.scalar_tensor(int(offset) % (1 << 32), dtype=torch.int64, device=x.device)
    k = _u32_hash(indices + offset_tensor, seed) >> 8
    threshold = torch.floor(remainder * 16777216.0)
    rounded = (q + (k < threshold).to(torch.float32)) * ulp
    rounded = torch.copysign(rounded, x)
    if not saturate:
        rounded = torch.where(torch.isinf(x), x, rounded)
    rounded = torch.where(torch.isnan(x), x, rounded)
    return rounded.to(dtype)


def minifloat_neighbors(x: torch.Tensor, *, dtype: torch.dtype):
    """Ordered target-dtype neighbors after the format's saturation policy.

    Exact values have identical neighbors. FP16 finite overflow is bracketed
    by max finite and infinity. NaN/Inf follow the format's cast policy.
    """
    if x.dtype != torch.float32:
        raise TypeError(f"expected torch.float32 input, got {x.dtype}")
    if dtype not in _FORMATS:
        raise ValueError("neighbors supports E4M3, E5M2, and FP16")
    mantissa, emin, maximum, saturate = _FORMATS[dtype]
    a = x.abs().clamp_max(maximum)
    safe = torch.where(torch.isfinite(a), a, torch.zeros_like(a))
    q, remainder, ulp = _grid(safe, mantissa, emin)
    near = q * ulp
    away = (q + (remainder != 0).float()) * ulp
    if not saturate:
        near = torch.where(x.abs() > maximum, maximum, near)
        away = torch.where(x.abs() > maximum, float("inf"), away)
        near = torch.where(torch.isinf(x), float("inf"), near)
    near = torch.copysign(near, x)
    away = torch.copysign(away, x)
    lower = torch.where(torch.signbit(x), away, near)
    upper = torch.where(torch.signbit(x), near, away)
    lower = torch.where(torch.isnan(x), x, lower)
    upper = torch.where(torch.isnan(x), x, upper)
    return lower.to(dtype), upper.to(dtype)
