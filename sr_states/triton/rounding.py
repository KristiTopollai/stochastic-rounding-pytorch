"""Built-in JIT rounding callbacks; optimizer kernels contain no format logic."""

from ._common import tl, triton

if triton is not None:
    from .sr_cast import _stochastic_bf16_value, _u32_hash

    @triton.jit
    def identity(value, counter, dtype: tl.constexpr):
        return value

    @triton.jit
    def nearest(value, counter, dtype: tl.constexpr):
        # Some Triton versions lower FP32 -> FP8 through FP16 with truncation,
        # losing which side of an FP8 midpoint the original value occupied.
        # Use the native FP32 conversion; both packed lanes receive the value.
        if dtype == tl.float8e4nv:
            packed = tl.inline_asm_elementwise(
                "cvt.rn.satfinite.e4m3x2.f32 $0, $1, $1;",
                constraints="=h,f",
                args=[value],
                dtype=tl.uint16,
                is_pure=True,
                pack=1,
            )
            return packed.to(tl.uint8).to(dtype, bitcast=True)
        elif dtype == tl.float8e5:
            packed = tl.inline_asm_elementwise(
                "cvt.rn.satfinite.e5m2x2.f32 $0, $1, $1;",
                constraints="=h,f",
                args=[value],
                dtype=tl.uint16,
                is_pure=True,
                pack=1,
            )
            return packed.to(tl.uint8).to(dtype, bitcast=True)
        else:
            return value.to(dtype, fp_downcast_rounding="rtne")

    @triton.jit
    def stochastic_bf16(value, counter, dtype: tl.constexpr):
        return _stochastic_bf16_value(value, counter).to(dtype)

    @triton.jit
    def _minifloat_constants(dtype: tl.constexpr):
        if dtype == tl.float8e4nv:
            return 3, -6, 448.0, True
        elif dtype == tl.float8e5:
            return 2, -14, 57344.0, True
        else:
            tl.static_assert(dtype == tl.float16, "unsupported minifloat dtype")
            return 10, -14, 65504.0, False

    @triton.jit
    def nearest_minifloat(value, counter, dtype: tl.constexpr):
        _M, _EMIN, MAX, SATURATE = _minifloat_constants(dtype)
        if SATURATE:
            clamped = tl.minimum(tl.maximum(value, -MAX), MAX)
            value = tl.where(value != value, value, clamped)  # noqa: PLR0124
        return nearest(value, counter, dtype)

    @triton.jit
    def stochastic_minifloat(value, counter, dtype: tl.constexpr):
        M, EMIN, MAX, SATURATE = _minifloat_constants(dtype)
        a = tl.abs(value)
        if SATURATE:
            a = tl.minimum(a, MAX)
        safe = tl.where((a == a) & (a != float("inf")), a, 0.0)  # noqa: PLR0124
        bits = safe.to(tl.int32, bitcast=True)
        exponent = tl.maximum(((bits >> 23) & 0xFF) - 127, EMIN)
        ulp = ((exponent - M + 127) << 23).to(tl.float32, bitcast=True)
        reciprocal = ((127 - exponent + M) << 23).to(tl.float32, bitcast=True)
        scaled = safe * reciprocal
        q = tl.floor(scaled)
        remainder = scaled - q
        threshold = tl.floor(remainder * 16777216.0)
        k = _u32_hash(counter) >> 8
        rounded = (q + (k.to(tl.float32) < threshold).to(tl.float32)) * ulp
        sign = value.to(tl.uint32, bitcast=True) & 0x80000000
        rounded = (rounded.to(tl.uint32, bitcast=True) | sign).to(tl.float32, bitcast=True)
        if not SATURATE:
            rounded = tl.where(tl.abs(value) == float("inf"), value, rounded)
        rounded = tl.where(value != value, value, rounded)  # noqa: PLR0124
        return rounded.to(dtype, fp_downcast_rounding="rtne")
