"""Built-in JIT rounding callbacks; optimizer kernels contain no format logic."""

from ._common import tl, triton

if triton is not None:
    from .sr_cast import _stochastic_bf16_value

    @triton.jit
    def identity(value, counter, dtype: tl.constexpr):
        return value

    @triton.jit
    def nearest(value, counter, dtype: tl.constexpr):
        return value.to(dtype, fp_downcast_rounding="rtne")

    @triton.jit
    def stochastic_bf16(value, counter, dtype: tl.constexpr):
        return _stochastic_bf16_value(value, counter).to(dtype)
