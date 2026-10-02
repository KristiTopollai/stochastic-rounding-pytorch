"""torch.compile baselines for functional and in-place reference updates."""

from .functional import (
    compile_adamw_step,
    compile_adamw_step_,
    compile_sgdm_step,
    compile_sgdm_step_,
    compile_sr_cast,
)
from .optim import AdamWCompiledSR, SGDMCompiledSR

__all__ = [
    "AdamWCompiledSR",
    "SGDMCompiledSR",
    "compile_adamw_step",
    "compile_adamw_step_",
    "compile_sgdm_step",
    "compile_sgdm_step_",
    "compile_sr_cast",
]
