"""torch.compile baselines for functional and in-place reference updates."""

from .functional import (
    compile_adamw_step,
    compile_adamw_step_,
    compile_sgdm_step,
    compile_sgdm_step_,
    compile_sr_cast,
)

__all__ = [
    "compile_adamw_step",
    "compile_adamw_step_",
    "compile_sgdm_step",
    "compile_sgdm_step_",
    "compile_sr_cast",
]
