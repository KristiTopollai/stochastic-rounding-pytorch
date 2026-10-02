"""Optimizer wrappers with compiled SR updates and ordinary Python bookkeeping."""

from sr_states.optim.adamw import AdamWReferenceSR
from sr_states.optim.sgd import SGDMReferenceSR

from .functional import compile_adamw_step_, compile_sgdm_step_


def _update_function(compiler, compile_kwargs):
    settings = {"fullgraph": True, "dynamic": True, **(compile_kwargs or {})}
    if settings.get("backend", "inductor") == "inductor":
        settings.setdefault("options", {"triton.cudagraphs": False})
    # The optimizer's step() already supplies the no_grad context.
    return compiler(**settings).__wrapped__


class AdamWCompiledSR(AdamWReferenceSR):
    """AdamW with compiled FP32 updates and BF16 stochastic state writes."""

    def __init__(self, params, *, compile_kwargs=None, **kwargs):
        super().__init__(params, **kwargs)
        self._update = _update_function(compile_adamw_step_, compile_kwargs)


class SGDMCompiledSR(SGDMReferenceSR):
    """Momentum SGD with compiled FP32 updates and BF16 stochastic state writes."""

    def __init__(self, params, lr, momentum=0.9, *, compile_kwargs=None, **kwargs):
        super().__init__(params, lr=lr, momentum=momentum, **kwargs)
        self._update = _update_function(compile_sgdm_step_, compile_kwargs)
