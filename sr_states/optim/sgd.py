"""Readable PyTorch SGDM optimizers with an explicit state precision policy."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch.optim import Optimizer

from sr_states.reference.sgd import sgdm_step


class SGDMReferenceSR(Optimizer):
    """Momentum SGD with FP32 math and FP32/BF16 state storage.

    BF16 state uses stochastic rounding by default. RNG state is represented by
    ``sr_seed`` and ``sr_offset`` in each parameter group, so optimizer
    ``state_dict`` checkpoints reproduce the subsequent random choices.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float,
        momentum: float = 0.9,
        *,
        state_dtype: torch.dtype = torch.bfloat16,
        rounding: str = "stochastic",
        seed: int = 0,
    ) -> None:
        if lr < 0:
            raise ValueError(f"invalid learning rate: {lr}")
        if not 0 <= momentum < 1:
            raise ValueError(f"invalid momentum value: {momentum}")
        if state_dtype not in {torch.bfloat16, torch.float32}:
            raise ValueError("state_dtype must be torch.bfloat16 or torch.float32")
        if rounding not in {"stochastic", "nearest"}:
            raise ValueError("rounding must be 'stochastic' or 'nearest'")
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "state_dtype": state_dtype,
            "rounding": rounding,
            "sr_seed": int(seed),
            "sr_offset": 0,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            offset = int(group["sr_offset"])
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.grad.is_sparse:
                    raise RuntimeError("SGDMReferenceSR does not support sparse gradients")
                state = self.state[parameter]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(
                        parameter, dtype=group["state_dtype"], memory_format=torch.preserve_format
                    )
                updated_p, updated_u = sgdm_step(
                    parameter,
                    parameter.grad,
                    state["momentum_buffer"],
                    lr=group["lr"],
                    momentum_factor=group["momentum"],
                    state_dtype=group["state_dtype"],
                    rounding=group["rounding"],
                    seed=group["sr_seed"],
                    offset=offset,
                )
                parameter.copy_(updated_p)
                state["momentum_buffer"].copy_(updated_u)
                offset += parameter.numel()
            group["sr_offset"] = offset
        return loss


class SGDMReferenceRN(SGDMReferenceSR):
    """Deterministic BF16 round-to-nearest SGDM reference."""

    def __init__(self, params, lr: float, momentum: float = 0.9, **kwargs) -> None:
        kwargs["rounding"] = "nearest"
        super().__init__(params, lr=lr, momentum=momentum, **kwargs)
