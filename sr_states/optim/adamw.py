"""Readable PyTorch AdamW optimizers with low-precision state storage."""

from __future__ import annotations

from collections.abc import Iterable

import torch

from sr_states.reference.adamw import adamw_step

from ._state import StateOptimizer


class AdamWReferenceSR(StateOptimizer):
    """AdamW with FP32 arithmetic and FP32/BF16 moment-state storage."""

    _state_names = ("exp_avg", "exp_avg_sq")

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        *,
        state_dtype: torch.dtype = torch.bfloat16,
        rounding: str = "stochastic",
        seed: int = 0,
    ) -> None:
        beta1, beta2 = betas
        if lr < 0:
            raise ValueError(f"invalid learning rate: {lr}")
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError(f"invalid beta values: {betas}")
        if eps < 0:
            raise ValueError(f"invalid epsilon value: {eps}")
        if weight_decay < 0:
            raise ValueError(f"invalid weight_decay value: {weight_decay}")
        if state_dtype not in {torch.bfloat16, torch.float32}:
            raise ValueError("state_dtype must be torch.bfloat16 or torch.float32")
        if rounding not in {"stochastic", "nearest"}:
            raise ValueError("rounding must be 'stochastic' or 'nearest'")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
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

        offset = self._next_offset()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.grad.is_sparse:
                    raise RuntimeError("AdamWReferenceSR does not support sparse gradients")
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        parameter, dtype=group["state_dtype"], memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        parameter, dtype=group["state_dtype"], memory_format=torch.preserve_format
                    )
                state["step"] += 1
                updated_p, updated_m, updated_v = adamw_step(
                    parameter,
                    parameter.grad,
                    state["exp_avg"],
                    state["exp_avg_sq"],
                    lr=group["lr"],
                    beta1=beta1,
                    beta2=beta2,
                    eps=group["eps"],
                    weight_decay=group["weight_decay"],
                    step=state["step"],
                    state_dtype=group["state_dtype"],
                    rounding=group["rounding"],
                    seed=group["sr_seed"],
                    offset=offset,
                )
                parameter.copy_(updated_p)
                state["exp_avg"].copy_(updated_m)
                state["exp_avg_sq"].copy_(updated_v)
                offset += 2 * parameter.numel()
                group["sr_offset"] = offset
        return loss


class AdamWReferenceRN(AdamWReferenceSR):
    """Deterministic BF16 round-to-nearest AdamW reference."""

    def __init__(self, params, **kwargs) -> None:
        kwargs["rounding"] = "nearest"
        super().__init__(params, **kwargs)
