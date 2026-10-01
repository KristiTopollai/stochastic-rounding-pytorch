"""Fused AdamW with FP32 arithmetic and stochastic BF16 moment storage."""

from __future__ import annotations

from collections.abc import Iterable

import torch

from sr_states.optim._state import StateOptimizer

from ._common import require_triton, tl, triton
from ._optimizer import nonnegative, validate_options, validate_tensors

if triton is not None:
    from .sr_cast import _stochastic_bf16_value

    @triton.jit
    def _fused_adamw_kernel(
        parameter_ptr,
        gradient_ptr,
        exp_avg_ptr,
        exp_avg_sq_ptr,
        n_elements,
        beta1,
        beta2,
        one_minus_beta1,
        one_minus_beta2,
        step_size,
        correction2_sqrt,
        decay,
        eps,
        seed,
        logical_offset,
        STOCHASTIC: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        indices = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = indices < n_elements
        parameter = tl.load(parameter_ptr + indices, mask=mask, other=0.0).to(tl.float32)
        gradient = tl.load(gradient_ptr + indices, mask=mask, other=0.0).to(tl.float32)
        m = tl.load(exp_avg_ptr + indices, mask=mask, other=0.0).to(tl.float32)
        v = tl.load(exp_avg_sq_ptr + indices, mask=mask, other=0.0).to(tl.float32)
        updated_m = beta1 * m + one_minus_beta1 * gradient
        updated_v = beta2 * v + one_minus_beta2 * (gradient * gradient)
        denominator = tl.sqrt(updated_v) / correction2_sqrt + eps
        updated_parameter = parameter * decay - step_size * (updated_m / denominator)

        if STOCHASTIC:
            counter = (
                indices.to(tl.uint32)
                + tl.cast(seed, tl.uint32)
                + tl.cast(logical_offset, tl.uint32)
            )
            stored_m = _stochastic_bf16_value(updated_m, counter)
            stored_v = _stochastic_bf16_value(updated_v, counter + tl.cast(n_elements, tl.uint32))
        else:
            stored_m, stored_v = updated_m, updated_v
        tl.store(parameter_ptr + indices, updated_parameter, mask=mask)
        tl.store(exp_avg_ptr + indices, stored_m, mask=mask)
        tl.store(exp_avg_sq_ptr + indices, stored_v, mask=mask)


@torch.no_grad()
def triton_adamw_step_(
    parameter: torch.Tensor,
    gradient: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    *,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    step: int,
    seed: int = 0,
    offset: int = 0,
    rounding: str = "stochastic",
    block_size: int = 256,
) -> None:
    """Update parameter and moments in place; SR applies only to BF16 moments.

    Inputs must be contiguous, disjoint CUDA tensors with identical shapes.
    Parameters and gradients may be FP32 or BF16. Both moments must share an
    FP32 or BF16 dtype. The parameter update uses the unrounded FP32 moments.
    ``step`` starts at one; the two moments use consecutive RNG counter ranges.
    """
    require_triton()
    validate_options(rounding, block_size, exp_avg.dtype)
    _validate_hyperparameters(lr, (beta1, beta2), eps, weight_decay)
    if not isinstance(step, int) or isinstance(step, bool) or step < 1:
        raise ValueError("step must be a positive integer")
    validate_tensors(parameter, gradient, exp_avg, exp_avg_sq)
    if parameter.numel() == 0:
        return
    with torch.cuda.device(parameter.device):
        _fused_adamw_kernel[(triton.cdiv(parameter.numel(), block_size),)](
            parameter,
            gradient,
            exp_avg,
            exp_avg_sq,
            parameter.numel(),
            beta1,
            beta2,
            1.0 - beta1,
            1.0 - beta2,
            lr / (1.0 - beta1**step),
            (1.0 - beta2**step) ** 0.5,
            1.0 - lr * weight_decay,
            eps,
            int(seed) & 0xFFFFFFFF,
            int(offset) & 0xFFFFFFFF,
            STOCHASTIC=rounding == "stochastic" and exp_avg.dtype == torch.bfloat16,
            BLOCK_SIZE=block_size,
            num_warps=4,
        )


def _validate_hyperparameters(lr, betas, eps, weight_decay):
    nonnegative("learning rate", lr)
    nonnegative("epsilon", eps)
    nonnegative("weight_decay", weight_decay)
    if len(betas) != 2 or any(not 0 <= beta < 1 for beta in betas):
        raise ValueError(f"invalid beta values: {betas}")


class AdamWTriton(StateOptimizer):
    """AdamW with fused state writes, using BF16 stochastic storage by default.

    Parameters retain their dtype. FP32 state storage and BF16 nearest rounding
    are available as baselines. AMSGrad and sparse gradients are unsupported.
    """

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
        block_size: int = 256,
    ) -> None:
        _validate_hyperparameters(lr, betas, eps, weight_decay)
        validate_options(rounding, block_size, state_dtype)
        super().__init__(
            params,
            {
                "lr": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
                "state_dtype": state_dtype,
                "rounding": rounding,
                "sr_seed": int(seed),
                "sr_offset": 0,
                "block_size": block_size,
            },
        )

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
                    raise RuntimeError("AdamWTriton does not support sparse gradients")
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    for name in self._state_names:
                        state[name] = torch.zeros_like(parameter, dtype=group["state_dtype"])
                next_step = state["step"] + 1
                triton_adamw_step_(
                    parameter,
                    parameter.grad,
                    state["exp_avg"],
                    state["exp_avg_sq"],
                    lr=group["lr"],
                    beta1=beta1,
                    beta2=beta2,
                    eps=group["eps"],
                    weight_decay=group["weight_decay"],
                    step=next_step,
                    rounding=group["rounding"],
                    seed=group["sr_seed"],
                    offset=offset,
                    block_size=group["block_size"],
                )
                state["step"] = next_step
                offset += 2 * parameter.numel()
                group["sr_offset"] = offset
        return loss
