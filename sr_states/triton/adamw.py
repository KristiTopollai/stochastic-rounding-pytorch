"""Fused AdamW with FP32 arithmetic and independent moment storage formats."""

from __future__ import annotations

import torch

from sr_states.formats import StateFormat, resolve_formats
from sr_states.optim.adamw import AdamWReferenceSR
from sr_states.rounding import get_triton_rounding

from ._common import require_triton, tl, triton, validate_fp8_device, validate_update_device
from ._keys import RNG_WORD_ARGS, split_words
from ._multi_tensor import CHUNK_TILES, prepare_group
from ._optimizer import nonnegative, validate_options, validate_tensors

if triton is not None:
    from ._common import _chunk_info
    from ._keys import _state_key, _update_key

    @triton.jit
    def _adamw_tile(
        parameter_ptr,
        gradient_ptr,
        exp_avg_ptr,
        exp_avg_sq_ptr,
        indices,
        mask,
        beta1,
        beta2,
        one_minus_beta1,
        one_minus_beta2,
        step_size,
        correction2_sqrt,
        decay,
        eps,
        key,
        logical_offset,
        ROUND_M: tl.constexpr,
        ROUND_V: tl.constexpr,
    ):
        parameter = tl.load(parameter_ptr + indices, mask=mask, other=0.0).to(tl.float32)
        gradient = tl.load(gradient_ptr + indices, mask=mask, other=0.0).to(tl.float32)
        m = tl.load(exp_avg_ptr + indices, mask=mask, other=0.0).to(tl.float32)
        v = tl.load(exp_avg_sq_ptr + indices, mask=mask, other=0.0).to(tl.float32)
        updated_m = beta1 * m + one_minus_beta1 * gradient
        updated_v = beta2 * v + one_minus_beta2 * (gradient * gradient)
        denominator = tl.sqrt(updated_v) / correction2_sqrt + eps
        updated_parameter = parameter * decay - step_size * (updated_m / denominator)

        global_index = indices.to(tl.uint32) + tl.cast(logical_offset, tl.uint32)
        counter_m = global_index + _state_key(key, 0)
        counter_v = global_index + _state_key(key, 1)
        stored_m = ROUND_M(updated_m, counter_m, exp_avg_ptr.dtype.element_ty)
        stored_v = ROUND_V(updated_v, counter_v, exp_avg_sq_ptr.dtype.element_ty)
        tl.store(parameter_ptr + indices, updated_parameter, mask=mask)
        tl.store(exp_avg_ptr + indices, stored_m, mask=mask)
        tl.store(exp_avg_sq_ptr + indices, stored_v, mask=mask)

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
        ROUND_M: tl.constexpr,
        ROUND_V: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        indices = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        _adamw_tile(
            parameter_ptr,
            gradient_ptr,
            exp_avg_ptr,
            exp_avg_sq_ptr,
            indices,
            indices < n_elements,
            beta1,
            beta2,
            one_minus_beta1,
            one_minus_beta2,
            step_size,
            correction2_sqrt,
            decay,
            eps,
            tl.cast(seed, tl.uint32),
            logical_offset,
            ROUND_M,
            ROUND_V,
        )

    @triton.jit(do_not_specialize=RNG_WORD_ARGS)
    def _multi_adamw_kernel(
        meta_ptr,
        chunks_ptr,
        pointers_ptr,
        parameter_witness,
        gradient_witness,
        exp_avg_witness,
        exp_avg_sq_witness,
        beta1,
        beta2,
        one_minus_beta1,
        one_minus_beta2,
        step_size,
        correction2_sqrt,
        decay,
        eps,
        seed_lo,
        seed_hi,
        step_lo,
        step_hi,
        ROUND_M: tl.constexpr,
        ROUND_V: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        CHUNK_TILES: tl.constexpr,
    ):
        row, n_elements, param_index, shard_start, start, tiles = _chunk_info(
            meta_ptr, chunks_ptr, BLOCK_SIZE, CHUNK_TILES
        )
        parameter_ptr = tl.load(pointers_ptr + 4 * row).to(parameter_witness.dtype)
        gradient_ptr = tl.load(pointers_ptr + 4 * row + 1).to(gradient_witness.dtype)
        exp_avg_ptr = tl.load(pointers_ptr + 4 * row + 2).to(exp_avg_witness.dtype)
        exp_avg_sq_ptr = tl.load(pointers_ptr + 4 * row + 3).to(exp_avg_sq_witness.dtype)
        key = _update_key(seed_lo, seed_hi, step_lo, step_hi, param_index)
        for tile in range(tiles):
            indices = start + tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            _adamw_tile(
                parameter_ptr,
                gradient_ptr,
                exp_avg_ptr,
                exp_avg_sq_ptr,
                indices,
                indices < n_elements,
                beta1,
                beta2,
                one_minus_beta1,
                one_minus_beta2,
                step_size,
                correction2_sqrt,
                decay,
                eps,
                key,
                shard_start,
                ROUND_M,
                ROUND_V,
            )


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
    rounding: str | None = None,
    formats: tuple | dict | None = None,
    block_size: int = 256,
) -> None:
    """Update parameter and moments in place; each moment uses its resolved storage format.

    Inputs must be contiguous, disjoint CUDA tensors with identical shapes.
    Parameters and gradients may be FP32 or BF16. Moments may use different registered storage dtypes. The parameter update uses the unrounded FP32 moments.
    ``step`` starts at one. Each moment derives its own counter base from the
    update key ``seed``. ``offset`` is the flat starting index of this shard.
    """
    require_triton()
    if formats is None:
        formats = (
            StateFormat(exp_avg.dtype, "stochastic" if rounding is None else rounding),
            StateFormat(exp_avg_sq.dtype, "stochastic" if rounding is None else rounding),
        )
        rounding = None
    resolved = resolve_formats(
        ("exp_avg", "exp_avg_sq"), formats, rounding=rounding, backend="triton"
    )
    for state, fmt in zip((exp_avg, exp_avg_sq), resolved):
        validate_options(fmt.rounding, block_size, fmt.dtype)
        if state.dtype != fmt.dtype:
            raise ValueError("state dtype does not match its state format")
    _validate_hyperparameters(lr, (beta1, beta2), eps, weight_decay)
    if not isinstance(step, int) or isinstance(step, bool) or step < 1:
        raise ValueError("step must be a positive integer")
    validate_tensors(parameter, gradient, exp_avg, exp_avg_sq)
    validate_fp8_device(parameter.device, *(fmt.dtype for fmt in resolved))
    if parameter.numel() == 0:
        return
    with torch.cuda.device(parameter.device):
        _fused_adamw_kernel[(triton.cdiv(parameter.numel(), block_size),)](
            parameter,
            gradient,
            exp_avg,
            exp_avg_sq,
            parameter.numel(),
            *_adamw_scalars(lr, beta1, beta2, eps, weight_decay, step),
            int(seed) & 0xFFFFFFFF,
            int(offset) & 0xFFFFFFFF,
            ROUND_M=get_triton_rounding(resolved[0].dtype, resolved[0].rounding),
            ROUND_V=get_triton_rounding(resolved[1].dtype, resolved[1].rounding),
            BLOCK_SIZE=block_size,
            num_warps=4,
        )


def _adamw_scalars(lr, beta1, beta2, eps, weight_decay, step):
    return (
        beta1,
        beta2,
        1.0 - beta1,
        1.0 - beta2,
        lr / (1.0 - beta1**step),
        (1.0 - beta2**step) ** 0.5,
        1.0 - lr * weight_decay,
        eps,
    )


def _validate_hyperparameters(lr, betas, eps, weight_decay):
    nonnegative("learning rate", lr)
    nonnegative("epsilon", eps)
    nonnegative("weight_decay", weight_decay)
    if len(betas) != 2 or any(not 0 <= beta < 1 for beta in betas):
        raise ValueError(f"invalid beta values: {betas}")


class AdamWTriton(AdamWReferenceSR):
    """Fused AdamW with FP32 arithmetic and independently configured moments."""

    _rounding_backend = "triton"

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        *,
        state_format=None,
        state_dtype=None,
        rounding=None,
        seed=0,
        block_size=256,
    ):
        self._initial_block_size = block_size
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            state_format=state_format,
            state_dtype=state_dtype,
            rounding=rounding,
            seed=seed,
        )
        self.defaults["block_size"] = block_size

    def _validate_group(self, group):
        super()._validate_group(group)
        validate_options(
            "nearest", group.get("block_size", self._initial_block_size), torch.float32
        )

    def add_param_group(self, param_group):
        super().add_param_group(
            {"block_size": self.defaults.get("block_size", self._initial_block_size), **param_group}
        )

    def _step_group(self, group, formats, items, rng_step):
        items = list(items)
        if not items:
            return
        require_triton()
        launches = prepare_group(self, group, formats, items, validate_update_device)
        beta1, beta2 = group["betas"]
        lr, eps, weight_decay = group["lr"], group["eps"], group["weight_decay"]
        scalars = {
            step: _adamw_scalars(lr, beta1, beta2, eps, weight_decay, step)
            for _, step, _, _ in launches
        }
        words = (*split_words(group["sr_seed"]), *split_words(rng_step))
        round_m, round_v = (get_triton_rounding(fmt.dtype, fmt.rounding) for fmt in formats)
        for device, step, rows, tables in launches:
            with torch.cuda.device(device):
                _multi_adamw_kernel[(tables.chunks.numel(),)](
                    tables.meta,
                    tables.chunks,
                    tables.pointers,
                    *rows[0].tensors,
                    *scalars[step],
                    *words,
                    ROUND_M=round_m,
                    ROUND_V=round_v,
                    BLOCK_SIZE=group["block_size"],
                    CHUNK_TILES=CHUNK_TILES,
                    num_warps=4,
                )
        for _, state, _, step, _ in items:
            state["step"] = step
