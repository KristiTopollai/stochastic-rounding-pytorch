"""Shared optimizer state allocation, storage, step loop and checkpoint handling."""

from __future__ import annotations

import torch
from torch.optim import Optimizer

from sr_states._hash import state_key, update_key
from sr_states.formats import StateFormat, force_rounding, resolve_formats
from sr_states.rounding import cast

if torch.distributed.is_available():
    from torch.distributed.tensor import DTensor


def _validate_rng_numel(numel: int) -> None:
    if numel > 1 << 32:
        raise ValueError("optimizer RNG supports at most 2**32 elements per parameter")


class StateOptimizer(Optimizer):
    """Base for optimizers whose recurrence reads and returns FP32 tensors.

    Declare ``_state_names`` and implement ``_compute(p32, g32, states32,
    group, step) -> (new_p32, tuple_of_new_states32)``. The base supplies zero
    initialization, per-state storage, counters, closures and checkpoints.
    Optimizers with fused/compiled implementations can override ``_step_parameter``.
    Set ``_forced_rounding`` to fix every state's rule; formats, shorthands,
    group overrides and checkpoints then choose only each state's dtype.
    """

    _state_names: tuple[str, ...] = ()
    _rounding_backend = "reference"
    _uses_step = True
    _forced_rounding: str | None = None

    def __init__(
        self, params, defaults, *, state_format=None, state_dtype=None, rounding=None, seed=0
    ):
        if not self._state_names or len(set(self._state_names)) != len(self._state_names):
            raise ValueError("_state_names must contain unique state names")
        # Keep the public API strict; tuples are an internal functional interface.
        if state_format is not None and not isinstance(state_format, (StateFormat, dict)):
            raise TypeError("state_format must be a StateFormat or a dict keyed by state name")
        formats = resolve_formats(
            self._state_names,
            state_format,
            state_dtype=state_dtype,
            rounding=rounding,
            forced_rounding=self._forced_rounding,
            backend=self._rounding_backend,
        )
        defaults = {
            **defaults,
            "state_format": self._serialize(formats),
            "sr_seed": int(seed),
            "sr_step": 0,
        }
        super().__init__(params, defaults)

    def _serialize(self, formats):
        return {name: fmt.to_dict() for name, fmt in zip(self._state_names, formats)}

    def _formats(self, group):
        return tuple(
            StateFormat.from_dict(group["state_format"][name]) for name in self._state_names
        )

    def _validate_group(self, group):
        """Subclass hook for hyperparameters, also run on add/load."""

    def _normalize_group(self, group, inherited=None, *, checkpoint=False):
        group = dict(group)
        explicit = group.get("state_format")
        if explicit is not None:
            if not isinstance(explicit, (StateFormat, dict)):
                raise TypeError("state_format must be a StateFormat or a dict keyed by state name")
            if "state_dtype" in group or "rounding" in group:
                raise ValueError("state_format cannot be combined with state_dtype or rounding")
            if checkpoint:
                if not isinstance(explicit, dict) or set(explicit) != set(self._state_names):
                    raise ValueError(f"state_format keys must be exactly {self._state_names}")
                explicit = {name: StateFormat.from_dict(fmt) for name, fmt in explicit.items()}
            formats = resolve_formats(
                self._state_names,
                explicit,
                forced_rounding=self._forced_rounding,
                backend=self._rounding_backend,
            )
        elif "state_dtype" in group or "rounding" in group:
            previous = (
                self._formats(inherited)
                if inherited
                else (StateFormat(torch.bfloat16),) * len(self._state_names)
            )
            formats = tuple(
                StateFormat(
                    group.get("state_dtype", fmt.dtype), group.get("rounding", fmt.rounding)
                )
                for fmt in previous
            )
        else:
            formats = (
                self._formats(inherited)
                if inherited
                else (StateFormat(torch.bfloat16),) * len(self._state_names)
            )
        formats = force_rounding(formats, self._forced_rounding)
        for fmt in formats:
            fmt.validate(self._rounding_backend)
        group.pop("state_dtype", None)
        group.pop("rounding", None)
        group.pop("sr_offset", None)
        group.setdefault("sr_step", 0)
        group["state_format"] = self._serialize(formats)
        self._validate_group({**(inherited or {}), **group})
        return group

    def add_param_group(self, param_group):
        normalized = self._normalize_group(param_group, self.defaults)
        super().add_param_group(normalized)

    def _step_parameter(self, parameter, state, group, formats, step, seed, offset):
        """Update one parameter using its RNG key and flat shard start."""
        updated_p, proposals = self._compute(
            parameter.float(),
            parameter.grad.float(),
            tuple(state[name].float() for name in self._state_names),
            group,
            step,
        )
        if len(proposals) != len(self._state_names):
            raise ValueError("_compute must return one tensor per state")
        stored = [
            cast(
                value,
                dtype=fmt.dtype,
                rounding=fmt.rounding,
                seed=state_key(seed, i),
                offset=offset,
            )
            for i, (value, fmt) in enumerate(zip(proposals, formats))
        ]
        parameter.copy_(updated_p)
        for name, value in zip(self._state_names, stored):
            state[name].copy_(value)

    def _shard_start(self, parameter) -> int:
        """Return the flat global index of the parameter's first local element."""
        if torch.distributed.is_available() and isinstance(parameter, DTensor):
            raise NotImplementedError("DTensor optimizer parameters require FSDP2 support")
        return 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        rng_step = self._rng_step() + 1
        for group in self.param_groups:
            group["sr_step"] = rng_step
        first_index = 0
        for group in self.param_groups:
            formats = self._formats(group)
            for param_index, parameter in enumerate(group["params"], start=first_index):
                if parameter.grad is None:
                    continue
                if parameter.grad.is_sparse:
                    raise RuntimeError(f"{type(self).__name__} does not support sparse gradients")
                offset = self._shard_start(parameter)
                _validate_rng_numel(parameter.numel())
                state = self.state[parameter]
                for name, fmt in zip(self._state_names, formats):
                    if name not in state:
                        state[name] = torch.zeros_like(
                            parameter, dtype=fmt.dtype, memory_format=torch.preserve_format
                        )
                    elif state[name].dtype != fmt.dtype:
                        raise ValueError(f"{name} dtype does not match its state format")
                next_step = state.get("step", 0) + 1
                seed = update_key(group["sr_seed"], rng_step, param_index)
                self._step_parameter(parameter, state, group, formats, next_step, seed, offset)
                if self._uses_step:
                    state["step"] = next_step
            first_index += len(group["params"])
        return loss

    def load_state_dict(self, state_dict):
        # Normalize a copy: never mutate the caller's checkpoint. Preserve state
        # before PyTorch casts floating tensors to the parameter's dtype.
        checkpoint = dict(state_dict)
        checkpoint["param_groups"] = [
            self._normalize_group(g, checkpoint=True) for g in state_dict["param_groups"]
        ]
        saved = {}

        def capture(optimizer, source):
            for group, source_group in zip(optimizer.param_groups, source["param_groups"]):
                for parameter, key in zip(group["params"], source_group["params"]):
                    original = source["state"].get(key, {})
                    saved[parameter] = {
                        name: original[name] for name in self._state_names if name in original
                    }

        def restore(optimizer):
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    for name, fmt in zip(self._state_names, self._formats(group)):
                        if name in saved.get(parameter, {}):
                            optimizer.state[parameter][name] = (
                                saved[parameter][name]
                                .to(device=parameter.device, dtype=fmt.dtype)
                                .clone()
                            )

        before = self.register_load_state_dict_pre_hook(capture)
        after = self.register_load_state_dict_post_hook(restore, prepend=True)
        try:
            return super().load_state_dict(checkpoint)
        finally:
            before.remove()
            after.remove()

    def _rng_step(self) -> int:
        return max((int(group["sr_step"]) for group in self.param_groups), default=0)
