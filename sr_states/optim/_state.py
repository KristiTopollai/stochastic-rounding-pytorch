"""Optimizer state storage and checkpoint handling."""

from __future__ import annotations

import torch
from torch.optim import Optimizer


class StateOptimizer(Optimizer):
    _state_names: tuple[str, ...] = ()

    def load_state_dict(self, state_dict):
        # PyTorch casts floating state to the parameter dtype. Preserve the
        # original tensors so FP32 state also survives loading with BF16 parameters.
        saved = {}

        def capture(optimizer, checkpoint):
            for group, source_group in zip(optimizer.param_groups, checkpoint["param_groups"]):
                for parameter, key in zip(group["params"], source_group["params"]):
                    source = checkpoint["state"].get(key, {})
                    saved[parameter] = {
                        name: source[name] for name in self._state_names if name in source
                    }

        def restore(optimizer):
            for group in optimizer.param_groups:
                dtype = group.get("state_dtype", torch.bfloat16)
                for parameter in group["params"]:
                    for name, value in saved.get(parameter, {}).items():
                        optimizer.state[parameter][name] = value.to(
                            device=parameter.device, dtype=dtype
                        ).clone()

        before = self.register_load_state_dict_pre_hook(capture)
        after = self.register_load_state_dict_post_hook(restore, prepend=True)
        try:
            return super().load_state_dict(state_dict)
        finally:
            before.remove()
            after.remove()

    def _next_offset(self) -> int:
        return max((int(group["sr_offset"]) for group in self.param_groups), default=0)
