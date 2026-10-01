import copy

import pytest
import torch

from sr_states.triton import (
    SGDMTriton,
    is_triton_available,
    rn_cast_bf16,
    sr_cast_bf16,
    triton_sgdm_step_,
)


def test_optional_triton_api_imports_without_triton_installed():
    assert callable(sr_cast_bf16)
    assert callable(rn_cast_bf16)
    assert callable(triton_sgdm_step_)
    assert issubclass(SGDMTriton, torch.optim.Optimizer)
    assert isinstance(is_triton_available(), bool)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"lr": -0.1}, "learning rate"),
        ({"lr": 0.1, "momentum": 1.0}, "momentum value"),
        ({"lr": 0.1, "rounding": "invalid"}, "rounding"),
        ({"lr": 0.1, "block_size": 64}, "block_size"),
    ],
)
def test_triton_optimizer_rejects_invalid_options(kwargs, message):
    parameter = torch.nn.Parameter(torch.ones(4))
    with pytest.raises(ValueError, match=message):
        SGDMTriton([parameter], **kwargs)


@pytest.mark.parametrize("source_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("target_dtype", [torch.float32, torch.bfloat16])
def test_triton_checkpoint_preserves_bf16_state_on_cpu(source_dtype, target_dtype):
    # Serialization needs no CUDA kernel, so CPU CI can guard the storage
    # contract even when the end-to-end GPU resume test is skipped.
    source = torch.nn.Parameter(torch.ones(4, dtype=source_dtype))
    unused = torch.nn.Parameter(torch.ones(4, dtype=source_dtype))
    optimizer = SGDMTriton([{"params": [source]}, {"params": [unused]}], lr=0.1, seed=44)
    buffer = torch.tensor([0.0, -0.0, 1.0078125, -0.50390625], dtype=torch.bfloat16)
    optimizer.state[source]["momentum_buffer"] = buffer.clone()
    optimizer.param_groups[0]["sr_offset"] = 2**32 + 17
    checkpoint = copy.deepcopy(optimizer.state_dict())

    target = torch.nn.Parameter(torch.ones(4, dtype=target_dtype))
    target_unused = torch.nn.Parameter(torch.ones(4, dtype=target_dtype))
    resumed = SGDMTriton([{"params": [target]}, {"params": [target_unused]}], lr=0.9, seed=999)
    resumed.load_state_dict(checkpoint)

    restored = resumed.state[target]["momentum_buffer"]
    assert restored.dtype == torch.bfloat16
    assert restored.device == target.device
    assert torch.equal(restored.view(torch.int16), buffer.view(torch.int16))
    assert target_unused not in resumed.state
    assert resumed.param_groups[0]["sr_offset"] == 2**32 + 17
    assert resumed.param_groups[1]["sr_offset"] == 0
    assert resumed.param_groups[0]["sr_seed"] == 44
    assert resumed.param_groups[0]["lr"] == 0.1
    assert checkpoint["state"][0]["momentum_buffer"].dtype == torch.bfloat16


def test_triton_checkpoint_before_first_step_keeps_state_lazy():
    parameter = torch.nn.Parameter(torch.ones(4))
    optimizer = SGDMTriton([parameter], lr=0.1)
    optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    assert not optimizer.state
