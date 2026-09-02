import pytest
import torch

from sr_states.triton import SGDMTriton, is_triton_available, sr_cast_bf16, triton_sgdm_step_


def test_optional_triton_api_imports_without_triton_installed():
    assert callable(sr_cast_bf16)
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
