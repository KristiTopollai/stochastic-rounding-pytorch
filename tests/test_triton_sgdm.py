import copy

import pytest
import torch

from sr_states.reference.sgd import sgdm_step
from sr_states.triton import SGDMTriton, is_triton_available, triton_sgdm_step_

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not is_triton_available(), reason="requires Triton and CUDA"),
]


@pytest.mark.parametrize("rounding", ["stochastic", "nearest"])
@pytest.mark.parametrize("parameter_dtype", [torch.float32, torch.bfloat16])
def test_fused_sgdm_matches_reference(rounding, parameter_dtype):
    parameter = torch.linspace(-1, 1, 100_003, device="cuda", dtype=parameter_dtype)
    gradient = torch.cos(parameter.float()).to(parameter_dtype)
    momentum = torch.sin(parameter.float()).to(torch.bfloat16)
    expected_parameter, expected_momentum = sgdm_step(
        parameter,
        gradient,
        momentum,
        lr=0.01,
        momentum_factor=0.9,
        rounding=rounding,
        seed=72,
        offset=100,
    )

    triton_sgdm_step_(
        parameter,
        gradient,
        momentum,
        lr=0.01,
        momentum_factor=0.9,
        rounding=rounding,
        seed=72,
        offset=100,
    )

    # Triton may contract the multiply-add, moving the FP32 intermediate by
    # one ULP relative to the eager reference before the BF16 write-back.
    torch.testing.assert_close(parameter, expected_parameter, rtol=8e-3, atol=1e-6)
    torch.testing.assert_close(momentum, expected_momentum, rtol=8e-3, atol=1e-6)


def test_fused_sgdm_is_invariant_to_block_size():
    initial_parameter = torch.linspace(-3, 7, 10_003, device="cuda")
    gradient = torch.sin(initial_parameter)
    initial_momentum = torch.cos(initial_parameter).to(torch.bfloat16)
    outputs = []

    for block_size in (128, 256, 512, 1024):
        parameter = initial_parameter.clone()
        momentum = initial_momentum.clone()
        triton_sgdm_step_(
            parameter,
            gradient,
            momentum,
            lr=0.03,
            momentum_factor=0.8,
            seed=991,
            offset=17,
            block_size=block_size,
        )
        outputs.append((parameter, momentum))

    for parameter, momentum in outputs[1:]:
        assert torch.equal(parameter, outputs[0][0])
        assert torch.equal(momentum, outputs[0][1])


@pytest.mark.parametrize("rounding", ["stochastic", "nearest"])
@pytest.mark.parametrize("parameter_dtype", [torch.float32, torch.bfloat16])
def test_triton_optimizer_checkpoint_restores_rng_position(rounding, parameter_dtype):
    initial = torch.linspace(-1, 1, 4099, device="cuda", dtype=parameter_dtype)
    first = torch.nn.Parameter(initial.clone())
    first_optimizer = SGDMTriton([first], lr=0.01, rounding=rounding, seed=44)
    first.grad = torch.sin(initial)
    first_optimizer.step()

    resumed = torch.nn.Parameter(first.detach().clone())
    resumed_optimizer = SGDMTriton([resumed], lr=0.01, seed=999)
    resumed_optimizer.load_state_dict(copy.deepcopy(first_optimizer.state_dict()))
    assert resumed_optimizer.state[resumed]["momentum_buffer"].dtype == torch.bfloat16
    assert resumed_optimizer.state[resumed]["momentum_buffer"].device == resumed.device

    for step in range(1, 4):
        gradient = torch.cos(initial.float() * step).to(parameter_dtype)
        first.grad = gradient.clone()
        resumed.grad = gradient.clone()
        first_optimizer.step()
        resumed_optimizer.step()

        assert torch.equal(first, resumed)
        assert torch.equal(
            first_optimizer.state[first]["momentum_buffer"],
            resumed_optimizer.state[resumed]["momentum_buffer"],
        )
        expected_offset = (step + 1) * initial.numel()
        assert first_optimizer.param_groups[0]["sr_offset"] == expected_offset
        assert resumed_optimizer.param_groups[0]["sr_offset"] == expected_offset
