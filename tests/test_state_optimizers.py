import copy

import pytest
import torch

from sr_states import AdamWReferenceSR, SGDMReferenceSR
from sr_states.reference.adamw import adamw_step_
from sr_states.reference.sgd import sgdm_step_
from sr_states.triton import AdamWTriton, SGDMTriton

OPTIMIZERS = [SGDMReferenceSR, AdamWReferenceSR, SGDMTriton, AdamWTriton]


@pytest.mark.parametrize("optimizer_class", OPTIMIZERS)
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("parameter_dtype", [torch.float32, torch.bfloat16])
def test_checkpoint_keeps_state_precision(optimizer_class, state_dtype, parameter_dtype):
    p = torch.nn.Parameter(torch.ones(4, dtype=parameter_dtype))
    optimizer = optimizer_class([p], lr=0.01, state_dtype=state_dtype, seed=42)
    original = torch.tensor([0.0, -0.0, 1.003125, 0.503125], dtype=state_dtype)
    for name in optimizer._state_names:
        optimizer.state[p][name] = original.clone()
    if "exp_avg" in optimizer._state_names:
        optimizer.state[p]["step"] = 7
    optimizer.param_groups[0]["sr_offset"] = 2**32 + 31
    checkpoint = copy.deepcopy(optimizer.state_dict())

    q = torch.nn.Parameter(p.detach().clone())
    restored = optimizer_class([q], lr=0.3, state_dtype=torch.float32, seed=99)
    seen = []
    restored.register_load_state_dict_post_hook(
        lambda opt: seen.append(opt.state[q][opt._state_names[0]].dtype)
    )
    restored.load_state_dict(checkpoint)
    assert seen == [state_dtype]
    for name in optimizer._state_names:
        actual = restored.state[q][name]
        assert actual.dtype == state_dtype
        bits = torch.int32 if state_dtype == torch.float32 else torch.int16
        assert torch.equal(actual.view(bits), original.view(bits))
        assert actual.data_ptr() != checkpoint["state"][0][name].data_ptr()
    assert restored.param_groups[0]["sr_offset"] == 2**32 + 31
    assert restored.param_groups[0]["sr_seed"] == 42
    if "exp_avg" in optimizer._state_names:
        assert restored.state[q]["step"] == 7


@pytest.fixture
def cpu_state_kernels(monkeypatch):
    # Exercise optimizer bookkeeping on CPU; GPU tests cover the actual kernels.
    def sgdm(p, g, m, *, block_size, **kwargs):
        sgdm_step_(p, g, m, state_dtype=m.dtype, **kwargs)

    def adamw(p, g, m, v, *, block_size, **kwargs):
        adamw_step_(p, g, m, v, state_dtype=m.dtype, **kwargs)

    monkeypatch.setattr("sr_states.triton.sgd.triton_sgdm_step_", sgdm)
    monkeypatch.setattr("sr_states.triton.adamw.triton_adamw_step_", adamw)


@pytest.mark.parametrize("optimizer_class", OPTIMIZERS)
def test_steps_groups_missing_gradients_and_resume(optimizer_class, cpu_state_kernels):
    parameters = [torch.nn.Parameter(torch.linspace(-1, 1, 17)) for _ in range(3)]
    optimizer = optimizer_class(
        [{"params": parameters[:2]}, {"params": parameters[2:]}], lr=0.1, seed=4
    )
    state_count = len(optimizer._state_names)
    initial_bytes = 17 * 2 * state_count
    for index in (0, 2):
        parameters[index].grad = torch.ones_like(parameters[index]) * 0.103
    optimizer.step()
    assert parameters[1] not in optimizer.state
    assert optimizer.param_groups[0]["sr_offset"] == state_count * 17
    assert optimizer.param_groups[1]["sr_offset"] == 2 * state_count * 17
    state = optimizer.state[parameters[0]]
    assert sum(state[name].nbytes for name in optimizer._state_names) == initial_bytes

    resumed_parameters = [torch.nn.Parameter(p.detach().clone()) for p in parameters]
    resumed = optimizer_class(
        [{"params": resumed_parameters[:2]}, {"params": resumed_parameters[2:]}], lr=0.5
    )
    resumed.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    for i in range(4):
        for p, q in zip(parameters, resumed_parameters):
            p.grad = torch.sin(p.detach() + i)
            q.grad = p.grad.clone()
        optimizer.step()
        resumed.step()
        for p, q in zip(parameters, resumed_parameters):
            assert torch.equal(p, q)
            for name in optimizer._state_names:
                assert torch.equal(optimizer.state[p][name], resumed.state[q][name])
    if state_count == 2:
        assert [optimizer.state[p]["step"] for p in parameters] == [5, 4, 5]
    assert optimizer._next_offset() == (2 + 4 * 3) * 17 * state_count


@pytest.mark.parametrize("optimizer_class", [AdamWTriton, SGDMTriton])
def test_closure_and_sparse_gradient(optimizer_class, cpu_state_kernels):
    p = torch.nn.Parameter(torch.ones(4))
    optimizer = optimizer_class([p], lr=0.01)

    def closure():
        optimizer.zero_grad()
        loss = p.square().sum()
        loss.backward()
        return loss

    assert optimizer.step(closure).item() == 4
    p.grad = torch.sparse_coo_tensor([[0]], [1.0], size=(4,), check_invariants=True)
    offset = optimizer._next_offset()
    with pytest.raises(RuntimeError, match="sparse"):
        optimizer.step()
    assert optimizer._next_offset() == offset


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lr": -1},
        {"lr": float("nan")},
        {"eps": -1},
        {"eps": float("inf")},
        {"betas": (0.9, 1)},
        {"betas": (float("nan"), 0.9)},
        {"weight_decay": -1},
        {"state_dtype": torch.float16},
        {"rounding": "invalid"},
        {"block_size": 64},
    ],
)
def test_adamw_rejects_invalid_options(kwargs):
    with pytest.raises(ValueError):
        AdamWTriton([torch.nn.Parameter(torch.ones(1))], **kwargs)
