import copy

import torch

from sr_states import AdamWReferenceRN, AdamWReferenceSR, SGDMReferenceRN, SGDMReferenceSR


def test_sgdm_fp32_state_matches_torch_sgd():
    initial = torch.linspace(-1, 1, 257)
    ours = torch.nn.Parameter(initial.clone())
    theirs = torch.nn.Parameter(initial.clone())
    ours_optim = SGDMReferenceSR([ours], lr=0.03, momentum=0.8, state_dtype=torch.float32)
    torch_optim = torch.optim.SGD([theirs], lr=0.03, momentum=0.8)

    for step in range(5):
        gradient = torch.sin(initial * (step + 1))
        ours.grad = gradient.clone()
        theirs.grad = gradient.clone()
        ours_optim.step()
        torch_optim.step()

    torch.testing.assert_close(ours, theirs, rtol=1e-6, atol=1e-7)


def test_adamw_fp32_state_matches_torch_adamw():
    initial = torch.linspace(-1, 1, 257)
    ours = torch.nn.Parameter(initial.clone())
    theirs = torch.nn.Parameter(initial.clone())
    options = {"lr": 3e-3, "betas": (0.8, 0.95), "eps": 1e-6, "weight_decay": 0.1}
    ours_optim = AdamWReferenceSR([ours], state_dtype=torch.float32, **options)
    torch_optim = torch.optim.AdamW([theirs], foreach=False, **options)

    for step in range(5):
        gradient = torch.cos(initial * (step + 1))
        ours.grad = gradient.clone()
        theirs.grad = gradient.clone()
        ours_optim.step()
        torch_optim.step()

    torch.testing.assert_close(ours, theirs, rtol=2e-6, atol=2e-7)


def test_bf16_adamw_state_uses_half_the_fp32_state_bytes():
    p_bf16 = torch.nn.Parameter(torch.ones(128))
    p_fp32 = torch.nn.Parameter(torch.ones(128))
    bf16_optim = AdamWReferenceSR([p_bf16], state_dtype=torch.bfloat16)
    fp32_optim = AdamWReferenceSR([p_fp32], state_dtype=torch.float32)
    p_bf16.grad = torch.ones_like(p_bf16)
    p_fp32.grad = torch.ones_like(p_fp32)
    bf16_optim.step()
    fp32_optim.step()

    bf16_state = bf16_optim.state[p_bf16]
    fp32_state = fp32_optim.state[p_fp32]
    bf16_bytes = sum(
        value.numel() * value.element_size()
        for value in bf16_state.values()
        if torch.is_tensor(value)
    )
    fp32_bytes = sum(
        value.numel() * value.element_size()
        for value in fp32_state.values()
        if torch.is_tensor(value)
    )
    assert bf16_bytes * 2 == fp32_bytes


def test_optimizer_state_dict_restores_rng_position():
    initial = torch.linspace(-1, 1, 1024)
    first = torch.nn.Parameter(initial.clone())
    first_optim = SGDMReferenceSR([first], lr=0.01, seed=44)
    first.grad = torch.sin(initial)
    first_optim.step()

    resumed = torch.nn.Parameter(first.detach().clone())
    resumed_optim = SGDMReferenceSR([resumed], lr=0.01, seed=999)
    resumed_optim.load_state_dict(copy.deepcopy(first_optim.state_dict()))
    gradient = torch.cos(initial)
    first.grad = gradient.clone()
    resumed.grad = gradient.clone()
    first_optim.step()
    resumed_optim.step()
    assert torch.equal(first, resumed)
    assert torch.equal(
        first_optim.state[first]["momentum_buffer"],
        resumed_optim.state[resumed]["momentum_buffer"],
    )


def test_round_to_nearest_variants_store_deterministic_bf16_state():
    sgd_parameter = torch.nn.Parameter(torch.linspace(-1, 1, 64))
    sgd_parameter.grad = torch.cos(sgd_parameter.detach())
    sgd = SGDMReferenceRN([sgd_parameter], lr=0.01)
    sgd.step()
    expected_momentum = sgd_parameter.grad.float().to(torch.bfloat16)
    assert torch.equal(sgd.state[sgd_parameter]["momentum_buffer"], expected_momentum)

    adam_parameter = torch.nn.Parameter(torch.linspace(-1, 1, 64))
    adam_parameter.grad = torch.sin(adam_parameter.detach())
    adam = AdamWReferenceRN([adam_parameter], lr=0.01)
    adam.step()
    assert adam.state[adam_parameter]["exp_avg"].dtype == torch.bfloat16
    assert adam.state[adam_parameter]["exp_avg_sq"].dtype == torch.bfloat16
