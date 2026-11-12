import copy
import sys

import pytest
import torch

from sr_states import AdamWReferenceSR, SGDMReferenceSR, StateFormat, cast
from sr_states._hash import state_key, update_key
from sr_states.compiled import (
    AdamWCompiledSR,
    SGDMCompiledSR,
    compile_adamw_step,
    compile_sgdm_step,
)
from sr_states.optim._state import _validate_rng_numel
from sr_states.reference.adamw import adamw_step
from sr_states.reference.sgd import sgdm_step

STATE_DTYPES = (torch.bfloat16, torch.float16, torch.float8_e4m3fn, torch.float8_e5m2)
SEED = 0xFEDCBA9876543210


def assert_bytes_equal(actual, expected):
    assert actual.dtype == expected.dtype
    assert torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    )


def gradient():
    return 1.0 + torch.arange(1031, dtype=torch.float32) / 2048


def adamw_options():
    return {"betas": (0.5, 0.5), "eps": 0.125, "weight_decay": 0.0}


@pytest.mark.parametrize(
    "seed,step,index,expected",
    [
        (0, 0, 0, 0x0EAA7511),
        (0, 1, 0, 0x7D2B5EA8),
        (42, 17, 9, 0xFA76534D),
        (0x12345678ABCDEFFE, 0xFEDCBA9876543210, 0xA5A5A5A5, 0xF820C188),
        (-1, 1 << 32, 1, 0x03D350C7),
    ],
)
def test_update_key_known_answers(seed, step, index, expected):
    assert update_key(seed, step, index) == expected


@pytest.mark.parametrize(
    "key,index,expected",
    [
        (0, 0, 0),
        (0, 1, 0x688990C0),
        (1, 0, 0x688990C0),
        (0x89ABCDEF, 1, 0x575E21BD),
        (0xFFFFFFFF, 2, 0x5681FFD9),
    ],
)
def test_state_key_known_answers(key, index, expected):
    assert state_key(key, index) == expected


def test_changing_one_key_word_changes_the_key():
    seed, step, index = SEED, 0x123456789ABCDEF0, 13
    original = update_key(seed, step, index)
    for bit in range(64):
        assert update_key(seed ^ (1 << bit), step, index) != original
        assert update_key(seed, step ^ (1 << bit), index) != original
    for bit in range(32):
        assert update_key(seed, step, index ^ (1 << bit)) != original
    assert update_key(seed + (1 << 64), step + (1 << 64), index) == original
    assert len({state_key(original, i) for i in range(3)}) == 3


def test_sampled_update_keys_are_distinct():
    keys = {update_key(42, step, index) for step in range(1, 1001) for index in range(120)}
    assert len(keys) == 120_000


@pytest.mark.parametrize("adamw", [True, False])
@pytest.mark.parametrize("compiled", [True, False])
@pytest.mark.parametrize("dtype", STATE_DTYPES)
def test_optimizer_state_bytes_follow_rng_spec(adamw, compiled, dtype):
    torch._dynamo.reset()
    p = torch.nn.Parameter(torch.ones(1031))
    p.grad = gradient()
    skipped = torch.nn.Parameter(torch.ones(3))
    options = {"compile_kwargs": {"backend": "eager"}} if compiled else {}
    if adamw:
        cls = AdamWCompiledSR if compiled else AdamWReferenceSR
        options.update(adamw_options())
        proposals = (p.grad * 0.5, p.grad.square() * 0.5)
    else:
        cls = SGDMCompiledSR if compiled else SGDMReferenceSR
        proposals = (p.grad,)
    optimizer = cls(
        [{"params": [skipped]}, {"params": [p], "sr_seed": SEED}],
        lr=0.01,
        state_dtype=dtype,
        **options,
    )
    # Exercise the high step word and the distinction between the global RNG
    # clock and the parameter's own AdamW bias-correction step.
    optimizer.param_groups[0]["sr_step"] = 1 << 32
    optimizer.step()
    assert [group["sr_step"] for group in optimizer.param_groups] == [(1 << 32) + 1] * 2
    assert skipped not in optimizer.state
    if adamw:
        assert optimizer.state[p]["step"] == 1
    key = update_key(SEED, (1 << 32) + 1, 1)
    for index, (name, proposal) in enumerate(zip(optimizer._state_names, proposals)):
        expected = cast(proposal, dtype=dtype, seed=state_key(key, index), offset=0)
        assert_bytes_equal(optimizer.state[p][name], expected)


@pytest.mark.parametrize("adamw", [True, False])
@pytest.mark.parametrize("compiled", [True, False])
@pytest.mark.parametrize("dtype", STATE_DTYPES)
def test_functional_update_matches_flat_shards(adamw, compiled, dtype):
    torch._dynamo.reset()
    g = gradient()
    p = torch.ones_like(g)
    m = torch.linspace(0, 1, g.numel()).to(dtype)
    if adamw:
        tensors = (p, g, m, m.clone())
        compiler, update = compile_adamw_step, adamw_step
        options = {"beta1": 0.5, "beta2": 0.5, "eps": 0.125, "weight_decay": 0.0, "step": 3}
    else:
        tensors = (p, g, m)
        compiler, update = compile_sgdm_step, sgdm_step
        options = {"momentum_factor": 0.5}
    if compiled:
        update = compiler(backend="eager", fullgraph=True, dynamic=True)
    options.update(lr=0.01, state_dtype=dtype, seed=update_key(SEED, 7, 3))
    start = (1 << 32) - 211
    full = update(*tensors, offset=start, **options)
    chunks = [
        update(*(tensor[lo:hi] for tensor in tensors), offset=start + lo, **options)
        for lo, hi in ((0, 11), (11, 227), (227, 728), (728, g.numel()))
    ]
    for expected, parts in zip(full, zip(*chunks)):
        assert torch.equal(
            expected.view(torch.uint8), torch.cat([p.view(torch.uint8) for p in parts])
        )


@pytest.mark.parametrize("cls", [AdamWReferenceSR, SGDMReferenceSR])
def test_other_parameters_and_appended_groups_do_not_change_stream(cls):
    g = gradient()
    parameters = [torch.nn.Parameter(torch.ones_like(g)) for _ in range(3)]
    optimizers = []
    for p, size in zip(parameters, (11, 11, 300)):
        previous = torch.nn.Parameter(torch.ones(size))
        optimizers.append(cls([{"params": [previous]}, {"params": [p]}], lr=0.01, seed=SEED))
    for index in range(5):
        if index == 2:
            extra = torch.nn.Parameter(torch.ones(333))
            extra.grad = torch.full_like(extra, 0.103)
            optimizers[1].add_param_group({"params": [extra], "sr_seed": 123})
        for run, (p, optimizer) in enumerate(zip(parameters, optimizers)):
            previous = optimizer.param_groups[0]["params"][0]
            previous.grad = torch.full_like(previous, 0.103) if run > 0 else None
            p.grad = g if index != 1 else None
            optimizer.step()
        for p, optimizer in zip(parameters[1:], optimizers[1:]):
            assert_bytes_equal(p, parameters[0])
            for name in optimizer._state_names:
                assert_bytes_equal(
                    optimizer.state[p][name], optimizers[0].state[parameters[0]][name]
                )
    assert [group["sr_step"] for group in optimizers[1].param_groups] == [5, 5, 5]


def test_legacy_checkpoint_keeps_state_bytes_and_restarts_rng():
    p = torch.nn.Parameter(torch.ones(1031, dtype=torch.bfloat16))
    formats = {
        "exp_avg": StateFormat(torch.float8_e4m3fn),
        "exp_avg_sq": StateFormat(torch.float32),
    }
    optimizer = AdamWReferenceSR([p], state_format=formats, seed=SEED, **adamw_options())
    p.grad = gradient().bfloat16()
    optimizer.step()
    optimizer.step()
    checkpoint = copy.deepcopy(optimizer.state_dict())
    group = checkpoint["param_groups"][0]
    del group["sr_step"]
    group["sr_offset"] = 2**32 + 29

    q = torch.nn.Parameter(p.detach().clone())
    resumed = AdamWReferenceSR([q], seed=7)
    resumed.load_state_dict(checkpoint)
    restored_group = resumed.param_groups[0]
    assert restored_group["sr_step"] == 0
    assert restored_group["sr_seed"] == SEED
    assert "sr_offset" not in restored_group
    assert group["sr_offset"] == 2**32 + 29 and "sr_step" not in group
    assert restored_group["state_format"] == group["state_format"]
    assert resumed.state[q]["step"] == 2
    for name in formats:
        assert_bytes_equal(resumed.state[q][name], checkpoint["state"][0][name])
    q.grad = gradient().bfloat16()
    proposal = 0.5 * (resumed.state[q]["exp_avg"].float() + q.grad.float())
    resumed.step()
    assert restored_group["sr_step"] == 1
    assert resumed.state[q]["step"] == 3
    expected = cast(
        proposal, dtype=formats["exp_avg"].dtype, seed=state_key(update_key(SEED, 1, 0), 0)
    )
    assert_bytes_equal(resumed.state[q]["exp_avg"], expected)


def test_rng_numel_limit():
    _validate_rng_numel(1 << 32)
    with pytest.raises(ValueError, match="at most"):
        _validate_rng_numel((1 << 32) + 1)


@pytest.mark.skipif(not torch.distributed.is_available(), reason="requires torch.distributed")
def test_dtensor_is_rejected():
    import torch.distributed as dist
    from torch.distributed.tensor import DeviceMesh, DTensor, Replicate

    # A single fake rank constructs a real DTensor without opening sockets.
    pytest.importorskip("torch.testing._internal.distributed.fake_pg")
    dist.init_process_group("fake", store=dist.HashStore(), rank=0, world_size=1)
    try:
        mesh = DeviceMesh("cpu", [0])
        value = DTensor.from_local(torch.ones(3), mesh, [Replicate()])
        p = torch.nn.Parameter(value)
        p.grad = value.clone()
        optimizer = SGDMReferenceSR([p], lr=0.01)
        with pytest.raises(NotImplementedError, match="DTensor.*FSDP2"):
            optimizer.step()
        assert p not in optimizer.state
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("adamw", [True, False])
@pytest.mark.parametrize("backend", ["eager", "inductor"])
def test_compiled_optimizer_reuses_graph_across_rng_steps(adamw, backend):
    if backend == "inductor" and sys.platform != "linux":
        pytest.skip("Linux CI exercises CPU Inductor")
    torch._dynamo.reset()
    graphs = []

    def count_graph(graph, inputs):
        # Inductor may restart the initial trace to specialize fixed floats.
        # Count the compiled callable only after the backend returns it.
        update = torch._dynamo.lookup_backend(backend)(graph, inputs)
        graphs.append(graph)
        return update

    g = gradient()
    p, q = (torch.nn.Parameter(torch.ones_like(g)) for _ in range(2))
    ref_cls, compiled_cls = (
        (AdamWReferenceSR, AdamWCompiledSR) if adamw else (SGDMReferenceSR, SGDMCompiledSR)
    )
    options = adamw_options() if adamw else {"momentum": 0.5}
    reference = ref_cls([p], lr=0.0, seed=SEED, **options)
    compiled = compiled_cls(
        [q], lr=0.0, seed=SEED, compile_kwargs={"backend": count_graph}, **options
    )
    for _ in range(8):
        p.grad = q.grad = g
        reference.step()
        compiled.step()
        for name in reference._state_names:
            assert_bytes_equal(reference.state[p][name], compiled.state[q][name])
    assert len(graphs) == 1
