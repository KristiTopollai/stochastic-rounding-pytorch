import importlib
from itertools import chain

import pytest
import torch

from sr_states import StateFormat
from sr_states._hash import state_key, update_key
from sr_states.triton import AdamWTriton, SGDMTriton, is_triton_available
from sr_states.triton._common import tl, triton
from sr_states.triton._keys import RNG_WORD_ARGS, split_words
from sr_states.triton._multi_tensor import CHUNK_TILES

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not is_triton_available(), reason="requires CUDA and Triton"),
]

DTYPES = (torch.float32, torch.bfloat16, torch.float16, torch.float8_e4m3fn, torch.float8_e5m2)


class _MixedGradientParameter(torch.nn.Parameter):
    # Simulate independently typed gradient storage; Tensor.grad normally
    # requires the same dtype as the parameter.
    @property
    def grad(self):
        return getattr(self, "_gradient", None)

    @grad.setter
    def grad(self, value):
        self._gradient = value


def assert_bytes(actual, expected):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.parametrize("kind", ["adamw", "sgd"])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("rounding", ["nearest", "stochastic"])
def test_multi_tensor_matches_functional_updates_byte_for_byte(kind, dtype, rounding):
    if dtype in DTYPES[-2:] and torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("FP8 requires sm_89")
    adamw = kind == "adamw"
    cls = AdamWTriton if adamw else SGDMTriton
    module = importlib.import_module(f"sr_states.triton.{kind}")
    update = getattr(module, "triton_adamw_step_" if adamw else "triton_sgdm_step_")
    block_size = 128
    chunk = CHUNK_TILES * block_size
    start_step = 2**32 + 8

    class OffsetOptimizer(cls):
        def _shard_start(self, parameter):
            return offsets[parameter]

    def make_parameter(n, ptype):
        values = torch.linspace(-0.8, 0.7, n + 1, device="cuda")
        return _MixedGradientParameter(values.to(ptype)[1:])  # An unaligned view.

    pairs = ((torch.float32, torch.float32), (torch.bfloat16, torch.bfloat16))
    layouts = [
        [(0, *pairs[0]), (1, *pairs[0]), (chunk + 13, *pairs[0]), (chunk, *pairs[1])],
        [(chunk + 1, *pairs[0]), (61, pairs[0][0], pairs[1][1])],
    ]
    groups = [[make_parameter(n, pdtype) for n, pdtype, _ in rows] for rows in layouts]
    formats = [(StateFormat(dtype, rounding),) * len(cls._state_names)]
    if adamw:
        second = torch.float8_e5m2 if dtype == torch.float8_e4m3fn else torch.bfloat16
        other_rule = "nearest" if rounding == "stochastic" else "stochastic"
        formats.append((StateFormat(dtype, rounding), StateFormat(second, other_rule)))
    else:
        formats.append((StateFormat(torch.bfloat16, rounding),))
    seeds = (-(2**48) - 1, 2**78 + 0xABCD000100000010)
    opt = OffsetOptimizer(
        [
            {
                "params": params,
                "state_format": dict(zip(cls._state_names, fmts)),
                "sr_seed": seed,
                "sr_step": start_step,
            }
            for params, fmts, seed in zip(groups, formats, seeds)
        ],
        lr=0.06,
        block_size=block_size,
    )
    offsets = {p: 2**32 - 83 + i for i, p in enumerate(chain.from_iterable(groups))}
    expected = {p: p.detach().clone() for p in offsets}
    states = {}
    steps = {}
    generator = torch.Generator(device="cuda").manual_seed(756)
    for i in range(4):
        if i == 2:
            later = make_parameter(35, torch.float32)
            offsets[later] = -(2**32) - 13
            expected[later] = later.detach().clone()
            groups.append([later])
            layouts.append([(35, *pairs[0])])
            formats.append(formats[0])
            opt.add_param_group(
                {
                    "params": [later],
                    "state_format": dict(zip(cls._state_names, formats[0])),
                    "sr_seed": seeds[0],
                }
            )

        # Keep previous gradients live until replacements are allocated to
        # force different addresses, even on a caching allocator.
        old_grads = [p.grad for params in groups for p in params]
        opt.zero_grad(set_to_none=True)
        rng_step = start_step + i + 1
        index = 0
        for group, params, rows, fmt in zip(opt.param_groups, groups, layouts, formats):
            for p, (n, _, gdtype) in zip(params, rows):
                if index != 2 or i not in (1, 3):
                    p.grad = torch.randn(n, device="cuda", generator=generator).to(gdtype)
                    if p not in states:
                        states[p] = tuple(torch.zeros_like(p, dtype=f.dtype) for f in fmt)
                    steps[p] = steps.get(p, 0) + 1
                    kwargs = (
                        {
                            "beta1": group["betas"][0],
                            "beta2": group["betas"][1],
                            "eps": group["eps"],
                            "weight_decay": group["weight_decay"],
                            "step": steps[p],
                        }
                        if adamw
                        else {"momentum_factor": group["momentum"]}
                    )
                    update(
                        expected[p],
                        p.grad,
                        *states[p],
                        lr=group["lr"],
                        seed=update_key(group["sr_seed"], rng_step, index),
                        offset=offsets[p],
                        formats=fmt,
                        block_size=block_size,
                        **kwargs,
                    )
                index += 1
        del old_grads
        opt.step()
        assert opt._rng_step() == rng_step
        for p in offsets:
            assert_bytes(p, expected[p])
            for name, state in zip(cls._state_names, states.get(p, ())):
                assert_bytes(opt.state[p][name], state)
            assert opt.state[p].get("step") == (steps.get(p) if adamw else None)


if triton is not None:
    from sr_states.triton._keys import _state_key, _update_key

    @triton.jit(do_not_specialize=RNG_WORD_ARGS)
    def _derive_keys(out_ptr, index_ptr, seed_lo, seed_hi, step_lo, step_hi):
        index = tl.load(index_ptr)
        key = _update_key(seed_lo, seed_hi, step_lo, step_hi, index)
        tl.store(out_ptr, key)
        tl.store(out_ptr + 1, _state_key(key, 0))
        tl.store(out_ptr + 2, _state_key(key, 1))


@pytest.mark.parametrize(
    "seed,step,index",
    [
        (1, 16, 0),
        (-1, 2**32 + 1, 2**32 - 1),
        (2**87 + 7, 2**64 + 2**63 + 16, 2**39 + 13),
        (-(2**48) + 15, 2**32 - 1, 2**31 + 1),
    ],
)
def test_device_keys_match_host(seed, step, index):
    result = torch.empty(3, device="cuda", dtype=torch.int64)
    index_tensor = torch.tensor(index, device="cuda", dtype=torch.int64)
    _derive_keys[(1,)](result, index_tensor, *split_words(seed), *split_words(step))
    key = update_key(seed, step, index)
    assert result.tolist() == [key, state_key(key, 0), state_key(key, 1)]


@pytest.mark.parametrize("kind,cls", [("adamw", AdamWTriton), ("sgd", SGDMTriton)])
def test_one_launch_per_bucket_and_no_launch_on_group_validation_error(kind, cls, monkeypatch):
    module = importlib.import_module(f"sr_states.triton.{kind}")
    kernel = getattr(module, "_multi_adamw_kernel" if kind == "adamw" else "_multi_sgdm_kernel")
    run = kernel.run
    grids = []

    def count(*args, grid, **kwargs):
        grids.append(grid)
        return run(*args, grid=grid, **kwargs)

    monkeypatch.setattr(kernel, "run", count)
    chunk_size = 256 * CHUNK_TILES

    def parameter(n, dtype=torch.float32):
        p = torch.nn.Parameter(torch.ones(n, device="cuda", dtype=dtype))
        p.grad = torch.ones_like(p)
        return p

    p, q, r, empty, second = (
        parameter(1),
        parameter(chunk_size + 1),
        parameter(3, torch.bfloat16),
        parameter(0),
        parameter(31),
    )
    opt = cls([{"params": [p, q, r, empty]}, {"params": [second]}], lr=0.01)
    opt.step()
    assert grids == [(3,), (1,), (1,)]
    grids.clear()
    q.grad = None
    opt.step()
    q.grad = torch.ones_like(q)
    grids.clear()
    opt.step()
    assert grids == ([(1,), (2,), (1,), (1,)] if kind == "adamw" else [(3,), (1,), (1,)])

    # An invalid later bucket must not update the valid earlier ones.
    before = p.detach().clone()
    grids.clear()
    opt.state[r][opt._state_names[0]] = torch.zeros(4, device=r.device, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="shapes"):
        opt.step()
    assert not grids
    assert_bytes(p, before)


@pytest.mark.parametrize("cls", [AdamWTriton, SGDMTriton])
def test_cached_tables_follow_the_launch_stream(cls):
    p, q = [torch.nn.Parameter(torch.ones(257, device="cuda")) for _ in range(2)]
    base = torch.cuda.current_stream()
    p.grad = q.grad = torch.full_like(p, 0.123)
    alternating = cls([p], lr=0.02)
    single_stream = cls([q], lr=0.02)
    alternate = torch.cuda.Stream()
    for _ in range(3):
        alternating.step()
        alternate.wait_stream(base)
        with torch.cuda.stream(alternate):
            alternating.step()
        base.wait_stream(alternate)
        single_stream.step()
        single_stream.step()
    assert_bytes(p, q)
    for name in alternating._state_names:
        assert_bytes(alternating.state[p][name], single_stream.state[q][name])
