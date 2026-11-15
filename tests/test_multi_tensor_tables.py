import copy
from types import SimpleNamespace

import pytest
import torch

from sr_states._hash import update_key
from sr_states.optim._state import StateOptimizer
from sr_states.triton import AdamWTriton
from sr_states.triton._keys import split_words
from sr_states.triton._multi_tensor import GroupTableCache, bucket_items

MOMENT = ("m",)


def item(n, index=0, step=1, *, dtype=torch.float32, offset=0, device="cpu", grad_dtype=None):
    p = SimpleNamespace(
        grad=torch.zeros(n, dtype=grad_dtype or dtype, device=device),
        tensor=torch.zeros(n, dtype=dtype, device=device),
    )
    # Use an ordinary tensor proxy so the host bucketer can be tested with
    # different gradient dtypes, which Parameter.grad assignment forbids.
    p = _Parameter(p.tensor, p.grad)
    return p, {"m": torch.zeros(n, device=device)}, index, step, offset


class _Parameter:
    def __init__(self, tensor, grad):
        self.tensor = tensor
        self.grad = grad

    def __getattr__(self, name):
        return getattr(self.tensor, name)


def test_buckets_split_device_dtypes_and_adamw_steps():
    items = [
        item(0, 0),
        item(17, 2),
        item(5, 4, step=2),
        item(3, 5, dtype=torch.bfloat16),
        item(1, 6, grad_dtype=torch.bfloat16),
        item(2, 7, device="meta"),
        item(1, 9),
    ]
    adamw = bucket_items(items, MOMENT, uses_step=True)
    sgdm = bucket_items(items, MOMENT, uses_step=False)
    assert [[r.param_index for r in rows] for rows in adamw.values()] == [
        [2, 9],
        [4],
        [5],
        [6],
        [7],
    ]
    assert [[r.param_index for r in rows] for rows in sgdm.values()] == [[2, 4, 9], [5], [6], [7]]
    (device, dtypes, step), *_ = adamw
    assert (device.type, dtypes, step) == ("cpu", (torch.float32,) * 3, 1)


def test_chunk_boundaries_and_layout_cache(monkeypatch):
    monkeypatch.setattr("sr_states.triton._multi_tensor.CHUNK_TILES", 2)
    items = [item(n, i, offset=2**32 - 20 + i) for i, n in enumerate((1, 255, 256, 257))]
    cache = GroupTableCache()
    buckets = bucket_items(items, MOMENT, uses_step=True)
    ((_, _, _, first),) = cache.prepare(buckets, 128)
    assert first.chunks.tolist() == [0, 1, 2, 3, 3]
    assert first.meta.tolist() == [
        [n, i, i, 2**32 - 20 + i] for i, n in enumerate((1, 255, 256, 257))
    ]
    assert first.pointers.tolist() == [
        [p.data_ptr(), p.grad.data_ptr(), state["m"].data_ptr()] for p, state, *_ in items
    ]
    # AdamW's step is a launch scalar, so progressing together reuses the tables.
    ((_, step, _, same),) = cache.prepare(
        bucket_items([(*row[:3], 2, row[-1]) for row in items], MOMENT, uses_step=True), 128
    )
    assert step == 2 and same is first
    ((_, _, _, changed),) = cache.prepare(buckets, 256)
    assert changed.chunks.tolist() == [0, 1, 2, 3]
    assert changed.meta is not first.meta
    assert changed.pointers is first.pointers

    # Same pointer row, new RNG coordinates: only the layout changes.
    shifted = [(*row[:3], 1, row[-1] + 2**40 + 20) for row in items]
    ((_, _, _, changed_offset),) = cache.prepare(bucket_items(shifted, MOMENT, uses_step=True), 256)
    assert changed_offset.meta[:, 3].tolist() == [0, 1, 2, 3]
    assert changed_offset.pointers is changed.pointers


def test_pointer_table_refresh_after_gradient_parameter_or_checkpoint_replacement():
    p = torch.nn.Parameter(torch.ones(17))
    opt = AdamWTriton([p])
    group = opt.param_groups[0]
    formats = opt._formats(group)
    cache = GroupTableCache()
    p.grad = torch.ones_like(p)

    def tables():
        items = opt._group_items(opt.param_groups[0], formats, 0)
        ((_, _, _, result),) = cache.prepare(
            bucket_items(items, opt._state_names, uses_step=True), 256
        )
        return result

    first = tables()
    assert tables() is first
    old_tensors = [p.grad, p.data, *opt.state[p].values()]
    p.grad = torch.zeros_like(p)
    second = tables()
    p.data = torch.full_like(p, 2)
    third = tables()
    opt.load_state_dict(copy.deepcopy(opt.state_dict()))
    fourth = tables()
    for prev, current, indices in (
        (first, second, [1]),
        (second, third, [0]),
        (third, fourth, [2, 3]),
    ):
        assert prev.meta is current.meta and prev.chunks is current.chunks
        assert (prev.pointers[0] != current.pointers[0]).nonzero().flatten().tolist() == indices
    assert len(old_tensors) == 4  # Keep the allocator from reusing old storage during this test.


@pytest.mark.parametrize("different_steps", [False, True])
def test_overlapping_parameter_writes_rejected_even_across_buckets(different_steps):
    shared = torch.ones(12)
    a, b = item(8, 0), item(8, 1, step=1 + different_steps)
    a[0].tensor, b[0].tensor = shared[:8], shared[4:]
    with pytest.raises(ValueError, match="overlap"):
        GroupTableCache().prepare(bucket_items([a, b], MOMENT, uses_step=True), 256)


def test_shared_gradients_and_disjoint_storage_views_are_allowed():
    shared = torch.ones(16)
    a, b = item(8, 0), item(8, 1)
    a[0].tensor, b[0].tensor = shared[:8], shared[8:]
    b[0].grad = a[0].grad
    cache = GroupTableCache()
    cache.prepare(bucket_items([a, b], MOMENT, uses_step=True), 256)

    # Repointing one gradient into a parameter must invalidate overlap validation.
    b[0].grad = a[0].tensor
    with pytest.raises(ValueError, match="overlap"):
        cache.prepare(bucket_items([a, b], MOMENT, uses_step=True), 256)


def test_layout_is_checked_even_when_addresses_match():
    p, state, *rest = item(16)
    p.tensor = p.tensor.view(4, 4)
    p.grad = p.grad.view(4, 4)
    state["m"] = state["m"].view(4, 4)
    cache = GroupTableCache()
    cache.prepare(bucket_items([(p, state, *rest)], MOMENT, uses_step=True), 256)
    state["m"] = state["m"].T
    with pytest.raises(ValueError, match="contiguous"):
        bucket_items([(p, state, *rest)], MOMENT, uses_step=True)


def test_parameter_hook_keeps_order_rng_indices_and_partial_failure():
    class Recorder(StateOptimizer):
        _state_names = MOMENT

        def _step_parameter(self, parameter, state, group, formats, step, seed, offset):
            calls.append((parameter, step, seed, offset))

    calls = []
    p, skipped, q, bad = [torch.nn.Parameter(torch.ones(2)) for _ in range(4)]
    opt = Recorder([{"params": [p, skipped], "sr_seed": 7}, {"params": [q, bad]}], {}, seed=3)
    p.grad = q.grad = torch.ones(2)
    bad.grad = torch.sparse_coo_tensor([[0]], [1.0], (2,), check_invariants=True)
    with pytest.raises(RuntimeError, match="sparse"):
        opt.step()
    assert calls == [(p, 1, update_key(7, 1, 0), 0), (q, 1, update_key(3, 1, 2), 0)]
    assert opt.state[p]["step"] == opt.state[q]["step"] == opt._rng_step() == 1
    assert skipped not in opt.state and bad not in opt.state


@pytest.mark.parametrize("value", [0, 1, 16, -1, 2**31, 2**63 + 7, 2**90 + 2**33 + 4])
def test_split_words_represents_low_64_bits(value):
    lo, hi = split_words(value)
    assert all(-(2**31) <= word < 2**31 for word in (lo, hi))
    assert ((hi & 0xFFFFFFFF) << 32) | (lo & 0xFFFFFFFF) == value % 2**64
