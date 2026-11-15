"""Host-side bucket preparation and device table caching (no Triton dependency)."""

from collections import defaultdict
from dataclasses import dataclass

import torch

from ._optimizer import validate_no_overlap, validate_tensor_layout

# Use one tile per program until a bandwidth sweep justifies a larger chunk.
CHUNK_TILES = 1


@dataclass(frozen=True)
class _Row:
    tensors: tuple
    param_index: int
    numel: int
    offset: int


def bucket_items(items, state_names, *, uses_step):
    buckets = {}
    for parameter, state, param_index, step, offset in items:
        gradient = parameter.grad
        tensors = (parameter, gradient, *(state[name] for name in state_names))
        # A resize or restride can keep data_ptr unchanged. Check the layout
        # before comparing addresses or using a cached overlap result.
        validate_tensor_layout(tensors)
        if uses_step and (not isinstance(step, int) or step < 1):
            raise ValueError("step must be a positive integer")
        key = (parameter.device, tuple(t.dtype for t in tensors), step if uses_step else None)
        rows = buckets.setdefault(key, [])
        if numel := parameter.numel():
            rows.append(_Row(tensors, param_index, numel, int(offset) & 0xFFFFFFFF))
    return buckets


def _table(values, dtype, device):
    return torch.tensor(values, dtype=dtype, device="cpu").to(device, non_blocking=True)


@dataclass(frozen=True)
class _Tables:
    meta: torch.Tensor
    chunks: torch.Tensor
    pointers: torch.Tensor


@dataclass(frozen=True)
class _Entry:
    layout: tuple
    addresses: tuple
    stream: object
    tables: _Tables


class GroupTableCache:
    """Keep one set of tables per active bucket; keys exclude the changing AdamW step."""

    def __init__(self):
        self._entries = {}

    def prepare(self, buckets, block_size):
        layouts = {}
        addresses = {}
        for (device, dtypes, _), rows in buckets.items():
            if not rows:
                continue
            key = (device, dtypes, rows[0].param_index)
            layouts[key] = (block_size, tuple((r.numel, r.param_index, r.offset) for r in rows))
            addresses[key] = tuple(tuple(t.data_ptr() for t in r.tensors) for r in rows)

        if layouts.keys() != self._entries.keys() or any(
            (layout, addresses[key]) != (self._entries[key].layout, self._entries[key].addresses)
            for key, layout in layouts.items()
        ):
            # Bucket dispatch can reorder updates, so check dependencies even
            # across buckets. Equal addresses on different devices are unrelated.
            spans = defaultdict(list)
            for (device, dtypes, _), rows in buckets.items():
                if not rows:
                    continue
                sizes = tuple(t.element_size() for t in rows[0].tensors)
                for row, pointers in zip(rows, addresses[device, dtypes, rows[0].param_index]):
                    spans[device].extend(
                        (ptr, ptr + row.numel * size, role != 1)
                        for role, (ptr, size) in enumerate(zip(pointers, sizes))
                    )
            for device_spans in spans.values():
                validate_no_overlap(device_spans)

        streams = {
            device: torch.cuda.current_stream(device) if device.type == "cuda" else None
            for device in {key[0] for key in layouts}
        }
        entries = {}
        launches = []
        for (device, dtypes, step), rows in buckets.items():
            if not rows:
                continue
            key = (device, dtypes, rows[0].param_index)
            old = self._entries.get(key)
            layout, pointers, stream = layouts[key], addresses[key], streams[device]
            # Rebuild on a stream change so cached allocations and their
            # asynchronous uploads are only ever used on their owning stream.
            if old is not None and (layout, pointers, stream) == (
                old.layout,
                old.addresses,
                old.stream,
            ):
                entry = old
            else:
                if old is not None and stream == old.stream and layout == old.layout:
                    meta, chunks = old.tables.meta, old.tables.chunks
                else:
                    meta_rows, chunk_rows = [], []
                    chunk_size = CHUNK_TILES * block_size
                    for tensor_id, (numel, param_index, offset) in enumerate(layout[1]):
                        meta_rows.append((numel, len(chunk_rows), param_index, offset))
                        chunk_rows.extend([tensor_id] * ((numel + chunk_size - 1) // chunk_size))
                    meta = _table(meta_rows, torch.int64, device)
                    chunks = _table(chunk_rows, torch.int32, device)
                ptr_table = (
                    old.tables.pointers
                    if old is not None and stream == old.stream and pointers == old.addresses
                    else _table(pointers, torch.int64, device)
                )
                entry = _Entry(layout, pointers, stream, _Tables(meta, chunks, ptr_table))
            entries[key] = entry
            # The caller holds all input tensor objects through kernel enqueue.
            launches.append((device, step, rows, entry.tables))
        self._entries = entries
        return launches


def prepare_group(optimizer, group, formats, items, validate_device):
    """Collect and validate the entire group before the first update launch."""
    buckets = bucket_items(items, optimizer._state_names, uses_step=optimizer._uses_step)
    if not buckets:
        return []
    optimizer._validate_group(group)
    for fmt in formats:
        fmt.validate("triton")
    for device in {key[0] for key in buckets}:
        validate_device(device, *(fmt.dtype for fmt in formats))

    caches = getattr(optimizer, "_triton_tables", {})
    key = id(group)
    if key not in caches:
        # Checkpoint loads replace param_groups; keep no caches for old groups.
        live = {id(g) for g in optimizer.param_groups}
        caches = {k: v for k, v in caches.items() if k in live}
        caches[key] = GroupTableCache()
        optimizer._triton_tables = caches
    return caches[key].prepare(buckets, group["block_size"])
