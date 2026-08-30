from __future__ import annotations

import csv
import statistics
from pathlib import Path

import torch


def cuda_median_ms(function, *, warmup: int, repetitions: int) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()

    samples = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def result_row(name: str, n: int, milliseconds: float, bytes_per_element: int) -> dict:
    seconds = milliseconds / 1e3
    return {
        "method": name,
        "elements": n,
        "median_ms": milliseconds,
        "elements_per_second": n / seconds,
        "effective_gbps": n * bytes_per_element / seconds / 1e9,
    }


def write_rows(rows: list[dict], destination: str | None) -> None:
    if not rows:
        return
    if destination is None:
        writer = csv.DictWriter(__import__("sys").stdout, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
        return

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
