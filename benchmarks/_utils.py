from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

import torch


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def cuda_samples_ms(function, *, warmup: int, repetitions: int, before_call=None) -> list[float]:
    if warmup < 0 or repetitions <= 0:
        raise ValueError("warmup must be nonnegative and repetitions must be positive")
    # Always finish compilation and lazy initialization outside the timed region,
    # including when the caller explicitly requests zero warmup iterations.
    if before_call is not None:
        before_call()
    function()
    for _ in range(warmup):
        if before_call is not None:
            before_call()
        function()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # CUDA events themselves are lazily initialized.
    start.record()
    end.record()
    end.synchronize()
    samples = []
    for _ in range(repetitions):
        if before_call is not None:
            before_call()
        start.record()
        function()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    if any(not math.isfinite(value) or value <= 0 for value in samples):
        raise RuntimeError("CUDA timing returned a nonpositive or nonfinite sample")
    return samples


def cuda_median_ms(function, *, warmup: int, repetitions: int) -> float:
    return statistics.median(cuda_samples_ms(function, warmup=warmup, repetitions=repetitions))


def timing_row(name: str, n: int, samples: list[float], bytes_per_element: int) -> dict:
    row = result_row(name, n, statistics.median(samples), bytes_per_element)
    deciles = statistics.quantiles(samples, n=10, method="inclusive") if len(samples) > 1 else []
    row.update(
        p10_ms=deciles[0] if deciles else samples[0],
        p90_ms=deciles[-1] if deciles else samples[0],
        samples_ms=json.dumps(samples),
        timing="cuda_events",
    )
    return row


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
        writer = csv.DictWriter(sys.stdout, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
        return

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
