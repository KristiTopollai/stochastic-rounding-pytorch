"""Audit the archived run and reproduce its summaries using the standard library."""

import argparse
import csv
import hashlib
import json
import math
import statistics
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

STEP_METHODS = ("native_nr", "triton_nr", "eager_sr", "compiled_sr", "triton_sr")
UPDATE_METHODS = tuple(
    f"{backend}_{policy}"
    for backend in ("eager", "compiled", "triton")
    for policy in ("bf16_sr", "bf16_nr", "fp32")
)
WORKLOADS = {
    **{f"quadratic-{n}-{t}": (n, t) for n in (1048576, 16777216) for t in (1, 32)},
    "mlp-512-4-64": (813072, 10),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label):
    require(math.isclose(float(actual), expected, rel_tol=1e-10, abs_tol=1e-12), label)


def samples(row, prefix):
    values = json.loads(row[f"{prefix}samples_ms"])
    require(len(values) == 100 and all(math.isfinite(v) and v > 0 for v in values), "samples")
    deciles = statistics.quantiles(values, n=10, method="inclusive")
    median = statistics.median(values)
    for name, expected in (("median_ms", median), ("p10_ms", deciles[0]), ("p90_ms", deciles[-1])):
        close(row[prefix + name], expected, prefix + name)
    return median


def tests(path, total, skipped):
    root = ET.parse(path).getroot()
    cases = list(root.iter("testcase"))
    require(len(cases) == total, f"test count: {path}")
    require(
        not any(c.find("failure") is not None or c.find("error") is not None for c in cases),
        "tests failed",
    )
    names = {c.attrib["name"] for c in cases if c.find("skipped") is not None}
    require(names == set(skipped), f"unexpected skips: {names}")


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def audit(bundle, output):
    manifest = json.loads((bundle / "manifest.json").read_text())
    archived = {str(p.relative_to(bundle)) for p in (bundle / "run").rglob("*") if p.is_file()}
    require(archived == set(manifest["files"]), "archive file matrix")
    for name, expected in manifest["files"].items():
        content = (bundle / name).read_bytes()
        require(len(content) == expected["bytes"], f"file size: {name}")
        require(hashlib.sha256(content).hexdigest() == expected["sha256"], f"checksum: {name}")
    run = bundle / "run"
    status = json.loads((run / "status.json").read_text())
    environment = json.loads((run / "benchmark/environment.json").read_text())
    source = json.loads((run / "source.json").read_text())
    submission = json.loads((run / "submission.json").read_text())
    require(status["state"] == "completed" and status["exit_code"] == 0, "run incomplete")
    require(status["job_id"] == submission["job_id"] == manifest["job_id"] == "18994353", "job ID")
    require(
        all(
            meta["commit"] == manifest["source_commit"]
            for meta in (status, environment, source, submission)
        ),
        "source commit",
    )
    patch = (run / "source.patch").read_text()
    require(environment["tracked_changes"] == source["tracked_changes"] == patch, "source patch")
    tests(run / "cpu-tests.xml", 117, ())
    tests(
        run / "gpu-tests.xml",
        301,
        (
            "test_state_update_uses_tensor_device[sgdm]",
            "test_state_update_uses_tensor_device[adamw]",
        ),
    )
    groups = defaultdict(dict)
    scalar_samples = 0
    timing_rows = 0
    expected_files = {}
    for repeat in range(3):
        expected_files[f"optimizer-steps-r{repeat}.csv"] = 50
        for kind in ("adamw", "sgdm"):
            expected_files[f"{kind}-single-r{repeat}.csv"] = 27
            expected_files[f"{kind}-multi-r{repeat}.csv"] = 18
    require(submission["expected_csv_rows"] == expected_files, "submission matrix")
    require(
        {p.name for p in (run / "benchmark").glob("*.csv")} == set(expected_files), "CSV matrix"
    )
    for name, count in sorted(expected_files.items()):
        repeat = int(name[-5])
        step = name.startswith("optimizer-steps")
        with (run / "benchmark" / name).open() as stream:
            rows = list(csv.DictReader(stream))
        require(len(rows) == count, f"row count: {name}")
        keys = set()
        paired = {}
        for row in rows:
            kind, method = row["optimizer"], row["method"]
            for field, expected in (
                ("order_seed", repeat),
                ("warmup", 20),
                ("repetitions", 100),
                ("seed", 7),
            ):
                require(int(row[field]) == expected, field)
            require(
                row["correctness"] == "passed" and row["parameter_dtype"] == "fp32",
                "correctness / dtype",
            )
            for field in ("torch", "triton", "python"):
                require(row[field + "_version"] == environment[field], field + " version")
            require(row["gpu"] == environment["gpu"] == "NVIDIA H200", "GPU")
            require(
                row["block_size"] == ("256" if method.startswith("triton") else ""), "block size"
            )
            elements, tensors = int(row["elements"]), int(row["tensors"])
            states = 2 if kind == "adamw" else 1
            if step:
                workload = row["problem"]
                require((elements, tensors) == WORKLOADS[workload], "workload dimensions")
                require(row["timing"] == "optimizer_step_synchronized_wall", "step timing")
                require(row["state_dtype"] == "bf16", "step state dtype")
                require(
                    row["rounding"] == ("nearest" if method.endswith("nr") else "stochastic"),
                    "step rounding",
                )
                require(int(row["state_bytes"]) == 2 * states * elements, "state bytes")
                require(int(row["final_rng_offset"]) == 120 * states * elements, "RNG offset")
                require(int(row["timed_compiled_graphs"]) == 0, "timed compilation")
                require(
                    (int(row["compiled_graphs"]) > 0) == (method == "compiled_sr"),
                    "compiled graphs",
                )
                require(
                    (
                        int(row["first_timed_step"]),
                        int(row["last_timed_step"]),
                        int(row["rehearsal_steps"]),
                    )
                    == (21, 120, 120),
                    "step counters",
                )
                shapes = json.loads(row["shapes"])
                require(
                    len(shapes) == tensors and sum(math.prod(s) for s in shapes) == elements,
                    "shapes",
                )
                prefixes = (("wall", ""), ("cuda", "cuda_"), ("submit", "submit_"))
            else:
                n = int(row["tensor_elements"])
                require(elements == n * tensors, "update dimensions")
                workload = f"{n}x{tensors}"
                require(
                    row["reset"] == "outside_timing" and row["timing"] == "cuda_events",
                    "update timing",
                )
                fp32 = method.endswith("fp32")
                require(
                    row["state_dtype"] == ("torch.float32" if fp32 else "torch.bfloat16"),
                    "update state dtype",
                )
                require(
                    row["rounding"] == ("stochastic" if method.endswith("sr") else "nearest"),
                    "update rounding",
                )
                require(row["step"] == ("100" if kind == "adamw" else ""), "update step")
                close(
                    row["elements_per_second"],
                    elements * 1000 / float(row["median_ms"]),
                    "throughput",
                )
                traffic = 12 + 2 * states * (4 if fp32 else 2)
                close(
                    row["effective_gbps"],
                    elements * traffic / float(row["median_ms"]) / 1e6,
                    "bandwidth",
                )
                prefixes = (("cuda", ""),)
            key = (kind, workload, method)
            require(key not in keys, "duplicate row")
            keys.add(key)
            paired[key] = row
            for metric, prefix in prefixes:
                group = ("step" if step else "update", kind, workload, metric, method)
                require(repeat not in groups[group], "duplicate repeat")
                groups[group][repeat] = samples(row, prefix)
                scalar_samples += 100
            timing_rows += 1
        if step:
            expected = {
                (k, w, m) for k in ("adamw", "sgdm") for w in WORKLOADS for m in STEP_METHODS
            }
            for (kind, workload, _), row in paired.items():
                for baseline in ("native_nr", "triton_nr"):
                    base = paired[kind, workload, baseline]
                    for prefix in ("", "cuda_"):
                        close(
                            row[f"{prefix}speedup_vs_{baseline}"],
                            float(base[prefix + "median_ms"]) / float(row[prefix + "median_ms"]),
                            "stored speedup",
                        )
        else:
            kind, layout = name.split("-")[:2]
            dimensions = (
                ((1024, 1), (1048576, 1), (16777216, 1))
                if layout == "single"
                else ((4096, 32), (65536, 32))
            )
            expected = {(kind, f"{n}x{t}", m) for n, t in dimensions for m in UPDATE_METHODS}
        require(keys == expected, f"method/workload matrix: {name}")
    summary, comparisons = [], []
    for key, repeats in sorted(groups.items()):
        require(set(repeats) == {0, 1, 2}, "incomplete repeats")
        family, kind, workload, metric, method = key
        identity = dict(zip(("family", "optimizer", "workload", "metric", "method"), key))
        times = list(repeats.values())
        summary.append(
            {
                **identity,
                "repeats": 3,
                "median_us": statistics.median(times) * 1000,
                "min_us": min(times) * 1000,
                "max_us": max(times) * 1000,
            }
        )
        if method != ("triton_sr" if family == "step" else "triton_bf16_sr"):
            continue
        baselines = (
            ("native_nr", "triton_nr", "eager_sr", "compiled_sr")
            if family == "step"
            else ("eager_bf16_sr", "compiled_bf16_sr", "triton_bf16_nr", "triton_fp32")
        )
        for baseline in baselines:
            base = groups[(family, kind, workload, metric, baseline)]
            speedups = [base[i] / repeats[i] for i in range(3)]
            overheads = [(repeats[i] / base[i] - 1) * 100 for i in range(3)]
            comparisons.append(
                {
                    **identity,
                    "baseline": baseline,
                    "median_speedup": statistics.median(speedups),
                    "min_speedup": min(speedups),
                    "max_speedup": max(speedups),
                    "median_overhead_pct": statistics.median(overheads),
                    "min_overhead_pct": min(overheads),
                    "max_overhead_pct": max(overheads),
                }
            )
    require(
        (timing_rows, scalar_samples, len(summary), len(comparisons)) == (420, 72000, 240, 160),
        "aggregate matrix",
    )
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "summary.csv", summary)
    write_csv(output / "comparisons.csv", comparisons)
    result = {
        "job_id": status["job_id"],
        "source_commit": source["commit"],
        "archived_files": len(archived),
        "csv_files": 15,
        "timing_rows": timing_rows,
        "timed_updates": 42000,
        "recorded_latency_values": scalar_samples,
        "summary_rows": len(summary),
        "comparison_rows": len(comparisons),
        "tests_passed": 299,
        "tests_skipped": 2,
        "cpu_tests_passed": 117,
        "timed_compiled_graphs": 0,
    }
    (output / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.bundle, args.output), indent=2))
