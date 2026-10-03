# H200 optimizer timing results

Completed job **18994353** on H200 node `gh105` in **14m 17s**. The run finished on
2026-10-02 at 01:43 UTC (2026-10-01, 21:43 America/New_York). It used Python 3.13.5,
PyTorch 2.11.0+cu128, CUDA 12.8, and Triton 3.6.0.

The source is commit [de62a14](https://github.com/KristiTopollai/stochastic-rounding-pytorch/commit/de62a14b33163f64e6d61178288522075ccf71f2)
plus the [recorded test-isolation patch](run/source.patch), now included in the
repository. The archive records this commit as `ef5c8b4` before author attribution
was corrected; its source tree is unchanged. All 43 original artifacts are preserved byte for byte under
[run/](run/), with SHA-256 checksums in [manifest.json](manifest.json).

## Full optimizer steps

These measurements time complete `optimizer.step()` calls with FP32 parameters
and gradients and BF16 states. Each method advances its own optimizer trajectory.
The quadratic label gives **total parameters / tensor count**; the MLP has width
512, four hidden layers, and batch size 64. Gradient computation is outside timing.

Times below are synchronized wall latency in **microseconds**, including the
event wait, reported as the median of three process medians (100 samples per
process). Compiled SR compiles each tensor update while optimizer bookkeeping
remains in Python. Archived `native_nr` rows retain the eager FP32 reference
using PyTorch BF16 casts; they are omitted from these tables.

| Optimizer | Workload: total / tensors | Triton NR | Eager SR | Compiled SR | Triton SR |
| --- | --- | ---: | ---: | ---: | ---: |
| AdamW | Quadratic 1,048,576 / 1 | 84.49 | 647.75 | 135.37 | 85.19 |
| AdamW | Quadratic 1,048,576 / 32 | 1,279.72 | 18,701.92 | 2,564.69 | 1,296.36 |
| AdamW | Quadratic 16,777,216 / 1 | 145.51 | 5,033.19 | 381.07 | 145.90 |
| AdamW | Quadratic 16,777,216 / 32 | 1,304.61 | 18,717.01 | 2,593.57 | 1,302.90 |
| AdamW | MLP 813,072 / 10 | 516.81 | 6,209.80 | 1,198.95 | 520.50 |
| SGDM | Quadratic 1,048,576 / 1 | 78.23 | 345.00 | 115.28 | 79.68 |
| SGDM | Quadratic 1,048,576 / 32 | 1,055.79 | 9,105.32 | 1,955.62 | 1,067.28 |
| SGDM | Quadratic 16,777,216 / 1 | 122.31 | 2,494.40 | 147.20 | 122.94 |
| SGDM | Quadratic 16,777,216 / 32 | 1,071.03 | 9,169.62 | 1,952.92 | 1,080.12 |
| SGDM | MLP 813,072 / 10 | 453.51 | 3,141.02 | 942.90 | 455.33 |

Speedups are medians of **within-repeat ratios**, rather than ratios of the
aggregate times above. Overhead is `100 * (SR / NR - 1)` in each repeat; its range
is the minimum to maximum of three repeats, not a confidence interval.

| Optimizer | Workload: total / tensors | vs. eager SR | vs. compiled SR | Matched Triton NR overhead: median [range] |
| --- | --- | ---: | ---: | ---: |
| AdamW | Quadratic 1,048,576 / 1 | 7.67× | 1.57× | +1.50% [-0.44%, +1.56%] |
| AdamW | Quadratic 1,048,576 / 32 | 14.23× | 1.96× | +1.83% [+1.30%, +1.86%] |
| AdamW | Quadratic 16,777,216 / 1 | 34.50× | 2.61× | +0.27% [+0.24%, +0.36%] |
| AdamW | Quadratic 16,777,216 / 32 | 14.27× | 1.98× | -0.13% [-0.34%, +0.28%] |
| AdamW | MLP 813,072 / 10 | 11.90× | 2.28× | +1.44% [-1.10%, +1.62%] |
| SGDM | Quadratic 1,048,576 / 1 | 4.31× | 1.45× | +1.14% [-0.46%, +1.86%] |
| SGDM | Quadratic 1,048,576 / 32 | 8.43× | 1.80× | +1.09% [+1.01%, +1.56%] |
| SGDM | Quadratic 16,777,216 / 1 | 20.29× | 1.20× | +0.25% [+0.11%, +0.93%] |
| SGDM | Quadratic 16,777,216 / 32 | 8.45× | 1.80× | +0.70% [+0.20%, +0.85%] |
| SGDM | MLP 813,072 / 10 | 6.86× | 2.05× | +0.74% [-0.16%, +1.31%] |

On the largest single tensor, Triton SR is **2.61× faster than compiled SR for
AdamW** and **1.20× for SGDM**. Matched Triton NR overhead is **+0.27%** and
**+0.25%**, respectively. Across the ten full-step workloads, median SR overhead
versus matched Triton NR ranges from **−0.13% to +1.83%**. Differences this small
should not be interpreted as a precise isolated cost of random-number generation.

## Fixed-input update microbenchmarks

These CUDA-event timings cover in-place updates with preallocated buffers.
Inputs, RNG offsets, and the AdamW step are reset before every sample, outside
timing. Here the shape is **elements per tensor × number of tensors**, unlike
the total-parameter convention for full optimizer steps. The table shows Triton
variants and compiled SR; all nine eager/compiled/Triton × BF16-SR/BF16-NR/FP32
methods are retained in the raw files and [summary.csv](report/summary.csv).

| Optimizer | Tensor shape | Compiled BF16 SR (µs) | Triton BF16 SR (µs) | Triton BF16 NR (µs) | Triton FP32 state (µs) | SR speedup vs. compiled SR |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| AdamW | 1,024 × 1 | 108.88 | 56.21 | 55.55 | 55.42 | 1.95× |
| AdamW | 1,048,576 × 1 | 116.77 | 56.16 | 55.42 | 55.68 | 2.09× |
| AdamW | 16,777,216 × 1 | 495.89 | 84.03 | 83.84 | 114.99 | 5.89× |
| AdamW | 4,096 × 32 | 2,684.56 | 1,202.54 | 1,194.74 | 1,167.20 | 2.19× |
| AdamW | 65,536 × 32 | 2,664.06 | 1,197.87 | 1,198.37 | 1,182.80 | 2.25× |
| SGDM | 1,024 × 1 | 88.90 | 49.39 | 48.64 | 48.50 | 1.80× |
| SGDM | 1,048,576 × 1 | 88.22 | 47.86 | 48.34 | 48.58 | 1.82× |
| SGDM | 16,777,216 × 1 | 86.18 | 67.97 | 67.94 | 83.10 | 1.27× |
| SGDM | 4,096 × 32 | 2,122.66 | 1,012.37 | 995.52 | 993.98 | 2.10× |
| SGDM | 65,536 × 32 | 2,133.84 | 1,006.24 | 990.64 | 989.55 | 2.06× |

## Validation and measurement scope

- **299 tests passed**, with only the two tests requiring a second GPU skipped.
  The separate CPU preflight passed all **117 tests**. See the [test log](run/benchmark/tests.log)
  and [JUnit results](run/gpu-tests.xml).
- Fifteen CSVs contain **420 timing rows** across three process repeats and
  **42,000 timed updates**. Full steps record wall, CUDA-event, and CPU-submission
  latency; microbenchmarks record CUDA-event latency. This gives **72,000 recorded
  latency values**. Every sample count, median, decile, stored speedup, and method
  matrix was audited; every measured case passed its correctness check.
- Full-step measurements use 120 untimed rehearsal steps, restart, 20 warmup
  steps, and 100 measured steps (steps 21–120). RNG offsets advance across states
  and steps. **No compiled graphs were generated during timing**.
- Triton uses block size 256 and the default four warps. CUDA graphs are disabled.
  Each repeat randomizes method order using order seeds 0, 1, and 2; data/RNG seed
  is 7. Compilation, gradient computation, and initialization are excluded.
- CUDA-event timing includes host submission gaps. Wall timing includes event
  synchronization overhead, while CPU-submission timing ends when `step()` returns.
  These metrics measure different parts of the call and should not be mixed.
- All repeats ran in one allocation on one H200. The ranges describe these three
  process repeats; they are not cross-device uncertainty estimates. The results
  describe synthetic optimizer timing, not end-to-end training throughput or
  convergence quality.

## Artifacts and reproduction

- [Summary medians and process ranges](report/summary.csv): all methods and timing channels.
- [Paired speedups and overhead ranges](report/comparisons.csv).
- [Audit counts](report/audit.json), [source hashes](run/source.json),
  [environment](run/benchmark/environment.json), [submission](run/submission.json),
  [Slurm accounting](slurm-accounting.txt), and [full job log](run/slurm-18994353.out).

Reproduce the audit and summaries without PyTorch or a GPU:

```bash
python results/h200-states-20261002/summarize.py --output results/raw/states-summary-reproduced
```

Reproduce the tests and measurements from the recorded source with CUDA/Triton:

```bash
bash scripts/run_state_validation.sh results/raw/states-rerun
```

The archived batch script records the H200 resource request and scratch/cache
settings used for this run; its absolute paths are specific to the original
cluster. Use a fresh result directory when rerunning.
