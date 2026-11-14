# Optimizer regression fixtures

`optimizer_v01.pt` records parameter/state bytes and RNG offsets from the
pre-refactor baseline `393c527`, captured with PyTorch 2.13.0 on macOS ARM CPU.
Its 24 cases cover AdamW and SGDM, eager and compiled (`backend="eager"`),
FP32/BF16 parameters, FP32/BF16 nearest/stochastic states, parameter groups,
missing gradients, and counter wraparound over seven steps.

`baseline_393c527.zip` contains the matching frozen `sr_states` source, produced
with `git archive --format=zip 393c527 sr_states`. On other CPU environments,
`tests/test_state_formats.py` replays this source locally before comparing bytes;
it does not assume that FP32 arithmetic is identical across platforms or Torch
versions. The replay and fixture comparisons use FP32 states and BF16 nearest
states for each optimizer and parameter dtype. The committed capture retains the
original stochastic cases and offsets for reference. RNG stream bytes are
checked against the current specification in `tests/test_rng_streams.py`.
That test module also runs the frozen optimizer in a subprocess to write a
legacy checkpoint and verifies that it loads and resumes with the new RNG.

To capture the baseline, copy `scripts/record_optimizer_golden.py` into the frozen
baseline's `scripts/` directory, then run it from that source tree:

```sh
python scripts/record_optimizer_golden.py --output optimizer_v01.pt
```

The generator refuses to overwrite existing captures. For GPU regression, add
`--device cuda --source-commit 393c527b6c4d859759eef6a7bce8509599572dd8`, then set
`SR_STATES_GPU_GOLDEN` to the capture when running `tests/test_minifloat_gpu.py`.
Capture and comparison must use the same GPU/software environment. Compiled eager
checks graph tracing; the separate Inductor tests check real code generation.

Library execution and GPU baseline regressions were validated on H200, and the
complete suite including FP8 state validation on L40S, using Python 3.13.5,
PyTorch 2.11.0+cu128, CUDA 12.8, and Triton 3.6.0.
