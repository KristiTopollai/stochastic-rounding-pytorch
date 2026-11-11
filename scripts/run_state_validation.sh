#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
result_dir=""
tests_only=0
for argument in "$@"; do
  case "$argument" in
    --tests-only) tests_only=1 ;;
    *)
      if [[ -n "$result_dir" ]]; then
        printf '%s\n' "Usage: $0 [result-directory] [--tests-only]" >&2
        exit 2
      fi
      result_dir=$argument
      ;;
  esac
done
result_dir=${result_dir:-results/raw/states-$(date -u +%Y%m%dT%H%M%SZ)}
mkdir -p "$result_dir"
if [[ -e "$result_dir/environment.json" ]]; then
  printf '%s\n' "Results already exist: $result_dir" >&2
  exit 1
fi
python - "$result_dir/environment.json" <<'PY'
import json
import platform
import subprocess
import sys
from pathlib import Path
import torch
from sr_states.triton import is_triton_available
if not is_triton_available():
    raise SystemExit("CUDA and Triton are required")
import triton
metadata = {
    "python": platform.python_version(), "torch": torch.__version__,
    "triton": triton.__version__, "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(),
    "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "tracked_changes": subprocess.check_output(["git", "diff", "HEAD", "--", "sr_states", "benchmarks", "tests", "scripts"], text=True),
}
Path(sys.argv[1]).write_text(json.dumps(metadata, indent=2) + "\n")
PY
# Replay the frozen pre-refactor package on the same GPU/software environment.
python - "$result_dir" <<'PY_BASELINE'
import shutil
import sys
import zipfile
from pathlib import Path
result = Path(sys.argv[1]).resolve()
baseline = result / "baseline-source"
with zipfile.ZipFile("tests/golden/baseline_393c527.zip") as archive:
    archive.extractall(baseline)
(baseline / "scripts").mkdir()
shutil.copyfile("scripts/record_optimizer_golden.py", baseline / "scripts/record_optimizer_golden.py")
PY_BASELINE
python "$result_dir/baseline-source/scripts/record_optimizer_golden.py" \
  --device cuda --source-commit 393c527b6c4d859759eef6a7bce8509599572dd8 \
  --output "$result_dir/optimizer_v01_cuda.pt" 2>&1 | tee "$result_dir/golden-capture.log"
export SR_STATES_GPU_GOLDEN="$result_dir/optimizer_v01_cuda.pt"
python -m pytest 2>&1 | tee "$result_dir/tests.log"
if [[ "$tests_only" == 1 ]]; then
  exit 0
fi
for repeat in 0 1 2; do
  for optimizer in sgdm adamw; do
    python -m "benchmarks.bench_$optimizer" --sizes 1024 1048576 16777216 \
      --order-seed "$repeat" --output "$result_dir/$optimizer-single-r$repeat.csv" \
      2>&1 | tee "$result_dir/$optimizer-single-r$repeat.log"
    python -m "benchmarks.bench_$optimizer" --sizes 4096 65536 --tensors 32 \
      --order-seed "$repeat" --output "$result_dir/$optimizer-multi-r$repeat.csv" \
      2>&1 | tee "$result_dir/$optimizer-multi-r$repeat.log"
  done
  python -m benchmarks.bench_optimizer_steps --sizes 1048576 16777216 --tensors 1 32 \
    --order-seed "$repeat" --output "$result_dir/optimizer-steps-r$repeat.csv" \
    2>&1 | tee "$result_dir/optimizer-steps-r$repeat.log"
done
