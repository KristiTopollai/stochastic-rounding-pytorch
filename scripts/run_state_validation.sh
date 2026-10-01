#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
result_dir=${1:-results/raw/states-$(date -u +%Y%m%dT%H%M%SZ)}
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
python -m pytest 2>&1 | tee "$result_dir/tests.log"
for repeat in 0 1 2; do
  for optimizer in sgdm adamw; do
    python -m "benchmarks.bench_$optimizer" --sizes 1024 1048576 16777216 \
      --order-seed "$repeat" --output "$result_dir/$optimizer-single-r$repeat.csv" \
      2>&1 | tee "$result_dir/$optimizer-single-r$repeat.log"
    python -m "benchmarks.bench_$optimizer" --sizes 4096 65536 --tensors 32 \
      --order-seed "$repeat" --output "$result_dir/$optimizer-multi-r$repeat.csv" \
      2>&1 | tee "$result_dir/$optimizer-multi-r$repeat.log"
  done
done
