#!/usr/bin/env bash
#SBATCH --job-name=sr-main-timing
#SBATCH --account=torch_pr_103_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH --constraint=h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH="/scratch/kt2664/venvs/tilde/bin:$PATH"
export PYTHONPATH="$PWD/.tools"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export TMPDIR="/scratch/kt2664/Stochastic-Rounding-for-Numerical-Optimization-with-State-Quantization/tmp/main-state-timing-20261002T011750Z"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/torchinductor"
export TRITON_CACHE_DIR="$TMPDIR/triton"
export SR_TIMING_RUN="/scratch/kt2664/Stochastic-Rounding-for-Numerical-Optimization-with-State-Quantization/results/raw/main-state-timing-20261002T011750Z"
export PYTEST_ADDOPTS="--junitxml=$SR_TIMING_RUN/gpu-tests.xml --durations=15"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
finish() {
    local result=$?
    trap - EXIT
    python - "$result" <<'PYEND'
from datetime import datetime, timezone
from pathlib import Path
import json,os,sys
path=Path(os.environ['SR_TIMING_RUN'])/'status.json'
status=json.loads(path.read_text()) if path.exists() else {}
status.update(state='completed' if int(sys.argv[1])==0 else 'failed',exit_code=int(sys.argv[1]),finished_utc=datetime.now(timezone.utc).isoformat())
path.write_text(json.dumps(status,indent=2)+'\n')
PYEND
    exit "$result"
}
trap finish EXIT
python - <<'PYENV'
from datetime import datetime,timezone
import hashlib,json,os,subprocess
from pathlib import Path
import torch,triton
out=Path(os.environ['SR_TIMING_RUN'])
source=json.loads((out/'source.json').read_text())
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==source['commit']
for name,digest in source['sources'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
assert torch.cuda.is_available(), 'CUDA unavailable'
status={'state':'running','started_utc':datetime.now(timezone.utc).isoformat(),'job_id':os.environ['SLURM_JOB_ID'],'commit':source['commit'],'gpu':torch.cuda.get_device_name(),'device_count':torch.cuda.device_count(),'command':['bash','scripts/run_state_validation.sh',str(out/'benchmark')]}
(out/'status.json').write_text(json.dumps(status,indent=2)+'\n')
print(json.dumps(status,indent=2),flush=True)
PYENV
bash scripts/run_state_validation.sh "$SR_TIMING_RUN/benchmark"
