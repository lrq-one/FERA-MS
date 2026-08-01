#!/usr/bin/env bash
set -euo pipefail

CFG="$1"
JOB="$2"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p benchmark_audit/checkpoints/"$JOB"
mkdir -p benchmark_audit/profile/"$JOB"

echo "============================================================"
echo "JOB: $JOB"
echo "CFG: $CFG"
echo "REPO_ROOT: $REPO_ROOT"
echo "CKPT_DIR: $REPO_ROOT/benchmark_audit/checkpoints/$JOB"
echo "PROFILE_DIR: $REPO_ROOT/benchmark_audit/profile/$JOB"
echo "START: $(date)"
echo "============================================================"

python - <<'PY2'
import torch
print("python check ok")
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY2

python scripts/run_pl_model_fit.py \
  -t config/template.yml \
  -c "$CFG" \
  -w disabled \
  -j "$JOB"

echo "============================================================"
echo "DONE: $JOB"
echo "END: $(date)"
echo "============================================================"
