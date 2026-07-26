#!/usr/bin/env bash

set -u

ROOT="/home/lwh/projects/lrq2/fragnnet-main/ms2spectra_v1_r119"
ABLATION_ROOT="$ROOT/ablation_studies/fera_ms_core_ablation"

cd "$ROOT" || {
    echo "无法进入项目目录：$ROOT"
    exit 1
}

export PYTHONPATH="$ROOT/code/src:$ROOT/code:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

mkdir -p \
  "$ABLATION_ROOT/logs" \
  "$ABLATION_ROOT/results"

python -u \
  "$ABLATION_ROOT/src/preflight.py" \
  2>&1 | tee \
  "$ABLATION_ROOT/logs/00_preflight.log"

CODE=${PIPESTATUS[0]}

echo
echo "PREFLIGHT_EXIT_CODE=$CODE"
echo "LOG=$ABLATION_ROOT/logs/00_preflight.log"
echo "REPORT=$ABLATION_ROOT/results/preflight_report.json"

exit "$CODE"
