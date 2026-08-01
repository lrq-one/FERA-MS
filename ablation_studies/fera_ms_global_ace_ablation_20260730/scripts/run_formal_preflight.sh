#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
export PYTHONPATH="$ROOT/code/src:$ROOT/code:$ROOT"
python -u "$ROOT/ablation_studies/fera_ms_global_ace_ablation_20260730/scripts/run_formal_preflight.py"
