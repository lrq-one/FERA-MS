#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
python -u "$ROOT/ablation_studies/fera_ms_global_ace_ablation_20260730/pipeline_src/formal_pipeline.py" --aggregate
