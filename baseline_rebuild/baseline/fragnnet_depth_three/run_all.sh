#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SPLIT="${1:-random}"

for SEED in 42 43 44; do
  python "$BASELINE_ROOT/tools_local/run_formal_one.py" \
    --model fragnnet_d3 --split "$SPLIT" --seed "$SEED"
done
