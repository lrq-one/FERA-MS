#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT="${FERA_MS_BASELINE_SOURCE:-$BASELINE_ROOT/shared/fragnnet_main}"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

bash benchmark_audit/run_one_pl_seed.sh "$SCRIPT_DIR/configs/neims_ace_reference.yml" neims_ace_reference
bash benchmark_audit/run_one_pl_seed.sh "$SCRIPT_DIR/configs/neims_ace_replicate_alpha.yml" neims_ace_replicate_alpha
bash benchmark_audit/run_one_pl_seed.sh "$SCRIPT_DIR/configs/neims_ace_replicate_beta.yml" neims_ace_replicate_beta
