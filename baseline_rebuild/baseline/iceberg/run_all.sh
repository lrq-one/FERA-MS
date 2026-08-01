#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT="${FERA_MS_ICEBERG_SOURCE:-$BASELINE_ROOT/shared/iceberg_core}"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

bash benchmark_audit/run_one_pl_seed.sh "$SCRIPT_DIR/configs/iceberg_core_reference.yml" iceberg_core_reference
bash benchmark_audit/run_one_pl_seed.sh "$SCRIPT_DIR/configs/iceberg_core_replicate_alpha.yml" iceberg_core_replicate_alpha
bash benchmark_audit/run_one_pl_seed.sh "$SCRIPT_DIR/configs/iceberg_core_replicate_beta.yml" iceberg_core_replicate_beta
