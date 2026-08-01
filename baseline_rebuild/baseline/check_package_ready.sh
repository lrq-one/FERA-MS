#!/usr/bin/env bash
set -euo pipefail

BASELINE_ROOT="${FERA_MS_BASELINE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
MAIN="${FERA_MS_BASELINE_SOURCE:-$BASELINE_ROOT/shared/fragnnet_main}"
ICEBERG="${FERA_MS_ICEBERG_SOURCE:-$BASELINE_ROOT/shared/iceberg_core}"

for script in \
  "$BASELINE_ROOT/neims/run_all.sh" \
  "$BASELINE_ROOT/massformer/run_all.sh" \
  "$BASELINE_ROOT/fragnnet_d3/run_all.sh" \
  "$BASELINE_ROOT/graff_ms/run_all.sh" \
  "$BASELINE_ROOT/iceberg/run_all.sh" \
  "$BASELINE_ROOT/fiora/run_final.sh"; do
  bash -n "$script"
done

test -f "$MAIN/benchmark_audit/run_one_pl_seed.sh" || {
  echo "Set FERA_MS_BASELINE_SOURCE to the external baseline source checkout." >&2
  exit 2
}
test -f "$ICEBERG/benchmark_audit/run_one_pl_seed.sh" || {
  echo "Set FERA_MS_ICEBERG_SOURCE to the external ICEBERG source checkout." >&2
  exit 2
}

python "$BASELINE_ROOT/tools_local/check_local_runtime.py"
echo "BASELINE_PREFLIGHT_OK"
