#!/usr/bin/env bash
set -euo pipefail
ROOT="${FERA_MS_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
if [[ "${1:-}" == "--dry-run" ]]; then
  for seed in 42 43 44; do FERA_DRY_RUN=1 "$ROOT/ablation_studies/fera_ms_global_ace_ablation_20260730/run_one_seed.sh" "$seed"; done
  echo "AGGREGATION\tDRY_RUN"
  exit 0
fi
for seed in 42 43 44; do "$ROOT/ablation_studies/fera_ms_global_ace_ablation_20260730/run_one_seed.sh" "$seed"; done
exec "$ROOT/ablation_studies/fera_ms_global_ace_ablation_20260730/scripts/run_formal_aggregate.sh"
