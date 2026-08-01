#!/usr/bin/env bash
set -euo pipefail
seed="${1:?seed required}"
ROOT="${FERA_MS_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
ABL="$ROOT/ablation_studies/fera_ms_global_ace_ablation_20260730"
if [[ "${FERA_DRY_RUN:-0}" != "1" ]]; then
  source "$ABL/scripts/require_gpu.sh"
fi
for forbidden in run_seed42_smoke; do
  if [[ "$ABL/runs/seed_$seed" == *"$forbidden"* ]]; then exit 99; fi
done
echo "Formal global-only seed $seed control-flow entrypoint."
echo "This script dispatches the locked V2A/V2C and R146-R184B global-only CE pipeline."
args=(--seed "$seed")
if [[ "${FERA_SMOKE_MODE:-0}" == "1" ]]; then args+=(--smoke-mode); fi
if [[ "${FERA_DRY_RUN:-0}" == "1" ]]; then args+=(--dry-run); fi
exec python -u "$ABL/pipeline_src/formal_pipeline.py" "${args[@]}"
