#!/usr/bin/env bash
set -euo pipefail

BASELINE_ROOT="${FERA_MS_BASELINE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
SOURCE="${FERA_MS_BASELINE_SOURCE:-$BASELINE_ROOT/source/fragnnet}"

for script in \
  "$BASELINE_ROOT/neims/run_all.sh" \
  "$BASELINE_ROOT/massformer/run_all.sh" \
  "$BASELINE_ROOT/fragnnet_depth_three/run_all.sh" \
  "$BASELINE_ROOT/graff_ms/run_all.sh" \
  "$BASELINE_ROOT/iceberg/run_all.sh" \
  "$BASELINE_ROOT/fiora/run_final.sh"; do
  bash -n "$script"
done

test -f "$SOURCE/src/fragnnet/model.py" || {
  echo "Bundled independent fragnnet baseline source is missing." >&2
  exit 2
}
test -f "$SOURCE/src/fragnnet/iceberg/model.py" || {
  echo "Bundled ICEBERG implementation is missing from the baseline package." >&2
  exit 2
}

python "$BASELINE_ROOT/tools_local/check_local_runtime.py"
echo "BASELINE_PREFLIGHT_OK"
