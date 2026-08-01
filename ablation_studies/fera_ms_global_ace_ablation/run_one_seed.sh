#!/usr/bin/env bash
set -euo pipefail
seed="${1:?seed required}"
case "$seed" in 42|43|44) ;; *) exit 2;; esac
ROOT="${FERA_MS_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
exec "$ROOT/ablation_studies/fera_ms_global_ace_ablation/scripts/run_formal_seed.sh" "$seed"
