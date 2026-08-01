#!/usr/bin/env bash
set -euo pipefail

BASELINE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="${FERA_MS_FIORA_WORKSPACE:-$BASELINE_ROOT/shared/fiora_work}"
FIORA_PREDICT="${FIORA_PREDICT:-fiora-predict}"

cd "$WORK/fiora-main"

mkdir -p "$WORK/fiora_audit/preds"
mkdir -p "$WORK/fiora_audit/summaries"

PRED="$WORK/fiora_audit/preds/fiora_os_qtof_final.mgf"
OUT="$WORK/fiora_audit/summaries/fiora_os_qtof_final_test.csv"

echo "============================================================"
echo "Running FIORA official zero-shot QTOF baseline"
echo "WORK: $WORK"
echo "PRED: $PRED"
echo "OUT:  $OUT"
echo "============================================================"

"$FIORA_PREDICT" \
  -i "$WORK/safe19659/fiora_safe19659_test_qtof_ace_FROM_FULL.csv" \
  -o "$PRED" \
  --dev cuda:0 \
  --min_prob 0 \
  --no-rt \
  --no-ccs \
  --no-annotation

python \
  "$WORK/fiora_audit/scripts/eval_fiora_against_library_csv.py" \
  --pred_mgf "$PRED" \
  --ref_csv "$WORK/safe19659/fiora_safe19659_library_qtof_ace.csv" \
  --split test \
  --top_k 0 \
  --rel_thresh 0.0 \
  --out_csv "$OUT"

echo "FIORA final evaluation saved to: $OUT"
