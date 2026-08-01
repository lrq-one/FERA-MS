#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SOURCE="$BASELINE_ROOT/source/fiora"
INPUT="${FERA_MS_FIORA_INPUT:?Set FERA_MS_FIORA_INPUT to the local test-query CSV}"
REFERENCE="${FERA_MS_FIORA_REFERENCE:?Set FERA_MS_FIORA_REFERENCE to the local reference CSV}"
MODEL="${FERA_MS_FIORA_MODEL:?Set FERA_MS_FIORA_MODEL to the external FIORA .pt model}"
OUTPUT_DIR="${FERA_MS_BASELINE_OUTPUT_DIR:-$BASELINE_ROOT/results_local}/fiora"

mkdir -p "$OUTPUT_DIR"

PRED="$OUTPUT_DIR/fiora_os_qtof_final.mgf"
OUT="$OUTPUT_DIR/fiora_os_qtof_final_test.csv"

echo "Running FIORA official zero-shot QTOF baseline"
echo "SOURCE: $SOURCE"
echo "MODEL:  $MODEL"
echo "PRED:   $PRED"
echo "OUT:    $OUT"

PYTHONPATH="$SOURCE${PYTHONPATH:+:$PYTHONPATH}" python -m fiora.cli.predict \
  -i "$INPUT" \
  -o "$PRED" \
  --model "$MODEL" \
  --dev "${FIORA_DEVICE:-cpu}" \
  --min_prob 0 \
  --no-rt \
  --no-ccs \
  --no-annotation

python "$SCRIPT_DIR/eval_fiora_against_library_csv.py" \
  --pred_mgf "$PRED" \
  --ref_csv "$REFERENCE" \
  --split test \
  --top_k 0 \
  --rel_thresh 0.0 \
  --out_csv "$OUT"

echo "FIORA final evaluation saved to: $OUT"
