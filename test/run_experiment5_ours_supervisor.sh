#!/usr/bin/env bash
set -uo pipefail

ROOT="/home/lwh/projects/lrq2/fragnnet-main/ms2spectra_v1_r119"

# EXPERIMENT5_FORCE_LRQ_ENV_V1
source "/home/lwh/anaconda3/etc/profile.d/conda.sh"
conda activate lrq_q
PYTHON_BIN="/home/lwh/anaconda3/envs/lrq_q/bin/python"
export PATH="/home/lwh/anaconda3/envs/lrq_q/bin:$PATH"
OUT="$ROOT/runs/experiments/molecular_retrieval/pubchem_legacy_full/ours_r184b_experiment5_20260724"
LOG="$OUT/run.log"
STATUS="$OUT/supervisor_status.json"
BATCH_FILE="$OUT/current_batch_size.txt"

cd "$ROOT" || exit 1
mkdir -p "$OUT"

exec 9>"$OUT/supervisor.lock"

if ! flock -n 9; then
    echo "[supervisor] another supervisor already holds the lock"
    exit 0
fi

echo "$$" > "$OUT/supervisor.pid"

if [ ! -s "$BATCH_FILE" ]; then
    echo "32" > "$BATCH_FILE"
fi

attempt=0
max_attempts=50

write_status() {
    local state="$1"
    local code="${2:-0}"
    local batch_size="${3:-0}"

    python - \
        "$STATUS" \
        "$state" \
        "$attempt" \
        "$code" \
        "$batch_size" <<'PY'
from pathlib import Path
import json
import os
import sys
import time

path = Path(sys.argv[1])

payload = {
    "status": sys.argv[2],
    "attempt": int(sys.argv[3]),
    "last_exit_code": int(sys.argv[4]),
    "batch_size": int(sys.argv[5]),
    "updated_at_epoch": time.time(),
}

temporary = path.with_name(path.name + ".tmp")
temporary.write_text(
    json.dumps(
        payload,
        indent=2,
        ensure_ascii=False,
    )
    + "\n",
    encoding="utf-8",
)
os.replace(temporary, path)
PY
}

while true; do
    if "$PYTHON_BIN" - "$OUT/run_manifest.json" <<'PY'
from pathlib import Path
import json
import sys

path = Path(sys.argv[1])

if not path.is_file():
    raise SystemExit(1)

try:
    state = json.loads(
        path.read_text(encoding="utf-8")
    )
except Exception:
    raise SystemExit(1)

raise SystemExit(
    0 if state.get("status") == "complete" else 1
)
PY
    then
        current_batch=$(cat "$BATCH_FILE")
        write_status "complete" 0 "$current_batch"
        echo "[supervisor] all combinations complete"
        exit 0
    fi

    attempt=$((attempt + 1))

    if [ "$attempt" -gt "$max_attempts" ]; then
        current_batch=$(cat "$BATCH_FILE")
        write_status "gave_up" 99 "$current_batch"
        echo "[supervisor] exceeded max attempts=$max_attempts"
        exit 99
    fi

    batch_size=$(cat "$BATCH_FILE")

    case "$batch_size" in
        32|24|20|16) ;;
        *)
            batch_size=32
            echo "$batch_size" > "$BATCH_FILE"
            ;;
    esac

    start_line=$(wc -l < "$LOG" 2>/dev/null || echo 0)

    {
        echo
        echo "================================================================"
        echo "[supervisor] attempt=$attempt"
        echo "[supervisor] start=$(date '+%F %T %Z')"
        echo "[supervisor] batch_size=$batch_size"
        echo "================================================================"
    } >> "$LOG"

    write_status "running" 0 "$batch_size"

    env \
        PYTHONPATH="$ROOT/code/src" \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        "$PYTHON_BIN" -u test/run_experiment5_ours.py \
            --splits random scaffold \
            --seeds 42 43 44 \
            --batch-size "$batch_size" \
            --num-workers 4 \
            >> "$LOG" 2>&1

    exit_code=$?

    if [ "$exit_code" -eq 0 ]; then
        write_status "complete" 0 "$batch_size"
        echo "[supervisor] complete"
        exit 0
    fi

    attempt_log="$OUT/attempt_${attempt}_tail.log"

    tail -n +"$((start_line + 1))" "$LOG" \
        > "$attempt_log" 2>/dev/null || true

    next_batch="$batch_size"

    if grep -Eqi \
        'CUDA out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED' \
        "$attempt_log"; then

        case "$batch_size" in
            32) next_batch=24 ;;
            24) next_batch=20 ;;
            20) next_batch=16 ;;
            16) next_batch=16 ;;
        esac

        if [ "$next_batch" -lt "$batch_size" ]; then
            echo "$next_batch" > "$BATCH_FILE"

            {
                echo "[supervisor] CUDA OOM detected"
                echo "[supervisor] batch_size $batch_size -> $next_batch"
                echo "[supervisor] resume from latest completed row"
            } >> "$LOG"
        fi
    fi

    write_status "restarting" "$exit_code" "$next_batch"

    {
        echo "[supervisor] process exited code=$exit_code"
        echo "[supervisor] restart after 30 seconds"
    } >> "$LOG"

    sleep 30
done
