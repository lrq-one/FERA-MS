#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE="${FERA_MS_BASELINE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PROJECT="${FERA_MS_ROOT:-$(cd "$BASELINE/../.." && pwd)}"
REPO="${FERA_MS_BASELINE_SOURCE:-$BASELINE/source/fragnnet}"

RUNS_ROOT="${FERA_MS_RUNS_DIR:-$PROJECT/runs}"
ROOT="$RUNS_ROOT/experiments/molecular_retrieval/pubchem_legacy_full"
RETRIEVAL_ROOT="$ROOT/baseline_molecular_retrieval"
FROZEN="$RETRIEVAL_ROOT/_frozen_inputs"

TOOLS="$BASELINE/tools_local"
RUNNER="$TOOLS/run_baseline_retrieval_formal.py"
PLAN="$FROZEN/molecular_retrieval_run_plan.csv"

PY="${PYTHON:-python}"

mkdir -p "$RETRIEVAL_ROOT"

exec 9>"$RETRIEVAL_ROOT/.three_baseline_supervisor.lock"

if ! flock -n 9; then
    echo "ANOTHER_THREE_BASELINE_SUPERVISOR_IS_RUNNING"
    exit 31
fi

export PYTHONPATH="$REPO/src:$REPO"
export DGLBACKEND=pytorch
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
unset TORCH_FORCE_WEIGHTS_ONLY_LOAD || true

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled

echo "======================================================================"
echo "THREE-BASELINE MOLECULAR-RETRIEVAL SUPERVISOR"
echo "start=$(date -Is)"
echo "plan=$PLAN"
echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
echo "concurrent_ablation_allowed=true"
echo "======================================================================"

update_status() {
    local model="$1"
    local split="$2"
    local seed="$3"
    local status="$4"

    "$PY" - \
        "$PLAN" \
        "$model" \
        "$split" \
        "$seed" \
        "$status" <<'PY'
import os
import sys
from pathlib import Path

import pandas as pd


plan_path = Path(sys.argv[1])
model = sys.argv[2]
split = sys.argv[3]
seed = int(sys.argv[4])
status = sys.argv[5]

plan = pd.read_csv(
    plan_path,
    low_memory=False,
)

mask = (
    plan["model"].astype(str)
    .eq(model)
    & plan["split"].astype(str)
    .eq(split)
    & plan["seed"].astype(int)
    .eq(seed)
)

if int(mask.sum()) != 1:
    raise RuntimeError(
        f"Cannot update unique plan row: "
        f"{model}/{split}/{seed}"
    )

plan.loc[
    mask,
    "status",
] = status

temporary = plan_path.with_suffix(
    plan_path.suffix + ".tmp"
)

plan.to_csv(
    temporary,
    index=False,
)

os.replace(
    temporary,
    plan_path,
)
PY
}

while IFS=$'\t' read -r \
    model split seed checkpoint config output_dir
do
    success="$output_dir/_SUCCESS.json"

    echo
    echo "======================================================================"
    echo "RUN $model $split seed=$seed"
    echo "output=$output_dir"
    echo "======================================================================"

    mkdir -p "$output_dir"

    if [ -f "$success" ]; then
        echo "SKIP_SUCCESS $success"
        update_status \
            "$model" \
            "$split" \
            "$seed" \
            "SUCCESS"

        "$PY" -u "$RUNNER" \
            --aggregate-only

        continue
    fi

    case "$model" in
        neims)
            eval_batch=64
            chunk_size=16384
            ;;
        massformer)
            eval_batch=32
            chunk_size=8192
            ;;
        fragnnet_d3)
            eval_batch=32
            chunk_size=2048
            ;;
        *)
            echo "UNKNOWN_MODEL=$model"
            exit 32
            ;;
    esac

    update_status \
        "$model" \
        "$split" \
        "$seed" \
        "RUNNING"

    run_ok=0

    for attempt in 1 2 3 4 5
    do
        attempt_log="$output_dir/attempt_${attempt}.log"
        run_log="$output_dir/run.log"

        echo | tee -a "$run_log"
        echo "------------------------------------------------------------" \
            | tee -a "$run_log"
        echo "attempt=$attempt" \
            | tee -a "$run_log"
        echo "start=$(date -Is)" \
            | tee -a "$run_log"
        echo "eval_batch=$eval_batch" \
            | tee -a "$run_log"
        echo "chunk_size=$chunk_size" \
            | tee -a "$run_log"
        echo "------------------------------------------------------------" \
            | tee -a "$run_log"

        set +e

        nice -n 10 \
        "$PY" -u "$RUNNER" \
            --model "$model" \
            --split "$split" \
            --seed "$seed" \
            --checkpoint "$checkpoint" \
            --config "$config" \
            --output-dir "$output_dir" \
            --chunk-size "$chunk_size" \
            --eval-batch-size "$eval_batch" \
            --num-workers 0 \
            --device cuda:0 \
            2>&1 \
        | tee -a \
            "$run_log" \
            "$attempt_log"

        rc=${PIPESTATUS[0]}

        set -e

        echo "attempt_rc=$rc" \
            | tee -a "$run_log"

        if [ "$rc" -eq 0 ] \
           && [ -f "$success" ]
        then
            run_ok=1
            break
        fi

        if grep -Eqi \
            'CUDA out of memory|out of memory|CUBLAS_STATUS_ALLOC_FAILED' \
            "$attempt_log"
        then
            if [ "$eval_batch" -le 1 ]; then
                echo "OOM_AT_MINIMUM_BATCH"
                break
            fi

            eval_batch=$((eval_batch / 2))

            if [ "$eval_batch" -lt 1 ]; then
                eval_batch=1
            fi

            echo "OOM_RETRY_WITH_EVAL_BATCH=$eval_batch"

            sleep 20
            continue
        fi

        echo "NON_OOM_FAILURE_NO_AUTOMATIC_RETRY"
        break
    done

    if [ "$run_ok" -ne 1 ]; then
        update_status \
            "$model" \
            "$split" \
            "$seed" \
            "FAILED"

        echo "FORMAL_RUN_FAILED $model $split seed=$seed"
        exit 40
    fi

    update_status \
        "$model" \
        "$split" \
        "$seed" \
        "SUCCESS"

    "$PY" -u "$RUNNER" \
        --aggregate-only

done < <(
    "$PY" - "$PLAN" <<'PY'
import sys
from pathlib import Path

import pandas as pd


plan_path = Path(sys.argv[1])

plan = pd.read_csv(
    plan_path,
    low_memory=False,
)

expected_models = {
    "neims",
    "massformer",
    "fragnnet_d3",
}

plan = plan[
    plan[
        "model"
    ].astype(str).isin(
        expected_models
    )
].copy()

if len(plan) != 18:
    raise RuntimeError(
        f"Expected 18 plan rows, "
        f"found {len(plan)}"
    )

for row in plan.itertuples(
    index=False
):
    print(
        "\t".join(
            [
                str(row.model),
                str(row.split),
                str(int(row.seed)),
                str(row.checkpoint),
                str(row.config),
                str(row.output_dir),
            ]
        )
    )
PY
)

echo
echo "======================================================================"
echo "FINAL AGGREGATION"
echo "======================================================================"

"$PY" -u "$RUNNER" \
    --aggregate-only \
    --require-complete

echo
echo "======================================================================"
echo "THREE-BASELINE MOLECULAR-RETRIEVAL COMPLETE"
echo "completed=$(date -Is)"
echo "======================================================================"
