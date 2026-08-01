#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${FERA_MS_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
RUNS_ROOT="${FERA_MS_RUNS_DIR:-$ROOT/runs}"
ABLATION_ROOT="$ROOT/ablation_studies/fera_ms_core_ablation"
RUN_ROOT="$ABLATION_ROOT/runs/no_spectrum_allocator"
LOG_ROOT="$ABLATION_ROOT/logs/no_spectrum_allocator"
TEMPLATE="$RUNS_ROOT/_config/template.yml"
RERANKER_SCRIPT="$ROOT/train/_impl/refinement_steps/candidate_reranker.py"
HPARAM_ENV="$RUN_ROOT/mainline_reranker_hparams.env"

cd "$ROOT" || exit 1
export PYTHONPATH="$ROOT/code/src:$ROOT/code:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

mkdir -p "$RUN_ROOT" "$LOG_ROOT"

python - "$ROOT/config/train.yml" > "$HPARAM_ENV" <<'PY'
from pathlib import Path
import shlex
import sys
import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
params = config.get("postprocessing_env")
if not isinstance(params, dict):
    raise RuntimeError("config/train.yml is missing postprocessing_env")
for key, value in params.items():
    print(f"{key}={shlex.quote(str(value))}")
PY

if [ "$?" -ne 0 ]; then
    echo "MAINLINE_HYPERPARAMETER_READ_FAILED"
    exit 1
fi

# shellcheck disable=SC1090
source "$HPARAM_ENV"

echo "============================================================"
echo "NO_SPECTRUM_ALLOCATOR"
echo "============================================================"
echo "Candidate reranker: retained"
echo "Spectrum allocator: omitted"
echo "Selection: validation-only alpha selection"
echo "Test evaluation: once after selection"
echo "============================================================"

FAILED=0
SOURCE_ROOT="$RUNS_ROOT/experiments/molecule_disjoint_three_seeds"

for SEED in 42 43 44
do
    SOURCE="$SOURCE_ROOT/seed_${SEED}"
    FULL_MODEL="$SOURCE/full_fera_ms"
    CONFIG="$SOURCE/global_ace_control_ce_trajectory_ablation/control/config.yml"
    CHECKPOINT="$FULL_MODEL/final_peak_distillation/final_peak_distillation_best_state.pt"
    REGRESSOR="$FULL_MODEL/candidate_reranking/candidate_reranker_regressor.pkl"
    OUT="$RUN_ROOT/random/seed_${SEED}"
    LOG="$LOG_ROOT/random_seed_${SEED}.log"

    mkdir -p "$OUT"

    MISSING=0
    for path in "$TEMPLATE" "$CONFIG" "$CHECKPOINT" "$REGRESSOR" "$RERANKER_SCRIPT"
    do
        if [ ! -s "$path" ]; then
            echo "MISSING_REQUIRED_FILE=$path"
            MISSING=1
        fi
    done
    if [ "$MISSING" -ne 0 ]; then
        FAILED=1
        continue
    fi

    python -u "$RERANKER_SCRIPT" \
        --template "$TEMPLATE" \
        --config "$CONFIG" \
        --ckpt_path "$CHECKPOINT" \
        --out_dir "$OUT" \
        --seed "$SEED" \
        --backend lightgbm \
        --load_regressor "$REGRESSOR" \
        --max_train_rows "$CANDIDATE_RERANKER_MAX_TRAIN_ROWS" \
        --neg_topk_per_batch "$CANDIDATE_RERANKER_NEG_TOPK" \
        --neg_rand_per_batch "$CANDIDATE_RERANKER_NEG_RANDOM" \
        --mz_tol 0.01 \
        --mz_sigma 0.003 \
        --target_bin_res 0.01 \
        --local_bin_res 0.01 \
        --eval_bin_res 0.01 \
        --residual_clip "$CANDIDATE_RERANKER_RESIDUAL_CLIP" \
        --neg_residual "$CANDIDATE_RERANKER_NEG_RESIDUAL" \
        --score_clip "$CANDIDATE_RERANKER_SCORE_CLIP" \
        --low_w "$CANDIDATE_RERANKER_LOW_W" \
        --mid_w "$CANDIDATE_RERANKER_MID_W" \
        --high_w "$CANDIDATE_RERANKER_HIGH_W" \
        --pos_weight "$CANDIDATE_RERANKER_POS_W" \
        --pos_intensity_weight "$CANDIDATE_RERANKER_POS_INTENSITY_W" \
        --neg_weight "$CANDIDATE_RERANKER_NEG_W" \
        --neg_prob_weight "$CANDIDATE_RERANKER_NEG_PROB_W" \
        --n_estimators "$CANDIDATE_RERANKER_N_ESTIMATORS" \
        --gbdt_lr "$CANDIDATE_RERANKER_GBDT_LR" \
        --num_leaves "$CANDIDATE_RERANKER_NUM_LEAVES" \
        --max_depth "$CANDIDATE_RERANKER_MAX_DEPTH" \
        --min_child_samples "$CANDIDATE_RERANKER_MIN_CHILD" \
        --subsample "$CANDIDATE_RERANKER_SUBSAMPLE" \
        --colsample_bytree "$CANDIDATE_RERANKER_COLSAMPLE" \
        --reg_alpha "$CANDIDATE_RERANKER_REG_ALPHA" \
        --reg_lambda "$CANDIDATE_RERANKER_REG_LAMBDA" \
        --num_workers "$CANDIDATE_RERANKER_WORKERS" \
        --max_extra_dims "$CANDIDATE_RERANKER_EXTRA_DIMS" \
        --alpha_grid "$CANDIDATE_RERANKER_ALPHA_GRID" \
        --eval_test \
        2>&1 | tee "$LOG"

    CODE=${PIPESTATUS[0]}
    if [ "$CODE" -ne 0 ] || [ ! -s "$OUT/candidate_reranker_best_test.csv" ]; then
        echo "NO_SPECTRUM_ALLOCATOR_FAILED seed=$SEED code=$CODE"
        FAILED=1
    fi
done

if [ "$FAILED" -ne 0 ]; then
    exit 1
fi

echo "NO_SPECTRUM_ALLOCATOR_COMPLETE"
