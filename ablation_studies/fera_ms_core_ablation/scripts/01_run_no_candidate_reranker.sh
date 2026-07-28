#!/usr/bin/env bash

set -u

ROOT="/home/lwh/projects/lrq2/fragnnet-main/ms2spectra_v1_r119"
ABLATION_ROOT="$ROOT/ablation_studies/fera_ms_core_ablation"
RUN_ROOT="$ABLATION_ROOT/runs/no_candidate_reranker"
LOG_ROOT="$ABLATION_ROOT/logs/no_candidate_reranker"
RESULT_ROOT="$ABLATION_ROOT/results"

TEMPLATE="$ROOT/runs/_config/template.yml"
ALLOCATOR_SCRIPT="$ROOT/train/_impl/refinement_steps/spectrum_allocator.py"
HPARAM_ENV="$ABLATION_ROOT/config/mainline_allocator_hparams.env"

cd "$ROOT" || exit 1

export PYTHONPATH="$ROOT/code/src:$ROOT/code:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

mkdir -p \
  "$RUN_ROOT" \
  "$LOG_ROOT" \
  "$RESULT_ROOT"

# ----------------------------------------------------------------------
# Read the exact locked allocator hyperparameters used by the main line.
# ----------------------------------------------------------------------

python - "$ROOT/config/train.yml" > "$HPARAM_ENV" <<'PY'
from pathlib import Path
import shlex
import sys
import yaml

path = Path(sys.argv[1])

config = yaml.safe_load(
    path.read_text(encoding="utf-8")
)

params = config.get("postprocessing_env")

if not isinstance(params, dict):
    raise RuntimeError(
        "config/train.yml 缺少 postprocessing_env"
    )

for key, value in params.items():
    print(
        f"{key}={shlex.quote(str(value))}"
    )
PY

HPARAM_CODE=$?

if [ "$HPARAM_CODE" -ne 0 ]; then
    echo "MAINLINE_HPARAMETER_READ_FAILED"
    exit "$HPARAM_CODE"
fi

# shellcheck disable=SC1090
source "$HPARAM_ENV"

required_vars=(
    B_EPOCHS
    B_LR
    B_WEIGHT_DECAY
    B_HIDDEN
    B_LAYERS
    B_DROPOUT
    B_SCORE_CLIP
    B_LGBM_SCORE_CLIP
    B_RESIDUAL_SCALE
    B_TEMPERATURE
    B_COS_WEIGHT
    B_JSS_WEIGHT
    B_TARGET_CE_WEIGHT
    B_POS_RECALL_WEIGHT
    B_BASE_KL_WEIGHT
    B_RESIDUAL_L2_WEIGHT
    B_LOW_W
    B_MID_W
    B_HIGH_W
    R172_EXTRA_DIMS
)

for name in "${required_vars[@]}"
do
    if [ -z "${!name:-}" ]; then
        echo "MISSING_HPARAMETER=$name"
        exit 1
    fi
done

echo "============================================================"
echo "A5: WITHOUT CHEMICAL CANDIDATE RERANKING"
echo "============================================================"
echo "LightGBM alpha : 0.0"
echo "Allocator      : retrained from random initialization"
echo "Test selection : false"
echo "Test evaluation: after validation-only checkpoint selection"
echo "Output         : $RUN_ROOT"
echo "============================================================"

FAILED=0

for SPLIT in random
do
    if [ "$SPLIT" = "random" ]; then
        SOURCE_ROOT="$ROOT/runs/experiments/molecule_disjoint_3seeds"
        EXPECTED_TEST_COUNT=3931
    else
        SOURCE_ROOT="$ROOT/runs/experiments/scaffold_disjoint_3seeds"
        EXPECTED_TEST_COUNT=3960
    fi

    for SEED in 42 43 44
    do
        SOURCE="$SOURCE_ROOT/seed_${SEED}/v2e_full_063"

        CONFIG="$SOURCE_ROOT/seed_${SEED}/v2c_ce_trajectory_ablation/control/config.yml"
        R160="$SOURCE/08_R160/r160_best_state.pt"
        REGRESSOR="$SOURCE/09_R172D/r170_regressor.pkl"

        OUT="$RUN_ROOT/$SPLIT/seed_${SEED}"
        LOG="$LOG_ROOT/${SPLIT}_seed_${SEED}.log"

        mkdir -p "$OUT"

        echo
        echo "============================================================"
        echo "START split=$SPLIT seed=$SEED"
        echo "============================================================"
        echo "config     : $CONFIG"
        echo "R160       : $R160"
        echo "feature pack: $REGRESSOR"
        echo "output     : $OUT"
        echo "============================================================"

        if [ ! -s "$CONFIG" ]; then
            echo "MISSING_CONFIG=$CONFIG"
            FAILED=1
            continue
        fi

        if [ ! -s "$R160" ]; then
            echo "MISSING_R160=$R160"
            FAILED=1
            continue
        fi

        if [ ! -s "$REGRESSOR" ]; then
            echo "MISSING_FEATURE_PACKAGE=$REGRESSOR"
            FAILED=1
            continue
        fi

        if [ -s "$OUT/r184_best_test.csv" ] \
           && [ -s "$OUT/r184_allocator_best.pt" ]; then
            echo "[RESUME] split=$SPLIT seed=$SEED already complete"
            continue
        fi

        python -u "$ALLOCATOR_SCRIPT" \
          --template "$TEMPLATE" \
          --config "$CONFIG" \
          --ckpt_path "$R160" \
          --regressor_path "$REGRESSOR" \
          --out_dir "$OUT" \
          --seed "$SEED" \
          --epochs "$B_EPOCHS" \
          --lr "$B_LR" \
          --weight_decay "$B_WEIGHT_DECAY" \
          --grad_clip 5.0 \
          --hidden "$B_HIDDEN" \
          --layers "$B_LAYERS" \
          --dropout "$B_DROPOUT" \
          --score_clip "$B_SCORE_CLIP" \
          --alpha 0.0 \
          --lgbm_score_clip "$B_LGBM_SCORE_CLIP" \
          --residual_scale "$B_RESIDUAL_SCALE" \
          --temperature "$B_TEMPERATURE" \
          --cos_weight "$B_COS_WEIGHT" \
          --jss_weight "$B_JSS_WEIGHT" \
          --target_ce_weight "$B_TARGET_CE_WEIGHT" \
          --pos_recall_weight "$B_POS_RECALL_WEIGHT" \
          --base_kl_weight "$B_BASE_KL_WEIGHT" \
          --residual_l2_weight "$B_RESIDUAL_L2_WEIGHT" \
          --low_w "$B_LOW_W" \
          --mid_w "$B_MID_W" \
          --high_w "$B_HIGH_W" \
          --mz_tol 0.01 \
          --mz_sigma 0.003 \
          --target_bin_res 0.01 \
          --local_bin_res 0.01 \
          --eval_bin_res 0.01 \
          --residual_clip 6.0 \
          --neg_residual 4.0 \
          --max_extra_dims "$R172_EXTRA_DIMS" \
          --max_train_batches 0 \
          --eval_test \
          2>&1 | tee "$LOG"

        CODE=${PIPESTATUS[0]}

        echo
        echo "RUN_EXIT_CODE=$CODE"

        if [ "$CODE" -ne 0 ]; then
            FAILED=1
            echo "RUN_FAILED split=$SPLIT seed=$SEED"
            continue
        fi

        python - \
          "$OUT/r184_best_test.csv" \
          "$EXPECTED_TEST_COUNT" \
          "$OUT/run_validation.json" <<'PY'
from pathlib import Path
import json
import sys

import pandas as pd

table_path = Path(sys.argv[1])
expected = int(sys.argv[2])
output_path = Path(sys.argv[3])

table = pd.read_csv(table_path)

global_rows = table[
    table["ce_bucket"].astype(str) == "global"
]

if len(global_rows) != 1:
    raise RuntimeError(
        f"expected one global row, found {len(global_rows)}"
    )

row = global_rows.iloc[0]

count = int(row["spec_count"])
cosine = float(row["cos"])
jss = float(row["jss"])

if count != expected:
    raise RuntimeError(
        f"test spectrum count mismatch: {count} != {expected}"
    )

if not (0.0 <= cosine <= 1.0):
    raise RuntimeError(
        f"invalid cosine: {cosine}"
    )

if not (0.0 <= jss <= 1.0):
    raise RuntimeError(
        f"invalid JSS: {jss}"
    )

output_path.write_text(
    json.dumps(
        {
            "test_spectrum_count": count,
            "micro_cbin": cosine,
            "micro_jss": jss,
            "candidate_reranker_alpha": 0.0,
            "test_used_for_selection": False,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)

print(
    f"VALIDATED count={count} "
    f"CBIN={cosine:.9f} "
    f"JSS={jss:.9f}"
)
PY

        VALIDATE_CODE=$?

        if [ "$VALIDATE_CODE" -ne 0 ]; then
            FAILED=1
            echo "VALIDATION_FAILED split=$SPLIT seed=$SEED"
        fi
    done
done

# ----------------------------------------------------------------------
# Aggregate every completed run. The summary is only declared complete
# when all six split/seed combinations are present.
# ----------------------------------------------------------------------

python - "$RUN_ROOT" "$RESULT_ROOT" <<'PY'
from pathlib import Path
import json
import sys

import pandas as pd

run_root = Path(sys.argv[1])
result_root = Path(sys.argv[2])

rows = []

for split in ("random",):
    for seed in (42, 43, 44):
        path = (
            run_root
            / split
            / f"seed_{seed}"
            / "run_validation.json"
        )

        if not path.is_file():
            continue

        data = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        rows.append(
            {
                "split": split,
                "seed": seed,
                "model_variant":
                    "Without chemical candidate reranking",
                "micro_cbin":
                    float(data["micro_cbin"]),
                "micro_jss":
                    float(data["micro_jss"]),
                "test_spectrum_count":
                    int(data["test_spectrum_count"]),
                "candidate_reranker_alpha":
                    float(data["candidate_reranker_alpha"]),
            }
        )

raw = pd.DataFrame(rows)

raw_path = (
    result_root
    / "no_candidate_reranker_raw.csv"
)

raw.to_csv(
    raw_path,
    index=False,
)

if raw.empty:
    print("NO_COMPLETED_RUNS")
    raise SystemExit(0)

summary = (
    raw.groupby(
        "split",
        as_index=False,
    )
    .agg(
        micro_cbin_mean=(
            "micro_cbin",
            "mean",
        ),
        micro_cbin_std=(
            "micro_cbin",
            lambda values:
                values.std(ddof=1),
        ),
        micro_jss_mean=(
            "micro_jss",
            "mean",
        ),
        micro_jss_std=(
            "micro_jss",
            lambda values:
                values.std(ddof=1),
        ),
        n_seeds=(
            "seed",
            "nunique",
        ),
    )
)

summary_path = (
    result_root
    / "no_candidate_reranker_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False,
)

formatted_rows = []

for split in ("random",):
    selected = summary[
        summary["split"] == split
    ]

    if len(selected) != 1:
        continue

    row = selected.iloc[0]

    formatted_rows.append(
        {
            "Split":
                split.capitalize(),
            "Model variant":
                "Without chemical candidate reranking",
            "Micro CBIN":
                (
                    f"{row['micro_cbin_mean']:.6f}"
                    f" ± "
                    f"{row['micro_cbin_std']:.6f}"
                ),
            "Micro JSS":
                (
                    f"{row['micro_jss_mean']:.6f}"
                    f" ± "
                    f"{row['micro_jss_std']:.6f}"
                ),
            "Seeds":
                int(row["n_seeds"]),
        }
    )

publication = pd.DataFrame(
    formatted_rows
)

publication_path = (
    result_root
    / "no_candidate_reranker_publication_table.csv"
)

publication.to_csv(
    publication_path,
    index=False,
)

print()
print("=" * 110)
print("A5 COMPLETED RUN SUMMARY")
print("=" * 110)
print(
    publication.to_string(
        index=False
    )
)
print("=" * 110)
print("raw        :", raw_path)
print("summary    :", summary_path)
print("publication:", publication_path)

complete = (
    len(raw) == 3
    and set(summary["n_seeds"].astype(int))
    == {3}
)

if complete:
    print("NO_CANDIDATE_RERANKER_ABLATION_COMPLETE")
else:
    print(
        "NO_CANDIDATE_RERANKER_ABLATION_PARTIAL "
        f"completed={len(raw)}/3"
    )
PY

AGGREGATE_CODE=$?

if [ "$AGGREGATE_CODE" -ne 0 ]; then
    FAILED=1
fi

echo
echo "============================================================"
echo "A5 FINAL STATUS"
echo "============================================================"
echo "FAILED=$FAILED"
echo "RUN_ROOT=$RUN_ROOT"
echo "LOG_ROOT=$LOG_ROOT"
echo "RESULT_ROOT=$RESULT_ROOT"
echo "============================================================"

exit "$FAILED"
