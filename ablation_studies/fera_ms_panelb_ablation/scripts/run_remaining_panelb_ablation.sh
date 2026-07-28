#!/usr/bin/env bash

ROOT="/home/lwh/projects/lrq2/fragnnet-main/ms2spectra_v1_r119"
ABLATION_ROOT="$ROOT/ablation_studies/fera_ms_panelb_ablation"

cd "$ROOT" || exit 1

source /home/lwh/anaconda3/etc/profile.d/conda.sh
conda activate lrq_q || exit 1

export PYTHONPATH="$ROOT/code/src:$ROOT/code:$ROOT"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

SEED42_SUMMARY="$ABLATION_ROOT/runs/without_mz_offset_expansion/seed_42/evaluation/test_summary.json"

python - "$SEED42_SUMMARY" <<'PY'
import json
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])

if not path.is_file():
    raise FileNotFoundError(path)

data = json.loads(
    path.read_text(encoding="utf-8")
)

assert data["test_spectrum_count"] == 3931
assert math.isfinite(float(data["micro_cbin"]))
assert math.isfinite(float(data["micro_jss"]))

print("SEED42_LOCKED_TEST_GATE_PASS")
PY

if [ "$?" -ne 0 ]; then
    echo "Seed42 test gate failed."
    exit 1
fi

run_case () {
    VARIANT="$1"
    SEED="$2"

    TRAIN_LOG="$ABLATION_ROOT/logs/${VARIANT}_seed${SEED}.log"
    TEST_LOG="$ABLATION_ROOT/logs/${VARIANT}_seed${SEED}_test.log"

    echo
    echo "================================================================================"
    echo "START TRAIN: $VARIANT seed=$SEED"
    echo "================================================================================"

    python \
      "$ABLATION_ROOT/src/run_one_panelb_ablation.py" \
      --variant "$VARIANT" \
      --seed "$SEED" \
      > "$TRAIN_LOG" 2>&1

    CODE="$?"

    if [ "$CODE" -ne 0 ]; then
        echo "TRAIN FAILED: $VARIANT seed=$SEED code=$CODE"
        exit "$CODE"
    fi

    echo
    echo "================================================================================"
    echo "START TEST: $VARIANT seed=$SEED"
    echo "================================================================================"

    python \
      "$ABLATION_ROOT/src/evaluate_one_panelb_ablation.py" \
      --variant "$VARIANT" \
      --seed "$SEED" \
      > "$TEST_LOG" 2>&1

    CODE="$?"

    if [ "$CODE" -ne 0 ]; then
        echo "TEST FAILED: $VARIANT seed=$SEED code=$CODE"
        exit "$CODE"
    fi

    cat \
      "$ABLATION_ROOT/runs/$VARIANT/seed_$SEED/evaluation/test_summary.json"
}

run_case without_mz_offset_expansion 43
run_case without_mz_offset_expansion 44

run_case without_rendered_peak_gate 42
run_case without_rendered_peak_gate 43
run_case without_rendered_peak_gate 44

python - <<'PY'
from pathlib import Path
import json
import pandas as pd

root = Path(
    "ablation_studies/"
    "fera_ms_panelb_ablation"
)

rows = []

for path in sorted(
    root.glob(
        "runs/*/seed_*/evaluation/"
        "test_summary.json"
    )
):
    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    rows.append(
        {
            "variant":
                data["variant"],

            "seed":
                data["seed"],

            "test_spectrum_count":
                data["test_spectrum_count"],

            "micro_cbin":
                data["micro_cbin"],

            "micro_jss":
                data["micro_jss"],

            "delta_cbin_vs_full":
                data.get(
                    "delta_cbin_vs_full"
                ),

            "delta_jss_vs_full":
                data.get(
                    "delta_jss_vs_full"
                ),
        }
    )

table = pd.DataFrame(rows)

results_dir = root / "results"
results_dir.mkdir(
    parents=True,
    exist_ok=True,
)

table.to_csv(
    results_dir
    / "panelb_seed_results.csv",
    index=False,
)

summary = (
    table
    .groupby("variant", as_index=False)
    .agg(
        n_seeds=("seed", "count"),
        micro_cbin_mean=("micro_cbin", "mean"),
        micro_cbin_std=("micro_cbin", "std"),
        micro_jss_mean=("micro_jss", "mean"),
        micro_jss_std=("micro_jss", "std"),
    )
)

summary.to_csv(
    results_dir
    / "panelb_3seed_summary.csv",
    index=False,
)

print()
print("SEED RESULTS")
print(table.to_string(index=False))

print()
print("THREE-SEED SUMMARY")
print(summary.to_string(index=False))
PY

echo
echo "PANEL-B ALL REMAINING RUNS COMPLETE"
