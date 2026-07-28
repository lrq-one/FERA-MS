#!/usr/bin/env bash

set -u

ROOT="/home/lwh/projects/lrq2/fragnnet-main/ms2spectra_v1_r119"
ABLATION_ROOT="$ROOT/ablation_studies/fera_ms_core_ablation"

VARIANT="${1:-}"

shift || true

if [ "$#" -gt 0 ]; then
    SEEDS=("$@")
else
    SEEDS=(42 43 44)
fi

case "$VARIANT" in
    fragment_node_mlp)
        FORMAL_NAME="Fragment-wise node encoder"
        ;;
    topology_only_dag)
        FORMAL_NAME="Topology-only fragment DAG"
        ;;
    global_molecular_context)
        FORMAL_NAME="Global molecular context"
        ;;
    global_ace_only)
        FORMAL_NAME="Global ACE conditioning only"
        ;;
    *)
        echo "Unknown variant: $VARIANT"
        echo
        echo "Allowed:"
        echo "  fragment_node_mlp"
        echo "  topology_only_dag"
        echo "  global_molecular_context"
        echo "  global_ace_only"
        exit 1
        ;;
esac

cd "$ROOT" || exit 1

export PYTHONPATH="$ROOT/code/src:$ROOT/code:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

RUN_ROOT="$ABLATION_ROOT/runs/$VARIANT"
LOG_ROOT="$ABLATION_ROOT/logs/$VARIANT"
RESULT_ROOT="$ABLATION_ROOT/results"

mkdir -p \
  "$RUN_ROOT" \
  "$LOG_ROOT" \
  "$RESULT_ROOT" \
  "$ABLATION_ROOT/recovery"

ACTIVE_BASE="$ROOT/runs/v2a_gine_cutchem_only"
ACTIVE_CONTROL="$ROOT/runs/v2c_ce_trajectory_ablation"
ACTIVE_REFINEMENT="$ROOT/runs/v2e_full_063"

clean_active_path() {
    PATH_VALUE="$1"

    if [ -L "$PATH_VALUE" ]; then
        rm "$PATH_VALUE"
        return
    fi

    if [ -e "$PATH_VALUE" ]; then
        RECOVERY="$ABLATION_ROOT/recovery/$(basename "$PATH_VALUE")_$(date +%Y%m%d_%H%M%S)"
        echo "[RECOVERY MOVE]"
        echo "$PATH_VALUE"
        echo "→ $RECOVERY"
        mv "$PATH_VALUE" "$RECOVERY"
    fi
}

link_active_path() {
    TARGET="$1"
    LINK="$2"

    if [ -L "$LINK" ]; then
        rm "$LINK"
    elif [ -e "$LINK" ]; then
        clean_active_path "$LINK"
    fi

    ln -s "$TARGET" "$LINK"
}

echo "================================================================================================"
echo "STRICT NEURAL ABLATION"
echo "================================================================================================"
echo "Variant : $VARIANT"
echo "Name    : $FORMAL_NAME"
echo "Split   : random molecule-disjoint only"
echo "Seeds   : ${SEEDS[*]}"
echo "Stop at : R160 neural backbone"
echo "Scaffold: disabled"
echo "================================================================================================"

for SEED in "${SEEDS[@]}"
do
    SEED_ROOT="$RUN_ROOT/seed_${SEED}"
    RESULT_JSON="$SEED_ROOT/evaluation/result.json"

    mkdir -p \
      "$SEED_ROOT/logs" \
      "$SEED_ROOT/evaluation"

    if [ -s "$RESULT_JSON" ]; then
        echo
        echo "[RESUME] $VARIANT seed=$SEED already complete"
        continue
    fi

    echo
    echo
    echo "################################################################################################"
    echo "START $VARIANT SEED $SEED"
    echo "################################################################################################"

    export MS2_GLOBAL_SEED="$SEED"
    export PYTHONHASHSEED="$SEED"
    export MS2_ABLATION_VARIANT="$VARIANT"

    cat > "$SEED_ROOT/effective_environment.env" <<EOF
MS2_GLOBAL_SEED=$SEED
PYTHONHASHSEED=$SEED
MS2_ABLATION_VARIANT=$VARIANT
SPLIT=random_molecule_disjoint
EOF

    clean_active_path "$ACTIVE_BASE"
    clean_active_path "$ACTIVE_CONTROL"
    clean_active_path "$ACTIVE_REFINEMENT"

    # ------------------------------------------------------------------
    # Base training
    # ------------------------------------------------------------------

    if [ ! -s "$SEED_ROOT/base_training/final/model.ckpt" ]; then
        python -u \
          train/train.py base \
          2>&1 | tee \
          "$SEED_ROOT/logs/01_base_training.log"

        CODE=${PIPESTATUS[0]}

        if [ "$CODE" -ne 0 ]; then
            echo "BASE_TRAINING_FAILED seed=$SEED code=$CODE"
            exit "$CODE"
        fi

        mv \
          "$ACTIVE_BASE" \
          "$SEED_ROOT/base_training"
    fi

    link_active_path \
      "$SEED_ROOT/base_training" \
      "$ACTIVE_BASE"

    # ------------------------------------------------------------------
    # Control finetuning
    # ------------------------------------------------------------------

    if [ ! -s "$SEED_ROOT/control_finetuning/control/model_best.ckpt" ]; then
        python -u \
          train/train.py control \
          2>&1 | tee \
          "$SEED_ROOT/logs/02_control_finetuning.log"

        CODE=${PIPESTATUS[0]}

        if [ "$CODE" -ne 0 ]; then
            echo "CONTROL_FINETUNING_FAILED seed=$SEED code=$CODE"
            exit "$CODE"
        fi

        mv \
          "$ACTIVE_CONTROL" \
          "$SEED_ROOT/control_finetuning"
    fi

    link_active_path \
      "$SEED_ROOT/control_finetuning" \
      "$ACTIVE_CONTROL"

    # ------------------------------------------------------------------
    # Neural refinement to R160 only
    # ------------------------------------------------------------------

    if [ ! -s "$SEED_ROOT/neural_refinement/08_R160/r160_best_state.pt" ]; then
        if [ "$VARIANT" = "global_ace_only" ]; then
            bash \
              "$ABLATION_ROOT/scripts/refine_global_ace_only.sh" \
              --fresh \
              2>&1 | tee \
              "$SEED_ROOT/logs/03_neural_refinement.log"
        else
            export MS2_STOP_AFTER_R160=1

            bash \
              train/_impl/run_refinement.sh \
              --fresh \
              2>&1 | tee \
              "$SEED_ROOT/logs/03_neural_refinement.log"

            unset MS2_STOP_AFTER_R160
        fi

        CODE=${PIPESTATUS[0]}

        if [ "$CODE" -ne 0 ]; then
            echo "NEURAL_REFINEMENT_FAILED seed=$SEED code=$CODE"
            exit "$CODE"
        fi

        mv \
          "$ACTIVE_REFINEMENT" \
          "$SEED_ROOT/neural_refinement"
    fi

    # ------------------------------------------------------------------
    # Locked test evaluation after validation-only model selection
    # ------------------------------------------------------------------

    python -u \
      "$ABLATION_ROOT/src/evaluate_r160_ablation.py" \
      --template "$ROOT/runs/_config/template.yml" \
      --config "$SEED_ROOT/control_finetuning/control/config.yml" \
      --checkpoint "$SEED_ROOT/neural_refinement/08_R160/r160_best_state.pt" \
      --out-dir "$SEED_ROOT/evaluation" \
      --variant "$VARIANT" \
      --seed "$SEED" \
      2>&1 | tee \
      "$SEED_ROOT/logs/04_r160_test_evaluation.log"

    CODE=${PIPESTATUS[0]}

    if [ "$CODE" -ne 0 ]; then
        echo "R160_EVALUATION_FAILED seed=$SEED code=$CODE"
        exit "$CODE"
    fi

    if [ -L "$ACTIVE_BASE" ]; then
        rm "$ACTIVE_BASE"
    fi

    if [ -L "$ACTIVE_CONTROL" ]; then
        rm "$ACTIVE_CONTROL"
    fi

    if [ -L "$ACTIVE_REFINEMENT" ]; then
        rm "$ACTIVE_REFINEMENT"
    fi

    echo
    echo "COMPLETE $VARIANT SEED $SEED"
done

python - \
  "$RUN_ROOT" \
  "$RESULT_ROOT" \
  "$VARIANT" \
  "$FORMAL_NAME" <<'PY'
from pathlib import Path
import json
import sys

import pandas as pd


run_root = Path(sys.argv[1])
result_root = Path(sys.argv[2])
variant = sys.argv[3]
formal_name = sys.argv[4]

rows = []

for seed in (42, 43, 44):
    path = (
        run_root
        / f"seed_{seed}"
        / "evaluation"
        / "result.json"
    )

    if not path.is_file():
        continue

    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    rows.append(data)

raw = pd.DataFrame(rows)

raw_path = (
    result_root
    / f"{variant}_raw.csv"
)

raw.to_csv(
    raw_path,
    index=False,
)

if len(raw) != 3:
    print(
        f"{variant.upper()}_PARTIAL "
        f"completed={len(raw)}/3"
    )
    raise SystemExit(1)

summary = {
    "model_variant":
        formal_name,

    "micro_cbin_mean":
        float(
            raw["micro_cbin"].mean()
        ),

    "micro_cbin_std":
        float(
            raw["micro_cbin"].std(
                ddof=1
            )
        ),

    "micro_jss_mean":
        float(
            raw["micro_jss"].mean()
        ),

    "micro_jss_std":
        float(
            raw["micro_jss"].std(
                ddof=1
            )
        ),

    "n_seeds":
        3,
}

summary_frame = pd.DataFrame(
    [summary]
)

summary_path = (
    result_root
    / f"{variant}_summary.csv"
)

summary_frame.to_csv(
    summary_path,
    index=False,
)

publication = pd.DataFrame(
    [
        {
            "Model variant":
                formal_name,

            "Random micro CBIN":
                (
                    f"{summary['micro_cbin_mean']:.6f}"
                    f" ± "
                    f"{summary['micro_cbin_std']:.6f}"
                ),

            "Random micro JSS":
                (
                    f"{summary['micro_jss_mean']:.6f}"
                    f" ± "
                    f"{summary['micro_jss_std']:.6f}"
                ),
        }
    ]
)

publication_path = (
    result_root
    / f"{variant}_publication_table.csv"
)

publication.to_csv(
    publication_path,
    index=False,
)

print()
print("=" * 110)
print(formal_name)
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
print(
    f"{variant.upper()}_ABLATION_COMPLETE"
)
PY
