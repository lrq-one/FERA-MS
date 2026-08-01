#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${FERA_MS_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
RUNS_ROOT="${FERA_MS_RUNS_DIR:-$ROOT/runs}"
OUT="$RUNS_ROOT/full_fera_ms"
DIAG="$ROOT/train/_impl/refinement_steps"
TEMPLATE="$RUNS_ROOT/_config/template.yml"

BASE_CONFIG="$RUNS_ROOT/global_ace_control_ce_trajectory_ablation/control/config.yml"

BASE_CHECKPOINT="$RUNS_ROOT/global_ace_control_ce_trajectory_ablation/control/model_best.ckpt"

FRESH=0

if [ "${1:-}" = "--fresh" ]; then
    FRESH=1
fi

cd "$ROOT" || {
    echo "无法进入：$ROOT"
    exit 1
}

export PYTHONPATH="$ROOT/code/src:$ROOT/code:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export FERA_MS_ROOT="$ROOT"
export FERA_MS_RUNS_DIR="$RUNS_ROOT"

if [ "$FRESH" -eq 1 ] && [ -d "$OUT" ]; then
    BACKUP="${OUT}.bak_$(date +%Y%m%d_%H%M%S)"
    mv "$OUT" "$BACKUP"
    echo "旧输出已备份：$BACKUP"
fi

mkdir -p \
    "$OUT/logs" \
    "$OUT/preflight" \
    "$OUT/formula_composition_refinement" \
    "$OUT/collision_energy_response_refinement" \
    "$OUT/neural_refinement" \
    "$OUT/peak_distillation_warmup" \
    "$OUT/peak_distillation_continuation" \
    "$OUT/fragment_representation_refinement" \
    "$OUT/bounded_residual_flow_refinement" \
    "$OUT/final_peak_distillation" \
    "$OUT/candidate_reranking" \
    "$OUT/spectrum_allocation" \

if [ ! -f "$TEMPLATE" ]; then
    echo "缺少模板：$TEMPLATE"
    exit 1
fi

if [ ! -f "$BASE_CONFIG" ]; then
    echo "缺少global ACE control配置：$BASE_CONFIG"
    exit 1
fi

if [ ! -f "$BASE_CHECKPOINT" ]; then
    echo "缺少global ACE control checkpoint：$BASE_CHECKPOINT"
    exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "没有检测到nvidia-smi，停止。"
    exit 1
fi

python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA不可用")

print("CUDA:", torch.cuda.get_device_name(0))

try:
    import lightgbm
    print("LightGBM:", lightgbm.__version__)
except Exception as exc:
    raise SystemExit(
        f"LightGBM不可用：{exc!r}"
    )
PY

PREFLIGHT_CODE=$?

if [ "$PREFLIGHT_CODE" -ne 0 ]; then
    echo "环境检查失败。"
    exit 1
fi

python - "$ROOT/config/train.yml" \
    > "$OUT/mainline_hparams.env" \
    2> "$OUT/mainline_hparams.log" <<'PY_HPARAMS'
import os
from pathlib import Path
import shlex
import sys
import yaml

config = yaml.safe_load(
    Path(sys.argv[1]).read_text(
        encoding="utf-8"
    )
)

params = config.get("postprocessing_env")

if not isinstance(params, dict):
    raise RuntimeError(
        "缺少postprocessing_env配置"
    )

for key, value in params.items():
    print(
        f"{key}={shlex.quote(str(value))}"
    )
PY_HPARAMS

HPARAM_CODE=$?

if [ "$HPARAM_CODE" -ne 0 ]; then
    echo "主线超参数读取失败。"
    exit 1
fi

# shellcheck disable=SC1090
source "$OUT/mainline_hparams.env"


if [ -n "${MS2_GLOBAL_SEED:-}" ]; then
    CANDIDATE_RERANKER_SEED="$MS2_GLOBAL_SEED"
fi

cat > "$OUT/effective_seed.env" <<EOF
MS2_GLOBAL_SEED=${MS2_GLOBAL_SEED:-}
CANDIDATE_RERANKER_SEED=$CANDIDATE_RERANKER_SEED
EOF

echo "Effective pipeline seed: ${MS2_GLOBAL_SEED:-unset}"
echo "Effective candidate reranker/spectrum allocator seed: $CANDIDATE_RERANKER_SEED"

run_stage() {
    LABEL="$1"
    shift

    LOG_FILE="$OUT/logs/${LABEL}.log"
    COMMAND_FILE="$OUT/logs/${LABEL}.command.txt"

    printf '%q ' "$@" \
        > "$COMMAND_FILE"

    printf '\n' \
        >> "$COMMAND_FILE"

    echo
    echo "================================================================================================"
    echo "$LABEL"
    echo "================================================================================================"
    cat "$COMMAND_FILE"
    echo

    "$@" 2>&1 \
        | tee "$LOG_FILE"

    CODE=${PIPESTATUS[0]}

    echo
    echo "$LABEL exit code: $CODE"

    return "$CODE"
}

metric() {
    python "$ROOT/train/_impl/stage_metrics.py" \
        metric \
        "$1"
}

echo
echo "================================================================================================"
echo "FERA-MS REFINEMENT PIPELINE"
echo "================================================================================================"
echo "test used      : False"
echo "output         : $OUT"
echo "================================================================================================"

python - <<'PY' \
    2>&1 \
    | tee "$OUT/preflight/preflight.log"
import os
from pathlib import Path

import torch

from ms2spectra.workflow import (
    load_config,
    init_dataset,
    init_dataloader,
)
from ms2spectra.training import FragGNNPL

import importlib.util


root = Path(os.environ["FERA_MS_ROOT"])

script = (
    root
    / "train/_impl/refinement_steps/"
    "formula_composition.py"
)

spec = importlib.util.spec_from_file_location(
    "formula_composition_preflight",
    str(script),
)

module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Args:
    hidden = 128
    dropout = 0.05
    delta_scale = 0.05
    formula_comp_feat_size = 18
    bin_res = 0.01
    max_bins = 0
    ce_binned_aux_weight = 0.0015
    low_w = 0.30
    mid_w = 1.50
    high_w = 2.00
    support_oracle_weight = 0.0
    support_oracle_false_weight = 0.25


config = load_config(
    root / "runs/_config/template.yml",
    root
    / "runs/global_ace_control_ce_trajectory_ablation/"
    "control/config.yml",
)

config = module.override_cfg(
    config,
    Args(),
)

train_dataset = init_dataset(
    config,
    splits=("train",),
)[0]

loader = init_dataloader(
    train_dataset,
    config,
)

model = FragGNNPL(
    **config
)

checkpoint = torch.load(
    root
    / "runs/global_ace_control_ce_trajectory_ablation/"
    "control/model_best.ckpt",
    map_location="cpu",
    weights_only=False,
)

state_dict = (
    checkpoint["state_dict"]
    if (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    )
    else checkpoint
)

missing, unexpected = model.load_state_dict(
    state_dict,
    strict=False,
)

print("missing keys:", len(missing))
for key in missing[:40]:
    print("  missing:", key)

print("unexpected keys:", len(unexpected))
for key in unexpected[:40]:
    print("  unexpected:", key)

allowed_missing_prefixes = (
    "model.formula_comp_residual_head",
)

bad_missing = [
    key
    for key in missing
    if not key.startswith(
        allowed_missing_prefixes
    )
]

if bad_missing:
    raise RuntimeError(
        "global ACE control→formula-composition refinement存在非预期missing keys："
        + repr(
            bad_missing[:40]
        )
    )

if unexpected:
    raise RuntimeError(
        "global ACE control→formula-composition refinement存在unexpected keys："
        + repr(
            unexpected[:40]
        )
    )

batch = next(
    iter(loader)
)

device = torch.device("cuda")
model = model.to(device)
batch = module.move_to_device(
    batch,
    device,
)

model.eval()

with torch.no_grad():
    result = model._common_step(
        batch,
        split="train",
        log=False,
    )

if not torch.isfinite(
    result["mean_loss"]
):
    raise RuntimeError(
        "preflight mean_loss非有限值"
    )

print(
    "preflight mean_loss:",
    float(
        result["mean_loss"]
        .detach()
        .cpu()
    ),
)

print("GLOBAL_ACE_CONTROL_TO_FORMULA_COMPOSITION_PREFLIGHT_PASSED")
PY

PREFLIGHT_RUN=${PIPESTATUS[0]}

if [ "$PREFLIGHT_RUN" -ne 0 ]; then
    echo "global ACE control→formula-composition refinement兼容性检查失败。"
    exit 1
fi


# =============================================================================
# formula-composition refinement
# =============================================================================

FORMULA_COMPOSITION_CKPT="$OUT/formula_composition_refinement/formula_composition_best_state.pt"

if [ ! -f "$FORMULA_COMPOSITION_CKPT" ]; then
    run_stage \
        "formula_composition_refinement" \
        python -u \
        "$DIAG/formula_composition.py" \
        --template "$TEMPLATE" \
        --config "$BASE_CONFIG" \
        --ckpt_path "$BASE_CHECKPOINT" \
        --out_dir "$OUT/formula_composition_refinement" \
        --epochs 4 \
        --max_train_batches -1 \
        --lr 5e-5 \
        --weight_decay 1e-5 \
        --hidden 128 \
        --dropout 0.05 \
        --delta_scale 0.05 \
        --formula_comp_feat_size 18 \
        --bin_res 0.01 \
        --max_bins 0 \
        --ce_binned_aux_weight 0.0015 \
        --support_oracle_weight 0.0 \
        --support_oracle_false_weight 0.25 \
        --low_w 0.30 \
        --mid_w 1.50 \
        --high_w 2.00

    if [ $? -ne 0 ]; then
        exit 1
    fi
else
    echo "[RESUME] formula-composition refinement checkpoint已存在。"
fi


# =============================================================================
# collision-energy response refinement
# =============================================================================

COLLISION_ENERGY_RESPONSE_CKPT="$OUT/collision_energy_response_refinement/collision_energy_response_best_state.pt"

if [ ! -f "$COLLISION_ENERGY_RESPONSE_CKPT" ]; then
    run_stage \
        "collision_energy_response_refinement" \
        python -u \
        "$DIAG/collision_energy_response.py" \
        --template "$TEMPLATE" \
        --config "$BASE_CONFIG" \
        --ckpt_path "$FORMULA_COMPOSITION_CKPT" \
        --out_dir "$OUT/collision_energy_response_refinement" \
        --epochs 4 \
        --max_train_batches -1 \
        --lr 5e-5 \
        --weight_decay 1e-5 \
        --formula_comp_hidden 128 \
        --formula_comp_dropout 0.05 \
        --formula_comp_delta_scale 0.05 \
        --formula_comp_feat_size 18 \
        --ce_hidden 128 \
        --ce_dropout 0.05 \
        --ce_delta_scale 0.025 \
        --ce_use_formula_comp \
        --ce_use_depth \
        --ce_use_h \
        --bin_res 0.01 \
        --max_bins 0 \
        --ce_binned_aux_weight 0.0015 \
        --support_oracle_weight 0.0 \
        --support_oracle_false_weight 0.20 \
        --low_w 0.25 \
        --mid_w 1.75 \
        --high_w 2.25

    if [ $? -ne 0 ]; then
        exit 1
    fi
else
    echo "[RESUME] collision-energy response refinement checkpoint已存在。"
fi


FORMULA_COMMON=(
    --formula_comp_hidden 128
    --formula_comp_dropout 0.05
    --formula_comp_delta_scale 0.05
    --formula_comp_feat_size 18
    --ce_hidden 128
    --ce_dropout 0.05
    --ce_delta_scale 0.02
    --ce_use_formula_comp
    --ce_use_depth
    --ce_use_h
    --bin_res 0.01
    --max_bins 0
    --ce_binned_aux_weight 0.0015
    --support_oracle_weight 0.0
    --support_oracle_false_weight 0.20
    --low_w 0.25
    --mid_w 1.75
    --high_w 2.25
    --formula_aux_weight 0.0005
    --formula_tol 0.01
    --formula_mz_sigma 0.003
    --hard_formula_topk 3
    --formula_score_mode max
    --prob_alpha 0.0
    --formula_kl_weight 1.0
    --formula_rank_weight 0.2
    --formula_false_weight 0.01
    --formula_target_topk 5
    --formula_neg_topk 20
    --formula_margin 0.5
    --formula_neg_target_max 0.002
    --formula_low_w 0.05
    --formula_mid_w 1.5
    --formula_high_w 2.0
)


# =============================================================================
# neural refinement
# =============================================================================

NEURAL_REFINEMENT_CHECKPOINT="$OUT/neural_refinement/neural_refinement_best_state.pt"

if [ ! -f "$NEURAL_REFINEMENT_CHECKPOINT" ]; then
    run_stage \
        "neural_refinement" \
        python -u \
        "$DIAG/neural_refinement.py" \
        --template "$TEMPLATE" \
        --config "$BASE_CONFIG" \
        --ckpt_path "$COLLISION_ENERGY_RESPONSE_CKPT" \
        --out_dir "$OUT/neural_refinement" \
        --epochs 4 \
        --max_train_batches -1 \
        --lr 5e-6 \
        --weight_decay 1e-5 \
        "${FORMULA_COMMON[@]}" \
        --train_formula_composition \
        --train_formula_module

    if [ $? -ne 0 ]; then
        exit 1
    fi
else
    echo "[RESUME] neural refinement checkpoint已存在。"
fi


# =============================================================================
# peak distillation warmup
# =============================================================================

PEAK_DISTILLATION_WARMUP_CKPT="$OUT/peak_distillation_warmup/neural_refinement_best_state.pt"

if [ ! -f "$PEAK_DISTILLATION_WARMUP_CKPT" ]; then
    run_stage \
        "peak_distillation_warmup" \
        python -u \
        "$DIAG/neural_refinement.py" \
        --template "$TEMPLATE" \
        --config "$BASE_CONFIG" \
        --ckpt_path "$NEURAL_REFINEMENT_CHECKPOINT" \
        --out_dir "$OUT/peak_distillation_warmup" \
        --epochs 8 \
        --max_train_batches -1 \
        --lr 3e-6 \
        --weight_decay 1e-5 \
        "${FORMULA_COMMON[@]}" \
        --train_formula_composition \
        --train_formula_module

    if [ $? -ne 0 ]; then
        exit 1
    fi
else
    echo "[RESUME] peak distillation warmup checkpoint已存在。"
fi


# =============================================================================
# peak distillation continuation
# =============================================================================

PEAK_DISTILLATION_CONTINUATION_CKPT="$OUT/peak_distillation_continuation/neural_refinement_best_state.pt"

if [ ! -f "$PEAK_DISTILLATION_CONTINUATION_CKPT" ]; then
    run_stage \
        "peak_distillation_continuation" \
        python -u \
        "$DIAG/neural_refinement.py" \
        --template "$TEMPLATE" \
        --config "$BASE_CONFIG" \
        --ckpt_path "$PEAK_DISTILLATION_WARMUP_CKPT" \
        --out_dir "$OUT/peak_distillation_continuation" \
        --epochs 6 \
        --max_train_batches -1 \
        --lr 1e-6 \
        --weight_decay 1e-5 \
        "${FORMULA_COMMON[@]}" \
        --train_formula_composition \
        --train_formula_module

    if [ $? -ne 0 ]; then
        exit 1
    fi
else
    echo "[RESUME] peak distillation continuation checkpoint已存在。"
fi


# =============================================================================
# fragment representation refinement reconstructed from surviving train_frag_rep implementation
# =============================================================================

FRAGMENT_REPRESENTATION_CKPT="$OUT/fragment_representation_refinement/neural_refinement_best_state.pt"

if [ ! -f "$FRAGMENT_REPRESENTATION_CKPT" ]; then
    run_stage \
        "fragment_representation_refinement" \
        python -u \
        "$DIAG/neural_refinement.py" \
        --template "$TEMPLATE" \
        --config "$BASE_CONFIG" \
        --ckpt_path "$PEAK_DISTILLATION_CONTINUATION_CKPT" \
        --out_dir "$OUT/fragment_representation_refinement" \
        --epochs 6 \
        --max_train_batches -1 \
        --lr 8e-7 \
        --weight_decay 1e-5 \
        "${FORMULA_COMMON[@]}" \
        --train_formula_composition \
        --train_formula_module \
        --train_frag_rep

    if [ $? -ne 0 ]; then
        exit 1
    fi
else
    echo "[RESUME] fragment representation refinement checkpoint已存在。"
fi


# =============================================================================
# bounded residual flow bounded residual flow, validation gated
# =============================================================================

BOUNDED_RESIDUAL_FLOW_CKPT="$OUT/bounded_residual_flow_refinement/neural_refinement_best_state.pt"

if [ ! -f "$BOUNDED_RESIDUAL_FLOW_CKPT" ]; then
    run_stage \
        "bounded_residual_flow_refinement" \
        python -u \
        "$DIAG/neural_refinement.py" \
        --template "$TEMPLATE" \
        --config "$BASE_CONFIG" \
        --ckpt_path "$FRAGMENT_REPRESENTATION_CKPT" \
        --out_dir "$OUT/bounded_residual_flow_refinement" \
        --epochs 6 \
        --max_train_batches -1 \
        --lr 5e-7 \
        --weight_decay 1e-5 \
        "${FORMULA_COMMON[@]}" \
        --use_ce_flowfrag \
        --ce_flowfrag_lambda_max 0.15 \
        --ce_flowfrag_hidden 128 \
        --ce_flowfrag_dropout 0.05 \
        --ce_flowfrag_max_depth 4 \
        --ce_flowfrag_mixture_hidden 128 \
        --ce_flowfrag_mixture_dropout 0.05 \
        --ce_flowfrag_mixture_init_bias -3.0 \
        --ce_flowfrag_delta_clip 3.0 \
        --ce_flowfrag_use_direct_node \
        --ce_flowfrag_direct_mix 0.35 \
        --train_ce_flowfrag

    if [ $? -ne 0 ]; then
        exit 1
    fi
else
    echo "[RESUME] bounded residual flow checkpoint已存在。"
fi

BOUNDED_RESIDUAL_FLOW_SELECTED="$(
    python "$ROOT/train/_impl/stage_metrics.py" \
        select \
        --before "$OUT/bounded_residual_flow_refinement/neural_refinement_val_epoch0_before.csv" \
        --best "$OUT/bounded_residual_flow_refinement/neural_refinement_best_val.csv" \
        --parent "$FRAGMENT_REPRESENTATION_CKPT" \
        --child "$BOUNDED_RESIDUAL_FLOW_CKPT" \
        --decision "$OUT/bounded_residual_flow_refinement/validation_gate.json" \
        --min-delta 0.0
)"

echo "bounded residual flow selected checkpoint: $BOUNDED_RESIDUAL_FLOW_SELECTED"


# =============================================================================
# final peak distillation
# =============================================================================

FINAL_PEAK_DISTILLATION_CKPT="$OUT/final_peak_distillation/final_peak_distillation_best_state.pt"

if [ ! -f "$FINAL_PEAK_DISTILLATION_CKPT" ]; then
    run_stage \
        "final_peak_distillation" \
        python -u \
        "$DIAG/peak_distillation.py" \
        --template "$TEMPLATE" \
        --config "$BASE_CONFIG" \
        --ckpt_path "$BOUNDED_RESIDUAL_FLOW_SELECTED" \
        --out_dir "$OUT/final_peak_distillation" \
        --epochs 8 \
        --max_train_batches -1 \
        --lr 2e-7 \
        --weight_decay 1e-5 \
        --formula_comp_hidden 128 \
        --formula_comp_dropout 0.05 \
        --formula_comp_delta_scale 0.05 \
        --formula_comp_feat_size 18 \
        --ce_hidden 128 \
        --ce_dropout 0.05 \
        --ce_delta_scale 0.020 \
        --ce_use_formula_comp \
        --ce_use_depth \
        --ce_use_h \
        --use_ce_flowfrag \
        --ce_flowfrag_lambda_max 0.15 \
        --ce_flowfrag_hidden 128 \
        --ce_flowfrag_dropout 0.05 \
        --ce_flowfrag_max_depth 4 \
        --ce_flowfrag_mixture_hidden 128 \
        --ce_flowfrag_mixture_dropout 0.05 \
        --ce_flowfrag_mixture_init_bias -3.0 \
        --ce_flowfrag_delta_clip 3.0 \
        --ce_flowfrag_use_direct_node \
        --ce_flowfrag_direct_mix 0.35 \
        --bin_res 0.01 \
        --max_bins 0 \
        --eval_bin_res 0.01 \
        --ce_binned_aux_weight 0.0015 \
        --low_w 0.25 \
        --mid_w 1.75 \
        --high_w 2.25 \
        --peak_oracle_weight 0.02 \
        --false_mass_weight 0.015 \
        --hit_mass_weight 0.003 \
        --oracle_bin_res 0.01 \
        --oracle_mz_tol 0.01 \
        --oracle_mz_sigma 0.003 \
        --oracle_low_w 0.50 \
        --oracle_mid_w 2.00 \
        --oracle_high_w 3.00 \
        --train_formula_composition \
        --train_formula_module \
        --train_frag_rep \
        --train_ce_flowfrag \
        --train_refiner \
        --train_render_gate

    if [ $? -ne 0 ]; then
        exit 1
    fi
else
    echo "[RESUME] final peak distillation checkpoint已存在。"
fi

FINAL_PEAK_DISTILLATION_COS="$(
    metric \
        "$OUT/final_peak_distillation/final_peak_distillation_best_val.csv"
)"

echo "final peak distillation validation cosine: $FINAL_PEAK_DISTILLATION_COS"


# =============================================================================
# candidate reranker exact residual scorer
# =============================================================================

CANDIDATE_RERANKER_PKL="$OUT/candidate_reranking/candidate_reranker_regressor.pkl"

run_candidate_reranker() {
    run_stage \
        "candidate_reranking" \
        python -u \
        "$DIAG/candidate_reranker.py" \
        --template "$TEMPLATE" \
        --config "$BASE_CONFIG" \
        --ckpt_path "$FINAL_PEAK_DISTILLATION_CKPT" \
        --out_dir "$OUT/candidate_reranking" \
        --seed "$CANDIDATE_RERANKER_SEED" \
        --backend lightgbm \
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
        "$@"
}

if [ ! -f "$CANDIDATE_RERANKER_PKL" ]; then
    run_candidate_reranker
    if [ $? -ne 0 ]; then
        exit 1
    fi
elif [ ! -s "$OUT/candidate_reranking/candidate_reranker_alpha_val.csv" ]; then
    echo "[RECOVER] regressor已存在；只重新运行validation alpha grid，不覆盖regressor。"
    REGRESSOR_SHA_BEFORE="$(sha256sum "$CANDIDATE_RERANKER_PKL" | awk '{print $1}')"
    run_candidate_reranker --load_regressor "$CANDIDATE_RERANKER_PKL"
    if [ $? -ne 0 ]; then
        exit 1
    fi
    REGRESSOR_SHA_AFTER="$(sha256sum "$CANDIDATE_RERANKER_PKL" | awk '{print $1}')"
    if [ "$REGRESSOR_SHA_BEFORE" != "$REGRESSOR_SHA_AFTER" ]; then
        echo "alpha-only recovery错误覆盖了已有regressor。"
        exit 1
    fi
else
    echo "[RESUME] candidate reranker regressor和alpha summary均已存在。"
fi

BEST_ALPHA="$(
    python - "$OUT/candidate_reranking/candidate_reranker_alpha_val.csv" <<'PY_ALPHA'
import math
import sys

import pandas as pd


path = sys.argv[1]
frame = pd.read_csv(path)

required = {
    "alpha",
    "val_cos",
}

missing = required - set(frame.columns)

if missing:
    raise RuntimeError(
        "alpha表缺少字段："
        + repr(sorted(missing))
    )

frame["alpha"] = pd.to_numeric(
    frame["alpha"],
    errors="coerce",
)

frame["val_cos"] = pd.to_numeric(
    frame["val_cos"],
    errors="coerce",
)

if "val_jss" in frame.columns:
    frame["val_jss"] = pd.to_numeric(
        frame["val_jss"],
        errors="coerce",
    )
else:
    frame["val_jss"] = 0.0

frame = frame[
    frame["alpha"].notna()
    & frame["val_cos"].notna()
].copy()

frame = frame[
    frame["alpha"].map(math.isfinite)
    & frame["val_cos"].map(math.isfinite)
]

if frame.empty:
    raise RuntimeError(
        "alpha表没有有效结果"
    )

best = (
    frame
    .sort_values(
        [
            "val_cos",
            "val_jss",
            "alpha",
        ],
        ascending=[
            False,
            False,
            True,
        ],
    )
    .iloc[0]
)

print(
    f"{float(best['alpha']):.12g}"
)
PY_ALPHA
)"

ALPHA_CODE=$?

if [ "$ALPHA_CODE" -ne 0 ] || [ -z "$BEST_ALPHA" ]; then
    echo "无法提取candidate reranker最佳alpha。"
    exit 1
fi

printf '%s\n' "$BEST_ALPHA" \
    > "$OUT/candidate_reranking/best_alpha.txt"

if [ ! -s "$OUT/candidate_reranking/candidate_reranker_alpha_val.csv" ]; then
    echo "[STOP] candidate reranker缺少candidate_reranker_alpha_val.csv"
    exit 1
fi

if ! python -c '
import math
import sys

value = float(sys.argv[1])

if not math.isfinite(value):
    raise ValueError(value)
' "$BEST_ALPHA"
then
    echo "[STOP] candidate reranker最佳alpha无效：'$BEST_ALPHA'"
    exit 1
fi

echo "candidate reranker validated alpha      : $BEST_ALPHA"

CANDIDATE_RERANKER_COS="$(
    metric \
        "$OUT/candidate_reranking/candidate_reranker_best_val.csv"
)"

echo "candidate reranker selected alpha      : $BEST_ALPHA"
echo "candidate reranker validation cosine  : $CANDIDATE_RERANKER_COS"


# =============================================================================
# spectrum allocator sibling, stronger residual allocator
# =============================================================================

SPECTRUM_ALLOCATOR_CKPT="$OUT/spectrum_allocation/spectrum_allocator_allocator_best.pt"

if [ ! -f "$SPECTRUM_ALLOCATOR_CKPT" ]; then
    run_stage \
        "spectrum_allocation" \
        python -u \
        "$DIAG/spectrum_allocator.py" \
        --template "$TEMPLATE" \
        --config "$BASE_CONFIG" \
        --ckpt_path "$FINAL_PEAK_DISTILLATION_CKPT" \
        --regressor_path "$CANDIDATE_RERANKER_PKL" \
        --out_dir "$OUT/spectrum_allocation" \
        --seed "$CANDIDATE_RERANKER_SEED" \
        --epochs "$B_EPOCHS" \
        --lr "$B_LR" \
        --weight_decay "$B_WEIGHT_DECAY" \
        --grad_clip 5.0 \
        --hidden "$B_HIDDEN" \
        --layers "$B_LAYERS" \
        --dropout "$B_DROPOUT" \
        --score_clip "$B_SCORE_CLIP" \
        --alpha "$BEST_ALPHA" \
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
        --max_extra_dims "$CANDIDATE_RERANKER_EXTRA_DIMS" \
        --max_train_batches 0

    if [ $? -ne 0 ]; then
        exit 1
    fi
else
    echo "[RESUME] spectrum allocator allocator已存在。"
fi



echo
echo "============================================================"
echo "FINAL MAINLINE REFINEMENT COMPLETE"
echo "============================================================"
echo "Refined backbone checkpoint:"
echo "$FINAL_PEAK_DISTILLATION_CKPT"
echo
echo "Candidate reranker:"
echo "$CANDIDATE_RERANKER_PKL"
echo
echo "Final spectrum allocator:"
echo "$SPECTRUM_ALLOCATOR_CKPT"
echo
echo "Test used for selection: False"
echo "============================================================"
