#!/usr/bin/env python3

from __future__ import annotations
import os

import copy
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml

try:
    import lightning.pytorch as pl
except ModuleNotFoundError:
    import pytorch_lightning as pl


ROOT = Path(os.environ.get("FERA_MS_ROOT", Path(__file__).resolve().parents[2])).resolve()
RUNS_ROOT = Path(os.environ.get("FERA_MS_RUNS_DIR", ROOT / "runs")).resolve()

GLOBAL_SEED = int(
    os.environ.get(
        "MS2_GLOBAL_SEED",
        "42",
    )
)

sys.path.insert(
    0,
    str(ROOT / "code/src"),
)

sys.path.insert(
    0,
    str(ROOT / "code"),
)

sys.path.insert(
    0,
    str(ROOT),
)

from ms2spectra import workflow
from ms2spectra.training import FragGNNPL

from train._impl.trainer_helpers import (
    load_state_dict_any,
    make_trainer,
)

from train._impl.config_builder import (
    load_yaml,
    materialize_training_config,
    prepare_effective_config,
)


CONFIG_BUNDLE = ROOT / "config/train.yml"

RUNTIME_CONFIG = materialize_training_config(
    CONFIG_BUNDLE,
    RUNS_ROOT / "_config",
)

TEMPLATE = RUNTIME_CONFIG["template"]
BASE_MODEL_CUSTOM = RUNTIME_CONFIG["base_stage"]
RETAINED_CONTROL_CUSTOM = RUNTIME_CONFIG["continuation_stage"]

OUTPUT_ROOT = RUNS_ROOT / "structural_backbone_gine_cutchem_only"

INITIAL_TRAINING_DIR = OUTPUT_ROOT / "initial_training_base_model_40ep"
CONTINUATION_DIR = OUTPUT_ROOT / "continuation_retained_control_10ep"
FINAL_DIR = OUTPUT_ROOT / "final"

MONITOR = "val_cos_sim_0.01_epoch/mean"

BASE_MODEL_INITIAL_TRAINING_BASELINE = 0.5897715092
FORMAL_RETAINED_CONTROL_BASELINE = 0.5927286148


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(
                8 * 1024 * 1024
            )

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def write_yaml(
    path: Path,
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        yaml.safe_dump(
            config,
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )


def rotate_output() -> None:
    if not OUTPUT_ROOT.exists():
        return

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    backup = OUTPUT_ROOT.with_name(
        OUTPUT_ROOT.name
        + ".bak_"
        + timestamp
    )

    OUTPUT_ROOT.rename(backup)

    print(
        "[BACKUP]",
        backup,
    )


def build_stage_configs() -> tuple[
    dict[str, Any],
    dict[str, Any],
]:
    original_base_config = copy.deepcopy(
        dict(
            workflow.load_config(
                str(TEMPLATE),
                str(BASE_MODEL_CUSTOM),
            )
        )
    )

    initial_training = copy.deepcopy(
        original_base_config
    )

    # ==========================================================
    # ==========================================================
    initial_training["frag_gnn_type"] = "GINE"

    frag_params = copy.deepcopy(
        initial_training["frag_params"]
    )

    frag_params[
        "pyg_edge_feats"
    ] = [
        "cut_chem",
    ]

    initial_training["frag_params"] = frag_params

    # ==========================================================
    # ==========================================================
    initial_training.update(
        {
            "seed":
                GLOBAL_SEED,

            "min_epochs":
                1,

            "max_epochs":
                40,

            "eval_test_split":
                False,

            "disable_checkpoints":
                False,

            "delete_checkpoints":
                False,

            "upload_checkpoints":
                False,

            "checkpoint_save_last":
                True,

            "checkpoint_metric":
                MONITOR,

            "checkpoint_metric_mode":
                "max",

            "wandb_name":
                "STRUCTURAL_BACKBONE_GINE_CUTCHEM_INITIAL_TRAINING",

            "wandb_group":
                "STRUCTURAL_BACKBONE_GINE_CUTCHEM_ONLY",
        }
    )

    template_custom = load_yaml(
        TEMPLATE
    )

    modified_base_model_custom = load_yaml(
        BASE_MODEL_CUSTOM
    )

    modified_base_model_custom[
        "frag_gnn_type"
    ] = "GINE"

    modified_frag_params = copy.deepcopy(
        modified_base_model_custom[
            "frag_params"
        ]
    )

    modified_frag_params[
        "pyg_edge_feats"
    ] = [
        "cut_chem",
    ]

    modified_base_model_custom[
        "frag_params"
    ] = modified_frag_params

    retained_control_custom = load_yaml(
        RETAINED_CONTROL_CUSTOM
    )

    continuation, copied_training = (
        prepare_effective_config(
            template_cfg=template_custom,
            base_model_custom=modified_base_model_custom,
            retained_control_custom=retained_control_custom,
            epochs=10,
        )
    )

    continuation.update(
        {
            "seed":
                GLOBAL_SEED,

            "eval_test_split":
                False,

            "disable_checkpoints":
                False,

            "delete_checkpoints":
                False,

            "upload_checkpoints":
                False,

            "checkpoint_save_last":
                True,

            "checkpoint_metric":
                MONITOR,

            "checkpoint_metric_mode":
                "max",

            "wandb_name":
                "STRUCTURAL_BACKBONE_GINE_CUTCHEM_CONTINUATION",

            "wandb_group":
                "STRUCTURAL_BACKBONE_GINE_CUTCHEM_ONLY",
        }
    )

    split_override = os.environ.get(
        "MS2_SPLIT_DP"
    )

    if split_override:
        initial_training["split_dp"] = split_override
        continuation["split_dp"] = split_override

        print(
            "[SPLIT OVERRIDE]",
            split_override,
        )

    # ==========================================================
    # ==========================================================
    assertions = {
        "initial_training_frag_gnn_type":
            initial_training["frag_gnn_type"],

        "continuation_frag_gnn_type":
            continuation["frag_gnn_type"],

        "initial_training_edge_feats":
            initial_training[
                "frag_params"
            ][
                "pyg_edge_feats"
            ],

        "continuation_edge_feats":
            continuation[
                "frag_params"
            ][
                "pyg_edge_feats"
            ],

        "initial_training_ce_insert_type":
            initial_training["ce_insert_type"],

        "continuation_ce_insert_type":
            continuation["ce_insert_type"],

        

        

        "initial_training_ce_depth_rank":
            initial_training[
                "use_ce_depth_rank_loss"
            ],

        "continuation_ce_depth_rank":
            continuation[
                "use_ce_depth_rank_loss"
            ],

        "initial_training_lr":
            initial_training["lr"],

        "initial_training_weight_decay":
            initial_training["weight_decay"],

        "initial_training_frag_layers":
            initial_training["frag_num_layers"],

        "initial_training_frag_hidden":
            initial_training["frag_hidden_size"],

        "initial_training_batch_size":
            initial_training["train_batch_size"],
    }

    assert (
        initial_training["frag_gnn_type"]
        == "GINE"
    )

    assert (
        continuation["frag_gnn_type"]
        == "GINE"
    )

    assert (
        initial_training["frag_params"][
            "pyg_edge_feats"
        ]
        == ["cut_chem"]
    )

    assert (
        continuation["frag_params"][
            "pyg_edge_feats"
        ]
        == ["cut_chem"]
    )

    assert (
        initial_training["ce_insert_type"]
        == "embed"
    )

    assert (
        continuation["ce_insert_type"]
        == "embed"
    )



    assert not bool(
        initial_training[
            "use_ce_depth_rank_loss"
        ]
    )

    assert not bool(
        continuation[
            "use_ce_depth_rank_loss"
        ]
    )

    assert (
        "use_metric_aligned_loss"
        not in initial_training
    )

    assert (
        "use_metric_aligned_loss"
        not in continuation
    )

    assert (
        int(
            initial_training[
                "frag_num_layers"
            ]
        )
        == 3
    )

    assert (
        float(initial_training["lr"])
        == float(original_base_config["lr"])
    )

    assert (
        float(
            initial_training[
                "weight_decay"
            ]
        )
        == float(
            original_base_config[
                "weight_decay"
            ]
        )
    )

    print()
    print("=" * 96)
    print("structural backbone CONFIG AUDIT")
    print("=" * 96)

    print(
        json.dumps(
            assertions,
            indent=2,
            ensure_ascii=False,
        )
    )

    print()
    print(
        "[retained control TRAINING SETTINGS]"
    )

    print(
        json.dumps(
            copied_training,
            indent=2,
            ensure_ascii=False,
        )
    )

    print("=" * 96)

    initial_training[
        "eval_test_split"
    ] = False

    continuation[
        "eval_test_split"
    ] = False

    return initial_training, continuation


def architecture_preflight(
    config: dict[str, Any],
) -> None:
    pl.seed_everything(
        int(config["seed"]),
        workers=True,
    )

    model = FragGNNPL(
        **config
    )

    fragment_model = model.model
    wrapper = (
        fragment_model
        .frag_embedder
    )

    wrapper_class = (
        wrapper
        .__class__
        .__name__
    )

    inner_class = (
        wrapper
        .gnn
        .__class__
        .__name__
    )

    gnn_type = str(
        wrapper.gnn_type
    )

    edge_dim = int(
        fragment_model
        .frag_edge_feats_size
    )

    ce_type = str(
        fragment_model
        .ce_insert_type
    )

    print()
    print("=" * 96)
    print("structural backbone ARCHITECTURE PREFLIGHT")
    print("=" * 96)

    print(
        "wrapper          :",
        wrapper_class,
    )

    print(
        "gnn_type         :",
        gnn_type,
    )

    print(
        "inner encoder    :",
        inner_class,
    )

    print(
        "edge feature dim :",
        edge_dim,
    )

    print(
        "CE encoder       :",
        ce_type,
    )

    if (
        gnn_type != "GINE"
        or inner_class != "GINE"
        or edge_dim != 10
        or ce_type != "embed"
    ):
        raise RuntimeError(
            "structural backbone结构检查失败："
            f"wrapper={wrapper_class}, "
            f"gnn_type={gnn_type}, "
            f"inner={inner_class}, "
            f"edge_dim={edge_dim}, "
            f"ce_type={ce_type}"
        )

    print(
        "[PREFLIGHT PASS] "
        "唯一架构变化为GINE+cut_chem"
    )

    print("=" * 96)

    del model


def train_stage(
    name: str,
    run_dir: Path,
    config: dict[str, Any],
    initialization_checkpoint: Path | None,
    patience: int,
    min_delta: float,
) -> tuple[float, Path]:
    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    config_path = (
        run_dir
        / "config.yml"
    )

    write_yaml(
        config_path,
        config,
    )

    print()
    print("=" * 96)
    print(name)
    print("=" * 96)

    print(
        "run directory :",
        run_dir,
    )

    print(
        "epochs        :",
        config["max_epochs"],
    )

    print(
        "lr            :",
        config["lr"],
    )

    print(
        "weight decay  :",
        config["weight_decay"],
    )

    print(
        "initialization:",
        (
            "random"
            if initialization_checkpoint
            is None
            else initialization_checkpoint
        ),
    )

    print(
        "test used     : False"
    )

    print("=" * 96)

    pl.seed_everything(
        int(config["seed"]),
        workers=True,
    )

    train_dataset, val_dataset = (
        workflow.init_dataset(
            config,
            splits=(
                "train",
                "val",
            ),
        )
    )

    train_loader = (
        workflow.init_dataloader(
            train_dataset,
            config,
        )
    )

    val_loader = (
        workflow.init_dataloader(
            val_dataset,
            config,
        )
    )

    model = FragGNNPL(
        **config
    )

    if (
        initialization_checkpoint
        is not None
    ):
        state_dict = (
            load_state_dict_any(
                initialization_checkpoint
            )
        )

        incompatibility = (
            model.load_state_dict(
                state_dict,
                strict=True,
            )
        )

        print(
            "[STRICT WEIGHT LOAD]",
            incompatibility,
        )

    trainable = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    total = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    print(
        "[PARAMETERS] "
        f"trainable={trainable}, "
        f"total={total}"
    )

    trainer, checkpoint_callback = (
        make_trainer(
            run_dir=run_dir,
            config=config,
            patience=patience,
            min_delta=min_delta,
            training=True,
        )
    )

    resume_ckpt_path = os.environ.get("MS2_RESUME_CKPT")

    checkpoint_dir = str(
        trainer.checkpoint_callback.dirpath
    )

    if (
        resume_ckpt_path
        and "initial_training_base_model_40ep"
        not in checkpoint_dir
    ):
        resume_ckpt_path = None

    if resume_ckpt_path:
        print(
            f"[RESUME] {resume_ckpt_path}"
        )

    trainer.fit(
        model,
        ckpt_path=resume_ckpt_path,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    if checkpoint_callback is None:
        raise RuntimeError(
            "checkpoint callback未创建"
        )

    best_source = Path(
        checkpoint_callback
        .best_model_path
    )

    if not best_source.is_file():
        raise FileNotFoundError(
            best_source
        )

    canonical_best = (
        run_dir
        / "model_best.ckpt"
    )

    shutil.copy2(
        best_source,
        canonical_best,
    )

    last_source = (
        run_dir
        / "checkpoints"
        / "last.ckpt"
    )

    if last_source.is_file():
        shutil.copy2(
            last_source,
            run_dir
            / "model_last.ckpt",
        )

    best_score = float(
        checkpoint_callback
        .best_model_score
        .detach()
        .cpu()
    )

    result = {
        "name":
            name,

        "best_val_cosine":
            best_score,

        "best_checkpoint":
            str(canonical_best),

        "checkpoint_sha256":
            sha256_file(
                canonical_best
            ),

        "initialization_checkpoint":
            (
                None
                if initialization_checkpoint
                is None
                else str(
                    initialization_checkpoint
                )
            ),

        "test_used":
            False,
    }

    (
        run_dir
        / "stage_summary.json"
    ).write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        f"[{name}] "
        f"best validation cosine = "
        f"{best_score:.10f}"
    )

    return (
        best_score,
        canonical_best,
    )


def main() -> None:
    required = [
        TEMPLATE,
        BASE_MODEL_CUSTOM,
        RETAINED_CONTROL_CUSTOM,
        ROOT
        / "code/src/ms2spectra/model.py",
        ROOT
        / "code/src/ms2spectra/training.py",
        ROOT
        / "code/src/ms2spectra/utils/nn_utils.py",
        ROOT
        / "code/src/ms2spectra/utils/frag_utils.py",
        ROOT
        / "data",
    ]

    missing = [
        path
        for path in required
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "缺少必要文件：\n"
            + "\n".join(
                str(path)
                for path in missing
            )
        )

    rotate_output()

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=False,
    )

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = (
            True
        )

        torch.backends.cudnn.allow_tf32 = (
            True
        )

    try:
        torch.set_float32_matmul_precision(
            "high"
        )
    except Exception:
        pass

    initial_training_config, continuation_config = (
        build_stage_configs()
    )

    architecture_preflight(
        initial_training_config
    )

    initial_training_score, initial_training_checkpoint = (
        train_stage(
            name="INITIAL_TRAINING_STRUCTURAL_BACKBONE_GINE_40EP",
            run_dir=INITIAL_TRAINING_DIR,
            config=initial_training_config,
            initialization_checkpoint=None,
            patience=1000,
            min_delta=0.0,
        )
    )

    continuation_score, continuation_checkpoint = (
        train_stage(
            name="CONTINUATION_STRUCTURAL_BACKBONE_RETAINED_CONTROL_10EP",
            run_dir=CONTINUATION_DIR,
            config=continuation_config,
            initialization_checkpoint=(
                initial_training_checkpoint
            ),
            patience=4,
            min_delta=1.0e-4,
        )
    )

    if continuation_score >= initial_training_score:
        selected_stage = "retained_control_continuation"
        selected_score = continuation_score
        selected_checkpoint = (
            continuation_checkpoint
        )
        selected_config = continuation_config
    else:
        selected_stage = "structural_backbone_initial_training"
        selected_score = initial_training_score
        selected_checkpoint = (
            initial_training_checkpoint
        )
        selected_config = initial_training_config

    FINAL_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    final_checkpoint = (
        FINAL_DIR
        / "model.ckpt"
    )

    shutil.copy2(
        selected_checkpoint,
        final_checkpoint,
    )

    write_yaml(
        FINAL_DIR
        / "config.yml",
        selected_config,
    )

    if (
        selected_score
        >= FORMAL_RETAINED_CONTROL_BASELINE
    ):
        decision = (
            "PASS_OR_EXCEED_FORMAL_BASELINE"
        )

        next_action = (
            "retain_GINE_cutchem_and_test_"
            "continuous_CE_as_separate_ablation"
        )

    elif (
        selected_score
        >= BASE_MODEL_INITIAL_TRAINING_BASELINE
    ):
        decision = (
            "GINE_CUTCHEM_NEUTRAL_BUT_"
            "RETAINED_CONTROL_GAIN_NOT_RECOVERED"
        )

        next_action = (
            "inspect_stagewise_allocation_"
            "before_any_new_module"
        )

    else:
        decision = (
            "REJECT_CURRENT_GINE_CUTCHEM"
        )

        next_action = (
            "return_to_NodeMLP_and_do_not_"
            "add_continuous_CE"
        )

    summary = {
        "experiment":
            "STRUCTURAL_BACKBONE_GINE_CUTCHEM_ONLY",

        "only_model_changes": {
            "frag_gnn_type":
                "NodeMLP -> GINE",

            "frag_params."
            "pyg_edge_feats":
                "[] -> [cut_chem]",
        },

        "initial_training_best_val_cosine":
            initial_training_score,

        "continuation_best_val_cosine":
            continuation_score,

        "selected_stage":
            selected_stage,

        "selected_best_val_cosine":
            selected_score,

        "base_model_initial_training_baseline":
            BASE_MODEL_INITIAL_TRAINING_BASELINE,

        "formal_retained_control_baseline":
            FORMAL_RETAINED_CONTROL_BASELINE,

        "delta_vs_structural_backbone_initial_training":
            (
                selected_score
                - BASE_MODEL_INITIAL_TRAINING_BASELINE
            ),

        "delta_vs_formal_retained_control":
            (
                selected_score
                - FORMAL_RETAINED_CONTROL_BASELINE
            ),

        "decision":
            decision,

        "next_action":
            next_action,

        "selected_checkpoint":
            str(final_checkpoint),

        "selected_checkpoint_sha256":
            sha256_file(
                final_checkpoint
            ),

        "test_used":
            False,

        "github_modified":
            False,
    }

    (
        FINAL_DIR
        / "decision.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print("structural backbone FINAL VALIDATION RESULT")
    print("=" * 100)

    print(
        "initial_training best cosine :",
        f"{initial_training_score:.10f}",
    )

    print(
        "continuation best cosine :",
        f"{continuation_score:.10f}",
    )

    print(
        "selected stage     :",
        selected_stage,
    )

    print(
        "selected cosine    :",
        f"{selected_score:.10f}",
    )

    print(
        "delta vs base model initial_training :",
        f"{selected_score - BASE_MODEL_INITIAL_TRAINING_BASELINE:+.10f}",
    )

    print(
        "delta vs formal    :",
        f"{selected_score - FORMAL_RETAINED_CONTROL_BASELINE:+.10f}",
    )

    print(
        "DECISION           :",
        decision,
    )

    print(
        "NEXT               :",
        next_action,
    )

    print(
        "test used          : False"
    )

    print(
        "GitHub modified    : False"
    )

    print(
        "final checkpoint   :",
        final_checkpoint,
    )

    print(
        "decision file      :",
        FINAL_DIR
        / "decision.json",
    )

    print("=" * 100)


if __name__ == "__main__":
    main()
