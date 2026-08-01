#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(os.environ.get("FERA_MS_ROOT", Path(__file__).resolve().parents[3])).resolve()
ABLATION_ROOT = (
    ROOT
    / "ablation_studies"
    / "fera_ms_panelb_ablation"
)

os.chdir(ROOT)

sys.path.insert(0, str(ROOT / "code/src"))
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT))

from ms2spectra.training import FragGNNPL
from train._impl import base_training as base_training
from train._impl import control_finetuning


RANDOM_SPLIT = (
    "data/split/"
    "nist20_qtof_cid_safe19659_qcv1_trainonly"
)

VARIANTS: dict[str, dict[str, Any]] = {
    "without_mz_offset_expansion": {
        "use_mz_offset_peak_expansion": False,
        "mz_offset_peak_steps": [0.0],
    },
    "without_rendered_peak_gate": {
        "use_rendered_peak_drop_gate": False,
    },
}

EXPECTED_CHANGED_KEYS = {
    "without_mz_offset_expansion": {
        "use_mz_offset_peak_expansion",
        "mz_offset_peak_steps",
    },
    "without_rendered_peak_gate": {
        "use_rendered_peak_drop_gate",
    },
}

IGNORED_DIFF_KEYS = {
    "seed",
    "wandb_name",
    "wandb_group",
}


def ensure_inside_ablation_root(path: Path) -> None:
    resolved = path.resolve()
    root = ABLATION_ROOT.resolve()

    if not resolved.is_relative_to(root):
        raise RuntimeError(
            "检测到输出路径逃离ablation_studies："
            f"{resolved}"
        )


def changed_top_level_keys(
    reference: dict[str, Any],
    variant: dict[str, Any],
) -> set[str]:
    keys = set(reference) | set(variant)

    changed = {
        key
        for key in keys
        if reference.get(key) != variant.get(key)
    }

    return changed - IGNORED_DIFF_KEYS


def apply_variant(
    config: dict[str, Any],
    variant: str,
    seed: int,
    stage_name: str,
) -> dict[str, Any]:
    reference = copy.deepcopy(config)
    modified = copy.deepcopy(config)

    modified.update(
        VARIANTS[variant]
    )

    modified["seed"] = seed
    modified["split_dp"] = RANDOM_SPLIT
    modified["wandb_name"] = (
        f"PANELB_{variant}_{stage_name}_seed{seed}"
    )
    modified["wandb_group"] = (
        "FERA_MS_PANELB_ABLATION"
    )

    changed = changed_top_level_keys(
        reference,
        modified,
    )

    expected = EXPECTED_CHANGED_KEYS[variant]

    if changed != expected:
        raise RuntimeError(
            f"{stage_name}: 检测到非预期配置变化。\n"
            f"actual={sorted(changed)}\n"
            f"expected={sorted(expected)}"
        )

    return modified


def inspect_real_model(
    config: dict[str, Any],
    variant: str,
    stage_name: str,
    audit_path: Path,
) -> None:
    model_wrapper = FragGNNPL(**config)
    model = model_wrapper.model

    actual = {
        "stage":
            stage_name,

        "variant":
            variant,

        "seed":
            int(config["seed"]),

        "split_dp":
            config["split_dp"],

        "frag_gnn_type_config":
            config.get("frag_gnn_type"),

        "frag_gnn_type_model":
            str(model.frag_embedder.gnn_type),

        "fragment_edge_features":
            (
                config.get("frag_params", {})
                .get("pyg_edge_feats")
            ),

        "fragment_edge_feature_dim":
            int(model.frag_edge_feats_size),

        "candidate_refiner_enabled":
            bool(
                config.get(
                    "use_spectrum_candidate_refiner"
                )
            ),

        "candidate_refiner_present":
            (
                getattr(
                    model,
                    "spectrum_candidate_refiner",
                    None,
                )
                is not None
            ),

        "binned_renderer_enabled":
            bool(
                config.get(
                    "use_binned_spectrum_renderer"
                )
            ),

        "mz_expansion_config":
            bool(
                config.get(
                    "use_mz_offset_peak_expansion"
                )
            ),

        "mz_expansion_model":
            bool(
                model.use_mz_offset_peak_expansion
            ),

        "mz_offset_steps_config":
            [
                float(value)
                for value in config.get(
                    "mz_offset_peak_steps",
                    []
                )
            ],

        "mz_offset_steps_model":
            [
                float(value)
                for value in model.mz_offset_peak_steps
            ],

        "rendered_gate_config":
            bool(
                config.get(
                    "use_rendered_peak_drop_gate"
                )
            ),

        "rendered_gate_model":
            bool(
                model.use_rendered_peak_drop_gate
            ),

        "rendered_gate_module":
            (
                None
                if model.rendered_peak_drop_gate is None
                else model.rendered_peak_drop_gate
                    .__class__.__name__
            ),

        "trainable_parameters":
            sum(
                parameter.numel()
                for parameter
                in model_wrapper.parameters()
                if parameter.requires_grad
            ),

        "total_parameters":
            sum(
                parameter.numel()
                for parameter
                in model_wrapper.parameters()
            ),
    }

    assert (
        actual["frag_gnn_type_config"]
        == "GINE"
    )
    assert (
        actual["frag_gnn_type_model"]
        == "GINE"
    )
    assert (
        actual["fragment_edge_features"]
        == ["cut_chem"]
    )
    assert (
        actual["fragment_edge_feature_dim"]
        == 10
    )
    assert (
        actual["candidate_refiner_enabled"]
        is True
    )
    assert (
        actual["candidate_refiner_present"]
        is True
    )
    assert (
        actual["binned_renderer_enabled"]
        is True
    )

    if variant == "without_mz_offset_expansion":
        assert (
            actual["mz_expansion_config"]
            is False
        )
        assert (
            actual["mz_expansion_model"]
            is False
        )
        assert (
            actual["mz_offset_steps_config"]
            == [0.0]
        )
        assert (
            actual["mz_offset_steps_model"]
            == [0.0]
        )
        assert (
            actual["rendered_gate_config"]
            is True
        )
        assert (
            actual["rendered_gate_model"]
            is True
        )
        assert (
            actual["rendered_gate_module"]
            is not None
        )

    elif variant == "without_rendered_peak_gate":
        expected_steps = [
            -0.002,
            -0.001,
            0.0,
            0.001,
            0.002,
        ]

        assert (
            actual["mz_expansion_config"]
            is True
        )
        assert (
            actual["mz_expansion_model"]
            is True
        )
        assert (
            actual["mz_offset_steps_config"]
            == expected_steps
        )
        assert (
            actual["mz_offset_steps_model"]
            == expected_steps
        )
        assert (
            actual["rendered_gate_config"]
            is False
        )
        assert (
            actual["rendered_gate_model"]
            is False
        )
        assert (
            actual["rendered_gate_module"]
            is None
        )

    audit_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit_path.write_text(
        json.dumps(
            actual,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print("LOCKED ABLATION PREFLIGHT PASS")
    print("=" * 100)

    for key, value in actual.items():
        print(f"{key:<42s}: {value}")

    print("audit:", audit_path)
    print("=" * 100)

    del model_wrapper
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def write_final_backbone(
    final_dir: Path,
    checkpoint: Path,
    config: dict[str, Any],
    selected_stage: str,
    selected_score: float,
    initial_training_score: float,
    continuation_score: float,
) -> None:
    final_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    final_checkpoint = (
        final_dir
        / "model.ckpt"
    )

    shutil.copy2(
        checkpoint,
        final_checkpoint,
    )

    base_training.write_yaml(
        final_dir / "config.yml",
        config,
    )

    summary = {
        "selected_stage":
            selected_stage,

        "selected_validation_cosine":
            selected_score,

        "initial_training_validation_cosine":
            initial_training_score,

        "continuation_validation_cosine":
            continuation_score,

        "checkpoint":
            str(final_checkpoint),

        "checkpoint_sha256":
            base_training.sha256_file(
                final_checkpoint
            ),

        "test_used":
            False,
    }

    (
        final_dir
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--variant",
        required=True,
        choices=sorted(VARIANTS),
    )

    parser.add_argument(
        "--seed",
        required=True,
        type=int,
        choices=[42, 43, 44],
    )

    args = parser.parse_args()

    variant = args.variant
    seed = args.seed

    os.environ["MS2_GLOBAL_SEED"] = str(seed)
    os.environ["MS2_SPLIT_DP"] = RANDOM_SPLIT
    os.environ.pop(
        "MS2_RESUME_CKPT",
        None,
    )

    base_training.GLOBAL_SEED = seed
    control_finetuning.GLOBAL_SEED = seed

    run_root = (
        ABLATION_ROOT
        / "runs"
        / variant
        / f"seed_{seed}"
    )

    ensure_inside_ablation_root(run_root)

    if run_root.exists():
        raise FileExistsError(
            "该运行目录已经存在。"
            "为避免覆盖，本程序拒绝启动：\n"
            f"{run_root}"
        )

    structural_backbone_root = run_root / "structural_backbone"
    initial_training_dir = (
        structural_backbone_root
        / "structural_backbone_initial"
    )
    continuation_dir = (
        structural_backbone_root
        / "retained_control_continuation"
    )
    final_dir = (
        structural_backbone_root
        / "final"
    )
    audit_root = run_root / "audit"

    for path in (
        initial_training_dir,
        continuation_dir,
        final_dir,
        audit_root,
    ):
        ensure_inside_ablation_root(path)

    print("=" * 100)
    print("FERA-MS PANEL-B STRICT ABLATION")
    print("=" * 100)
    print("variant :", variant)
    print("seed    :", seed)
    print("split   :", RANDOM_SPLIT)
    print("output  :", run_root)
    print("test used during training: False")
    print("=" * 100)

    initial_training_reference, continuation_reference = (
        base_training.build_stage_configs()
    )

    initial_training_config = apply_variant(
        initial_training_reference,
        variant,
        seed,
        "structural_backbone_initial_training",
    )

    continuation_config = apply_variant(
        continuation_reference,
        variant,
        seed,
        "structural_backbone_continuation",
    )

    inspect_real_model(
        initial_training_config,
        variant,
        "structural_backbone_initial_training",
        audit_root
        / "structural_backbone_initial_training_preflight.json",
    )

    inspect_real_model(
        continuation_config,
        variant,
        "structural_backbone_continuation",
        audit_root
        / "structural_backbone_continuation_preflight.json",
    )

    initial_training_score, initial_training_checkpoint = (
        base_training.train_stage(
            name=(
                "PANELB_ABLATION_"
                "STRUCTURAL_BACKBONE_INITIAL_TRAINING"
            ),
            run_dir=initial_training_dir,
            config=initial_training_config,
            initialization_checkpoint=None,
            patience=1000,
            min_delta=0.0,
        )
    )

    continuation_score, continuation_checkpoint = (
        base_training.train_stage(
            name=(
                "PANELB_ABLATION_"
                "STRUCTURAL_BACKBONE_CONTINUATION"
            ),
            run_dir=continuation_dir,
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

    write_final_backbone(
        final_dir=final_dir,
        checkpoint=selected_checkpoint,
        config=selected_config,
        selected_stage=selected_stage,
        selected_score=selected_score,
        initial_training_score=initial_training_score,
        continuation_score=continuation_score,
    )

    inspect_real_model(
        selected_config,
        variant,
        "structural_backbone_selected_final",
        audit_root
        / "structural_backbone_selected_final_preflight.json",
    )

    control_finetuning.BASE_CONFIG = (
        final_dir
        / "config.yml"
    )
    control_finetuning.BASE_CHECKPOINT = (
        final_dir
        / "model.ckpt"
    )
    control_finetuning.OUTPUT_ROOT = (
        run_root
        / "global_ace_control"
    )
    control_finetuning.STRUCTURAL_BACKBONE_BASELINE = (
        selected_score
    )
    control_finetuning.VARIANTS = {
        "control": {}
    }

    ensure_inside_ablation_root(
        control_finetuning.OUTPUT_ROOT
    )

    global_ace_control_config = (
        control_finetuning.build_config(
            "control"
        )
    )

    inspect_real_model(
        global_ace_control_config,
        variant,
        "global_ace_control_control",
        audit_root
        / "global_ace_control_control_preflight.json",
    )

    global_ace_control_summary = (
        control_finetuning.train_variant(
            "control"
        )
    )

    final_summary = {
        "variant":
            variant,

        "seed":
            seed,

        "split":
            RANDOM_SPLIT,

        "structural_backbone_selected_stage":
            selected_stage,

        "structural_backbone_validation_cosine":
            selected_score,

        "global_ace_control_validation_cosine":
            global_ace_control_summary[
                "best_val_cosine"
            ],

        "global_ace_control_delta_vs_backbone":
            global_ace_control_summary[
                "delta_vs_backbone"
            ],

        "global_ace_control_checkpoint":
            global_ace_control_summary[
                "checkpoint"
            ],

        "test_used_during_training":
            False,

        "output_root":
            str(run_root),
    }

    (
        run_root
        / "run_summary.json"
    ).write_text(
        json.dumps(
            final_summary,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print("PANEL-B ABLATION TRAINING COMPLETE")
    print("=" * 100)
    print(
        json.dumps(
            final_summary,
            indent=2,
            ensure_ascii=False,
        )
    )
    print("=" * 100)


if __name__ == "__main__":
    main()
