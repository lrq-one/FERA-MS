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
import yaml


ROOT = Path(__file__).resolve().parents[3]
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

    # 所有实验共同保持的结构。
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


def write_final_v2a(
    final_dir: Path,
    checkpoint: Path,
    config: dict[str, Any],
    selected_stage: str,
    selected_score: float,
    stage1_score: float,
    stage2_score: float,
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

        "stage1_validation_cosine":
            stage1_score,

        "stage2_validation_cosine":
            stage2_score,

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

    v2a_root = run_root / "v2a"
    stage1_dir = (
        v2a_root
        / "stage1_v1_40ep"
    )
    stage2_dir = (
        v2a_root
        / "stage2_r119_10ep"
    )
    final_dir = (
        v2a_root
        / "final"
    )
    audit_root = run_root / "audit"

    for path in (
        stage1_dir,
        stage2_dir,
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

    # 构建与正式V2A完全一致的两个阶段。
    stage1_reference, stage2_reference = (
        base_training.build_stage_configs()
    )

    stage1_config = apply_variant(
        stage1_reference,
        variant,
        seed,
        "v2a_stage1",
    )

    stage2_config = apply_variant(
        stage2_reference,
        variant,
        seed,
        "v2a_stage2",
    )

    inspect_real_model(
        stage1_config,
        variant,
        "v2a_stage1",
        audit_root
        / "v2a_stage1_preflight.json",
    )

    inspect_real_model(
        stage2_config,
        variant,
        "v2a_stage2",
        audit_root
        / "v2a_stage2_preflight.json",
    )

    stage1_score, stage1_checkpoint = (
        base_training.train_stage(
            name=(
                "PANELB_ABLATION_"
                "V2A_STAGE1"
            ),
            run_dir=stage1_dir,
            config=stage1_config,
            initialization_checkpoint=None,
            patience=1000,
            min_delta=0.0,
        )
    )

    stage2_score, stage2_checkpoint = (
        base_training.train_stage(
            name=(
                "PANELB_ABLATION_"
                "V2A_STAGE2"
            ),
            run_dir=stage2_dir,
            config=stage2_config,
            initialization_checkpoint=(
                stage1_checkpoint
            ),
            patience=4,
            min_delta=1.0e-4,
        )
    )

    if stage2_score >= stage1_score:
        selected_stage = "stage2_r119"
        selected_score = stage2_score
        selected_checkpoint = (
            stage2_checkpoint
        )
        selected_config = stage2_config
    else:
        selected_stage = "stage1_v1"
        selected_score = stage1_score
        selected_checkpoint = (
            stage1_checkpoint
        )
        selected_config = stage1_config

    write_final_v2a(
        final_dir=final_dir,
        checkpoint=selected_checkpoint,
        config=selected_config,
        selected_stage=selected_stage,
        selected_score=selected_score,
        stage1_score=stage1_score,
        stage2_score=stage2_score,
    )

    # 再次从最终保存配置实例化检查。
    inspect_real_model(
        selected_config,
        variant,
        "v2a_selected_final",
        audit_root
        / "v2a_selected_final_preflight.json",
    )

    # 将原V2C代码的输入和输出全部改到当前消融目录。
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
        / "v2c"
    )
    control_finetuning.V2A_BASELINE = (
        selected_score
    )
    control_finetuning.VARIANTS = {
        "control": {}
    }

    ensure_inside_ablation_root(
        control_finetuning.OUTPUT_ROOT
    )

    v2c_config = (
        control_finetuning.build_config(
            "control"
        )
    )

    inspect_real_model(
        v2c_config,
        variant,
        "v2c_control",
        audit_root
        / "v2c_control_preflight.json",
    )

    v2c_summary = (
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

        "v2a_selected_stage":
            selected_stage,

        "v2a_validation_cosine":
            selected_score,

        "v2c_validation_cosine":
            v2c_summary[
                "best_val_cosine"
            ],

        "v2c_delta_vs_v2a":
            v2c_summary[
                "delta_vs_v2a"
            ],

        "v2c_checkpoint":
            v2c_summary[
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
