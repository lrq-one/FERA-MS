#!/usr/bin/env python3
"""Materialize and merge the locked training configuration bundle."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"YAML top level must be a mapping: {path}")
    return data


def deep_merge(
    base: dict[str, Any],
    patch: dict[str, Any],
) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in patch.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def materialize_training_config(
    bundle_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    bundle_path = Path(bundle_path)
    output_dir = Path(output_dir)
    bundle = load_yaml(bundle_path)

    names = {
        "template": "template.yml",
        "base_stage": "base_stage.yml",
        "continuation_stage": "continuation_stage.yml",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}

    for key, filename in names.items():
        value = bundle.get(key)
        if not isinstance(value, dict):
            raise KeyError(f"Configuration bundle is missing mapping: {key}")

        path = output_dir / filename
        path.write_text(
            yaml.safe_dump(
                value,
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
        result[key] = path

    return result


def prepare_effective_config(
    template_cfg: dict[str, Any],
    base_model_custom: dict[str, Any],
    retained_control_custom: dict[str, Any],
    epochs: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    effective_base_config = deep_merge(
        template_cfg,
        base_model_custom,
    )
    effective_retained_control_config = deep_merge(
        template_cfg,
        retained_control_custom,
    )

    transfer = copy.deepcopy(effective_base_config)
    copied: dict[str, Any] = {}

    exact_training_keys = {
        "optimizer",
        "lr",
        "weight_decay",
        "lr_schedule",
        "lr_decay_rate",
        "lr_warmup_steps",
        "lr_decay_steps",
        "gradient_clip_val",
        "gradient_clip_algorithm",
        "accumulate_grad_batches",
        "precision",
        "use_tensor_float32",
        "deterministic",
        "automatic_optimization",
    }

    for key in sorted(exact_training_keys):
        if key in effective_retained_control_config:
            transfer[key] = copy.deepcopy(
                effective_retained_control_config[key]
            )
            copied[key] = copy.deepcopy(
                effective_retained_control_config[key]
            )

    for key, value in effective_retained_control_config.items():
        key_lower = key.lower()
        if (
            "scheduler" in key_lower
            or key_lower.startswith("lr_")
            or key_lower.endswith("_lr")
            or key_lower.startswith("optimizer_")
        ):
            transfer[key] = copy.deepcopy(value)
            copied[key] = copy.deepcopy(value)

    transfer["min_epochs"] = 1
    transfer["max_epochs"] = int(epochs)

    for scope_key in (
        "spectrum_refiner_train_scope",
        "train_scope",
    ):
        if scope_key in transfer:
            transfer[scope_key] = "all"
            copied[scope_key] = "all"

    if "use_support_oracle_support_oracle_reweight_loss" in transfer:
        transfer["use_support_oracle_support_oracle_reweight_loss"] = False
        copied["use_support_oracle_support_oracle_reweight_loss"] = False

    if "support_oracle_support_oracle_weight" in transfer:
        transfer["support_oracle_support_oracle_weight"] = 0.0
        copied["support_oracle_support_oracle_weight"] = 0.0

    transfer.update(
        {
            "eval_test_split": True,
            "disable_checkpoints": False,
            "delete_checkpoints": False,
            "upload_checkpoints": False,
            "checkpoint_save_last": True,
            "checkpoint_metric_mode": "max",
        }
    )

    checkpoint_metric = transfer.get(
        "checkpoint_metric",
        "val_cos_sim_0.01_epoch/mean",
    )
    if "cos_sim" not in str(checkpoint_metric):
        checkpoint_metric = "val_cos_sim_0.01_epoch/mean"
    transfer["checkpoint_metric"] = checkpoint_metric

    copied.update(
        {
            "min_epochs": transfer["min_epochs"],
            "max_epochs": transfer["max_epochs"],
            "eval_test_split": True,
            "checkpoint_metric": checkpoint_metric,
            "checkpoint_metric_mode": "max",
            "checkpoint_save_last": True,
        }
    )

    assert not bool(
        transfer.get(
            "use_support_oracle_support_oracle_reweight_loss",
            False,
        )
    )
    assert float(
        transfer.get("support_oracle_support_oracle_weight", 0.0)
    ) == 0.0

    return transfer, copied
