#!/usr/bin/env python3
"""Read the consolidated, immutable global-ACE ablation configuration."""

from __future__ import annotations

from pathlib import Path
from copy import deepcopy

import yaml


def _merge(base: dict, overrides: dict) -> dict:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


def load_locked_config(ablation_root: Path, seed: int, stage: str) -> dict:
    store_path = ablation_root / "configs" / "locked.yml"
    store = yaml.safe_load(store_path.read_text(encoding="utf-8"))
    if int(seed) not in store["seeds"]:
        raise KeyError(f"Unsupported locked seed: {seed}")
    config = deepcopy(store["common"])
    config["seed"] = int(seed)
    for stage_entry in store["stages"]:
        _merge(config, stage_entry.get("overrides", {}))
        if stage_entry["name"] == stage:
            return config
    raise KeyError(f"Missing locked functional stage: {stage}")


def write_effective_config(
    ablation_root: Path,
    seed: int,
    stage: str,
    destination: Path,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(
            load_locked_config(ablation_root, seed, stage),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return destination
