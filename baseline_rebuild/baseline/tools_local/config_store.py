#!/usr/bin/env python3
"""Access consolidated baseline configurations without generated seed files."""

from __future__ import annotations

from pathlib import Path

import yaml


def locked_config_path(base: Path, model: str) -> Path:
    directory = {
        "fragnnet_d3": "fragnnet_depth_three",
    }.get(model, model)
    return base / directory / "configs" / "locked.yml"


def load_locked_config(base: Path, model: str, split: str, seed: int) -> dict:
    path = locked_config_path(base, model)
    store = yaml.safe_load(path.read_text(encoding="utf-8"))
    try:
        config = store["configurations"][split][int(seed)]
    except KeyError as exc:
        raise KeyError(
            f"Missing locked baseline config: model={model}, split={split}, seed={seed}"
        ) from exc
    return dict(config)
