#!/usr/bin/env python3
"""Non-training preflight for the bundled manuscript baseline sources."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

import yaml


BASE = Path(
    os.environ.get(
        "FERA_MS_BASELINE_ROOT",
        Path(__file__).resolve().parents[1],
    )
).resolve()
FRAGNNET_SOURCE = Path(
    os.environ.get(
        "FERA_MS_BASELINE_SOURCE",
        BASE / "source" / "fragnnet",
    )
).resolve()
FIORA_SOURCE = (BASE / "source" / "fiora").resolve()
DATA_ROOT = Path(
    os.environ.get(
        "FERA_MS_BASELINE_DATA_DIR",
        BASE.parents[1] / "data",
    )
).resolve()

MODEL_INFO = {
    "neims": ("neims", "fragnnet.pl_model"),
    "massformer": ("massformer", "fragnnet.massformer.pl_model"),
    "fragnnet_d3": ("frag_gnn", "fragnnet.pl_model"),
    "iceberg": ("iceberg_inten", "fragnnet.iceberg.pl_model"),
    "graff_ms": ("graff", "fragnnet.graff.pl_model"),
}
DATA_KEYS = ("spec_fp", "mol_fp", "split_dp", "frag_dp", "magma_dp", "ann_fp")


def resolve_data_path(value: str) -> Path:
    path = Path(value)
    if path.parts and path.parts[0] == "data":
        return DATA_ROOT.joinpath(*path.parts[1:])
    return path


def check_locked_configs(model: str, require_data: bool) -> None:
    expected_model_type, _ = MODEL_INFO[model]
    for split in ("random", "scaffold"):
        split_token = {
            "random": "qcv1_trainonly",
            "scaffold": "scaffold60_20_20_seed42",
        }[split]
        for seed in (42, 43, 44):
            path = BASE / model / "configs" / split / f"seed_{seed}.yml"
            if not path.is_file():
                raise FileNotFoundError(path)
            config = yaml.safe_load(path.read_text(encoding="utf-8"))
            if int(config["seed"]) != seed:
                raise RuntimeError(f"Seed mismatch in {path}")
            if config["model_type"] != expected_model_type:
                raise RuntimeError(f"Model type mismatch in {path}")
            if split_token not in str(config["split_dp"]):
                raise RuntimeError(f"Split mismatch in {path}")
            if require_data:
                for key in DATA_KEYS:
                    value = config.get(key)
                    if isinstance(value, str):
                        resolved = resolve_data_path(value)
                        if not resolved.exists():
                            raise FileNotFoundError(f"{key}: {resolved}")
            print("CONFIG_OK", model, split, seed, path)


def import_model(model: str) -> None:
    _, module_name = MODEL_INFO[model]
    source_path = str(FRAGNNET_SOURCE / "src")
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    importlib.import_module(module_name)
    print("IMPORT_OK", model, module_name)


def check_fiora(do_import: bool) -> None:
    required = (
        FIORA_SOURCE / "LICENSE",
        FIORA_SOURCE / "pyproject.toml",
        FIORA_SOURCE / "fiora" / "cli" / "predict.py",
        BASE / "fiora" / "run_final.sh",
        BASE / "fiora" / "eval_fiora_against_library_csv.py",
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if do_import:
        source_path = str(FIORA_SOURCE)
        if source_path not in sys.path:
            sys.path.insert(0, source_path)
        importlib.import_module("fiora.cli.predict")
        print("IMPORT_OK", "fiora", "fiora.cli.predict")
    print("SOURCE_OK", "fiora", FIORA_SOURCE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=[*MODEL_INFO, "fiora", "all"],
        default="all",
    )
    parser.add_argument("--require-data", action="store_true")
    parser.add_argument("--skip-imports", action="store_true")
    args = parser.parse_args()

    required_source = (
        FRAGNNET_SOURCE / "LICENSE",
        FRAGNNET_SOURCE / "config" / "template.yml",
        FRAGNNET_SOURCE / "scripts" / "run_pl_model_fit.py",
        FRAGNNET_SOURCE / "src" / "fragnnet" / "model.py",
        FRAGNNET_SOURCE / "src" / "fragnnet" / "iceberg" / "model.py",
    )
    for path in required_source:
        if not path.is_file():
            raise FileNotFoundError(path)

    selected = list(MODEL_INFO) if args.model == "all" else [args.model]
    for model in selected:
        if model == "fiora":
            check_fiora(not args.skip_imports)
            continue
        check_locked_configs(model, args.require_data)
        if not args.skip_imports:
            import_model(model)
        print("SOURCE_OK", model, FRAGNNET_SOURCE)

    if args.model == "all":
        check_fiora(not args.skip_imports)

    print("BASELINE_PREFLIGHT_OK")


if __name__ == "__main__":
    main()
