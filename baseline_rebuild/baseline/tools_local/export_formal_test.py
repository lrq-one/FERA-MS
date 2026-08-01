#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml


def sha256_file(path: Path) -> str:
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


def plain(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(
                value.detach().cpu()
            )
        return (
            value.detach()
            .cpu()
            .tolist()
        )

    if isinstance(value, dict):
        return {
            str(key): plain(child)
            for key, child in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            plain(child)
            for child in value
        ]

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ) or value is None:
        return value

    return str(value)


def normalize_id(value: Any) -> str:
    text = str(value)

    if text.endswith(".0"):
        text = text[:-2]

    return text


def move_batch(
    model,
    batch,
    device,
):
    try:
        return model.transfer_batch_to_device(
            batch,
            device,
            0,
        )
    except Exception:
        pass

    if isinstance(batch, torch.Tensor):
        return batch.to(device)

    if isinstance(batch, dict):
        return {
            key: move_batch(
                model,
                value,
                device,
            )
            for key, value in batch.items()
        }

    if isinstance(batch, list):
        return [
            move_batch(
                model,
                value,
                device,
            )
            for value in batch
        ]

    if isinstance(batch, tuple):
        return tuple(
            move_batch(
                model,
                value,
                device,
            )
            for value in batch
        )

    if hasattr(batch, "to"):
        try:
            return batch.to(device)
        except Exception:
            return batch

    return batch


def find_metric_key(
    keys,
    metric: str,
) -> str:
    keys = list(keys)

    preferred = {
        "cbin": [
            "cos_sim_0.01",
            "cos_sim_0.010",
            "cosine_0.01",
            "cos_sim",
        ],
        "jss": [
            "jss_0.01",
            "jss_0.010",
            "jss",
        ],
    }[metric]

    for key in preferred:
        if key in keys:
            return key

    if metric == "cbin":
        candidates = [
            key
            for key in keys
            if "cos_sim" in key
            and "sqrt" not in key
            and "_np" not in key
            and "hun" not in key
        ]
    else:
        candidates = [
            key
            for key in keys
            if key.startswith("jss")
            and "sqrt" not in key
            and "_np" not in key
        ]

    candidates_001 = [
        key
        for key in candidates
        if "0.01" in key
    ]

    if len(candidates_001) == 1:
        return candidates_001[0]

    if len(candidates) == 1:
        return candidates[0]

    raise RuntimeError(
        f"Cannot uniquely resolve {metric}. "
        f"Available keys: {sorted(keys)}"
    )


def get_git_commit(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            [
                "git",
                "-C",
                str(repo),
                "rev-parse",
                "HEAD",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def callback_metadata(
    checkpoint: dict,
) -> dict:
    output = {}

    callbacks = checkpoint.get(
        "callbacks",
        {},
    )

    if not isinstance(callbacks, dict):
        return output

    for callback_name, state in callbacks.items():
        if not isinstance(state, dict):
            continue

        if (
            "best_model_score" in state
            or "best_model_path" in state
        ):
            output = {
                "callback_name":
                    str(callback_name),
                "monitor":
                    plain(
                        state.get("monitor")
                    ),
                "mode":
                    plain(
                        state.get("mode")
                    ),
                "best_model_score":
                    plain(
                        state.get(
                            "best_model_score"
                        )
                    ),
                "best_model_path":
                    plain(
                        state.get(
                            "best_model_path"
                        )
                    ),
                "best_k_models":
                    plain(
                        state.get(
                            "best_k_models"
                        )
                    ),
            }
            break

    return output


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo",
        required=True,
    )

    parser.add_argument(
        "--template",
        required=True,
    )

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--out-dir",
        required=True,
    )

    parser.add_argument(
        "--model-name",
        required=True,
    )

    parser.add_argument(
        "--split-name",
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
    )

    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    config_path = Path(
        args.config
    ).resolve()
    checkpoint_path = Path(
        args.checkpoint
    ).resolve()
    output_dir = Path(
        args.out_dir
    ).resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    os.chdir(repo)

    sys.path.insert(
        0,
        str(repo / "src"),
    )
    sys.path.insert(
        0,
        str(repo),
    )

    from fragnnet.runner import (
        init_dataloader,
        init_dataset,
        load_config,
    )
    from fragnnet.pl_model import (
        FragGNNPL,
        NeimsPL,
    )
    from fragnnet.massformer.pl_model import (
        MassFormerPL,
    )
    from fragnnet.iceberg.pl_model import (
        IcebergIntenPL,
    )

    config = load_config(
        args.template,
        str(config_path),
    )

    config["eval_test_split"] = True
    config["auxiliary_scores"] = [
        "cos_sim",
        "jss",
    ]
    config["eval_mz_bin_res"] = [
        0.01,
    ]
    config["sparse_cosine_similarity"] = True
    config["compile"] = False

    model_classes = {
        "neims": NeimsPL,
        "massformer": MassFormerPL,
        "frag_gnn": FragGNNPL,
        "iceberg_inten": IcebergIntenPL,
    }

    model_type = config[
        "model_type"
    ]

    if model_type not in model_classes:
        raise RuntimeError(
            f"Unsupported model_type: "
            f"{model_type}"
        )

    model_class = model_classes[
        model_type
    ]

    print(
        "MODEL_CLASS",
        model_class.__name__,
    )

    print(
        "CHECKPOINT",
        checkpoint_path,
    )

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Checkpoint type: "
            f"{type(checkpoint)}"
        )

    state_dict = checkpoint.get(
        "state_dict",
        checkpoint,
    )

    model = model_class(
        **config
    )

    load_result = model.load_state_dict(
        state_dict,
        strict=True,
    )

    print(
        "STRICT_CHECKPOINT_LOAD_OK",
        load_result,
    )

    test_dataset, = init_dataset(
        config,
        splits=("test",),
    )

    test_loader = init_dataloader(
        test_dataset,
        config,
    )

    test_metadata = (
        test_dataset
        .spec_df
        .reset_index(drop=True)
        .copy()
    )

    expected_rows = (
        3931
        if args.split_name == "random"
        else 3960
    )

    expected_molecules = (
        456
        if args.split_name == "random"
        else 450
    )

    if len(test_metadata) != expected_rows:
        raise RuntimeError(
            f"Unexpected test rows: "
            f"{len(test_metadata)} "
            f"!= {expected_rows}"
        )

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = model.to(device)
    model.eval()

    cbin_values = []
    jss_values = []

    resolved_keys = None

    with torch.no_grad():
        for batch_index, batch in enumerate(
            test_loader
        ):
            batch = move_batch(
                model,
                batch,
                device,
            )

            results = model._common_step(
                batch,
                split="test",
                log=False,
            )

            if resolved_keys is None:
                cbin_key = find_metric_key(
                    results.keys(),
                    "cbin",
                )

                jss_key = find_metric_key(
                    results.keys(),
                    "jss",
                )

                resolved_keys = {
                    "cbin": cbin_key,
                    "jss": jss_key,
                }

                print(
                    "RESOLVED_METRIC_KEYS",
                    resolved_keys,
                )

            batch_cbin = (
                results[
                    resolved_keys["cbin"]
                ]
                .detach()
                .float()
                .cpu()
                .reshape(-1)
                .numpy()
            )

            batch_jss = (
                results[
                    resolved_keys["jss"]
                ]
                .detach()
                .float()
                .cpu()
                .reshape(-1)
                .numpy()
            )

            if len(batch_cbin) != len(
                batch_jss
            ):
                raise RuntimeError(
                    "CBIN/JSS batch length mismatch"
                )

            cbin_values.extend(
                batch_cbin.tolist()
            )
            jss_values.extend(
                batch_jss.tolist()
            )

            if (
                batch_index == 0
                or (batch_index + 1) % 50 == 0
            ):
                print(
                    "INFERENCE_PROGRESS",
                    batch_index + 1,
                    len(cbin_values),
                )

    if len(cbin_values) != len(
        test_metadata
    ):
        raise RuntimeError(
            f"Inference row mismatch: "
            f"{len(cbin_values)} "
            f"!= {len(test_metadata)}"
        )

    detail = pd.DataFrame()

    detail["split"] = (
        args.split_name
    )
    detail["model"] = (
        args.model_name
    )
    detail["seed"] = int(
        args.seed
    )

    for column in [
        "spec_id",
        "mol_id",
        "group_id",
    ]:
        if column in test_metadata.columns:
            detail[column] = (
                test_metadata[column]
                .map(normalize_id)
            )
        else:
            detail[column] = ""

    ce_column = None

    for candidate in [
        "ace",
        "nce",
        "collision_energy",
    ]:
        if candidate in test_metadata.columns:
            ce_column = candidate
            break

    if ce_column is None:
        detail["ace"] = np.nan
    else:
        detail["ace"] = pd.to_numeric(
            test_metadata[ce_column],
            errors="coerce",
        )

    detail["cbin"] = np.asarray(
        cbin_values,
        dtype=np.float64,
    )

    detail["jss"] = np.asarray(
        jss_values,
        dtype=np.float64,
    )

    if detail["spec_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate spec_id in test export"
        )

    if detail["mol_id"].nunique() != (
        expected_molecules
    ):
        raise RuntimeError(
            f"Unexpected molecule count: "
            f"{detail['mol_id'].nunique()} "
            f"!= {expected_molecules}"
        )

    if not np.isfinite(
        detail["cbin"]
    ).all():
        raise RuntimeError(
            "Non-finite CBIN values"
        )

    if not np.isfinite(
        detail["jss"]
    ).all():
        raise RuntimeError(
            "Non-finite JSS values"
        )

    per_molecule = (
        detail
        .groupby(
            "mol_id",
            as_index=False,
        )
        .agg(
            spectrum_count=(
                "spec_id",
                "size",
            ),
            mean_ace=(
                "ace",
                "mean",
            ),
            cbin=(
                "cbin",
                "mean",
            ),
            jss=(
                "jss",
                "mean",
            ),
        )
    )

    metrics = {
        "model":
            args.model_name,
        "model_type":
            model_type,
        "split":
            args.split_name,
        "seed":
            int(args.seed),
        "test_spectra":
            int(len(detail)),
        "test_molecules":
            int(
                detail[
                    "mol_id"
                ].nunique()
            ),
        "micro_cbin":
            float(
                detail["cbin"].mean()
            ),
        "macro_cbin":
            float(
                per_molecule[
                    "cbin"
                ].mean()
            ),
        "micro_jss":
            float(
                detail["jss"].mean()
            ),
        "macro_jss":
            float(
                per_molecule[
                    "jss"
                ].mean()
            ),
        "metric_keys":
            resolved_keys,
    }

    detail.to_csv(
        output_dir
        / "test_per_spectrum.csv.gz",
        index=False,
        compression="gzip",
    )

    per_molecule.to_csv(
        output_dir
        / "test_per_molecule.csv",
        index=False,
    )

    (
        output_dir
        / "metrics.json"
    ).write_text(
        json.dumps(
            metrics,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    split_dir = (
        repo
        / config["split_dp"]
    )

    split_hashes = {}

    for filename in [
        "train_ids.csv",
        "val_ids.csv",
        "test_ids.csv",
    ]:
        path = split_dir / filename

        if not path.is_file():
            raise FileNotFoundError(path)

        split_hashes[
            filename
        ] = sha256_file(path)

    checkpoint_meta = (
        callback_metadata(checkpoint)
    )

    manifest = {
        "model":
            args.model_name,
        "model_type":
            model_type,
        "split":
            args.split_name,
        "seed":
            int(args.seed),
        "checkpoint":
            str(checkpoint_path),
        "checkpoint_sha256":
            sha256_file(
                checkpoint_path
            ),
        "checkpoint_bytes":
            int(
                checkpoint_path
                .stat()
                .st_size
            ),
        "checkpoint_epoch":
            plain(
                checkpoint.get(
                    "epoch"
                )
            ),
        "checkpoint_global_step":
            plain(
                checkpoint.get(
                    "global_step"
                )
            ),
        "checkpoint_callback":
            checkpoint_meta,
        "strict_load":
            True,
        "config":
            str(config_path),
        "config_sha256":
            sha256_file(
                config_path
            ),
        "template":
            str(
                Path(
                    args.template
                ).resolve()
            ),
        "split_dir":
            str(
                split_dir.resolve()
            ),
        "split_sha256":
            split_hashes,
        "code_commit":
            get_git_commit(repo),
        "python":
            sys.version,
        "torch":
            torch.__version__,
        "torch_cuda":
            torch.version.cuda,
        "gpu":
            (
                torch.cuda
                .get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
        "test_metrics":
            metrics,
    }

    (
        output_dir
        / "model_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print("FORMAL_TEST_EXPORT_COMPLETE")
    print(
        json.dumps(
            metrics,
            indent=2,
            ensure_ascii=False,
        )
    )
    print("=" * 100)


if __name__ == "__main__":
    main()
