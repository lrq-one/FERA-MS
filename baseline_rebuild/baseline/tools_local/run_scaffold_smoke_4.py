#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


BASE = Path(os.environ.get("FERA_MS_BASELINE_ROOT", Path(__file__).resolve().parents[1])).resolve()

SOURCE = Path(
    os.environ.get(
        "FERA_MS_BASELINE_SOURCE",
        BASE / "source" / "fragnnet",
    )
).resolve()

RESULT_ROOT = (
    Path(os.environ.get("FERA_MS_BASELINE_OUTPUT_DIR", BASE / "results_local"))
    / "scaffold_smoke"
)

SPLIT = (
    "data/split/"
    "nist20_qtof_cid_safe19659_"
    "scaffold60_20_20_seed42"
)

OLD_ROOT = os.environ.get("FERA_MS_LEGACY_BASELINE_ROOT", str(BASE))

JOBS = [
    {
        "model": "neims",
        "repo": SOURCE,
        "source_config":
            BASE / "neims/configs/scaffold/seed_42.yml",
    },
    {
        "model": "massformer",
        "repo": SOURCE,
        "source_config":
            BASE / "massformer/configs/scaffold/seed_42.yml",
    },
    {
        "model": "fragnnet_d3",
        "repo": SOURCE,
        "source_config":
            BASE / "fragnnet_d3/configs/scaffold/seed_42.yml",
    },
    {
        "model": "iceberg",
        "repo": SOURCE,
        "source_config":
            BASE / "iceberg/configs/scaffold/seed_42.yml",
    },
]


def rewrite_paths(value):
    if isinstance(value, dict):
        return {
            key: rewrite_paths(child)
            for key, child in value.items()
        }

    if isinstance(value, list):
        return [
            rewrite_paths(child)
            for child in value
        ]

    if isinstance(value, str):
        return value.replace(
            OLD_ROOT,
            str(BASE),
        )

    return value


def make_config(
    source: Path,
    target: Path,
):
    config = yaml.safe_load(
        source.read_text(
            encoding="utf-8"
        )
    )

    config = rewrite_paths(config)

    config["seed"] = 42
    config["split_dp"] = SPLIT
    config["min_epochs"] = 1
    config["max_epochs"] = 1
    config["eval_test_split"] = True
    config["auxiliary_scores"] = [
        "cos_sim",
        "jss",
    ]
    config["eval_mz_bin_res"] = [
        0.01,
    ]

    target.write_text(
        yaml.safe_dump(
            config,
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def clean_repo(repo: Path):
    shutil.rmtree(
        repo / "tmp_ckpt",
        ignore_errors=True,
    )

    shutil.rmtree(
        repo / "tmp_profile",
        ignore_errors=True,
    )


def run_job(job):
    model = job["model"]
    repo = job["repo"]

    result_dir = (
        RESULT_ROOT
        / model
        / "seed42"
    )

    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    status_path = (
        result_dir
        / "status.txt"
    )

    if (
        status_path.is_file()
        and status_path.read_text().strip()
        == "SUCCESS"
    ):
        print(
            "SKIP SUCCESS:",
            model,
        )
        return

    config_path = (
        result_dir
        / "config.yml"
    )

    make_config(
        job["source_config"],
        config_path,
    )

    clean_repo(repo)

    command = [
        sys.executable,
        "scripts/run_pl_model_fit.py",
        "-t",
        "config/template.yml",
        "-c",
        str(config_path),
        "-w",
        "disabled",
        "-j",
        f"local_scaffold_{model}_seed42_smoke",
    ]

    environment = os.environ.copy()

    # LOCAL_REPO_PYTHONPATH_PATCH_V1
    # Each packaged repository has its own
    # fragnnet implementation. Do not install
    # both under the same environment; expose
    # the correct repository only for this job.
    python_paths = [
        str((repo / "src").resolve()),
        str(repo.resolve()),
    ]

    inherited_pythonpath = (
        environment
        .get("PYTHONPATH", "")
        .strip()
    )

    if inherited_pythonpath:
        python_paths.append(
            inherited_pythonpath
        )

    environment["PYTHONPATH"] = (
        os.pathsep.join(
            python_paths
        )
    )

    print(
        "job PYTHONPATH:",
        environment["PYTHONPATH"],
    )


    environment[
        "DGLBACKEND"
    ] = "pytorch"

    environment[
        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"
    ] = "1"

    environment.pop(
        "TORCH_FORCE_WEIGHTS_ONLY_LOAD",
        None,
    )

    log_path = (
        result_dir
        / "train.log"
    )

    print()
    print("=" * 100)
    print(
        "SMOKE:",
        model,
    )
    print(
        "repo:",
        repo,
    )
    print(
        "config:",
        config_path,
    )
    print("=" * 100)

    with log_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        process = subprocess.run(
            command,
            cwd=repo,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )

    if process.returncode != 0:
        status_path.write_text(
            f"FAILED:{process.returncode}\n"
        )

        tail = log_path.read_text(
            encoding="utf-8",
            errors="replace",
        )[-15000:]

        print(tail)

        raise RuntimeError(
            f"Smoke failed: {model}; "
            f"see {log_path}"
        )

    # A one-epoch smoke run only verifies that
    # training/validation finishes successfully.
    # Some model configurations do not emit a
    # checkpoint during a one-epoch smoke.
    checkpoint_root = repo / "tmp_ckpt"

    checkpoints = []

    if checkpoint_root.exists():
        checkpoints = sorted(
            checkpoint_root.rglob("*.ckpt"),
            key=lambda item: (
                item.stat().st_mtime,
                str(item),
            ),
        )

    checkpoint = (
        checkpoints[-1]
        if checkpoints
        else None
    )

    smoke_record = {
        "model": model,
        "seed": 42,
        "split": "scaffold",
        "epochs": 1,
        "training_process_returncode": 0,
        "checkpoint_produced":
            checkpoint is not None,
        "checkpoint_name":
            (
                checkpoint.name
                if checkpoint is not None
                else None
            ),
        "checkpoint_bytes":
            (
                checkpoint.stat().st_size
                if checkpoint is not None
                else 0
            ),
        "completed_at":
            datetime.now().isoformat(
                timespec="seconds"
            ),
    }

    (
        result_dir
        / "smoke_record.json"
    ).write_text(
        json.dumps(
            smoke_record,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    status_path.write_text(
        "SUCCESS\n",
        encoding="utf-8",
    )

    if checkpoint is None:
        print(
            "SUCCESS:",
            model,
            "training completed; "
            "no smoke checkpoint emitted",
        )
    else:
        print(
            "SUCCESS:",
            model,
            checkpoint.name,
            checkpoint.stat().st_size,
        )

    # Smoke checkpoints are intentionally discarded.
    # Formal runs will force and verify best.ckpt.
    clean_repo(repo)


def main():
    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    for job in JOBS:
        run_job(job)

    print()
    print("=" * 100)
    print("ALL_4_MODEL_SCAFFOLD_SMOKE_SUCCESS")
    print("=" * 100)


if __name__ == "__main__":
    main()
