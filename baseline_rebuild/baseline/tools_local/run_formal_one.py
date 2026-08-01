#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import yaml


BASE = Path(os.environ.get("FERA_MS_BASELINE_ROOT", Path(__file__).resolve().parents[1])).resolve()

MAIN = (
    Path(os.environ.get("FERA_MS_BASELINE_SOURCE", BASE / "shared/fragnnet_main"))
)

ICEBERG = (
    Path(os.environ.get("FERA_MS_ICEBERG_SOURCE", BASE / "shared/iceberg_core"))
)

RESULT_ROOT = (
    Path(os.environ.get("FERA_MS_BASELINE_OUTPUT_DIR", BASE / "results_local"))
    / "formal_v1"
)

TOOLS = BASE / "tools_local"

MODEL_INFO = {
    "neims": {
        "repo": MAIN,
        "config":
            MAIN
            / "benchmark_audit/"
            "configs_clean/"
            "neims_ace_reference.yml",
        "epochs": 60,
    },
    "massformer": {
        "repo": MAIN,
        "config":
            MAIN
            / "benchmark_audit/"
            "configs_clean/"
            "massformer_ace_reference.yml",
        "epochs": 80,
    },
    "fragnnet_d3": {
        "repo": MAIN,
        "config":
            MAIN
            / "benchmark_audit/"
            "configs_clean/"
            "fragnnet_fragmentation_ace_reference.yml",
        "epochs": 80,
    },
    "iceberg": {
        "repo": ICEBERG,
        "config":
            ICEBERG
            / "benchmark_audit/"
            "configs_clean/"
            "iceberg_core_reference.yml",
        "epochs": 60,
    },
}

SPLITS = {
    "random": (
        "data/split/"
        "nist20_qtof_cid_safe19659_"
        "qcv1_trainonly"
    ),
    "scaffold": (
        "data/split/"
        "nist20_qtof_cid_safe19659_"
        "scaffold60_20_20_seed42"
    ),
}


def make_config(
    source: Path,
    target: Path,
    split: str,
    seed: int,
    epochs: int,
) -> None:
    config = yaml.safe_load(
        source.read_text(
            encoding="utf-8"
        )
    )

    config["seed"] = int(seed)
    config["split_dp"] = SPLITS[
        split
    ]

    config["min_epochs"] = 1
    config["max_epochs"] = int(
        epochs
    )

    config["eval_test_split"] = True

    config["auxiliary_scores"] = [
        "cos_sim",
        "jss",
    ]

    config["eval_mz_bin_res"] = [
        0.01,
    ]

    config[
        "sparse_cosine_similarity"
    ] = True

    config["compile"] = False

    target.write_text(
        yaml.safe_dump(
            config,
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def clean_temp(repo: Path) -> None:
    shutil.rmtree(
        repo / "tmp_ckpt",
        ignore_errors=True,
    )

    shutil.rmtree(
        repo / "tmp_profile",
        ignore_errors=True,
    )


# CHECKPOINT_MIRROR_V3
def mirror_checkpoints(
    repo: Path,
    staging_dir: Path,
    stop_event: threading.Event,
) -> None:
    staging_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    observed = {}

    while not stop_event.is_set():
        root = repo / "tmp_ckpt"

        if root.exists():
            for source in root.rglob("*.ckpt"):
                try:
                    stat = source.stat()

                    if stat.st_size <= 0:
                        continue

                    identity = (
                        stat.st_size,
                        stat.st_mtime_ns,
                    )

                    key = str(source.resolve())
                    previous = observed.get(key)
                    observed[key] = identity

                    # 等待连续两次大小不变，避免复制未写完的文件。
                    if previous != identity:
                        continue

                    target = staging_dir / source.name
                    temporary = staging_dir / (
                        source.name + ".copying"
                    )

                    if target.exists():
                        target_stat = target.stat()

                        if (
                            target_stat.st_size
                            == stat.st_size
                            and target_stat.st_mtime_ns
                            == stat.st_mtime_ns
                        ):
                            continue

                    try:
                        shutil.copy2(
                            source,
                            temporary,
                        )

                        os.replace(
                            temporary,
                            target,
                        )

                        print(
                            "CHECKPOINT_MIRRORED",
                            source,
                            "->",
                            target,
                            target.stat().st_size,
                            flush=True,
                        )

                    except Exception as exc:
                        temporary.unlink(
                            missing_ok=True
                        )

                        print(
                            "CHECKPOINT_MIRROR_RETRY",
                            repr(exc),
                            flush=True,
                        )

                except FileNotFoundError:
                    continue

        stop_event.wait(0.4)


def run() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=MODEL_INFO,
        required=True,
    )

    parser.add_argument(
        "--split",
        choices=SPLITS,
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        choices=[42, 43, 44],
        required=True,
    )

    args = parser.parse_args()

    info = MODEL_INFO[
        args.model
    ]

    repo = info["repo"]
    epochs = int(
        info["epochs"]
    )

    result_dir = (
        RESULT_ROOT
        / args.split
        / args.model
        / f"seed{args.seed}"
    )

    status_path = (
        result_dir
        / "status.txt"
    )

    if (
        status_path.is_file()
        and status_path
        .read_text(
            encoding="utf-8"
        )
        .strip()
        == "SUCCESS"
    ):
        print(
            "ALREADY_SUCCESS",
            result_dir,
        )
        return

    if result_dir.exists():
        backup = result_dir.with_name(
            result_dir.name
            + ".failed_"
            + time.strftime(
                "%Y%m%d_%H%M%S"
            )
        )

        shutil.move(
            result_dir,
            backup,
        )

        print(
            "MOVED_OLD_PARTIAL_RUN",
            backup,
        )

    result_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    config_path = (
        result_dir
        / "config.yml"
    )

    make_config(
        source=info["config"],
        target=config_path,
        split=args.split,
        seed=args.seed,
        epochs=epochs,
    )

    clean_temp(repo)

    environment = (
        os.environ.copy()
    )

    environment[
        "DGLBACKEND"
    ] = "pytorch"

    environment[
        "PYTHONUNBUFFERED"
    ] = "1"

    environment[
        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"
    ] = "1"

    environment.pop(
        "TORCH_FORCE_WEIGHTS_ONLY_LOAD",
        None,
    )

    environment["PYTHONPATH"] = (
        str(repo / "src")
        + os.pathsep
        + str(repo)
    )

    environment[
        "CUDA_VISIBLE_DEVICES"
    ] = environment.get(
        "CUDA_VISIBLE_DEVICES",
        "0",
    )

    job_name = (
        f"formal_{args.split}_"
        f"{args.model}_"
        f"seed{args.seed}"
    )

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
        job_name,
    ]

    training_log = (
        result_dir
        / "train.log"
    )

    print("=" * 100)
    print("FORMAL TRAINING")
    print("model =", args.model)
    print("split =", args.split)
    print("seed =", args.seed)
    print("epochs =", epochs)
    print("repo =", repo)
    print("result =", result_dir)
    print("command =", " ".join(command))
    print("=" * 100)

    status_path.write_text(
        "TRAINING\n",
        encoding="utf-8",
    )

    staging_dir = (
        result_dir
        / "checkpoint_staging"
    )

    shutil.rmtree(
        staging_dir,
        ignore_errors=True,
    )

    stop_event = threading.Event()

    mirror_thread = threading.Thread(
        target=mirror_checkpoints,
        args=(
            repo,
            staging_dir,
            stop_event,
        ),
        daemon=True,
    )

    mirror_thread.start()

    try:
        with training_log.open(
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
    finally:
        stop_event.set()
        mirror_thread.join(timeout=30)

    if process.returncode != 0:
        status_path.write_text(
            f"TRAIN_FAILED:"
            f"{process.returncode}\n",
            encoding="utf-8",
        )

        print(
            training_log
            .read_text(
                encoding="utf-8",
                errors="replace",
            )[-30000:]
        )

        raise RuntimeError(
            f"Training failed: "
            f"{process.returncode}"
        )

    checkpoint_dir = repo / "tmp_ckpt"

    live_candidates = (
        list(checkpoint_dir.rglob("*.ckpt"))
        if checkpoint_dir.exists()
        else []
    )

    staged_candidates = (
        list(staging_dir.rglob("*.ckpt"))
        if staging_dir.exists()
        else []
    )

    candidates_by_name = {}

    for candidate in (
        staged_candidates
        + live_candidates
    ):
        if candidate.name == "last.ckpt":
            continue

        candidates_by_name[
            candidate.name
        ] = candidate

    training_text = training_log.read_text(
        encoding="utf-8",
        errors="replace",
    )

    patterns = [
        (
            r"Loaded model weights from the "
            r"checkpoint at\s+(.+?\.ckpt)"
        ),
        (
            r"Restoring states from the checkpoint "
            r"path at\s+(.+?\.ckpt)"
        ),
    ]

    loaded_names = []

    for pattern in patterns:
        for match in re.findall(
            pattern,
            training_text,
            flags=re.MULTILINE,
        ):
            loaded_names.append(
                Path(match.strip()).name
            )

    selected_name = (
        loaded_names[-1]
        if loaded_names
        else None
    )

    if (
        selected_name
        and selected_name in candidates_by_name
    ):
        best_source = candidates_by_name[
            selected_name
        ]

    elif len(candidates_by_name) == 1:
        best_source = next(
            iter(candidates_by_name.values())
        )

    else:
        inventory = {
            "loaded_checkpoint_names":
                loaded_names,
            "selected_name":
                selected_name,
            "live_candidates": [
                {
                    "path": str(item),
                    "bytes":
                        item.stat().st_size,
                }
                for item in live_candidates
                if item.exists()
            ],
            "staged_candidates": [
                {
                    "path": str(item),
                    "bytes":
                        item.stat().st_size,
                }
                for item in staged_candidates
                if item.exists()
            ],
        }

        (
            result_dir
            / "checkpoint_inventory.json"
        ).write_text(
            json.dumps(
                inventory,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        raise RuntimeError(
            "Cannot identify validation-best "
            "checkpoint; see "
            "checkpoint_inventory.json"
        )

    print(
        "VALIDATION_BEST_RECOVERED",
        selected_name,
        best_source,
        best_source.stat().st_size,
    )

    best_target = (
        result_dir
        / "best.ckpt"
    )

    shutil.copy2(
        best_source,
        best_target,
    )

    print(
        "BEST_CHECKPOINT_COPIED",
        best_source,
        "->",
        best_target,
        best_target.stat().st_size,
    )

    export_command = [
        sys.executable,
        str(
            TOOLS
            / "export_formal_test.py"
        ),
        "--repo",
        str(repo),
        "--template",
        str(repo / "config/template.yml"),
        "--config",
        str(config_path),
        "--checkpoint",
        str(best_target),
        "--out-dir",
        str(result_dir),
        "--model-name",
        args.model,
        "--split-name",
        args.split,
        "--seed",
        str(args.seed),
    ]

    export_log = (
        result_dir
        / "export.log"
    )

    status_path.write_text(
        "EXPORTING\n",
        encoding="utf-8",
    )

    with export_log.open(
        "w",
        encoding="utf-8",
    ) as handle:
        process = subprocess.run(
            export_command,
            cwd=repo,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )

    if process.returncode != 0:
        status_path.write_text(
            f"EXPORT_FAILED:"
            f"{process.returncode}\n",
            encoding="utf-8",
        )

        print(
            export_log
            .read_text(
                encoding="utf-8",
                errors="replace",
            )[-30000:]
        )

        raise RuntimeError(
            f"Export failed: "
            f"{process.returncode}"
        )

    required = [
        "best.ckpt",
        "config.yml",
        "train.log",
        "export.log",
        "test_per_spectrum.csv.gz",
        "test_per_molecule.csv",
        "metrics.json",
        "model_manifest.json",
    ]

    missing = [
        filename
        for filename in required
        if not (
            result_dir
            / filename
        ).is_file()
    ]

    if missing:
        raise RuntimeError(
            f"Missing formal artifacts: "
            f"{missing}"
        )

    # Only after all persistent outputs pass.
    clean_temp(repo)

    shutil.rmtree(
        result_dir
        / "checkpoint_staging",
        ignore_errors=True,
    )

    status_path.write_text(
        "SUCCESS\n",
        encoding="utf-8",
    )

    metrics = json.loads(
        (
            result_dir
            / "metrics.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    print()
    print("=" * 100)
    print("FORMAL_RUN_SUCCESS")
    print(
        json.dumps(
            metrics,
            indent=2,
            ensure_ascii=False,
        )
    )
    print("RESULT_DIR =", result_dir)
    print("=" * 100)


if __name__ == "__main__":
    run()
