#!/usr/bin/env python3

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from config_store import load_locked_config, locked_config_path


BASE = Path(os.environ.get("FERA_MS_BASELINE_ROOT", Path(__file__).resolve().parents[1])).resolve()

TOOLS = BASE / "tools_local"

ROOT = (
    Path(os.environ.get("FERA_MS_BASELINE_OUTPUT_DIR", BASE / "results_local"))
    / "formal"
)

SEEDS = [
    42,
    43,
    44,
]

JOBS = []

# Complete one model at a time so that an
# molecular-retrieval-ready model family becomes
# available as early as possible.
for model in [
    "neims",
    "massformer",
    "fragnnet_d3",
    "iceberg",
    "graff_ms",
]:
    for split in [
        "random",
        "scaffold",
    ]:
        for seed in SEEDS:
            JOBS.append(
                {
                    "model": model,
                    "split": split,
                    "seed": seed,
                }
            )


def append_progress(
    record: dict,
) -> None:
    path = (
        ROOT
        / "matrix_progress.jsonl"
    )

    with path.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )
        handle.flush()
        os.fsync(
            handle.fileno()
        )


def run_is_successful(
    job: dict,
) -> bool:
    status = (
        ROOT
        / job["split"]
        / job["model"]
        / f"seed{job['seed']}"
        / "status.txt"
    )

    return (
        status.is_file()
        and status.read_text(
            encoding="utf-8"
        ).strip()
        == "SUCCESS"
    )


def read_metrics(
    job: dict,
):
    path = (
        ROOT
        / job["split"]
        / job["model"]
        / f"seed{job['seed']}"
        / "metrics.json"
    )

    if not path.is_file():
        return None

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the locked 30-job manuscript baseline matrix.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the matrix without creating outputs or training",
    )
    args = parser.parse_args()

    if args.dry_run:
        missing = []
        for job in JOBS:
            config = locked_config_path(BASE, job["model"])
            if not config.is_file():
                missing.append(str(config))
                continue
            load_locked_config(BASE, job["model"], job["split"], job["seed"])
        if missing:
            raise FileNotFoundError("\n".join(missing))
        print("FORMAL_BASELINE_MATRIX_DRY_RUN_OK", len(JOBS))
        return

    ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    lock_path = (
        ROOT
        / "formal_matrix.lock"
    )

    lock_handle = lock_path.open(
        "w",
        encoding="utf-8",
    )

    try:
        fcntl.flock(
            lock_handle,
            fcntl.LOCK_EX
            | fcntl.LOCK_NB,
        )
    except BlockingIOError:
        raise SystemExit(
            "ANOTHER_FORMAL_MATRIX_IS_RUNNING"
        )

    environment = os.environ.copy()

    environment[
        "PYTHONUNBUFFERED"
    ] = "1"

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

    environment[
        "CUDA_VISIBLE_DEVICES"
    ] = "0"

    print("=" * 110)
    print("FORMAL BASELINE MATRIX")
    print("jobs =", len(JOBS))
    print("python =", sys.executable)
    print("root =", ROOT)
    print("=" * 110)

    completed = 0

    for index, job in enumerate(
        JOBS,
        start=1,
    ):
        identity = (
            f"{job['model']}/"
            f"{job['split']}/"
            f"seed{job['seed']}"
        )

        print()
        print("=" * 110)
        print(
            f"JOB {index}/"
            f"{len(JOBS)}:"
        )
        print(identity)
        print("=" * 110)

        if run_is_successful(job):
            metrics = read_metrics(job)

            print(
                "SKIP_EXISTING_SUCCESS",
                identity,
            )

            append_progress(
                {
                    "time":
                        datetime.now()
                        .isoformat(
                            timespec="seconds"
                        ),
                    "job_index":
                        index,
                    "identity":
                        identity,
                    "status":
                        "SKIPPED_SUCCESS",
                    "metrics":
                        metrics,
                }
            )

            completed += 1
            continue

        command = [
            sys.executable,
            str(
                TOOLS
                / "run_formal_one.py"
            ),
            "--model",
            job["model"],
            "--split",
            job["split"],
            "--seed",
            str(job["seed"]),
        ]

        append_progress(
            {
                "time":
                    datetime.now()
                    .isoformat(
                        timespec="seconds"
                    ),
                "job_index":
                    index,
                "identity":
                    identity,
                "status":
                    "STARTED",
            }
        )

        process = subprocess.run(
            command,
            env=environment,
            check=False,
        )

        if (
            process.returncode != 0
            or not run_is_successful(job)
        ):
            append_progress(
                {
                    "time":
                        datetime.now()
                        .isoformat(
                            timespec="seconds"
                        ),
                    "job_index":
                        index,
                    "identity":
                        identity,
                    "status":
                        "FAILED",
                    "returncode":
                        process.returncode,
                }
            )

            print(
                "FORMAL_MATRIX_STOPPED_ON_FAILURE",
                identity,
                process.returncode,
            )

            raise SystemExit(
                process.returncode
                if process.returncode != 0
                else 1
            )

        metrics = read_metrics(job)

        append_progress(
            {
                "time":
                    datetime.now()
                    .isoformat(
                        timespec="seconds"
                    ),
                "job_index":
                    index,
                "identity":
                    identity,
                "status":
                    "SUCCESS",
                "metrics":
                    metrics,
            }
        )

        completed += 1

        print(
            "MATRIX_JOB_SUCCESS",
            identity,
        )

        if metrics:
            print(
                json.dumps(
                    metrics,
                    indent=2,
                    ensure_ascii=False,
                )
            )

    print()
    print("=" * 110)
    print(
        "ALL_FORMAL_MATRIX_JOBS_SUCCESS"
    )
    print(
        "completed =",
        completed,
    )
    print("=" * 110)

    aggregate_command = [
        sys.executable,
        str(
            TOOLS
            / "aggregate_formal_results.py"
        ),
    ]

    aggregate_process = subprocess.run(
        aggregate_command,
        env=environment,
        check=False,
    )

    if aggregate_process.returncode != 0:
        raise SystemExit(
            "FORMAL_AGGREGATION_FAILED"
        )

    print(
        "FORMAL_MATRIX_AND_AGGREGATION_COMPLETE"
    )


if __name__ == "__main__":
    main()
