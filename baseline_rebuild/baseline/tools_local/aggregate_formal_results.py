#!/usr/bin/env python3

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(os.environ.get("FERA_MS_BASELINE_ROOT", Path(__file__).resolve().parents[1])).resolve()

ROOT = (
    Path(os.environ.get("FERA_MS_BASELINE_OUTPUT_DIR", BASE / "results_local"))
    / "formal_v1"
)

MODELS = [
    "neims",
    "massformer",
    "fragnnet_d3",
    "iceberg",
]

SPLITS = [
    "random",
    "scaffold",
]

SEEDS = [
    42,
    43,
    44,
]

METRICS = [
    "micro_cbin",
    "macro_cbin",
    "micro_jss",
    "macro_jss",
]

OLD = {
    ("random", "neims"): {
        "micro_cbin": 0.539813,
        "macro_cbin": 0.609104,
        "micro_jss": 0.527694,
        "macro_jss": 0.587309,
    },
    ("random", "massformer"): {
        "micro_cbin": 0.488705,
        "macro_cbin": 0.569718,
        "micro_jss": 0.476723,
        "macro_jss": 0.545117,
    },
    ("random", "fragnnet_d3"): {
        "micro_cbin": 0.397917,
        "macro_cbin": 0.444995,
        "micro_jss": 0.380211,
        "macro_jss": 0.409460,
    },
    ("random", "iceberg"): {
        "micro_cbin": 0.355295,
        "macro_cbin": 0.425834,
        "micro_jss": 0.358493,
        "macro_jss": 0.419959,
    },
    ("scaffold", "neims"): {
        "micro_cbin": 0.485762,
        "macro_cbin": 0.547049,
        "micro_jss": 0.475262,
        "macro_jss": 0.522164,
    },
    ("scaffold", "massformer"): {
        "micro_cbin": 0.428516,
        "macro_cbin": 0.502220,
        "micro_jss": 0.416702,
        "macro_jss": 0.474760,
    },
    ("scaffold", "fragnnet_d3"): {
        "micro_cbin": 0.376104,
        "macro_cbin": 0.435349,
        "micro_jss": 0.361346,
        "macro_jss": 0.396282,
    },
    ("scaffold", "iceberg"): {
        "micro_cbin": 0.320724,
        "macro_cbin": 0.385709,
        "micro_jss": 0.323407,
        "macro_jss": 0.374149,
    },
}


def classify(delta: float) -> str:
    absolute = abs(delta)

    if absolute <= 0.010:
        return "NORMAL"

    if absolute <= 0.020:
        return "REVIEW"

    return "ABNORMAL"


def main() -> None:
    rows = []
    missing = []

    for split in SPLITS:
        for model in MODELS:
            for seed in SEEDS:
                run_dir = (
                    ROOT
                    / split
                    / model
                    / f"seed{seed}"
                )

                status_path = (
                    run_dir
                    / "status.txt"
                )

                metrics_path = (
                    run_dir
                    / "metrics.json"
                )

                manifest_path = (
                    run_dir
                    / "model_manifest.json"
                )

                checkpoint_path = (
                    run_dir
                    / "best.ckpt"
                )

                if not (
                    status_path.is_file()
                    and status_path.read_text(
                        encoding="utf-8"
                    ).strip() == "SUCCESS"
                    and metrics_path.is_file()
                    and manifest_path.is_file()
                    and checkpoint_path.is_file()
                ):
                    missing.append(
                        {
                            "split": split,
                            "model": model,
                            "seed": seed,
                            "run_dir": str(run_dir),
                        }
                    )
                    continue

                metrics = json.loads(
                    metrics_path.read_text(
                        encoding="utf-8"
                    )
                )

                manifest = json.loads(
                    manifest_path.read_text(
                        encoding="utf-8"
                    )
                )

                row = {
                    "split": split,
                    "model": model,
                    "seed": seed,
                    "test_spectra":
                        metrics["test_spectra"],
                    "test_molecules":
                        metrics["test_molecules"],
                    "checkpoint_epoch":
                        manifest.get(
                            "checkpoint_epoch"
                        ),
                    "checkpoint_sha256":
                        manifest.get(
                            "checkpoint_sha256"
                        ),
                    "checkpoint_bytes":
                        manifest.get(
                            "checkpoint_bytes"
                        ),
                }

                for metric in METRICS:
                    row[metric] = float(
                        metrics[metric]
                    )

                rows.append(row)

    per_seed = pd.DataFrame(rows)

    per_seed.to_csv(
        ROOT / "formal_per_seed.csv",
        index=False,
    )

    (
        ROOT
        / "missing_runs.json"
    ).write_text(
        json.dumps(
            missing,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    if per_seed.empty:
        raise RuntimeError(
            "No successful formal runs found"
        )

    summary_rows = []

    for (
        split,
        model,
    ), frame in per_seed.groupby(
        ["split", "model"],
        sort=False,
    ):
        row = {
            "split": split,
            "model": model,
            "runs": int(len(frame)),
        }

        for metric in METRICS:
            values = frame[
                metric
            ].to_numpy(
                dtype=np.float64
            )

            row[
                f"{metric}_mean"
            ] = float(
                values.mean()
            )

            row[
                f"{metric}_std"
            ] = (
                float(
                    values.std(ddof=1)
                )
                if len(values) > 1
                else np.nan
            )

        summary_rows.append(row)

    summary = pd.DataFrame(
        summary_rows
    ).sort_values(
        ["split", "model"]
    )

    summary.to_csv(
        ROOT
        / "formal_three_seed_summary.csv",
        index=False,
    )

    comparison_rows = []

    for row in summary.to_dict(
        orient="records"
    ):
        split = row["split"]
        model = row["model"]

        reference = OLD[
            (split, model)
        ]

        for metric in METRICS:
            new_value = float(
                row[
                    f"{metric}_mean"
                ]
            )

            old_value = float(
                reference[metric]
            )

            delta = (
                new_value
                - old_value
            )

            comparison_rows.append(
                {
                    "split": split,
                    "model": model,
                    "metric": metric,
                    "old_mean": old_value,
                    "new_mean": new_value,
                    "delta": delta,
                    "absolute_delta":
                        abs(delta),
                    "status":
                        classify(delta),
                }
            )

    comparison = pd.DataFrame(
        comparison_rows
    )

    comparison.to_csv(
        ROOT
        / "comparison_to_previous.csv",
        index=False,
    )

    order_records = []

    expected_order = MODELS

    for split in SPLITS:
        split_summary = (
            summary[
                summary["split"]
                == split
            ]
            .sort_values(
                "micro_cbin_mean",
                ascending=False,
            )
        )

        actual_order = (
            split_summary[
                "model"
            ].tolist()
        )

        order_records.append(
            {
                "split": split,
                "expected_order":
                    expected_order,
                "actual_order":
                    actual_order,
                "order_ok":
                    actual_order
                    == expected_order,
            }
        )

    (
        ROOT
        / "model_order_audit.json"
    ).write_text(
        json.dumps(
            order_records,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 120)
    print("FORMAL PER-SEED RESULTS")
    print("=" * 120)

    print(
        per_seed[
            [
                "split",
                "model",
                "seed",
                *METRICS,
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("=" * 120)
    print("FORMAL THREE-SEED SUMMARY")
    print("=" * 120)

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 120)
    print("COMPARISON TO PREVIOUS")
    print("=" * 120)

    print(
        comparison.to_string(
            index=False
        )
    )

    print()
    print("SUCCESSFUL_RUNS =", len(per_seed))
    print("MISSING_RUNS =", len(missing))

    abnormal = comparison[
        comparison["status"]
        == "ABNORMAL"
    ]

    review = comparison[
        comparison["status"]
        == "REVIEW"
    ]

    print(
        "ABNORMAL_COMPARISONS =",
        len(abnormal),
    )

    print(
        "REVIEW_COMPARISONS =",
        len(review),
    )

    if len(per_seed) == 24:
        print(
            "ALL_24_FORMAL_RUNS_PRESENT"
        )

    if (
        len(per_seed) == 24
        and len(abnormal) == 0
        and all(
            item["order_ok"]
            for item in order_records
        )
    ):
        print(
            "FORMAL_BASELINE_RESULTS_ACCEPTED"
        )
    elif len(per_seed) == 24:
        print(
            "FORMAL_BASELINE_RESULTS_REQUIRE_REVIEW"
        )


if __name__ == "__main__":
    main()
