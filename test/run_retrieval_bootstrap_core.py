from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

RETRIEVAL_ROOT = (
    ROOT
    / "runs/experiments/molecular_retrieval/"
      "pubchem_fixed50"
)

OUT_DIR = (
    ROOT
    / "runs/experiments/reviewer_analysis/"
      "retrieval_bootstrap"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "FERA-MS": (
        RETRIEVAL_ROOT
        / "ours_spectrum_allocator_molecular_retrieval"
    ),
    "FraGNNet-D3": (
        RETRIEVAL_ROOT
        / "baseline_molecular_retrieval/fragnnet_d3"
    ),
    "NEIMS": (
        RETRIEVAL_ROOT
        / "baseline_molecular_retrieval/neims"
    ),
    "MassFormer": (
        RETRIEVAL_ROOT
        / "baseline_molecular_retrieval/massformer"
    ),
}

BASELINES = [
    "FraGNNet-D3",
    "NEIMS",
    "MassFormer",
]

SEEDS = [42, 43, 44]
SPLITS = ["random", "scaffold"]
METHOD = "cbin_sqrt"
BOOTSTRAPS = 20_000
RANDOM_SEED = 20260729

EXPECTED = {
    "random": {
        "queries": 3917,
        "molecules": 454,
    },
    "scaffold": {
        "queries": 3949,
        "molecules": 448,
    },
}


def load_one(
    model: str,
    split: str,
    seed: int,
) -> pd.DataFrame:
    path = (
        MODELS[model]
        / split
        / f"seed_{seed}"
        / "true_candidate_ranks.csv.gz"
    )

    if not path.is_file():
        raise FileNotFoundError(path)

    frame = pd.read_csv(path)

    required = {
        "seed",
        "method",
        "query_spec_id",
        "target_mol_id",
        "retrieval_rank",
        "pool_size_scored",
    }

    missing = required - set(frame.columns)

    if missing:
        raise RuntimeError(
            f"{path} missing columns: "
            f"{sorted(missing)}"
        )

    frame = frame[
        frame["method"].astype(str).eq(METHOD)
        & frame["seed"].astype(int).eq(seed)
        & frame["pool_size_scored"].astype(int).eq(50)
    ].copy()

    frame = frame[
        [
            "query_spec_id",
            "target_mol_id",
            "retrieval_rank",
            "pool_size_scored",
        ]
    ]

    frame["query_spec_id"] = (
        frame["query_spec_id"].astype(int)
    )
    frame["target_mol_id"] = (
        frame["target_mol_id"].astype(int)
    )
    frame["retrieval_rank"] = (
        frame["retrieval_rank"].astype(float)
    )

    # Files may contain exact rows appended by resumed runs.
    frame = frame.drop_duplicates()

    duplicate_queries = frame[
        frame.duplicated(
            "query_spec_id",
            keep=False,
        )
    ]

    if len(duplicate_queries):
        conflict = (
            duplicate_queries
            .groupby("query_spec_id")
            .agg(
                rank_n=(
                    "retrieval_rank",
                    "nunique",
                ),
                mol_n=(
                    "target_mol_id",
                    "nunique",
                ),
            )
        )

        conflict = conflict[
            (conflict["rank_n"] > 1)
            | (conflict["mol_n"] > 1)
        ]

        if len(conflict):
            raise RuntimeError(
                f"Conflicting duplicated queries in {path}:\n"
                f"{conflict.head(20)}"
            )

        frame = frame.drop_duplicates(
            "query_spec_id",
            keep="first",
        )

    return frame.sort_values(
        "query_spec_id"
    ).reset_index(drop=True)


def holm_adjust(
    p_values: np.ndarray,
) -> np.ndarray:
    p_values = np.asarray(
        p_values,
        dtype=float,
    )

    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)

    running = 0.0
    m = len(p_values)

    for position, index in enumerate(order):
        value = (
            (m - position)
            * p_values[index]
        )

        running = max(
            running,
            value,
        )

        adjusted[index] = min(
            running,
            1.0,
        )

    return adjusted


def metric_vectors(
    ranks: np.ndarray,
) -> dict[str, np.ndarray]:
    ranks = np.asarray(
        ranks,
        dtype=float,
    )

    return {
        "Top-1":
            (ranks <= 1.0).astype(float),
        "Top-5":
            (ranks <= 5.0).astype(float),
        "Top-10":
            (ranks <= 10.0).astype(float),
        "MRR":
            1.0 / ranks,
    }


def paired_bootstrap(
    query_diff: np.ndarray,
    molecule_ids: np.ndarray,
    aggregation: str,
    rng: np.random.Generator,
) -> tuple[
    float,
    float,
    float,
    float,
]:
    work = pd.DataFrame({
        "mol_id": molecule_ids,
        "diff": query_diff,
    })

    grouped = (
        work.groupby(
            "mol_id",
            sort=True,
        )["diff"]
        .agg(["sum", "count", "mean"])
    )

    molecule_sums = (
        grouped["sum"].to_numpy(
            dtype=float
        )
    )
    molecule_counts = (
        grouped["count"].to_numpy(
            dtype=float
        )
    )
    molecule_means = (
        grouped["mean"].to_numpy(
            dtype=float
        )
    )

    n_molecules = len(grouped)

    if aggregation == "Micro":
        point = float(
            query_diff.mean()
        )
    elif aggregation == "Macro":
        point = float(
            molecule_means.mean()
        )
    else:
        raise ValueError(aggregation)

    bootstrap_values = np.empty(
        BOOTSTRAPS,
        dtype=float,
    )

    batch_size = 500

    for start in range(
        0,
        BOOTSTRAPS,
        batch_size,
    ):
        stop = min(
            start + batch_size,
            BOOTSTRAPS,
        )

        indices = rng.integers(
            0,
            n_molecules,
            size=(
                stop - start,
                n_molecules,
            ),
        )

        if aggregation == "Micro":
            numerator = (
                molecule_sums[indices]
                .sum(axis=1)
            )
            denominator = (
                molecule_counts[indices]
                .sum(axis=1)
            )

            bootstrap_values[
                start:stop
            ] = (
                numerator / denominator
            )
        else:
            bootstrap_values[
                start:stop
            ] = (
                molecule_means[indices]
                .mean(axis=1)
            )

    ci_low, ci_high = np.quantile(
        bootstrap_values,
        [0.025, 0.975],
    )

    p_lower = (
        (
            np.count_nonzero(
                bootstrap_values <= 0.0
            )
            + 1
        )
        / (BOOTSTRAPS + 1)
    )

    p_upper = (
        (
            np.count_nonzero(
                bootstrap_values >= 0.0
            )
            + 1
        )
        / (BOOTSTRAPS + 1)
    )

    p_value = min(
        1.0,
        2.0 * min(
            p_lower,
            p_upper,
        ),
    )

    return (
        point,
        float(ci_low),
        float(ci_high),
        float(p_value),
    )


def main():
    loaded = {}

    for split in SPLITS:
        for model in MODELS:
            for seed in SEEDS:
                key = (
                    split,
                    model,
                    seed,
                )

                loaded[key] = load_one(
                    model=model,
                    split=split,
                    seed=seed,
                )

    cohort_rows = []
    point_rows = []
    comparison_rows = []

    master_tables = {}

    for split in SPLITS:
        query_sets = [
            set(
                loaded[
                    split,
                    model,
                    seed,
                ]["query_spec_id"]
            )
            for model in MODELS
            for seed in SEEDS
        ]

        common_queries = set.intersection(
            *query_sets
        )

        model_tables = {}

        for model in MODELS:
            seed_tables = []

            for seed in SEEDS:
                frame = loaded[
                    split,
                    model,
                    seed,
                ]

                frame = frame[
                    frame["query_spec_id"]
                    .isin(common_queries)
                ].copy()

                frame = frame.rename(
                    columns={
                        "retrieval_rank":
                            f"rank_{seed}",
                    }
                )

                seed_tables.append(
                    frame[
                        [
                            "query_spec_id",
                            "target_mol_id",
                            f"rank_{seed}",
                        ]
                    ]
                )

            merged = seed_tables[0]

            for seed_frame in seed_tables[1:]:
                merged = merged.merge(
                    seed_frame,
                    on=[
                        "query_spec_id",
                        "target_mol_id",
                    ],
                    how="inner",
                    validate="one_to_one",
                )

            merged[
                f"rank_{model}"
            ] = merged[
                [
                    f"rank_{seed}"
                    for seed in SEEDS
                ]
            ].mean(axis=1)

            model_tables[model] = merged[
                [
                    "query_spec_id",
                    "target_mol_id",
                    f"rank_{model}",
                ]
            ]

        master = model_tables["FERA-MS"]

        for model in BASELINES:
            master = master.merge(
                model_tables[model],
                on=[
                    "query_spec_id",
                    "target_mol_id",
                ],
                how="inner",
                validate="one_to_one",
            )

        master = master.sort_values(
            "query_spec_id"
        ).reset_index(drop=True)

        expected = EXPECTED[split]

        if (
            len(master)
            != expected["queries"]
        ):
            raise RuntimeError(
                f"{split}: expected "
                f"{expected['queries']} common queries, "
                f"found {len(master)}"
            )

        molecule_count = int(
            master["target_mol_id"]
            .nunique()
        )

        if (
            molecule_count
            != expected["molecules"]
        ):
            raise RuntimeError(
                f"{split}: expected "
                f"{expected['molecules']} molecules, "
                f"found {molecule_count}"
            )

        master_tables[split] = master

        cohort_rows.append({
            "split": split,
            "method": METHOD,
            "queries": len(master),
            "molecules": molecule_count,
            "pool_size": 50,
            "seeds": "42|43|44",
            "models": "|".join(MODELS),
        })

        molecule_ids = (
            master["target_mol_id"]
            .to_numpy(dtype=int)
        )

        model_metric_vectors = {}

        for model in MODELS:
            vectors = metric_vectors(
                master[
                    f"rank_{model}"
                ].to_numpy(dtype=float)
            )

            model_metric_vectors[
                model
            ] = vectors

            for metric, values in vectors.items():
                point_rows.append({
                    "split": split,
                    "aggregation": "Micro",
                    "model": model,
                    "metric": metric,
                    "value": float(
                        values.mean()
                    ),
                    "queries": len(master),
                    "molecules": molecule_count,
                })

                macro_value = float(
                    pd.DataFrame({
                        "mol_id": molecule_ids,
                        "value": values,
                    })
                    .groupby(
                        "mol_id",
                        sort=True,
                    )["value"]
                    .mean()
                    .mean()
                )

                point_rows.append({
                    "split": split,
                    "aggregation": "Macro",
                    "model": model,
                    "metric": metric,
                    "value": macro_value,
                    "queries": len(master),
                    "molecules": molecule_count,
                })

        for baseline in BASELINES:
            for metric in [
                "Top-1",
                "Top-5",
                "Top-10",
                "MRR",
            ]:
                difference = (
                    model_metric_vectors[
                        "FERA-MS"
                    ][metric]
                    - model_metric_vectors[
                        baseline
                    ][metric]
                )

                for aggregation in [
                    "Micro",
                    "Macro",
                ]:
                    seed_offset = (
                        sum(
                            ord(character)
                            for character in (
                                split
                                + baseline
                                + metric
                                + aggregation
                            )
                        )
                    )

                    rng = np.random.default_rng(
                        RANDOM_SEED
                        + seed_offset
                    )

                    (
                        point,
                        ci_low,
                        ci_high,
                        p_value,
                    ) = paired_bootstrap(
                        query_diff=difference,
                        molecule_ids=molecule_ids,
                        aggregation=aggregation,
                        rng=rng,
                    )

                    comparison_rows.append({
                        "split": split,
                        "aggregation":
                            aggregation,
                        "metric": metric,
                        "baseline": baseline,
                        "difference":
                            point,
                        "ci_low":
                            ci_low,
                        "ci_high":
                            ci_high,
                        "p_value_raw":
                            p_value,
                        "bootstrap_replicates":
                            BOOTSTRAPS,
                    })

    cohort = pd.DataFrame(
        cohort_rows
    )

    points = pd.DataFrame(
        point_rows
    )

    comparisons = pd.DataFrame(
        comparison_rows
    )

    if len(comparisons) != 48:
        raise RuntimeError(
            "Expected exactly 48 retrieval comparisons "
            "(3 baselines x 2 splits x 4 metrics x "
            "2 aggregations), got "
            f"{len(comparisons)}."
        )

    comparisons[
        "p_value_holm"
    ] = holm_adjust(
        comparisons[
            "p_value_raw"
        ].to_numpy(dtype=float)
    )

    cohort.to_csv(
        OUT_DIR
        / "retrieval_common_cohort.csv",
        index=False,
    )

    points.to_csv(
        OUT_DIR
        / "retrieval_seed_averaged_metrics.csv",
        index=False,
    )

    comparisons.to_csv(
        OUT_DIR
        / "retrieval_paired_bootstrap.csv",
        index=False,
    )

    for split, frame in master_tables.items():
        frame.to_csv(
            OUT_DIR
            / f"{split}_seed_averaged_ranks.csv.gz",
            index=False,
            compression="gzip",
        )

    print("\nCOMMON COHORT")
    print(
        cohort.to_string(index=False)
    )

    print("\nSEED-AVERAGED METRICS")
    print(
        points.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.6f}",
        )
    )

    print("\nPAIRED BOOTSTRAP")
    print(
        comparisons.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.6f}",
        )
    )

    print("\nOUTPUT:", OUT_DIR)


if __name__ == "__main__":
    main()
