from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

OLD_SCRIPT = (
    ROOT
    / "test/run_retrieval_bootstrap_core.py"
)

OUT_DIR = (
    ROOT
    / "runs/experiments/reviewer_stage2/"
      "retrieval_bootstrap_v2"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_old_module():
    spec = importlib.util.spec_from_file_location(
        "retrieval_bootstrap_base",
        OLD_SCRIPT,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Cannot import {OLD_SCRIPT}"
        )

    module = importlib.util.module_from_spec(
        spec
    )
    spec.loader.exec_module(module)

    return module


def metric_vectors_from_rank_matrix(
    rank_matrix: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Compute each metric within each seed first, then average
    the metric across seeds for every query.
    """
    ranks = np.asarray(
        rank_matrix,
        dtype=float,
    )

    if ranks.ndim != 2:
        raise ValueError(
            f"Expected query x seed matrix, got {ranks.shape}"
        )

    return {
        "Top-1":
            (ranks <= 1.0).mean(axis=1),
        "Top-5":
            (ranks <= 5.0).mean(axis=1),
        "Top-10":
            (ranks <= 10.0).mean(axis=1),
        "MRR":
            (1.0 / ranks).mean(axis=1),
    }


def scalar_metric(
    values: np.ndarray,
    molecule_ids: np.ndarray,
    aggregation: str,
) -> float:
    values = np.asarray(
        values,
        dtype=float,
    )

    if aggregation == "Micro":
        return float(values.mean())

    if aggregation == "Macro":
        return float(
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

    raise ValueError(aggregation)


def main():
    base = load_old_module()

    loaded = {}

    for split in base.SPLITS:
        for model in base.MODELS:
            for seed in base.SEEDS:
                loaded[
                    split,
                    model,
                    seed,
                ] = base.load_one(
                    model=model,
                    split=split,
                    seed=seed,
                )

    cohort_rows = []
    per_seed_rows = []
    seed_mean_rows = []
    comparison_rows = []

    for split in base.SPLITS:
        query_sets = [
            set(
                loaded[
                    split,
                    model,
                    seed,
                ]["query_spec_id"]
                .astype(int)
            )
            for model in base.MODELS
            for seed in base.SEEDS
        ]

        common_queries = set.intersection(
            *query_sets
        )

        model_tables = {}

        for model in base.MODELS:
            merged_model = None

            for seed in base.SEEDS:
                frame = loaded[
                    split,
                    model,
                    seed,
                ].copy()

                frame = frame[
                    frame["query_spec_id"].isin(
                        common_queries
                    )
                ][
                    [
                        "query_spec_id",
                        "target_mol_id",
                        "retrieval_rank",
                    ]
                ].copy()

                frame = frame.rename(
                    columns={
                        "retrieval_rank":
                            f"rank_{model}_{seed}",
                    }
                )

                if merged_model is None:
                    merged_model = frame
                else:
                    merged_model = (
                        merged_model.merge(
                            frame,
                            on=[
                                "query_spec_id",
                                "target_mol_id",
                            ],
                            how="inner",
                            validate="one_to_one",
                        )
                    )

            model_tables[model] = merged_model

        master = model_tables["FERA-MS"]

        for model in base.BASELINES:
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

        expected = base.EXPECTED[split]

        query_count = len(master)
        molecule_count = int(
            master["target_mol_id"].nunique()
        )

        if query_count != expected["queries"]:
            raise RuntimeError(
                f"{split}: expected "
                f"{expected['queries']} queries, "
                f"found {query_count}"
            )

        if molecule_count != expected["molecules"]:
            raise RuntimeError(
                f"{split}: expected "
                f"{expected['molecules']} molecules, "
                f"found {molecule_count}"
            )

        molecule_ids = (
            master["target_mol_id"]
            .to_numpy(dtype=int)
        )

        cohort_rows.append({
            "split": split,
            "method": base.METHOD,
            "queries": query_count,
            "molecules": molecule_count,
            "pool_size": 50,
            "seeds": "42|43|44",
        })

        query_vectors_by_model = {}

        for model in base.MODELS:
            rank_columns = [
                f"rank_{model}_{seed}"
                for seed in base.SEEDS
            ]

            rank_matrix = master[
                rank_columns
            ].to_numpy(dtype=float)

            query_vectors_by_model[
                model
            ] = metric_vectors_from_rank_matrix(
                rank_matrix
            )

            # Per-seed point estimates, preserving Table 3 protocol.
            for column_index, seed in enumerate(
                base.SEEDS
            ):
                seed_ranks = rank_matrix[
                    :,
                    column_index,
                ]

                seed_vectors = {
                    "Top-1":
                        (seed_ranks <= 1).astype(float),
                    "Top-5":
                        (seed_ranks <= 5).astype(float),
                    "Top-10":
                        (seed_ranks <= 10).astype(float),
                    "MRR":
                        1.0 / seed_ranks,
                }

                for metric, values in (
                    seed_vectors.items()
                ):
                    for aggregation in [
                        "Micro",
                        "Macro",
                    ]:
                        per_seed_rows.append({
                            "split": split,
                            "aggregation":
                                aggregation,
                            "model": model,
                            "seed": seed,
                            "metric": metric,
                            "value": scalar_metric(
                                values,
                                molecule_ids,
                                aggregation,
                            ),
                            "queries": query_count,
                            "molecules":
                                molecule_count,
                        })

            # Point estimate based on seed-averaged query metrics.
            for metric, values in (
                query_vectors_by_model[
                    model
                ].items()
            ):
                for aggregation in [
                    "Micro",
                    "Macro",
                ]:
                    seed_mean_rows.append({
                        "split": split,
                        "aggregation":
                            aggregation,
                        "model": model,
                        "metric": metric,
                        "value": scalar_metric(
                            values,
                            molecule_ids,
                            aggregation,
                        ),
                        "queries": query_count,
                        "molecules":
                            molecule_count,
                    })

        for baseline in base.BASELINES:
            for metric in [
                "Top-1",
                "Top-5",
                "Top-10",
                "MRR",
            ]:
                query_difference = (
                    query_vectors_by_model[
                        "FERA-MS"
                    ][metric]
                    - query_vectors_by_model[
                        baseline
                    ][metric]
                )

                for aggregation in [
                    "Micro",
                    "Macro",
                ]:
                    seed_offset = sum(
                        ord(character)
                        for character in (
                            split
                            + baseline
                            + metric
                            + aggregation
                        )
                    )

                    rng = np.random.default_rng(
                        base.RANDOM_SEED
                        + seed_offset
                    )

                    (
                        difference,
                        ci_low,
                        ci_high,
                        p_value,
                    ) = base.paired_bootstrap(
                        query_diff=
                            query_difference,
                        molecule_ids=
                            molecule_ids,
                        aggregation=
                            aggregation,
                        rng=rng,
                    )

                    comparison_rows.append({
                        "split": split,
                        "aggregation":
                            aggregation,
                        "metric": metric,
                        "baseline": baseline,
                        "difference":
                            difference,
                        "ci_low":
                            ci_low,
                        "ci_high":
                            ci_high,
                        "p_value_raw":
                            p_value,
                        "bootstrap_replicates":
                            base.BOOTSTRAPS,
                    })

        master.to_csv(
            OUT_DIR
            / f"{split}_per_seed_ranks.csv.gz",
            index=False,
            compression="gzip",
        )

    cohort = pd.DataFrame(
        cohort_rows
    )

    per_seed = pd.DataFrame(
        per_seed_rows
    )

    seed_mean = pd.DataFrame(
        seed_mean_rows
    )

    mean_std = (
        per_seed
        .groupby(
            [
                "split",
                "aggregation",
                "model",
                "metric",
                "queries",
                "molecules",
            ],
            as_index=False,
            sort=True,
        )["value"]
        .agg(["mean", "std"])
        .reset_index()
    )

    comparisons = pd.DataFrame(
        comparison_rows
    )

    comparisons[
        "p_value_holm"
    ] = base.holm_adjust(
        comparisons[
            "p_value_raw"
        ].to_numpy(dtype=float)
    )

    cohort.to_csv(
        OUT_DIR
        / "retrieval_common_cohort.csv",
        index=False,
    )

    per_seed.to_csv(
        OUT_DIR
        / "retrieval_per_seed_metrics.csv",
        index=False,
    )

    mean_std.to_csv(
        OUT_DIR
        / "retrieval_mean_std_metrics.csv",
        index=False,
    )

    seed_mean.to_csv(
        OUT_DIR
        / "retrieval_seed_averaged_query_metrics.csv",
        index=False,
    )

    comparisons.to_csv(
        OUT_DIR
        / "retrieval_paired_bootstrap.csv",
        index=False,
    )

    print("\nCOMMON COHORT")
    print(cohort.to_string(index=False))

    print("\nMEAN ± SD ACROSS SEEDS")
    print(
        mean_std.to_string(
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
