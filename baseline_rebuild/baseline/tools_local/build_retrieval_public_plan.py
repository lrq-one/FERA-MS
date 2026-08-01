#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


OURS = Path(os.environ.get("FERA_MS_ROOT", Path(__file__).resolve().parents[3])).resolve()

ROOT = (
    Path(os.environ.get("FERA_MS_RUNS_DIR", OURS / "runs"))
    / "experiments"
    / "molecular_retrieval"
    / "pubchem_legacy_full"
)

OUT = (
    ROOT
    / "baseline_molecular_retrieval"
    / "_common_plan_v1"
)

QUERY_SPEC_PATH = (
    OURS
    / "data"
    / "proc"
    / "nist20_qtof_cid_safe19659"
    / "spec_df.pkl"
)

CANDIDATE_PATH = (
    ROOT
    / "inference_ready_pools_20260723"
    / "molecular_retrieval_inference_ready_candidates.csv.gz"
)

MAPPING_PATH = (
    ROOT
    / "candidate_d3_20260723"
    / "proc"
    / "candidate_structure_mapping.csv.gz"
)

DAG_DIR = (
    ROOT
    / "candidate_d3_20260723"
    / "frag"
    / "dags"
)

MEMBERSHIP_DIR = (
    ROOT
    / "frozen_manifest_20260723"
)

SPLIT_PATHS = {
    "random": (
        OURS
        / "data"
        / "split"
        / "nist20_qtof_cid_safe19659_qcv1_trainonly"
        / "test_ids.csv"
    ),
    "scaffold": (
        OURS
        / "data"
        / "split"
        / "nist20_qtof_cid_safe19659_scaffold60_20_20_seed42"
        / "test_ids.csv"
    ),
}

EXPECTED = {
    ("random", "all_test"):
        (3931, 456),
    ("random", "available_pool"):
        (3930, 455),
    ("random", "fixed50"):
        (3917, 454),
    ("random", "exact_formula"):
        (3917, 454),

    ("scaffold", "all_test"):
        (3960, 450),
    ("scaffold", "available_pool"):
        (3960, 450),
    ("scaffold", "fixed50"):
        (3949, 448),
    ("scaffold", "exact_formula"):
        (3949, 448),
}

EXPECTED_RANDOM_EXPANDED_ROWS = 195_993
EXPECTED_RANDOM_CANDIDATE_STRUCTURES = 21_385
EXPECTED_INCOMPLETE_TARGETS = {"17995"}
EXPECTED_MISSING_DAG_STRUCTURES = 8


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(
                8 * 1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def normalize_id(value) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip()

    try:
        number = float(text)

        if np.isfinite(number):
            integer = int(number)

            if number == integer:
                return str(integer)

    except Exception:
        pass

    return text


def normalize_key(value) -> str:
    if pd.isna(value):
        return ""

    return str(value).strip().upper()


def count_queries(
    frame: pd.DataFrame,
) -> tuple[int, int]:
    return (
        int(len(frame)),
        int(
            frame[
                "mol_id"
            ].nunique()
        ),
    )


def save_csv_gz(
    frame: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.to_csv(
        path,
        index=False,
        compression="gzip",
    )


def load_memberships(
    cohort: str,
) -> pd.DataFrame:
    path = (
        MEMBERSHIP_DIR
        / (
            "molecular_retrieval_"
            f"{cohort}_memberships.csv"
        )
    )

    frame = pd.read_csv(
        path,
        low_memory=False,
    )

    frame["_target_norm"] = (
        frame[
            "target_mol_id"
        ].map(normalize_id)
    )

    return frame


def main() -> None:
    required = [
        QUERY_SPEC_PATH,
        CANDIDATE_PATH,
        MAPPING_PATH,
        DAG_DIR,
        SPLIT_PATHS["random"],
        SPLIT_PATHS["scaffold"],
    ]

    missing = [
        str(path)
        for path in required
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing required inputs:\n"
            + "\n".join(missing)
        )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 110)
    print("PHASE 1: LOAD FROZEN PUBLIC ASSETS")
    print("=" * 110)

    candidates = pd.read_csv(
        CANDIDATE_PATH,
        low_memory=False,
    )

    mapping = pd.read_csv(
        MAPPING_PATH,
        low_memory=False,
    )

    query_spec = pd.read_pickle(
        QUERY_SPEC_PATH
    )

    print(
        "candidate_membership_rows =",
        len(candidates),
    )

    print(
        "candidate_structures =",
        len(mapping),
    )

    print(
        "original_query_spectra =",
        len(query_spec),
    )

    candidates["_target_norm"] = (
        candidates[
            "target_mol_id"
        ].map(normalize_id)
    )

    candidates["_candidate_key_norm"] = (
        candidates[
            "candidate_connectivity_key"
        ].map(normalize_key)
    )

    mapping["_candidate_key_norm"] = (
        mapping[
            "candidate_connectivity_key"
        ].map(normalize_key)
    )

    if mapping[
        "_candidate_key_norm"
    ].duplicated().any():
        sample = mapping.loc[
            mapping[
                "_candidate_key_norm"
            ].duplicated(
                keep=False
            ),
            [
                "candidate_connectivity_key",
                "mol_id",
            ],
        ].head(20)

        raise RuntimeError(
            "Duplicate candidate structure keys:\n"
            + sample.to_string(
                index=False
            )
        )

    mapping_small = mapping[
        [
            "_candidate_key_norm",
            "mol_id",
            "spec_id",
            "candidate_structure_id",
            "candidate_connectivity_key",
            "candidate_smiles",
            "canonical_smiles",
            "full_inchikey",
            "formula",
            "exact_mw",
        ]
    ].rename(
        columns={
            "mol_id":
                "candidate_internal_mol_id",
            "spec_id":
                "candidate_template_spec_id",
            "candidate_connectivity_key":
                "mapping_connectivity_key",
            "candidate_smiles":
                "mapping_candidate_smiles",
            "formula":
                "structure_formula",
            "exact_mw":
                "structure_exact_mw",
        }
    )

    bound = candidates.merge(
        mapping_small,
        on="_candidate_key_norm",
        how="left",
        validate="many_to_one",
    )

    if bound[
        "candidate_internal_mol_id"
    ].isna().any():
        sample = bound.loc[
            bound[
                "candidate_internal_mol_id"
            ].isna(),
            [
                "target_mol_id",
                "candidate_connectivity_key",
            ],
        ].head(20)

        raise RuntimeError(
            "Candidate structure mapping missing:\n"
            + sample.to_string(
                index=False
            )
        )

    if not (
        bound[
            "candidate_smiles"
        ].astype(str)
        ==
        bound[
            "mapping_candidate_smiles"
        ].astype(str)
    ).all():
        raise RuntimeError(
            "Frozen candidate SMILES does not "
            "match candidate structure mapping"
        )

    # candidate_formula is intentionally retained exactly as frozen.
    # It is the protocol namespace used by exact_formula.
    bound[
        "protocol_candidate_formula"
    ] = bound[
        "candidate_formula"
    ]

    print()
    print("=" * 110)
    print("PHASE 2: REPRODUCE OURS INCOMPLETE-POOL POLICY")
    print("=" * 110)

    dag_ids = {
        path.name.split(".")[0]
        for path in DAG_DIR.iterdir()
        if path.is_file()
    }

    bound["has_ours_d3_dag"] = (
        bound[
            "candidate_internal_mol_id"
        ].map(normalize_id)
        .isin(dag_ids)
    )

    missing_dag_rows = bound.loc[
        ~bound[
            "has_ours_d3_dag"
        ]
    ].copy()

    missing_dag_candidates = int(
        missing_dag_rows[
            "_candidate_key_norm"
        ].nunique()
    )

    incomplete_targets = set(
        missing_dag_rows[
            "_target_norm"
        ].unique()
        .tolist()
    )

    print(
        "missing_dag_candidate_structures =",
        missing_dag_candidates,
    )

    print(
        "incomplete_target_count =",
        len(incomplete_targets),
    )

    print(
        "incomplete_targets =",
        sorted(incomplete_targets),
    )

    if (
        missing_dag_candidates
        != EXPECTED_MISSING_DAG_STRUCTURES
    ):
        raise RuntimeError(
            "Unexpected missing-D3 structure count: "
            f"{missing_dag_candidates}"
        )

    if (
        incomplete_targets
        != EXPECTED_INCOMPLETE_TARGETS
    ):
        raise RuntimeError(
            "Unexpected incomplete target set: "
            f"{sorted(incomplete_targets)}"
        )

    excluded_rows = bound.loc[
        bound[
            "_target_norm"
        ].isin(incomplete_targets)
    ].copy()

    excluded_path = (
        OUT
        / "excluded_incomplete_pools.csv"
    )

    excluded_summary = (
        excluded_rows.groupby(
            [
                "target_mol_id",
                "target_connectivity_key",
            ],
            dropna=False,
        )
        .agg(
            candidate_count=(
                "candidate_connectivity_key",
                "size",
            ),
            missing_d3_candidates=(
                "has_ours_d3_dag",
                lambda values: int(
                    (~values).sum()
                ),
            ),
        )
        .reset_index()
    )

    excluded_summary.to_csv(
        excluded_path,
        index=False,
    )

    effective_candidates = bound.loc[
        ~bound[
            "_target_norm"
        ].isin(incomplete_targets)
    ].copy()

    if (
        ~effective_candidates[
            "has_ours_d3_dag"
        ]
    ).any():
        raise RuntimeError(
            "Missing D3 candidates remain after "
            "whole-target exclusion"
        )

    print(
        "effective_candidate_membership_rows =",
        len(effective_candidates),
    )

    print(
        "excluded_membership_rows =",
        len(excluded_rows),
    )

    print(
        "excluded_pool_policy = whole_target"
    )

    effective_candidate_path = (
        OUT
        / "effective_available_candidates.csv.gz"
    )

    effective_candidates.to_csv(
        effective_candidate_path,
        index=False,
        compression="gzip",
    )

    available_memberships = (
        load_memberships(
            "available_pool"
        )
    )

    fixed_memberships = (
        load_memberships(
            "fixed50"
        )
    )

    all_manifests = {}
    combined_plan_parts = []

    print()
    print("=" * 110)
    print("PHASE 3: BUILD IMMUTABLE QUERY-CANDIDATE PLANS")
    print("=" * 110)

    for split in [
        "random",
        "scaffold",
    ]:
        split_label = (
            f"{split}_test"
        )

        test_ids = pd.read_csv(
            SPLIT_PATHS[split],
            low_memory=False,
        )

        required_id_columns = {
            "spec_id",
            "mol_id",
            "group_id",
        }

        if not required_id_columns.issubset(
            test_ids.columns
        ):
            raise RuntimeError(
                f"{split}: unexpected test_ids "
                f"columns {list(test_ids.columns)}"
            )

        test_spec_ids = set(
            test_ids[
                "spec_id"
            ].map(normalize_id)
        )

        query_all = query_spec.loc[
            query_spec[
                "spec_id"
            ].map(normalize_id)
            .isin(test_spec_ids)
        ].copy()

        if query_all[
            "spec_id"
        ].duplicated().any():
            raise RuntimeError(
                f"{split}: duplicate query spec_id"
            )

        all_counts = count_queries(
            query_all
        )

        print()
        print(
            split,
            "all_test =",
            all_counts,
        )

        if (
            all_counts
            != EXPECTED[
                (
                    split,
                    "all_test",
                )
            ]
        ):
            raise RuntimeError(
                f"{split}: all-test mismatch "
                f"{all_counts}"
            )

        available_original = set(
            available_memberships.loc[
                available_memberships[
                    "split"
                ]
                == split_label,
                "_target_norm",
            ].tolist()
        )

        fixed_original = set(
            fixed_memberships.loc[
                fixed_memberships[
                    "split"
                ]
                == split_label,
                "_target_norm",
            ].tolist()
        )

        available_effective = (
            available_original
            - incomplete_targets
        )

        fixed_effective = (
            fixed_original
            - incomplete_targets
        )

        query_all["_target_norm"] = (
            query_all[
                "mol_id"
            ].map(normalize_id)
        )

        query_all["_query_spec_norm"] = (
            query_all[
                "spec_id"
            ].map(normalize_id)
        )

        query_available = query_all.loc[
            query_all[
                "_target_norm"
            ].isin(
                available_effective
            )
        ].copy()

        query_fixed = query_all.loc[
            query_all[
                "_target_norm"
            ].isin(
                fixed_effective
            )
        ].copy()

        available_counts = count_queries(
            query_available
        )

        fixed_counts = count_queries(
            query_fixed
        )

        print(
            split,
            "available_pool =",
            available_counts,
        )

        print(
            split,
            "fixed50 =",
            fixed_counts,
        )

        print(
            split,
            "exact_formula =",
            fixed_counts,
        )

        if (
            available_counts
            != EXPECTED[
                (
                    split,
                    "available_pool",
                )
            ]
        ):
            raise RuntimeError(
                f"{split}: available-pool "
                f"count mismatch "
                f"{available_counts}"
            )

        if (
            fixed_counts
            != EXPECTED[
                (
                    split,
                    "fixed50",
                )
            ]
        ):
            raise RuntimeError(
                f"{split}: fixed50 "
                f"count mismatch "
                f"{fixed_counts}"
            )

        split_dir = (
            OUT
            / split
        )

        split_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        query_columns = [
            "spec_id",
            "mol_id",
            "ace",
            "peaks",
            "prec_mz",
            "group_id",
            "dset_spec_id",
            "_target_norm",
            "_query_spec_norm",
        ]

        missing_query_columns = [
            column
            for column in query_columns
            if column not in query_available.columns
        ]

        if missing_query_columns:
            raise RuntimeError(
                f"{split}: missing query columns "
                f"{missing_query_columns}"
            )

        query_table = query_available[
            query_columns
        ].rename(
            columns={
                "spec_id":
                    "query_spec_id",
                "mol_id":
                    "query_target_mol_id",
                "ace":
                    "query_ace",
                "peaks":
                    "query_peaks",
                "prec_mz":
                    "query_precursor_mz",
                "group_id":
                    "query_group_id",
                "dset_spec_id":
                    "query_dset_spec_id",
            }
        ).copy()

        query_path = (
            split_dir
            / "query_spectra.pkl.gz"
        )

        query_table.to_pickle(
            query_path,
            compression="gzip",
        )

        candidate_pool = (
            effective_candidates.loc[
                effective_candidates[
                    "_target_norm"
                ].isin(
                    available_effective
                )
            ].copy()
        )

        query_small = query_table[
            [
                "_target_norm",
                "_query_spec_norm",
                "query_spec_id",
                "query_target_mol_id",
                "query_ace",
                "query_precursor_mz",
            ]
        ]

        expanded = query_small.merge(
            candidate_pool,
            on="_target_norm",
            how="inner",
            validate="many_to_many",
        )

        expanded[
            "candidate_internal_mol_id"
        ] = expanded[
            "candidate_internal_mol_id"
        ].astype(int)

        expanded[
            "candidate_rank"
        ] = expanded[
            "candidate_rank"
        ].astype(int)

        expanded[
            "is_true_candidate"
        ] = expanded[
            "is_true_candidate"
        ].astype(int)

        expanded[
            "_candidate_sort"
        ] = expanded[
            "candidate_internal_mol_id"
        ]

        expanded[
            "_query_sort"
        ] = pd.to_numeric(
            expanded[
                "query_spec_id"
            ],
            errors="raise",
        )

        expanded = (
            expanded.sort_values(
                [
                    "_candidate_sort",
                    "_query_sort",
                    "candidate_rank",
                ],
                kind="mergesort",
            )
            .reset_index(
                drop=True
            )
        )

        expanded[
            "inference_spec_id"
        ] = np.arange(
            1,
            len(expanded) + 1,
            dtype=np.int64,
        )

        expanded[
            "split"
        ] = split

        expanded[
            "is_fixed50_target"
        ] = expanded[
            "_target_norm"
        ].isin(
            fixed_effective
        )

        true_formula = (
            expanded.loc[
                expanded[
                    "is_true_candidate"
                ]
                == 1,
                [
                    "_query_spec_norm",
                    "protocol_candidate_formula",
                ],
            ]
            .rename(
                columns={
                    "protocol_candidate_formula":
                        "true_protocol_formula",
                }
            )
        )

        true_formula_counts = (
            true_formula.groupby(
                "_query_spec_norm"
            ).size()
        )

        if not (
            true_formula_counts == 1
        ).all():
            raise RuntimeError(
                f"{split}: query does not have "
                "exactly one true candidate"
            )

        expanded = expanded.merge(
            true_formula,
            on="_query_spec_norm",
            how="left",
            validate="many_to_one",
        )

        expanded[
            "protocol_formula_match"
        ] = (
            expanded[
                "protocol_candidate_formula"
            ].astype(str)
            ==
            expanded[
                "true_protocol_formula"
            ].astype(str)
        )

        # Hard identity audits.
        if not expanded[
            "inference_spec_id"
        ].is_unique:
            raise RuntimeError(
                f"{split}: inference_spec_id "
                "is not unique"
            )

        target_per_query = (
            expanded.groupby(
                "_query_spec_norm"
            )[
                "_target_norm"
            ].nunique()
        )

        if not (
            target_per_query == 1
        ).all():
            raise RuntimeError(
                f"{split}: query maps to "
                "multiple targets"
            )

        ace_per_query = (
            expanded.groupby(
                "_query_spec_norm"
            )[
                "query_ace"
            ].nunique(
                dropna=False
            )
        )

        if not (
            ace_per_query == 1
        ).all():
            raise RuntimeError(
                f"{split}: query has "
                "multiple ACE values"
            )

        true_per_query = (
            expanded.groupby(
                "_query_spec_norm"
            )[
                "is_true_candidate"
            ].sum()
        )

        if not (
            true_per_query == 1
        ).all():
            raise RuntimeError(
                f"{split}: query true-candidate "
                "count mismatch"
            )

        actual_count = (
            expanded.groupby(
                "_query_spec_norm"
            ).size()
        )

        declared_count = (
            expanded.groupby(
                "_query_spec_norm"
            )[
                "candidate_count"
            ].first()
            .astype(int)
        )

        if not (
            actual_count
            == declared_count
        ).all():
            bad = pd.DataFrame(
                {
                    "actual":
                        actual_count,
                    "declared":
                        declared_count,
                }
            ).loc[
                lambda frame:
                    frame[
                        "actual"
                    ]
                    != frame[
                        "declared"
                    ]
            ].head(20)

            raise RuntimeError(
                f"{split}: candidate-count "
                "mismatch:\n"
                + bad.to_string()
            )

        fixed_query_ids = set(
            query_fixed[
                "spec_id"
            ].map(normalize_id)
        )

        fixed_rows = expanded.loc[
            expanded[
                "_query_spec_norm"
            ].isin(
                fixed_query_ids
            )
        ]

        fixed_sizes = (
            fixed_rows.groupby(
                "_query_spec_norm"
            ).size()
        )

        if not (
            fixed_sizes == 50
        ).all():
            raise RuntimeError(
                f"{split}: fixed50 query "
                "does not have exactly "
                "50 candidates"
            )

        rank_duplicates = (
            expanded.duplicated(
                [
                    "_query_spec_norm",
                    "candidate_rank",
                ],
                keep=False,
            )
        )

        if rank_duplicates.any():
            raise RuntimeError(
                f"{split}: duplicate frozen "
                "candidate_rank within query"
            )

        plan_columns = [
            "inference_spec_id",
            "split",
            "query_spec_id",
            "query_target_mol_id",
            "query_ace",
            "query_precursor_mz",
            "candidate_connectivity_key",
            "candidate_smiles",
            "protocol_candidate_formula",
            "structure_formula",
            "candidate_internal_mol_id",
            "candidate_template_spec_id",
            "candidate_structure_id",
            "candidate_source_id",
            "candidate_rank",
            "original_generation_rank",
            "local_valid_rank",
            "candidate_count",
            "is_true_candidate",
            "target_morgan_tanimoto",
            "candidate_origin",
            "analysis_cohort",
            "is_fixed50_target",
            "protocol_formula_match",
            "true_protocol_formula",
            "structure_exact_mw",
            "_target_norm",
            "_query_spec_norm",
            "_candidate_key_norm",
        ]

        plan = expanded[
            plan_columns
        ].copy()

        plan_path = (
            split_dir
            / "expanded_run_plan.csv.gz"
        )

        save_csv_gz(
            plan,
            plan_path,
        )

        fixed_ids_frame = (
            query_fixed[
                [
                    "spec_id",
                    "mol_id",
                ]
            ]
            .rename(
                columns={
                    "spec_id":
                        "query_spec_id",
                    "mol_id":
                        "target_mol_id",
                }
            )
            .drop_duplicates()
        )

        fixed_ids_path = (
            split_dir
            / "fixed50_query_ids.csv.gz"
        )

        save_csv_gz(
            fixed_ids_frame,
            fixed_ids_path,
        )

        # exact_formula uses the same query set as fixed50.
        exact_ids_path = (
            split_dir
            / "exact_formula_query_ids.csv.gz"
        )

        shutil.copy2(
            fixed_ids_path,
            exact_ids_path,
        )

        used_candidates = int(
            plan[
                "candidate_internal_mol_id"
            ].nunique()
        )

        print(
            split,
            "expanded_rows =",
            len(plan),
        )

        print(
            split,
            "expanded_queries =",
            plan[
                "query_spec_id"
            ].nunique(),
        )

        print(
            split,
            "expanded_targets =",
            plan[
                "query_target_mol_id"
            ].nunique(),
        )

        print(
            split,
            "candidate_structures_used =",
            used_candidates,
        )

        if split == "random":
            if (
                len(plan)
                != EXPECTED_RANDOM_EXPANDED_ROWS
            ):
                raise RuntimeError(
                    "Random expanded-row mismatch: "
                    f"{len(plan)}"
                )

            if (
                used_candidates
                != EXPECTED_RANDOM_CANDIDATE_STRUCTURES
            ):
                raise RuntimeError(
                    "Random candidate-structure "
                    "count mismatch: "
                    f"{used_candidates}"
                )

        manifest = {
            "split": split,
            "all_test_spectra":
                all_counts[0],
            "all_test_molecules":
                all_counts[1],
            "available_query_spectra":
                available_counts[0],
            "available_query_molecules":
                available_counts[1],
            "fixed50_query_spectra":
                fixed_counts[0],
            "fixed50_query_molecules":
                fixed_counts[1],
            "exact_formula_query_spectra":
                fixed_counts[0],
            "exact_formula_query_molecules":
                fixed_counts[1],
            "expanded_rows":
                int(len(plan)),
            "candidate_structures_used":
                used_candidates,
            "incomplete_targets_excluded":
                sorted(incomplete_targets),
            "query_path":
                str(query_path),
            "query_sha256":
                sha256_file(query_path),
            "plan_path":
                str(plan_path),
            "plan_sha256":
                sha256_file(plan_path),
            "fixed50_ids_path":
                str(fixed_ids_path),
            "fixed50_ids_sha256":
                sha256_file(
                    fixed_ids_path
                ),
            "exact_formula_rule":
                (
                    "fixed50 post-processing "
                    "using frozen "
                    "protocol_candidate_formula "
                    "equality to true candidate"
                ),
            "sort_rule": [
                "candidate_internal_mol_id",
                "query_spec_id",
                "candidate_rank",
            ],
            "join_rule":
                (
                    "model output joins metadata "
                    "one-to-one by "
                    "inference_spec_id"
                ),
        }

        manifest_path = (
            split_dir
            / "plan_manifest.json"
        )

        manifest_path.write_text(
            json.dumps(
                manifest,
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        all_manifests[split] = (
            manifest
        )

        combined_plan_parts.append(
            plan.assign(
                public_plan_split=split
            )
        )

    print()
    print("=" * 110)
    print("PHASE 4: CROSS-SPLIT AND CROSS-SEED CONTRACT")
    print("=" * 110)

    public_contract = {
        "source_query_spectra":
            str(QUERY_SPEC_PATH),
        "source_candidates":
            str(CANDIDATE_PATH),
        "source_structure_mapping":
            str(MAPPING_PATH),
        "incomplete_pool_policy":
            "exclude_whole_target",
        "incomplete_targets":
            sorted(incomplete_targets),
        "missing_d3_structures":
            missing_dag_candidates,
        "candidate_membership_rebuilt":
            False,
        "pubchem_redownloaded":
            False,
        "negative_candidates_reselected":
            False,
        "candidate_rank_changed":
            False,
        "candidate_formula_changed":
            False,
        "candidate_tanimoto_changed":
            False,
        "query_ace_source":
            (
                "original NIST20 "
                "query spectrum ace"
            ),
        "query_peaks_source":
            (
                "original NIST20 "
                "query spectrum peaks"
            ),
        "model_input_semantics":
            (
                "candidate structure "
                "+ query ACE"
            ),
        "model_output_contract": [
            "inference_spec_id",
            "predicted_mz",
            "predicted_intensity",
        ],
        "model_output_join":
            (
                "one-to-one on "
                "inference_spec_id"
            ),
        "candidate_failure_policy":
            (
                "never silently drop a "
                "single candidate"
            ),
        "cohort_derivation": {
            "available_pool":
                (
                    "effective available "
                    "targets after whole-pool "
                    "D3 completeness exclusion"
                ),
            "fixed50":
                (
                    "frozen fixed50 "
                    "target membership"
                ),
            "exact_formula":
                (
                    "fixed50 score rows "
                    "filtered by frozen "
                    "protocol candidate formula"
                ),
        },
        "score_methods": [
            "cbin",
            "cbin_sqrt",
            "jss",
        ],
        "bin_width_da": 0.01,
        "bin_assignment":
            "numpy.rint(mz / 0.01)",
        "splits":
            all_manifests,
        "seed_identity_contract":
            (
                "The exact same split plan "
                "is used for seeds "
                "42, 43 and 44"
            ),
    }

    contract_path = (
        OUT
        / "public_plan_contract.json"
    )

    contract_path.write_text(
        json.dumps(
            public_contract,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    (
        OUT
        / "PUBLIC_PLAN_READY"
    ).write_text(
        "MOLECULAR_RETRIEVAL_PUBLIC_PLAN_READY\n",
        encoding="utf-8",
    )

    print(
        "PUBLIC_PLAN_CONTRACT =",
        contract_path,
    )

    print(
        "MOLECULAR_RETRIEVAL_PUBLIC_PLAN_READY"
    )


if __name__ == "__main__":
    main()
