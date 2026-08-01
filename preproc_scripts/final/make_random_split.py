#!/usr/bin/env python3
"""Select the fixed benchmark cohort, then create its random split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


QC_COLUMNS = (
    "n_peaks",
    "max_intensity_frac",
    "entropy_norm",
    "precursor_survival_yield",
    "support_PR_abs006",
    "support_PWR_abs006",
    "true_oos_intensity",
)


def load_eligible_pool(split_dir: Path) -> pd.DataFrame:
    frames = [
        pd.read_csv(split_dir / f"{split}_ids.csv")
        for split in ("train", "val", "test")
    ]
    pool = pd.concat(frames, ignore_index=True)[
        ["spec_id", "mol_id", "group_id"]
    ]
    if pool["spec_id"].duplicated().any():
        raise RuntimeError("Eligible split union contains duplicate spec_id values")
    return pool


def score_qc(pool: pd.DataFrame) -> pd.DataFrame:
    result = pool.copy()
    missing = sorted(set(QC_COLUMNS) - set(result.columns))
    if missing:
        raise RuntimeError(f"QC table is missing columns: {missing}")

    result["n_peaks"] = result["n_peaks"].fillna(0)
    result["max_intensity_frac"] = result["max_intensity_frac"].fillna(1.0)
    result["entropy_norm"] = result["entropy_norm"].fillna(0.0)
    result["precursor_survival_yield"] = result[
        "precursor_survival_yield"
    ].fillna(1.0)
    result["support_PWR_abs006"] = result["support_PWR_abs006"].fillna(0.0)
    result["support_PR_abs006"] = result["support_PR_abs006"].fillna(0.0)

    result["bad_too_few_peaks"] = result["n_peaks"] < 3
    result["bad_low_support"] = result["support_PWR_abs006"] < 0.35
    result["bad_precursor_dominated"] = (
        result["precursor_survival_yield"] > 0.95
    )
    result["bad_single_peak"] = result["max_intensity_frac"] > 0.98
    result["bad_low_entropy"] = result["entropy_norm"] < 0.03
    hard_bad_columns = (
        "bad_too_few_peaks",
        "bad_low_support",
        "bad_precursor_dominated",
        "bad_single_peak",
        "bad_low_entropy",
    )
    result["hard_bad"] = result[list(hard_bad_columns)].any(axis=1)

    result["qcv1_bad_score"] = (
        0.45 * (1.0 - result["support_PWR_abs006"]).clip(0.0, 1.0)
        + 0.20 * result["precursor_survival_yield"].clip(0.0, 1.0)
        + 0.15 * result["max_intensity_frac"].clip(0.0, 1.0)
        + 0.10 * (1.0 - result["entropy_norm"]).clip(0.0, 1.0)
        + 0.10
        * (1.0 / np.sqrt(result["n_peaks"].clip(lower=1))).clip(0.0, 1.0)
    )
    return result


def select_fixed_cohort(
    scored: pd.DataFrame,
    target_count: int,
    target_molecules: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = scored.loc[~scored["hard_bad"]].sort_values(
        ["qcv1_bad_score", "spec_id"],
        kind="mergesort",
    )
    if len(candidates) < target_count:
        raise RuntimeError(
            f"Only {len(candidates)} spectra pass hard QC; {target_count} required"
        )

    best_per_molecule = candidates.drop_duplicates(
        "connectivity_key",
        keep="first",
    )
    if best_per_molecule["connectivity_key"].nunique() != target_molecules:
        raise RuntimeError(
            "QC-passing cohort does not contain the locked number of molecules: "
            f"{best_per_molecule['connectivity_key'].nunique()} != {target_molecules}"
        )

    selected_ids = set(best_per_molecule["spec_id"])
    remaining = candidates.loc[~candidates["spec_id"].isin(selected_ids)]
    selected = pd.concat(
        [
            best_per_molecule,
            remaining.head(target_count - len(best_per_molecule)),
        ],
        ignore_index=True,
    ).sort_values("spec_id")
    dropped = scored.loc[~scored["spec_id"].isin(selected["spec_id"])].copy()
    return selected, dropped


def exact_subset(
    groups: pd.DataFrame,
    target: int,
    seed: int,
) -> set[str]:
    shuffled = groups.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    previous: dict[int, tuple[int, int]] = {0: (-1, -1)}
    for index, size in enumerate(shuffled["spectrum_count"].astype(int)):
        for subtotal in sorted(tuple(previous), reverse=True):
            updated = subtotal + int(size)
            if updated <= target and updated not in previous:
                previous[updated] = (subtotal, index)
        if target in previous:
            break

    if target not in previous:
        raise RuntimeError(
            f"Cannot construct an exact molecule-disjoint partition of {target} spectra"
        )
    chosen_total = target
    chosen_indices = []
    while chosen_total:
        chosen_total, index = previous[chosen_total]
        chosen_indices.append(index)
    return set(shuffled.loc[chosen_indices, "connectivity_key"].astype(str))


def make_random_split(cohort: pd.DataFrame, seed: int) -> dict[str, pd.DataFrame]:
    groups = cohort.groupby("connectivity_key", as_index=False).agg(
        spectrum_count=("spec_id", "nunique")
    )
    target = int(np.floor(len(cohort) * 0.20))
    test_keys = None
    val_keys = None
    for attempt in range(100):
        candidate_test = exact_subset(groups, target, seed + 2 * attempt)
        remaining = groups.loc[
            ~groups["connectivity_key"].isin(candidate_test)
        ]
        try:
            candidate_val = exact_subset(
                remaining,
                target,
                seed + 2 * attempt + 1,
            )
        except RuntimeError:
            continue
        test_keys = candidate_test
        val_keys = candidate_val
        break
    if test_keys is None or val_keys is None:
        raise RuntimeError(
            "Unable to construct exact random validation/test spectrum counts"
        )
    train_keys = set(groups["connectivity_key"].astype(str)) - test_keys - val_keys

    result = {}
    for split, keys in (
        ("train", train_keys),
        ("val", val_keys),
        ("test", test_keys),
    ):
        result[split] = cohort.loc[
            cohort["connectivity_key"].astype(str).isin(keys),
            ["spec_id", "mol_id", "group_id"],
        ].sort_values("spec_id")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_split_dp", type=Path, required=True)
    parser.add_argument("--qc_csv", type=Path, required=True)
    parser.add_argument("--mol_df", type=Path, required=True)
    parser.add_argument("--out_split_dp", type=Path, required=True)
    parser.add_argument("--target_count", type=int, default=19659)
    parser.add_argument("--target_molecules", type=int, default=2274)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    eligible = load_eligible_pool(args.base_split_dp)
    qc = pd.read_csv(args.qc_csv)
    molecules = pd.read_pickle(args.mol_df)
    if "mol_id" not in molecules.columns:
        molecules = molecules.reset_index()
    molecules = molecules[["mol_id", "inchikey_s"]].drop_duplicates("mol_id")

    scored = eligible.merge(
        qc,
        on=["spec_id", "mol_id", "group_id"],
        how="left",
        validate="one_to_one",
    ).merge(molecules, on="mol_id", how="left", validate="many_to_one")
    scored = scored.rename(columns={"inchikey_s": "connectivity_key"})
    if scored["connectivity_key"].isna().any():
        raise RuntimeError("Some eligible spectra lack a connectivity key")
    if scored[list(QC_COLUMNS)].isna().any().any():
        raise RuntimeError("QC table does not cover every eligible spectrum")

    scored = score_qc(scored)
    cohort, dropped = select_fixed_cohort(
        scored,
        args.target_count,
        args.target_molecules,
    )
    splits = make_random_split(cohort, args.seed)
    expected_eval_count = int(np.floor(args.target_count * 0.20))
    for split in ("val", "test"):
        if len(splits[split]) != expected_eval_count:
            raise RuntimeError(
                f"{split} contains {len(splits[split])} spectra; "
                f"expected {expected_eval_count}"
            )

    args.out_split_dp.mkdir(parents=True, exist_ok=True)
    for split, frame in splits.items():
        frame.to_csv(args.out_split_dp / f"{split}_ids.csv", index=False)
    pd.DataFrame(columns=["spec_id", "mol_id", "group_id"]).to_csv(
        args.out_split_dp / "secondary_ids.csv",
        index=False,
    )
    cohort.to_csv(args.out_split_dp / "cohort_ids.csv", index=False)
    cohort.to_csv(args.out_split_dp / "qcv1_keep_details.csv", index=False)
    dropped.to_csv(args.out_split_dp / "qcv1_drop_details.csv", index=False)

    overlap = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap[f"{left}_{right}"] = len(
            set(splits[left]["mol_id"]) & set(splits[right]["mol_id"])
        )
    if any(overlap.values()):
        raise RuntimeError(f"Molecule overlap across random splits: {overlap}")

    audit = {
        "protocol": "cohort-wide QC before molecule-disjoint random splitting",
        "seed": args.seed,
        "cohort_spectra": int(len(cohort)),
        "cohort_molecules": int(cohort["connectivity_key"].nunique()),
        "split_spectra": {name: int(len(frame)) for name, frame in splits.items()},
        "split_molecules": {
            name: int(frame["mol_id"].nunique()) for name, frame in splits.items()
        },
        "molecule_overlap": overlap,
    }
    (args.out_split_dp / "audit.json").write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
