#!/usr/bin/env python3
"""Restore and verify the exact random cohort used by the paper.

The historical random assignment is an input identity, not a split that can be
recreated from a new random seed.  The formal route starts from the archived
``safe19707`` processed tables and split, removes only molecules for which the
required depth-3 DAG is unavailable, and verifies the locked paper files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
IDENTITY_PATH = ROOT / "config/paper_experiment_identity.json"
ID_COLUMNS = ["spec_id", "mol_id", "group_id"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_identity(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise RuntimeError(f"Unsupported identity schema: {path}")
    return payload


def normalized_molecules(path: Path) -> pd.DataFrame:
    frame = pd.read_pickle(path)
    if "mol_id" not in frame.columns:
        if frame.index.name != "mol_id":
            raise RuntimeError(f"mol_df has no mol_id: {path}")
        frame = frame.reset_index()
    return frame


def dag_exists(directory: Path, mol_id: int) -> bool:
    return any(
        (directory / f"{mol_id}{suffix}").is_file()
        for suffix in (".pickle.bz2", ".json.bz2", ".pkl.bz2", ".pickle")
    )


def assert_split_identity(
    split_dir: Path,
    expected: dict,
    *,
    verify_file_hashes: bool,
) -> dict:
    frames: dict[str, pd.DataFrame] = {}
    audit: dict[str, dict] = {}
    for name in ("train", "val", "test"):
        path = split_dir / f"{name}_ids.csv"
        frame = pd.read_csv(path)[ID_COLUMNS]
        frames[name] = frame
        actual = {
            "spectra": int(frame["spec_id"].nunique()),
            "molecules": int(frame["mol_id"].nunique()),
            "sha256": sha256_file(path),
        }
        wanted = expected[name]
        if actual["spectra"] != wanted["spectra"]:
            raise RuntimeError(f"{name} spectrum identity mismatch: {actual} != {wanted}")
        if actual["molecules"] != wanted["molecules"]:
            raise RuntimeError(f"{name} molecule identity mismatch: {actual} != {wanted}")
        if verify_file_hashes and actual["sha256"] != wanted["sha256"]:
            raise RuntimeError(f"{name} SHA-256 mismatch: {actual['sha256']} != {wanted['sha256']}")
        audit[name] = actual

    if pd.concat(frames.values(), ignore_index=True)["spec_id"].duplicated().any():
        raise RuntimeError("A spectrum occurs in more than one random subset")
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = set(frames[left]["mol_id"]) & set(frames[right]["mol_id"])
        if overlap:
            raise RuntimeError(f"Molecule overlap between {left} and {right}: {len(overlap)}")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Restore the exact safe19707 -> safe19659 paper cohort and random split."
    )
    parser.add_argument("--source-proc", type=Path, required=True)
    parser.add_argument("--source-split", type=Path, required=True)
    parser.add_argument("--dag-dir", type=Path, required=True)
    parser.add_argument("--output-proc", type=Path, required=True)
    parser.add_argument("--output-split", type=Path, required=True)
    parser.add_argument("--identity", type=Path, default=IDENTITY_PATH)
    args = parser.parse_args()

    identity = load_identity(args.identity)
    cohort_contract = identity["cohort"]
    source_contract = cohort_contract["historical_source"]
    split_contract = identity["splits"]["random"]

    source_spec = pd.read_pickle(args.source_proc / "spec_df.pkl")
    source_mol = normalized_molecules(args.source_proc / "mol_df.pkl")
    source_ann_path = args.source_proc / "ann_df.pkl"
    source_splits = {
        name: pd.read_csv(args.source_split / f"{name}_ids.csv")[ID_COLUMNS]
        for name in ("train", "val", "test")
    }

    if len(source_spec) != source_contract["spectra"]:
        raise RuntimeError(f"Historical source spectra mismatch: {len(source_spec)}")
    if source_mol["mol_id"].nunique() != source_contract["molecules"]:
        raise RuntimeError(f"Historical source molecules mismatch: {source_mol['mol_id'].nunique()}")

    exclusions = source_contract["excluded_molecules"]
    excluded_ids = {int(item["mol_id"]) for item in exclusions}
    for item in exclusions:
        mol_id = int(item["mol_id"])
        row = source_mol.loc[source_mol["mol_id"] == mol_id]
        if len(row) != 1:
            raise RuntimeError(f"Expected one historical molecule row for mol_id={mol_id}")
        if str(row.iloc[0]["inchikey_s"]) != item["connectivity_key"]:
            raise RuntimeError(f"Connectivity identity mismatch for historical mol_id={mol_id}")
        count = int((source_spec["mol_id"] == mol_id).sum())
        if count != int(item["spectrum_count"]):
            raise RuntimeError(f"Spectrum identity mismatch for historical mol_id={mol_id}")
        if dag_exists(args.dag_dir, mol_id):
            raise RuntimeError(
                f"Locked exclusion mol_id={mol_id} now has a DAG; review the cohort identity rather than silently excluding it"
            )
        if not (source_splits["train"]["mol_id"] == mol_id).any():
            raise RuntimeError(f"Locked exclusion mol_id={mol_id} is not in the historical training split")

    output_spec = source_spec.loc[~source_spec["mol_id"].isin(excluded_ids)].copy()
    output_mol = source_mol.loc[~source_mol["mol_id"].isin(excluded_ids)].copy()
    output_splits = {
        "train": source_splits["train"].loc[
            ~source_splits["train"]["mol_id"].isin(excluded_ids)
        ].copy(),
        "val": source_splits["val"].copy(),
        "test": source_splits["test"].copy(),
    }

    remaining_missing = sorted(
        mol_id
        for mol_id in output_mol["mol_id"].astype(int).unique()
        if not dag_exists(args.dag_dir, int(mol_id))
    )
    if remaining_missing:
        raise RuntimeError(f"Final cohort still has missing required DAGs: {remaining_missing[:20]}")
    if len(output_spec) != cohort_contract["spectra"]:
        raise RuntimeError(f"Final cohort spectrum count mismatch: {len(output_spec)}")
    if output_mol["mol_id"].nunique() != cohort_contract["molecules"]:
        raise RuntimeError(f"Final cohort molecule count mismatch: {output_mol['mol_id'].nunique()}")

    args.output_proc.mkdir(parents=True, exist_ok=True)
    args.output_split.mkdir(parents=True, exist_ok=True)
    output_spec.to_pickle(args.output_proc / "spec_df.pkl")
    output_mol.to_pickle(args.output_proc / "mol_df.pkl")
    shutil.copyfile(source_ann_path, args.output_proc / "ann_df.pkl")
    for name, frame in output_splits.items():
        frame.to_csv(args.output_split / f"{name}_ids.csv", index=False)
    pd.DataFrame(columns=ID_COLUMNS).to_csv(
        args.output_split / "secondary_ids.csv", index=False
    )
    cohort_ids = pd.concat(output_splits.values(), ignore_index=True).sort_values("spec_id")
    cohort_ids.to_csv(args.output_split / "cohort_ids.csv", index=False)

    split_audit = assert_split_identity(
        args.output_split,
        split_contract,
        verify_file_hashes=True,
    )
    file_audit = {
        name: sha256_file(args.output_proc / name)
        for name in ("spec_df.pkl", "mol_df.pkl", "ann_df.pkl")
    }
    for name, expected_sha in cohort_contract["files"].items():
        if file_audit[name] != expected_sha:
            raise RuntimeError(f"{name} SHA-256 mismatch: {file_audit[name]} != {expected_sha}")

    audit = {
        "protocol": "historical_safe19707_pruned_only_for_unavailable_required_depth3_dag",
        "identity_contract": str(args.identity.resolve()),
        "cohort": {
            "spectra": int(len(output_spec)),
            "molecules": int(output_mol["mol_id"].nunique()),
            "file_sha256": file_audit,
        },
        "excluded_molecules": exclusions,
        "random_split": split_audit,
        "molecule_overlap": 0,
        "remaining_missing_required_dags": 0,
        "byte_identity_enforced": True,
    }
    (args.output_split / "audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
