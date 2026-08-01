#!/usr/bin/env python3
"""Materialize and verify the paper safe19659 depth-3 nl_v1 DAG cache."""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IDENTITY = ROOT / "config/paper_experiment_identity.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dag_molecule_id(path: Path) -> int:
    token = path.name.split(".", 1)[0]
    try:
        return int(token)
    except ValueError as exc:
        raise RuntimeError(f"DAG filename does not begin with a molecule ID: {path.name}") from exc


def materialize(source: Path, target: Path, mode: str) -> None:
    if target.exists():
        if not target.is_file() or sha256_file(target) != sha256_file(source):
            raise RuntimeError(f"Existing DAG differs from the archived source: {target}")
        return
    if mode == "copy":
        shutil.copy2(source, target)
    elif mode == "symlink":
        target.symlink_to(source.resolve())
    else:
        try:
            os.link(source, target)
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise
            shutil.copy2(source, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dag-dir", type=Path, required=True)
    parser.add_argument("--cohort-ids", type=Path, required=True)
    parser.add_argument("--output-dag-dir", type=Path, required=True)
    parser.add_argument("--identity", type=Path, default=DEFAULT_IDENTITY)
    parser.add_argument("--mode", choices=("hardlink", "copy", "symlink"), default="hardlink")
    args = parser.parse_args()

    identity = json.loads(args.identity.read_text(encoding="utf-8"))
    contract = identity["dag_cache"]
    cohort = pd.read_csv(args.cohort_ids)
    if "mol_id" not in cohort.columns:
        raise RuntimeError(f"cohort file has no mol_id column: {args.cohort_ids}")
    molecule_ids = sorted({int(value) for value in cohort["mol_id"].tolist()})
    expected_count = int(contract["molecules"])
    if len(molecule_ids) != expected_count:
        raise RuntimeError(f"Formal cohort has {len(molecule_ids)} molecules; expected {expected_count}")

    source_files: dict[int, Path] = {}
    for path in args.source_dag_dir.iterdir():
        if not path.is_file():
            continue
        mol_id = dag_molecule_id(path)
        if mol_id in source_files:
            raise RuntimeError(f"Multiple archived DAG files found for molecule {mol_id}")
        source_files[mol_id] = path
    missing = [mol_id for mol_id in molecule_ids if mol_id not in source_files]
    if missing:
        raise RuntimeError(f"Archived DAG cache is missing {len(missing)} formal molecules: {missing[:20]}")

    args.output_dag_dir.mkdir(parents=True, exist_ok=True)
    expected_names = {source_files[mol_id].name for mol_id in molecule_ids}
    existing_names = {path.name for path in args.output_dag_dir.iterdir() if path.is_file() or path.is_symlink()}
    extras = sorted(existing_names - expected_names)
    if extras:
        raise RuntimeError(f"Formal DAG directory contains unexpected files: {extras[:20]}")

    for mol_id in molecule_ids:
        source = source_files[mol_id]
        materialize(source, args.output_dag_dir / source.name, args.mode)

    lines = []
    for name in sorted(expected_names):
        path = args.output_dag_dir / name
        lines.append(f"{sha256_file(path)}  {name}\n")
    manifest_sha = hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()
    expected_sha = contract["manifest_sha256"]
    if manifest_sha != expected_sha:
        raise RuntimeError(f"Formal DAG manifest SHA-256 mismatch: {manifest_sha} != {expected_sha}")

    audit = {
        "source_dag_dir": str(args.source_dag_dir.resolve()),
        "output_dag_dir": str(args.output_dag_dir.resolve()),
        "mode": args.mode,
        "molecules": len(molecule_ids),
        "files": len(lines),
        "manifest_sha256": manifest_sha,
    }
    audit_path = args.output_dag_dir.parent / "formal_dag_cache_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
