#!/usr/bin/env python3
"""Build FIORA input/reference CSVs from a licensed local processed cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_ids(split_dir: Path, name: str) -> set[int]:
    frame = pd.read_csv(split_dir / f"{name}_ids.csv")
    for column in ("spec_id", "id"):
        if column in frame.columns:
            return set(frame[column].astype(int))
    return set(frame.iloc[:, 0].astype(int))


def clean_peaks(peaks) -> list[tuple[float, float]]:
    output = []
    for mz, intensity in list(peaks):
        try:
            mz = float(mz)
            intensity = float(intensity)
        except (TypeError, ValueError):
            continue
        if np.isfinite(mz) and np.isfinite(intensity) and intensity > 0:
            output.append((mz, intensity))
    return output


def peaks_json(peaks) -> str:
    values = clean_peaks(peaks)
    return json.dumps(
        {
            "mz": [value[0] for value in values],
            "intensity": [value[1] for value in values],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    spec = pd.read_pickle(args.processed_dir / "spec_df.pkl")
    mol = pd.read_pickle(args.processed_dir / "mol_df.pkl")

    split_map = {}
    for split in ("train", "val", "test"):
        split_map.update(
            {spec_id: split for spec_id in read_ids(args.split_dir, split)}
        )

    smiles_col = next(
        (
            name
            for name in ("smiles", "SMILES", "smi", "can_smiles", "canonical_smiles")
            if name in mol.columns
        ),
        None,
    )
    if smiles_col is None:
        raise ValueError("Cannot find a SMILES column in mol_df.pkl")

    ce_col = next(
        (name for name in ("ace", "CE", "collision_energy") if name in spec.columns),
        None,
    )
    if ce_col is None:
        raise ValueError("Cannot find an ACE/CE column in spec_df.pkl")

    mol_small = mol[["mol_id", smiles_col]].rename(columns={smiles_col: "SMILES"})
    frame = spec.merge(mol_small, on="mol_id", how="left")
    frame["datasplit"] = frame["spec_id"].astype(int).map(split_map)
    frame = frame[frame["datasplit"].isin(("train", "val", "test"))].copy()

    def summary_json(row) -> str:
        return json.dumps(
            {
                "Name": str(int(row["spec_id"])),
                "Precursor_type": "[M+H]+",
                "CE": float(row[ce_col]),
                "Instrument_type": "QTOF",
                "collision_energy": float(row[ce_col]),
                "precursor_mode": "[M+H]+",
                "instrument": "QTOF",
            }
        )

    output = pd.DataFrame(
        {
            "Name": frame["spec_id"].astype(int).astype(str),
            "SMILES": frame["SMILES"].astype(str),
            "Precursor_type": "[M+H]+",
            "CE": frame[ce_col].astype(float),
            "Instrument_type": "QTOF",
            "collision_energy": frame[ce_col].astype(float),
            "precursor_mode": "[M+H]+",
            "instrument": "QTOF",
            "peaks": frame["peaks"].apply(peaks_json),
            "summary": frame.apply(summary_json, axis=1),
            "datasplit": frame["datasplit"],
            "mol_id": frame["mol_id"].astype(int),
            "spec_id": frame["spec_id"].astype(int),
        }
    ).dropna()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print("FIORA_INPUTS_WRITTEN", args.output, len(output))
    print(output["datasplit"].value_counts().to_string())


if __name__ == "__main__":
    main()
