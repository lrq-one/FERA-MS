#!/usr/bin/env python3
"""Generate the locked spectrum-level QC table from tracked inputs."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


QC_COLUMNS = (
    "n_peaks",
    "max_intensity_frac",
    "entropy_norm",
    "precursor_survival_yield",
    "support_PR_abs006",
    "support_PWR_abs006",
    "true_oos_intensity",
)


def deep_merge(base: dict, patch: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in patch.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_frag_params(bundle_path: Path) -> dict:
    bundle = yaml.safe_load(bundle_path.read_text(encoding="utf-8"))
    config = deep_merge(bundle["template"], bundle["base_stage"])
    params = copy.deepcopy(config["frag_params"])
    params["pyg"] = False
    params["formula_peak_mzs"] = True
    params["formula_peak_probs"] = True
    return params


def normalized_peaks(peaks, mz_max: float) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(peaks, dtype=np.float64)
    if values.size == 0:
        return np.empty(0), np.empty(0)
    values = values.reshape(-1, 2)
    values = values[
        np.isfinite(values).all(axis=1)
        & (values[:, 0] > 0)
        & (values[:, 0] <= mz_max)
        & (values[:, 1] > 0)
    ]
    if len(values) == 0:
        return np.empty(0), np.empty(0)
    intensities = values[:, 1] / values[:, 1].sum()
    return values[:, 0], intensities


def support_recall(
    true_mz: np.ndarray,
    true_intensity: np.ndarray,
    candidate_mz: np.ndarray,
    tolerance: float,
) -> tuple[float, float]:
    candidate_mz = np.asarray(candidate_mz, dtype=np.float64).reshape(-1)
    candidate_mz = np.unique(
        np.round(candidate_mz[np.isfinite(candidate_mz) & (candidate_mz > 0)], 6)
    )
    if len(true_mz) == 0 or len(candidate_mz) == 0:
        return 0.0, 0.0

    matched = np.zeros(len(true_mz), dtype=bool)
    for index, mass in enumerate(true_mz):
        left = np.searchsorted(candidate_mz, mass - tolerance, side="left")
        right = np.searchsorted(candidate_mz, mass + tolerance, side="right")
        matched[index] = right > left
    return float(matched.mean()), float(true_intensity[matched].sum())


def normalized_entropy(intensities: np.ndarray) -> float:
    if len(intensities) <= 1:
        return 0.0
    entropy = -float(np.sum(intensities * np.log(intensities + 1e-12)))
    return entropy / np.log(len(intensities))


def load_eligible_ids(split_dir: Path) -> pd.DataFrame:
    frames = []
    for split in ("train", "val", "test"):
        path = split_dir / f"{split}_ids.csv"
        frame = pd.read_csv(path)[["spec_id", "mol_id", "group_id"]]
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    if result["spec_id"].duplicated().any():
        raise RuntimeError("Eligible split union contains duplicate spec_id values")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-bundle", type=Path, default=Path("config/train.yml"))
    parser.add_argument("--spec-fp", type=Path, required=True)
    parser.add_argument("--frag-dp", type=Path, required=True)
    parser.add_argument("--eligible-split", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mz-max", type=float, default=1500.0)
    parser.add_argument("--tolerance", type=float, default=0.006)
    args = parser.parse_args()

    from ms2spectra.data import SpecMolFragDataset
    from ms2spectra.utils.frag_utils import load_frag_d

    frag_params = load_frag_params(args.config_bundle)
    spectra = pd.read_pickle(args.spec_fp)
    eligible = load_eligible_ids(args.eligible_split)
    spectra = eligible.merge(
        spectra,
        on=["spec_id", "mol_id", "group_id"],
        how="left",
        validate="one_to_one",
    )
    if spectra["peaks"].isna().any():
        raise RuntimeError("Some eligible spectra are missing from spec_df")

    processor = object.__new__(SpecMolFragDataset)
    processor.frag_params = frag_params
    rows = []

    for row in spectra.itertuples(index=False):
        entry = pd.Series(row._asdict())
        true_mz, true_intensity = normalized_peaks(entry["peaks"], args.mz_max)
        fragment = load_frag_d(
            int(entry["mol_id"]),
            str(args.frag_dp),
            bool(frag_params.get("compressed", False)),
        )
        processed = processor._process_frag(fragment, entry)
        candidate_mz = processed["frag_formula_peak_mzs"].detach().cpu().numpy()
        peak_recall, weighted_recall = support_recall(
            true_mz,
            true_intensity,
            candidate_mz,
            args.tolerance,
        )
        precursor_yield = float(
            true_intensity[
                np.abs(true_mz - float(entry["prec_mz"])) <= args.tolerance
            ].sum()
        )
        rows.append(
            {
                "spec_id": int(entry["spec_id"]),
                "mol_id": int(entry["mol_id"]),
                "group_id": int(entry["group_id"]),
                "n_peaks": int(len(true_mz)),
                "max_intensity_frac": float(true_intensity.max()) if len(true_intensity) else 0.0,
                "entropy_norm": normalized_entropy(true_intensity),
                "precursor_survival_yield": precursor_yield,
                "support_PR_abs006": peak_recall,
                "support_PWR_abs006": weighted_recall,
                "true_oos_intensity": 1.0 - weighted_recall,
            }
        )

    output = pd.DataFrame(rows)
    if output[list(QC_COLUMNS)].isna().any().any():
        raise RuntimeError("Generated QC table contains missing values")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    output.sort_values("spec_id").to_csv(args.out, index=False)
    print(f"wrote {len(output)} QC rows to {args.out}")


if __name__ == "__main__":
    main()
