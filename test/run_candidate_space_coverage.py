from __future__ import annotations

import bz2
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ms2spectra.utils.frag_utils import get_node_feats
from ms2spectra.utils.formula_utils import PREC_TYPE_TO_MASS_DIFF


ROOT = Path.cwd()

SPEC_FP = (
    ROOT
    / "data/proc/nist20_qtof_cid_safe19659/spec_df.pkl"
)

FRAG_DP = (
    ROOT
    / "data/frag/"
      "nist20_qtof_cid_safe19659_d3_mhp_qtof_cid_nl_v1/dags"
)

SPLITS = {
    "random": (
        ROOT
        / "data/split/"
          "nist20_qtof_cid_safe19659_qcv1_trainonly/test_ids.csv"
    ),
    "scaffold": (
        ROOT
        / "data/split/"
          "nist20_qtof_cid_safe19659_scaffold60_20_20_seed42/"
          "test_ids.csv"
    ),
}

EXPECTED_SPECTRA = {
    "random": 3931,
    "scaffold": 3960,
}

OUT_DIR = (
    ROOT
    / "runs/experiments/"
      "candidate_space_coverage_d1_d3_h_nl"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOLERANCE_DA = 0.01
NUM_ISOTOPES = 1
NL_MAX_PEAKS = 3

VARIANTS = [
    ("D1_only", 1, False, False),
    ("D2_only", 2, False, False),
    ("D3_only", 3, False, False),
    ("D3+H-transfer", 3, True, False),
    ("D3+H-transfer+NL_support", 3, True, True),
]

VARIANT_ORDER = [
    item[0]
    for item in VARIANTS
]


def as_numpy(value, dtype=None):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()

    array = np.asarray(value)

    if dtype is not None:
        array = array.astype(
            dtype,
            copy=False,
        )

    return array


def normalize_mol_id(value):
    numeric = float(value)

    if numeric.is_integer():
        return str(int(numeric))

    return str(value)


def load_fragment_cache(mol_id):
    stem = normalize_mol_id(mol_id)

    compressed_path = (
        FRAG_DP
        / f"{stem}.pickle.bz2"
    )

    plain_path = (
        FRAG_DP
        / f"{stem}.pickle"
    )

    if compressed_path.is_file():
        with bz2.BZ2File(
            compressed_path,
            "rb",
        ) as handle:
            return pickle.load(handle)

    if plain_path.is_file():
        with plain_path.open(
            "rb",
        ) as handle:
            return pickle.load(handle)

    raise FileNotFoundError(
        f"Missing fragment cache for mol_id={mol_id}: "
        f"{compressed_path}"
    )


def get_formula_indices(
    fragment_cache,
    depth_limit,
    include_all_h,
):
    dag = fragment_cache["dag"]

    node_feature_indices = (
        dag.node_feat_idxs.reshape(-1)
    )

    h_formula_indices = get_node_feats(
        dag.x,
        node_feature_indices,
        "h_formulae_idx",
    )

    h_formula_indices = as_numpy(
        h_formula_indices,
        np.int64,
    )

    node_depths = as_numpy(
        fragment_cache["nodes_min_depth"],
        np.int64,
    ).reshape(-1)

    if (
        h_formula_indices.shape[0]
        != node_depths.shape[0]
    ):
        raise RuntimeError(
            "Node depth and formula-index "
            "dimensions do not match: "
            f"{node_depths.shape} versus "
            f"{h_formula_indices.shape}"
        )

    selected_nodes = (
        node_depths
        <= int(depth_limit)
    )

    if not np.any(selected_nodes):
        return np.empty(
            0,
            dtype=np.int64,
        )

    formula_indices_by_h = (
        fragment_cache["idx_by_h_delta"]
    )

    if include_all_h:
        selected_channels = range(
            h_formula_indices.shape[1]
        )
    else:
        # h_formulae_idx column 0 corresponds
        # to the base ΔH=0 formula.
        selected_channels = (0,)

    selected_formula_indices = set()

    for channel in selected_channels:
        if channel >= len(
            formula_indices_by_h
        ):
            continue

        formula_indices_from_nodes = set(
            int(index)
            for index
            in h_formula_indices[
                selected_nodes,
                channel,
            ].reshape(-1)
        )

        valid_formula_indices = set(
            int(index)
            for index
            in formula_indices_by_h[channel]
        )

        selected_formula_indices.update(
            formula_indices_from_nodes
            & valid_formula_indices
        )

    return np.asarray(
        sorted(selected_formula_indices),
        dtype=np.int64,
    )


def extract_positive_masses(
    mass_matrix,
    probability_matrix,
):
    mass_matrix = as_numpy(
        mass_matrix,
        np.float64,
    )

    probability_matrix = as_numpy(
        probability_matrix,
        np.float64,
    )

    if (
        mass_matrix.shape
        != probability_matrix.shape
    ):
        raise RuntimeError(
            "Mass and probability shapes differ: "
            f"{mass_matrix.shape} versus "
            f"{probability_matrix.shape}"
        )

    valid = (
        np.isfinite(mass_matrix)
        & np.isfinite(probability_matrix)
        & (mass_matrix > 0.0)
        & (probability_matrix > 0.0)
    )

    masses = mass_matrix[valid]

    if masses.size == 0:
        return np.empty(
            0,
            dtype=np.float64,
        )

    return np.unique(masses)


def build_candidate_sets(
    fragment_cache,
):
    base_mass_matrix = as_numpy(
        fragment_cache[
            "formula_peak_mzs"
        ],
        np.float64,
    )[:, :NUM_ISOTOPES]

    base_probability_matrix = as_numpy(
        fragment_cache[
            "formula_peak_probs"
        ],
        np.float64,
    )[:, :NUM_ISOTOPES]

    output = {}

    for (
        variant_name,
        depth_limit,
        include_all_h,
        include_nl,
    ) in VARIANTS:
        formula_indices = get_formula_indices(
            fragment_cache=fragment_cache,
            depth_limit=depth_limit,
            include_all_h=include_all_h,
        )

        candidate_parts = []

        if formula_indices.size > 0:
            candidate_parts.append(
                extract_positive_masses(
                    base_mass_matrix[
                        formula_indices
                    ],
                    base_probability_matrix[
                        formula_indices
                    ],
                )
            )

        if include_nl:
            required_nl_fields = {
                "nl_formula_peak_mzs",
                "nl_formula_peak_probs",
            }

            missing_nl_fields = (
                required_nl_fields
                - set(fragment_cache)
            )

            if missing_nl_fields:
                raise KeyError(
                    "Locked D3 cache is missing "
                    f"NL fields: "
                    f"{sorted(missing_nl_fields)}"
                )

            nl_mass_matrix = as_numpy(
                fragment_cache[
                    "nl_formula_peak_mzs"
                ],
                np.float64,
            )[
                formula_indices,
                :NL_MAX_PEAKS,
            ]

            nl_probability_matrix = as_numpy(
                fragment_cache[
                    "nl_formula_peak_probs"
                ],
                np.float64,
            )[
                formula_indices,
                :NL_MAX_PEAKS,
            ]

            candidate_parts.append(
                extract_positive_masses(
                    nl_mass_matrix,
                    nl_probability_matrix,
                )
            )

        candidate_parts = [
            part
            for part in candidate_parts
            if part.size > 0
        ]

        if candidate_parts:
            output[variant_name] = (
                np.unique(
                    np.concatenate(
                        candidate_parts
                    )
                )
            )
        else:
            output[variant_name] = (
                np.empty(
                    0,
                    dtype=np.float64,
                )
            )

    return output


def match_experimental_peaks(
    experimental_masses,
    candidate_masses,
):
    experimental_masses = np.asarray(
        experimental_masses,
        dtype=np.float64,
    )

    candidate_masses = np.sort(
        np.asarray(
            candidate_masses,
            dtype=np.float64,
        )
    )

    matched = np.zeros(
        experimental_masses.size,
        dtype=bool,
    )

    if (
        experimental_masses.size == 0
        or candidate_masses.size == 0
    ):
        return matched

    insertion_positions = np.searchsorted(
        candidate_masses,
        experimental_masses,
        side="left",
    )

    right_valid = (
        insertion_positions
        < candidate_masses.size
    )

    right_rows = np.where(
        right_valid
    )[0]

    matched[right_rows] |= (
        np.abs(
            candidate_masses[
                insertion_positions[
                    right_rows
                ]
            ]
            - experimental_masses[
                right_rows
            ]
        )
        < TOLERANCE_DA
    )

    left_valid = (
        insertion_positions
        > 0
    )

    left_rows = np.where(
        left_valid
    )[0]

    matched[left_rows] |= (
        np.abs(
            candidate_masses[
                insertion_positions[
                    left_rows
                ]
                - 1
            ]
            - experimental_masses[
                left_rows
            ]
        )
        < TOLERANCE_DA
    )

    return matched


def clean_experimental_peaks(peaks):
    peak_array = np.asarray(
        peaks,
        dtype=np.float64,
    )

    if (
        peak_array.ndim != 2
        or peak_array.shape[1] < 2
    ):
        raise ValueError(
            f"Invalid peak array shape: "
            f"{peak_array.shape}"
        )

    masses = peak_array[:, 0]
    intensities = peak_array[:, 1]

    valid = (
        np.isfinite(masses)
        & np.isfinite(intensities)
        & (masses > 0.0)
        & (intensities >= 0.0)
    )

    return (
        masses[valid],
        intensities[valid],
    )


def evaluate_split(
    split_name,
    test_ids_path,
    spectrum_dataframe,
):
    test_ids = pd.read_csv(
        test_ids_path
    )

    selected_spectra = spectrum_dataframe[
        spectrum_dataframe[
            "spec_id"
        ].isin(
            test_ids["spec_id"]
        )
    ].copy()

    selected_spectra = (
        selected_spectra
        .sort_values(
            ["mol_id", "spec_id"]
        )
        .reset_index(drop=True)
    )

    expected_count = (
        EXPECTED_SPECTRA[
            split_name
        ]
    )

    if (
        len(selected_spectra)
        != expected_count
    ):
        raise RuntimeError(
            f"{split_name}: expected "
            f"{expected_count} spectra, "
            f"found {len(selected_spectra)}"
        )

    print(
        f"\n[{split_name}] "
        f"spectra={len(selected_spectra)}, "
        f"molecules="
        f"{selected_spectra['mol_id'].nunique()}",
        flush=True,
    )

    result_rows = []

    molecule_groups = (
        selected_spectra
        .groupby(
            "mol_id",
            sort=True,
        )
    )

    for (
        mol_id,
        molecule_spectra,
    ) in tqdm(
        molecule_groups,
        total=selected_spectra[
            "mol_id"
        ].nunique(),
        desc=f"{split_name} molecules",
    ):
        fragment_cache = (
            load_fragment_cache(
                mol_id
            )
        )

        unshifted_candidate_sets = (
            build_candidate_sets(
                fragment_cache
            )
        )

        for spectrum in (
            molecule_spectra
            .itertuples(index=False)
        ):
            (
                experimental_masses,
                experimental_intensities,
            ) = clean_experimental_peaks(
                spectrum.peaks
            )

            precursor_type = str(
                spectrum.prec_type
            )

            if (
                precursor_type
                not in PREC_TYPE_TO_MASS_DIFF
            ):
                raise KeyError(
                    f"Unknown precursor type: "
                    f"{precursor_type}"
                )

            adduct_mass_shift = float(
                PREC_TYPE_TO_MASS_DIFF[
                    precursor_type
                ]
            )

            precursor_mass = float(
                spectrum.prec_mz
            )

            precursor_excluded_mask = (
                np.abs(
                    experimental_masses
                    - precursor_mass
                )
                >= TOLERANCE_DA
            )

            top_count = min(
                10,
                experimental_masses.size,
            )

            top_indices = np.argsort(
                experimental_intensities,
                kind="stable",
            )[-top_count:]

            spectrum_covered_counts = []

            for variant_name in VARIANT_ORDER:
                candidate_masses = (
                    unshifted_candidate_sets[
                        variant_name
                    ]
                    + adduct_mass_shift
                )

                matched = (
                    match_experimental_peaks(
                        experimental_masses,
                        candidate_masses,
                    )
                )

                spectrum_covered_counts.append(
                    int(matched.sum())
                )

                result_rows.append(
                    {
                        "split":
                            split_name,
                        "spec_id":
                            int(
                                spectrum.spec_id
                            ),
                        "mol_id":
                            int(
                                spectrum.mol_id
                            ),
                        "ace":
                            float(
                                spectrum.ace
                            ),
                        "variant":
                            variant_name,
                        "n_experimental_peaks":
                            int(
                                experimental_masses.size
                            ),
                        "n_covered_peaks":
                            int(
                                matched.sum()
                            ),
                        "total_experimental_intensity":
                            float(
                                experimental_intensities.sum()
                            ),
                        "covered_experimental_intensity":
                            float(
                                experimental_intensities[
                                    matched
                                ].sum()
                            ),
                        "n_top10_peaks":
                            int(
                                top_count
                            ),
                        "n_top10_covered":
                            int(
                                matched[
                                    top_indices
                                ].sum()
                            ),
                        "n_precursor_excluded_peaks":
                            int(
                                precursor_excluded_mask.sum()
                            ),
                        "n_precursor_excluded_covered":
                            int(
                                (
                                    matched
                                    & precursor_excluded_mask
                                ).sum()
                            ),
                    }
                )

            if any(
                later < earlier
                for earlier, later
                in zip(
                    spectrum_covered_counts,
                    spectrum_covered_counts[1:],
                )
            ):
                raise RuntimeError(
                    "Coverage is not monotonic: "
                    f"split={split_name}, "
                    f"spec_id={spectrum.spec_id}, "
                    f"counts="
                    f"{spectrum_covered_counts}"
                )

    return result_rows


def safe_ratio(
    numerator,
    denominator,
):
    if denominator <= 0:
        return np.nan

    return float(
        numerator
        / denominator
    )


def summarize_results(
    per_spectrum_dataframe,
):
    summary_rows = []

    grouped = (
        per_spectrum_dataframe
        .groupby(
            ["split", "variant"],
            sort=False,
        )
    )

    for (
        split_name,
        variant_name,
    ), group in grouped:
        summary_rows.append(
            {
                "split":
                    split_name,
                "variant":
                    variant_name,
                "spectra":
                    int(
                        group[
                            "spec_id"
                        ].nunique()
                    ),
                "molecules":
                    int(
                        group[
                            "mol_id"
                        ].nunique()
                    ),
                "peak_recall":
                    safe_ratio(
                        group[
                            "n_covered_peaks"
                        ].sum(),
                        group[
                            "n_experimental_peaks"
                        ].sum(),
                    ),
                "explained_intensity":
                    safe_ratio(
                        group[
                            "covered_experimental_intensity"
                        ].sum(),
                        group[
                            "total_experimental_intensity"
                        ].sum(),
                    ),
                "top10_peak_recall":
                    safe_ratio(
                        group[
                            "n_top10_covered"
                        ].sum(),
                        group[
                            "n_top10_peaks"
                        ].sum(),
                    ),
                "precursor_excluded_peak_recall":
                    safe_ratio(
                        group[
                            "n_precursor_excluded_covered"
                        ].sum(),
                        group[
                            "n_precursor_excluded_peaks"
                        ].sum(),
                    ),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary["variant"] = (
        pd.Categorical(
            summary["variant"],
            categories=VARIANT_ORDER,
            ordered=True,
        )
    )

    return (
        summary
        .sort_values(
            ["split", "variant"]
        )
        .reset_index(drop=True)
    )


def main():
    required_paths = [
        SPEC_FP,
        FRAG_DP,
        *SPLITS.values(),
    ]

    for required_path in required_paths:
        if not required_path.exists():
            raise FileNotFoundError(
                required_path
            )

    print("=" * 110)
    print(
        "CANDIDATE-SPACE COVERAGE: "
        "D1 / D2 / D3 / H-TRANSFER / NL"
    )
    print("=" * 110)
    print("spec_df :", SPEC_FP)
    print("frag_dp :", FRAG_DP)
    print("tolerance:", TOLERANCE_DA, "Da")
    print("variants :", VARIANT_ORDER)
    print("output   :", OUT_DIR)
    print("=" * 110)

    spectrum_dataframe = (
        pd.read_pickle(
            SPEC_FP
        )
    )

    all_result_rows = []

    for (
        split_name,
        test_ids_path,
    ) in SPLITS.items():
        all_result_rows.extend(
            evaluate_split(
                split_name=split_name,
                test_ids_path=test_ids_path,
                spectrum_dataframe=spectrum_dataframe,
            )
        )

    per_spectrum = pd.DataFrame(
        all_result_rows
    )

    summary = summarize_results(
        per_spectrum
    )

    per_spectrum_path = (
        OUT_DIR
        / "candidate_space_coverage_per_spectrum.csv.gz"
    )

    summary_path = (
        OUT_DIR
        / "candidate_space_coverage_summary.csv"
    )

    per_spectrum.to_csv(
        per_spectrum_path,
        index=False,
        compression="gzip",
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    displayed_summary = (
        summary.copy()
    )

    metric_columns = [
        "peak_recall",
        "explained_intensity",
        "top10_peak_recall",
        "precursor_excluded_peak_recall",
    ]

    for column in metric_columns:
        displayed_summary[column] = (
            100.0
            * displayed_summary[column]
        ).map(
            lambda value:
                f"{value:.3f}%"
        )

    print()
    print("=" * 110)
    print("FINAL COVERAGE SUMMARY")
    print("=" * 110)
    print(
        displayed_summary.to_string(
            index=False
        )
    )
    print("=" * 110)
    print(
        "per-spectrum:",
        per_spectrum_path,
    )
    print(
        "summary     :",
        summary_path,
    )
    print(
        "CANDIDATE_SPACE_COVERAGE_COMPLETE"
    )


if __name__ == "__main__":
    main()
