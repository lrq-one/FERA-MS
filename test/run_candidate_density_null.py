from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]

SOURCE_PATH = (
    ROOT
    / "test/run_candidate_space_coverage.py"
)

OUT_DIR = (
    ROOT
    / "runs/experiments/reviewer_stage2/"
      "candidate_density_null"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

NULL_REPLICATES = 100
RANDOM_SEED = 20260729


def load_source_module():
    spec = importlib.util.spec_from_file_location(
        "coverage_source",
        SOURCE_PATH,
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            f"Cannot import {SOURCE_PATH}"
        )

    module = importlib.util.module_from_spec(
        spec
    )
    spec.loader.exec_module(module)

    return module


def local_candidate_counts(
    experimental_masses,
    candidate_masses,
    tolerance,
):
    experimental_masses = np.asarray(
        experimental_masses,
        dtype=float,
    )

    candidate_masses = np.sort(
        np.asarray(
            candidate_masses,
            dtype=float,
        )
    )

    left = np.searchsorted(
        candidate_masses,
        experimental_masses - tolerance,
        side="left",
    )

    right = np.searchsorted(
        candidate_masses,
        experimental_masses + tolerance,
        side="right",
    )

    return right - left


def summarize_distribution(
    values,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    return {
        "mean":
            float(np.mean(values)),
        "sd":
            float(np.std(
                values,
                ddof=1,
            )),
        "minimum":
            float(np.min(values)),
        "p25":
            float(np.quantile(
                values,
                0.25,
            )),
        "median":
            float(np.median(values)),
        "p75":
            float(np.quantile(
                values,
                0.75,
            )),
        "p95":
            float(np.quantile(
                values,
                0.95,
            )),
        "maximum":
            float(np.max(values)),
    }


def main():
    source = load_source_module()

    spec_df = pd.read_pickle(
        source.SPEC_FP
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    molecule_rows = []
    spectrum_rows = []
    null_rows = []

    for (
        split_name,
        test_ids_path,
    ) in source.SPLITS.items():
        test_ids = pd.read_csv(
            test_ids_path
        )

        selected = spec_df[
            spec_df["spec_id"].isin(
                test_ids["spec_id"]
            )
        ].copy()

        selected = selected.sort_values(
            ["mol_id", "spec_id"]
        )

        null_peak_numerators = np.zeros(
            NULL_REPLICATES,
            dtype=float,
        )
        null_peak_denominators = np.zeros(
            NULL_REPLICATES,
            dtype=float,
        )
        null_intensity_numerators = np.zeros(
            NULL_REPLICATES,
            dtype=float,
        )
        null_intensity_denominators = np.zeros(
            NULL_REPLICATES,
            dtype=float,
        )

        observed_peak_numerator = 0.0
        observed_peak_denominator = 0.0
        observed_intensity_numerator = 0.0
        observed_intensity_denominator = 0.0

        molecule_groups = selected.groupby(
            "mol_id",
            sort=True,
        )

        for (
            mol_id,
            molecule_spectra,
        ) in tqdm(
            molecule_groups,
            total=selected[
                "mol_id"
            ].nunique(),
            desc=f"{split_name} density/null",
        ):
            cache = source.load_fragment_cache(
                mol_id
            )

            candidate_sets = (
                source.build_candidate_sets(
                    cache
                )
            )

            precursor_reference = float(
                molecule_spectra[
                    "prec_mz"
                ].median()
            )

            for (
                variant_name,
                masses,
            ) in candidate_sets.items():
                count = int(
                    len(masses)
                )

                density = (
                    100.0
                    * count
                    / precursor_reference
                    if precursor_reference > 0
                    else np.nan
                )

                molecule_rows.append({
                    "split":
                        split_name,
                    "mol_id":
                        int(mol_id),
                    "variant":
                        variant_name,
                    "candidate_count":
                        count,
                    "precursor_mz":
                        precursor_reference,
                    "candidates_per_100_Da":
                        density,
                })

            first_spectrum = next(
                molecule_spectra.itertuples(
                    index=False
                )
            )

            precursor_type = str(
                first_spectrum.prec_type
            )

            adduct_shift = float(
                source.PREC_TYPE_TO_MASS_DIFF[
                    precursor_type
                ]
            )

            full_candidates = (
                candidate_sets[
                    "D3+H-transfer+NL_support"
                ]
                + adduct_shift
            )

            all_experimental_masses = []
            cleaned_spectra = []

            for spectrum in (
                molecule_spectra
                .itertuples(index=False)
            ):
                (
                    experimental_masses,
                    experimental_intensities,
                ) = source.clean_experimental_peaks(
                    spectrum.peaks
                )

                cleaned_spectra.append((
                    spectrum,
                    experimental_masses,
                    experimental_intensities,
                ))

                if len(experimental_masses):
                    all_experimental_masses.append(
                        experimental_masses
                    )

            maxima = [
                precursor_reference,
                float(
                    np.max(full_candidates)
                )
                if len(full_candidates)
                else precursor_reference,
            ]

            if all_experimental_masses:
                maxima.append(
                    float(
                        np.max(
                            np.concatenate(
                                all_experimental_masses
                            )
                        )
                    )
                )

            null_width = max(
                maxima
            ) + source.TOLERANCE_DA

            offsets = rng.uniform(
                0.0,
                null_width,
                size=NULL_REPLICATES,
            )

            shifted_candidate_sets = [
                np.sort(
                    np.mod(
                        full_candidates
                        + offset,
                        null_width,
                    )
                )
                for offset in offsets
            ]

            for (
                spectrum,
                experimental_masses,
                experimental_intensities,
            ) in cleaned_spectra:
                local_counts = (
                    local_candidate_counts(
                        experimental_masses,
                        full_candidates,
                        source.TOLERANCE_DA,
                    )
                )

                observed_matched = (
                    local_counts > 0
                )

                spectrum_rows.append({
                    "split":
                        split_name,
                    "spec_id":
                        int(spectrum.spec_id),
                    "mol_id":
                        int(spectrum.mol_id),
                    "ace":
                        float(spectrum.ace),
                    "experimental_peaks":
                        int(
                            len(
                                experimental_masses
                            )
                        ),
                    "full_candidate_count":
                        int(
                            len(
                                full_candidates
                            )
                        ),
                    "mean_candidates_within_0p01_Da":
                        float(
                            local_counts.mean()
                        )
                        if len(local_counts)
                        else np.nan,
                    "median_candidates_within_0p01_Da":
                        float(
                            np.median(
                                local_counts
                            )
                        )
                        if len(local_counts)
                        else np.nan,
                    "p95_candidates_within_0p01_Da":
                        float(
                            np.quantile(
                                local_counts,
                                0.95,
                            )
                        )
                        if len(local_counts)
                        else np.nan,
                    "maximum_candidates_within_0p01_Da":
                        int(
                            local_counts.max()
                        )
                        if len(local_counts)
                        else 0,
                    "fraction_peaks_with_multiple_candidates":
                        float(
                            np.mean(
                                local_counts > 1
                            )
                        )
                        if len(local_counts)
                        else np.nan,
                })

                observed_peak_numerator += float(
                    observed_matched.sum()
                )
                observed_peak_denominator += float(
                    len(
                        experimental_masses
                    )
                )
                observed_intensity_numerator += float(
                    experimental_intensities[
                        observed_matched
                    ].sum()
                )
                observed_intensity_denominator += float(
                    experimental_intensities.sum()
                )

                for replicate, shifted in enumerate(
                    shifted_candidate_sets
                ):
                    null_counts = (
                        local_candidate_counts(
                            experimental_masses,
                            shifted,
                            source.TOLERANCE_DA,
                        )
                    )

                    null_matched = (
                        null_counts > 0
                    )

                    null_peak_numerators[
                        replicate
                    ] += float(
                        null_matched.sum()
                    )
                    null_peak_denominators[
                        replicate
                    ] += float(
                        len(
                            experimental_masses
                        )
                    )

                    null_intensity_numerators[
                        replicate
                    ] += float(
                        experimental_intensities[
                            null_matched
                        ].sum()
                    )
                    null_intensity_denominators[
                        replicate
                    ] += float(
                        experimental_intensities.sum()
                    )

        observed_peak_recall = (
            observed_peak_numerator
            / observed_peak_denominator
        )

        observed_explained_intensity = (
            observed_intensity_numerator
            / observed_intensity_denominator
        )

        for replicate in range(
            NULL_REPLICATES
        ):
            null_rows.append({
                "split":
                    split_name,
                "replicate":
                    replicate,
                "observed_peak_recall":
                    observed_peak_recall,
                "observed_explained_intensity":
                    observed_explained_intensity,
                "null_peak_recall":
                    (
                        null_peak_numerators[
                            replicate
                        ]
                        / null_peak_denominators[
                            replicate
                        ]
                    ),
                "null_explained_intensity":
                    (
                        null_intensity_numerators[
                            replicate
                        ]
                        / null_intensity_denominators[
                            replicate
                        ]
                    ),
            })

    molecule_df = pd.DataFrame(
        molecule_rows
    )

    spectrum_df = pd.DataFrame(
        spectrum_rows
    )

    null_df = pd.DataFrame(
        null_rows
    )

    summary_rows = []

    for (
        split_name,
        variant_name,
    ), group in molecule_df.groupby(
        ["split", "variant"],
        sort=True,
    ):
        count_stats = summarize_distribution(
            group["candidate_count"]
        )

        density_stats = summarize_distribution(
            group[
                "candidates_per_100_Da"
            ]
        )

        summary_rows.append({
            "split":
                split_name,
            "variant":
                variant_name,
            "molecules":
                int(len(group)),
            **{
                f"candidate_count_{key}":
                    value
                for key, value
                in count_stats.items()
            },
            **{
                f"density_per_100Da_{key}":
                    value
                for key, value
                in density_stats.items()
            },
        })

    density_summary = pd.DataFrame(
        summary_rows
    )

    null_summary_rows = []

    for (
        split_name,
        group,
    ) in null_df.groupby(
        "split",
        sort=True,
    ):
        null_summary_rows.append({
            "split":
                split_name,
            "replicates":
                len(group),
            "observed_peak_recall":
                float(
                    group[
                        "observed_peak_recall"
                    ].iloc[0]
                ),
            "null_peak_recall_mean":
                float(
                    group[
                        "null_peak_recall"
                    ].mean()
                ),
            "null_peak_recall_p2p5":
                float(
                    group[
                        "null_peak_recall"
                    ].quantile(0.025)
                ),
            "null_peak_recall_p97p5":
                float(
                    group[
                        "null_peak_recall"
                    ].quantile(0.975)
                ),
            "observed_explained_intensity":
                float(
                    group[
                        "observed_explained_intensity"
                    ].iloc[0]
                ),
            "null_explained_intensity_mean":
                float(
                    group[
                        "null_explained_intensity"
                    ].mean()
                ),
            "null_explained_intensity_p2p5":
                float(
                    group[
                        "null_explained_intensity"
                    ].quantile(0.025)
                ),
            "null_explained_intensity_p97p5":
                float(
                    group[
                        "null_explained_intensity"
                    ].quantile(0.975)
                ),
        })

    null_summary = pd.DataFrame(
        null_summary_rows
    )

    molecule_df.to_csv(
        OUT_DIR
        / "candidate_density_per_molecule.csv.gz",
        index=False,
        compression="gzip",
    )

    spectrum_df.to_csv(
        OUT_DIR
        / "candidate_multiplicity_per_spectrum.csv.gz",
        index=False,
        compression="gzip",
    )

    density_summary.to_csv(
        OUT_DIR
        / "candidate_density_summary.csv",
        index=False,
    )

    null_df.to_csv(
        OUT_DIR
        / "same_count_circular_shift_null_replicates.csv",
        index=False,
    )

    null_summary.to_csv(
        OUT_DIR
        / "same_count_circular_shift_null_summary.csv",
        index=False,
    )

    print("\nCANDIDATE DENSITY SUMMARY")
    print(
        density_summary.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.4f}",
        )
    )

    print("\nSAME-COUNT NULL SUMMARY")
    print(
        null_summary.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.6f}",
        )
    )

    print("\nOUTPUT:", OUT_DIR)


if __name__ == "__main__":
    main()
