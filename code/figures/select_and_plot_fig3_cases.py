#!/usr/bin/env python3

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from matplotlib.offsetbox import (
    AnnotationBbox,
    OffsetImage,
)

from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]

for candidate in (
    ROOT / "code" / "src",
    ROOT / "code",
    ROOT,
):
    value = str(candidate)

    if value not in sys.path:
        sys.path.insert(0, value)


CORE_PATH = (
    Path(__file__).resolve().parent
    / "select_and_plot_fig3_cases_core.py"
)

spec = importlib.util.spec_from_file_location(
    "fig3_core",
    str(CORE_PATH),
)

if spec is None or spec.loader is None:
    raise RuntimeError(
        f"Cannot load core module: {CORE_PATH}"
    )

core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


# Plot colours matched to the manuscript palette.
OBSERVED_BLUE = "#2F80C3"
PREDICTED_ORANGE = "#F28E2B"
INK = "#20262E"
MUTED = "#667085"
GRID = "#D9DEE5"


BLACKLIST = {
    "Random": {21101},
    "Scaffold": {6626},
}


def rebuild_loader(
    model_pack,
    batch_size,
):
    """
    Rebuild the test DataLoader with a small inference batch.

    This prevents the rich-feature extraction stage from allocating
    the very large tensors that caused the previous CUDA OOM.
    """
    old_loader = model_pack["loader"]

    model_pack["loader"] = DataLoader(
        old_loader.dataset,
        batch_size=max(4, int(batch_size)),
        shuffle=False,
        num_workers=0,
        collate_fn=old_loader.collate_fn,
        pin_memory=False,
        drop_last=False,
    )

    return model_pack


def build_broad_quality_pool(
    scores,
    protocol,
    pool_size,
):
    """
    Keep good but not near-perfect spectra and sample broadly across
    that quality range. Peak richness is evaluated later from the
    actual observed spectra.
    """
    frame = scores.copy()

    frame = frame[
        ~frame["spec_id"]
        .astype(int)
        .isin(BLACKLIST[protocol])
    ].copy()

    lower = frame[
        "consensus_score"
    ].quantile(0.50)

    upper = frame[
        "consensus_score"
    ].quantile(0.965)

    frame = frame[
        (
            frame["consensus_score"]
            >= lower
        )
        &
        (
            frame["consensus_score"]
            <= upper
        )
    ].copy()

    # Prefer useful collision-energy ranges, but do not make CE
    # more important than spectral richness.
    if protocol == "Random":
        preferred = frame[
            (
                frame["ce"] >= 15.0
            )
            &
            (
                frame["ce"] <= 45.0
            )
        ]
    else:
        preferred = frame[
            frame["ce"] >= 20.0
        ]

    if len(preferred) >= min(
        80,
        int(pool_size),
    ):
        frame = preferred

    frame = frame.sort_values(
        [
            "consensus_score",
            "cbin_sd",
            "jss_sd",
        ],
        ascending=[
            False,
            True,
            True,
        ],
    ).reset_index(drop=True)

    if len(frame) > int(pool_size):
        # Take spectra across the full retained quality interval,
        # rather than selecting only the easiest near-perfect cases.
        indices = np.linspace(
            0,
            len(frame) - 1,
            int(pool_size),
        )

        indices = np.unique(
            np.round(indices)
            .astype(int)
        )

        frame = frame.iloc[
            indices
        ].copy()

    return frame.reset_index(drop=True)


def peak_statistics(
    case,
):
    true_mz = np.asarray(
        case["true_mz"],
        dtype=float,
    )

    true_intensity = core.normalize_to_100(
        case["true_intensity"]
    )

    keep = (
        np.isfinite(true_mz)
        &
        np.isfinite(true_intensity)
        &
        (true_mz > 0)
        &
        (true_intensity >= 0.15)
    )

    mz = true_mz[keep]
    intensity = true_intensity[keep]

    peak_count = int(len(mz))

    major_peak_count = int(
        np.sum(intensity >= 2.0)
    )

    medium_peak_count = int(
        np.sum(intensity >= 0.75)
    )

    if len(mz) >= 2:
        mz_span = float(
            np.quantile(mz, 0.95)
            -
            np.quantile(mz, 0.05)
        )
    else:
        mz_span = 0.0

    return {
        "display_peak_count":
            peak_count,

        "major_peak_count":
            major_peak_count,

        "medium_peak_count":
            medium_peak_count,

        "mz_span":
            mz_span,
    }


def rank_percentile(
    series,
):
    return series.rank(
        method="average",
        pct=True,
    )


def select_rich_representative(
    pool,
    spectra,
    protocol,
    min_peaks,
    min_major_peaks,
):
    rows = []

    for score_row in pool.itertuples(
        index=False
    ):
        spec_id = int(
            score_row.spec_id
        )

        case = spectra.get(
            spec_id
        )

        if case is None:
            continue

        stats = peak_statistics(
            case
        )

        rows.append({
            **case,
            **stats,

            "cbin_mean":
                float(
                    score_row.cbin_mean
                ),

            "cbin_sd":
                float(
                    score_row.cbin_sd
                ),

            "jss_mean":
                float(
                    score_row.jss_mean
                ),

            "jss_sd":
                float(
                    score_row.jss_sd
                ),

            "consensus_score":
                float(
                    score_row.consensus_score
                ),
        })

    frame = pd.DataFrame(rows)

    if frame.empty:
        raise RuntimeError(
            f"No spectra extracted for {protocol}."
        )

    # Avoid extreme near-perfect examples while retaining genuinely
    # good predictions.
    quality_mask = (
        (frame["cbin_mean"] >= 0.78)
        &
        (frame["cbin_mean"] <= 0.970)
        &
        (frame["jss_mean"] >= 0.65)
    )

    eligible = frame[
        quality_mask
        &
        (
            frame["display_peak_count"]
            >= int(min_peaks)
        )
        &
        (
            frame["major_peak_count"]
            >= int(min_major_peaks)
        )
        &
        (
            frame["mz_span"]
            >= 70.0
        )
    ].copy()

    # The script no longer falls back to sparse 10-peak examples.
    # It may relax only slightly, with a hard floor of 18 peaks.
    if eligible.empty:
        relaxed_min = max(
            18,
            int(min_peaks) - 4,
        )

        print(
            f"[{protocol}] No case met the strict "
            f"{min_peaks}-peak threshold; "
            f"trying the explicit relaxed threshold "
            f"of {relaxed_min} peaks."
        )

        eligible = frame[
            quality_mask
            &
            (
                frame["display_peak_count"]
                >= relaxed_min
            )
            &
            (
                frame["major_peak_count"]
                >= max(
                    6,
                    int(min_major_peaks) - 2,
                )
            )
            &
            (
                frame["mz_span"]
                >= 60.0
            )
        ].copy()

    if eligible.empty:
        best_richness = int(
            frame["display_peak_count"]
            .max()
        )

        raise RuntimeError(
            f"No publication-suitable {protocol} case found. "
            f"Maximum available display peak count in the "
            f"examined pool was {best_richness}. "
            f"Increase --pool_size rather than accepting a "
            f"sparse example."
        )

    richness = rank_percentile(
        eligible["display_peak_count"]
    )

    major = rank_percentile(
        eligible["major_peak_count"]
    )

    span = rank_percentile(
        eligible["mz_span"]
    )

    quality = rank_percentile(
        eligible["consensus_score"]
    )

    instability = (
        0.5
        * rank_percentile(
            eligible["cbin_sd"]
        )
        +
        0.5
        * rank_percentile(
            eligible["jss_sd"]
        )
    )

    eligible["selection_score"] = (
        0.42 * richness
        + 0.18 * major
        + 0.12 * span
        + 0.20 * quality
        + 0.08 * (1.0 - instability)
    )

    selected = eligible.sort_values(
        [
            "selection_score",
            "display_peak_count",
            "major_peak_count",
            "consensus_score",
        ],
        ascending=[
            False,
            False,
            False,
            False,
        ],
    ).iloc[0]

    return selected.to_dict()


def square_root_relative(
    intensity,
):
    relative = (
        core.normalize_to_100(
            intensity
        )
        / 100.0
    )

    return np.sqrt(
        np.clip(
            relative,
            0.0,
            None,
        )
    )


def annotate_strong_peaks(
    axis,
    mz,
    height,
    positive,
    color,
    top_n=6,
):
    if len(mz) == 0:
        return

    indices = np.argsort(
        height
    )[::-1]

    indices = [
        index
        for index in indices
        if height[index] >= 0.22
    ][:int(top_n)]

    for index in indices:
        value = (
            height[index]
            if positive
            else -height[index]
        )

        offset = (
            0.025
            if positive
            else -0.025
        )

        axis.text(
            mz[index],
            value + offset,
            f"{mz[index]:.1f}",
            rotation=90,
            ha="center",
            va=(
                "bottom"
                if positive
                else "top"
            ),
            fontsize=6.3,
            color=color,
        )


def add_structure_in_negative_half(
    axis,
    case,
):
    structure = core.draw_structure(
        case.get("smiles")
    )

    if structure is None:
        return

    image = OffsetImage(
        np.asarray(structure),
        zoom=0.27,
    )

    image.set_alpha(0.96)

    box = AnnotationBbox(
        image,
        (0.22, 0.23),
        xycoords="axes fraction",
        frameon=True,
        pad=0.10,
        bboxprops={
            "facecolor":
                "white",

            "edgecolor":
                "none",

            "alpha":
                0.90,
        },
        zorder=5,
    )

    axis.add_artist(box)


def plot_mirror_case(
    axis,
    case,
    panel,
    protocol,
):
    true_mz = np.asarray(
        case["true_mz"],
        dtype=float,
    )

    true_percent = core.normalize_to_100(
        case["true_intensity"]
    )

    pred_mz = np.asarray(
        case["pred_mz"],
        dtype=float,
    )

    pred_percent = core.normalize_to_100(
        case["pred_intensity"]
    )

    true_keep = (
        np.isfinite(true_mz)
        &
        np.isfinite(true_percent)
        &
        (true_mz > 0)
        &
        (true_percent >= 0.15)
    )

    pred_keep = (
        np.isfinite(pred_mz)
        &
        np.isfinite(pred_percent)
        &
        (pred_mz > 0)
        &
        (pred_percent >= 0.15)
    )

    true_mz = true_mz[
        true_keep
    ]

    true_height = square_root_relative(
        true_percent[
            true_keep
        ]
    )

    pred_mz = pred_mz[
        pred_keep
    ]

    pred_height = square_root_relative(
        pred_percent[
            pred_keep
        ]
    )

    axis.axhline(
        0,
        color="#707984",
        linewidth=0.8,
        zorder=1,
    )

    axis.vlines(
        true_mz,
        0,
        true_height,
        color=OBSERVED_BLUE,
        linewidth=1.05,
        zorder=3,
    )

    axis.vlines(
        pred_mz,
        0,
        -pred_height,
        color=PREDICTED_ORANGE,
        linewidth=1.05,
        zorder=3,
    )

    annotate_strong_peaks(
        axis,
        true_mz,
        true_height,
        positive=True,
        color=OBSERVED_BLUE,
    )

    annotate_strong_peaks(
        axis,
        pred_mz,
        pred_height,
        positive=False,
        color=PREDICTED_ORANGE,
    )

    precursor = core.safe_float(
        case.get("precursor_mz")
    )

    maxima = [100.0]

    if len(true_mz):
        maxima.append(
            float(true_mz.max())
        )

    if len(pred_mz):
        maxima.append(
            float(pred_mz.max())
        )

    if np.isfinite(precursor):
        maxima.append(
            float(precursor)
        )

    axis.set_xlim(
        -12,
        max(maxima) * 1.04,
    )

    axis.set_ylim(
        -1.08,
        1.08,
    )

    axis.set_yticks(
        [
            -1.0,
            -0.5,
            0.0,
            0.5,
            1.0,
        ]
    )

    axis.set_yticklabels(
        [
            "1.0",
            "0.5",
            "0",
            "0.5",
            "1.0",
        ]
    )

    axis.set_ylabel(
        "Square root of\nrelative intensity",
        fontsize=9,
    )

    axis.set_xlabel(
        "Mass/Charge (m/z)",
        fontsize=9,
    )

    axis.grid(
        axis="y",
        color=GRID,
        linewidth=0.55,
        alpha=0.55,
    )

    for spine in axis.spines.values():
        spine.set_color(
            "#707984"
        )

        spine.set_linewidth(
            0.8
        )

    axis.tick_params(
        labelsize=8,
    )

    axis.text(
        -0.095,
        1.02,
        f"({panel})",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color=INK,
    )

    axis.text(
        0.00,
        1.02,
        f"{protocol} test example",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
        color=INK,
    )

    metadata = (
        f"ACE {case['ce']:.1f} eV"
        f"   |   CBIN {case['seed42_cbin']:.3f}"
        f"   |   JSS {case['seed42_jss']:.3f}"
        f"   |   observed peaks "
        f"{case['display_peak_count']}"
    )

    axis.text(
        0.99,
        1.02,
        metadata,
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        color=MUTED,
    )

    axis.text(
        0.012,
        0.955,
        "Observed",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        fontweight="bold",
        color=OBSERVED_BLUE,
    )

    axis.text(
        0.012,
        0.045,
        "FERA-MS",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        color=PREDICTED_ORANGE,
    )

    add_structure_in_negative_half(
        axis,
        case,
    )


def save_case_table(
    cases,
    output_path,
):
    rows = []

    for case in cases:
        rows.append({
            "protocol":
                case["protocol"],

            "spec_id":
                case["spec_id"],

            "mol_id":
                case.get(
                    "mol_id",
                    "",
                ),

            "ce":
                case["ce"],

            "precursor_mz":
                case.get(
                    "precursor_mz",
                    "",
                ),

            "seed42_cbin":
                case["seed42_cbin"],

            "seed42_jss":
                case["seed42_jss"],

            "three_seed_cbin_mean":
                case["cbin_mean"],

            "three_seed_cbin_sd":
                case["cbin_sd"],

            "three_seed_jss_mean":
                case["jss_mean"],

            "three_seed_jss_sd":
                case["jss_sd"],

            "display_peak_count":
                case["display_peak_count"],

            "major_peak_count":
                case["major_peak_count"],

            "mz_span":
                case["mz_span"],

            "smiles":
                case.get(
                    "smiles",
                    "",
                ),

            "formula":
                case.get(
                    "formula",
                    "",
                ),
        })

    pd.DataFrame(
        rows
    ).to_csv(
        output_path,
        index=False,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pool_size",
        type=int,
        default=192,
    )

    parser.add_argument(
        "--min_peaks",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--min_major_peaks",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--inference_batch_size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--output_prefix",
        default=(
            "Fig3_representative_"
            "rich_cases"
        ),
    )

    args = parser.parse_args()

    output_directory = (
        ROOT / "figure"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected_cases = []

    for protocol, directory in (
        core.PROTOCOLS.items()
    ):
        print()
        print("=" * 72)
        print(
            protocol,
            "rich representative-case selection",
        )
        print("=" * 72)

        scores = core.load_three_seed_scores(
            directory
        )

        pool = build_broad_quality_pool(
            scores,
            protocol,
            args.pool_size,
        )

        print(
            "test spectra:",
            len(scores),
        )

        print(
            "broad candidate pool:",
            len(pool),
        )

        print(
            "consensus range:",
            f"{pool['consensus_score'].min():.4f}",
            "to",
            f"{pool['consensus_score'].max():.4f}",
        )

        model_pack = core.load_seed42_model(
            directory
        )

        model_pack = rebuild_loader(
            model_pack,
            args.inference_batch_size,
        )

        wanted_ids = set(
            pool["spec_id"]
            .astype(int)
            .tolist()
        )

        spectra = core.collect_spectra(
            model_pack,
            wanted_ids,
        )

        selected = select_rich_representative(
            pool,
            spectra,
            protocol,
            args.min_peaks,
            args.min_major_peaks,
        )

        selected["protocol"] = protocol

        selected = core.add_metadata(
            selected
        )

        selected_cases.append(
            selected
        )

        print(
            "selected spec_id:",
            selected["spec_id"],
        )

        print(
            "ACE:",
            selected["ce"],
        )

        print(
            "display peaks:",
            selected[
                "display_peak_count"
            ],
        )

        print(
            "major peaks:",
            selected[
                "major_peak_count"
            ],
        )

        print(
            "m/z span:",
            f"{selected['mz_span']:.1f}",
        )

        print(
            "seed42 CBIN/JSS:",
            f"{selected['seed42_cbin']:.4f}",
            f"{selected['seed42_jss']:.4f}",
        )

        print(
            "three-seed CBIN/JSS:",
            f"{selected['cbin_mean']:.4f}",
            f"{selected['jss_mean']:.4f}",
        )

        del model_pack

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    csv_path = (
        output_directory
        / (
            args.output_prefix
            + "_selected_cases.csv"
        )
    )

    save_case_table(
        selected_cases,
        csv_path,
    )

    plt.rcParams.update({
        "font.family":
            "DejaVu Sans",

        "pdf.fonttype":
            42,

        "ps.fonttype":
            42,
    })

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(
            7.4,
            8.2,
        ),
        facecolor="white",
    )

    plot_mirror_case(
        axes[0],
        selected_cases[0],
        "a",
        "Random",
    )

    plot_mirror_case(
        axes[1],
        selected_cases[1],
        "b",
        "Scaffold",
    )

    figure.subplots_adjust(
        left=0.12,
        right=0.985,
        top=0.965,
        bottom=0.075,
        hspace=0.20,
    )

    png_path = (
        output_directory
        / (
            args.output_prefix
            + ".png"
        )
    )

    pdf_path = (
        output_directory
        / (
            args.output_prefix
            + ".pdf"
        )
    )

    figure.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
    )

    figure.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    print()
    print("=" * 72)
    print("GENERATED")
    print("=" * 72)
    print(csv_path)
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
