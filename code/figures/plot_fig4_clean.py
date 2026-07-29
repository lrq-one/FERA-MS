#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(
    "/home/lwh/projects/lrq2/fragnnet-main/"
    "ms2spectra_v1_r119"
)

FIGURE_DIR = ROOT / "figure"

INPUT_FILES = {
    "random_query": (
        FIGURE_DIR
        / "Fig4_random_query_level_paired.csv"
    ),
    "scaffold_query": (
        FIGURE_DIR
        / "Fig4_scaffold_query_level_paired.csv"
    ),
    "random_molecule": (
        FIGURE_DIR
        / "Fig4_random_molecule_level_paired.csv"
    ),
    "scaffold_molecule": (
        FIGURE_DIR
        / "Fig4_scaffold_molecule_level_paired.csv"
    ),
}

OUT_PREFIX = FIGURE_DIR / "Fig4_clean_v3"


# ============================================================
# Academic colours
# ============================================================

ORANGE_DARK = "#C9842C"
ORANGE_LIGHT = "#E4AD63"
GREY = "#B7BEC6"
BLUE_LIGHT = "#79ACD8"
BLUE_DARK = "#3F84C5"

GRID = "#D9DEE4"
TEXT = "#24292F"
REFERENCE = "#68727D"


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 12.5,
        "axes.titlesize": 20.5,
        "axes.labelsize": 14.5,
        "xtick.labelsize": 12.0,
        "ytick.labelsize": 12.0,
        "axes.linewidth": 0.9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


# ============================================================
# Loading and checking
# ============================================================

def load_inputs():
    for name, path in INPUT_FILES.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing input file for {name}:\n{path}"
            )

    random_query = pd.read_csv(
        INPUT_FILES["random_query"]
    )

    scaffold_query = pd.read_csv(
        INPUT_FILES["scaffold_query"]
    )

    random_molecule = pd.read_csv(
        INPUT_FILES["random_molecule"]
    )

    scaffold_molecule = pd.read_csv(
        INPUT_FILES["scaffold_molecule"]
    )

    query_required = {
        "query_spec_id",
        "target_mol_id",
        "delta_rank",
    }

    molecule_required = {
        "target_mol_id",
        "fera_mrr",
        "fragnnet_mrr",
        "delta_mrr",
    }

    for name, frame in (
        ("random query", random_query),
        ("scaffold query", scaffold_query),
    ):
        missing = query_required.difference(
            frame.columns
        )

        if missing:
            raise RuntimeError(
                f"{name} file is missing columns: "
                f"{sorted(missing)}"
            )

    for name, frame in (
        ("random molecule", random_molecule),
        ("scaffold molecule", scaffold_molecule),
    ):
        missing = molecule_required.difference(
            frame.columns
        )

        if missing:
            raise RuntimeError(
                f"{name} file is missing columns: "
                f"{sorted(missing)}"
            )

    return (
        random_query,
        scaffold_query,
        random_molecule,
        scaffold_molecule,
    )


# ============================================================
# Panel A / B
# ============================================================

def summarize_rank_shift(delta):
    delta = np.asarray(
        delta,
        dtype=float,
    )

    eps = 1e-12

    large_loss = delta <= -5
    small_loss = (
        (delta > -5)
        & (delta < -eps)
    )

    tied = np.isclose(
        delta,
        0.0,
        atol=eps,
    )

    small_gain = (
        (delta > eps)
        & (delta < 5)
    )

    large_gain = delta >= 5

    return {
        "large_loss": (
            100.0 * large_loss.mean()
        ),
        "small_loss": (
            100.0 * small_loss.mean()
        ),
        "tied": (
            100.0 * tied.mean()
        ),
        "small_gain": (
            100.0 * small_gain.mean()
        ),
        "large_gain": (
            100.0 * large_gain.mean()
        ),
        "improved": (
            100.0 * (delta > eps).mean()
        ),
        "worsened": (
            100.0 * (delta < -eps).mean()
        ),
        "mean_delta": float(
            np.mean(delta)
        ),
        "median_delta": float(
            np.median(delta)
        ),
    }


def draw_rank_shift(
    ax,
    frame,
    panel_letter,
    split_label,
):
    delta = frame[
        "delta_rank"
    ].to_numpy(dtype=float)

    result = summarize_rank_shift(
        delta
    )

    tied_half = (
        result["tied"] / 2.0
    )

    segments = [
        {
            "name": "Large\nloss",
            "value": result["large_loss"],
            "left": -(
                result["large_loss"]
                + result["small_loss"]
                + tied_half
            ),
            "colour": ORANGE_DARK,
            "text_colour": "white",
        },
        {
            "name": "Small\nloss",
            "value": result["small_loss"],
            "left": -(
                result["small_loss"]
                + tied_half
            ),
            "colour": ORANGE_LIGHT,
            "text_colour": TEXT,
        },
        {
            "name": "Tied",
            "value": result["tied"],
            "left": -tied_half,
            "colour": GREY,
            "text_colour": TEXT,
        },
        {
            "name": "Small\ngain",
            "value": result["small_gain"],
            "left": tied_half,
            "colour": BLUE_LIGHT,
            "text_colour": TEXT,
        },
        {
            "name": "Large\ngain",
            "value": result["large_gain"],
            "left": (
                tied_half
                + result["small_gain"]
            ),
            "colour": BLUE_DARK,
            "text_colour": "white",
        },
    ]

    for segment in segments:
        value = segment["value"]
        left = segment["left"]

        ax.barh(
            0,
            value,
            left=left,
            height=0.34,
            color=segment["colour"],
            edgecolor="white",
            linewidth=1.0,
            zorder=3,
        )

        centre = left + value / 2.0

        if value >= 4.0:
            ax.text(
                centre,
                0,
                f"{value:.1f}%",
                ha="center",
                va="center",
                fontsize=12.6,
                fontweight="bold",
                color=segment["text_colour"],
                zorder=4,
            )

        if value >= 7.0:
            ax.text(
                centre,
                0.335,
                segment["name"],
                ha="center",
                va="center",
                multialignment="center",
                linespacing=0.92,
                fontsize=12.2,
                color=TEXT,
                clip_on=False,
            )

    ax.axvline(
        0,
        color=REFERENCE,
        linestyle="--",
        linewidth=1.0,
        zorder=2,
    )

    ax.text(
        0.0,
        1.035,
        (
            f"{result['improved']:.1f}% improved · "
            f"{result['tied']:.1f}% tied · "
            f"{result['worsened']:.1f}% worsened"
            "\n"
            f"mean Δrank = {result['mean_delta']:.2f} · "
            f"median = {result['median_delta']:.2f}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        multialignment="left",
        linespacing=1.15,
        fontsize=11.7,
        color=TEXT,
        clip_on=False,
    )

    ax.set_title(
        f"{panel_letter}  Query-level rank shifts — {split_label}",
        loc="left",
        fontsize=20.5,
        fontweight="bold",
        y=1.205,
        pad=0,
    )

    ax.set_xlim(
        -70,
        70,
    )

    ticks = np.arange(
        -60,
        61,
        20,
    )

    ax.set_xticks(
        ticks
    )

    ax.set_xticklabels(
        [str(abs(int(tick))) for tick in ticks]
    )

    ax.set_xlabel(
        "Queries (%)"
    )

    ax.set_ylim(
        -0.64,
        0.64,
    )

    ax.set_yticks([])

    ax.text(
        -68,
        -0.48,
        "FraGNNet-D3 better",
        ha="left",
        va="center",
        fontsize=12.5,
        color=ORANGE_DARK,
    )

    ax.text(
        68,
        -0.48,
        "FERA-MS better",
        ha="right",
        va="center",
        fontsize=12.5,
        color=BLUE_DARK,
    )

    ax.grid(
        axis="x",
        color=GRID,
        linewidth=0.7,
        zorder=0,
    )

    ax.spines["top"].set_visible(
        False
    )
    ax.spines["right"].set_visible(
        False
    )
    ax.spines["left"].set_visible(
        False
    )

    return result


# ============================================================
# Panel C / D
# ============================================================

def draw_molecule_scatter(
    ax,
    frame,
    panel_letter,
    split_label,
):
    x = frame[
        "fragnnet_mrr"
    ].to_numpy(dtype=float)

    y = frame[
        "fera_mrr"
    ].to_numpy(dtype=float)

    delta = (
        y - x
    )

    eps = 1e-12

    higher = delta > eps
    lower = delta < -eps
    tied = np.isclose(
        delta,
        0.0,
        atol=eps,
    )

    ax.scatter(
        x[lower],
        y[lower],
        s=16,
        alpha=0.50,
        color=ORANGE_DARK,
        edgecolors="none",
        rasterized=True,
        zorder=3,
    )

    if tied.any():
        ax.scatter(
            x[tied],
            y[tied],
            s=14,
            alpha=0.48,
            color=GREY,
            edgecolors="none",
            rasterized=True,
            zorder=3,
        )

    ax.scatter(
        x[higher],
        y[higher],
        s=16,
        alpha=0.50,
        color=BLUE_DARK,
        edgecolors="none",
        rasterized=True,
        zorder=3,
    )

    ax.plot(
        [0, 1],
        [0, 1],
        color=REFERENCE,
        linestyle="--",
        linewidth=1.0,
        zorder=2,
    )

    higher_pct = (
        100.0 * higher.mean()
    )

    lower_pct = (
        100.0 * lower.mean()
    )

    median_delta = float(
        np.median(delta)
    )

    mean_delta = float(
        np.mean(delta)
    )

    ax.text(
        0.0,
        1.085,
        (
            f"{higher_pct:.1f}% of molecules higher with FERA-MS"
            "\n"
            f"mean ΔMRR = {mean_delta:.3f} · "
            f"median = {median_delta:.3f}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        multialignment="left",
        linespacing=1.15,
        fontsize=11.7,
        color=TEXT,
        clip_on=False,
    )

    ax.set_title(
        f"{panel_letter}  Molecule-level paired MRR — {split_label}",
        loc="left",
        fontsize=20.5,
        fontweight="bold",
        y=1.205,
        pad=0,
    )

    ax.set_xlabel(
        "FraGNNet-D3 molecule-level MRR"
    )

    ax.set_ylabel(
        "FERA-MS molecule-level MRR"
    )

    ax.set_xlim(
        -0.015,
        1.015,
    )

    ax.set_ylim(
        -0.015,
        1.015,
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.grid(
        color=GRID,
        linewidth=0.65,
        zorder=0,
    )

    ax.text(
        0.08,
        0.91,
        "FERA-MS better",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=12.5,
        color=BLUE_DARK,
    )

    ax.text(
        0.94,
        0.08,
        "FraGNNet-D3 better",
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=12.5,
        color=ORANGE_DARK,
    )

    ax.spines["top"].set_visible(
        False
    )
    ax.spines["right"].set_visible(
        False
    )

    return {
        "fera_higher_pct": higher_pct,
        "fragnnet_higher_pct": lower_pct,
        "mean_delta_mrr": mean_delta,
        "median_delta_mrr": median_delta,
    }


# ============================================================
# Main
# ============================================================


def main():
    (
        random_query,
        scaffold_query,
        random_molecule,
        scaffold_molecule,
    ) = load_inputs()

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(16.8, 11.6),
        gridspec_kw={
            "height_ratios": [0.78, 1.22],
        },
    )

    summary_rows = []

    random_rank_summary = draw_rank_shift(
        axes[0, 0],
        random_query,
        panel_letter="A",
        split_label="Random",
    )

    scaffold_rank_summary = draw_rank_shift(
        axes[0, 1],
        scaffold_query,
        panel_letter="B",
        split_label="Scaffold",
    )

    random_molecule_summary = (
        draw_molecule_scatter(
            axes[1, 0],
            random_molecule,
            panel_letter="C",
            split_label="Random",
        )
    )

    scaffold_molecule_summary = (
        draw_molecule_scatter(
            axes[1, 1],
            scaffold_molecule,
            panel_letter="D",
            split_label="Scaffold",
        )
    )

    summary_rows.extend(
        [
            {
                "panel": "A",
                "split": "random",
                **random_rank_summary,
            },
            {
                "panel": "B",
                "split": "scaffold",
                **scaffold_rank_summary,
            },
            {
                "panel": "C",
                "split": "random",
                **random_molecule_summary,
            },
            {
                "panel": "D",
                "split": "scaffold",
                **scaffold_molecule_summary,
            },
        ]
    )

    fig.subplots_adjust(
        left=0.070,
        right=0.985,
        bottom=0.075,
        top=0.925,
        wspace=0.285,
        hspace=0.515,
    )

    # Square scatter panels appear optically left-shifted because
    # the y-axis title and tick labels occupy extra space.
    # Move them slightly right and reduce their width minimally
    # to preserve the outer margins.
    for ax, dx in (
        (axes[1, 0], 0.016),
        (axes[1, 1], 0.008),
    ):
        position = ax.get_position()

        ax.set_position(
            [
                position.x0 + dx,
                position.y0,
                position.width * 0.96,
                position.height,
            ]
        )

    # Align the titles and summary lines of C/D with the left
    # boundaries of A/B, while keeping the square scatter areas
    # optically centred.
    for upper_ax, lower_ax in (
        (axes[0, 0], axes[1, 0]),
        (axes[0, 1], axes[1, 1]),
    ):
        upper_position = upper_ax.get_position()
        lower_position = lower_ax.get_position()

        # Convert the upper-panel left boundary from figure
        # coordinates into lower-axis coordinates.
        aligned_x = (
            upper_position.x0
            - lower_position.x0
        ) / lower_position.width

        # The title created with loc="left" is stored separately
        # from the centred title.
        lower_ax._left_title.set_x(
            aligned_x
        )

        # Move only the summary line located above the lower axes.
        # Keep the in-panel labels unchanged.
        for label in lower_ax.texts:
            x_value, y_value = label.get_position()

            if y_value > 1.0:
                label.set_x(
                    aligned_x
                )

    fig.savefig(
        f"{OUT_PREFIX}.png",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.10,
    )

    fig.savefig(
        f"{OUT_PREFIX}.pdf",
        bbox_inches="tight",
        pad_inches=0.10,
    )

    fig.savefig(
        f"{OUT_PREFIX}.svg",
        bbox_inches="tight",
        pad_inches=0.10,
    )

    plt.close(
        fig
    )

    pd.DataFrame(
        summary_rows
    ).to_csv(
        FIGURE_DIR
        / "Fig4_clean_v3_summary.csv",
        index=False,
    )

    print("=" * 80)
    print("CLEAN FIGURE 4 COMPLETE")
    print("=" * 80)
    print(f"PNG: {OUT_PREFIX}.png")
    print(f"PDF: {OUT_PREFIX}.pdf")
    print(f"SVG: {OUT_PREFIX}.svg")
    print(
        "SUMMARY:",
        FIGURE_DIR
        / "Fig4_clean_v3_summary.csv",
    )
    print("=" * 80)


if __name__ == "__main__":
    main()