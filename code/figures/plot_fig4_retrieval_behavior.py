#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# Paths
# ============================================================

ROOT = Path(
    "/home/lwh/projects/lrq2/fragnnet-main/"
    "ms2spectra_v1_r119"
)

EXP_ROOT = (
    ROOT
    / "runs"
    / "experiments"
    / "molecular_retrieval"
    / "pubchem_legacy_full"
)

FERA_ROOT = (
    EXP_ROOT
    / "ours_r184b_experiment5_20260724"
)

FRAGNNET_ROOT = (
    EXP_ROOT
    / "baseline_experiment5"
    / "fragnnet_d3"
)

OUT_DIR = ROOT / "figure"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PREFIX = OUT_DIR / "Fig4_retrieval_behavior"

SEEDS = (42, 43, 44)
SPLITS = ("random", "scaffold")

EXPECTED_COUNTS = {
    "random": {
        "queries": 3917,
        "molecules": 454,
    },
    "scaffold": {
        "queries": 3949,
        "molecules": 448,
    },
}


# ============================================================
# Plot style
# ============================================================

COLOR_BETTER = "#3C82C4"
COLOR_WORSE = "#D59645"
COLOR_TIE = "#A7ADB5"
COLOR_DIAGONAL = "#5E6873"
COLOR_GRID = "#D9DEE4"
COLOR_TEXT = "#20252B"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


# ============================================================
# Data loading
# ============================================================

REQUIRED_COLUMNS = {
    "split",
    "seed",
    "cohort",
    "method",
    "query_spec_id",
    "target_mol_id",
    "retrieval_rank",
}


def load_rank_file(
    model_name: str,
    model_root: Path,
    split: str,
    seed: int,
) -> pd.DataFrame:
    path = (
        model_root
        / split
        / f"seed_{seed}"
        / "true_candidate_ranks.csv.gz"
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing rank file: {path}"
        )

    frame = pd.read_csv(
        path,
        low_memory=False,
    )

    missing = REQUIRED_COLUMNS.difference(
        frame.columns
    )

    if missing:
        raise RuntimeError(
            f"{path} is missing columns: "
            f"{sorted(missing)}"
        )

    frame = frame.loc[
        (frame["cohort"] == "fixed50")
        & (frame["method"] == "cbin_sqrt")
    ].copy()

    frame = frame[
        [
            "split",
            "seed",
            "query_spec_id",
            "target_mol_id",
            "query_ace",
            "retrieval_rank",
            "pool_size_scored",
            "true_candidate_similarity",
        ]
    ].copy()

    frame["model"] = model_name
    frame["retrieval_rank"] = pd.to_numeric(
        frame["retrieval_rank"],
        errors="raise",
    )

    if frame["query_spec_id"].duplicated().any():
        duplicated = frame.loc[
            frame["query_spec_id"].duplicated(False),
            "query_spec_id",
        ].head(10).tolist()

        raise RuntimeError(
            f"Duplicate query_spec_id values in {path}: "
            f"{duplicated}"
        )

    expected_queries = EXPECTED_COUNTS[
        split
    ]["queries"]

    if len(frame) != expected_queries:
        raise RuntimeError(
            f"{model_name} {split} seed {seed}: "
            f"expected {expected_queries} fixed50/cbin_sqrt "
            f"queries, found {len(frame)}"
        )

    return frame


def load_all_model_ranks(
    model_name: str,
    model_root: Path,
    split: str,
) -> pd.DataFrame:
    frames = [
        load_rank_file(
            model_name=model_name,
            model_root=model_root,
            split=split,
            seed=seed,
        )
        for seed in SEEDS
    ]

    return pd.concat(
        frames,
        ignore_index=True,
    )


# ============================================================
# Metric validation
# ============================================================

def calculate_seed_metrics(
    frame: pd.DataFrame,
    model_name: str,
    split: str,
    seed: int,
) -> list[dict]:
    ranks = frame["retrieval_rank"].to_numpy(
        dtype=float
    )

    reciprocal_ranks = 1.0 / ranks

    spectrum_row = {
        "model": model_name,
        "split": split,
        "seed": seed,
        "unit": "spectrum_micro",
        "n_spectra": len(frame),
        "n_molecules": frame[
            "target_mol_id"
        ].nunique(),
        "top1": np.mean(ranks <= 1),
        "top5": np.mean(ranks <= 5),
        "top10": np.mean(ranks <= 10),
        "mrr": np.mean(reciprocal_ranks),
        "median_rank": np.median(ranks),
        "mean_rank": np.mean(ranks),
    }

    molecule_frame = frame.copy()
    molecule_frame["top1"] = (
        molecule_frame["retrieval_rank"] <= 1
    ).astype(float)
    molecule_frame["top5"] = (
        molecule_frame["retrieval_rank"] <= 5
    ).astype(float)
    molecule_frame["top10"] = (
        molecule_frame["retrieval_rank"] <= 10
    ).astype(float)
    molecule_frame["rr"] = (
        1.0 / molecule_frame["retrieval_rank"]
    )

    per_molecule = (
        molecule_frame
        .groupby(
            "target_mol_id",
            as_index=False,
        )
        .agg(
            top1=("top1", "mean"),
            top5=("top5", "mean"),
            top10=("top10", "mean"),
            mrr=("rr", "mean"),
            median_rank=(
                "retrieval_rank",
                "median",
            ),
            mean_rank=(
                "retrieval_rank",
                "mean",
            ),
        )
    )

    molecule_row = {
        "model": model_name,
        "split": split,
        "seed": seed,
        "unit": "molecule_macro",
        "n_spectra": len(frame),
        "n_molecules": len(per_molecule),
        "top1": per_molecule["top1"].mean(),
        "top5": per_molecule["top5"].mean(),
        "top10": per_molecule["top10"].mean(),
        "mrr": per_molecule["mrr"].mean(),
        "median_rank": (
            per_molecule["median_rank"].mean()
        ),
        "mean_rank": (
            per_molecule["mean_rank"].mean()
        ),
    }

    return [
        spectrum_row,
        molecule_row,
    ]


# ============================================================
# Build paired query and molecule data
# ============================================================

def aggregate_query_level(
    frame: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    frame = frame.copy()
    frame["reciprocal_rank"] = (
        1.0 / frame["retrieval_rank"]
    )

    output = (
        frame
        .groupby(
            [
                "query_spec_id",
                "target_mol_id",
            ],
            as_index=False,
        )
        .agg(
            **{
                f"{prefix}_rank_mean": (
                    "retrieval_rank",
                    "mean",
                ),
                f"{prefix}_rank_std": (
                    "retrieval_rank",
                    "std",
                ),
                f"{prefix}_rr_mean": (
                    "reciprocal_rank",
                    "mean",
                ),
                f"{prefix}_n_seeds": (
                    "seed",
                    "nunique",
                ),
            }
        )
    )

    return output


def build_paired_split(
    split: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fera_all = load_all_model_ranks(
        model_name="FERA-MS",
        model_root=FERA_ROOT,
        split=split,
    )

    frag_all = load_all_model_ranks(
        model_name="FraGNNet-D3",
        model_root=FRAGNNET_ROOT,
        split=split,
    )

    validation_rows = []

    for seed in SEEDS:
        validation_rows.extend(
            calculate_seed_metrics(
                frame=fera_all.loc[
                    fera_all["seed"] == seed
                ],
                model_name="FERA-MS",
                split=split,
                seed=seed,
            )
        )

        validation_rows.extend(
            calculate_seed_metrics(
                frame=frag_all.loc[
                    frag_all["seed"] == seed
                ],
                model_name="FraGNNet-D3",
                split=split,
                seed=seed,
            )
        )

    validation = pd.DataFrame(
        validation_rows
    )

    fera_query = aggregate_query_level(
        fera_all,
        prefix="fera",
    )

    frag_query = aggregate_query_level(
        frag_all,
        prefix="fragnnet",
    )

    outer = fera_query.merge(
        frag_query,
        on=[
            "query_spec_id",
            "target_mol_id",
        ],
        how="outer",
        indicator=True,
    )

    unmatched = outer.loc[
        outer["_merge"] != "both"
    ]

    if not unmatched.empty:
        raise RuntimeError(
            f"{split}: FERA-MS and FraGNNet-D3 "
            f"query sets do not match. "
            f"Unmatched rows: {len(unmatched)}"
        )

    paired = outer.drop(
        columns="_merge"
    ).copy()

    if not (
        paired["fera_n_seeds"] == 3
    ).all():
        raise RuntimeError(
            f"{split}: some FERA-MS queries do not "
            f"contain all three seeds"
        )

    if not (
        paired["fragnnet_n_seeds"] == 3
    ).all():
        raise RuntimeError(
            f"{split}: some FraGNNet-D3 queries do not "
            f"contain all three seeds"
        )

    paired["delta_rank"] = (
        paired["fragnnet_rank_mean"]
        - paired["fera_rank_mean"]
    )

    paired["delta_rr"] = (
        paired["fera_rr_mean"]
        - paired["fragnnet_rr_mean"]
    )

    expected_queries = EXPECTED_COUNTS[
        split
    ]["queries"]

    expected_molecules = EXPECTED_COUNTS[
        split
    ]["molecules"]

    if len(paired) != expected_queries:
        raise RuntimeError(
            f"{split}: expected {expected_queries} "
            f"paired queries, found {len(paired)}"
        )

    if (
        paired["target_mol_id"].nunique()
        != expected_molecules
    ):
        raise RuntimeError(
            f"{split}: expected {expected_molecules} "
            f"paired molecules, found "
            f"{paired['target_mol_id'].nunique()}"
        )

    molecule = (
        paired
        .groupby(
            "target_mol_id",
            as_index=False,
        )
        .agg(
            n_queries=(
                "query_spec_id",
                "nunique",
            ),
            fera_mrr=(
                "fera_rr_mean",
                "mean",
            ),
            fragnnet_mrr=(
                "fragnnet_rr_mean",
                "mean",
            ),
        )
    )

    molecule["delta_mrr"] = (
        molecule["fera_mrr"]
        - molecule["fragnnet_mrr"]
    )

    return paired, molecule, validation


# ============================================================
# Plotting
# ============================================================

def panel_title(
    ax,
    letter: str,
    title: str,
) -> None:
    ax.set_title(
        f"{letter}  {title}",
        loc="left",
        fontweight="bold",
        pad=8,
    )


def plot_rank_distribution(
    ax,
    paired: pd.DataFrame,
    letter: str,
    split_label: str,
) -> dict:
    delta = paired["delta_rank"].to_numpy(
        dtype=float
    )

    # Mean rank difference can have thirds because three
    # seeds are averaged. Two-rank bins keep the figure clean.
    bin_edges = np.arange(
        -50,
        52,
        2,
        dtype=float,
    )

    counts, edges = np.histogram(
        delta,
        bins=bin_edges,
    )

    percentages = (
        counts
        / len(delta)
        * 100.0
    )

    centers = (
        edges[:-1]
        + edges[1:]
    ) / 2.0

    colors = []

    for center in centers:
        if center < -1:
            colors.append(
                COLOR_WORSE
            )
        elif center > 1:
            colors.append(
                COLOR_BETTER
            )
        else:
            colors.append(
                COLOR_TIE
            )

    ax.bar(
        centers,
        percentages,
        width=np.diff(edges) * 0.92,
        color=colors,
        edgecolor="white",
        linewidth=0.25,
        zorder=3,
    )

    ax.axvline(
        0,
        color=COLOR_DIAGONAL,
        linewidth=1.0,
        linestyle="--",
        zorder=4,
    )

    improved = np.mean(
        delta > 1e-12
    ) * 100.0

    tied = np.mean(
        np.isclose(
            delta,
            0.0,
            atol=1e-12,
        )
    ) * 100.0

    worsened = np.mean(
        delta < -1e-12
    ) * 100.0

    median_delta = np.median(
        delta
    )

    mean_delta = np.mean(
        delta
    )

    ax.text(
        0.025,
        0.965,
        (
            f"Improved {improved:.1f}%   "
            f"Tied {tied:.1f}%   "
            f"Worsened {worsened:.1f}%\n"
            f"Mean Δrank {mean_delta:.2f}   "
            f"Median Δrank {median_delta:.2f}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color=COLOR_TEXT,
    )

    panel_title(
        ax,
        letter,
        (
            f"Query-level rank improvement "
            f"({split_label})"
        ),
    )

    ax.set_xlabel(
        "Rank improvement "
        "(FraGNNet-D3 − FERA-MS)"
    )

    ax.set_ylabel(
        "Queries (%)"
    )

    ax.set_xlim(
        -50,
        50,
    )

    ymax = max(
        5.0,
        percentages.max() * 1.22,
    )

    ax.set_ylim(
        0,
        ymax,
    )

    ax.grid(
        axis="y",
        color=COLOR_GRID,
        linewidth=0.7,
        zorder=0,
    )

    ax.spines["top"].set_visible(
        False
    )
    ax.spines["right"].set_visible(
        False
    )

    return {
        "split": split_label.lower(),
        "n_queries": len(paired),
        "improved_pct": improved,
        "tied_pct": tied,
        "worsened_pct": worsened,
        "mean_delta_rank": mean_delta,
        "median_delta_rank": median_delta,
    }


def plot_molecule_scatter(
    ax,
    molecule: pd.DataFrame,
    letter: str,
    split_label: str,
    show_legend: bool,
) -> dict:
    tolerance = 1e-12

    better = (
        molecule["delta_mrr"]
        > tolerance
    )

    worse = (
        molecule["delta_mrr"]
        < -tolerance
    )

    tied = ~better & ~worse

    ax.scatter(
        molecule.loc[
            worse,
            "fragnnet_mrr",
        ],
        molecule.loc[
            worse,
            "fera_mrr",
        ],
        s=24,
        alpha=0.70,
        color=COLOR_WORSE,
        edgecolors="none",
        zorder=3,
    )

    ax.scatter(
        molecule.loc[
            tied,
            "fragnnet_mrr",
        ],
        molecule.loc[
            tied,
            "fera_mrr",
        ],
        s=24,
        alpha=0.70,
        color=COLOR_TIE,
        edgecolors="none",
        zorder=3,
    )

    ax.scatter(
        molecule.loc[
            better,
            "fragnnet_mrr",
        ],
        molecule.loc[
            better,
            "fera_mrr",
        ],
        s=24,
        alpha=0.70,
        color=COLOR_BETTER,
        edgecolors="none",
        zorder=3,
    )

    ax.plot(
        [0, 1],
        [0, 1],
        color=COLOR_DIAGONAL,
        linestyle="--",
        linewidth=1.0,
        zorder=2,
    )

    higher_pct = (
        better.mean()
        * 100.0
    )

    lower_pct = (
        worse.mean()
        * 100.0
    )

    tied_pct = (
        tied.mean()
        * 100.0
    )

    mean_delta = (
        molecule["delta_mrr"].mean()
    )

    median_delta = (
        molecule["delta_mrr"].median()
    )

    ax.text(
        0.025,
        0.965,
        (
            f"FERA-MS higher {higher_pct:.1f}%   "
            f"Lower {lower_pct:.1f}%\n"
            f"Mean ΔMRR {mean_delta:.3f}   "
            f"Median ΔMRR {median_delta:.3f}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color=COLOR_TEXT,
    )

    panel_title(
        ax,
        letter,
        (
            f"Molecule-level paired MRR "
            f"({split_label})"
        ),
    )

    ax.set_xlabel(
        "FraGNNet-D3 molecule-level MRR"
    )

    ax.set_ylabel(
        "FERA-MS molecule-level MRR"
    )

    ax.set_xlim(
        -0.02,
        1.02,
    )

    ax.set_ylim(
        -0.02,
        1.02,
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.grid(
        color=COLOR_GRID,
        linewidth=0.7,
        zorder=0,
    )

    ax.spines["top"].set_visible(
        False
    )
    ax.spines["right"].set_visible(
        False
    )

    if show_legend:
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=COLOR_BETTER,
                markeredgecolor="none",
                markersize=6,
                label="FERA-MS higher",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=COLOR_WORSE,
                markeredgecolor="none",
                markersize=6,
                label="FERA-MS lower",
            ),
            Line2D(
                [0],
                [0],
                color=COLOR_DIAGONAL,
                linestyle="--",
                linewidth=1.0,
                label="Equal performance",
            ),
        ]

        ax.legend(
            handles=handles,
            loc="lower right",
            frameon=False,
        )

    return {
        "split": split_label.lower(),
        "n_molecules": len(molecule),
        "fera_higher_pct": higher_pct,
        "fera_lower_pct": lower_pct,
        "tied_pct": tied_pct,
        "mean_delta_mrr": mean_delta,
        "median_delta_mrr": median_delta,
    }


# ============================================================
# Main
# ============================================================

def main() -> None:
    paired_by_split = {}
    molecule_by_split = {}
    validation_frames = []

    for split in SPLITS:
        paired, molecule, validation = (
            build_paired_split(
                split
            )
        )

        paired["split"] = split
        molecule["split"] = split

        paired_by_split[split] = paired
        molecule_by_split[split] = molecule
        validation_frames.append(
            validation
        )

    validation_all = pd.concat(
        validation_frames,
        ignore_index=True,
    )

    validation_summary = (
        validation_all
        .groupby(
            [
                "model",
                "split",
                "unit",
            ],
            as_index=False,
        )
        .agg(
            n_seeds=("seed", "nunique"),
            top1_mean=("top1", "mean"),
            top1_std=("top1", "std"),
            top5_mean=("top5", "mean"),
            top5_std=("top5", "std"),
            top10_mean=("top10", "mean"),
            top10_std=("top10", "std"),
            mrr_mean=("mrr", "mean"),
            mrr_std=("mrr", "std"),
            median_rank_mean=(
                "median_rank",
                "mean",
            ),
            mean_rank_mean=(
                "mean_rank",
                "mean",
            ),
        )
    )

    # ----------------------
    # Figure
    # ----------------------

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12.2, 9.3),
    )

    rank_summaries = []
    molecule_summaries = []

    rank_summaries.append(
        plot_rank_distribution(
            axes[0, 0],
            paired_by_split["random"],
            letter="A",
            split_label="Random split",
        )
    )

    rank_summaries.append(
        plot_rank_distribution(
            axes[0, 1],
            paired_by_split["scaffold"],
            letter="B",
            split_label="Scaffold split",
        )
    )

    molecule_summaries.append(
        plot_molecule_scatter(
            axes[1, 0],
            molecule_by_split["random"],
            letter="C",
            split_label="Random split",
            show_legend=True,
        )
    )

    molecule_summaries.append(
        plot_molecule_scatter(
            axes[1, 1],
            molecule_by_split["scaffold"],
            letter="D",
            split_label="Scaffold split",
            show_legend=False,
        )
    )

    fig.subplots_adjust(
        left=0.08,
        right=0.985,
        bottom=0.075,
        top=0.975,
        wspace=0.23,
        hspace=0.28,
    )

    fig.savefig(
        f"{OUT_PREFIX}.png",
        dpi=450,
        bbox_inches="tight",
    )

    fig.savefig(
        f"{OUT_PREFIX}.pdf",
        bbox_inches="tight",
    )

    fig.savefig(
        f"{OUT_PREFIX}.svg",
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    # ----------------------
    # Audit outputs
    # ----------------------

    for split in SPLITS:
        paired_by_split[
            split
        ].to_csv(
            (
                OUT_DIR
                / f"Fig4_{split}_query_level_paired.csv"
            ),
            index=False,
        )

        molecule_by_split[
            split
        ].to_csv(
            (
                OUT_DIR
                / f"Fig4_{split}_molecule_level_paired.csv"
            ),
            index=False,
        )

    validation_all.to_csv(
        (
            OUT_DIR
            / "Fig4_seed_level_metric_validation.csv"
        ),
        index=False,
    )

    validation_summary.to_csv(
        (
            OUT_DIR
            / "Fig4_three_seed_metric_validation.csv"
        ),
        index=False,
    )

    figure_summary = pd.DataFrame(
        rank_summaries
        + molecule_summaries
    )

    figure_summary.to_csv(
        (
            OUT_DIR
            / "Fig4_behavior_summary.csv"
        ),
        index=False,
    )

    # ----------------------
    # Terminal report
    # ----------------------

    print("=" * 90)
    print("FIGURE 4 COMPLETE")
    print("=" * 90)

    for split in SPLITS:
        paired = paired_by_split[
            split
        ]

        molecule = molecule_by_split[
            split
        ]

        print(
            f"{split.upper()}: "
            f"{len(paired)} paired queries, "
            f"{len(molecule)} paired molecules"
        )

        print(
            "  Query rank improvement: "
            f"mean={paired['delta_rank'].mean():.4f}, "
            f"median={paired['delta_rank'].median():.4f}, "
            f"improved={(paired['delta_rank'] > 0).mean() * 100:.2f}%"
        )

        print(
            "  Molecule MRR improvement: "
            f"mean={molecule['delta_mrr'].mean():.4f}, "
            f"median={molecule['delta_mrr'].median():.4f}, "
            f"FERA-MS higher={(molecule['delta_mrr'] > 0).mean() * 100:.2f}%"
        )

    print()
    print("THREE-SEED VALIDATION:")
    print(
        validation_summary[
            [
                "model",
                "split",
                "unit",
                "top1_mean",
                "top5_mean",
                "top10_mean",
                "mrr_mean",
                "median_rank_mean",
                "mean_rank_mean",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("OUTPUTS:")
    print(f"  {OUT_PREFIX}.png")
    print(f"  {OUT_PREFIX}.pdf")
    print(f"  {OUT_PREFIX}.svg")
    print(
        f"  {OUT_DIR / 'Fig4_behavior_summary.csv'}"
    )
    print(
        f"  {OUT_DIR / 'Fig4_three_seed_metric_validation.csv'}"
    )
    print("=" * 90)


if __name__ == "__main__":
    main()
