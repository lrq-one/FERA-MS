#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    RDKit_AVAILABLE = True
except Exception:
    RDKit_AVAILABLE = False


# ---------- parsing helpers ----------
def parse_num_list(x):
    if pd.isna(x):
        return np.array([], dtype=float)
    x = str(x).strip()
    if not x:
        return np.array([], dtype=float)
    parts = re.split(r"[;, \t\n]+", x)
    vals = [float(p) for p in parts if p != ""]
    return np.array(vals, dtype=float)


def normalize_to_100(arr):
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return arr
    maxv = arr.max()
    if maxv <= 0:
        return arr
    return 100.0 * arr / maxv


def draw_molecule(smiles, size=(300, 220)):
    if (not RDKit_AVAILABLE) or (smiles is None) or (str(smiles).strip() == ""):
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Draw.MolToImage(mol, size=size)


def annotate_top_peaks(ax, mz, inten, top_n=8, positive=True, color="#111827"):
    if len(mz) == 0:
        return

    order = np.argsort(inten)[::-1]
    order = order[: min(top_n, len(order))]
    chosen = sorted(order, key=lambda i: mz[i])

    for i in chosen:
        x = mz[i]
        y = inten[i] if positive else -inten[i]
        va = "bottom" if positive else "top"
        dy = 2.5 if positive else -2.5
        ax.text(
            x, y + dy, f"{x:.1f}",
            fontsize=6.5,
            rotation=90,
            ha="center",
            va=va,
            color=color,
        )


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#D1D5DB", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=9)


def plot_mirror(ax, exp_mz, exp_int, pred_mz, pred_int, cbin=None, jss=None):
    exp_int = normalize_to_100(exp_int)
    pred_int = normalize_to_100(pred_int)

    obs_color = "#1F2937"     # dark gray
    pred_color = "#2F6DB5"    # blue

    ax.axhline(0.0, color="#4B5563", linewidth=1.0)

    if len(exp_mz):
        ax.vlines(exp_mz, 0, exp_int, color=obs_color, linewidth=1.2, alpha=0.95, label="Observed")
    if len(pred_mz):
        ax.vlines(pred_mz, 0, -pred_int, color=pred_color, linewidth=1.2, alpha=0.95, label="FERA-MS")

    annotate_top_peaks(ax, exp_mz, exp_int, top_n=7, positive=True, color=obs_color)
    annotate_top_peaks(ax, pred_mz, pred_int, top_n=7, positive=False, color=pred_color)

    xmax = 0
    if len(exp_mz):
        xmax = max(xmax, float(np.max(exp_mz)))
    if len(pred_mz):
        xmax = max(xmax, float(np.max(pred_mz)))
    xmax = max(100.0, xmax)
    ax.set_xlim(0, xmax * 1.03)

    ax.set_ylim(-108, 108)
    ax.set_yticks([-100, -50, 0, 50, 100])
    ax.set_yticklabels(["100", "50", "0", "50", "100"])
    ax.set_xlabel("m/z", fontsize=10)
    ax.set_ylabel("Relative intensity", fontsize=10)

    style_axis(ax)

    # labels for observed / predicted
    ax.text(
        0.01, 0.94, "Observed",
        transform=ax.transAxes,
        fontsize=9.5,
        fontweight="bold",
        color=obs_color,
        ha="left", va="top",
    )
    ax.text(
        0.01, 0.06, "FERA-MS",
        transform=ax.transAxes,
        fontsize=9.5,
        fontweight="bold",
        color=pred_color,
        ha="left", va="bottom",
    )

    # metrics box
    lines = []
    if cbin is not None and not pd.isna(cbin):
        lines.append(f"CBIN = {float(cbin):.3f}")
    if jss is not None and not pd.isna(jss):
        lines.append(f"JSS = {float(jss):.3f}")
    if lines:
        ax.text(
            0.985, 0.94,
            "\n".join(lines),
            transform=ax.transAxes,
            fontsize=8.5,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#CBD5E1", alpha=0.95),
        )


def add_molecule_panel(ax, row):
    ax.axis("off")

    # panel title
    panel = str(row.get("panel", "")).strip()
    title = str(row.get("title", "")).strip()
    if panel:
        ax.text(
            0.00, 1.03, panel,
            transform=ax.transAxes,
            fontsize=13,
            fontweight="bold",
            ha="left", va="bottom",
        )
    if title:
        ax.text(
            0.12, 1.03, title,
            transform=ax.transAxes,
            fontsize=12.5,
            fontweight="bold",
            ha="left", va="bottom",
        )

    # molecule image
    smiles = row.get("smiles", "")
    img = draw_molecule(smiles, size=(320, 230))
    if img is not None:
        ax.imshow(img)
        ax.set_xlim(0, img.size[0])
        ax.set_ylim(img.size[1], 0)
    else:
        ax.text(
            0.5, 0.65, "Structure unavailable",
            transform=ax.transAxes,
            fontsize=10,
            ha="center", va="center",
        )

    # metadata block
    meta_lines = []
    if str(row.get("name", "")).strip():
        meta_lines.append(f"Name: {row['name']}")
    if str(row.get("formula", "")).strip():
        meta_lines.append(f"Formula: {row['formula']}")
    if str(row.get("precursor_mz", "")).strip():
        meta_lines.append(f"Precursor m/z: {row['precursor_mz']}")
    if str(row.get("ace", "")).strip():
        meta_lines.append(f"ACE: {row['ace']} eV")

    meta = "\n".join(meta_lines)
    ax.text(
        0.02, -0.04, meta,
        transform=ax.transAxes,
        fontsize=8.8,
        ha="left", va="top",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_csv", required=True, help="CSV containing representative cases")
    parser.add_argument("--out_png", default="figure/Fig3_representative_cases.png")
    parser.add_argument("--out_pdf", default="figure/Fig3_representative_cases.pdf")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    df = pd.read_csv(args.case_csv)
    if df.empty:
        raise ValueError("Empty case CSV")

    n = len(df)
    fig = plt.figure(figsize=(10.5, 4.8 * n))
    gs = GridSpec(
        nrows=n, ncols=2,
        width_ratios=[1.2, 2.8],
        height_ratios=[1.0] * n,
        wspace=0.16, hspace=0.48,
        figure=fig
    )

    for idx, (_, row) in enumerate(df.iterrows()):
        ax_left = fig.add_subplot(gs[idx, 0])
        ax_right = fig.add_subplot(gs[idx, 1])

        add_molecule_panel(ax_left, row)

        exp_mz = parse_num_list(row["exp_mz"])
        exp_int = parse_num_list(row["exp_int"])
        pred_mz = parse_num_list(row["pred_mz"])
        pred_int = parse_num_list(row["pred_int"])

        plot_mirror(
            ax_right,
            exp_mz=exp_mz,
            exp_int=exp_int,
            pred_mz=pred_mz,
            pred_int=pred_int,
            cbin=row.get("cbin", np.nan),
            jss=row.get("jss", np.nan),
        )

    plt.tight_layout()
    Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_pdf).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out_png, dpi=args.dpi, bbox_inches="tight")
    plt.savefig(args.out_pdf, dpi=args.dpi, bbox_inches="tight")
    print("FIG3_PNG =", args.out_png)
    print("FIG3_PDF =", args.out_pdf)


if __name__ == "__main__":
    main()
