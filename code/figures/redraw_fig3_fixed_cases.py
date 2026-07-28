#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from rdkit import Chem
from rdkit.Chem import rdDepictor


ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = ROOT / "figure"

CACHE_NPZ = (
    FIGURE_DIR
    / "Fig3_fixed_cases_spectra_cache.npz"
)

CACHE_JSON = (
    FIGURE_DIR
    / "Fig3_fixed_cases_metadata.json"
)

OUT_PNG = (
    FIGURE_DIR
    / "Fig3_representative_cases_clean.png"
)

OUT_PDF = (
    FIGURE_DIR
    / "Fig3_representative_cases_clean.pdf"
)

OUT_SVG = (
    FIGURE_DIR
    / "Fig3_representative_cases_clean.svg"
)


OBSERVED_BLUE = "#2F80C3"
PREDICTED_ORANGE = "#F28E2B"
INK = "#20262E"
GRID = "#D9DEE5"
SPINE = "#69727D"


def sqrt_relative(
    intensity: np.ndarray,
) -> np.ndarray:
    values = np.asarray(
        intensity,
        dtype=float,
    )

    values = np.clip(
        values,
        0.0,
        None,
    )

    if (
        values.size == 0
        or float(values.max()) <= 0.0
    ):
        return values

    # Leave space between the tallest peak and the border.
    return (
        np.sqrt(
            values / float(values.max())
        )
        * 0.86
    )


def collapse_peaks(
    mz: np.ndarray,
    intensity: np.ndarray,
    decimals: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    mz = np.asarray(
        mz,
        dtype=float,
    ).reshape(-1)

    intensity = np.asarray(
        intensity,
        dtype=float,
    ).reshape(-1)

    keep = (
        np.isfinite(mz)
        & np.isfinite(intensity)
        & (mz > 0.0)
        & (intensity > 0.0)
    )

    mz = mz[keep]
    intensity = intensity[keep]

    if mz.size == 0:
        return mz, intensity

    rounded = np.round(
        mz,
        decimals,
    )

    unique_mz, inverse = np.unique(
        rounded,
        return_inverse=True,
    )

    summed = np.zeros(
        len(unique_mz),
        dtype=float,
    )

    np.add.at(
        summed,
        inverse,
        intensity,
    )

    order = np.argsort(
        unique_mz
    )

    return (
        unique_mz[order],
        summed[order],
    )


def spaced_top_indices(
    mz: np.ndarray,
    height: np.ndarray,
    top_n: int,
    min_separation: float = 12.0,
) -> list[int]:
    if len(mz) == 0:
        return []

    selected: list[int] = []

    for index in np.argsort(
        height
    )[::-1]:
        if height[index] < 0.28:
            continue

        current_mz = float(
            mz[index]
        )

        separated = all(
            abs(
                current_mz
                - float(mz[chosen])
            )
            >= min_separation
            for chosen in selected
        )

        if separated:
            selected.append(
                int(index)
            )

        if len(selected) >= top_n:
            break

    return sorted(
        selected,
        key=lambda index: float(mz[index]),
    )


def annotate_major_peaks(
    axis: plt.Axes,
    mz: np.ndarray,
    height: np.ndarray,
    positive: bool,
    color: str,
    top_n: int,
) -> None:
    indices = spaced_top_indices(
        mz,
        height,
        top_n=top_n,
    )

    for index in indices:
        value = float(
            height[index]
        )

        y = (
            value
            if positive
            else -value
        )

        offset = (
            0.020
            if positive
            else -0.020
        )

        label_y = (
            min(y + offset, 1.01)
            if positive
            else max(y + offset, -1.01)
        )

        axis.text(
            float(mz[index]),
            label_y,
            f"{float(mz[index]):.1f}",
            rotation=90,
            ha="center",
            va=(
                "bottom"
                if positive
                else "top"
            ),
            fontsize=6.5,
            color=color,
            clip_on=True,
            zorder=5,
        )


def atom_color(
    symbol: str,
) -> str:
    colors = {
        "N": "#3155D5",
        "O": "#E52B2B",
        "S": "#C99A00",
        "P": "#D77A00",
        "F": "#2E9B55",
        "Cl": "#2E9B55",
        "Br": "#8B3A2E",
        "I": "#7048A8",
    }

    return colors.get(
        symbol,
        INK,
    )


def draw_vector_molecule(
    axis: plt.Axes,
    smiles: str,
) -> None:
    axis.set_axis_off()
    axis.set_facecolor("white")

    if not smiles:
        return

    molecule = Chem.MolFromSmiles(
        str(smiles)
    )

    if molecule is None:
        return

    molecule = Chem.RemoveHs(
        molecule
    )

    rdDepictor.Compute2DCoords(
        molecule
    )

    conformer = molecule.GetConformer()

    coordinates = np.array(
        [
            [
                conformer
                .GetAtomPosition(index)
                .x,
                -conformer
                .GetAtomPosition(index)
                .y,
            ]
            for index
            in range(
                molecule.GetNumAtoms()
            )
        ],
        dtype=float,
    )

    if coordinates.size == 0:
        return

    coordinates -= coordinates.mean(
        axis=0
    )

    span = max(
        float(
            np.ptp(
                coordinates,
                axis=0,
            ).max()
        ),
        1.0,
    )

    coordinates /= span

    def draw_line(
        point_1: np.ndarray,
        point_2: np.ndarray,
        offset: float,
        linewidth: float,
    ) -> None:
        direction = (
            point_2
            - point_1
        )

        norm = float(
            np.linalg.norm(
                direction
            )
        )

        if norm <= 0.0:
            return

        perpendicular = np.array(
            [
                -direction[1],
                direction[0],
            ]
        ) / norm

        shift = (
            perpendicular
            * offset
        )

        axis.plot(
            [
                point_1[0] + shift[0],
                point_2[0] + shift[0],
            ],
            [
                point_1[1] + shift[1],
                point_2[1] + shift[1],
            ],
            color="#111820",
            linewidth=linewidth,
            solid_capstyle="round",
            zorder=1,
        )

    for bond in molecule.GetBonds():
        point_1 = coordinates[
            bond.GetBeginAtomIdx()
        ]

        point_2 = coordinates[
            bond.GetEndAtomIdx()
        ]

        order = float(
            bond.GetBondTypeAsDouble()
        )

        if bond.GetIsAromatic():
            draw_line(
                point_1,
                point_2,
                0.0,
                1.5,
            )

            draw_line(
                point_1,
                point_2,
                0.018,
                0.7,
            )

        elif order >= 2.9:
            draw_line(
                point_1,
                point_2,
                -0.020,
                1.15,
            )

            draw_line(
                point_1,
                point_2,
                0.0,
                1.15,
            )

            draw_line(
                point_1,
                point_2,
                0.020,
                1.15,
            )

        elif order >= 1.9:
            draw_line(
                point_1,
                point_2,
                -0.014,
                1.25,
            )

            draw_line(
                point_1,
                point_2,
                0.014,
                1.25,
            )

        else:
            draw_line(
                point_1,
                point_2,
                0.0,
                1.5,
            )

    for index, atom in enumerate(
        molecule.GetAtoms()
    ):
        symbol = atom.GetSymbol()
        charge = int(
            atom.GetFormalCharge()
        )

        if (
            symbol == "C"
            and charge == 0
        ):
            continue

        charge_text = ""

        if charge == 1:
            charge_text = "+"

        elif charge > 1:
            charge_text = (
                f"{charge}+"
            )

        elif charge == -1:
            charge_text = "−"

        elif charge < -1:
            charge_text = (
                f"{abs(charge)}−"
            )

        x_value, y_value = coordinates[
            index
        ]

        axis.text(
            x_value,
            y_value,
            symbol + charge_text,
            ha="center",
            va="center",
            fontsize=11.5,
            color=atom_color(
                symbol
            ),
            bbox={
                "facecolor":
                    "white",

                "edgecolor":
                    "none",

                "pad":
                    0.55,
            },
            zorder=3,
        )

    padding = 0.08

    axis.set_xlim(
        coordinates[:, 0].min()
        - padding,
        coordinates[:, 0].max()
        + padding,
    )

    axis.set_ylim(
        coordinates[:, 1].max()
        + padding,
        coordinates[:, 1].min()
        - padding,
    )

    axis.set_aspect(
        "equal",
        adjustable="box",
    )


def add_molecule_inset(
    axis: plt.Axes,
    smiles: str,
) -> None:
    # Enlarged and moved to the far-right lower blank region.
    inset = axis.inset_axes(
        [
            0.660,  # left: 向图内移动
            0.035,  # bottom
            0.320,  # width: 放大
            0.390,  # height: 放大
        ],
        zorder=10,
    )

    inset.patch.set_facecolor(
        "white"
    )

    inset.patch.set_alpha(
        1.00
    )

    draw_vector_molecule(
        inset,
        smiles,
    )


def draw_panel(
    axis: plt.Axes,
    true_mz: np.ndarray,
    true_intensity: np.ndarray,
    pred_mz: np.ndarray,
    pred_intensity: np.ndarray,
    smiles: str,
    panel: str,
    title: str,
) -> None:
    true_mz, true_intensity = (
        collapse_peaks(
            true_mz,
            true_intensity,
        )
    )

    pred_mz, pred_intensity = (
        collapse_peaks(
            pred_mz,
            pred_intensity,
        )
    )

    true_height = sqrt_relative(
        true_intensity
    )

    pred_height = sqrt_relative(
        pred_intensity
    )

    true_keep = (
        (true_mz >= 50.0)
        & (true_mz <= 450.0)
        & (true_height >= 0.035)
    )

    pred_keep = (
        (pred_mz >= 50.0)
        & (pred_mz <= 450.0)
        & (pred_height >= 0.035)
    )

    true_mz = true_mz[
        true_keep
    ]

    true_height = true_height[
        true_keep
    ]

    pred_mz = pred_mz[
        pred_keep
    ]

    pred_height = pred_height[
        pred_keep
    ]

    axis.axhline(
        0.0,
        color="#818995",
        linewidth=0.8,
        zorder=1,
    )

    axis.vlines(
        true_mz,
        0.0,
        true_height,
        color=OBSERVED_BLUE,
        linewidth=0.65,
        alpha=0.96,
        zorder=3,
    )

    axis.vlines(
        pred_mz,
        0.0,
        -pred_height,
        color=PREDICTED_ORANGE,
        linewidth=0.65,
        alpha=0.96,
        zorder=3,
    )

    annotate_major_peaks(
        axis,
        true_mz,
        true_height,
        positive=True,
        color=OBSERVED_BLUE,
        top_n=6,
    )

    annotate_major_peaks(
        axis,
        pred_mz,
        pred_height,
        positive=False,
        color=PREDICTED_ORANGE,
        top_n=5,
    )

    axis.set_xlim(
        50.0,
        450.0,
    )

    axis.set_ylim(
        -1.18,
        1.18,
    )

    axis.set_xticks(
        [
            50,
            100,
            150,
            200,
            250,
            300,
            350,
            400,
            450,
        ]
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

    axis.set_xlabel(
        "Mass/Charge (m/z)",
        fontsize=10.5,
    )

    axis.set_ylabel(
        "Square root of\nrelative intensity",
        fontsize=10.5,
    )

    axis.grid(
        axis="y",
        color=GRID,
        linewidth=0.55,
        alpha=0.70,
        zorder=0,
    )

    axis.tick_params(
        axis="both",
        labelsize=8.5,
        width=0.8,
        length=4,
    )

    for spine in axis.spines.values():
        spine.set_color(
            SPINE
        )

        spine.set_linewidth(
            0.8
        )

    # Panel label and title are inside the panel and cannot overlap.
    axis.text(
        0.006,
        1.025,
        f"({panel})",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=11.5,
        fontweight="bold",
        color=INK,
        clip_on=False,
    )

    axis.text(
        0.055,
        1.025,
        title,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=11.5,
        fontweight="bold",
        color=INK,
        clip_on=False,
    )

    axis.text(
        0.012,
        0.955,
        "Observed",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
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
        fontsize=8.2,
        fontweight="bold",
        color=PREDICTED_ORANGE,
    )

    add_molecule_inset(
        axis,
        smiles,
    )


def main() -> None:
    if (
        not CACHE_NPZ.is_file()
        or not CACHE_JSON.is_file()
    ):
        raise FileNotFoundError(
            "Missing fixed-case cache. "
            "Do not rescan; check the existing cache files."
        )

    arrays = np.load(
        CACHE_NPZ
    )

    metadata = json.loads(
        CACHE_JSON.read_text(
            encoding="utf-8"
        )
    )

    plt.rcParams.update({
        "font.family":
            "DejaVu Sans",

        "pdf.fonttype":
            42,

        "ps.fonttype":
            42,

        "svg.fonttype":
            "none",
    })

    figure, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(
            13.8,
            9.4,
        ),
        facecolor="white",
    )

    draw_panel(
        axes[0],
        arrays[
            "random_true_mz"
        ],
        arrays[
            "random_true_intensity"
        ],
        arrays[
            "random_pred_mz"
        ],
        arrays[
            "random_pred_intensity"
        ],
        metadata[
            "random"
        ].get(
            "smiles",
            "",
        ),
        panel="a",
        title="Random test example",
    )

    draw_panel(
        axes[1],
        arrays[
            "scaffold_true_mz"
        ],
        arrays[
            "scaffold_true_intensity"
        ],
        arrays[
            "scaffold_pred_mz"
        ],
        arrays[
            "scaffold_pred_intensity"
        ],
        metadata[
            "scaffold"
        ].get(
            "smiles",
            "",
        ),
        panel="b",
        title="Scaffold test example",
    )

    figure.subplots_adjust(
        left=0.075,
        right=0.988,
        top=0.965,
        bottom=0.075,
        hspace=0.25,
    )

    figure.savefig(
        OUT_PNG,
        dpi=1200,
        bbox_inches="tight",
        facecolor="white",
    )

    figure.savefig(
        OUT_PDF,
        bbox_inches="tight",
        facecolor="white",
    )

    figure.savefig(
        OUT_SVG,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(
        figure
    )

    print("FIGURE GENERATED")
    print(OUT_PNG)
    print(OUT_PDF)
    print(OUT_SVG)


if __name__ == "__main__":
    main()
