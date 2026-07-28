#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from rdkit import Chem
from rdkit.Chem import rdDepictor


ROOT = Path(__file__).resolve().parents[2]

for candidate in (
    ROOT / "code" / "src",
    ROOT / "code",
    ROOT,
):
    value = str(candidate)

    if value not in sys.path:
        sys.path.insert(0, value)


OUT_DIR = (
    ROOT
    / "figure"
    / "supp_ce_examples"
)

MANIFEST = (
    OUT_DIR
    / "ce_case_selected_locked.csv"
)

CACHE_NPZ = (
    OUT_DIR
    / "Supp_CE_cases_cache.npz"
)

CACHE_JSON = (
    OUT_DIR
    / "Supp_CE_cases_metadata.json"
)

OUT_PNG = (
    OUT_DIR
    / "Supp_Fig_CE_resolved_examples.png"
)

OUT_PDF = (
    OUT_DIR
    / "Supp_Fig_CE_resolved_examples.pdf"
)

OUT_SVG = (
    OUT_DIR
    / "Supp_Fig_CE_resolved_examples.svg"
)

TABLE_CSV = (
    OUT_DIR
    / "Supp_Table_CE_resolved_cases.csv"
)

TABLE_TEX = (
    OUT_DIR
    / "Supp_Table_CE_resolved_cases.tex"
)


BLUE = "#2F80C3"
ORANGE = "#F28E2B"
INK = "#20262E"
GRID = "#D9DEE5"
SPINE = "#69727D"


def load_core():
    path = (
        Path(__file__).resolve().parent
        / "select_and_plot_fig3_cases_core.py"
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing core helper: {path}"
        )

    spec = importlib.util.spec_from_file_location(
        "fig3_core_supp",
        str(path),
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            f"Cannot import {path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


def collapse_peaks(
    mz,
    intensity,
    decimals=5,
):
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
        & (mz > 0)
        & (intensity > 0)
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


def normalized_percent(
    intensity,
):
    values = np.asarray(
        intensity,
        dtype=float,
    )

    if (
        values.size == 0
        or float(values.max()) <= 0
    ):
        return values

    return (
        100.0
        * values
        / float(values.max())
    )


def sqrt_relative(
    intensity,
):
    values = np.asarray(
        intensity,
        dtype=float,
    )

    if (
        values.size == 0
        or float(values.max()) <= 0
    ):
        return values

    relative = (
        values
        / float(values.max())
    )

    return (
        np.sqrt(
            np.clip(
                relative,
                0,
                None,
            )
        )
        * 0.86
    )


def export_cache():
    if not MANIFEST.is_file():
        raise FileNotFoundError(
            MANIFEST
        )

    manifest = pd.read_csv(
        MANIFEST
    )

    required = {
        "split",
        "mol_id",
        "spec_id",
        "ace",
        "ace_stratum",
    }

    missing = (
        required
        - set(manifest.columns)
    )

    if missing:
        raise RuntimeError(
            "Manifest missing columns: "
            f"{sorted(missing)}"
        )

    core = load_core()

    arrays = {}
    metadata = {
        "cases": [],
    }

    for split in (
        "random",
        "scaffold",
    ):
        rows = manifest[
            manifest["split"]
            .astype(str)
            .str.lower()
            == split
        ].copy()

        if len(rows) != 3:
            raise RuntimeError(
                f"Expected 3 rows for {split}, "
                f"found {len(rows)}"
            )

        rows = rows.sort_values(
            "ace"
        ).reset_index(
            drop=True
        )

        protocol_name = (
            "Random"
            if split == "random"
            else "Scaffold"
        )

        directory = (
            core.PROTOCOLS[
                protocol_name
            ]
        )

        wanted = set(
            rows["spec_id"]
            .astype(int)
            .tolist()
        )

        print()
        print("=" * 72)
        print(
            f"Exporting {protocol_name}: "
            f"{sorted(wanted)}"
        )
        print("=" * 72)

        model_pack = (
            core.load_seed42_model(
                directory
            )
        )

        spectra = (
            core.collect_spectra(
                model_pack,
                wanted,
            )
        )

        missing_ids = (
            wanted
            - set(spectra)
        )

        if missing_ids:
            raise RuntimeError(
                "Could not export "
                f"{protocol_name} spec IDs: "
                f"{sorted(missing_ids)}"
            )

        for row in rows.itertuples(
            index=False
        ):
            spec_id = int(
                row.spec_id
            )

            case = core.add_metadata(
                spectra[spec_id]
            )

            stratum = str(
                row.ace_stratum
            ).lower()

            key = (
                f"{split}_{stratum}"
            )

            arrays[
                f"{key}_true_mz"
            ] = np.asarray(
                case["true_mz"],
                dtype=np.float64,
            )

            arrays[
                f"{key}_true_intensity"
            ] = np.asarray(
                case["true_intensity"],
                dtype=np.float64,
            )

            arrays[
                f"{key}_pred_mz"
            ] = np.asarray(
                case["pred_mz"],
                dtype=np.float64,
            )

            arrays[
                f"{key}_pred_intensity"
            ] = np.asarray(
                case["pred_intensity"],
                dtype=np.float64,
            )

            true_percent = (
                normalized_percent(
                    case["true_intensity"]
                )
            )

            peak_count = int(
                np.sum(
                    true_percent >= 0.5
                )
            )

            smiles = case.get(
                "smiles",
                "",
            )

            formula = case.get(
                "formula",
                "",
            )

            metadata["cases"].append({
                "split":
                    split,

                "protocol":
                    protocol_name,

                "mol_id":
                    int(row.mol_id),

                "spec_id":
                    spec_id,

                "ace":
                    float(row.ace),

                "ace_stratum":
                    stratum,

                "seed42_cbin":
                    float(
                        case["seed42_cbin"]
                    ),

                "seed42_jss":
                    float(
                        case["seed42_jss"]
                    ),

                "observed_peaks":
                    peak_count,

                "smiles":
                    ""
                    if pd.isna(smiles)
                    else str(smiles),

                "formula":
                    ""
                    if pd.isna(formula)
                    else str(formula),
            })

        del model_pack

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    np.savez_compressed(
        CACHE_NPZ,
        **arrays,
    )

    CACHE_JSON.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("CACHE:", CACHE_NPZ)
    print("META :", CACHE_JSON)


def atom_color(
    symbol,
):
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
    axis,
    smiles,
):
    axis.set_axis_off()

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

    def line(
        point_1,
        point_2,
        offset=0.0,
        linewidth=1.2,
    ):
        direction = (
            point_2
            - point_1
        )

        norm = float(
            np.linalg.norm(
                direction
            )
        )

        if norm <= 0:
            return

        perpendicular = np.array([
            -direction[1],
            direction[0],
        ]) / norm

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
            line(
                point_1,
                point_2,
                0.0,
                1.25,
            )

            line(
                point_1,
                point_2,
                0.016,
                0.55,
            )

        elif order >= 2.9:
            line(
                point_1,
                point_2,
                -0.019,
                1.0,
            )

            line(
                point_1,
                point_2,
                0.0,
                1.0,
            )

            line(
                point_1,
                point_2,
                0.019,
                1.0,
            )

        elif order >= 1.9:
            line(
                point_1,
                point_2,
                -0.013,
                1.05,
            )

            line(
                point_1,
                point_2,
                0.013,
                1.05,
            )

        else:
            line(
                point_1,
                point_2,
                0.0,
                1.25,
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

        charge_text = (
            "+"
            if charge == 1
            else (
                "−"
                if charge == -1
                else ""
            )
        )

        x_value, y_value = (
            coordinates[index]
        )

        axis.text(
            x_value,
            y_value,
            symbol + charge_text,
            ha="center",
            va="center",
            fontsize=8.5,
            color=atom_color(
                symbol
            ),
            bbox={
                "facecolor":
                    "white",

                "edgecolor":
                    "none",

                "pad":
                    0.45,
            },
        )

    padding = 0.09

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


def top_indices(
    mz,
    height,
    top_n,
    separation=18.0,
):
    selected = []

    for index in np.argsort(
        height
    )[::-1]:
        if height[index] < 0.30:
            continue

        current = float(
            mz[index]
        )

        separated = all(
            abs(
                current
                - float(mz[chosen])
            )
            >= separation
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
        key=lambda index: float(
            mz[index]
        ),
    )


def annotate_peaks(
    axis,
    mz,
    height,
    positive,
    color,
    top_n,
):
    for index in top_indices(
        mz,
        height,
        top_n,
    ):
        y_value = (
            float(height[index])
            if positive
            else -float(height[index])
        )

        label_y = (
            min(
                y_value + 0.018,
                0.98,
            )
            if positive
            else max(
                y_value - 0.018,
                -0.98,
            )
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
            fontsize=5.8,
            color=color,
            clip_on=True,
        )


def draw_mirror(
    axis,
    true_mz,
    true_intensity,
    pred_mz,
    pred_intensity,
    x_limit,
    panel,
    show_y,
    show_labels,
):
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

    keep_true = (
        (true_mz >= x_limit[0])
        & (true_mz <= x_limit[1])
        & (true_height >= 0.035)
    )

    keep_pred = (
        (pred_mz >= x_limit[0])
        & (pred_mz <= x_limit[1])
        & (pred_height >= 0.035)
    )

    true_mz = true_mz[
        keep_true
    ]

    true_height = true_height[
        keep_true
    ]

    pred_mz = pred_mz[
        keep_pred
    ]

    pred_height = pred_height[
        keep_pred
    ]

    axis.axhline(
        0,
        color="#818995",
        linewidth=0.7,
    )

    axis.vlines(
        true_mz,
        0,
        true_height,
        color=BLUE,
        linewidth=0.55,
    )

    axis.vlines(
        pred_mz,
        0,
        -pred_height,
        color=ORANGE,
        linewidth=0.55,
    )

    annotate_peaks(
        axis,
        true_mz,
        true_height,
        True,
        BLUE,
        3,
    )

    annotate_peaks(
        axis,
        pred_mz,
        pred_height,
        False,
        ORANGE,
        2,
    )

    axis.set_xlim(
        *x_limit
    )

    axis.set_ylim(
        -1.08,
        1.08,
    )

    axis.grid(
        axis="y",
        color=GRID,
        linewidth=0.45,
        alpha=0.70,
    )

    axis.tick_params(
        labelsize=7,
        width=0.7,
        length=3,
    )

    for spine in axis.spines.values():
        spine.set_color(
            SPINE
        )

        spine.set_linewidth(
            0.7
        )

    axis.text(
        0.015,
        0.95,
        f"({panel})",
        transform=axis.transAxes,
        fontsize=8.5,
        fontweight="bold",
        ha="left",
        va="top",
    )

    if show_labels:
        axis.text(
            0.02,
            0.89,
            "Observed",
            transform=axis.transAxes,
            fontsize=7,
            fontweight="bold",
            color=BLUE,
            ha="left",
            va="top",
        )

        axis.text(
            0.02,
            0.08,
            "FERA-MS",
            transform=axis.transAxes,
            fontsize=7,
            fontweight="bold",
            color=ORANGE,
            ha="left",
            va="bottom",
        )

    axis.set_yticks([
        -1.0,
        -0.5,
        0,
        0.5,
        1.0,
    ])

    if show_y:
        axis.set_ylabel(
            "Square root of\nrelative intensity",
            fontsize=8,
        )

        axis.set_yticklabels([
            "1.0",
            "0.5",
            "0",
            "0.5",
            "1.0",
        ])

    else:
        axis.set_yticklabels([])


def write_table(
    metadata,
):
    frame = pd.DataFrame(
        metadata["cases"]
    )

    order_map = {
        "low": 0,
        "middle": 1,
        "high": 2,
    }

    frame["_order"] = (
        frame["ace_stratum"]
        .map(order_map)
    )

    frame = (
        frame
        .sort_values([
            "split",
            "_order",
        ])
        .drop(
            columns="_order"
        )
    )

    frame.to_csv(
        TABLE_CSV,
        index=False,
    )

    lines = [
        r"\begin{table*}[!t]",
        (
            r"\caption{Collision-energy-resolved representative "
            r"cases used in Supplementary "
            r"Fig.~\ref{fig:supp_ce_examples}. Peak counts denote "
            r"observed peaks with at least 0.5\% normalized relative "
            r"intensity. CBIN and JSS correspond to the plotted "
            r"locked seed-42 predictions.}"
        ),
        r"\label{tab:supp_ce_cases}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        (
            r"\begin{tabular*}{\textwidth}"
            r"{@{\extracolsep{\fill}}lllrrrrr}"
        ),
        r"\toprule",
        (
            r"Split & Molecule & ACE stratum & ACE (eV) & "
            r"Spec ID & Observed peaks & CBIN & JSS \\"
        ),
        r"\midrule",
    ]

    for row in frame.itertuples(
        index=False
    ):
        lines.append(
            f"{row.protocol} & "
            f"{row.mol_id} & "
            f"{row.ace_stratum.capitalize()} & "
            f"{row.ace:.1f} & "
            f"{row.spec_id} & "
            f"{row.observed_peaks} & "
            f"{row.seed42_cbin:.3f} & "
            f"{row.seed42_jss:.3f} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular*}",
        r"\end{table*}",
    ])

    TABLE_TEX.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def create_figure():
    if (
        not CACHE_NPZ.is_file()
        or not CACHE_JSON.is_file()
    ):
        raise FileNotFoundError(
            "Missing cache; run once with "
            "--refresh-cache"
        )

    arrays = np.load(
        CACHE_NPZ
    )

    metadata = json.loads(
        CACHE_JSON.read_text(
            encoding="utf-8"
        )
    )

    cases = {
        (
            case["split"],
            case["ace_stratum"],
        ):
            case
        for case
        in metadata["cases"]
    }

    row_limits = {}

    for split in (
        "random",
        "scaffold",
    ):
        maxima = [
            50.0,
        ]

        for stratum in (
            "low",
            "middle",
            "high",
        ):
            key = (
                f"{split}_{stratum}"
            )

            for suffix in (
                "true_mz",
                "pred_mz",
            ):
                values = np.asarray(
                    arrays[
                        f"{key}_{suffix}"
                    ],
                    dtype=float,
                )

                if values.size:
                    maxima.append(
                        float(
                            np.nanmax(values)
                        )
                    )

        maximum = max(
            maxima
        )

        x_max = max(
            250.0,
            np.ceil(
                maximum / 50.0
            )
            * 50.0,
        )

        row_limits[split] = (
            50.0,
            x_max,
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

    figure = plt.figure(
        figsize=(
            15.5,
            7.4,
        ),
        facecolor="white",
    )

    grid = figure.add_gridspec(
        2,
        4,
        width_ratios=[
            0.80,
            1.35,
            1.35,
            1.35,
        ],
        left=0.035,
        right=0.99,
        top=0.93,
        bottom=0.09,
        wspace=0.12,
        hspace=0.28,
    )

    panel_letters = iter(
        "abcdef"
    )

    for row_index, split in enumerate(
        (
            "random",
            "scaffold",
        )
    ):
        structure_axis = (
            figure.add_subplot(
                grid[
                    row_index,
                    0,
                ]
            )
        )

        middle_case = cases[
            (
                split,
                "middle",
            )
        ]

        draw_vector_molecule(
            structure_axis,
            middle_case.get(
                "smiles",
                "",
            ),
        )

        structure_axis.text(
            0.5,
            1.02,
            (
                f"{'(A)' if row_index == 0 else '(B)'}  "
                f"{split.capitalize()} test molecule\n"
                f"Molecule {middle_case['mol_id']}"
            ),
            transform=structure_axis.transAxes,
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )

        for column_index, stratum in enumerate(
            (
                "low",
                "middle",
                "high",
            ),
            start=1,
        ):
            axis = figure.add_subplot(
                grid[
                    row_index,
                    column_index,
                ]
            )

            key = (
                f"{split}_{stratum}"
            )

            draw_mirror(
                axis,
                arrays[
                    f"{key}_true_mz"
                ],
                arrays[
                    f"{key}_true_intensity"
                ],
                arrays[
                    f"{key}_pred_mz"
                ],
                arrays[
                    f"{key}_pred_intensity"
                ],
                row_limits[split],
                next(panel_letters),
                show_y=(
                    column_index == 1
                ),
                show_labels=(
                    column_index == 1
                ),
            )

            case = cases[
                (
                    split,
                    stratum,
                )
            ]

            if row_index == 0:
                axis.set_title(
                    (
                        f"{stratum.capitalize()} ACE\n"
                        f"{case['ace']:.0f} eV"
                    ),
                    fontsize=10,
                    fontweight="bold",
                    pad=8,
                )

            if row_index == 1:
                axis.set_xlabel(
                    "Mass/Charge (m/z)",
                    fontsize=8.5,
                )

    figure.savefig(
        OUT_PNG,
        dpi=900,
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

    write_table(
        metadata
    )

    print()
    print("FIGURE:", OUT_PNG)
    print("FIGURE:", OUT_PDF)
    print("FIGURE:", OUT_SVG)
    print("TABLE :", TABLE_CSV)
    print("TABLE :", TABLE_TEX)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--refresh-cache",
        action="store_true",
    )

    arguments = parser.parse_args()

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if arguments.refresh_cache:
        export_cache()

    create_figure()


if __name__ == "__main__":
    main()
