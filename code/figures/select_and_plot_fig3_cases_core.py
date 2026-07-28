from __future__ import annotations

# --- FIG3_CORE_PATH_BOOTSTRAP ---
import sys as _fig3_sys
from pathlib import Path as _Fig3Path

_FIG3_PROJECT_ROOT = (
    _Fig3Path(__file__)
    .resolve()
    .parents[2]
)

for _fig3_candidate in (
    _FIG3_PROJECT_ROOT / "code" / "src",
    _FIG3_PROJECT_ROOT / "code",
    _FIG3_PROJECT_ROOT,
):
    _fig3_value = str(_fig3_candidate)

    if _fig3_value not in _fig3_sys.path:
        _fig3_sys.path.insert(
            0,
            _fig3_value,
        )
# --- END FIG3_CORE_PATH_BOOTSTRAP ---

import sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parents[2]
for _p in [str(_PROJECT_ROOT), str(_PROJECT_ROOT / "code" / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

#!/usr/bin/env python3

# --- local package path bootstrap ---
import sys
from pathlib import Path as _BootstrapPath

_PROJECT_ROOT_FOR_IMPORT = (
    _BootstrapPath(__file__)
    .resolve()
    .parents[2]
)

for _candidate in (
    _PROJECT_ROOT_FOR_IMPORT / "code" / "src",
    _PROJECT_ROOT_FOR_IMPORT / "code",
    _PROJECT_ROOT_FOR_IMPORT,
):
    _candidate_str = str(_candidate)

    if _candidate_str not in sys.path:
        sys.path.insert(
            0,
            _candidate_str,
        )
# --- end local package path bootstrap ---

import argparse
import importlib.util
import pickle
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

try:
    from rdkit import Chem
    from rdkit.Chem import Draw
except Exception:
    Chem = None
    Draw = None


ROOT = Path(__file__).resolve().parents[2]

PROTOCOLS = {
    "Random": "molecule_disjoint_3seeds",
    "Scaffold": "scaffold_disjoint_3seeds",
}

SEEDS = (42, 43, 44)


# FERA-MS palette
NAVY = "#173F7A"
BLUE = "#3B82C4"
INK = "#1E2A38"
MUTED = "#667085"
GRID = "#D9DEE5"
OBSERVED = "#252A31"


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(
        name,
        str(path),
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Cannot import module: {path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def choose_file(
    paths: list[Path],
    label: str,
) -> Path:
    paths = [
        path
        for path in paths
        if path.is_file()
    ]

    if not paths:
        raise FileNotFoundError(
            f"Cannot find {label}"
        )

    def score(path: Path):
        text = str(path).lower()

        value = 0

        if "final_locked_evaluation" in text:
            value += 20

        if "final" in text:
            value += 5

        if "/old/" in text:
            value -= 20

        return (
            value,
            -len(text),
        )

    return sorted(
        paths,
        key=score,
        reverse=True,
    )[0]


def seed_root(
    protocol_directory: str,
    seed: int,
) -> Path:
    path = (
        ROOT
        / "runs"
        / "experiments"
        / protocol_directory
        / f"seed_{seed}"
    )

    if not path.is_dir():
        raise FileNotFoundError(path)

    return path


def locate_artifacts(
    protocol_directory: str,
    seed: int,
) -> dict[str, Path]:
    current_seed_root = seed_root(
        protocol_directory,
        seed,
    )

    run_root = (
        current_seed_root
        / "v2e_full_063"
    )

    if not run_root.is_dir():
        candidates = [
            path
            for path
            in current_seed_root.rglob(
                "v2e_full_063"
            )
            if path.is_dir()
        ]

        if not candidates:
            raise FileNotFoundError(
                f"No v2e_full_063 under "
                f"{current_seed_root}"
            )

        run_root = candidates[0]

    config_candidates = list(
        current_seed_root.rglob(
            "v2c_ce_trajectory_ablation/"
            "control/config.yml"
        )
    )

    return {
        "seed_root":
            current_seed_root,

        "run_root":
            run_root,

        "backbone":
            choose_file(
                list(
                    run_root.rglob(
                        "r160_best_state.pt"
                    )
                ),
                "R160 checkpoint",
            ),

        "reranker":
            choose_file(
                list(
                    run_root.rglob(
                        "r170_regressor.pkl"
                    )
                ),
                "R172D reranker",
            ),

        "allocator":
            choose_file(
                list(
                    run_root.rglob(
                        "r184_allocator_best.pt"
                    )
                ),
                "R184B allocator",
            ),

        "per_spectrum":
            choose_file(
                list(
                    run_root.rglob(
                        "test_per_spectrum.csv"
                    )
                ),
                "test_per_spectrum.csv",
            ),

        "config":
            choose_file(
                config_candidates,
                "V2C control config",
            ),
    }


def load_three_seed_scores(
    protocol_directory: str,
) -> pd.DataFrame:
    frames = []

    for seed in SEEDS:
        artifacts = locate_artifacts(
            protocol_directory,
            seed,
        )

        frame = pd.read_csv(
            artifacts["per_spectrum"]
        )

        required = {
            "spec_id",
            "ce",
            "cos",
            "jss",
        }

        missing = (
            required
            - set(frame.columns)
        )

        if missing:
            raise RuntimeError(
                f"{artifacts['per_spectrum']} "
                f"missing columns: "
                f"{sorted(missing)}"
            )

        frame = frame[
            [
                "spec_id",
                "ce",
                "cos",
                "jss",
            ]
        ].copy()

        frame = frame.rename(
            columns={
                "ce":
                    f"ce_{seed}",

                "cos":
                    f"cbin_{seed}",

                "jss":
                    f"jss_{seed}",
            }
        )

        frames.append(frame)

    merged = frames[0]

    for frame in frames[1:]:
        merged = merged.merge(
            frame,
            on="spec_id",
            how="inner",
            validate="one_to_one",
        )

    cbin_columns = [
        f"cbin_{seed}"
        for seed in SEEDS
    ]

    jss_columns = [
        f"jss_{seed}"
        for seed in SEEDS
    ]

    ce_columns = [
        f"ce_{seed}"
        for seed in SEEDS
    ]

    merged["ce"] = (
        merged[ce_columns]
        .mean(axis=1)
    )

    merged["cbin_mean"] = (
        merged[cbin_columns]
        .mean(axis=1)
    )

    merged["cbin_sd"] = (
        merged[cbin_columns]
        .std(axis=1, ddof=1)
    )

    merged["jss_mean"] = (
        merged[jss_columns]
        .mean(axis=1)
    )

    merged["jss_sd"] = (
        merged[jss_columns]
        .std(axis=1, ddof=1)
    )

    merged["consensus_score"] = (
        0.60
        * merged["cbin_mean"]
        +
        0.40
        * merged["jss_mean"]
    )

    return merged


def build_candidate_pool(
    scores: pd.DataFrame,
    protocol: str,
    pool_size: int,
) -> pd.DataFrame:
    lower = scores[
        "consensus_score"
    ].quantile(0.75)

    upper = scores[
        "consensus_score"
    ].quantile(0.95)

    pool = scores[
        (
            scores["consensus_score"]
            >= lower
        )
        &
        (
            scores["consensus_score"]
            <= upper
        )
    ].copy()

    if protocol == "Random":
        preferred = pool[
            (
                pool["ce"] > 20.0
            )
            &
            (
                pool["ce"] <= 40.0
            )
        ]
    else:
        preferred = pool[
            pool["ce"] > 40.0
        ]

        if len(preferred) < 12:
            preferred = pool[
                (
                    pool["ce"] > 20.0
                )
                &
                (
                    pool["ce"] <= 40.0
                )
            ]

    if len(preferred) >= 12:
        pool = preferred

    pool = pool.sort_values(
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
    )

    return (
        pool
        .head(pool_size)
        .reset_index(drop=True)
    )


def predict_with_feature_names(
    regressor: Any,
    features: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    score_clip: float,
) -> torch.Tensor:
    array = (
        features
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    feature_names = getattr(
        regressor,
        "feature_names_in_",
        None,
    )

    if feature_names is None:
        feature_names = getattr(
            regressor,
            "feature_name_",
            None,
        )

    predict_input: Any = array

    if feature_names is not None:
        names = [
            str(name)
            for name
            in list(feature_names)
        ]

        if len(names) == int(
            array.shape[1]
        ):
            predict_input = pd.DataFrame(
                array,
                columns=names,
            )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        score = (
            regressor
            .predict(predict_input)
            .astype(np.float32)
        )

    score = np.clip(
        score,
        -float(score_clip),
        float(score_clip),
    )

    return torch.from_numpy(
        score
    ).to(
        device=device,
        dtype=dtype,
    )


def load_seed42_model(
    protocol_directory: str,
):
    artifacts = locate_artifacts(
        protocol_directory,
        42,
    )

    diagnostics = (
        ROOT
        / "train"
        / "_impl"
        / "refinement_steps"
    )

    candidate_reranker = load_module(
        diagnostics
        / "candidate_reranker.py",
        f"r172_{protocol_directory}",
    )

    spectrum_allocator = load_module(
        diagnostics
        / "spectrum_allocator.py",
        f"r184_{protocol_directory}",
    )

    spectrum_allocator.lgbm_predict = (
        predict_with_feature_names
    )

    from ms2spectra.workflow import (
        load_config,
        init_dataset,
        init_dataloader,
    )

    from ms2spectra.training import (
        FragGNNPL
    )

    template_path = (
        ROOT
        / "runs"
        / "_config"
        / "template.yml"
    )

    config = load_config(
        template_path,
        artifacts["config"],
    )

    config = (
        candidate_reranker
        .force_r160_arch(config)
    )

    _, test_dataset = init_dataset(
        config,
        splits=(
            "val",
            "test",
        ),
    )

    test_loader = init_dataloader(
        test_dataset,
        config,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    backbone = FragGNNPL(
        **config
    )

    backbone_state = (
        candidate_reranker
        .load_state_dict_any(
            artifacts["backbone"]
        )
    )

    backbone.load_state_dict(
        backbone_state,
        strict=False,
    )

    backbone = backbone.to(
        device
    )

    backbone.eval()

    for parameter in (
        backbone.parameters()
    ):
        parameter.requires_grad_(False)

    with artifacts[
        "reranker"
    ].open("rb") as handle:
        reranker_package = (
            pickle.load(handle)
        )

    regressor = (
        reranker_package["model"]
    )

    allocator_package = torch.load(
        artifacts["allocator"],
        map_location="cpu",
        weights_only=False,
    )

    saved_arguments = dict(
        allocator_package["args"]
    )

    allocator_arguments = (
        SimpleNamespace(
            **saved_arguments
        )
    )

    extra_schema = (
        allocator_package.get(
            "extra_schema",
            reranker_package.get(
                "extra_schema",
                [],
            ),
        )
    )

    allocator = (
        spectrum_allocator
        .ResidualAllocator(
            input_dim=int(
                allocator_package[
                    "input_dim"
                ]
            ),

            hidden=int(
                saved_arguments[
                    "hidden"
                ]
            ),

            layers=int(
                saved_arguments[
                    "layers"
                ]
            ),

            dropout=float(
                saved_arguments[
                    "dropout"
                ]
            ),

            score_clip=float(
                saved_arguments[
                    "score_clip"
                ]
            ),
        )
        .to(device)
    )

    allocator.load_state_dict(
        allocator_package["model"]
    )

    allocator.eval()

    return {
        "loader":
            test_loader,

        "device":
            device,

        "backbone":
            backbone,

        "allocator":
            allocator,

        "regressor":
            regressor,

        "args":
            allocator_arguments,

        "extra_schema":
            extra_schema,

        "candidate_reranker":
            candidate_reranker,

        "spectrum_allocator":
            spectrum_allocator,
    }


def collapse_peaks(
    mz: np.ndarray,
    intensity: np.ndarray,
):
    mz = np.asarray(
        mz,
        dtype=float,
    ).reshape(-1)

    intensity = np.asarray(
        intensity,
        dtype=float,
    ).reshape(-1)

    valid = (
        np.isfinite(mz)
        &
        np.isfinite(intensity)
        &
        (mz > 0)
        &
        (intensity > 0)
    )

    mz = mz[valid]
    intensity = intensity[valid]

    if mz.size == 0:
        return mz, intensity

    rounded = np.round(
        mz,
        6,
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


def normalize_to_100(
    intensity: np.ndarray,
):
    intensity = np.asarray(
        intensity,
        dtype=float,
    )

    if (
        intensity.size == 0
        or intensity.max() <= 0
    ):
        return intensity

    return (
        100.0
        * intensity
        / intensity.max()
    )


@torch.no_grad()
def collect_spectra(
    model_pack,
    wanted_ids: set[int],
):
    backbone = model_pack[
        "backbone"
    ]

    allocator = model_pack[
        "allocator"
    ]

    regressor = model_pack[
        "regressor"
    ]

    loader = model_pack[
        "loader"
    ]

    device = model_pack[
        "device"
    ]

    r172 = model_pack[
        "candidate_reranker"
    ]

    r184 = model_pack[
        "spectrum_allocator"
    ]

    arguments = model_pack[
        "args"
    ]

    extra_schema = model_pack[
        "extra_schema"
    ]

    collected = {}

    for batch in loader:
        batch = r172.move_to_device(
            batch,
            device,
        )

        (
            result,
            features,
            lgbm_score,
            target_mass,
            positive,
        ) = r184.build_batch_tensors(
            backbone,
            batch,
            r172,
            regressor,
            extra_schema,
            arguments,
            split="test",
        )

        output = r184.forward_allocator(
            backbone,
            allocator,
            batch,
            result,
            features,
            lgbm_score,
            target_mass,
            r172,
            arguments,
        )

        identifiers = (
            result["unique_id"]
            .detach()
            .cpu()
            .reshape(-1)
            .numpy()
            .astype(int)
        )

        ce, _ = r172.find_ce(
            batch
        )

        ce = (
            ce
            .detach()
            .cpu()
            .reshape(-1)
            .numpy()
        )

        predicted_mz = (
            result["pred_mzs"]
            .detach()
            .cpu()
            .numpy()
        )

        predicted_probability = (
            output["new_logp"]
            .exp()
            .detach()
            .cpu()
            .numpy()
        )

        predicted_batch = (
            result["pred_batch_idxs"]
            .detach()
            .cpu()
            .numpy()
            .astype(int)
        )

        true_mz = (
            result["true_mzs"]
            .detach()
            .cpu()
            .numpy()
        )

        true_probability = (
            result["true_logprobs"]
            .exp()
            .detach()
            .cpu()
            .numpy()
        )

        true_batch = (
            result["true_batch_idxs"]
            .detach()
            .cpu()
            .numpy()
            .astype(int)
        )

        cosine = (
            output["cos"]
            .detach()
            .cpu()
            .numpy()
        )

        jss = (
            output["jss"]
            .detach()
            .cpu()
            .numpy()
        )

        for local_index, spec_id in enumerate(
            identifiers
        ):
            spec_id = int(spec_id)

            if spec_id not in wanted_ids:
                continue

            pred_mz_i, pred_int_i = (
                collapse_peaks(
                    predicted_mz[
                        predicted_batch
                        == local_index
                    ],
                    predicted_probability[
                        predicted_batch
                        == local_index
                    ],
                )
            )

            true_mz_i, true_int_i = (
                collapse_peaks(
                    true_mz[
                        true_batch
                        == local_index
                    ],
                    true_probability[
                        true_batch
                        == local_index
                    ],
                )
            )

            true_percent = (
                normalize_to_100(
                    true_int_i
                )
            )

            rich_peak_count = int(
                np.sum(
                    true_percent >= 0.5
                )
            )

            collected[spec_id] = {
                "spec_id":
                    spec_id,

                "ce":
                    float(
                        ce[local_index]
                    ),

                "seed42_cbin":
                    float(
                        cosine[local_index]
                    ),

                "seed42_jss":
                    float(
                        jss[local_index]
                    ),

                "true_mz":
                    true_mz_i,

                "true_intensity":
                    true_int_i,

                "pred_mz":
                    pred_mz_i,

                "pred_intensity":
                    pred_int_i,

                "rich_peak_count":
                    rich_peak_count,
            }

        if wanted_ids.issubset(
            collected.keys()
        ):
            break

    return collected


def select_best_rich_case(
    pool: pd.DataFrame,
    spectra: dict,
    min_peaks: int,
):
    rows = []

    for row in pool.itertuples(
        index=False
    ):
        case = spectra.get(
            int(row.spec_id)
        )

        if case is None:
            continue

        rows.append({
            **case,

            "cbin_mean":
                float(row.cbin_mean),

            "cbin_sd":
                float(row.cbin_sd),

            "jss_mean":
                float(row.jss_mean),

            "jss_sd":
                float(row.jss_sd),

            "consensus_score":
                float(
                    row.consensus_score
                ),
        })

    frame = pd.DataFrame(rows)

    if frame.empty:
        raise RuntimeError(
            "No candidate spectra extracted."
        )

    eligible = frame[
        frame["rich_peak_count"]
        >= min_peaks
    ].copy()

    if eligible.empty:
        eligible = frame.copy()

    richness_rank = (
        eligible["rich_peak_count"]
        .rank(pct=True)
    )

    quality_rank = (
        eligible["consensus_score"]
        .rank(pct=True)
    )

    instability = (
        0.5
        * eligible["cbin_sd"]
        .rank(pct=True)
        +
        0.5
        * eligible["jss_sd"]
        .rank(pct=True)
    )

    eligible["selection_score"] = (
        0.55
        * richness_rank
        +
        0.35
        * quality_rank
        +
        0.10
        * (1.0 - instability)
    )

    best = eligible.sort_values(
        [
            "selection_score",
            "rich_peak_count",
            "consensus_score",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    ).iloc[0]

    return best.to_dict()


def first_column(
    frame: pd.DataFrame,
    aliases: tuple[str, ...],
):
    lookup = {
        str(column).lower():
            column
        for column
        in frame.columns
    }

    for alias in aliases:
        if alias.lower() in lookup:
            return lookup[
                alias.lower()
            ]

    return None


def reset_identifier(
    frame: pd.DataFrame,
    identifier: str,
):
    if identifier in frame.columns:
        return frame.copy()

    frame = frame.reset_index()

    if (
        identifier
        not in frame.columns
        and "index" in frame.columns
    ):
        frame = frame.rename(
            columns={
                "index":
                    identifier
            }
        )

    return frame


def add_metadata(
    case: dict,
):
    output = dict(case)

    spec_paths = [
        path
        for path
        in (
            ROOT
            / "data"
            / "proc"
        ).rglob("spec_df.pkl")
        if "safe19659" in str(path)
    ]

    mol_paths = [
        path
        for path
        in (
            ROOT
            / "data"
            / "proc"
        ).rglob("mol_df.pkl")
        if "safe19659" in str(path)
    ]

    if not spec_paths:
        return output

    spec_df = pd.read_pickle(
        spec_paths[0]
    )

    spec_df = reset_identifier(
        spec_df,
        "spec_id",
    )

    selected_spec = spec_df[
        spec_df["spec_id"].astype(int)
        == int(case["spec_id"])
    ]

    if selected_spec.empty:
        return output

    selected_spec = (
        selected_spec.iloc[0]
    )

    mol_id_column = first_column(
        spec_df,
        (
            "mol_id",
            "molecule_id",
        ),
    )

    precursor_column = first_column(
        spec_df,
        (
            "precursor_mz",
            "precursor_mass",
            "parent_mz",
            "parentmass",
        ),
    )

    if mol_id_column:
        output["mol_id"] = (
            selected_spec[
                mol_id_column
            ]
        )

    if precursor_column:
        output["precursor_mz"] = (
            selected_spec[
                precursor_column
            ]
        )

    if (
        not mol_paths
        or "mol_id" not in output
    ):
        return output

    mol_df = pd.read_pickle(
        mol_paths[0]
    )

    mol_df = reset_identifier(
        mol_df,
        "mol_id",
    )

    selected_mol = mol_df[
        mol_df["mol_id"].astype(str)
        == str(output["mol_id"])
    ]

    if selected_mol.empty:
        return output

    selected_mol = selected_mol.iloc[0]

    smiles_column = first_column(
        mol_df,
        (
            "smiles",
            "canonical_smiles",
            "smiles_canonical",
            "mol_smiles",
        ),
    )

    formula_column = first_column(
        mol_df,
        (
            "formula",
            "molecular_formula",
            "mol_formula",
        ),
    )

    if smiles_column:
        output["smiles"] = (
            selected_mol[
                smiles_column
            ]
        )

    if formula_column:
        output["formula"] = (
            selected_mol[
                formula_column
            ]
        )

    return output


def safe_float(
    value,
    default=np.nan,
):
    try:
        number = float(value)

        if np.isfinite(number):
            return number

    except Exception:
        pass

    return default


def draw_structure(
    smiles,
):
    if (
        Chem is None
        or Draw is None
        or smiles is None
        or str(smiles) == "nan"
    ):
        return None

    molecule = Chem.MolFromSmiles(
        str(smiles)
    )

    if molecule is None:
        return None

    return Draw.MolToImage(
        molecule,
        size=(
            650,
            280,
        ),
    )


def annotate_peaks(
    axis,
    mz,
    intensity,
    positive,
    color,
):
    if len(mz) == 0:
        return

    indices = (
        np.argsort(intensity)[::-1]
    )

    indices = [
        index
        for index
        in indices
        if intensity[index] >= 10.0
    ][:6]

    for index in indices:
        y = (
            intensity[index]
            if positive
            else -intensity[index]
        )

        axis.text(
            mz[index],
            y
            + (
                3.0
                if positive
                else -3.0
            ),
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


def plot_case(
    figure,
    top_cell,
    bottom_cell,
    case,
    panel,
    protocol,
):
    meta_axis = figure.add_subplot(
        top_cell
    )

    spectrum_axis = figure.add_subplot(
        bottom_cell
    )

    meta_axis.axis("off")

    structure = draw_structure(
        case.get("smiles")
    )

    if structure is not None:
        meta_axis.imshow(
            structure
        )

        meta_axis.set_aspect(
            "auto"
        )

    meta_axis.text(
        0.00,
        1.03,
        panel,
        transform=meta_axis.transAxes,
        fontsize=14,
        fontweight="bold",
        color=INK,
        va="bottom",
    )

    meta_axis.text(
        0.07,
        1.03,
        f"{protocol} test example",
        transform=meta_axis.transAxes,
        fontsize=11.5,
        fontweight="bold",
        color=INK,
        va="bottom",
    )

    precursor = safe_float(
        case.get("precursor_mz")
    )

    metadata = [
        f"spec ID = {case['spec_id']}",
        f"ACE = {case['ce']:.1f} eV",
    ]

    if np.isfinite(precursor):
        metadata.append(
            f"precursor m/z = "
            f"{precursor:.4f}"
        )

    metadata.extend([
        f"CBIN = "
        f"{case['seed42_cbin']:.3f}",

        f"JSS = "
        f"{case['seed42_jss']:.3f}",

        f"observed peaks = "
        f"{case['rich_peak_count']}",
    ])

    meta_axis.text(
        0.98,
        0.91,
        "\n".join(metadata),
        transform=meta_axis.transAxes,
        ha="right",
        va="top",
        fontsize=7.9,
        color=MUTED,
        bbox={
            "boxstyle":
                "round,pad=0.35",

            "facecolor":
                "white",

            "edgecolor":
                GRID,

            "linewidth":
                0.8,
        },
    )

    true_mz = np.asarray(
        case["true_mz"],
        dtype=float,
    )

    true_intensity = (
        normalize_to_100(
            case["true_intensity"]
        )
    )

    pred_mz = np.asarray(
        case["pred_mz"],
        dtype=float,
    )

    pred_intensity = (
        normalize_to_100(
            case["pred_intensity"]
        )
    )

    true_keep = (
        true_intensity >= 0.5
    )

    pred_keep = (
        pred_intensity >= 0.5
    )

    true_mz = true_mz[
        true_keep
    ]

    true_intensity = (
        true_intensity[
            true_keep
        ]
    )

    pred_mz = pred_mz[
        pred_keep
    ]

    pred_intensity = (
        pred_intensity[
            pred_keep
        ]
    )

    spectrum_axis.axhline(
        0,
        color="#7A8491",
        linewidth=0.8,
    )

    spectrum_axis.vlines(
        true_mz,
        0,
        true_intensity,
        color=OBSERVED,
        linewidth=1.0,
    )

    spectrum_axis.vlines(
        pred_mz,
        0,
        -pred_intensity,
        color=BLUE,
        linewidth=1.0,
    )

    annotate_peaks(
        spectrum_axis,
        true_mz,
        true_intensity,
        True,
        OBSERVED,
    )

    annotate_peaks(
        spectrum_axis,
        pred_mz,
        pred_intensity,
        False,
        BLUE,
    )

    maxima = [
        100.0,
    ]

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
            precursor
        )

    spectrum_axis.set_xlim(
        0,
        max(maxima) * 1.03,
    )

    spectrum_axis.set_ylim(
        -112,
        112,
    )

    spectrum_axis.set_yticks(
        [
            -100,
            -50,
            0,
            50,
            100,
        ]
    )

    spectrum_axis.set_yticklabels(
        [
            "100",
            "50",
            "0",
            "50",
            "100",
        ]
    )

    spectrum_axis.set_xlabel(
        "m/z",
        fontsize=9,
    )

    spectrum_axis.set_ylabel(
        "Relative intensity",
        fontsize=9,
    )

    spectrum_axis.grid(
        axis="y",
        color=GRID,
        linewidth=0.6,
        alpha=0.8,
    )

    spectrum_axis.spines[
        "top"
    ].set_visible(False)

    spectrum_axis.spines[
        "right"
    ].set_visible(False)

    spectrum_axis.tick_params(
        labelsize=8,
    )

    spectrum_axis.text(
        0.01,
        0.95,
        "Observed",
        transform=spectrum_axis.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        fontweight="bold",
        color=OBSERVED,
    )

    spectrum_axis.text(
        0.01,
        0.05,
        "FERA-MS",
        transform=spectrum_axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
        color=BLUE,
    )


def save_selection_table(
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

            "observed_peak_count":
                case["rich_peak_count"],

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
        default=48,
    )

    parser.add_argument(
        "--min_peaks",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--output_prefix",
        default=(
            "Fig3_representative_"
            "cases_preview"
        ),
    )

    arguments = parser.parse_args()

    output_directory = (
        ROOT
        / "figure"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected_cases = []

    for protocol, directory in (
        PROTOCOLS.items()
    ):
        print()
        print(
            "=" * 72
        )
        print(
            protocol,
            "three-seed consensus selection"
        )
        print(
            "=" * 72
        )

        scores = load_three_seed_scores(
            directory
        )

        pool = build_candidate_pool(
            scores,
            protocol,
            arguments.pool_size,
        )

        print(
            "test spectra:",
            len(scores),
        )

        print(
            "candidate pool:",
            len(pool),
        )

        print(
            "consensus range:",
            f"{pool['consensus_score'].min():.4f}",
            "to",
            f"{pool['consensus_score'].max():.4f}",
        )

        model_pack = load_seed42_model(
            directory
        )

        wanted_ids = set(
            pool["spec_id"]
            .astype(int)
            .tolist()
        )

        spectra = collect_spectra(
            model_pack,
            wanted_ids,
        )

        selected = select_best_rich_case(
            pool,
            spectra,
            arguments.min_peaks,
        )

        selected["protocol"] = (
            protocol
        )

        selected = add_metadata(
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
            "observed peaks:",
            selected[
                "rich_peak_count"
            ],
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
            arguments.output_prefix
            + "_selected_cases.csv"
        )
    )

    save_selection_table(
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

    figure = plt.figure(
        figsize=(
            11.2,
            6.6,
        ),
        facecolor="white",
    )

    grid = figure.add_gridspec(
        2,
        2,
        height_ratios=[
            0.34,
            0.66,
        ],
        width_ratios=[
            1.0,
            1.0,
        ],
        hspace=0.08,
        wspace=0.14,
    )

    plot_case(
        figure,
        grid[0, 0],
        grid[1, 0],
        selected_cases[0],
        "A",
        "Random",
    )

    plot_case(
        figure,
        grid[0, 1],
        grid[1, 1],
        selected_cases[1],
        "B",
        "Scaffold",
    )

    png_path = (
        output_directory
        / (
            arguments.output_prefix
            + ".png"
        )
    )

    pdf_path = (
        output_directory
        / (
            arguments.output_prefix
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
    print(
        "=" * 72
    )
    print(
        "GENERATED"
    )
    print(
        "=" * 72
    )
    print(csv_path)
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
