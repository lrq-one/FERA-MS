#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


PROJECT = Path(os.environ.get("FERA_MS_ROOT", Path(__file__).resolve().parents[3])).resolve()

BASELINE = (
    Path(os.environ.get("FERA_MS_BASELINE_ROOT", Path(__file__).resolve().parents[1]))
)

REPO = (
    Path(
        os.environ.get(
            "FERA_MS_BASELINE_SOURCE",
            BASELINE / "source" / "fragnnet",
        )
    )
)

SRC = REPO / "src"

ROOT = (
    Path(os.environ.get("FERA_MS_RUNS_DIR", PROJECT / "runs"))
    / "experiments"
    / "molecular_retrieval"
    / "pubchem_legacy_full"
)

EXP5 = ROOT / "baseline_experiment5"
FROZEN = EXP5 / "_frozen_inputs"
COMMON = EXP5 / "_common_plan_v1"

RUN_PLAN = (
    FROZEN
    / "experiment5_run_plan.csv"
)

TEMPLATE = (
    REPO
    / "config"
    / "template.yml"
)

CANDIDATE_PROC = (
    ROOT
    / "candidate_d3_20260723"
    / "proc"
)

CANDIDATE_MOL = (
    CANDIDATE_PROC
    / "mol_df.pkl"
)

CANDIDATE_SPEC = (
    CANDIDATE_PROC
    / "spec_df.pkl"
)

FRAG_DP = (
    ROOT
    / "candidate_d3_20260723"
    / "frag"
    / "dags"
)

BIN_RES = 0.01
MZ_MAX = 1500.0

VALID_MODELS = {
    "neims",
    "massformer",
    "fragnnet_d3",
}

sys.path.insert(
    0,
    str(SRC),
)

sys.path.insert(
    1,
    str(REPO),
)

from fragnnet.runner import load_config
from fragnnet.utils.ms2c_utils import (
    model_type_to_model_cls,
    run_spectra_prediction,
)


def now_iso() -> str:
    return time.strftime(
        "%Y-%m-%dT%H:%M:%S%z"
    )


def norm_id(value: Any) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    text = str(value).strip()

    if text.endswith(".0"):
        text = text[:-2]

    return text


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(
                8 * 1024 * 1024
            )

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_numpy_1d(value: Any) -> np.ndarray:
    if value is None:
        return np.empty(
            0,
            dtype=np.float64,
        )

    if torch.is_tensor(value):
        value = (
            value.detach()
            .cpu()
            .numpy()
        )

    try:
        return np.asarray(
            value,
            dtype=np.float64,
        ).reshape(-1)
    except Exception:
        return np.empty(
            0,
            dtype=np.float64,
        )


def peak_list_to_arrays(
    peaks: Any,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(peaks, str):
        text = peaks.strip()

        if not text:
            return (
                np.empty(0),
                np.empty(0),
            )

        try:
            peaks = ast.literal_eval(
                text
            )
        except Exception:
            return (
                np.empty(0),
                np.empty(0),
            )

    if isinstance(
        peaks,
        np.ndarray,
    ):
        peaks = peaks.tolist()

    if not isinstance(
        peaks,
        (list, tuple),
    ):
        return (
            np.empty(0),
            np.empty(0),
        )

    mzs = []
    intensities = []

    for peak in peaks:
        if (
            isinstance(
                peak,
                (
                    list,
                    tuple,
                    np.ndarray,
                ),
            )
            and len(peak) >= 2
        ):
            try:
                mzs.append(
                    float(peak[0])
                )

                intensities.append(
                    float(peak[1])
                )

            except Exception:
                continue

    return (
        np.asarray(
            mzs,
            dtype=np.float64,
        ),
        np.asarray(
            intensities,
            dtype=np.float64,
        ),
    )


def aggregate_sparse_bins(
    mzs: np.ndarray,
    intensities: np.ndarray,
    bin_res: float = BIN_RES,
) -> tuple[np.ndarray, np.ndarray]:
    mzs = np.asarray(
        mzs,
        dtype=np.float64,
    ).reshape(-1)

    intensities = np.asarray(
        intensities,
        dtype=np.float64,
    ).reshape(-1)

    if mzs.shape != intensities.shape:
        return (
            np.empty(
                0,
                dtype=np.int64,
            ),
            np.empty(
                0,
                dtype=np.float64,
            ),
        )

    valid = (
        np.isfinite(mzs)
        & np.isfinite(intensities)
        & (mzs >= 0.0)
        & (mzs <= MZ_MAX)
        & (intensities > 0.0)
    )

    mzs = mzs[valid]
    intensities = intensities[valid]

    if mzs.size == 0:
        return (
            np.empty(
                0,
                dtype=np.int64,
            ),
            np.empty(
                0,
                dtype=np.float64,
            ),
        )

    bins = np.rint(
        mzs / float(bin_res)
    ).astype(np.int64)

    order = np.argsort(
        bins,
        kind="mergesort",
    )

    bins = bins[order]
    intensities = intensities[order]

    unique_bins, starts = np.unique(
        bins,
        return_index=True,
    )

    values = np.add.reduceat(
        intensities,
        starts,
    )

    return unique_bins, values


def sparse_cosine(
    pred_bins: np.ndarray,
    pred_values: np.ndarray,
    query_bins: np.ndarray,
    query_values: np.ndarray,
    sqrt_transform: bool = False,
) -> float:
    pred_values = np.asarray(
        pred_values,
        dtype=np.float64,
    )

    query_values = np.asarray(
        query_values,
        dtype=np.float64,
    )

    if sqrt_transform:
        pred_values = np.sqrt(
            np.clip(
                pred_values,
                0.0,
                None,
            )
        )

        query_values = np.sqrt(
            np.clip(
                query_values,
                0.0,
                None,
            )
        )

    pred_norm = float(
        np.linalg.norm(
            pred_values
        )
    )

    query_norm = float(
        np.linalg.norm(
            query_values
        )
    )

    if (
        pred_norm <= 0.0
        or query_norm <= 0.0
    ):
        return 0.0

    i = 0
    j = 0
    dot = 0.0

    while (
        i < len(pred_bins)
        and j < len(query_bins)
    ):
        if (
            pred_bins[i]
            == query_bins[j]
        ):
            dot += float(
                pred_values[i]
                * query_values[j]
            )

            i += 1
            j += 1

        elif (
            pred_bins[i]
            < query_bins[j]
        ):
            i += 1

        else:
            j += 1

    result = (
        dot
        / (
            pred_norm
            * query_norm
        )
    )

    return float(
        max(
            0.0,
            min(
                1.0,
                result,
            ),
        )
    )


def sparse_jss(
    pred_bins: np.ndarray,
    pred_values: np.ndarray,
    query_bins: np.ndarray,
    query_values: np.ndarray,
) -> float:
    pred_sum = float(
        pred_values.sum()
    )

    query_sum = float(
        query_values.sum()
    )

    if (
        pred_sum <= 0.0
        or query_sum <= 0.0
    ):
        return 0.0

    p = pred_values / pred_sum
    q = query_values / query_sum

    i = 0
    j = 0
    jsd = 0.0

    while (
        i < len(pred_bins)
        or j < len(query_bins)
    ):
        if (
            j >= len(query_bins)
            or (
                i < len(pred_bins)
                and pred_bins[i]
                < query_bins[j]
            )
        ):
            pv = float(p[i])
            qv = 0.0
            i += 1

        elif (
            i >= len(pred_bins)
            or query_bins[j]
            < pred_bins[i]
        ):
            pv = 0.0
            qv = float(q[j])
            j += 1

        else:
            pv = float(p[i])
            qv = float(q[j])
            i += 1
            j += 1

        mean = 0.5 * (
            pv + qv
        )

        if mean <= 0.0:
            continue

        if pv > 0.0:
            jsd += (
                0.5
                * pv
                * math.log(
                    pv / mean
                )
            )

        if qv > 0.0:
            jsd += (
                0.5
                * qv
                * math.log(
                    qv / mean
                )
            )

    return float(
        max(
            0.0,
            min(
                1.0,
                1.0
                - jsd
                / math.log(2.0),
            ),
        )
    )


def summarize_ranks(
    scores: pd.DataFrame,
    split: str,
    seed: int,
    fixed_targets: set[str],
    available_targets: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = scores.copy()

    scores["_target_norm"] = (
        scores[
            "target_mol_id"
        ].map(norm_id)
    )

    scores["_query_norm"] = (
        scores[
            "query_spec_id"
        ].map(norm_id)
    )

    scores[
        "candidate_connectivity_key"
    ] = (
        scores[
            "candidate_connectivity_key"
        ].astype(str)
    )

    scores["candidate_rank"] = (
        pd.to_numeric(
            scores[
                "candidate_rank"
            ],
            errors="raise",
        )
    )

    scores[
        "is_true_candidate"
    ] = (
        pd.to_numeric(
            scores[
                "is_true_candidate"
            ],
            errors="raise",
        )
        .astype(int)
    )

    true_formula_rows = (
        scores[
            scores[
                "is_true_candidate"
            ]
            == 1
        ]
        .copy()
    )

    if (
        true_formula_rows[
            "candidate_formula"
        ].isna().any()
    ):
        raise RuntimeError(
            "True candidate formula "
            "contains missing values"
        )

    true_formula_by_query = (
        true_formula_rows
        .drop_duplicates(
            "_query_norm"
        )
        .set_index(
            "_query_norm"
        )[
            "candidate_formula"
        ]
        .astype(str)
        .to_dict()
    )

    scores["_true_formula"] = (
        scores[
            "_query_norm"
        ].map(
            true_formula_by_query
        )
    )

    cohort_frames: list[
        tuple[
            str,
            pd.DataFrame,
        ]
    ] = [
        (
            "fixed50",
            scores[
                scores[
                    "_target_norm"
                ].isin(
                    fixed_targets
                )
            ].copy(),
        ),
        (
            "available_pool",
            scores[
                scores[
                    "_target_norm"
                ].isin(
                    available_targets
                )
            ].copy(),
        ),
    ]

    exact = scores[
        scores[
            "_target_norm"
        ].isin(
            fixed_targets
        )
        & (
            scores[
                "candidate_formula"
            ].astype(str)
            == scores[
                "_true_formula"
            ]
        )
    ].copy()

    cohort_frames.append(
        (
            "exact_formula",
            exact,
        )
    )

    rank_frames = []

    metric_columns = [
        "cbin",
        "cbin_sqrt",
        "jss",
    ]

    for cohort, frame in (
        cohort_frames
    ):
        if frame.empty:
            continue

        true_counts = (
            frame.groupby(
                "_query_norm"
            )[
                "is_true_candidate"
            ].sum()
        )

        valid_queries = set(
            true_counts[
                true_counts == 1
            ].index
        )

        frame = frame[
            frame[
                "_query_norm"
            ].isin(
                valid_queries
            )
        ].copy()

        for metric in (
            metric_columns
        ):
            ordered = (
                frame.sort_values(
                    [
                        "_query_norm",
                        metric,
                        "candidate_rank",
                        (
                            "candidate_"
                            "connectivity_key"
                        ),
                    ],
                    ascending=[
                        True,
                        False,
                        True,
                        True,
                    ],
                    kind="mergesort",
                )
                .copy()
            )

            ordered[
                "retrieval_rank"
            ] = (
                ordered.groupby(
                    "_query_norm"
                ).cumcount()
                + 1
            )

            true_rows = ordered[
                ordered[
                    "is_true_candidate"
                ]
                == 1
            ].copy()

            true_rows["split"] = (
                split
            )

            true_rows["seed"] = (
                seed
            )

            true_rows["cohort"] = (
                cohort
            )

            true_rows["method"] = (
                metric
            )

            pool_sizes = (
                ordered.groupby(
                    "_query_norm"
                ).size()
            )

            true_rows[
                "pool_size_scored"
            ] = (
                pool_sizes.reindex(
                    true_rows[
                        "_query_norm"
                    ]
                ).to_numpy()
            )

            rank_frames.append(
                true_rows[
                    [
                        "split",
                        "seed",
                        "cohort",
                        "method",
                        "query_spec_id",
                        "target_mol_id",
                        "query_ace",
                        "retrieval_rank",
                        "pool_size_scored",
                        metric,
                    ]
                ].rename(
                    columns={
                        metric:
                            (
                                "true_candidate_"
                                "similarity"
                            )
                    }
                )
            )

    if not rank_frames:
        raise RuntimeError(
            "No valid ranking frames "
            "were produced"
        )

    rankings = pd.concat(
        rank_frames,
        ignore_index=True,
    )

    summary_rows = []

    for (
        cohort,
        method,
    ), frame in rankings.groupby(
        [
            "cohort",
            "method",
        ],
        sort=True,
    ):
        frame = frame.copy()

        frame["top1"] = (
            frame[
                "retrieval_rank"
            ]
            <= 1
        ).astype(float)

        frame["top5"] = (
            frame[
                "retrieval_rank"
            ]
            <= 5
        ).astype(float)

        frame["top10"] = (
            frame[
                "retrieval_rank"
            ]
            <= 10
        ).astype(float)

        frame["rr"] = (
            1.0
            / frame[
                "retrieval_rank"
            ].astype(float)
        )

        summary_rows.append(
            {
                "split":
                    split,
                "seed":
                    seed,
                "cohort":
                    cohort,
                "method":
                    method,
                "unit":
                    "spectrum_micro",
                "n_spectra":
                    int(
                        len(frame)
                    ),
                "n_molecules":
                    int(
                        frame[
                            "target_mol_id"
                        ]
                        .map(norm_id)
                        .nunique()
                    ),
                "top1":
                    float(
                        frame[
                            "top1"
                        ].mean()
                    ),
                "top5":
                    float(
                        frame[
                            "top5"
                        ].mean()
                    ),
                "top10":
                    float(
                        frame[
                            "top10"
                        ].mean()
                    ),
                "mrr":
                    float(
                        frame[
                            "rr"
                        ].mean()
                    ),
                "median_rank":
                    float(
                        frame[
                            "retrieval_rank"
                        ].median()
                    ),
                "mean_rank":
                    float(
                        frame[
                            "retrieval_rank"
                        ].mean()
                    ),
            }
        )

        molecule = (
            frame.assign(
                _target=frame[
                    "target_mol_id"
                ].map(norm_id)
            )
            .groupby(
                "_target",
                as_index=False,
            )
            .agg(
                top1=(
                    "top1",
                    "mean",
                ),
                top5=(
                    "top5",
                    "mean",
                ),
                top10=(
                    "top10",
                    "mean",
                ),
                rr=(
                    "rr",
                    "mean",
                ),
                retrieval_rank=(
                    "retrieval_rank",
                    "mean",
                ),
            )
        )

        summary_rows.append(
            {
                "split":
                    split,
                "seed":
                    seed,
                "cohort":
                    cohort,
                "method":
                    method,
                "unit":
                    "molecule_macro",
                "n_spectra":
                    int(
                        len(frame)
                    ),
                "n_molecules":
                    int(
                        len(molecule)
                    ),
                "top1":
                    float(
                        molecule[
                            "top1"
                        ].mean()
                    ),
                "top5":
                    float(
                        molecule[
                            "top5"
                        ].mean()
                    ),
                "top10":
                    float(
                        molecule[
                            "top10"
                        ].mean()
                    ),
                "mrr":
                    float(
                        molecule[
                            "rr"
                        ].mean()
                    ),
                "median_rank":
                    float(
                        molecule[
                            "retrieval_rank"
                        ].median()
                    ),
                "mean_rank":
                    float(
                        molecule[
                            "retrieval_rank"
                        ].mean()
                    ),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    expected_keys = {
        (
            cohort,
            method,
            unit,
        )
        for cohort in [
            "available_pool",
            "fixed50",
            "exact_formula",
        ]
        for method in [
            "cbin",
            "cbin_sqrt",
            "jss",
        ]
        for unit in [
            "spectrum_micro",
            "molecule_macro",
        ]
    }

    actual_keys = set(
        zip(
            summary["cohort"],
            summary["method"],
            summary["unit"],
        )
    )

    if actual_keys != expected_keys:
        missing = sorted(
            expected_keys
            - actual_keys
        )

        extra = sorted(
            actual_keys
            - expected_keys
        )

        raise RuntimeError(
            "Retrieval summary key mismatch: "
            f"missing={missing}, "
            f"extra={extra}"
        )

    if len(summary) != 18:
        raise RuntimeError(
            "Expected 18 summary rows, "
            f"found {len(summary)}"
        )

    return rankings, summary


def load_query_bin_cache(
    split: str,
    plan: pd.DataFrame,
) -> dict[
    str,
    tuple[
        np.ndarray,
        np.ndarray,
    ],
]:
    path = (
        COMMON
        / split
        / "query_spectra.pkl.gz"
    )

    if not path.is_file():
        raise FileNotFoundError(
            path
        )

    obj = pd.read_pickle(
        path
    )

    cache = {}

    if isinstance(obj, dict):
        for key, peaks in (
            obj.items()
        ):
            mzs, intensities = (
                peak_list_to_arrays(
                    peaks
                )
            )

            bins, values = (
                aggregate_sparse_bins(
                    mzs,
                    intensities,
                )
            )

            cache[
                norm_id(key)
            ] = (
                bins,
                values,
            )

    elif isinstance(
        obj,
        pd.DataFrame,
    ):
        frame = obj.copy()

        id_column = None

        for candidate in [
            "query_spec_id",
            "spec_id",
            "_query_spec_norm",
        ]:
            if candidate in frame.columns:
                id_column = candidate
                break

        if id_column is None:
            if not isinstance(
                frame.index,
                pd.RangeIndex,
            ):
                frame = (
                    frame.reset_index()
                )

                id_column = (
                    frame.columns[0]
                )

            else:
                raise RuntimeError(
                    "Cannot identify query "
                    "spectrum ID column in "
                    f"{path}; columns="
                    f"{list(frame.columns)}"
                )

        peaks_column = None

        for candidate in [
            "query_peaks",
            "peaks",
        ]:
            if candidate in frame.columns:
                peaks_column = candidate
                break

        if peaks_column is None:
            raise RuntimeError(
                "Cannot identify query peaks "
                f"column in {path}; columns="
                f"{list(frame.columns)}"
            )

        for row in frame[
            [
                id_column,
                peaks_column,
            ]
        ].itertuples(
            index=False,
            name=None,
        ):
            key, peaks = row

            mzs, intensities = (
                peak_list_to_arrays(
                    peaks
                )
            )

            bins, values = (
                aggregate_sparse_bins(
                    mzs,
                    intensities,
                )
            )

            cache[
                norm_id(key)
            ] = (
                bins,
                values,
            )

    else:
        raise TypeError(
            "Unsupported query-spectra "
            f"object: {type(obj)}"
        )

    required_queries = set(
        plan[
            "query_spec_id"
        ].map(norm_id)
    )

    missing = sorted(
        required_queries
        - set(cache)
    )

    if missing:
        raise RuntimeError(
            "Query spectra missing from "
            f"frozen cache: {missing[:20]} "
            f"(n={len(missing)})"
        )

    empty_queries = [
        key
        for key in required_queries
        if (
            len(cache[key][0])
            == 0
        )
    ]

    if empty_queries:
        raise RuntimeError(
            "Frozen query spectra contain "
            "empty valid peak lists: "
            f"{empty_queries[:20]}"
        )

    return cache


def load_candidate_assets():
    if not CANDIDATE_MOL.is_file():
        raise FileNotFoundError(
            CANDIDATE_MOL
        )

    if not CANDIDATE_SPEC.is_file():
        raise FileNotFoundError(
            CANDIDATE_SPEC
        )

    mol_df = pd.read_pickle(
        CANDIDATE_MOL
    ).copy()

    spec_df = pd.read_pickle(
        CANDIDATE_SPEC
    ).copy()

    if "mol_id" not in mol_df:
        raise RuntimeError(
            "Candidate mol_df lacks mol_id"
        )

    if "mol_id" not in spec_df:
        raise RuntimeError(
            "Candidate spec_df lacks mol_id"
        )

    mol_df["_mol_norm"] = (
        mol_df[
            "mol_id"
        ].map(norm_id)
    )

    spec_df["_mol_norm"] = (
        spec_df[
            "mol_id"
        ].map(norm_id)
    )

    if (
        mol_df[
            "_mol_norm"
        ].duplicated().any()
    ):
        raise RuntimeError(
            "Candidate mol_df contains "
            "duplicate mol IDs"
        )

    spec_df = (
        spec_df.drop_duplicates(
            "_mol_norm",
            keep="first",
        )
    )

    spec_index = (
        spec_df.set_index(
            "_mol_norm",
            drop=False,
        )
    )

    return (
        mol_df,
        spec_index,
    )


def build_chunk_inputs(
    chunk: pd.DataFrame,
    candidate_mol: pd.DataFrame,
    candidate_spec_index: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    keys = (
        chunk[
            "candidate_internal_mol_id"
        ].map(norm_id)
    )

    template = (
        candidate_spec_index.reindex(
            keys.tolist()
        )
    )

    if template[
        "mol_id"
    ].isna().any():
        bad = (
            keys[
                template[
                    "mol_id"
                ].isna().to_numpy()
            ]
            .head(20)
            .tolist()
        )

        raise RuntimeError(
            "Candidate spectrum templates "
            f"missing: {bad}"
        )

    template = (
        template.reset_index(
            drop=True
        )
        .drop(
            columns=[
                "_mol_norm",
            ],
            errors="ignore",
        )
        .copy()
    )

    inference_ids = (
        pd.to_numeric(
            chunk[
                "inference_spec_id"
            ],
            errors="raise",
        )
        .astype(np.int64)
        .to_numpy()
    )

    template["spec_id"] = (
        inference_ids
    )

    template["group_id"] = (
        inference_ids
    )

    if (
        "dset_spec_id"
        in template.columns
    ):
        template[
            "dset_spec_id"
        ] = inference_ids

    if "dset" in template.columns:
        template["dset"] = (
            "experiment5_pubchem"
        )

    template["ace"] = (
        pd.to_numeric(
            chunk[
                "query_ace"
            ],
            errors="raise",
        )
        .astype(float)
        .to_numpy()
    )

    used_keys = set(keys)

    mol_input = (
        candidate_mol[
            candidate_mol[
                "_mol_norm"
            ].isin(
                used_keys
            )
        ]
        .drop(
            columns=[
                "_mol_norm",
            ],
            errors="ignore",
        )
        .copy()
    )

    found_keys = set(
        mol_input[
            "mol_id"
        ].map(norm_id)
    )

    missing_mols = sorted(
        used_keys - found_keys
    )

    if missing_mols:
        raise RuntimeError(
            "Candidate molecules missing: "
            f"{missing_mols[:20]}"
        )

    return (
        template,
        mol_input,
    )


def make_model_config(
    config_path: Path,
    eval_batch_size: int,
    num_workers: int,
) -> dict:
    config = load_config(
        str(TEMPLATE),
        str(config_path),
    )

    config[
        "eval_batch_size"
    ] = int(eval_batch_size)

    config[
        "num_workers"
    ] = int(num_workers)

    config[
        "pin_memory"
    ] = bool(
        torch.cuda.is_available()
    )

    config[
        "share_memory"
    ] = False

    for key in [
        "spec_params",
        "mol_params",
    ]:
        value = config.get(key)

        if isinstance(
            value,
            dict,
        ):
            value[
                "preprocess"
            ] = True

    frag_params = config.get(
        "frag_params"
    )

    if isinstance(
        frag_params,
        dict,
    ):
        frag_params[
            "preprocess"
        ] = False

        frag_params[
            "preload"
        ] = False

    return config


def load_model(
    checkpoint: Path,
    config: dict,
    device: torch.device,
):
    model_type = str(
        config[
            "model_type"
        ]
    )

    try:
        model_class = (
            model_type_to_model_cls[
                model_type
            ]
        )
    except KeyError as exc:
        raise RuntimeError(
            "No model class registered for "
            f"model_type={model_type}; "
            f"available={sorted(model_type_to_model_cls)}"
        ) from exc

    model = (
        model_class
        .load_from_checkpoint(
            str(checkpoint),
            map_location="cpu",
            strict=True,
            **config,
        )
    )

    model = model.to(
        device
    )

    model.eval()

    for parameter in (
        model.parameters()
    ):
        parameter.requires_grad = (
            False
        )

    return model


def expected_part_path(
    parts_dir: Path,
    index: int,
    start: int,
    end: int,
) -> Path:
    return (
        parts_dir
        / (
            f"part_{index:05d}_"
            f"{start:09d}_"
            f"{end:09d}.csv.gz"
        )
    )


def validate_score_part(
    path: Path,
    chunk: pd.DataFrame,
) -> bool:
    if not path.is_file():
        return False

    try:
        part = pd.read_csv(
            path,
            usecols=[
                "inference_spec_id",
            ],
        )

    except Exception:
        return False

    expected = (
        chunk[
            "inference_spec_id"
        ].map(norm_id)
        .tolist()
    )

    actual = (
        part[
            "inference_spec_id"
        ].map(norm_id)
        .tolist()
    )

    return (
        len(actual)
        == len(expected)
        and actual == expected
    )


def score_prediction_chunk(
    chunk: pd.DataFrame,
    predictions: pd.DataFrame,
    query_cache: dict,
    split: str,
    seed: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    required_prediction_columns = {
        "spec_id",
        "pred_mzs",
        "pred_ints",
    }

    missing_columns = (
        required_prediction_columns
        - set(predictions.columns)
    )

    if missing_columns:
        raise RuntimeError(
            "Prediction output missing "
            f"columns: {missing_columns}"
        )

    predictions = (
        predictions.copy()
    )

    predictions[
        "_infer_norm"
    ] = (
        predictions[
            "spec_id"
        ].map(norm_id)
    )

    if predictions[
        "_infer_norm"
    ].duplicated().any():
        bad = (
            predictions[
                predictions[
                    "_infer_norm"
                ].duplicated(
                    keep=False
                )
            ][
                "_infer_norm"
            ]
            .head(20)
            .tolist()
        )

        raise RuntimeError(
            "Duplicate prediction IDs: "
            f"{bad}"
        )

    expected_ids = set(
        chunk[
            "inference_spec_id"
        ].map(norm_id)
    )

    output_ids = set(
        predictions[
            "_infer_norm"
        ]
    )

    extra_ids = sorted(
        output_ids
        - expected_ids
    )

    if extra_ids:
        raise RuntimeError(
            "Prediction output contains "
            "unexpected IDs: "
            f"{extra_ids[:20]}"
        )

    prediction_map = {}

    for row in predictions[
        [
            "_infer_norm",
            "pred_mzs",
            "pred_ints",
        ]
    ].itertuples(
        index=False,
        name=None,
    ):
        key, mzs, intensities = row

        prediction_map[key] = (
            mzs,
            intensities,
        )

    output_rows = []
    failed_rows = []

    optional_columns = [
        "candidate_formula",
        "target_morgan_tanimoto",
        "candidate_count",
        "analysis_cohort",
        "candidate_origin",
    ]

    for record in chunk.to_dict(
        orient="records"
    ):
        inference_id = record[
            "inference_spec_id"
        ]

        inference_key = norm_id(
            inference_id
        )

        query_id = record[
            "query_spec_id"
        ]

        query_key = norm_id(
            query_id
        )

        query_bins, query_values = (
            query_cache[
                query_key
            ]
        )

        status = "SUCCESS"
        failure_reason = ""

        cbin = 0.0
        cbin_sqrt = 0.0
        jss = 0.0

        prediction = (
            prediction_map.get(
                inference_key
            )
        )

        if prediction is None:
            status = (
                "MISSING_MODEL_OUTPUT"
            )

            failure_reason = (
                "inference_spec_id absent "
                "from model prediction output"
            )

        else:
            pred_mzs = to_numpy_1d(
                prediction[0]
            )

            pred_intensities = (
                to_numpy_1d(
                    prediction[1]
                )
            )

            if (
                pred_mzs.shape
                != pred_intensities.shape
            ):
                status = (
                    "INVALID_PREDICTION_SHAPE"
                )

                failure_reason = (
                    "pred_mzs and pred_ints "
                    "have different lengths"
                )

            else:
                pred_bins, pred_values = (
                    aggregate_sparse_bins(
                        pred_mzs,
                        pred_intensities,
                    )
                )

                if len(pred_bins) == 0:
                    status = (
                        "EMPTY_VALID_PREDICTION"
                    )

                    failure_reason = (
                        "no finite positive peaks "
                        "inside the scoring m/z range"
                    )

                else:
                    cbin = sparse_cosine(
                        pred_bins,
                        pred_values,
                        query_bins,
                        query_values,
                        sqrt_transform=False,
                    )

                    cbin_sqrt = (
                        sparse_cosine(
                            pred_bins,
                            pred_values,
                            query_bins,
                            query_values,
                            sqrt_transform=True,
                        )
                    )

                    jss = sparse_jss(
                        pred_bins,
                        pred_values,
                        query_bins,
                        query_values,
                    )

                    values = np.asarray(
                        [
                            cbin,
                            cbin_sqrt,
                            jss,
                        ],
                        dtype=np.float64,
                    )

                    if (
                        not np.isfinite(
                            values
                        ).all()
                        or (
                            values < 0.0
                        ).any()
                        or (
                            values > 1.0
                        ).any()
                    ):
                        status = (
                            "INVALID_SCORE"
                        )

                        failure_reason = (
                            "nonfinite or "
                            "out-of-range score"
                        )

                        cbin = 0.0
                        cbin_sqrt = 0.0
                        jss = 0.0

        output = {
            "inference_spec_id":
                inference_id,
            "split":
                split,
            "seed":
                int(seed),
            "query_spec_id":
                query_id,
            "target_mol_id":
                record[
                    "target_mol_id"
                ],
            "query_ace":
                record[
                    "query_ace"
                ],
            "candidate_connectivity_key":
                record[
                    (
                        "candidate_"
                        "connectivity_key"
                    )
                ],
            "candidate_internal_mol_id":
                record[
                    (
                        "candidate_"
                        "internal_mol_id"
                    )
                ],
            "candidate_rank":
                record[
                    "candidate_rank"
                ],
            "is_true_candidate":
                record[
                    "is_true_candidate"
                ],
            "cbin":
                float(cbin),
            "cbin_sqrt":
                float(cbin_sqrt),
            "jss":
                float(jss),
            "prediction_status":
                status,
        }

        for column in optional_columns:
            output[column] = (
                record.get(
                    column,
                    np.nan,
                )
            )

        output_rows.append(output)

        if status != "SUCCESS":
            failed = {
                "inference_spec_id":
                    inference_id,
                "split":
                    split,
                "seed":
                    int(seed),
                "query_spec_id":
                    query_id,
                "target_mol_id":
                    record[
                        "target_mol_id"
                    ],
                "candidate_connectivity_key":
                    record[
                        (
                            "candidate_"
                            "connectivity_key"
                        )
                    ],
                "candidate_internal_mol_id":
                    record[
                        (
                            "candidate_"
                            "internal_mol_id"
                        )
                    ],
                "candidate_rank":
                    record[
                        "candidate_rank"
                    ],
                "is_true_candidate":
                    record[
                        "is_true_candidate"
                    ],
                "reason":
                    status,
                "detail":
                    failure_reason,
            }

            failed_rows.append(
                failed
            )

    scores = pd.DataFrame(
        output_rows
    )

    failures = pd.DataFrame(
        failed_rows
    )

    if len(scores) != len(chunk):
        raise RuntimeError(
            "Scored row count mismatch: "
            f"{len(scores)} vs "
            f"{len(chunk)}"
        )

    expected_order = (
        chunk[
            "inference_spec_id"
        ].map(norm_id)
        .tolist()
    )

    actual_order = (
        scores[
            "inference_spec_id"
        ].map(norm_id)
        .tolist()
    )

    if expected_order != actual_order:
        raise RuntimeError(
            "Score rows are not aligned "
            "with the frozen plan"
        )

    return scores, failures


def load_split_plan(
    split: str,
) -> pd.DataFrame:
    path = (
        COMMON
        / split
        / "expanded_run_plan.csv.gz"
    )

    if not path.is_file():
        raise FileNotFoundError(
            path
        )

    plan = pd.read_csv(
        path,
        low_memory=False,
    )

    # --------------------------------------------------------------
    # Normalize public-plan aliases without changing candidate order.
    # --------------------------------------------------------------
    original_columns = list(
        plan.columns
    )

    if "target_mol_id" not in plan.columns:
        target_aliases = [
            "query_mol_id",
            "query_internal_mol_id",
            "query_target_mol_id",
            "true_mol_id",
            "target_id",
            "target_structure_id",
        ]

        target_source = next(
            (
                column
                for column in target_aliases
                if column in plan.columns
            ),
            None,
        )

        if target_source is None:
            raise RuntimeError(
                "Cannot derive target_mol_id from "
                "the frozen public plan. "
                f"Columns={original_columns}"
            )

        plan["target_mol_id"] = (
            plan[target_source]
        )

        print(
            "> normalized target_mol_id from",
            target_source,
            flush=True,
        )

    # COMPOSITE_FROZEN_FORMULA_JOIN_V2
    #
    # Restore candidate_formula from the frozen candidate master.
    # The join must be target-aware. A connectivity key alone is not
    # globally unique in the frozen PubChem candidate namespace.
    if "candidate_formula" not in plan.columns:
        candidate_master_path = (
            ROOT
            / "inference_ready_pools_20260723"
            / (
                "experiment5_inference_ready_"
                "candidates.csv.gz"
            )
        )

        if not candidate_master_path.is_file():
            raise FileNotFoundError(
                candidate_master_path
            )

        candidate_master = pd.read_csv(
            candidate_master_path,
            low_memory=False,
        )

        def normalize_alias(
            frame,
            canonical,
            aliases,
            label,
        ):
            if canonical in frame.columns:
                return canonical

            source = next(
                (
                    column
                    for column in aliases
                    if column in frame.columns
                ),
                None,
            )

            if source is None:
                return None

            frame[canonical] = frame[source]

            print(
                "> normalized",
                label,
                canonical,
                "from",
                source,
                flush=True,
            )

            return canonical

        normalize_alias(
            candidate_master,
            "target_mol_id",
            [
                "query_target_mol_id",
                "query_mol_id",
                "true_mol_id",
                "target_id",
                "target_structure_id",
            ],
            "candidate master",
        )

        normalize_alias(
            candidate_master,
            "candidate_internal_mol_id",
            [
                "candidate_mol_id",
                "internal_mol_id",
                "candidate_structure_id",
                "candidate_internal_id",
            ],
            "candidate master",
        )

        normalize_alias(
            candidate_master,
            "candidate_connectivity_key",
            [
                "candidate_inchikey14",
                "candidate_inchi_key14",
                "connectivity_key",
                "candidate_connectivity",
            ],
            "candidate master",
        )

        normalize_alias(
            candidate_master,
            "candidate_rank",
            [
                "frozen_candidate_rank",
                "original_candidate_rank",
                "pubchem_rank",
                "pool_rank",
                "rank",
            ],
            "candidate master",
        )

        formula_source = next(
            (
                column
                for column in [
                    "candidate_formula",
                    "frozen_candidate_formula",
                    "candidate_molecular_formula",
                    "molecular_formula",
                    "formula",
                ]
                if column
                in candidate_master.columns
            ),
            None,
        )

        if formula_source is None:
            formula_like = [
                column
                for column
                in candidate_master.columns
                if "formula"
                in column.lower()
            ]

            raise RuntimeError(
                "Frozen candidate master contains "
                "no candidate-formula field. "
                f"Formula-like columns={formula_like}; "
                f"all columns="
                f"{list(candidate_master.columns)}"
            )

        def compose_key(
            frame,
            columns,
        ):
            pieces = [
                frame[column].map(norm_id)
                for column in columns
            ]

            key = pieces[0].copy()

            for piece in pieces[1:]:
                key = (
                    key
                    + "\x1f"
                    + piece
                )

            valid = pd.Series(
                True,
                index=frame.index,
            )

            for piece in pieces:
                valid &= piece.ne("")

            return key, valid

        candidate_identity_options = [
            [
                "target_mol_id",
                "candidate_internal_mol_id",
            ],
            [
                "target_mol_id",
                "candidate_connectivity_key",
                "candidate_rank",
            ],
            [
                "target_mol_id",
                "candidate_connectivity_key",
            ],
            [
                "target_mol_id",
                "candidate_rank",
            ],
            [
                "candidate_internal_mol_id",
            ],
        ]

        selected_columns = None
        selected_formula = None
        diagnostics = []

        for identity_columns in (
            candidate_identity_options
        ):
            if not all(
                column in plan.columns
                and column
                in candidate_master.columns
                for column
                in identity_columns
            ):
                diagnostics.append(
                    {
                        "columns":
                            identity_columns,
                        "status":
                            "MISSING_COLUMNS",
                    }
                )

                continue

            plan_key, plan_valid = (
                compose_key(
                    plan,
                    identity_columns,
                )
            )

            master_key, master_valid = (
                compose_key(
                    candidate_master,
                    identity_columns,
                )
            )

            if not bool(
                plan_valid.all()
            ):
                diagnostics.append(
                    {
                        "columns":
                            identity_columns,
                        "status":
                            "EMPTY_PLAN_IDENTITY",
                        "count":
                            int(
                                (
                                    ~plan_valid
                                ).sum()
                            ),
                    }
                )

                continue

            formula_frame = pd.DataFrame(
                {
                    "_join_key":
                        master_key,
                    "_valid_identity":
                        master_valid,
                    "candidate_formula":
                        candidate_master[
                            formula_source
                        ],
                }
            )

            formula_frame = formula_frame[
                formula_frame[
                    "_valid_identity"
                ]
                & formula_frame[
                    "candidate_formula"
                ].notna()
            ].copy()

            formula_frame[
                "candidate_formula"
            ] = (
                formula_frame[
                    "candidate_formula"
                ]
                .astype(str)
                .str.strip()
            )

            formula_frame = formula_frame[
                formula_frame[
                    "candidate_formula"
                ].ne("")
            ].copy()

            conflicts = (
                formula_frame.groupby(
                    "_join_key"
                )[
                    "candidate_formula"
                ]
                .nunique(
                    dropna=True
                )
            )

            conflicts = conflicts[
                conflicts > 1
            ]

            if len(conflicts):
                diagnostics.append(
                    {
                        "columns":
                            identity_columns,
                        "status":
                            "FORMULA_CONFLICT",
                        "conflict_keys":
                            int(
                                len(conflicts)
                            ),
                        "examples":
                            list(
                                conflicts.index[
                                    :5
                                ]
                            ),
                    }
                )

                continue

            lookup = (
                formula_frame[
                    [
                        "_join_key",
                        "candidate_formula",
                    ]
                ]
                .drop_duplicates(
                    "_join_key",
                    keep="first",
                )
                .set_index(
                    "_join_key"
                )[
                    "candidate_formula"
                ]
            )

            restored = plan_key.map(
                lookup
            )

            missing_count = int(
                restored.isna().sum()
            )

            if missing_count:
                diagnostics.append(
                    {
                        "columns":
                            identity_columns,
                        "status":
                            "INCOMPLETE_COVERAGE",
                        "missing_rows":
                            missing_count,
                    }
                )

                continue

            selected_columns = (
                identity_columns
            )

            selected_formula = (
                restored
            )

            diagnostics.append(
                {
                    "columns":
                        identity_columns,
                    "status":
                        "SELECTED",
                    "covered_rows":
                        int(
                            len(restored)
                        ),
                }
            )

            break

        if selected_columns is None:
            raise RuntimeError(
                "Unable to restore frozen "
                "candidate_formula with a unique, "
                "fully covered composite identity. "
                f"Diagnostics={diagnostics}; "
                f"plan columns={list(plan.columns)}; "
                f"master columns="
                f"{list(candidate_master.columns)}"
            )

        plan["candidate_formula"] = (
            selected_formula.to_numpy()
        )

        if (
            plan[
                "candidate_formula"
            ].isna().any()
        ):
            raise RuntimeError(
                "candidate_formula contains missing "
                "values after selected composite join"
            )

        print(
            "> restored candidate_formula",
            "source=",
            candidate_master_path,
            "formula_column=",
            formula_source,
            "identity_columns=",
            selected_columns,
            "rows=",
            len(plan),
            flush=True,
        )

        print(
            "> formula join diagnostics =",
            diagnostics,
            flush=True,
        )

    required = {
        "inference_spec_id",
        "query_spec_id",
        "target_mol_id",
        "query_ace",
        "candidate_connectivity_key",
        "candidate_internal_mol_id",
        "candidate_rank",
        "is_true_candidate",
        "candidate_formula",
    }

    missing = (
        required
        - set(plan.columns)
    )

    if missing:
        raise RuntimeError(
            "Normalized public plan still "
            "missing columns: "
            f"{sorted(missing)}"
        )

    if plan[
        "inference_spec_id"
    ].duplicated().any():
        raise RuntimeError(
            "Frozen plan contains duplicate "
            "inference_spec_id values"
        )

    expected_ids = [
        str(value)
        for value
        in range(
            1,
            len(plan) + 1,
        )
    ]

    actual_ids = (
        plan[
            "inference_spec_id"
        ].map(norm_id)
        .tolist()
    )

    if actual_ids != expected_ids:
        raise RuntimeError(
            "Frozen plan inference IDs "
            "are not exact 1..N order"
        )

    return plan


def build_fixed_target_sets(
    scores: pd.DataFrame,
) -> tuple[
    set[str],
    set[str],
]:
    work = scores.copy()

    work["_target_norm"] = (
        work[
            "target_mol_id"
        ].map(norm_id)
    )

    pool_sizes = (
        work.groupby(
            "_target_norm"
        )[
            "candidate_connectivity_key"
        ].nunique()
    )

    available = set(
        pool_sizes.index
    )

    fixed = set(
        pool_sizes[
            pool_sizes == 50
        ].index
    )

    if not fixed:
        raise RuntimeError(
            "No fixed-50 targets detected"
        )

    return fixed, available


def concatenate_parts(
    parts_dir: Path,
    expected_rows: int,
) -> pd.DataFrame:
    paths = sorted(
        parts_dir.glob(
            "part_*.csv.gz"
        )
    )

    if not paths:
        raise RuntimeError(
            f"No score parts found: "
            f"{parts_dir}"
        )

    frames = [
        pd.read_csv(
            path,
            low_memory=False,
        )
        for path in paths
    ]

    scores = pd.concat(
        frames,
        ignore_index=True,
    )

    if len(scores) != expected_rows:
        raise RuntimeError(
            "Combined score rows mismatch: "
            f"{len(scores)} vs "
            f"{expected_rows}"
        )

    scores = scores.sort_values(
        "inference_spec_id",
        kind="mergesort",
    ).reset_index(
        drop=True
    )

    if scores[
        "inference_spec_id"
    ].map(norm_id).tolist() != [
        str(value)
        for value
        in range(
            1,
            expected_rows + 1,
        )
    ]:
        raise RuntimeError(
            "Combined score identity/order "
            "audit failed"
        )

    if scores[
        "inference_spec_id"
    ].duplicated().any():
        raise RuntimeError(
            "Combined scores contain "
            "duplicate inference IDs"
        )

    for metric in [
        "cbin",
        "cbin_sqrt",
        "jss",
    ]:
        values = pd.to_numeric(
            scores[metric],
            errors="coerce",
        )

        if (
            values.isna().any()
            or (
                values < 0.0
            ).any()
            or (
                values > 1.0
            ).any()
        ):
            raise RuntimeError(
                f"Invalid final metric "
                f"values in {metric}"
            )

    return scores


def concatenate_failed_parts(
    failed_dir: Path,
) -> pd.DataFrame:
    paths = sorted(
        failed_dir.glob(
            "failed_*.csv.gz"
        )
    )

    if not paths:
        return pd.DataFrame(
            columns=[
                "inference_spec_id",
                "split",
                "seed",
                "query_spec_id",
                "target_mol_id",
                (
                    "candidate_"
                    "connectivity_key"
                ),
                (
                    "candidate_"
                    "internal_mol_id"
                ),
                "candidate_rank",
                "is_true_candidate",
                "reason",
                "detail",
            ]
        )

    return pd.concat(
        [
            pd.read_csv(
                path,
                low_memory=False,
            )
            for path in paths
        ],
        ignore_index=True,
    )


def write_coverage_summary(
    scores: pd.DataFrame,
    failures: pd.DataFrame,
    output_path: Path,
) -> None:
    rows = []

    total_rows = int(
        len(scores)
    )

    total_queries = int(
        scores[
            "query_spec_id"
        ].map(norm_id).nunique()
    )

    total_targets = int(
        scores[
            "target_mol_id"
        ].map(norm_id).nunique()
    )

    total_candidates = int(
        scores[
            "candidate_internal_mol_id"
        ].map(norm_id).nunique()
    )

    failed_ids = set(
        failures[
            "inference_spec_id"
        ].map(norm_id)
    ) if len(failures) else set()

    failed_queries = set(
        failures[
            "query_spec_id"
        ].map(norm_id)
    ) if len(failures) else set()

    rows.append(
        {
            "scope":
                "overall",
            "status":
                "all",
            "candidate_prediction_rows":
                total_rows,
            "query_spectra":
                total_queries,
            "target_molecules":
                total_targets,
            "candidate_structures":
                total_candidates,
            "failed_candidate_rows":
                int(
                    len(failed_ids)
                ),
            "failed_query_spectra":
                int(
                    len(
                        failed_queries
                    )
                ),
            "successful_candidate_rows":
                int(
                    total_rows
                    - len(failed_ids)
                ),
            "candidate_row_coverage":
                float(
                    (
                        total_rows
                        - len(failed_ids)
                    )
                    / max(
                        total_rows,
                        1,
                    )
                ),
        }
    )

    for status, count in (
        scores[
            "prediction_status"
        ]
        .value_counts(
            dropna=False
        )
        .items()
    ):
        rows.append(
            {
                "scope":
                    "prediction_status",
                "status":
                    str(status),
                "candidate_prediction_rows":
                    int(count),
                "query_spectra":
                    np.nan,
                "target_molecules":
                    np.nan,
                "candidate_structures":
                    np.nan,
                "failed_candidate_rows":
                    (
                        0
                        if str(status)
                        == "SUCCESS"
                        else int(count)
                    ),
                "failed_query_spectra":
                    np.nan,
                "successful_candidate_rows":
                    (
                        int(count)
                        if str(status)
                        == "SUCCESS"
                        else 0
                    ),
                "candidate_row_coverage":
                    float(
                        count
                        / max(
                            total_rows,
                            1,
                        )
                    ),
            }
        )

    pd.DataFrame(
        rows
    ).to_csv(
        output_path,
        index=False,
    )


def finalize_run(
    model_name: str,
    split: str,
    seed: int,
    checkpoint: Path,
    config_path: Path,
    output_dir: Path,
    split_plan: pd.DataFrame,
    parts_dir: Path,
    failed_parts_dir: Path,
    chunk_size: int,
    eval_batch_size: int,
    num_workers: int,
    started_at: str,
) -> None:
    scores = concatenate_parts(
        parts_dir,
        expected_rows=len(
            split_plan
        ),
    )

    failures = (
        concatenate_failed_parts(
            failed_parts_dir
        )
    )

    candidate_scores_path = (
        output_dir
        / "candidate_scores.csv.gz"
    )

    scores.to_csv(
        candidate_scores_path,
        index=False,
    )

    failed_path = (
        output_dir
        / "failed_candidates.csv.gz"
    )

    failures.to_csv(
        failed_path,
        index=False,
    )

    fixed_targets, available_targets = (
        build_fixed_target_sets(
            scores
        )
    )

    rankings, summary = (
        summarize_ranks(
            scores=scores,
            split=split,
            seed=seed,
            fixed_targets=fixed_targets,
            available_targets=(
                available_targets
            ),
        )
    )

    rankings_path = (
        output_dir
        / "true_candidate_ranks.csv.gz"
    )

    summary_path = (
        output_dir
        / "retrieval_summary.csv"
    )

    rankings.to_csv(
        rankings_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    coverage_path = (
        output_dir
        / "coverage_summary.csv"
    )

    write_coverage_summary(
        scores,
        failures,
        coverage_path,
    )

    plan_path = (
        COMMON
        / split
        / "expanded_run_plan.csv.gz"
    )

    success = {
        "status":
            "SUCCESS",
        "model":
            model_name,
        "split":
            split,
        "seed":
            int(seed),
        "started_at":
            started_at,
        "completed_at":
            now_iso(),
        "device":
            (
                "cuda:0"
                if torch.cuda.is_available()
                else "cpu"
            ),
        "chunk_size":
            int(chunk_size),
        "eval_batch_size":
            int(eval_batch_size),
        "num_workers":
            int(num_workers),
        "checkpoint":
            str(checkpoint),
        "checkpoint_sha256":
            sha256_file(
                checkpoint
            ),
        "config":
            str(config_path),
        "config_sha256":
            sha256_file(
                config_path
            ),
        "expanded_plan":
            str(plan_path),
        "expanded_plan_sha256":
            sha256_file(
                plan_path
            ),
        "candidate_prediction_rows":
            int(len(scores)),
        "query_spectra":
            int(
                scores[
                    "query_spec_id"
                ].map(norm_id).nunique()
            ),
        "query_molecules":
            int(
                scores[
                    "target_mol_id"
                ].map(norm_id).nunique()
            ),
        "candidate_structures":
            int(
                scores[
                    (
                        "candidate_"
                        "internal_mol_id"
                    )
                ].map(norm_id).nunique()
            ),
        "fixed50_target_count":
            int(
                len(
                    fixed_targets
                )
            ),
        "available_target_count":
            int(
                len(
                    available_targets
                )
            ),
        "failed_candidate_rows":
            int(
                len(failures)
            ),
        "retrieval_summary_rows":
            int(
                len(summary)
            ),
        "candidate_scores":
            str(
                candidate_scores_path
            ),
        "candidate_scores_sha256":
            sha256_file(
                candidate_scores_path
            ),
        "true_candidate_ranks":
            str(
                rankings_path
            ),
        "true_candidate_ranks_sha256":
            sha256_file(
                rankings_path
            ),
        "retrieval_summary":
            str(
                summary_path
            ),
        "retrieval_summary_sha256":
            sha256_file(
                summary_path
            ),
        "coverage_summary":
            str(
                coverage_path
            ),
        "coverage_summary_sha256":
            sha256_file(
                coverage_path
            ),
        "failed_candidates":
            str(
                failed_path
            ),
        "failed_candidates_sha256":
            sha256_file(
                failed_path
            ),
        "predicted_spectra_saved":
            False,
        "predicted_spectra_policy":
            (
                "Scores are computed immediately "
                "and only formal candidate-level "
                "similarities are retained, matching "
                "the locked Ours Experiment 5 "
                "storage policy."
            ),
        "identity_join":
            (
                "one_to_one_by_"
                "inference_spec_id"
            ),
        "bin_width_da":
            BIN_RES,
        "bin_assignment":
            "numpy.rint(mz / 0.01)",
        "rank_tie_break":
            [
                "score descending",
                (
                    "candidate_rank "
                    "ascending"
                ),
                (
                    "candidate_connectivity_"
                    "key ascending"
                ),
            ],
    }

    success_path = (
        output_dir
        / "_SUCCESS.json"
    )

    success_path.write_text(
        json.dumps(
            success,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    shutil.rmtree(
        parts_dir,
        ignore_errors=True,
    )

    shutil.rmtree(
        failed_parts_dir,
        ignore_errors=True,
    )

    progress_path = (
        output_dir
        / "candidate_scores.progress.json"
    )

    if progress_path.exists():
        progress_path.unlink()

    print()
    print("=" * 100)
    print("FORMAL RUN SUCCESS")
    print("=" * 100)
    print(
        "model =",
        model_name,
    )
    print(
        "split =",
        split,
    )
    print(
        "seed =",
        seed,
    )
    print(
        "candidate rows =",
        len(scores),
    )
    print(
        "failed candidates =",
        len(failures),
    )
    print(
        "summary rows =",
        len(summary),
    )
    print(
        "success =",
        success_path,
    )
    print("=" * 100)


def run_one(args) -> None:
    model_name = str(
        args.model
    ).strip().lower()

    if model_name not in (
        VALID_MODELS
    ):
        raise ValueError(
            f"Unsupported model: "
            f"{model_name}"
        )

    split = str(
        args.split
    ).strip().lower()

    if split not in {
        "random",
        "scaffold",
    }:
        raise ValueError(
            f"Unsupported split: {split}"
        )

    seed = int(args.seed)

    if seed not in {
        42,
        43,
        44,
    }:
        raise ValueError(
            f"Unsupported seed: {seed}"
        )

    checkpoint = Path(
        args.checkpoint
    ).resolve()

    config_path = Path(
        args.config
    ).resolve()

    output_dir = Path(
        args.output_dir
    ).resolve()

    success_path = (
        output_dir
        / "_SUCCESS.json"
    )

    if success_path.is_file():
        success = json.loads(
            success_path.read_text(
                encoding="utf-8"
            )
        )

        if (
            success.get("status")
            == "SUCCESS"
            and success.get("model")
            == model_name
            and success.get("split")
            == split
            and int(
                success.get(
                    "seed",
                    -1,
                )
            )
            == seed
        ):
            print(
                "SKIP_COMPLETE",
                success_path,
            )

            return

        raise RuntimeError(
            "Existing _SUCCESS.json does "
            "not match requested run"
        )

    if not checkpoint.is_file():
        raise FileNotFoundError(
            checkpoint
        )

    if not config_path.is_file():
        raise FileNotFoundError(
            config_path
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    started_at = now_iso()

    seed_all(seed)

    split_plan = load_split_plan(
        split
    )

    query_cache = (
        load_query_bin_cache(
            split,
            split_plan,
        )
    )

    candidate_mol, candidate_spec_index = (
        load_candidate_assets()
    )

    split_plan[
        "_candidate_norm"
    ] = (
        split_plan[
            "candidate_internal_mol_id"
        ].map(norm_id)
    )

    candidate_asset_ids = set(
        candidate_mol[
            "mol_id"
        ].map(norm_id)
    )

    missing_assets = sorted(
        set(
            split_plan[
                "_candidate_norm"
            ]
        )
        - candidate_asset_ids
    )

    if missing_assets:
        raise RuntimeError(
            "Frozen plan references missing "
            "candidate assets: "
            f"{missing_assets[:20]}"
        )

    parts_dir = (
        output_dir
        / "candidate_score_parts"
    )

    failed_parts_dir = (
        output_dir
        / "failed_candidate_parts"
    )

    parts_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    failed_parts_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    chunk_size = int(
        args.chunk_size
    )

    ranges = []

    for index, start in enumerate(
        range(
            0,
            len(split_plan),
            chunk_size,
        )
    ):
        end = min(
            start + chunk_size,
            len(split_plan),
        )

        ranges.append(
            (
                index,
                start,
                end,
            )
        )

    missing_ranges = []

    for (
        index,
        start,
        end,
    ) in ranges:
        chunk = split_plan.iloc[
            start:end
        ].copy()

        part_path = (
            expected_part_path(
                parts_dir,
                index,
                start,
                end,
            )
        )

        if validate_score_part(
            part_path,
            chunk,
        ):
            print(
                "VALID_RESUME_PART",
                part_path.name,
            )

            continue

        if part_path.exists():
            quarantine = (
                part_path.with_name(
                    part_path.name
                    + ".invalid_"
                    + time.strftime(
                        "%Y%m%d_%H%M%S"
                    )
                )
            )

            part_path.rename(
                quarantine
            )

            print(
                "QUARANTINED_INVALID_PART",
                quarantine,
            )

        missing_ranges.append(
            (
                index,
                start,
                end,
            )
        )

    print()
    print("=" * 100)
    print("FORMAL RUN")
    print("=" * 100)
    print(
        "model =",
        model_name,
    )
    print(
        "split =",
        split,
    )
    print(
        "seed =",
        seed,
    )
    print(
        "rows =",
        len(split_plan),
    )
    print(
        "chunks total =",
        len(ranges),
    )
    print(
        "chunks pending =",
        len(missing_ranges),
    )
    print(
        "eval batch size =",
        args.eval_batch_size,
    )
    print(
        "checkpoint =",
        checkpoint,
    )
    print("=" * 100)

    model = None
    config = None
    device = None

    if missing_ranges:
        if (
            args.device.startswith(
                "cuda"
            )
            and not torch.cuda.is_available()
        ):
            raise RuntimeError(
                "CUDA requested but "
                "not available"
            )

        device = torch.device(
            args.device
        )

        config = make_model_config(
            config_path,
            eval_batch_size=(
                args.eval_batch_size
            ),
            num_workers=(
                args.num_workers
            ),
        )

        expected_model_type = {
            "neims":
                "neims",
            "massformer":
                "massformer",
            "fragnnet_d3":
                "frag_gnn",
        }[
            model_name
        ]

        if (
            str(
                config[
                    "model_type"
                ]
            )
            != expected_model_type
        ):
            raise RuntimeError(
                "Model/config type mismatch: "
                f"model={model_name}, "
                f"config model_type="
                f"{config['model_type']}"
            )

        model = load_model(
            checkpoint,
            config,
            device,
        )

    completed_rows = 0

    for (
        index,
        start,
        end,
    ) in ranges:
        chunk = split_plan.iloc[
            start:end
        ].copy()

        part_path = (
            expected_part_path(
                parts_dir,
                index,
                start,
                end,
            )
        )

        failed_part_path = (
            failed_parts_dir
            / (
                f"failed_{index:05d}_"
                f"{start:09d}_"
                f"{end:09d}.csv.gz"
            )
        )

        if validate_score_part(
            part_path,
            chunk,
        ):
            completed_rows += (
                len(chunk)
            )

            continue

        print()
        print(
            f"[chunk {index + 1}/"
            f"{len(ranges)}] "
            f"rows {start}:{end}"
        )

        spec_input, mol_input = (
            build_chunk_inputs(
                chunk,
                candidate_mol,
                candidate_spec_index,
            )
        )

        predictions = (
            run_spectra_prediction(
                model=model,
                config_d=config,
                mol_data_ptr=mol_input,
                spec_data_ptr=spec_input,
                device=str(device),
                frag_dp=(
                    str(FRAG_DP)
                    if model_name
                    == "fragnnet_d3"
                    else None
                ),
                magma_dp=None,
                validate=False,
            )
        )

        scores, failures = (
            score_prediction_chunk(
                chunk=chunk,
                predictions=predictions,
                query_cache=query_cache,
                split=split,
                seed=seed,
            )
        )

        temporary_part = (
            part_path.with_suffix(
                part_path.suffix
                + ".tmp"
            )
        )

        scores.to_csv(
            temporary_part,
            index=False,
            compression="gzip",
        )

        temporary_part.replace(
            part_path
        )

        if len(failures):
            temporary_failed = (
                failed_part_path.with_suffix(
                    failed_part_path.suffix
                    + ".tmp"
                )
            )

            failures.to_csv(
                temporary_failed,
                index=False,
                compression="gzip",
            )

            temporary_failed.replace(
                failed_part_path
            )

        elif failed_part_path.exists():
            failed_part_path.unlink()

        completed_rows += len(chunk)

        progress = {
            "status":
                "RUNNING",
            "model":
                model_name,
            "split":
                split,
            "seed":
                seed,
            "completed_rows":
                int(
                    completed_rows
                ),
            "total_rows":
                int(
                    len(split_plan)
                ),
            "completed_chunks":
                int(
                    sum(
                        validate_score_part(
                            expected_part_path(
                                parts_dir,
                                idx,
                                lo,
                                hi,
                            ),
                            split_plan.iloc[
                                lo:hi
                            ],
                        )
                        for idx, lo, hi
                        in ranges
                    )
                ),
            "total_chunks":
                int(
                    len(ranges)
                ),
            "latest_part":
                str(part_path),
            "updated_at":
                now_iso(),
        }

        (
            output_dir
            / "candidate_scores.progress.json"
        ).write_text(
            json.dumps(
                progress,
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        print(
            "CHUNK_SUCCESS",
            part_path.name,
            "rows=",
            len(scores),
            "failed=",
            len(failures),
        )

        del spec_input
        del mol_input
        del predictions
        del scores
        del failures

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    finalize_run(
        model_name=model_name,
        split=split,
        seed=seed,
        checkpoint=checkpoint,
        config_path=config_path,
        output_dir=output_dir,
        split_plan=split_plan,
        parts_dir=parts_dir,
        failed_parts_dir=(
            failed_parts_dir
        ),
        chunk_size=chunk_size,
        eval_batch_size=(
            args.eval_batch_size
        ),
        num_workers=(
            args.num_workers
        ),
        started_at=started_at,
    )


def aggregate_results(
    require_complete: bool,
) -> None:
    if not RUN_PLAN.is_file():
        raise FileNotFoundError(
            RUN_PLAN
        )

    plan = pd.read_csv(
        RUN_PLAN,
        low_memory=False,
    )

    plan = plan[
        plan[
            "model"
        ].astype(str).isin(
            VALID_MODELS
        )
    ].copy()

    if len(plan) != 18:
        raise RuntimeError(
            "Expected 18 formal runs in "
            f"run plan, found {len(plan)}"
        )

    summary_frames = []
    rank_frames = []
    completed = []

    for row in plan.itertuples(
        index=False
    ):
        output_dir = Path(
            row.output_dir
        )

        success_path = (
            output_dir
            / "_SUCCESS.json"
        )

        summary_path = (
            output_dir
            / "retrieval_summary.csv"
        )

        ranks_path = (
            output_dir
            / "true_candidate_ranks.csv.gz"
        )

        if not success_path.is_file():
            continue

        success = json.loads(
            success_path.read_text(
                encoding="utf-8"
            )
        )

        if (
            success.get("status")
            != "SUCCESS"
        ):
            continue

        if not summary_path.is_file():
            raise FileNotFoundError(
                summary_path
            )

        if not ranks_path.is_file():
            raise FileNotFoundError(
                ranks_path
            )

        frame = pd.read_csv(
            summary_path
        )

        if len(frame) != 18:
            raise RuntimeError(
                f"{summary_path} has "
                f"{len(frame)} rows"
            )

        frame.insert(
            0,
            "model",
            row.model,
        )

        summary_frames.append(
            frame
        )

        ranks = pd.read_csv(
            ranks_path,
            low_memory=False,
        )

        ranks.insert(
            0,
            "model",
            row.model,
        )

        rank_frames.append(
            ranks
        )

        completed.append(
            (
                row.model,
                row.split,
                int(row.seed),
            )
        )

    if not summary_frames:
        print(
            "NO_COMPLETED_RUNS_TO_AGGREGATE"
        )

        return

    all_summary = pd.concat(
        summary_frames,
        ignore_index=True,
    )

    all_summary_path = (
        EXP5
        / "all_seed_summaries.csv"
    )

    all_summary.to_csv(
        all_summary_path,
        index=False,
    )

    group_columns = [
        "model",
        "split",
        "cohort",
        "method",
        "unit",
    ]

    metric_columns = [
        "top1",
        "top5",
        "top10",
        "mrr",
        "median_rank",
        "mean_rank",
    ]

    aggregate_rows = []

    for keys, frame in (
        all_summary.groupby(
            group_columns,
            sort=True,
        )
    ):
        row = dict(
            zip(
                group_columns,
                keys,
            )
        )

        row["n_seeds"] = int(
            frame[
                "seed"
            ].nunique()
        )

        row["n_spectra_min"] = int(
            frame[
                "n_spectra"
            ].min()
        )

        row["n_spectra_max"] = int(
            frame[
                "n_spectra"
            ].max()
        )

        row["n_molecules_min"] = int(
            frame[
                "n_molecules"
            ].min()
        )

        row["n_molecules_max"] = int(
            frame[
                "n_molecules"
            ].max()
        )

        for metric in metric_columns:
            values = pd.to_numeric(
                frame[metric],
                errors="raise",
            )

            row[
                f"{metric}_mean"
            ] = float(
                values.mean()
            )

            row[
                f"{metric}_std"
            ] = float(
                values.std(
                    ddof=1
                )
            ) if len(values) > 1 else np.nan

        aggregate_rows.append(
            row
        )

    aggregate = pd.DataFrame(
        aggregate_rows
    )

    aggregate_path = (
        EXP5
        / (
            "experiment5_baselines_"
            "aggregate.csv"
        )
    )

    aggregate.to_csv(
        aggregate_path,
        index=False,
    )

    for model in sorted(
        VALID_MODELS
    ):
        model_dir = (
            EXP5 / model
        )

        model_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        aggregate[
            aggregate[
                "model"
            ]
            == model
        ].to_csv(
            model_dir
            / "aggregate_3seeds.csv",
            index=False,
        )

    main_table = aggregate[
        (
            aggregate[
                "cohort"
            ]
            == "fixed50"
        )
        & (
            aggregate[
                "method"
            ]
            == "cbin_sqrt"
        )
    ].copy()

    main_table.to_csv(
        EXP5
        / (
            "main_fixed50_"
            "cbin_sqrt.csv"
        ),
        index=False,
    )

    all_ranks = pd.concat(
        rank_frames,
        ignore_index=True,
    )

    all_ranks.to_csv(
        EXP5
        / "all_true_candidate_ranks.csv.gz",
        index=False,
    )

    completion = {
        "completed_at":
            now_iso(),
        "completed_runs":
            int(
                len(completed)
            ),
        "expected_runs":
            18,
        "completed_run_keys":
            [
                {
                    "model":
                        model,
                    "split":
                        split,
                    "seed":
                        seed,
                }
                for (
                    model,
                    split,
                    seed,
                )
                in completed
            ],
        "seed_summary_rows":
            int(
                len(
                    all_summary
                )
            ),
        "aggregate_rows":
            int(
                len(
                    aggregate
                )
            ),
        "all_seed_summaries":
            str(
                all_summary_path
            ),
        "aggregate":
            str(
                aggregate_path
            ),
        "main_table":
            str(
                EXP5
                / (
                    "main_fixed50_"
                    "cbin_sqrt.csv"
                )
            ),
    }

    (
        EXP5
        / "completion_status.json"
    ).write_text(
        json.dumps(
            completion,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "AGGREGATE_COMPLETED_RUNS =",
        len(completed),
    )

    print(
        "ALL_SEED_SUMMARY_ROWS =",
        len(all_summary),
    )

    print(
        "AGGREGATE_ROWS =",
        len(aggregate),
    )

    if require_complete:
        if len(completed) != 18:
            raise RuntimeError(
                "Formal matrix incomplete: "
                f"{len(completed)}/18"
            )

        if len(all_summary) != 324:
            raise RuntimeError(
                "Expected 324 seed-level "
                f"summary rows, found "
                f"{len(all_summary)}"
            )

        if len(aggregate) != 108:
            raise RuntimeError(
                "Expected 108 aggregate rows, "
                f"found {len(aggregate)}"
            )

        readiness_path = (
            FROZEN
            / "readiness_v2.json"
        )

        if readiness_path.is_file():
            readiness = json.loads(
                readiness_path.read_text(
                    encoding="utf-8"
                )
            )
        else:
            readiness = {}

        readiness.update(
            {
                "checkpoint_count":
                    18,
                "checkpoint_gate":
                    True,
                "public_plan_gate":
                    True,
                (
                    "three_model_adapter_"
                    "smoke_gate"
                ):
                    True,
                (
                    "three_model_scorer_"
                    "ranker_gate"
                ):
                    True,
                "formal_runner_gate":
                    True,
                "formal_run_started":
                    True,
                "formal_run_complete":
                    True,
                "ready_for_formal_run":
                    True,
                "formal_run_count":
                    18,
                "completed_formal_runs":
                    18,
                (
                    "expected_seed_level_"
                    "summary_rows"
                ):
                    324,
                (
                    "actual_seed_level_"
                    "summary_rows"
                ):
                    324,
                (
                    "expected_three_seed_"
                    "aggregate_rows"
                ):
                    108,
                (
                    "actual_three_seed_"
                    "aggregate_rows"
                ):
                    108,
                "blockers":
                    [],
                "completion_status":
                    str(
                        EXP5
                        / (
                            "completion_"
                            "status.json"
                        )
                    ),
            }
        )

        readiness_path.write_text(
            json.dumps(
                readiness,
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        (
            EXP5
            / "THREE_BASELINE_EXPERIMENT5_SUCCESS"
        ).write_text(
            (
                "NEIMS_MASSFORMER_"
                "FRAGNNET_D3_"
                "EXPERIMENT5_SUCCESS\n"
            ),
            encoding="utf-8",
        )

        print(
            "THREE_BASELINE_EXPERIMENT5_COMPLETE"
        )


def scorer_self_test() -> None:
    rows = []

    for query_index in [
        1,
        2,
    ]:
        for candidate_rank in [
            1,
            2,
            3,
        ]:
            is_true = int(
                candidate_rank == 2
            )

            if query_index == 1:
                cbin_values = {
                    1: 0.9,
                    2: 0.9,
                    3: 0.1,
                }
            else:
                cbin_values = {
                    1: 0.2,
                    2: 0.8,
                    3: 0.1,
                }

            rows.append(
                {
                    "query_spec_id":
                        query_index,
                    "target_mol_id":
                        100,
                    "query_ace":
                        20.0
                        + query_index,
                    (
                        "candidate_"
                        "connectivity_key"
                    ):
                        (
                            f"KEY_"
                            f"{candidate_rank}"
                        ),
                    "candidate_rank":
                        candidate_rank,
                    "is_true_candidate":
                        is_true,
                    "candidate_formula":
                        (
                            "C6H6"
                            if candidate_rank
                            in {1, 2}
                            else "C7H8"
                        ),
                    "cbin":
                        cbin_values[
                            candidate_rank
                        ],
                    "cbin_sqrt":
                        cbin_values[
                            candidate_rank
                        ],
                    "jss":
                        cbin_values[
                            candidate_rank
                        ],
                }
            )

    scores = pd.DataFrame(
        rows
    )

    rankings, summary = (
        summarize_ranks(
            scores=scores,
            split="random",
            seed=42,
            fixed_targets={"100"},
            available_targets={"100"},
        )
    )

    if len(summary) != 18:
        raise RuntimeError(
            "Self-test summary row count "
            "is not 18"
        )

    check = rankings[
        (
            rankings[
                "cohort"
            ]
            == "fixed50"
        )
        & (
            rankings[
                "method"
            ]
            == "cbin"
        )
    ].sort_values(
        "query_spec_id"
    )

    ranks = (
        check[
            "retrieval_rank"
        ].astype(int).tolist()
    )

    if ranks != [2, 1]:
        raise RuntimeError(
            "Tie-break self-test failed: "
            f"{ranks}"
        )

    micro = summary[
        (
            summary[
                "cohort"
            ]
            == "fixed50"
        )
        & (
            summary[
                "method"
            ]
            == "cbin"
        )
        & (
            summary[
                "unit"
            ]
            == "spectrum_micro"
        )
    ].iloc[0]

    if abs(
        float(
            micro[
                "top1"
            ]
        )
        - 0.5
    ) > 1e-12:
        raise RuntimeError(
            "Micro summary self-test failed"
        )

    print(
        "SCORER_RANKER_SELF_TEST_PASS"
    )

    print(
        "SUMMARY_ROWS =",
        len(summary),
    )

    print(
        "TIE_BREAK_TRUE_RANKS =",
        ranks,
    )


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--self-test",
        action="store_true",
    )

    parser.add_argument(
        "--aggregate-only",
        action="store_true",
    )

    parser.add_argument(
        "--require-complete",
        action="store_true",
    )

    parser.add_argument(
        "--model",
    )

    parser.add_argument(
        "--split",
    )

    parser.add_argument(
        "--seed",
        type=int,
    )

    parser.add_argument(
        "--checkpoint",
    )

    parser.add_argument(
        "--config",
    )

    parser.add_argument(
        "--output-dir",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4096,
    )

    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
    )

    return parser


def main() -> None:
    args = (
        build_parser()
        .parse_args()
    )

    if args.self_test:
        scorer_self_test()
        return

    if args.aggregate_only:
        aggregate_results(
            require_complete=(
                args.require_complete
            )
        )
        return

    required = {
        "model":
            args.model,
        "split":
            args.split,
        "seed":
            args.seed,
        "checkpoint":
            args.checkpoint,
        "config":
            args.config,
        "output_dir":
            args.output_dir,
    }

    missing = [
        key
        for key, value
        in required.items()
        if value is None
    ]

    if missing:
        raise RuntimeError(
            "Missing run arguments: "
            f"{missing}"
        )

    run_one(args)


if __name__ == "__main__":
    try:
        main()

    except Exception:
        traceback.print_exc()
        sys.exit(1)
