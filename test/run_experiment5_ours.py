#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import gc
import gzip
import functools
import hashlib
import importlib.util
import json
import math
import os
import pickle
import random
import sys
import time
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runs/experiments/molecular_retrieval/pubchem_legacy_full"
POOL_DIR = BASE / "inference_ready_pools_20260723"
FROZEN_DIR = BASE / "frozen_manifest_20260723"
CANDIDATE_DIR = BASE / "candidate_d3_20260723"
PROC_DIR = CANDIDATE_DIR / "proc"
DAG_DIR = CANDIDATE_DIR / "frag/dags"
OUTPUT_ROOT = BASE / "ours_r184b_experiment5_20260724"
TEMPLATE_PATH = ROOT / "runs/_config/template.yml"
R170_PATH = ROOT / "train/_impl/refinement_steps/candidate_reranker.py"
R184_PATH = ROOT / "train/_impl/refinement_steps/spectrum_allocator.py"
BIN_RES = 0.01
MZ_MAX = 1500.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 5: frozen R160 -> R172D -> R184B PubChem retrieval."
    )
    parser.add_argument("--splits", nargs="+", default=["random", "scaffold"],
                        choices=["random", "scaffold"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--batch-size", type=int,
                        default=int(os.environ.get("EX5_BATCH_SIZE", "8")))
    parser.add_argument("--num-workers", type=int,
                        default=int(os.environ.get("EX5_NUM_WORKERS", "4")))
    parser.add_argument("--max-query-spectra", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def norm_id(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        try:
            return str(int(float(text)))
        except Exception:
            pass
    return text


def resolve_from_root(value: Any) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def config_path(split: str, seed: int) -> Path:
    family = (
        "molecule_disjoint_3seeds"
        if split == "random"
        else "scaffold_disjoint_3seeds"
    )
    return (
        ROOT / "runs/experiments" / family / f"seed_{seed}"
        / "v2c_ce_trajectory_ablation/control/config.yml"
    )


def seed_dir(split: str, seed: int) -> Path:
    family = (
        "molecule_disjoint_3seeds"
        if split == "random"
        else "scaffold_disjoint_3seeds"
    )
    return (
        ROOT / "runs/experiments" / family / f"seed_{seed}"
        / "v2e_full_063"
    )


def split_aliases(split: str) -> set[str]:
    if split == "random":
        return {"random", "random_test", "inchikey", "inchikey_test"}
    return {"scaffold", "scaffold_test"}


def filter_manifest_split(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    aliases = split_aliases(split)
    for column in ("split", "split_name", "split_type", "test_split"):
        if column in frame.columns:
            values = frame[column].astype(str).str.strip().str.lower()
            return frame[values.isin(aliases)].copy()
    for column in ("split_memberships", "membership", "memberships"):
        if column in frame.columns:
            token = "random_test" if split == "random" else "scaffold_test"
            return frame[
                frame[column].astype(str).str.lower().str.contains(token, na=False)
            ].copy()
    return frame.copy()


def target_set_from_manifest(path: Path, split: str) -> set[str]:
    if not path.is_file():
        return set()
    frame = filter_manifest_split(pd.read_csv(path), split)
    for column in ("target_mol_id", "mol_id"):
        if column in frame.columns:
            return {norm_id(value) for value in frame[column] if norm_id(value)}
    raise RuntimeError(f"target id column missing: {path}, columns={list(frame.columns)}")


def available_and_fixed_targets(split: str) -> tuple[set[str], set[str]]:
    available = target_set_from_manifest(
        FROZEN_DIR / "experiment5_available_pool_targets.csv", split
    )
    fixed = target_set_from_manifest(
        FROZEN_DIR / "experiment5_fixed50_targets.csv", split
    )

    summary_path = POOL_DIR / "experiment5_inference_ready_pool_summary.csv"
    if not available:
        summary = filter_manifest_split(pd.read_csv(summary_path), split)
        available = {norm_id(value) for value in summary["target_mol_id"]}

    if not fixed:
        summary = filter_manifest_split(pd.read_csv(summary_path), split)
        if "inference_ready_count" in summary.columns:
            summary = summary[
                pd.to_numeric(summary["inference_ready_count"], errors="coerce") == 50
            ]
        fixed = {norm_id(value) for value in summary["target_mol_id"]}

    return available, fixed


def choose_overlap_column(
    frame: pd.DataFrame,
    reference: set[str],
    label: str,
    excluded: set[str] | None = None,
) -> str:
    excluded = excluded or set()
    best_column = None
    best_overlap = -1
    for column in frame.columns:
        if column in excluded:
            continue
        values = {norm_id(v) for v in frame[column].dropna().head(20000)}
        overlap = len(values & reference)
        if overlap > best_overlap:
            best_overlap = overlap
            best_column = column
    if best_column is None or best_overlap <= 0:
        raise RuntimeError(
            f"cannot identify {label}; columns={list(frame.columns)}, "
            f"best={best_column}, overlap={best_overlap}"
        )
    print(f"[mapping] {label}: {best_column}, overlap={best_overlap}")
    return best_column


def dag_ids() -> set[str]:
    result = set()
    for path in DAG_DIR.iterdir():
        if path.is_file():
            result.add(path.name.split(".")[0])
    return result


def load_candidate_assets() -> dict[str, Any]:
    candidate_path = POOL_DIR / "experiment5_inference_ready_candidates.csv.gz"
    mapping_path = PROC_DIR / "candidate_structure_mapping.csv.gz"
    mol_path = PROC_DIR / "mol_df.pkl"
    spec_path = PROC_DIR / "spec_df.pkl"

    required = [
        candidate_path, mapping_path, mol_path, spec_path, DAG_DIR,
        TEMPLATE_PATH, R170_PATH, R184_PATH,
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    candidates = pd.read_csv(candidate_path)
    mapping = pd.read_csv(mapping_path)
    mol_df = pd.read_pickle(mol_path).copy()
    spec_df = pd.read_pickle(spec_path).copy()

    required_candidate_columns = {
        "target_mol_id",
        "candidate_connectivity_key",
        "is_true_candidate",
        "candidate_rank",
    }
    missing = required_candidate_columns - set(candidates.columns)
    if missing:
        raise RuntimeError(f"candidate columns missing: {sorted(missing)}")

    candidates["_target_norm"] = candidates["target_mol_id"].map(norm_id)
    candidates["_candidate_key_norm"] = (
        candidates["candidate_connectivity_key"].map(norm_id)
    )
    spec_df["_mol_norm"] = spec_df["mol_id"].map(norm_id)
    mol_df["_mol_norm"] = mol_df["mol_id"].map(norm_id)

    candidate_keys = set(candidates["_candidate_key_norm"])
    spec_mol_ids = set(spec_df["_mol_norm"])

    key_column = choose_overlap_column(
        mapping, candidate_keys, "candidate key"
    )
    id_column = choose_overlap_column(
        mapping, spec_mol_ids, "candidate internal mol_id",
        excluded={key_column},
    )

    mapping_small = mapping[[key_column, id_column]].copy()
    mapping_small["_candidate_key_norm"] = mapping_small[key_column].map(norm_id)
    mapping_small["_mol_norm"] = mapping_small[id_column].map(norm_id)
    mapping_small = mapping_small[
        ["_candidate_key_norm", "_mol_norm"]
    ].drop_duplicates("_candidate_key_norm")

    candidates = candidates.merge(
        mapping_small,
        on="_candidate_key_norm",
        how="left",
        validate="many_to_one",
    )
    if candidates["_mol_norm"].isna().any():
        bad = candidates[candidates["_mol_norm"].isna()][
            "candidate_connectivity_key"
        ].head(20).tolist()
        raise RuntimeError(f"candidate mapping missing: {bad}")

    available_dags = dag_ids()
    candidates["_dag_available"] = candidates["_mol_norm"].isin(available_dags)

    print(
        "[assets]",
        f"candidate_rows={len(candidates)}",
        f"candidate_structures={candidates['_mol_norm'].nunique()}",
        f"mol_rows={len(mol_df)}",
        f"spec_rows={len(spec_df)}",
        f"dags={len(available_dags)}",
    )

    return {
        "candidates": candidates,
        "mapping": mapping_small,
        "mol_df": mol_df,
        "spec_df": spec_df,
        "available_dags": available_dags,
    }


def query_spectra(
    split: str,
    config: dict[str, Any],
    target_ids: set[str],
    max_query_spectra: int,
) -> pd.DataFrame:
    original_spec_path = resolve_from_root(config["spec_fp"])
    split_dir = resolve_from_root(config["split_dp"])
    test_ids_path = split_dir / "test_ids.csv"

    if not original_spec_path.is_file():
        raise FileNotFoundError(original_spec_path)
    if not test_ids_path.is_file():
        raise FileNotFoundError(test_ids_path)

    spec = pd.read_pickle(original_spec_path).copy()
    test_ids = pd.read_csv(test_ids_path).copy()

    if "spec_id" not in test_ids.columns:
        raise RuntimeError(f"test_ids lacks spec_id: {test_ids_path}")

    test_spec_ids = {norm_id(v) for v in test_ids["spec_id"]}
    spec["_spec_norm"] = spec["spec_id"].map(norm_id)
    spec["_target_norm"] = spec["mol_id"].map(norm_id)

    query = spec[
        spec["_spec_norm"].isin(test_spec_ids)
        & spec["_target_norm"].isin(target_ids)
    ].copy()

    query["ace"] = pd.to_numeric(query["ace"], errors="coerce")
    query = query[query["ace"].notna()].copy()
    query = query.sort_values(["_target_norm", "spec_id"]).reset_index(drop=True)

    if max_query_spectra > 0:
        query = query.head(max_query_spectra).copy()

    query["_query_spec_norm"] = query["spec_id"].map(norm_id)

    print(
        f"[query] split={split}",
        f"spectra={len(query)}",
        f"molecules={query['_target_norm'].nunique()}",
        f"ace_mean={query['ace'].mean():.6f}",
    )
    return query


def build_expanded_candidate_input(
    query: pd.DataFrame,
    candidates: pd.DataFrame,
    candidate_spec: pd.DataFrame,
    candidate_mol: pd.DataFrame,
    target_ids: set[str],
    split: str,
    seed: int,
    audit_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, set[str]]:
    pool = candidates[candidates["_target_norm"].isin(target_ids)].copy()

    incomplete_targets = set(
        pool.loc[~pool["_dag_available"], "_target_norm"].unique()
    )
    complete_targets = set(target_ids) - incomplete_targets

    exclusion_rows = []
    for target in sorted(incomplete_targets):
        bad = pool[
            (pool["_target_norm"] == target)
            & (~pool["_dag_available"])
        ]
        for _, row in bad.iterrows():
            exclusion_rows.append({
                "split": split,
                "seed": seed,
                "target_mol_id": target,
                "candidate_connectivity_key": row["candidate_connectivity_key"],
                "candidate_internal_mol_id": row["_mol_norm"],
                "reason": "missing_D3_cache",
            })

    exclusions = pd.DataFrame(exclusion_rows)
    exclusions.to_csv(audit_dir / "excluded_incomplete_pools.csv", index=False)

    pool = pool[pool["_target_norm"].isin(complete_targets)].copy()
    query = query[query["_target_norm"].isin(complete_targets)].copy()

    if query.empty:
        raise RuntimeError(f"no query spectra remain for split={split}, seed={seed}")

    if pool.groupby("_target_norm")["is_true_candidate"].sum().ne(1).any():
        bad = (
            pool.groupby("_target_norm")["is_true_candidate"].sum()
            .loc[lambda x: x.ne(1)]
        )
        raise RuntimeError(f"true candidate count not one:\n{bad.head(20)}")

    query_small = query[
        ["_target_norm", "_query_spec_norm", "spec_id", "mol_id", "ace", "peaks"]
    ].copy()
    query_small = query_small.rename(columns={
        "spec_id": "query_spec_id",
        "mol_id": "query_target_mol_id",
        "ace": "query_ace",
        "peaks": "query_peaks",
    })

    metadata_columns = [
        "_target_norm",
        "_candidate_key_norm",
        "_mol_norm",
        "candidate_connectivity_key",
        "is_true_candidate",
        "candidate_rank",
    ]
    for optional in (
        "candidate_formula",
        "target_morgan_tanimoto",
        "candidate_count",
        "analysis_cohort",
        "candidate_origin",
    ):
        if optional in pool.columns:
            metadata_columns.append(optional)

    expanded = query_small.merge(
        pool[metadata_columns],
        on="_target_norm",
        how="inner",
        validate="many_to_many",
    )

    template = candidate_spec.drop_duplicates("_mol_norm").copy()
    template_columns = [column for column in template.columns if column != "_mol_norm"]
    template = template.rename(
        columns={column: f"tpl__{column}" for column in template_columns}
    )

    expanded = expanded.merge(
        template,
        on="_mol_norm",
        how="left",
        validate="many_to_one",
    )

    if expanded["tpl__mol_id"].isna().any():
        bad = expanded[expanded["tpl__mol_id"].isna()][
            ["candidate_connectivity_key", "_mol_norm"]
        ].head(20)
        raise RuntimeError(f"candidate spec template missing:\n{bad}")

    # Group repeated ACE evaluations of the same candidate together.
    # This makes the bounded per-worker DAG cache effective without
    # changing any query/candidate membership or ranking.
    expanded = expanded.sort_values(
        ["_mol_norm", "_query_spec_norm", "candidate_rank"]
    ).reset_index(drop=True)

    expanded["inference_spec_id"] = np.arange(
        1, len(expanded) + 1, dtype=np.int64
    )

    spec_input = pd.DataFrame()
    for column in template_columns:
        spec_input[column] = expanded[f"tpl__{column}"]

    spec_input["spec_id"] = expanded["inference_spec_id"].astype(np.int64)
    spec_input["group_id"] = expanded["inference_spec_id"].astype(np.int64)
    if "dset_spec_id" in spec_input.columns:
        spec_input["dset_spec_id"] = expanded["inference_spec_id"].astype(np.int64)
    if "dset" in spec_input.columns:
        spec_input["dset"] = "experiment5_pubchem"
    spec_input["ace"] = expanded["query_ace"].astype(float)
    spec_input["mol_id"] = expanded["tpl__mol_id"]

    used_mol_ids = set(spec_input["mol_id"].map(norm_id))
    mol_input = candidate_mol[
        candidate_mol["_mol_norm"].isin(used_mol_ids)
    ].drop(columns=["_mol_norm"], errors="ignore").copy()

    metadata = expanded[[
        "inference_spec_id",
        "_query_spec_norm",
        "query_spec_id",
        "query_target_mol_id",
        "_target_norm",
        "query_ace",
        "query_peaks",
        "candidate_connectivity_key",
        "_candidate_key_norm",
        "_mol_norm",
        "is_true_candidate",
        "candidate_rank",
    ] + [
        column for column in (
            "candidate_formula",
            "target_morgan_tanimoto",
            "candidate_count",
            "analysis_cohort",
            "candidate_origin",
        )
        if column in expanded.columns
    ]].copy()

    print(
        "[expanded]",
        f"rows={len(spec_input)}",
        f"query_spectra={metadata['_query_spec_norm'].nunique()}",
        f"targets={metadata['_target_norm'].nunique()}",
        f"candidate_molecules={len(used_mol_ids)}",
        f"excluded_incomplete_targets={len(incomplete_targets)}",
    )

    return spec_input, mol_input, metadata, incomplete_targets


def aggregate_sparse_bins(
    mzs: np.ndarray,
    intensities: np.ndarray,
    bin_res: float = BIN_RES,
) -> tuple[np.ndarray, np.ndarray]:
    mzs = np.asarray(mzs, dtype=np.float64).reshape(-1)
    intensities = np.asarray(intensities, dtype=np.float64).reshape(-1)
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
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

    bins = np.rint(mzs / float(bin_res)).astype(np.int64)
    order = np.argsort(bins, kind="mergesort")
    bins = bins[order]
    intensities = intensities[order]
    unique_bins, starts = np.unique(bins, return_index=True)
    values = np.add.reduceat(intensities, starts)
    return unique_bins, values


def peak_list_to_arrays(peaks: Any) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(peaks, np.ndarray):
        peaks = peaks.tolist()
    if not isinstance(peaks, (list, tuple)):
        return np.empty(0), np.empty(0)
    mzs = []
    ints = []
    for peak in peaks:
        if isinstance(peak, (list, tuple, np.ndarray)) and len(peak) >= 2:
            try:
                mzs.append(float(peak[0]))
                ints.append(float(peak[1]))
            except Exception:
                continue
    return np.asarray(mzs), np.asarray(ints)


def sparse_cosine(
    pred_bins: np.ndarray,
    pred_values: np.ndarray,
    query_bins: np.ndarray,
    query_values: np.ndarray,
    sqrt_transform: bool = False,
) -> float:
    if sqrt_transform:
        pred_values = np.sqrt(np.clip(pred_values, 0.0, None))
        query_values = np.sqrt(np.clip(query_values, 0.0, None))
    pred_norm = float(np.linalg.norm(pred_values))
    query_norm = float(np.linalg.norm(query_values))
    if pred_norm <= 0.0 or query_norm <= 0.0:
        return 0.0

    i = j = 0
    dot = 0.0
    while i < len(pred_bins) and j < len(query_bins):
        if pred_bins[i] == query_bins[j]:
            dot += float(pred_values[i] * query_values[j])
            i += 1
            j += 1
        elif pred_bins[i] < query_bins[j]:
            i += 1
        else:
            j += 1
    return float(dot / (pred_norm * query_norm))


def sparse_jss(
    pred_bins: np.ndarray,
    pred_values: np.ndarray,
    query_bins: np.ndarray,
    query_values: np.ndarray,
) -> float:
    pred_sum = float(pred_values.sum())
    query_sum = float(query_values.sum())
    if pred_sum <= 0.0 or query_sum <= 0.0:
        return 0.0

    p = pred_values / pred_sum
    q = query_values / query_sum
    i = j = 0
    jsd = 0.0
    while i < len(pred_bins) or j < len(query_bins):
        if j >= len(query_bins) or (
            i < len(pred_bins) and pred_bins[i] < query_bins[j]
        ):
            pv, qv = float(p[i]), 0.0
            i += 1
        elif i >= len(pred_bins) or query_bins[j] < pred_bins[i]:
            pv, qv = 0.0, float(q[j])
            j += 1
        else:
            pv, qv = float(p[i]), float(q[j])
            i += 1
            j += 1

        mean = 0.5 * (pv + qv)
        if pv > 0.0:
            jsd += 0.5 * pv * math.log(pv / mean)
        if qv > 0.0:
            jsd += 0.5 * qv * math.log(qv / mean)

    return float(max(0.0, min(1.0, 1.0 - jsd / math.log(2.0))))


def group_log_softmax(
    scores: torch.Tensor,
    batch_indices: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    output = torch.empty_like(scores)
    for index in range(int(batch_size)):
        mask = batch_indices == index
        if mask.any():
            output[mask] = torch.log_softmax(scores[mask], dim=0)
    return output


def predict_lgbm(
    regressor: Any,
    features: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    score_clip: float,
) -> torch.Tensor:
    array = features.detach().cpu().numpy().astype(np.float32)
    names = getattr(regressor, "feature_names_in_", None)
    if names is None:
        names = getattr(regressor, "feature_name_", None)

    model_input: Any = array
    if names is not None:
        names = [str(name) for name in list(names)]
        if len(names) == array.shape[1]:
            model_input = pd.DataFrame(array, columns=names)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = regressor.predict(model_input).astype(np.float32)

    scores = np.clip(scores, -float(score_clip), float(score_clip))
    return torch.from_numpy(scores).to(device=device, dtype=dtype)


def initialize_model(
    split: str,
    seed: int,
    base_config: dict[str, Any],
    r170: Any,
    r184: Any,
    device: torch.device,
) -> tuple[Any, Any, Any, Any, SimpleNamespace, list[Any], dict[str, str]]:
    from ms2spectra.training import FragGNNPL

    run_dir = seed_dir(split, seed)
    backbone_path = run_dir / "08_R160/r160_best_state.pt"
    reranker_path = run_dir / "09_R172D/r170_regressor.pkl"
    allocator_path = run_dir / "11_R184B/r184_allocator_best.pt"

    for path in (backbone_path, reranker_path, allocator_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    with reranker_path.open("rb") as handle:
        reranker_package = pickle.load(handle)
    regressor = reranker_package["model"]

    allocator_package = torch.load(
        allocator_path, map_location="cpu", weights_only=False
    )
    saved_arguments = dict(allocator_package["args"])
    allocator_arguments = SimpleNamespace(**saved_arguments)
    extra_schema = allocator_package.get(
        "extra_schema", reranker_package.get("extra_schema", [])
    )

    backbone = FragGNNPL(**base_config)
    state = r170.load_state_dict_any(backbone_path)
    missing, unexpected = backbone.load_state_dict(state, strict=False)
    print(
        "[model]",
        f"split={split}",
        f"seed={seed}",
        f"backbone_missing={len(missing)}",
        f"backbone_unexpected={len(unexpected)}",
    )
    backbone = backbone.to(device).eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)

    allocator = r184.ResidualAllocator(
        input_dim=int(allocator_package["input_dim"]),
        hidden=int(saved_arguments["hidden"]),
        layers=int(saved_arguments["layers"]),
        dropout=float(saved_arguments["dropout"]),
        score_clip=float(saved_arguments["score_clip"]),
    ).to(device)
    allocator.load_state_dict(allocator_package["model"])
    allocator.eval()
    for parameter in allocator.parameters():
        parameter.requires_grad_(False)

    hashes = {
        "r160": sha256(backbone_path),
        "r172d": sha256(reranker_path),
        "r184b": sha256(allocator_path),
    }
    return (
        backbone, allocator, regressor, allocator_package,
        allocator_arguments, extra_schema, hashes,
    )


def make_dataset_loader(
    base_config: dict[str, Any],
    spec_input: pd.DataFrame,
    mol_input: pd.DataFrame,
    batch_size: int,
    num_workers: int,
):
    from ms2spectra.workflow import init_dataset, init_dataloader

    config = copy.deepcopy(base_config)
    config["spec_fp"] = spec_input
    config["mol_fp"] = mol_input
    config["frag_dp"] = str(DAG_DIR)
    config["eval_batch_size"] = int(batch_size)
    config["num_workers"] = int(num_workers)
    config["share_memory"] = False
    config["dynamic_batch_sampler"] = False
    config["group_sampler"] = False
    config["simple_group_sampler"] = False
    config["frag_params"]["preload"] = False
    config["frag_params"]["preprocess"] = False
    config["mol_params"]["preprocess"] = True
    config["spec_params"]["preprocess"] = True
    config["pin_memory"] = bool(torch.cuda.is_available())

    dataset = init_dataset(config, splits=("predict_only",))[0]

    # Candidate structures are reused across the multiple ACE spectra of
    # their target molecule. Cache only a small number of raw DAG objects
    # per worker so repeated rows do not repeatedly decompress the same file.
    dataset._load_frag_entry = functools.lru_cache(maxsize=64)(
        dataset._load_frag_entry
    )

    loader = init_dataloader(dataset, config)
    return dataset, loader


def _count_gzip_csv_rows(path: Path) -> int:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return sum(1 for _ in reader)


def _scan_score_parts(parts_dir: Path) -> tuple[list[Path], int]:
    """Return ordered valid score parts and cumulative completed rows.

    Supports:
    1. legacy batch-based names:
       part_b00000001_b00000100_r000000001600.csv.gz
    2. new row-based names:
       part_r000000001601_r000000004800.csv.gz
    """
    import re

    legacy_pattern = re.compile(
        r"^part_b(\d+)_b(\d+)_r(\d+)\.csv\.gz$"
    )
    row_pattern = re.compile(
        r"^part_r(\d+)_r(\d+)\.csv\.gz$"
    )

    legacy_parts = []
    row_parts = []

    for part_path in parts_dir.glob("*.csv.gz"):
        legacy_match = legacy_pattern.match(part_path.name)
        if legacy_match is not None:
            legacy_parts.append((
                int(legacy_match.group(2)),
                int(legacy_match.group(3)),
                part_path,
            ))
            continue

        row_match = row_pattern.match(part_path.name)
        if row_match is not None:
            row_parts.append((
                int(row_match.group(1)),
                int(row_match.group(2)),
                part_path,
            ))

    legacy_parts.sort(key=lambda item: (item[0], item[1]))
    row_parts.sort(key=lambda item: (item[0], item[1]))

    ordered_paths: list[Path] = []
    completed_rows = 0

    for _, cumulative_rows, part_path in legacy_parts:
        actual_rows = _count_gzip_csv_rows(part_path)

        if completed_rows + actual_rows != cumulative_rows:
            raise RuntimeError(
                "legacy checkpoint cumulative row mismatch: "
                f"path={part_path}, "
                f"previous={completed_rows}, "
                f"part_rows={actual_rows}, "
                f"filename_total={cumulative_rows}"
            )

        ordered_paths.append(part_path)
        completed_rows = cumulative_rows

    for start_row, end_row, part_path in row_parts:
        expected_start = completed_rows + 1

        if start_row != expected_start:
            raise RuntimeError(
                "row checkpoint is not contiguous: "
                f"expected_start={expected_start}, "
                f"found_start={start_row}, "
                f"path={part_path}"
            )

        actual_rows = _count_gzip_csv_rows(part_path)
        expected_rows = end_row - start_row + 1

        if actual_rows != expected_rows:
            raise RuntimeError(
                "row checkpoint row count mismatch: "
                f"path={part_path}, "
                f"actual={actual_rows}, "
                f"expected={expected_rows}"
            )

        ordered_paths.append(part_path)
        completed_rows = end_row

    return ordered_paths, completed_rows


def run_predictions(
    split: str,
    seed: int,
    backbone: Any,
    allocator: Any,
    regressor: Any,
    allocator_arguments: SimpleNamespace,
    extra_schema: list[Any],
    metadata: pd.DataFrame,
    loader: Any,
    r170: Any,
    r184: Any,
    device: torch.device,
    score_path: Path,
    resume_rows: int,
) -> None:
    import os

    metadata_by_id = metadata.set_index(
        "inference_spec_id",
        drop=False,
    )

    query_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for query_id, rows in metadata.groupby(
        "_query_spec_norm",
        sort=False,
    ):
        mzs, intensities = peak_list_to_arrays(
            rows.iloc[0]["query_peaks"]
        )
        query_cache[str(query_id)] = aggregate_sparse_bins(
            mzs,
            intensities,
        )

    fieldnames = [
        "split",
        "seed",
        "query_spec_id",
        "target_mol_id",
        "query_ace",
        "candidate_connectivity_key",
        "candidate_internal_mol_id",
        "candidate_rank",
        "is_true_candidate",
        "candidate_formula",
        "target_morgan_tanimoto",
        "candidate_count",
        "analysis_cohort",
        "cbin",
        "cbin_sqrt",
        "jss",
    ]

    score_path.parent.mkdir(parents=True, exist_ok=True)

    parts_dir = score_path.parent / "candidate_score_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    progress_path = (
        score_path.parent
        / "candidate_scores.progress.json"
    )

    part_paths, discovered_rows = _scan_score_parts(parts_dir)

    if int(discovered_rows) != int(resume_rows):
        raise RuntimeError(
            "resume row mismatch: "
            f"parts={discovered_rows}, requested={resume_rows}"
        )

    processed = int(resume_rows)
    session_processed = 0
    session_start = time.time()

    buffer_rows: list[dict[str, Any]] = []
    checkpoint_batches = 100

    def atomic_json_write(
        destination: Path,
        payload: dict[str, Any],
    ) -> None:
        temporary = destination.with_name(
            destination.name + ".tmp"
        )
        temporary.write_text(
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)

    def write_checkpoint(
        session_batch_index: int,
    ) -> None:
        nonlocal buffer_rows
        nonlocal part_paths

        if not buffer_rows:
            return

        start_row = processed - len(buffer_rows) + 1
        end_row = processed

        final_part = parts_dir / (
            f"part_r{start_row:012d}"
            f"_r{end_row:012d}.csv.gz"
        )
        temporary_part = final_part.with_name(
            final_part.name + ".tmp"
        )

        with gzip.open(
            temporary_part,
            "wt",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
            )
            writer.writeheader()
            writer.writerows(buffer_rows)

        os.replace(temporary_part, final_part)
        part_paths.append(final_part)

        atomic_json_write(
            progress_path,
            {
                "status": "running",
                "checkpoint_version": 2,
                "checkpoint_unit": "rows",
                "split": split,
                "seed": int(seed),
                "batch_size": int(
                    getattr(loader, "batch_size", 0) or 0
                ),
                "completed_rows": int(processed),
                "metadata_rows": int(len(metadata)),
                "remaining_rows": int(
                    len(metadata) - processed
                ),
                "session_batches_completed": int(
                    session_batch_index
                ),
                "last_part": final_part.name,
                "updated_at_epoch": time.time(),
            },
        )

        print(
            "[checkpoint]",
            f"split={split}",
            f"seed={seed}",
            f"session_batches={session_batch_index}/{len(loader)}",
            f"candidate_predictions={processed}/{len(metadata)}",
            f"rows_saved={len(buffer_rows)}",
            f"path={final_part}",
            flush=True,
        )

        buffer_rows = []

    print(
        "[row-resume]",
        f"split={split}",
        f"seed={seed}",
        f"completed_rows={resume_rows}/{len(metadata)}",
        f"remaining_rows={len(metadata) - resume_rows}",
        f"current_batch_size="
        f"{getattr(loader, 'batch_size', 0) or 0}",
        flush=True,
    )

    with torch.inference_mode():
        for session_batch_index, batch in enumerate(
            loader,
            start=1,
        ):
            batch = r170.move_to_device(batch, device)

            result = backbone._common_step(
                batch,
                split="test",
                log=False,
            )

            result = r170.attach_raw_rich_features(
                backbone,
                batch,
                result,
                max_extra_dims=int(
                    allocator_arguments.max_extra_dims
                ),
                bin_res=float(
                    allocator_arguments.target_bin_res
                ),
            )

            result = r184.alias_rich_feature_keys(
                result,
                extra_schema,
            )

            features = r170.candidate_features(
                result,
                batch,
                mz_max=float(backbone.hparams.mz_max),
                local_bin_res=float(
                    allocator_arguments.local_bin_res
                ),
                extra_schema=extra_schema,
            ).float()

            lgbm_score = predict_lgbm(
                regressor,
                features,
                device=features.device,
                dtype=result["pred_logprobs"].dtype,
                score_clip=float(
                    allocator_arguments.lgbm_score_clip
                ),
            )

            batch_indices = result[
                "pred_batch_idxs"
            ].long()

            batch_size_actual = int(
                result["unique_id"].numel()
            )

            base_logits = (
                result["pred_logprobs"]
                .float()
                .clamp(-30.0, 5.0)
                + float(allocator_arguments.alpha)
                * lgbm_score
            )

            residual = allocator(features)

            logits = (
                base_logits
                + float(
                    allocator_arguments.residual_scale
                )
                * residual
            )

            logits = logits / max(
                float(allocator_arguments.temperature),
                1.0e-6,
            )

            new_logp = group_log_softmax(
                logits,
                batch_indices,
                batch_size_actual,
            )

            spec_ids = (
                result["unique_id"]
                .detach()
                .cpu()
                .reshape(-1)
                .numpy()
                .astype(int)
            )

            mzs_cpu = (
                result["pred_mzs"]
                .detach()
                .cpu()
                .numpy()
            )

            probs_cpu = (
                new_logp
                .exp()
                .detach()
                .cpu()
                .numpy()
            )

            batch_indices_cpu = (
                batch_indices
                .detach()
                .cpu()
                .numpy()
            )

            for local_index, inference_id in enumerate(
                spec_ids
            ):
                meta = metadata_by_id.loc[
                    int(inference_id)
                ]

                mask = (
                    batch_indices_cpu == local_index
                )

                pred_bins, pred_values = (
                    aggregate_sparse_bins(
                        mzs_cpu[mask],
                        probs_cpu[mask],
                    )
                )

                query_bins, query_values = query_cache[
                    str(meta["_query_spec_norm"])
                ]

                row = {
                    "split": split,
                    "seed": seed,
                    "query_spec_id":
                        meta["query_spec_id"],
                    "target_mol_id":
                        meta["query_target_mol_id"],
                    "query_ace":
                        float(meta["query_ace"]),
                    "candidate_connectivity_key":
                        meta[
                            "candidate_connectivity_key"
                        ],
                    "candidate_internal_mol_id":
                        meta["_mol_norm"],
                    "candidate_rank":
                        int(meta["candidate_rank"]),
                    "is_true_candidate":
                        int(meta["is_true_candidate"]),
                    "candidate_formula":
                        meta.get(
                            "candidate_formula",
                            "",
                        ),
                    "target_morgan_tanimoto":
                        meta.get(
                            "target_morgan_tanimoto",
                            np.nan,
                        ),
                    "candidate_count":
                        meta.get(
                            "candidate_count",
                            np.nan,
                        ),
                    "analysis_cohort":
                        meta.get(
                            "analysis_cohort",
                            "",
                        ),
                    "cbin": sparse_cosine(
                        pred_bins,
                        pred_values,
                        query_bins,
                        query_values,
                        False,
                    ),
                    "cbin_sqrt": sparse_cosine(
                        pred_bins,
                        pred_values,
                        query_bins,
                        query_values,
                        True,
                    ),
                    "jss": sparse_jss(
                        pred_bins,
                        pred_values,
                        query_bins,
                        query_values,
                    ),
                }

                buffer_rows.append(row)
                processed += 1
                session_processed += 1

            if (
                session_batch_index
                % checkpoint_batches
                == 0
                or session_batch_index == len(loader)
            ):
                write_checkpoint(session_batch_index)

            if session_batch_index % 100 == 0:
                elapsed = max(
                    time.time() - session_start,
                    1.0e-9,
                )

                print(
                    "[progress]",
                    f"split={split}",
                    f"seed={seed}",
                    f"session_batches="
                    f"{session_batch_index}/{len(loader)}",
                    f"candidate_predictions="
                    f"{processed}/{len(metadata)}",
                    f"session_rows_per_second="
                    f"{session_processed / elapsed:.2f}",
                    flush=True,
                )

    part_paths, final_rows = _scan_score_parts(
        parts_dir
    )

    if int(final_rows) != int(len(metadata)):
        raise RuntimeError(
            "prediction rows incomplete after loader: "
            f"{final_rows} != {len(metadata)}"
        )

    temporary_score = score_path.with_name(
        score_path.name + ".tmp"
    )

    combined_rows = 0

    with gzip.open(
        temporary_score,
        "wt",
        encoding="utf-8",
        newline="",
    ) as output_handle:
        output_writer = csv.DictWriter(
            output_handle,
            fieldnames=fieldnames,
        )
        output_writer.writeheader()

        for part_path in part_paths:
            with gzip.open(
                part_path,
                "rt",
                encoding="utf-8",
                newline="",
            ) as input_handle:
                reader = csv.DictReader(input_handle)

                if list(reader.fieldnames or []) != fieldnames:
                    raise RuntimeError(
                        "checkpoint schema mismatch: "
                        f"{part_path}"
                    )

                for row in reader:
                    output_writer.writerow(row)
                    combined_rows += 1

    if int(combined_rows) != int(len(metadata)):
        temporary_score.unlink(missing_ok=True)
        raise RuntimeError(
            "combined score rows mismatch: "
            f"{combined_rows} != {len(metadata)}"
        )

    os.replace(temporary_score, score_path)

    atomic_json_write(
        progress_path,
        {
            "status": "complete",
            "checkpoint_version": 2,
            "checkpoint_unit": "rows",
            "split": split,
            "seed": int(seed),
            "completed_rows": int(final_rows),
            "metadata_rows": int(len(metadata)),
            "remaining_rows": 0,
            "score_path": str(score_path),
            "completed_at_epoch": time.time(),
        },
    )

    print(
        "[predictions complete]",
        f"split={split}",
        f"seed={seed}",
        f"rows={final_rows}",
        f"path={score_path}",
        flush=True,
    )


def summarize_ranks(
    scores: pd.DataFrame,
    split: str,
    seed: int,
    fixed_targets: set[str],
    available_targets: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = scores.copy()
    scores["_target_norm"] = scores["target_mol_id"].map(norm_id)
    scores["_query_norm"] = scores["query_spec_id"].map(norm_id)

    true_formula_by_query = (
        scores[scores["is_true_candidate"] == 1]
        .drop_duplicates("_query_norm")
        .set_index("_query_norm")["candidate_formula"]
        .astype(str)
        .to_dict()
    )
    scores["_true_formula"] = scores["_query_norm"].map(true_formula_by_query)

    cohort_frames: list[tuple[str, pd.DataFrame]] = [
        (
            "fixed50",
            scores[scores["_target_norm"].isin(fixed_targets)].copy(),
        ),
        (
            "available_pool",
            scores[scores["_target_norm"].isin(available_targets)].copy(),
        ),
    ]

    exact = scores[
        scores["_target_norm"].isin(fixed_targets)
        & (scores["candidate_formula"].astype(str) == scores["_true_formula"])
    ].copy()
    cohort_frames.append(("exact_formula", exact))

    rank_frames = []
    metric_columns = ["cbin", "cbin_sqrt", "jss"]

    for cohort, frame in cohort_frames:
        if frame.empty:
            continue
        true_counts = frame.groupby("_query_norm")["is_true_candidate"].sum()
        valid_queries = set(true_counts[true_counts == 1].index)
        frame = frame[frame["_query_norm"].isin(valid_queries)].copy()

        for metric in metric_columns:
            ordered = frame.sort_values(
                ["_query_norm", metric, "candidate_rank",
                 "candidate_connectivity_key"],
                ascending=[True, False, True, True],
                kind="mergesort",
            ).copy()
            ordered["retrieval_rank"] = (
                ordered.groupby("_query_norm").cumcount() + 1
            )
            true_rows = ordered[ordered["is_true_candidate"] == 1].copy()
            true_rows["split"] = split
            true_rows["seed"] = seed
            true_rows["cohort"] = cohort
            true_rows["method"] = metric
            true_rows["pool_size_scored"] = (
                ordered.groupby("_query_norm").size()
                .reindex(true_rows["_query_norm"]).to_numpy()
            )
            rank_frames.append(true_rows[[
                "split", "seed", "cohort", "method", "query_spec_id",
                "target_mol_id", "query_ace", "retrieval_rank",
                "pool_size_scored", metric,
            ]].rename(columns={metric: "true_candidate_similarity"}))

    rankings = pd.concat(rank_frames, ignore_index=True)

    summary_rows = []
    for (cohort, method), frame in rankings.groupby(["cohort", "method"]):
        frame = frame.copy()
        frame["top1"] = (frame["retrieval_rank"] <= 1).astype(float)
        frame["top5"] = (frame["retrieval_rank"] <= 5).astype(float)
        frame["top10"] = (frame["retrieval_rank"] <= 10).astype(float)
        frame["rr"] = 1.0 / frame["retrieval_rank"].astype(float)

        summary_rows.append({
            "split": split,
            "seed": seed,
            "cohort": cohort,
            "method": method,
            "unit": "spectrum_micro",
            "n_spectra": int(len(frame)),
            "n_molecules": int(frame["target_mol_id"].map(norm_id).nunique()),
            "top1": float(frame["top1"].mean()),
            "top5": float(frame["top5"].mean()),
            "top10": float(frame["top10"].mean()),
            "mrr": float(frame["rr"].mean()),
            "median_rank": float(frame["retrieval_rank"].median()),
            "mean_rank": float(frame["retrieval_rank"].mean()),
        })

        molecule = (
            frame.assign(_target=frame["target_mol_id"].map(norm_id))
            .groupby("_target", as_index=False)
            .agg(
                top1=("top1", "mean"),
                top5=("top5", "mean"),
                top10=("top10", "mean"),
                rr=("rr", "mean"),
                retrieval_rank=("retrieval_rank", "mean"),
            )
        )
        summary_rows.append({
            "split": split,
            "seed": seed,
            "cohort": cohort,
            "method": method,
            "unit": "molecule_macro",
            "n_spectra": int(len(frame)),
            "n_molecules": int(len(molecule)),
            "top1": float(molecule["top1"].mean()),
            "top5": float(molecule["top5"].mean()),
            "top10": float(molecule["top10"].mean()),
            "mrr": float(molecule["rr"].mean()),
            "median_rank": float(molecule["retrieval_rank"].median()),
            "mean_rank": float(molecule["retrieval_rank"].mean()),
        })

    return rankings, pd.DataFrame(summary_rows)


def run_combination(
    split: str,
    seed: int,
    args: argparse.Namespace,
    assets: dict[str, Any],
    r170: Any,
    r184: Any,
    device: torch.device,
) -> None:
    import os
    import shutil

    combo_dir = OUTPUT_ROOT / split / f"seed_{seed}"
    success_path = combo_dir / "_SUCCESS.json"

    if success_path.is_file() and not args.force:
        print(
            f"[skip complete] {split} seed={seed}: "
            f"{success_path}",
            flush=True,
        )
        return

    combo_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    seed_everything(seed)

    from ms2spectra.workflow import load_config

    cfg_path = config_path(split, seed)

    if not cfg_path.is_file():
        raise FileNotFoundError(cfg_path)

    base_config = load_config(
        TEMPLATE_PATH,
        cfg_path,
    )
    base_config = r170.force_r160_arch(
        base_config
    )

    available_targets, fixed_targets = (
        available_and_fixed_targets(split)
    )

    query = query_spectra(
        split,
        base_config,
        available_targets,
        args.max_query_spectra,
    )

    (
        spec_input,
        mol_input,
        metadata,
        incomplete_targets,
    ) = build_expanded_candidate_input(
        query=query,
        candidates=assets["candidates"],
        candidate_spec=assets["spec_df"],
        candidate_mol=assets["mol_df"],
        target_ids=available_targets,
        split=split,
        seed=seed,
        audit_dir=combo_dir,
    )

    effective_available = (
        available_targets - incomplete_targets
    )
    effective_fixed = (
        fixed_targets - incomplete_targets
    )

    score_path = (
        combo_dir / "candidate_scores.csv.gz"
    )
    parts_dir = (
        combo_dir / "candidate_score_parts"
    )
    parts_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    _, resume_rows = _scan_score_parts(
        parts_dir
    )

    total_rows = int(len(spec_input))

    if resume_rows < 0 or resume_rows > total_rows:
        raise RuntimeError(
            "invalid row checkpoint: "
            f"{resume_rows}/{total_rows}"
        )

    remaining_spec_input = (
        spec_input
        .iloc[resume_rows:]
        .copy()
        .reset_index(drop=True)
    )

    print(
        "[row-resume-setup]",
        f"split={split}",
        f"seed={seed}",
        f"completed_rows={resume_rows}/{total_rows}",
        f"remaining_rows={len(remaining_spec_input)}",
        f"batch_size={int(args.batch_size)}",
        flush=True,
    )

    (
        backbone,
        allocator,
        regressor,
        allocator_package,
        allocator_arguments,
        extra_schema,
        hashes,
    ) = initialize_model(
        split,
        seed,
        base_config,
        r170,
        r184,
        device,
    )

    if remaining_spec_input.empty:
        dataset = None
        loader = ()
    else:
        dataset, loader = make_dataset_loader(
            base_config=base_config,
            spec_input=remaining_spec_input,
            mol_input=mol_input,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
        )

    run_predictions(
        split=split,
        seed=seed,
        backbone=backbone,
        allocator=allocator,
        regressor=regressor,
        allocator_arguments=allocator_arguments,
        extra_schema=extra_schema,
        metadata=metadata,
        loader=loader,
        r170=r170,
        r184=r184,
        device=device,
        score_path=score_path,
        resume_rows=resume_rows,
    )

    scores = pd.read_csv(score_path)

    rankings, summary = summarize_ranks(
        scores=scores,
        split=split,
        seed=seed,
        fixed_targets=effective_fixed,
        available_targets=effective_available,
    )

    rankings_path = (
        combo_dir
        / "true_candidate_ranks.csv.gz"
    )
    summary_path = (
        combo_dir
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

    audit = {
        "status": "complete",
        "split": split,
        "seed": int(seed),
        "device": str(device),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "checkpoint_version": 2,
        "checkpoint_unit": "rows",
        "query_spectra": int(
            metadata["_query_spec_norm"].nunique()
        ),
        "query_molecules": int(
            metadata["_target_norm"].nunique()
        ),
        "candidate_prediction_rows": int(
            len(metadata)
        ),
        "fixed50_target_count_original": int(
            len(fixed_targets)
        ),
        "available_target_count_original": int(
            len(available_targets)
        ),
        "incomplete_pool_target_count": int(
            len(incomplete_targets)
        ),
        "fixed50_target_count_scored": int(
            len(effective_fixed)
        ),
        "available_target_count_scored": int(
            len(effective_available)
        ),
        "d3_cache_count": int(
            len(assets["available_dags"])
        ),
        "weights_sha256": hashes,
        "allocator_best_val_cos": float(
            allocator_package.get(
                "best_val_cos",
                float("nan"),
            )
        ),
        "output_files": {
            "candidate_scores": str(score_path),
            "true_candidate_ranks": str(
                rankings_path
            ),
            "retrieval_summary": str(
                summary_path
            ),
        },
    }

    temporary_success = success_path.with_name(
        success_path.name + ".tmp"
    )
    temporary_success.write_text(
        json.dumps(
            audit,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(
        temporary_success,
        success_path,
    )

    # 完整组已合并并写出指标后，删除增量分片，
    # 避免同时保存一份分片和一份最终总文件。
    shutil.rmtree(
        parts_dir,
        ignore_errors=True,
    )

    print("\n" + "=" * 110)
    print(
        f"EXPERIMENT 5 COMPLETE: "
        f"split={split}, seed={seed}"
    )
    print("=" * 110)
    print(summary.to_string(index=False))
    print("=" * 110)

    del backbone
    del allocator
    del regressor
    del spec_input
    del mol_input
    del metadata
    del scores
    del rankings
    del summary

    if dataset is not None:
        del dataset

    del loader

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()



def aggregate_completed() -> None:
    frames = []
    for path in sorted(OUTPUT_ROOT.glob("*/seed_*/retrieval_summary.csv")):
        frames.append(pd.read_csv(path))
    if not frames:
        return

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(OUTPUT_ROOT / "all_seed_summaries.csv", index=False)

    metrics = ["top1", "top5", "top10", "mrr", "median_rank", "mean_rank"]
    grouped = combined.groupby(
        ["split", "cohort", "method", "unit"], as_index=False
    )

    rows = []
    for keys, frame in grouped:
        row = dict(zip(["split", "cohort", "method", "unit"], keys))
        row["seed_count"] = int(frame["seed"].nunique())
        row["seeds"] = "|".join(map(str, sorted(frame["seed"].unique())))
        row["n_spectra"] = int(round(frame["n_spectra"].mean()))
        row["n_molecules"] = int(round(frame["n_molecules"].mean()))
        for metric in metrics:
            row[f"{metric}_mean"] = float(frame[metric].mean())
            row[f"{metric}_std"] = float(frame[metric].std(ddof=1))
        rows.append(row)

    aggregate = pd.DataFrame(rows)
    aggregate.to_csv(OUTPUT_ROOT / "experiment5_ours_aggregate.csv", index=False)

    print("\n" + "=" * 110)
    print("EXPERIMENT 5 AGGREGATE")
    print("=" * 110)
    print(aggregate.to_string(index=False))
    print("=" * 110)


def main() -> None:
    args = parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(ROOT / "code/src"))
    r170 = load_module(R170_PATH, "experiment5_r170")
    r184 = load_module(R184_PATH, "experiment5_r184")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        "[start]",
        f"root={ROOT}",
        f"device={device}",
        f"splits={args.splits}",
        f"seeds={args.seeds}",
        f"batch_size={args.batch_size}",
        f"num_workers={args.num_workers}",
    )
    if device.type == "cuda":
        print("[gpu]", torch.cuda.get_device_name(0))

    assets = load_candidate_assets()

    run_manifest = {
        "status": "running",
        "started_at_epoch": time.time(),
        "splits": args.splits,
        "seeds": args.seeds,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "max_query_spectra": args.max_query_spectra,
        "scope": "ours_R160_R172D_R184B_only",
        "ranking_methods": ["cbin", "cbin_sqrt", "jss"],
        "cohorts": ["fixed50", "available_pool", "exact_formula"],
        "missing_d3_policy": "exclude_entire_affected_target_pool",
    }
    (OUTPUT_ROOT / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    try:
        for split in args.splits:
            for seed in args.seeds:
                run_combination(
                    split, seed, args, assets, r170, r184, device
                )
                # Refresh partial cross-seed tables after every completed group.
                aggregate_completed()
        aggregate_completed()
    except Exception:
        run_manifest["status"] = "failed"
        run_manifest["failed_at_epoch"] = time.time()
        (OUTPUT_ROOT / "run_manifest.json").write_text(
            json.dumps(run_manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        raise

    run_manifest["status"] = "complete"
    run_manifest["completed_at_epoch"] = time.time()
    (OUTPUT_ROOT / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("[all complete]", OUTPUT_ROOT)


if __name__ == "__main__":
    main()
