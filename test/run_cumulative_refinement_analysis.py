from __future__ import annotations

import gc
import json
import os
import pickle
import random
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from ms2spectra.training import FragGNNPL
from ms2spectra.workflow import (
    init_dataloader,
    init_dataset,
    load_config,
)

from train._impl.refinement_steps import (
    candidate_reranker,
    collision_energy_response,
    peak_distillation,
)


ROOT = Path(os.environ.get("FERA_MS_ROOT", Path(__file__).resolve().parents[1])).resolve()

TEMPLATE_PATH = (
    ROOT
    / "runs/_config/template.yml"
)

OUTPUT_ROOT = (
    ROOT
    / "runs/experiments/"
      "cumulative_refinement_analysis"
)

SPLIT_ROOTS = {
    "random": (
        ROOT
        / "runs/experiments/"
          "molecule_disjoint_3seeds"
    ),
    "scaffold": (
        ROOT
        / "runs/experiments/"
          "scaffold_disjoint_3seeds"
    ),
}

EXPECTED_TEST_COUNTS = {
    "random": 3931,
    "scaffold": 3960,
}

SEEDS = (42, 43, 44)

FORMAL_MODELS = [
    (
        1,
        "structured_fragment_dag_backbone",
        "Structured fragment-DAG backbone",
    ),
    (
        2,
        "formula_h_ce_refinement",
        "Formula/H-aware CE refinement",
    ),
    (
        3,
        "fragment_representation_refinement",
        "Fragment representation refinement",
    ),
    (
        4,
        "ce_flow_peak_entry_refinement",
        "CE-flow and peak-entry refinement",
    ),
    (
        5,
        "chemical_candidate_reranking",
        "Chemical candidate reranking",
    ),
    (
        6,
        "fera_ms",
        "FERA-MS",
    ),
]

MODEL_METADATA = {
    key: {
        "model_order": order,
        "model_variant": label,
    }
    for order, key, label in FORMAL_MODELS
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False


def release_cuda(*objects) -> None:
    for obj in objects:
        del obj

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)

    return path


def load_state_dict(path: Path):
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):
        return checkpoint["state_dict"]

    return checkpoint


def construct_model(
    config: dict,
    checkpoint_path: Path,
    device: torch.device,
    tag: str,
) -> FragGNNPL:
    model = FragGNNPL(**config)

    state_dict = load_state_dict(
        checkpoint_path
    )

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False,
    )

    print(
        f"[{tag}] missing={len(missing)} "
        f"unexpected={len(unexpected)}",
        flush=True,
    )

    if missing:
        for key in missing[:30]:
            print(
                f"  missing: {key}",
                flush=True,
            )

    if unexpected:
        for key in unexpected[:30]:
            print(
                f"  unexpected: {key}",
                flush=True,
            )

    if missing or unexpected:
        raise RuntimeError(
            f"{tag}: checkpoint architecture "
            f"does not match its reconstructed config"
        )

    model = model.to(device)
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return model


def formula_ce_args(
    use_ce_flowfrag: bool,
) -> SimpleNamespace:
    return SimpleNamespace(
        k3b_hidden=128,
        k3b_dropout=0.05,
        k3b_delta_scale=0.05,
        formula_comp_feat_size=18,

        ce_hidden=128,
        ce_dropout=0.05,
        ce_delta_scale=0.020,
        ce_use_formula_comp=True,
        ce_use_depth=True,
        ce_use_h=True,

        use_ce_flowfrag=bool(
            use_ce_flowfrag
        ),
        ce_flowfrag_lambda_max=(
            0.15
            if use_ce_flowfrag
            else 0.0
        ),
        ce_flowfrag_hidden=128,
        ce_flowfrag_dropout=0.05,
        ce_flowfrag_max_depth=4,
        ce_flowfrag_mixture_hidden=128,
        ce_flowfrag_mixture_dropout=0.05,
        ce_flowfrag_mixture_init_bias=-3.0,
        ce_flowfrag_delta_clip=3.0,
        ce_flowfrag_use_direct_node=True,
        ce_flowfrag_direct_mix=0.35,

        bin_res=0.01,
        max_bins=0,

        ce_binned_aux_weight=0.0015,
        r117_weight=0.0,
        r117_false_weight=0.20,

        low_w=0.25,
        mid_w=1.75,
        high_w=2.25,
    )


def load_test_loader(
    config: dict,
):
    datasets = init_dataset(
        config,
        splits=("test",),
    )

    if len(datasets) != 1:
        raise RuntimeError(
            "Expected exactly one test dataset"
        )

    return init_dataloader(
        datasets[0],
        config,
    )


def extract_global_metrics(
    table: pd.DataFrame,
    expected_count: int,
    tag: str,
) -> tuple[float, float, int]:
    global_rows = table[
        table["ce_bucket"].astype(str)
        == "global"
    ]

    if len(global_rows) != 1:
        raise RuntimeError(
            f"{tag}: expected one global row, "
            f"found {len(global_rows)}"
        )

    row = global_rows.iloc[0]

    count = int(row["spec_count"])

    if count != expected_count:
        raise RuntimeError(
            f"{tag}: expected {expected_count} "
            f"test spectra, found {count}"
        )

    return (
        float(row["cos"]),
        float(row["jss"]),
        count,
    )


def evaluate_neural_model(
    model: FragGNNPL,
    loader,
    device: torch.device,
    output_dir: Path,
    expected_count: int,
    tag: str,
) -> tuple[float, float, int]:
    eval_args = SimpleNamespace(
        eval_bin_res=0.01,
    )

    table, detail = (
        peak_distillation.eval_split(
            model=model,
            dl=loader,
            device=device,
            args=eval_args,
            split="test",
        )
    )

    table.to_csv(
        output_dir
        / "test_metrics.csv",
        index=False,
    )

    detail.to_csv(
        output_dir
        / "test_per_spectrum.csv.gz",
        index=False,
        compression="gzip",
    )

    cosine, jss, count = (
        extract_global_metrics(
            table=table,
            expected_count=expected_count,
            tag=tag,
        )
    )

    print()
    print(
        f"[{tag}] "
        f"micro CBIN={cosine:.9f}, "
        f"micro JSS={jss:.9f}",
        flush=True,
    )

    return cosine, jss, count


def find_best_alpha(
    reranker_dir: Path,
) -> float:
    best_alpha_path = (
        reranker_dir
        / "best_alpha.txt"
    )

    if best_alpha_path.is_file():
        return float(
            best_alpha_path
            .read_text(
                encoding="utf-8"
            )
            .strip()
        )

    alpha_table_path = require_file(
        reranker_dir
        / "r170_alpha_val.csv"
    )

    alpha_table = pd.read_csv(
        alpha_table_path
    )

    alpha_table["alpha"] = pd.to_numeric(
        alpha_table["alpha"],
        errors="raise",
    )

    alpha_table["val_cos"] = pd.to_numeric(
        alpha_table["val_cos"],
        errors="raise",
    )

    if "val_jss" not in alpha_table:
        alpha_table["val_jss"] = 0.0

    best = (
        alpha_table
        .sort_values(
            [
                "val_cos",
                "val_jss",
                "alpha",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )
        .iloc[0]
    )

    return float(best["alpha"])


def evaluate_reranked_model(
    backbone: FragGNNPL,
    loader,
    device: torch.device,
    reranker_path: Path,
    reranker_dir: Path,
    output_dir: Path,
    expected_count: int,
    tag: str,
) -> tuple[float, float, int, float]:
    with reranker_path.open(
        "rb"
    ) as handle:
        package = pickle.load(
            handle
        )

    regressor = package["model"]

    extra_schema = package.get(
        "extra_schema",
        [],
    )

    saved_args = dict(
        package.get(
            "args",
            {},
        )
    )

    alpha = find_best_alpha(
        reranker_dir
    )

    args = SimpleNamespace(
        max_extra_dims=int(
            saved_args.get(
                "max_extra_dims",
                96,
            )
        ),
        local_bin_res=float(
            saved_args.get(
                "local_bin_res",
                0.01,
            )
        ),
        score_clip=float(
            saved_args.get(
                "score_clip",
                6.0,
            )
        ),
        eval_bin_res=float(
            saved_args.get(
                "eval_bin_res",
                0.01,
            )
        ),
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                r"X does not have valid "
                r"feature names.*"
            ),
        )

        table = candidate_reranker.eval_split(
            base=backbone,
            regressor=regressor,
            extra_schema=extra_schema,
            dl=loader,
            device=device,
            args=args,
            split="test",
            alpha=alpha,
        )

    table.to_csv(
        output_dir
        / "test_metrics.csv",
        index=False,
    )

    (
        cosine,
        jss,
        count,
    ) = extract_global_metrics(
        table=table,
        expected_count=expected_count,
        tag=tag,
    )

    (
        output_dir
        / "selected_alpha.txt"
    ).write_text(
        f"{alpha:.12g}\n",
        encoding="utf-8",
    )

    print()
    print(
        f"[{tag}] "
        f"alpha={alpha:.12g}, "
        f"micro CBIN={cosine:.9f}, "
        f"micro JSS={jss:.9f}",
        flush=True,
    )

    return cosine, jss, count, alpha


def add_result(
    rows: list[dict],
    split_name: str,
    seed: int,
    model_key: str,
    cosine: float,
    jss: float,
    spectrum_count: int,
    artifact: Path,
    note: str = "",
) -> None:
    metadata = MODEL_METADATA[
        model_key
    ]

    rows.append(
        {
            "split":
                split_name,
            "seed":
                int(seed),
            "model_order":
                int(
                    metadata[
                        "model_order"
                    ]
                ),
            "model_key":
                model_key,
            "model_variant":
                metadata[
                    "model_variant"
                ],
            "micro_cbin":
                float(cosine),
            "micro_jss":
                float(jss),
            "test_spectrum_count":
                int(spectrum_count),
            "artifact":
                str(artifact),
            "note":
                str(note),
        }
    )


def evaluate_one_seed(
    split_name: str,
    seed: int,
    device: torch.device,
) -> list[dict]:
    split_root = SPLIT_ROOTS[
        split_name
    ]

    expected_count = (
        EXPECTED_TEST_COUNTS[
            split_name
        ]
    )

    seed_dir = (
        split_root
        / f"seed_{seed}"
    )

    base_dir = (
        seed_dir
        / "v2c_ce_trajectory_ablation/"
          "control"
    )

    refinement_dir = (
        seed_dir
        / "v2e_full_063"
    )

    config_path = require_file(
        base_dir
        / "config.yml"
    )

    backbone_checkpoint = require_file(
        base_dir
        / "model_best.ckpt"
    )

    formula_checkpoint = require_file(
        refinement_dir
        / "05_R150B/"
          "r148_best_state.pt"
    )

    fragment_checkpoint = require_file(
        refinement_dir
        / "06_R153/"
          "r148_best_state.pt"
    )

    distilled_checkpoint = require_file(
        refinement_dir
        / "08_R160/"
          "r160_best_state.pt"
    )

    reranker_dir = (
        refinement_dir
        / "09_R172D"
    )

    reranker_path = require_file(
        reranker_dir
        / "r170_regressor.pkl"
    )

    final_result_candidates = [
        (
            refinement_dir
            / "final_locked_evaluation/"
              "final_evaluation_with_"
              "molecule_aggregates.json"
        ),
        (
            refinement_dir
            / "final_locked_evaluation/"
              "final_evaluation.json"
        ),
    ]

    final_result_path = next(
        (
            path
            for path
            in final_result_candidates
            if path.is_file()
        ),
        None,
    )

    if final_result_path is None:
        raise FileNotFoundError(
            final_result_candidates[0]
        )

    output_dir = (
        OUTPUT_ROOT
        / split_name
        / f"seed_{seed}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    seed_result_path = (
        output_dir
        / "cumulative_refinement_seed_results.csv"
    )

    if seed_result_path.is_file():
        existing = pd.read_csv(
            seed_result_path
        )

        expected_model_keys = {
            key
            for _, key, _
            in FORMAL_MODELS
        }

        existing_model_keys = set(
            existing.get(
                "model_key",
                pd.Series(dtype=str),
            ).astype(str)
        )

        if (
            len(existing) == len(FORMAL_MODELS)
            and existing_model_keys
            == expected_model_keys
        ):
            print(
                f"[RESUME] {split_name} seed={seed} "
                "already complete",
                flush=True,
            )

            return existing.to_dict(
                orient="records"
            )

    print()
    print("#" * 110)
    print(
        f"START {split_name.upper()} "
        f"SEED {seed}"
    )
    print("#" * 110)
    print("config        :", config_path)
    print("base checkpoint:", backbone_checkpoint)
    print("formula ckpt   :", formula_checkpoint)
    print("fragment ckpt  :", fragment_checkpoint)
    print("distilled ckpt :", distilled_checkpoint)
    print("reranker       :", reranker_path)
    print("final result   :", final_result_path)
    print("#" * 110)

    seed_everything(seed)

    rows: list[dict] = []

    # ------------------------------------------------------------------
    # Shared base and rich test loaders
    # ------------------------------------------------------------------

    base_config = load_config(
        TEMPLATE_PATH,
        config_path,
    )

    rich_config = load_config(
        TEMPLATE_PATH,
        config_path,
    )

    rich_config = (
        candidate_reranker
        .force_r160_arch(
            rich_config
        )
    )

    print(
        "\nLoading base test dataset...",
        flush=True,
    )

    base_loader = load_test_loader(
        base_config
    )

    print(
        "Loading formula/rich test dataset...",
        flush=True,
    )

    rich_loader = load_test_loader(
        rich_config
    )

    # ------------------------------------------------------------------
    # Structured fragment-DAG backbone
    # ------------------------------------------------------------------

    variant_dir = (
        output_dir
        / "structured_fragment_dag_backbone"
    )

    variant_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    backbone_model = construct_model(
        config=base_config,
        checkpoint_path=backbone_checkpoint,
        device=device,
        tag=(
            f"{split_name}/seed_{seed}/"
            "Structured fragment-DAG backbone"
        ),
    )

    cosine, jss, count = (
        evaluate_neural_model(
            model=backbone_model,
            loader=base_loader,
            device=device,
            output_dir=variant_dir,
            expected_count=expected_count,
            tag=(
                f"{split_name}/seed_{seed}/"
                "Structured fragment-DAG backbone"
            ),
        )
    )

    add_result(
        rows=rows,
        split_name=split_name,
        seed=seed,
        model_key=(
            "structured_fragment_dag_backbone"
        ),
        cosine=cosine,
        jss=jss,
        spectrum_count=count,
        artifact=backbone_checkpoint,
    )

    release_cuda(
        backbone_model
    )

    # ------------------------------------------------------------------
    # Formula/H-aware CE refinement
    # ------------------------------------------------------------------

    variant_dir = (
        output_dir
        / "formula_h_ce_refinement"
    )

    variant_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    formula_config = load_config(
        TEMPLATE_PATH,
        config_path,
    )

    formula_config = (
        collision_energy_response
        .override_cfg_r147(
            formula_config,
            formula_ce_args(
                use_ce_flowfrag=False
            ),
        )
    )

    formula_model = construct_model(
        config=formula_config,
        checkpoint_path=formula_checkpoint,
        device=device,
        tag=(
            f"{split_name}/seed_{seed}/"
            "Formula-H-aware CE refinement"
        ),
    )

    cosine, jss, count = (
        evaluate_neural_model(
            model=formula_model,
            loader=rich_loader,
            device=device,
            output_dir=variant_dir,
            expected_count=expected_count,
            tag=(
                f"{split_name}/seed_{seed}/"
                "Formula-H-aware CE refinement"
            ),
        )
    )

    add_result(
        rows=rows,
        split_name=split_name,
        seed=seed,
        model_key=(
            "formula_h_ce_refinement"
        ),
        cosine=cosine,
        jss=jss,
        spectrum_count=count,
        artifact=formula_checkpoint,
    )

    release_cuda(
        formula_model
    )

    # ------------------------------------------------------------------
    # Fragment representation refinement
    # ------------------------------------------------------------------

    variant_dir = (
        output_dir
        / "fragment_representation_refinement"
    )

    variant_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fragment_config = load_config(
        TEMPLATE_PATH,
        config_path,
    )

    fragment_config = (
        collision_energy_response
        .override_cfg_r147(
            fragment_config,
            formula_ce_args(
                use_ce_flowfrag=False
            ),
        )
    )

    fragment_model = construct_model(
        config=fragment_config,
        checkpoint_path=fragment_checkpoint,
        device=device,
        tag=(
            f"{split_name}/seed_{seed}/"
            "Fragment representation refinement"
        ),
    )

    cosine, jss, count = (
        evaluate_neural_model(
            model=fragment_model,
            loader=rich_loader,
            device=device,
            output_dir=variant_dir,
            expected_count=expected_count,
            tag=(
                f"{split_name}/seed_{seed}/"
                "Fragment representation refinement"
            ),
        )
    )

    add_result(
        rows=rows,
        split_name=split_name,
        seed=seed,
        model_key=(
            "fragment_representation_refinement"
        ),
        cosine=cosine,
        jss=jss,
        spectrum_count=count,
        artifact=fragment_checkpoint,
        note=(
            "stable fragment-representation "
            "checkpoint before validation-gated "
            "CE-flow refinement"
        ),
    )

    release_cuda(
        fragment_model
    )

    # ------------------------------------------------------------------
    # CE-flow and peak-entry refinement
    # ------------------------------------------------------------------

    variant_dir = (
        output_dir
        / "ce_flow_peak_entry_refinement"
    )

    variant_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    distilled_model = construct_model(
        config=rich_config,
        checkpoint_path=distilled_checkpoint,
        device=device,
        tag=(
            f"{split_name}/seed_{seed}/"
            "CE-flow and peak-entry refinement"
        ),
    )

    cosine, jss, count = (
        evaluate_neural_model(
            model=distilled_model,
            loader=rich_loader,
            device=device,
            output_dir=variant_dir,
            expected_count=expected_count,
            tag=(
                f"{split_name}/seed_{seed}/"
                "CE-flow and peak-entry refinement"
            ),
        )
    )

    add_result(
        rows=rows,
        split_name=split_name,
        seed=seed,
        model_key=(
            "ce_flow_peak_entry_refinement"
        ),
        cosine=cosine,
        jss=jss,
        spectrum_count=count,
        artifact=distilled_checkpoint,
    )

    # ------------------------------------------------------------------
    # Chemical candidate reranking
    # ------------------------------------------------------------------

    variant_dir = (
        output_dir
        / "chemical_candidate_reranking"
    )

    variant_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        cosine,
        jss,
        count,
        alpha,
    ) = evaluate_reranked_model(
        backbone=distilled_model,
        loader=rich_loader,
        device=device,
        reranker_path=reranker_path,
        reranker_dir=reranker_dir,
        output_dir=variant_dir,
        expected_count=expected_count,
        tag=(
            f"{split_name}/seed_{seed}/"
            "Chemical candidate reranking"
        ),
    )

    add_result(
        rows=rows,
        split_name=split_name,
        seed=seed,
        model_key=(
            "chemical_candidate_reranking"
        ),
        cosine=cosine,
        jss=jss,
        spectrum_count=count,
        artifact=reranker_path,
        note=f"validated alpha={alpha:.12g}",
    )

    release_cuda(
        distilled_model
    )

    # ------------------------------------------------------------------
    # Final FERA-MS: read the already locked exact result
    # ------------------------------------------------------------------

    final_result = json.loads(
        final_result_path.read_text(
            encoding="utf-8"
        )
    )

    final_test = final_result[
        "test"
    ]

    final_cosine = float(
        final_test[
            "cosine"
        ]
    )

    final_jss = float(
        final_test[
            "jss"
        ]
    )

    final_count = int(
        final_test[
            "spectrum_count"
        ]
    )

    if final_count != expected_count:
        raise RuntimeError(
            f"{split_name}/seed_{seed}/FERA-MS: "
            f"expected {expected_count} spectra, "
            f"found {final_count}"
        )

    variant_dir = (
        output_dir
        / "fera_ms"
    )

    variant_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        variant_dir
        / "locked_result_source.txt"
    ).write_text(
        str(final_result_path)
        + "\n",
        encoding="utf-8",
    )

    add_result(
        rows=rows,
        split_name=split_name,
        seed=seed,
        model_key="fera_ms",
        cosine=final_cosine,
        jss=final_jss,
        spectrum_count=final_count,
        artifact=final_result_path,
        note=(
            "existing locked evaluation; "
            "test not used for selection"
        ),
    )

    print()
    print(
        f"[{split_name}/seed_{seed}/FERA-MS] "
        f"micro CBIN={final_cosine:.9f}, "
        f"micro JSS={final_jss:.9f}",
        flush=True,
    )

    seed_frame = (
        pd.DataFrame(rows)
        .sort_values(
            "model_order"
        )
        .reset_index(drop=True)
    )

    seed_frame.to_csv(
        output_dir
        / "cumulative_refinement_seed_results.csv",
        index=False,
    )

    print()
    print(
        seed_frame[
            [
                "model_variant",
                "micro_cbin",
                "micro_jss",
                "test_spectrum_count",
            ]
        ].to_string(
            index=False
        )
    )

    return rows


def build_summaries(
    raw: pd.DataFrame,
) -> None:
    raw_path = (
        OUTPUT_ROOT
        / "cumulative_refinement_raw.csv"
    )

    raw.to_csv(
        raw_path,
        index=False,
    )

    summary = (
        raw.groupby(
            [
                "model_order",
                "model_key",
                "model_variant",
                "split",
            ],
            as_index=False,
            sort=False,
        )
        .agg(
            micro_cbin_mean=(
                "micro_cbin",
                "mean",
            ),
            micro_cbin_std=(
                "micro_cbin",
                lambda values:
                    values.std(
                        ddof=1
                    ),
            ),
            micro_jss_mean=(
                "micro_jss",
                "mean",
            ),
            micro_jss_std=(
                "micro_jss",
                lambda values:
                    values.std(
                        ddof=1
                    ),
            ),
            n_seeds=(
                "seed",
                "nunique",
            ),
        )
        .sort_values(
            [
                "model_order",
                "split",
            ]
        )
        .reset_index(drop=True)
    )

    summary_path = (
        OUTPUT_ROOT
        / "cumulative_refinement_numeric_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    table_rows = []

    for (
        model_order,
        model_key,
        model_variant,
    ) in [
        (
            order,
            key,
            label,
        )
        for order, key, label
        in FORMAL_MODELS
    ]:
        row = {
            "Model variant":
                model_variant,
        }

        for split_name, heading in (
            ("random", "Random"),
            ("scaffold", "Scaffold"),
        ):
            selected = summary[
                (
                    summary[
                        "model_key"
                    ]
                    == model_key
                )
                & (
                    summary[
                        "split"
                    ]
                    == split_name
                )
            ]

            if len(selected) != 1:
                raise RuntimeError(
                    f"Missing summary row: "
                    f"{model_key}/{split_name}"
                )

            selected = selected.iloc[0]

            if int(
                selected[
                    "n_seeds"
                ]
            ) != 3:
                raise RuntimeError(
                    f"{model_key}/{split_name}: "
                    "expected 3 seeds"
                )

            row[
                f"{heading} micro CBIN"
            ] = (
                f"{float(selected['micro_cbin_mean']):.6f}"
                f" ± "
                f"{float(selected['micro_cbin_std']):.6f}"
            )

            row[
                f"{heading} micro JSS"
            ] = (
                f"{float(selected['micro_jss_mean']):.6f}"
                f" ± "
                f"{float(selected['micro_jss_std']):.6f}"
            )

        table_rows.append(row)

    publication_table = pd.DataFrame(
        table_rows
    )

    publication_path = (
        OUTPUT_ROOT
        / "cumulative_refinement_publication_table.csv"
    )

    publication_table.to_csv(
        publication_path,
        index=False,
    )

    print()
    print("=" * 140)
    print(
        "CUMULATIVE REFINEMENT ANALYSIS"
    )
    print("=" * 140)
    print(
        publication_table.to_string(
            index=False
        )
    )
    print("=" * 140)
    print("raw         :", raw_path)
    print("numeric     :", summary_path)
    print("publication :", publication_path)
    print(
        "CUMULATIVE_REFINEMENT_ANALYSIS_COMPLETE"
    )


def main() -> None:
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not TEMPLATE_PATH.is_file():
        raise FileNotFoundError(
            TEMPLATE_PATH
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available"
        )

    device = torch.device(
        "cuda"
    )

    print("=" * 110)
    print(
        "CUMULATIVE REFINEMENT ANALYSIS"
    )
    print("=" * 110)
    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )
    print(
        "Output:",
        OUTPUT_ROOT,
    )
    print(
        "Formal variants:",
    )

    for order, _, label in FORMAL_MODELS:
        print(
            f"  {order}. {label}"
        )

    print("=" * 110)

    all_rows = []

    for split_name in (
        "random",
        "scaffold",
    ):
        for seed in SEEDS:
            all_rows.extend(
                evaluate_one_seed(
                    split_name=split_name,
                    seed=seed,
                    device=device,
                )
            )

    raw = pd.DataFrame(
        all_rows
    )

    expected_rows = (
        len(SPLIT_ROOTS)
        * len(SEEDS)
        * len(FORMAL_MODELS)
    )

    if len(raw) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} result rows, "
            f"found {len(raw)}"
        )

    build_summaries(
        raw
    )


if __name__ == "__main__":
    main()
