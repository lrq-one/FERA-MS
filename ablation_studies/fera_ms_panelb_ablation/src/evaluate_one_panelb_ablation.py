#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch


ROOT = Path(os.environ.get("FERA_MS_ROOT", Path(__file__).resolve().parents[3])).resolve()
RUNS_ROOT = Path(os.environ.get("FERA_MS_RUNS_DIR", ROOT / "runs")).resolve()
ABLATION_ROOT = (
    ROOT
    / "ablation_studies"
    / "fera_ms_panelb_ablation"
)

os.chdir(ROOT)

sys.path.insert(0, str(ROOT / "code/src"))
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT))

from ms2spectra.training import FragGNNPL
from ms2spectra.workflow import (
    init_dataloader,
    init_dataset,
    load_config,
)
from train._impl.refinement_steps import (
    peak_distillation,
)


EXPECTED_TEST_COUNT = 3931

VARIANTS = {
    "without_mz_offset_expansion",
    "without_rendered_peak_gate",
}


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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def verify_model(
    wrapper: FragGNNPL,
    config: dict,
    variant: str,
) -> dict:
    model = wrapper.model

    actual = {
        "frag_gnn_type":
            config.get("frag_gnn_type"),

        "fragment_edge_features":
            (
                config.get("frag_params", {})
                .get("pyg_edge_feats")
            ),

        "candidate_refiner_present":
            (
                getattr(
                    model,
                    "spectrum_candidate_refiner",
                    None,
                )
                is not None
            ),

        "mz_expansion":
            bool(
                model.use_mz_offset_peak_expansion
            ),

        "mz_offset_steps":
            [
                float(value)
                for value
                in model.mz_offset_peak_steps
            ],

        "rendered_gate":
            bool(
                model.use_rendered_peak_drop_gate
            ),

        "rendered_gate_module":
            (
                None
                if model.rendered_peak_drop_gate is None
                else model.rendered_peak_drop_gate
                    .__class__.__name__
            ),
    }

    assert actual["frag_gnn_type"] == "GINE"
    assert (
        actual["fragment_edge_features"]
        == ["cut_chem"]
    )
    assert (
        actual["candidate_refiner_present"]
        is True
    )

    if variant == "without_mz_offset_expansion":
        assert actual["mz_expansion"] is False
        assert actual["mz_offset_steps"] == [0.0]
        assert actual["rendered_gate"] is True
        assert actual["rendered_gate_module"] is not None

    elif variant == "without_rendered_peak_gate":
        assert actual["mz_expansion"] is True
        assert actual["mz_offset_steps"] == [
            -0.002,
            -0.001,
            0.0,
            0.001,
            0.002,
        ]
        assert actual["rendered_gate"] is False
        assert actual["rendered_gate_module"] is None

    return actual


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--variant",
        required=True,
        choices=sorted(VARIANTS),
    )

    parser.add_argument(
        "--seed",
        required=True,
        type=int,
        choices=[42, 43, 44],
    )

    args = parser.parse_args()

    variant = args.variant
    seed = args.seed

    run_root = (
        ABLATION_ROOT
        / "runs"
        / variant
        / f"seed_{seed}"
    )

    config_path = (
        run_root
        / "global_ace_control"
        / "control"
        / "config.yml"
    )

    checkpoint_path = (
        run_root
        / "global_ace_control"
        / "control"
        / "model_best.ckpt"
    )

    output_dir = (
        run_root
        / "evaluation"
    )

    for path in (
        config_path,
        checkpoint_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    seed_everything(seed)

    config = load_config(
        RUNS_ROOT / "_config/template.yml",
        config_path,
    )

    config["eval_test_split"] = False

    wrapper = FragGNNPL(**config)

    architecture = verify_model(
        wrapper,
        config,
        variant,
    )

    state_dict = load_state_dict(
        checkpoint_path
    )

    incompatibility = wrapper.load_state_dict(
        state_dict,
        strict=True,
    )

    print(
        "[STRICT TEST LOAD]",
        incompatibility,
        flush=True,
    )

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    wrapper = wrapper.to(device)
    wrapper.eval()

    for parameter in wrapper.parameters():
        parameter.requires_grad_(False)

    datasets = init_dataset(
        config,
        splits=("test",),
    )

    if len(datasets) != 1:
        raise RuntimeError(
            f"Expected one test dataset, got {len(datasets)}"
        )

    loader = init_dataloader(
        datasets[0],
        config,
    )

    table, detail = (
        peak_distillation.eval_split(
            model=wrapper,
            dl=loader,
            device=device,
            args=SimpleNamespace(
                eval_bin_res=0.01,
            ),
            split="test",
        )
    )

    table.to_csv(
        output_dir / "test_metrics.csv",
        index=False,
    )

    detail.to_csv(
        output_dir
        / "test_per_spectrum.csv.gz",
        index=False,
        compression="gzip",
    )

    global_rows = table[
        table["ce_bucket"].astype(str)
        == "global"
    ]

    if len(global_rows) != 1:
        raise RuntimeError(
            "Global test row count is not one"
        )

    row = global_rows.iloc[0]

    count = int(row["spec_count"])
    cosine = float(row["cos"])
    jss = float(row["jss"])

    if count != EXPECTED_TEST_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_TEST_COUNT} spectra, "
            f"found {count}"
        )

    summary = {
        "variant":
            variant,

        "seed":
            seed,

        "checkpoint":
            str(checkpoint_path),

        "test_spectrum_count":
            count,

        "micro_cbin":
            cosine,

        "micro_jss":
            jss,

        "architecture":
            architecture,
    }

    reference_path = (
        ROOT
        / "runs"
        / "experiments"
        / "cumulative_refinement_analysis"
        / "random"
        / f"seed_{seed}"
        / "structured_fragment_dag_backbone"
        / "test_metrics.csv"
    )

    if reference_path.is_file():
        reference_table = pd.read_csv(
            reference_path
        )

        reference_global = reference_table[
            reference_table[
                "ce_bucket"
            ].astype(str)
            == "global"
        ]

        if len(reference_global) == 1:
            reference_row = (
                reference_global.iloc[0]
            )

            reference_cbin = float(
                reference_row["cos"]
            )
            reference_jss = float(
                reference_row["jss"]
            )

            summary.update(
                {
                    "full_backbone_micro_cbin":
                        reference_cbin,

                    "full_backbone_micro_jss":
                        reference_jss,

                    "delta_cbin_vs_full":
                        cosine
                        - reference_cbin,

                    "delta_jss_vs_full":
                        jss
                        - reference_jss,
                }
            )

    summary_path = (
        output_dir
        / "test_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print("PANEL-B LOCKED TEST COMPLETE")
    print("=" * 100)
    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
    )
    print("=" * 100)


if __name__ == "__main__":
    main()
