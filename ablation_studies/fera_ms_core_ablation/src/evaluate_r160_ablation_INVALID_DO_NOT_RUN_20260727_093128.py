from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

from ms2spectra.training import FragGNNPL
from ms2spectra.workflow import (
    apply_runtime_ablation,
    init_dataloader,
    init_dataset,
    load_config,
)

from train._impl.refinement_steps import (
    candidate_reranker,
    peak_distillation,
)


def load_state(path: Path):
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


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--template",
        required=True,
    )

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--out-dir",
        required=True,
    )

    parser.add_argument(
        "--variant",
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
    )

    args = parser.parse_args()

    os.environ[
        "MS2_ABLATION_VARIANT"
    ] = args.variant

    output_dir = Path(args.out_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = load_config(
        args.template,
        args.config,
    )

    config = (
        candidate_reranker
        .force_r160_arch(config)
    )

    config = apply_runtime_ablation(
        config
    )

    datasets = init_dataset(
        config,
        splits=("test",),
    )

    loader = init_dataloader(
        datasets[0],
        config,
    )

    model = FragGNNPL(
        **config
    )

    state = load_state(
        Path(args.checkpoint)
    )

    missing, unexpected = (
        model.load_state_dict(
            state,
            strict=False,
        )
    )

    print(
        "missing:",
        len(missing),
    )

    for key in missing[:30]:
        print("  missing:", key)

    print(
        "unexpected:",
        len(unexpected),
    )

    for key in unexpected[:30]:
        print("  unexpected:", key)

    if missing or unexpected:
        raise RuntimeError(
            "R160 checkpoint and effective "
            "ablation architecture do not match"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable"
        )

    device = torch.device(
        "cuda"
    )

    model = model.to(device)
    model.eval()

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

    global_rows = table[
        table["ce_bucket"].astype(str)
        == "global"
    ]

    if len(global_rows) != 1:
        raise RuntimeError(
            "Expected exactly one global row"
        )

    row = global_rows.iloc[0]

    count = int(row["spec_count"])
    cosine = float(row["cos"])
    jss = float(row["jss"])

    if count != 3931:
        raise RuntimeError(
            f"Random test count mismatch: "
            f"{count} != 3931"
        )

    result = {
        "variant":
            args.variant,

        "seed":
            int(args.seed),

        "split":
            "random_molecule_disjoint",

        "micro_cbin":
            cosine,

        "micro_jss":
            jss,

        "test_spectrum_count":
            count,

        "evaluation_bin_resolution":
            0.01,

        "checkpoint":
            str(
                Path(args.checkpoint)
            ),

        "test_used_for_selection":
            False,
    }

    result_path = (
        output_dir
        / "result.json"
    )

    result_path.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print("R160 ABLATION TEST RESULT")
    print("=" * 100)
    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )
    print("=" * 100)


if __name__ == "__main__":
    main()
