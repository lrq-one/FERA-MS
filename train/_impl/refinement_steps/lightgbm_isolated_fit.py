#!/usr/bin/env python3
"""Fit a LightGBM model without importing the PyTorch runtime."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", required=True)
    parser.add_argument("--y", required=True)
    parser.add_argument("--weight", required=True)
    parser.add_argument("--params", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if "torch" in sys.modules:
        raise RuntimeError(
            "Isolated LightGBM worker unexpectedly imported torch"
        )

    params = json.loads(
        Path(args.params).read_text(encoding="utf-8")
    )
    X = np.load(args.x, mmap_mode="r", allow_pickle=False)
    y = np.load(args.y, mmap_mode="r", allow_pickle=False)
    weight = np.load(
        args.weight,
        mmap_mode="r",
        allow_pickle=False,
    )

    print(
        "[isolated LightGBM]",
        f"version={lgb.__version__}",
        f"shape={X.shape}",
        f"workers={params.get('n_jobs')}",
        "torch_loaded=False",
        flush=True,
    )

    model = lgb.LGBMRegressor(**params)
    model.fit(X, y, sample_weight=weight)

    output = Path(args.output)
    with output.open("wb") as handle:
        pickle.dump(model, handle)

    print(
        "[isolated LightGBM] fit complete:",
        output,
        flush=True,
    )


if __name__ == "__main__":
    main()
