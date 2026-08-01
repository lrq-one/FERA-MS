#!/usr/bin/env python3

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


BASE = Path(os.environ.get("FERA_MS_BASELINE_ROOT", Path(__file__).resolve().parents[1])).resolve()

MAIN = Path(os.environ.get("FERA_MS_BASELINE_SOURCE", BASE / "shared/fragnnet_main")).resolve()
ICEBERG = Path(os.environ.get("FERA_MS_ICEBERG_SOURCE", BASE / "shared/iceberg_core")).resolve()


def import_one_of(names):
    errors = []

    for name in names:
        try:
            module = importlib.import_module(name)
            return name, module
        except Exception as exc:
            errors.append(
                f"{name}: {type(exc).__name__}: {exc}"
            )

    raise RuntimeError(
        "None of the alternatives imported:\n"
        + "\n".join(errors)
    )


print("python =", sys.executable)
print("version =", sys.version)

required = [
    "numpy",
    "pandas",
    "yaml",
    "torch",
    "torchvision",
    "torchmetrics",
    "rdkit",
    "dgl",
    "torch_geometric",
]

for name in required:
    module = importlib.import_module(name)

    print(
        name,
        "=",
        getattr(
            module,
            "__version__",
            "<no version>",
        ),
    )

lightning_name, lightning = import_one_of(
    [
        "lightning",
        "pytorch_lightning",
    ]
)

print(
    lightning_name,
    "=",
    getattr(
        lightning,
        "__version__",
        "<no version>",
    ),
)

import numpy as np
import torch

if np.lib.NumpyVersion(
    np.__version__
) >= "2.0.0":
    raise RuntimeError(
        f"NumPy 2.x is not allowed: {np.__version__}"
    )

print("torch_cuda =", torch.version.cuda)
print("cuda_available =", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available"
    )

print(
    "gpu_count =",
    torch.cuda.device_count(),
)

for index in range(
    torch.cuda.device_count()
):
    print(
        f"gpu[{index}] =",
        torch.cuda.get_device_name(index),
        "capability =",
        torch.cuda.get_device_capability(index),
    )

device = torch.device("cuda:0")

a = torch.randn(
    1024,
    1024,
    device=device,
)

b = torch.randn(
    1024,
    1024,
    device=device,
)

c = a @ b

torch.cuda.synchronize()

if not torch.isfinite(c).all():
    raise RuntimeError(
        "CUDA matmul produced non-finite values"
    )

print(
    "CUDA_MATMUL_OK",
    float(c.mean()),
)

import dgl
import dgl.function as fn

graph = dgl.graph(
    (
        torch.tensor(
            [0, 1],
            device=device,
        ),
        torch.tensor(
            [1, 2],
            device=device,
        ),
    ),
    num_nodes=3,
    device=device,
)

features = torch.randn(
    3,
    8,
    device=device,
    requires_grad=True,
)

graph.ndata["x"] = features

graph.update_all(
    fn.copy_u("x", "m"),
    fn.sum("m", "h"),
)

loss = graph.ndata["h"].square().sum()
loss.backward()
torch.cuda.synchronize()

print("DGL_CUDA_BACKWARD_OK")

try:
    from torch_scatter import scatter

    src = torch.tensor(
        [1.0, 2.0, 3.0],
        device=device,
    )

    index = torch.tensor(
        [0, 0, 1],
        device=device,
    )

    result = scatter(
        src,
        index,
        dim=0,
        reduce="sum",
    )

    assert torch.allclose(
        result,
        torch.tensor(
            [3.0, 3.0],
            device=device,
        ),
    )

    print("TORCH_SCATTER_CUDA_OK")

except ImportError:
    print(
        "TORCH_SCATTER_NOT_INSTALLED_SEPARATELY"
    )

required_paths = [
    MAIN
    / "data/proc/"
    "nist20_qtof_cid_safe19659/"
    "spec_df.pkl",

    MAIN
    / "data/proc/"
    "nist20_qtof_cid_safe19659/"
    "mol_df.pkl",

    MAIN
    / "data/split/"
    "nist20_qtof_cid_safe19659_"
    "qcv1_trainonly/"
    "train_ids.csv",

    MAIN
    / "data/split/"
    "nist20_qtof_cid_safe19659_"
    "scaffold60_20_20_seed42/"
    "train_ids.csv",

    ICEBERG
    / "data/proc/"
    "nist20_qtof_cid_safe19659/"
    "spec_df.pkl",

    ICEBERG
    / "data/proc/"
    "nist20_qtof_cid_safe19659/"
    "mol_df.pkl",

    ICEBERG
    / "data/split/"
    "nist20_qtof_cid_safe19659_"
    "qcv1_trainonly/"
    "train_ids.csv",

    ICEBERG
    / "data/split/"
    "nist20_qtof_cid_safe19659_"
    "scaffold60_20_20_seed42/"
    "train_ids.csv",
]

for path in required_paths:
    if not path.is_file():
        raise FileNotFoundError(path)

    print(
        "DATA_OK",
        path,
        path.stat().st_size,
    )


def load_main_configs():
    old_path = list(sys.path)
    old_modules = {
        key: value
        for key, value in sys.modules.items()
        if key == "fragnnet"
        or key.startswith("fragnnet.")
    }

    try:
        sys.path.insert(
            0,
            str(MAIN / "src"),
        )

        from fragnnet.runner import load_config

        configs = [
            (
                "neims",
                "benchmark_audit/configs_clean/"
                "neims_ace_reference.yml",
            ),
            (
                "massformer",
                "benchmark_audit/configs_clean/"
                "massformer_ace_reference.yml",
            ),
            (
                "fragnnet_d3",
                "benchmark_audit/configs_clean/"
                "fragnnet_fragmentation_ace_reference.yml",
            ),
        ]

        os.chdir(MAIN)

        for label, config in configs:
            loaded = load_config(
                "config/template.yml",
                config,
            )

            print(
                "MAIN_CONFIG_OK",
                label,
                loaded.get("model_type"),
                loaded.get("split_dp"),
                loaded.get("max_epochs"),
            )

    finally:
        sys.path[:] = old_path

        for key in list(sys.modules):
            if (
                key == "fragnnet"
                or key.startswith("fragnnet.")
            ):
                del sys.modules[key]

        sys.modules.update(old_modules)


def load_iceberg_config():
    old_path = list(sys.path)

    try:
        sys.path.insert(
            0,
            str(ICEBERG / "src"),
        )

        from fragnnet.runner import load_config

        os.chdir(ICEBERG)

        loaded = load_config(
            "config/template.yml",
            "benchmark_audit/configs_clean/"
            "iceberg_core_reference.yml",
        )

        print(
            "ICEBERG_CONFIG_OK",
            loaded.get("model_type"),
            loaded.get("split_dp"),
            loaded.get("max_epochs"),
        )

    finally:
        sys.path[:] = old_path


load_main_configs()
load_iceberg_config()

print("LOCAL_4_MODEL_RUNTIME_READY")
