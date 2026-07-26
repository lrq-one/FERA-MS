from __future__ import annotations

import importlib
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[3]

sys.path.insert(0, str(ROOT / "code/src"))
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT))

OUT_DIR = (
    ROOT
    / "ablation_studies"
    / "fera_ms_core_ablation"
    / "results"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

checks: list[dict[str, Any]] = []


def record(
    name: str,
    passed: bool,
    detail: str,
    critical: bool = True,
) -> None:
    checks.append(
        {
            "name": name,
            "passed": bool(passed),
            "critical": bool(critical),
            "detail": str(detail),
        }
    )

    if passed:
        status = "PASS"
    elif critical:
        status = "FAIL"
    else:
        status = "WARN"

    print(
        f"[{status}] {name}: {detail}",
        flush=True,
    )


print("=" * 110)
print("FERA-MS CORE ABLATION PREFLIGHT")
print("=" * 110)
print("ROOT:", ROOT)
print("CONDA:", os.environ.get("CONDA_DEFAULT_ENV", "unset"))
print("=" * 110)


required_files = [
    ROOT / "code/src/ms2spectra/model.py",
    ROOT / "code/src/ms2spectra/training.py",
    ROOT / "code/src/ms2spectra/utils/nn_utils.py",
    ROOT / "code/src/ms2spectra/utils/frag_utils.py",
    ROOT / "train/_impl/base_training.py",
    ROOT / "train/_impl/control_finetuning.py",
    ROOT / "train/_impl/run_refinement.sh",
    ROOT / "train/_impl/refinement_steps/neural_refinement.py",
    ROOT / "train/_impl/refinement_steps/peak_distillation.py",
    ROOT / "train/_impl/refinement_steps/candidate_reranker.py",
    ROOT / "train/_impl/refinement_steps/spectrum_allocator.py",
    ROOT / "test/evaluate.py",
    ROOT / "runs/_config/template.yml",
]

for path in required_files:
    record(
        f"file:{path.relative_to(ROOT)}",
        path.is_file(),
        str(path),
    )


try:
    import torch

    record(
        "torch import",
        True,
        torch.__version__,
    )

    record(
        "CUDA available",
        torch.cuda.is_available(),
        (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "CUDA unavailable"
        ),
    )
except Exception as exc:
    record(
        "torch/CUDA",
        False,
        repr(exc),
    )


for module_name in (
    "lightgbm",
    "pandas",
    "yaml",
    "torch_geometric",
):
    try:
        module = importlib.import_module(
            module_name
        )

        record(
            f"import:{module_name}",
            True,
            getattr(
                module,
                "__version__",
                "available",
            ),
        )
    except Exception as exc:
        record(
            f"import:{module_name}",
            False,
            repr(exc),
        )


model_path = (
    ROOT
    / "code/src/ms2spectra/model.py"
)

nn_utils_path = (
    ROOT
    / "code/src/ms2spectra/utils/nn_utils.py"
)

allocator_path = (
    ROOT
    / "train/_impl/refinement_steps/"
      "spectrum_allocator.py"
)

if nn_utils_path.is_file():
    nn_source = nn_utils_path.read_text(
        encoding="utf-8"
    )

    record(
        "A1 NodeMLP implementation",
        (
            'self.gnn_type == "NodeMLP"'
            in nn_source
            and "class NodeMLP"
            in nn_source
        ),
        "NodeMLP must exist in the shared GNN wrapper",
    )


if model_path.is_file():
    model_source = model_path.read_text(
        encoding="utf-8"
    )

    record(
        "cut-chemistry edge support",
        (
            '"cut_chem" in self.frag_edge_feats'
            in model_source
        ),
        "Full model must expose cut_chem edge features",
    )

    record(
        "atom-subset projection support",
        (
            "frag_node_mask_embed = scatter_reduce"
            in model_source
        ),
        "Full model must construct fragment-specific atom-subset embeddings",
    )

    record(
        "A2 generic-edge mode already present",
        (
            "ablation_topology_only"
            in model_source
        ),
        (
            "Expected to be absent before the dedicated "
            "ablation patch is installed"
        ),
        critical=False,
    )

    record(
        "A3 global-context interstage already present",
        (
            'cc_interstage_type == "global"'
            in model_source
        ),
        (
            "Expected to be absent before the dedicated "
            "ablation patch is installed"
        ),
        critical=False,
    )


if allocator_path.is_file():
    allocator_source = allocator_path.read_text(
        encoding="utf-8"
    )

    record(
        "A5 alpha-zero reranker removal",
        (
            "float(args.alpha) * lgbm_score"
            in allocator_source
        ),
        (
            "Allocator must support alpha=0 so it can be "
            "retrained without the candidate reranker"
        ),
    )


experiment_roots = {
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

artifact_suffixes = [
    (
        "control_config",
        Path(
            "v2c_ce_trajectory_ablation/"
            "control/config.yml"
        ),
    ),
    (
        "control_checkpoint",
        Path(
            "v2c_ce_trajectory_ablation/"
            "control/model_best.ckpt"
        ),
    ),
    (
        "refined_backbone",
        Path(
            "v2e_full_063/"
            "08_R160/r160_best_state.pt"
        ),
    ),
    (
        "candidate_reranker",
        Path(
            "v2e_full_063/"
            "09_R172D/r170_regressor.pkl"
        ),
    ),
    (
        "spectrum_allocator",
        Path(
            "v2e_full_063/"
            "11_R184B/r184_allocator_best.pt"
        ),
    ),
    (
        "locked_evaluation",
        Path(
            "v2e_full_063/"
            "final_locked_evaluation/"
            "final_evaluation.json"
        ),
    ),
]

resolved_splits: dict[str, str] = {}

for split_name, split_root in experiment_roots.items():
    for seed in (42, 43, 44):
        seed_root = (
            split_root
            / f"seed_{seed}"
        )

        for artifact_name, suffix in artifact_suffixes:
            path = seed_root / suffix

            record(
                (
                    f"{split_name}/seed_{seed}/"
                    f"{artifact_name}"
                ),
                path.is_file() and path.stat().st_size > 0,
                str(path),
            )

        config_path = (
            seed_root
            / "v2c_ce_trajectory_ablation/"
              "control/config.yml"
        )

        if config_path.is_file():
            config = yaml.safe_load(
                config_path.read_text(
                    encoding="utf-8"
                )
            )

            configured_seed = int(
                config.get(
                    "seed",
                    -1,
                )
            )

            record(
                (
                    f"{split_name}/seed_{seed}/"
                    "config_seed"
                ),
                configured_seed == seed,
                (
                    f"configured={configured_seed}, "
                    f"expected={seed}"
                ),
            )

            split_dp = Path(
                str(config["split_dp"])
            )

            if not split_dp.is_absolute():
                split_dp = ROOT / split_dp

            record(
                (
                    f"{split_name}/seed_{seed}/"
                    "split_directory"
                ),
                split_dp.is_dir(),
                str(split_dp),
            )

            resolved_splits[
                f"{split_name}_seed_{seed}"
            ] = str(split_dp)


try:
    from ms2spectra.model import FragGNNModel
    from ms2spectra.utils.nn_utils import GNN

    model_signature = inspect.signature(
        FragGNNModel.__init__
    )

    gnn_signature = inspect.signature(
        GNN.__init__
    )

    record(
        "FragGNNModel import",
        True,
        str(model_signature),
    )

    record(
        "GNN import",
        True,
        str(gnn_signature),
    )
except Exception as exc:
    record(
        "model imports",
        False,
        repr(exc),
    )


critical_failures = [
    item
    for item in checks
    if (
        item["critical"]
        and not item["passed"]
    )
]

report = {
    "root":
        str(ROOT),

    "conda_environment":
        os.environ.get(
            "CONDA_DEFAULT_ENV"
        ),

    "checks":
        checks,

    "resolved_splits":
        resolved_splits,

    "critical_failure_count":
        len(critical_failures),

    "ready_for_implementation":
        len(critical_failures) == 0,

    "planned_ablations": [
        "fragment_node_mlp",
        "topology_only_dag",
        "global_molecular_context",
        "global_ace_only",
        "no_candidate_reranker",
    ],
}

report_path = (
    OUT_DIR
    / "preflight_report.json"
)

report_path.write_text(
    json.dumps(
        report,
        indent=2,
        ensure_ascii=False,
    )
    + "\n",
    encoding="utf-8",
)

print()
print("=" * 110)
print(
    "CRITICAL_FAILURES:",
    len(critical_failures),
)
print(
    "READY_FOR_IMPLEMENTATION:",
    len(critical_failures) == 0,
)
print(
    "REPORT:",
    report_path,
)
print("=" * 110)

if critical_failures:
    raise SystemExit(1)
