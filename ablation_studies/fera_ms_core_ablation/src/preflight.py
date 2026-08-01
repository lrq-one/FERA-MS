#!/usr/bin/env python3
"""Preflight for the validated no-candidate-reranker control."""

from __future__ import annotations

import json
import os
from pathlib import Path

import yaml


ROOT = Path(os.environ.get("FERA_MS_ROOT", Path(__file__).resolve().parents[3])).resolve()
RUNS_ROOT = Path(os.environ.get("FERA_MS_RUNS_DIR", ROOT / "runs")).resolve()
ABLATION_ROOT = ROOT / "ablation_studies/fera_ms_core_ablation"
OUTPUT = ABLATION_ROOT / "results/preflight_report.json"

required_source = (
    ROOT / "config/train.yml",
    ROOT / "train/_impl/refinement_steps/spectrum_allocator.py",
    ABLATION_ROOT / "config/ablation_plan.yaml",
    ABLATION_ROOT / "scripts/run_no_candidate_reranker.sh",
)

checks: list[dict[str, object]] = []
for path in required_source:
    checks.append({"kind": "source", "path": str(path), "ok": path.is_file()})

plan = yaml.safe_load((ABLATION_ROOT / "config/ablation_plan.yaml").read_text(encoding="utf-8"))
definition = plan["experiments"]["no_candidate_reranker"]
checks.append(
    {
        "kind": "definition",
        "path": "no_candidate_reranker.reranker_alpha",
        "ok": float(definition["implementation"]["reranker_alpha"]) == 0.0,
    }
)

for seed in (42, 43, 44):
    seed_root = RUNS_ROOT / f"experiments/molecule_disjoint_3seeds/seed_{seed}"
    for relative in (
        "global_ace_control_ce_trajectory_ablation/control/config.yml",
        "full_model_full_063/08_R160/final_peak_distillation_best_state.pt",
        "full_model_full_063/09_candidate_reranker/candidate_reranker_regressor.pkl",
    ):
        path = seed_root / relative
        checks.append({"kind": "upstream_artifact", "path": str(path), "ok": path.is_file()})

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps({"checks": checks}, indent=2) + "\n", encoding="utf-8")

failed = [item for item in checks if not item["ok"]]
for item in checks:
    print("PASS" if item["ok"] else "FAIL", item["kind"], item["path"])
if failed:
    raise SystemExit(f"NO_RERANKER_PREFLIGHT_FAILED ({len(failed)} missing/invalid item(s))")
print("NO_RERANKER_PREFLIGHT_OK")
