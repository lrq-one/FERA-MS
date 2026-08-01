#!/usr/bin/env python3
"""Fail-fast checks for the public FERA-MS source release."""

from __future__ import annotations

import ast
import importlib
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
SOURCE_ROOTS = (
    ROOT / "code",
    ROOT / "train",
    ROOT / "test",
    ROOT / "preproc_scripts",
    ROOT / "ablation_studies",
    ROOT / "baseline_rebuild",
    ROOT / "config",
    ROOT / "scripts",
)
REQUIRED = (
    "README.md",
    "LICENSE_PENDING.md",
    "CITATION.cff",
    "requirements.txt",
    "environment.yml",
    "config/train.yml",
    "train/train.py",
    "train/_impl/base_training.py",
    "train/_impl/control_finetuning.py",
    "train/_impl/run_refinement.sh",
    "train/_impl/refinement_steps/candidate_reranker.py",
    "train/_impl/refinement_steps/spectrum_allocator.py",
    "test/evaluate.py",
    "preproc_scripts/03_prepare_dag_feats.py",
    "preproc_scripts/04_prepare_split.py",
    "docs/PIPELINE.md",
    "docs/REPRODUCIBILITY.md",
    "data/README.md",
)
FORBIDDEN_DIR_NAMES = {
    "__pycache__",
    "code_backup",
    "cysignals_crash_logs",
    "figure",
    "figures",
    "results_v2",
}
FORBIDDEN_ARTIFACT_SUFFIXES = {
    ".ckpt",
    ".pt",
    ".pth",
    ".pkl",
    ".pickle",
    ".joblib",
    ".npy",
    ".npz",
    ".png",
    ".pdf",
    ".svg",
    ".so",
}
TEXT_SUFFIXES = {".py", ".sh", ".yml", ".yaml", ".json", ".md", ".env", ".toml"}
ABSOLUTE_PATTERNS = (
    re.compile(r"/home/lwh(?:/|\b)"),
    re.compile(r"/mnt(?:/|\b)"),
    re.compile(r"/hy-tmp(?:/|\b)"),
    re.compile(r"fragnnet-main"),
    re.compile(r"ms2spectra_v1_r119"),
)


def fail(message: str, failures: list[str]) -> None:
    failures.append(message)
    print(f"FAIL: {message}")


def candidate_files() -> list[Path]:
    files: list[Path] = []
    for base in SOURCE_ROOTS:
        if not base.exists():
            continue
        files.extend(path for path in base.rglob("*") if path.is_file())
    return files


def git_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, check=True, text=True, capture_output=True
    )
    return [ROOT / line for line in result.stdout.splitlines() if line]


def main() -> int:
    failures: list[str] = []

    for relative in REQUIRED:
        if not (ROOT / relative).is_file():
            fail(f"missing required file: {relative}", failures)

    sys.path[:0] = [str(ROOT / "code/src"), str(ROOT)]
    for module_name in (
        "train._impl.config_builder",
        "ms2spectra.components.formula_features",
        "ms2spectra.losses.base",
    ):
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            fail(f"import {module_name}: {type(exc).__name__}: {exc}", failures)

    for path in candidate_files():
        relative = path.relative_to(ROOT)
        if any(part in FORBIDDEN_DIR_NAMES for part in relative.parts):
            fail(f"forbidden directory content: {relative}", failures)
        upper_name = path.name.upper()
        if "INVALID_DO_NOT_RUN" in upper_name or "INVALID_DO_NOT_RUN" in str(relative).upper():
            fail(f"invalid experiment marker: {relative}", failures)
        if path.suffix in FORBIDDEN_ARTIFACT_SUFFIXES:
            fail(f"generated/model artifact: {relative}", failures)
        if path.suffix == ".py":
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=str(relative))
            except (SyntaxError, UnicodeDecodeError) as exc:
                fail(f"Python syntax: {relative}: {exc}", failures)
        if path.suffix in TEXT_SUFFIXES:
            if path.resolve() == Path(__file__).resolve():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for pattern in ABSOLUTE_PATTERNS:
                if pattern.search(text):
                    fail(f"non-portable path token {pattern.pattern!r}: {relative}", failures)

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    referenced = set(
        re.findall(
            r"(?<![\w.-])([A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+\.(?:py|sh|yml|yaml))",
            readme,
        )
    )
    for relative in sorted(referenced):
        if relative.startswith(("data/", "runs/")):
            continue
        if not (ROOT / relative).is_file():
            fail(f"README references missing script/config: {relative}", failures)

    for path in git_files():
        if path.exists() and path.stat().st_size > 50 * 1024 * 1024:
            fail(f"tracked file exceeds 50 MB: {path.relative_to(ROOT)}", failures)
        relative = path.relative_to(ROOT)
        if path.suffix in FORBIDDEN_ARTIFACT_SUFFIXES:
            fail(f"tracked generated/model artifact: {relative}", failures)
        if "nist20" in path.name.lower() and path.suffix.lower() in {".msp", ".mol", ".pkl", ".bz2"}:
            fail(f"tracked NIST20 data: {relative}", failures)

    if failures:
        print(f"RELEASE_CHECK_FAILED ({len(failures)} issue(s))")
        return 1

    print("CORE_IMPORTS_OK")
    print("PORTABLE_PATHS_OK")
    print("NO_GENERATED_OR_LICENSED_ARTIFACTS")
    print("RELEASE_CHECK_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
