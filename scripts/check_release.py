#!/usr/bin/env python3
"""Fail-fast checks for the public FERA-MS source release."""

from __future__ import annotations

import ast
import importlib
import re
import subprocess
import sys
from pathlib import Path

import yaml


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
    "LICENSE",
    "CITATION.cff",
    "THIRD_PARTY_NOTICES.md",
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
    "preproc_scripts/prepare_dag_features.py",
    "preproc_scripts/prepare_split.py",
    "docs/PIPELINE.md",
    "docs/REPRODUCIBILITY.md",
    "docs/SOURCE_PROVENANCE.md",
    "docs/BASELINE_PROVENANCE.md",
    "data/README.md",
    "baseline_rebuild/baseline/source/fragnnet/LICENSE",
    "baseline_rebuild/baseline/source/fragnnet/UPSTREAM.md",
    "baseline_rebuild/baseline/source/fragnnet/src/fragnnet/model.py",
    "baseline_rebuild/baseline/source/fragnnet/src/fragnnet/iceberg/model.py",
    "baseline_rebuild/baseline/source/fiora/LICENSE",
    "baseline_rebuild/baseline/source/fiora/UPSTREAM.md",
    "baseline_rebuild/baseline/source/fiora/fiora/cli/predict.py",
    "baseline_rebuild/baseline/neims/UPSTREAM.md",
    "baseline_rebuild/baseline/massformer/UPSTREAM.md",
    "baseline_rebuild/baseline/fragnnet_depth_three/UPSTREAM.md",
    "baseline_rebuild/baseline/iceberg/UPSTREAM.md",
    "baseline_rebuild/baseline/graff_ms/UPSTREAM.md",
    "baseline_rebuild/baseline/fiora/UPSTREAM.md",
)
FORBIDDEN_DIR_NAMES = {
    "__pycache__",
    "code_backup",
    "cysignals_crash_logs",
    "figure",
    "figures",
    "results",
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
    re.compile(r"ms2spectra_base_model_[rv][0-9]+"),
)
NUMBERED_CODENAME_PATTERN = re.compile(
    r"(?:\b(?:R|V|K|r|v|k)[-_]?[0-9]+[A-Za-z0-9_-]*\b|"
    r"\b(?:stage|experiment)[-_ ]?[0-9]+[A-Za-z0-9_-]*\b)",
    re.IGNORECASE,
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
            relative_parts = relative.parts
            retained_baseline = relative_parts[:3] == (
                "baseline_rebuild",
                "baseline",
                "source",
            )
            locked_baseline_config = (
                relative_parts[:2] == ("baseline_rebuild", "baseline")
                and path.name == "locked.yml"
            )
            if (
                not retained_baseline
                and not locked_baseline_config
                and NUMBERED_CODENAME_PATTERN.search(text)
            ):
                fail(f"numbered internal codename: {relative}", failures)
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
        third_party_source = relative.parts[:3] == (
            "baseline_rebuild",
            "baseline",
            "source",
        )
        digit_name_allowlist = {"ms2spectra", "ms2c_utils.py"}
        if not third_party_source and any(
            any(character.isdigit() for character in part)
            and part not in digit_name_allowlist
            for part in relative.parts
        ):
            fail(f"numbered first-party path: {relative}", failures)
        if (
            len(relative.parts) >= 2
            and relative.parts[:2] == ("data", "split")
            and path.suffix.lower() == ".csv"
        ):
            fail(f"tracked record-level split CSV: {relative}", failures)
        if relative.parts and relative.parts[0] == "data" and any(
            part in {"raw", "proc", "frag"} for part in relative.parts[1:]
        ):
            fail(f"tracked licensed/derived data path: {relative}", failures)
        if path.suffix in FORBIDDEN_ARTIFACT_SUFFIXES:
            fail(f"tracked generated/model artifact: {relative}", failures)
        if "nist20" in path.name.lower() and path.suffix.lower() in {".msp", ".mol", ".pkl", ".bz2"}:
            fail(f"tracked NIST20 data: {relative}", failures)

    for nested_git in ROOT.rglob(".git"):
        if nested_git.resolve() != (ROOT / ".git").resolve():
            fail(f"nested third-party Git checkout: {nested_git.relative_to(ROOT)}", failures)

    notice = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    if "**BLOCKED**" in notice:
        fail("third-party notice still marks a retained baseline source route BLOCKED", failures)
    for baseline in ("NEIMS", "MassFormer", "FraGNNet-D3", "GrAFF-MS", "ICEBERG", "FIORA"):
        if baseline not in notice:
            fail(f"third-party notice missing baseline: {baseline}", failures)
    for retained_scope in (
        "baseline_rebuild/baseline/source/fragnnet/",
        "baseline_rebuild/baseline/source/fiora/",
    ):
        if retained_scope not in notice:
            fail(f"third-party notice missing retained scope: {retained_scope}", failures)

    fera_root = ROOT / "code/src/ms2spectra"
    for baseline_dir in ("massformer", "graff", "iceberg"):
        if (fera_root / baseline_dir).exists():
            fail(f"baseline implementation leaked into FERA-MS package: {baseline_dir}", failures)
    baseline_import = re.compile(r"ms2spectra\.(?:massformer|graff|iceberg)")
    for path in fera_root.rglob("*.py"):
        if baseline_import.search(path.read_text(encoding="utf-8")):
            fail(f"baseline import leaked into FERA-MS package: {path.relative_to(ROOT)}", failures)

    provenance = (ROOT / "docs/SOURCE_PROVENANCE.md").read_text(encoding="utf-8")
    source_suffixes = {".py", ".pyx", ".sh", ".yml", ".yaml", ".toml"}
    covered_roots = {
        "code",
        "train",
        "test",
        "preproc_scripts",
        "ablation_studies",
        "baseline_rebuild",
        "config",
        "scripts",
    }
    covered_root_files = {"setup.py", "pyproject.toml", "sitecustomize.py", "environment.yml"}
    for path in git_files():
        relative = path.relative_to(ROOT)
        if path.suffix.lower() not in source_suffixes:
            continue
        if relative.parts[0] not in covered_roots and str(relative) not in covered_root_files:
            fail(f"source file outside provenance scopes: {relative}", failures)
    for scope in (
        "code/src/ms2spectra/**",
        "baseline_rebuild/baseline/source/fragnnet/**",
        "baseline_rebuild/baseline/source/fiora/**",
    ):
        if scope not in provenance:
            fail(f"source provenance report lacks scope: {scope}", failures)

    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    expected_title = (
        "FERA-MS: Formula- and collision-energy-aware tandem mass spectrum "
        "prediction for molecular identification"
    )
    if citation.get("title") != expected_title:
        fail("CITATION.cff title mismatch", failures)
    if citation.get("version") != "0.1.0":
        fail("CITATION.cff version mismatch", failures)
    if str(citation.get("date-released")) != "2026-08-01":
        fail("CITATION.cff release date mismatch", failures)
    if citation.get("license") != "BSD-3-Clause":
        fail("CITATION.cff license mismatch", failures)
    readme_lower = readme.lower()
    split_disclosure = "record-level" in readme_lower and "not included" in readme_lower
    if not split_disclosure:
        fail("README lacks explicit record-level split non-distribution statement", failures)

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
