from __future__ import annotations

import bz2
import json
import os
import pickle
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

REFERENCE_DIR = (
    ROOT
    / "data/frag/"
      "nist20_qtof_cid_safe19659_d3_mhp_qtof_cid_nl_v1/dags"
)

TARGET_DIR = (
    ROOT
    / "runs/experiments/molecular_retrieval/"
      "pubchem_fixed50/candidate_dags/frag/dags"
)

AUDIT_PATH = (
    ROOT
    / "runs/experiments/molecular_retrieval/"
      "pubchem_fixed50/candidate_dags/"
      "nl_reconstruction_audit.json"
)

WORKERS = max(
    1,
    min(12, int(os.environ.get("EX5_NL_WORKERS", "8"))),
)

TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")

LOSSES = [
    ("H2O", {"H": 2, "O": 1}, 18.010564684),
    ("CO",  {"C": 1, "O": 1}, 27.994914620),
    ("CO2", {"C": 1, "O": 2}, 43.989829240),
    ("NH3", {"H": 3, "N": 1}, 17.026549101),
]

MATCH_TOLERANCE = 5.0e-3


def load_cache(path: Path) -> dict:
    with bz2.BZ2File(path, "rb") as handle:
        return pickle.load(handle)


def save_cache_atomic(path: Path, data: dict) -> None:
    temp = Path(str(path) + ".nl_tmp")
    try:
        with bz2.BZ2File(temp, "wb") as handle:
            pickle.dump(
                data,
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def as_numpy(value) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def formula_at(mapping, index: int) -> str:
    if isinstance(mapping, dict):
        value = mapping.get(index, mapping.get(str(index), ""))
    else:
        value = mapping[index]
    return "" if value is None else str(value)


def parse_formula(formula: str) -> dict[str, int]:
    formula = str(formula).strip()

    if not formula:
        return {}

    formula = formula.split("+", 1)[0].split("-", 1)[0]

    result: dict[str, int] = {}
    position = 0

    for match in TOKEN.finditer(formula):
        if match.start() != position:
            raise ValueError(f"Cannot parse molecular formula: {formula}")

        element, count = match.groups()
        result[element] = (
            result.get(element, 0)
            + (int(count) if count else 1)
        )
        position = match.end()

    if position != len(formula):
        raise ValueError(f"Cannot parse molecular formula: {formula}")

    return result


def sufficient(
    source: dict[str, int],
    loss: dict[str, int],
) -> bool:
    return all(
        source.get(element, 0) >= count
        for element, count in loss.items()
    )


def classify_delta(delta: float) -> str:
    candidates = sorted(
        (
            abs(delta - mass),
            name,
        )
        for name, _, mass in LOSSES
    )

    error, name = candidates[0]

    if error > MATCH_TOLERANCE:
        raise RuntimeError(
            f"Unknown NL mass difference: delta={delta:.9f}, "
            f"nearest={name}, error={error:.9f}"
        )

    return name


reference_files = sorted(
    REFERENCE_DIR.glob("*.pickle.bz2")
)

target_files = sorted(
    TARGET_DIR.glob("*.pickle.bz2")
)

if not reference_files:
    raise FileNotFoundError(
        f"Locked NL reference cache does not exist: {REFERENCE_DIR}"
    )

if not target_files:
    raise FileNotFoundError(
        f"Experiment 5 candidate cache does not exist: {TARGET_DIR}"
    )

print(
    f"[1/5] Read locked NL reference cache: "
    f"{len(reference_files)} files",
    flush=True,
)

widths = Counter()
probability_samples = {
    name: []
    for name, _, _ in LOSSES
}

reference_rows = []
unique_formulas = set()

for file_index, path in enumerate(reference_files, 1):
    cache = load_cache(path)

    required = {
        "formula_peak_mzs",
        "nl_formula_peak_mzs",
        "nl_formula_peak_probs",
        "idx_to_formula",
    }

    missing = required.difference(cache)

    if missing:
        raise RuntimeError(
            f"Reference cache missing fields: {path} -> {sorted(missing)}"
        )

    base = as_numpy(cache["formula_peak_mzs"])
    nl_mzs = as_numpy(cache["nl_formula_peak_mzs"])
    nl_probs = as_numpy(cache["nl_formula_peak_probs"])

    if (
        base.ndim != 2
        or nl_mzs.ndim != 2
        or nl_probs.ndim != 2
    ):
        raise RuntimeError(
            f"Unexpected reference-cache array dimensions: {path}"
        )

    if (
        nl_mzs.shape != nl_probs.shape
        or nl_mzs.shape[0] != base.shape[0]
    ):
        raise RuntimeError(
            f"Unexpected reference-cache array shape: {path}"
        )

    widths[int(nl_mzs.shape[1])] += 1
    mapping = cache["idx_to_formula"]

    for row in range(base.shape[0]):
        formula = formula_at(mapping, row)

        if not formula:
            continue

        composition = parse_formula(formula)
        base_mz = float(base[row, 0])
        actual = []

        for column in range(nl_mzs.shape[1]):
            probability = float(nl_probs[row, column])
            mz = float(nl_mzs[row, column])

            if probability <= 0.0 or mz <= 0.0:
                continue

            delta = base_mz - mz
            name = classify_delta(delta)

            if name in actual:
                raise RuntimeError(
                    f"Duplicate NL entries in the same row: "
                    f"{path.name}, formula={formula}, "
                    f"loss={name}"
                )

            actual.append(name)
            probability_samples[name].append(probability)

        reference_rows.append(
            (formula, composition, base_mz, tuple(actual))
        )
        unique_formulas.add(formula)

    if (
        file_index % 1000 == 0
        or file_index == len(reference_files)
    ):
        print(
            f"  Read {file_index}/{len(reference_files)}",
            flush=True,
        )

if len(widths) != 1:
    raise RuntimeError(
        f"Reference NL column count mismatch: {dict(widths)}"
    )

width = next(iter(widths))

mask_probability = {}

for name, _, _ in LOSSES:
    values = probability_samples[name]

    if not values:
        raise RuntimeError(
            f"Reference cache lacks NL template: {name}"
        )

    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))

    if np.max(np.abs(values - median)) > 1.0e-6:
        raise RuntimeError(
            f"NL mask probability is not fixed: {name}"
        )

    mask_probability[name] = median

print(
    f"[2/5] Lock NL template: "
    f"{len(LOSSES)} losses, cache column count={width}",
    flush=True,
)

for index, (name, loss, mass) in enumerate(
    LOSSES,
    1,
):
    print(
        f"  {index}. {name} "
        f"loss={loss} "
        f"mass={mass:.9f} "
        f"mask={mask_probability[name]:.6g}",
        flush=True,
    )

print(
    "[3/5] Verify 100% rule replay on the locked cache",
    flush=True,
)

mismatches = []

for formula, composition, base_mz, actual in reference_rows:
    predicted = tuple(
        name
        for name, loss, mass in LOSSES
        if sufficient(composition, loss)
        and composition != loss
        and base_mz > mass
    )[:width]

    if predicted != actual:
        mismatches.append(
            (formula, actual, predicted)
        )

        if len(mismatches) >= 20:
            break

if mismatches:
    print(
        "Rule replay did not reach 100%, will not modify experiment 5 cache."
    )

    for item in mismatches:
        print(item)

    raise RuntimeError(
        "LOCKED_NL_PARITY_FAILED"
    )

print(
    f"  LOCKED_NL_PARITY_OK："
    f"{len(reference_rows)} rows, "
    f"{len(unique_formulas)} unique formulas all match",
    flush=True,
)

print(
    f"[4/5] Enrich experiment 5 candidate cache: "
    f"{len(target_files)} files, workers={WORKERS}",
    flush=True,
)


def patch_one(path: Path) -> tuple[str, str]:
    cache = load_cache(path)

    base = cache.get("formula_peak_mzs")
    mapping = cache.get("idx_to_formula")

    if base is None or mapping is None:
        return (
            "error",
            f"{path.name}: "
            "missing formula_peak_mzs/idx_to_formula",
        )

    base_np = as_numpy(base)

    if base_np.ndim != 2:
        return (
            "error",
            f"{path.name}: invalid formula_peak_mzs",
        )

    mz_array = np.zeros(
        (base_np.shape[0], width),
        dtype=np.float32,
    )

    prob_array = np.zeros(
        (base_np.shape[0], width),
        dtype=np.float32,
    )

    for row in range(base_np.shape[0]):
        formula = formula_at(mapping, row)

        if not formula:
            continue

        composition = parse_formula(formula)
        base_mz = float(base_np[row, 0])

        selected = [
            (name, mass)
            for name, loss, mass in LOSSES
            if sufficient(composition, loss)
            and composition != loss
            and base_mz > mass
        ][:width]

        for column, (name, mass) in enumerate(selected):
            mz_array[row, column] = np.float32(
                base_mz - mass
            )

            prob_array[row, column] = np.float32(
                mask_probability[name]
            )

    dtype = (
        base.dtype
        if torch.is_tensor(base)
        else torch.float32
    )

    cache["nl_formula_peak_mzs"] = torch.as_tensor(
        mz_array,
        dtype=dtype,
    )

    cache["nl_formula_peak_probs"] = torch.as_tensor(
        prob_array,
        dtype=dtype,
    )

    save_cache_atomic(path, cache)

    return "patched", path.name


counts = Counter()
errors = []

with ThreadPoolExecutor(
    max_workers=WORKERS
) as executor:
    for index, (status, detail) in enumerate(
        executor.map(patch_one, target_files),
        1,
    ):
        counts[status] += 1

        if status == "error":
            errors.append(detail)

        if (
            index % 1000 == 0
            or index == len(target_files)
        ):
            print(
                f"  Processed {index}/{len(target_files)}: "
                f"{dict(counts)}",
                flush=True,
            )

if errors:
    raise RuntimeError(
        "\n".join(errors[:20])
    )

valid = 0

for path in target_files:
    cache = load_cache(path)
    base_rows = int(
        as_numpy(
            cache["formula_peak_mzs"]
        ).shape[0]
    )

    mz_shape = tuple(
        as_numpy(
            cache["nl_formula_peak_mzs"]
        ).shape
    )

    prob_shape = tuple(
        as_numpy(
            cache["nl_formula_peak_probs"]
        ).shape
    )

    if (
        mz_shape == (base_rows, width)
        and prob_shape == mz_shape
    ):
        valid += 1

if valid != len(target_files):
    raise RuntimeError(
        f"Unexpected number of complete caches after enrichment: "
        f"{valid}/{len(target_files)}"
    )

audit = {
    "status": "OK",
    "reference_cache_files": len(reference_files),
    "reference_rows": len(reference_rows),
    "reference_unique_formulas": len(unique_formulas),
    "target_cache_files": len(target_files),
    "target_augmented_valid": valid,
    "nl_width": width,
    "losses": [
        {
            "name": name,
            "composition": loss,
            "mass": mass,
            "mask_probability": mask_probability[name],
        }
        for name, loss, mass in LOSSES
    ],
    "operation_counts": dict(counts),
}

AUDIT_PATH.write_text(
    json.dumps(
        audit,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

print(
    f"[5/5] MOLECULAR_RETRIEVAL_NL_CACHE_READY："
    f"{valid}/{len(target_files)}",
    flush=True,
)

print(
    f"Audit file: {AUDIT_PATH}",
    flush=True,
)
