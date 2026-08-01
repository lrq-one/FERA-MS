"""Shared, locked feature contract for the paper LightGBM reranker."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


EXPLICIT_FEATURE_NAMES = (
    "mz_fraction", "log_mz", "sqrt_mz", "log_probability",
    "probability", "log_scaled_probability", "relative_probability",
    "base_peak_ratio", "probability_rank", "log_probability_rank",
    "ace_scaled", "ace_low", "ace_mid", "ace_high", "ace_mz_interaction",
    "precursor_mz_fraction", "mz_over_precursor", "neutral_loss_over_precursor",
    "positive_neutral_loss_fraction", "log_positive_neutral_loss",
    "mz_mass_defect", "neutral_loss_mass_defect", "not_above_precursor",
    "ace_mz_over_precursor", "ace_neutral_loss_over_precursor",
    "fragment_mass_window",
)
LOCAL_DENSITY_FEATURE_NAMES = (
    "local_count_radius_1", "local_count_radius_3", "local_count_radius_5",
    "local_probability_radius_1", "local_probability_radius_3",
    "local_probability_radius_5", "probability_share_radius_1",
    "probability_share_radius_5",
)
INTERNAL_COMPONENT_SCHEMA = (
    ("pred_spec_formula_logprobs", 1),
    ("pred_spec_formula_comp_feats", 18),
    ("pred_spec_base_peak_logprobs", 1),
    ("pred_spec_peak_logprobs", 1),
    ("pred_spec_peak_channels", 1),
    ("pred_rendered_peak_gate_logits", 1),
    ("pred_rendered_peak_gate_delta", 1),
    ("pred_refiner_delta", 1),
    ("pred_refiner_delta_valid_mask", 1),
    ("formula_pred_formula_logprobs", 1),
    ("agg_joint_logsum_logprob", 1),
    ("agg_joint_mean_logprob", 1),
    ("agg_joint_h_counts", 1),
    ("agg_joint_abs_h_counts", 1),
    ("agg_joint_joint_refinement_feats", 8),
    ("agg_node_logprobs_by_joint", 1),
    ("agg_node_depths_by_joint", 1),
    ("agg_node_formula_logprobs_by_joint", 1),
)
CANONICAL_INTERNAL_KEY = "fragment_rich_features"
LEGACY_INTERNAL_KEYS = ("r173_frag_rich_feats", "candidate_reranker_frag_rich_feats")
SUPPORTED_INTERNAL_KEYS = (CANONICAL_INTERNAL_KEY, *LEGACY_INTERNAL_KEYS)
EXPLICIT_DIM = 26
LOCAL_DENSITY_DIM = 8
INTERNAL_DIM = 42
TOTAL_FEATURE_DIM = 76
IDENTITY_PATH = Path(__file__).resolve().parents[3] / "config/paper_experiment_identity.json"

assert len(EXPLICIT_FEATURE_NAMES) == EXPLICIT_DIM
assert len(LOCAL_DENSITY_FEATURE_NAMES) == LOCAL_DENSITY_DIM
assert sum(dim for _, dim in INTERNAL_COMPONENT_SCHEMA) == INTERNAL_DIM


def alias_rich_feature_keys(results, extra_schema):
    """Resolve historical result keys without changing saved artifact schemas."""
    requested = {str(key) for key, _ in extra_schema}
    source = next(
        (key for key in SUPPORTED_INTERNAL_KEYS if isinstance(results.get(key), torch.Tensor)),
        None,
    )
    if source is None:
        return results
    for key in requested:
        if key in SUPPORTED_INTERNAL_KEYS and key not in results:
            results[key] = results[source]
    if CANONICAL_INTERNAL_KEY not in results:
        results[CANONICAL_INTERNAL_KEY] = results[source]
    return results


def validate_extra_schema(extra_schema):
    normalized = [(str(key), int(dim)) for key, dim in extra_schema]
    if len(normalized) != 1:
        raise RuntimeError(f"Paper reranker requires one 42D internal feature block: {normalized}")
    key, dim = normalized[0]
    if key not in SUPPORTED_INTERNAL_KEYS or dim != INTERNAL_DIM:
        raise RuntimeError(
            "Paper reranker internal schema mismatch: "
            f"{normalized}; expected a supported alias with dimension {INTERNAL_DIM}"
        )
    return normalized


def require_internal_feature_tensor(results, extra_schema, *, row_count, device):
    """Return the required 42D block or fail; formal features are never zero-filled."""
    normalized = validate_extra_schema(extra_schema)
    alias_rich_feature_keys(results, normalized)
    required_key = normalized[0][0]
    if required_key not in results:
        raise RuntimeError(f"Required 42D internal feature block is missing: {required_key}")
    internal = results[required_key]
    if (
        not isinstance(internal, torch.Tensor)
        or internal.ndim != 2
        or int(internal.shape[0]) != int(row_count)
        or int(internal.shape[1]) != INTERNAL_DIM
    ):
        shape = None if not isinstance(internal, torch.Tensor) else tuple(internal.shape)
        raise RuntimeError(f"Invalid internal feature tensor: {shape}")
    if internal.device != device:
        internal = internal.to(device)
    return internal.float()


def feature_schema_payload(extra_schema):
    validate_extra_schema(extra_schema)
    columns = list(EXPLICIT_FEATURE_NAMES) + list(LOCAL_DENSITY_FEATURE_NAMES)
    for name, dim in INTERNAL_COMPONENT_SCHEMA:
        columns.extend(name if dim == 1 else f"{name}[{index}]" for index in range(dim))
    payload = {
        "schema_version": 1,
        "groups": [
            {
                "name": "explicit",
                "source_module": "candidate_reranker.candidate_features",
                "dimension": EXPLICIT_DIM,
                "columns": list(EXPLICIT_FEATURE_NAMES),
            },
            {
                "name": "local_density",
                "source_module": "candidate_reranker.local_density_features",
                "dimension": LOCAL_DENSITY_DIM,
                "columns": list(LOCAL_DENSITY_FEATURE_NAMES),
            },
            {
                "name": "internal_backbone",
                "source_module": "candidate_reranker.attach_raw_rich_features",
                "dimension": INTERNAL_DIM,
                "canonical_result_key": CANONICAL_INTERNAL_KEY,
                "components": [
                    {"name": name, "dimension": dim} for name, dim in INTERNAL_COMPONENT_SCHEMA
                ],
            },
        ],
        "column_order": columns,
        "dimensions": {
            "explicit": EXPLICIT_DIM,
            "local_density": LOCAL_DENSITY_DIM,
            "internal_backbone": INTERNAL_DIM,
            "total": TOTAL_FEATURE_DIM,
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    payload["schema_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def write_feature_schema(out_dir, extra_schema):
    payload = feature_schema_payload(extra_schema)
    if IDENTITY_PATH.is_file():
        expected_sha = json.loads(IDENTITY_PATH.read_text(encoding="utf-8"))[
            "candidate_reranker"
        ]["schema_sha256"]
        if payload["schema_sha256"] != expected_sha:
            raise RuntimeError(
                "Candidate-reranker feature schema differs from the locked paper identity: "
                f"{payload['schema_sha256']} != {expected_sha}"
            )
    (Path(out_dir) / "feature_schema.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return payload
