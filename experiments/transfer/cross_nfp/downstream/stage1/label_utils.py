from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


PROBLEM2_LABELS = [
    "L_gradB",
    "aspect_ratio",
    "abs_edge_iota_over_nfp",
    "log10_qi",
    "edge_magnetic_mirror_ratio",
    "max_elongation",
]

CONSTRAINT_LABEL = "max_normalized_violation"

DERIVED_PROBLEM2_COLUMNS = {
    "aspect_ratio_violation",
    "iota_violation",
    "log10_qi_violation",
    "mirror_violation",
    "elongation_violation",
    CONSTRAINT_LABEL,
    "positive_max_normalized_violation",
    "feasible_under_problem_2",
    "score_problem_2",
}

METADATA_COLUMNS = {
    "sample_id",
    "boundary_json_path",
    "boundary.json",
    "method",
    "is_relaxed55",
}

EXCLUDED_AUX_PREFIXES = (
    "x_",
    "boundary.",
    "desc_",
    "metrics.",
    "misc.",
    "error",
    "omnigenous_field_and_targets.",
    "target.",
    "target_",
)


def unique_existing(columns: list[str], available: pd.Index | list[str]) -> list[str]:
    available_set = set(available)
    seen: set[str] = set()
    result = []
    for column in columns:
        if column in available_set and column not in seen:
            result.append(column)
            seen.add(column)
    return result


def is_auxiliary_metric_column(column: str, frame: pd.DataFrame) -> bool:
    if column in PROBLEM2_LABELS or column in DERIVED_PROBLEM2_COLUMNS or column in METADATA_COLUMNS:
        return False
    if column.startswith(EXCLUDED_AUX_PREFIXES):
        return False
    return bool(pd.api.types.is_numeric_dtype(frame[column]))


def discover_auxiliary_labels(
    frame: pd.DataFrame,
    min_finite_fraction: float = 0.98,
) -> tuple[list[str], dict[str, Any]]:
    candidates = [
        column
        for column in frame.columns
        if is_auxiliary_metric_column(str(column), frame)
    ]
    selected = []
    excluded = {}
    for column in candidates:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
        finite_fraction = float(np.isfinite(values).mean()) if values.size else 0.0
        if finite_fraction >= min_finite_fraction:
            selected.append(str(column))
        else:
            excluded[str(column)] = {
                "finite_fraction": finite_fraction,
                "reason": "below_min_finite_fraction",
            }
    return sorted(selected), {
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "min_finite_fraction": min_finite_fraction,
        "excluded": excluded,
    }


def regression_labels_from_frame(
    frame: pd.DataFrame,
    label_mode: str = "default_all",
    min_finite_fraction: float = 0.98,
) -> tuple[list[str], list[str], dict[str, Any]]:
    problem2 = unique_existing(PROBLEM2_LABELS, frame.columns)
    if label_mode == "problem2":
        return problem2, [], {
            "label_mode": label_mode,
            "auxiliary": {
                "candidate_count": 0,
                "selected_count": 0,
                "min_finite_fraction": min_finite_fraction,
                "excluded": {},
            },
        }
    if label_mode != "default_all":
        raise ValueError(f"Unknown label_mode={label_mode!r}; expected 'problem2' or 'default_all'.")
    auxiliary, aux_report = discover_auxiliary_labels(frame, min_finite_fraction=min_finite_fraction)
    labels = problem2 + [label for label in auxiliary if label not in problem2]
    return labels, auxiliary, {
        "label_mode": label_mode,
        "auxiliary": aux_report,
    }


def regression_label_weights(labels: list[str], surrogate_config: dict[str, Any]) -> np.ndarray:
    problem2_weight = float(surrogate_config.get("problem2_label_weight", 1.0))
    auxiliary_weight = float(surrogate_config.get("auxiliary_label_weight", 0.25))
    weights = np.full(len(labels), auxiliary_weight, dtype=np.float32)
    problem2_set = set(PROBLEM2_LABELS)
    for idx, label in enumerate(labels):
        if label in problem2_set:
            weights[idx] = problem2_weight
    for label, value in surrogate_config.get("label_weights", {}).items():
        if label in labels:
            weights[labels.index(label)] = float(value)
    weights = np.where(weights <= 0.0, 1.0, weights).astype(np.float32)
    return weights


def label_to_index(labels: list[str]) -> dict[str, int]:
    return {label: idx for idx, label in enumerate(labels)}
