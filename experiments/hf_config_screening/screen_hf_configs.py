from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import load_dataset

from constellaration.generative_model import bootstrap_dataset as bd


REPO = "proxima-fusion/constellaration"
CONFIGS = [
    "default",
    "finite_beta_1pct",
    "finite_beta_2pct",
    "finite_beta_3pct",
    "finite_beta_4pct",
    "finite_beta_5pct",
    "vmecpp_wout",
    "vmecpp_wout_finite_beta_1pct",
    "vmecpp_wout_finite_beta_2pct",
    "vmecpp_wout_finite_beta_3pct",
    "vmecpp_wout_finite_beta_4pct",
    "vmecpp_wout_finite_beta_5pct",
]

OUT_DIR = Path("experiments/hf_config_screening/outputs")
SUMMARY_JSON = OUT_DIR / "summary.json"
TOP_CSV = OUT_DIR / "default_top_candidates.csv"
NEAR_ALM_CSV = OUT_DIR / "default_near_alm_objective.csv"
CONFIG_CSV = OUT_DIR / "config_load_summary.csv"


def as_builtin(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if math.isnan(float(value)):
            return None
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=as_builtin),
        encoding="utf-8",
    )


def add_problem2_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["L_gradB"] = pd.to_numeric(
        df["minimum_normalized_magnetic_gradient_scale_length"], errors="coerce"
    )
    df["abs_edge_iota_over_nfp"] = pd.to_numeric(
        df["edge_rotational_transform_over_n_field_periods"], errors="coerce"
    ).abs()
    qi = pd.to_numeric(df["qi"], errors="coerce")
    df["log10_qi"] = np.log10(qi.where(qi > 0))
    df["aspect_ratio_violation"] = (
        pd.to_numeric(df["aspect_ratio"], errors="coerce") - 10.0
    ) / 10.0
    df["iota_violation"] = (0.25 - df["abs_edge_iota_over_nfp"]) / 0.25
    df["log10_qi_violation"] = (df["log10_qi"] + 4.0) / 4.0
    df["mirror_violation"] = (
        pd.to_numeric(df["edge_magnetic_mirror_ratio"], errors="coerce") - 0.2
    ) / 0.2
    df["elongation_violation"] = (
        pd.to_numeric(df["max_elongation"], errors="coerce") - 5.0
    ) / 5.0
    violation_cols = [
        "aspect_ratio_violation",
        "iota_violation",
        "log10_qi_violation",
        "mirror_violation",
        "elongation_violation",
    ]
    df["max_normalized_violation"] = df[violation_cols].max(axis=1)
    df["positive_max_normalized_violation"] = df[violation_cols].clip(lower=0).max(axis=1)
    df["feasible_problem2"] = df["max_normalized_violation"] <= 1e-2
    df["score_problem2"] = np.where(df["feasible_problem2"], df["L_gradB"] / 20.0, 0.0)
    return df


def compact_candidates(df: pd.DataFrame, n: int = 40) -> pd.DataFrame:
    cols = [
        "plasma_config_id",
        "L_gradB",
        "positive_max_normalized_violation",
        "max_normalized_violation",
        "score_problem2",
        "aspect_ratio",
        "abs_edge_iota_over_nfp",
        "log10_qi",
        "edge_magnetic_mirror_ratio",
        "max_elongation",
        "method",
        "boundary.json",
    ]
    keep = [col for col in cols if col in df.columns]
    return df.sort_values(
        ["positive_max_normalized_violation", "L_gradB"], ascending=[True, False]
    )[keep].head(n)


def load_default_no_error_nfp3() -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = load_dataset(REPO, "default", split="train")
    raw_rows = len(raw)
    df = bd.load_source_datasets_with_no_errors()
    no_error_rows = int(len(df))
    df = df[df["boundary.n_field_periods"] == 3].copy()
    nfp3_rows = int(len(df))
    df = bd._unflatten_metrics_and_concatenate(df)
    df = add_problem2_columns(df)
    meta = {
        "raw_rows": raw_rows,
        "no_error_rows": no_error_rows,
        "no_error_nfp3_rows": nfp3_rows,
        "columns": list(df.columns),
    }
    return df, meta


def try_load_config(config: str) -> dict[str, Any]:
    started = time.time()
    record: dict[str, Any] = {"config": config, "status": "pending"}
    try:
        ds = load_dataset(REPO, config, split="train")
        record.update(
            {
                "status": "loaded",
                "rows": len(ds),
                "seconds": round(time.time() - started, 3),
                "columns": list(ds.features.keys()),
            }
        )
        df = ds.to_pandas()
        record["has_boundary_json"] = "boundary.json" in df.columns
        record["has_source_plasma_config_id"] = (
            "misc.source_plasma_config_id" in df.columns
        )
        if "plasma_config_id" in df.columns:
            record["unique_plasma_config_id"] = int(df["plasma_config_id"].nunique())
        if "misc.source_plasma_config_id" in df.columns:
            record["unique_source_plasma_config_id"] = int(
                df["misc.source_plasma_config_id"].nunique()
            )
        return record
    except Exception as exc:
        record.update(
            {
                "status": "failed",
                "seconds": round(time.time() - started, 3),
                "error_type": type(exc).__name__,
                "error": str(exc)[:1200],
            }
        )
        return record


def main() -> None:
    os.environ.setdefault("HF_DATASETS_OFFLINE", "0")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    started = time.time()
    default_df, default_meta = load_default_no_error_nfp3()
    top_default = compact_candidates(default_df, n=80)
    top_default.to_csv(TOP_CSV, index=False)

    near_alm = default_df[
        default_df["L_gradB"].between(8.41, 8.81)
        | default_df["score_problem2"].between(0.421, 0.441)
        | default_df["positive_max_normalized_violation"].between(0.0, 0.05)
    ].copy()
    near_alm = near_alm.sort_values(
        ["positive_max_normalized_violation", "L_gradB"], ascending=[True, False]
    )
    compact_candidates(near_alm, n=200).to_csv(NEAR_ALM_CSV, index=False)

    config_records = [try_load_config(config) for config in CONFIGS]
    config_df = pd.DataFrame(config_records)
    config_df.to_csv(CONFIG_CSV, index=False)

    default_ids = set(default_df["plasma_config_id"]) if "plasma_config_id" in default_df else set()
    finite_beta_join_notes: list[dict[str, Any]] = []
    for record in config_records:
        if record["status"] != "loaded" or not record.get("has_source_plasma_config_id"):
            continue
        ds = load_dataset(REPO, record["config"], split="train")
        df = ds.to_pandas()
        src = set(df["misc.source_plasma_config_id"].dropna().astype(str))
        finite_beta_join_notes.append(
            {
                "config": record["config"],
                "source_rows": int(df["misc.source_plasma_config_id"].notna().sum()),
                "unique_sources": int(len(src)),
                "sources_found_in_default_no_error_nfp3": int(len(src & default_ids)),
                "sources_missing_from_default_no_error_nfp3": int(len(src - default_ids)),
            }
        )

    feasible = default_df[default_df["feasible_problem2"]]
    best_row = (
        top_default.head(1).to_dict(orient="records")[0] if not top_default.empty else {}
    )
    summary = {
        "repo": REPO,
        "elapsed_seconds": round(time.time() - started, 3),
        "default": {
            **default_meta,
            "official_feasible_rows": int(len(feasible)),
            "best_by_positive_max_violation": best_row,
            "alm_like_rows_positive_violation_lte_0p05": int(
                (default_df["positive_max_normalized_violation"] <= 0.05).sum()
            ),
            "alm_like_rows_score_0p421_to_0p441": int(
                default_df["score_problem2"].between(0.421, 0.441).sum()
            ),
            "alm_like_rows_L_gradB_8p41_to_8p81": int(
                default_df["L_gradB"].between(8.41, 8.81).sum()
            ),
        },
        "configs": config_records,
        "finite_beta_join_notes": finite_beta_join_notes,
        "outputs": {
            "summary_json": str(SUMMARY_JSON),
            "config_csv": str(CONFIG_CSV),
            "top_candidates_csv": str(TOP_CSV),
            "near_alm_csv": str(NEAR_ALM_CSV),
        },
    }
    write_json(SUMMARY_JSON, summary)
    print(json.dumps(summary, indent=2, sort_keys=True, default=as_builtin))


if __name__ == "__main__":
    main()
