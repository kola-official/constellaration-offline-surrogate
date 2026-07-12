from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import load_dataset

from constellaration.generative_model import bootstrap_dataset as bd


REPO = "proxima-fusion/constellaration"
DATA_DIR = Path("experiments/hf_config_screening/hf_mirror_data")
OUT_DIR = Path("experiments/hf_config_screening/outputs")


def as_builtin(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if math.isnan(float(value)):
            return None
        return float(value)
    if hasattr(value, "item"):
        return value.item()
    return value


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


def unflatten_finite_beta(df: pd.DataFrame) -> pd.DataFrame:
    rename = {
        col: col.removeprefix("metrics.")
        for col in df.columns
        if col.startswith("metrics.")
    }
    out = df.rename(columns=rename).copy()
    return add_problem2_columns(out)


def compact(df: pd.DataFrame, cols: list[str], n: int) -> pd.DataFrame:
    keep = [col for col in cols if col in df.columns]
    return df[keep].head(n)


def main() -> None:
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_dataset(REPO, "default", split="train").to_pandas()
    raw_default_ids = set(raw["plasma_config_id"].dropna().astype(str))

    default = bd.load_source_datasets_with_no_errors()
    no_error_default_ids = set(default["plasma_config_id"].dropna().astype(str))
    default_nfp3 = default[default["boundary.n_field_periods"] == 3].copy()
    no_error_nfp3_ids = set(default_nfp3["plasma_config_id"].dropna().astype(str))
    default_scored = bd._unflatten_metrics_and_concatenate(default_nfp3)
    default_scored = add_problem2_columns(default_scored)
    default_scored["source_plasma_config_id"] = default_scored["plasma_config_id"].astype(str)

    source_cols = [
        "source_plasma_config_id",
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
    default_source = default_scored[source_cols].rename(
        columns={
            "plasma_config_id": "default_plasma_config_id",
            "L_gradB": "default_L_gradB",
            "positive_max_normalized_violation": "default_positive_max_normalized_violation",
            "max_normalized_violation": "default_max_normalized_violation",
            "score_problem2": "default_score_problem2",
            "aspect_ratio": "default_aspect_ratio",
            "abs_edge_iota_over_nfp": "default_abs_edge_iota_over_nfp",
            "log10_qi": "default_log10_qi",
            "edge_magnetic_mirror_ratio": "default_edge_magnetic_mirror_ratio",
            "max_elongation": "default_max_elongation",
            "method": "default_method",
            "boundary.json": "default_boundary_json",
        }
    )

    summary_records: list[dict[str, Any]] = []
    joined_rows: list[pd.DataFrame] = []
    finite_problemlike_rows: list[pd.DataFrame] = []

    for path in sorted(DATA_DIR.glob("finite_beta_*/*.parquet")):
        config = path.parent.name
        df_raw = pd.read_parquet(path)
        df = unflatten_finite_beta(df_raw)
        df["config"] = config
        df["source_plasma_config_id"] = df["misc.source_plasma_config_id"].astype(str)
        error_col = "misc.has_neurips_2025_forward_model_error"
        ok = df[~df[error_col].fillna(False)].copy() if error_col in df else df.copy()
        source_ids = set(ok["source_plasma_config_id"].dropna().astype(str))

        joined = ok.merge(default_source, on="source_plasma_config_id", how="left")
        joined_rows.append(joined)
        finite_problemlike_rows.append(ok)

        official_joined = joined[joined["default_positive_max_normalized_violation"].notna()]
        best_official = (
            official_joined.sort_values(
                ["default_positive_max_normalized_violation", "default_L_gradB"],
                ascending=[True, False],
            )
            .head(1)
            .to_dict(orient="records")
        )
        best_finite = (
            ok.sort_values(
                ["positive_max_normalized_violation", "L_gradB"],
                ascending=[True, False],
            )
            .head(1)
            .to_dict(orient="records")
        )
        summary_records.append(
            {
                "config": config,
                "rows": int(len(df)),
                "no_error_rows": int(len(ok)),
                "unique_source_ids": int(len(source_ids)),
                "source_ids_in_default_raw": int(len(source_ids & raw_default_ids)),
                "source_ids_in_default_no_error": int(len(source_ids & no_error_default_ids)),
                "source_ids_in_default_no_error_nfp3": int(len(source_ids & no_error_nfp3_ids)),
                "joined_default_no_error_nfp3_rows": int(len(official_joined)),
                "joined_default_official_feasible_rows": int(
                    (official_joined["default_score_problem2"] > 0).sum()
                ),
                "joined_default_positive_violation_lte_0p05": int(
                    (official_joined["default_positive_max_normalized_violation"] <= 0.05).sum()
                ),
                "joined_default_score_0p421_to_0p441": int(
                    official_joined["default_score_problem2"].between(0.421, 0.441).sum()
                ),
                "joined_default_L_gradB_8p41_to_8p81": int(
                    official_joined["default_L_gradB"].between(8.41, 8.81).sum()
                ),
                "finite_beta_problemlike_feasible_rows": int(ok["feasible_problem2"].sum()),
                "finite_beta_problemlike_positive_violation_lte_0p05": int(
                    (ok["positive_max_normalized_violation"] <= 0.05).sum()
                ),
                "best_joined_default": best_official[0] if best_official else {},
                "best_finite_beta_problemlike": best_finite[0] if best_finite else {},
            }
        )

    summary = pd.DataFrame(summary_records)
    joined_all = pd.concat(joined_rows, ignore_index=True) if joined_rows else pd.DataFrame()
    finite_all = (
        pd.concat(finite_problemlike_rows, ignore_index=True)
        if finite_problemlike_rows
        else pd.DataFrame()
    )

    summary.to_csv(OUT_DIR / "finite_beta_summary.csv", index=False)
    if not joined_all.empty:
        top_joined = joined_all.sort_values(
            ["default_positive_max_normalized_violation", "default_L_gradB"],
            ascending=[True, False],
        )
        compact(
            top_joined,
            [
                "config",
                "source_plasma_config_id",
                "plasma_config_id",
                "default_L_gradB",
                "default_positive_max_normalized_violation",
                "default_max_normalized_violation",
                "default_score_problem2",
                "default_aspect_ratio",
                "default_abs_edge_iota_over_nfp",
                "default_log10_qi",
                "default_edge_magnetic_mirror_ratio",
                "default_max_elongation",
                "L_gradB",
                "positive_max_normalized_violation",
                "score_problem2",
            ],
            200,
        ).to_csv(OUT_DIR / "finite_beta_top_joined_default.csv", index=False)
    if not finite_all.empty:
        top_finite = finite_all.sort_values(
            ["positive_max_normalized_violation", "L_gradB"],
            ascending=[True, False],
        )
        compact(
            top_finite,
            [
                "config",
                "source_plasma_config_id",
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
            ],
            200,
        ).to_csv(OUT_DIR / "finite_beta_problemlike_top.csv", index=False)

    payload = {
        "elapsed_seconds": round(time.time() - started, 3),
        "default_raw_ids": len(raw_default_ids),
        "default_no_error_ids": len(no_error_default_ids),
        "default_no_error_nfp3_ids": len(no_error_nfp3_ids),
        "summary": summary_records,
        "outputs": {
            "finite_beta_summary_csv": str(OUT_DIR / "finite_beta_summary.csv"),
            "finite_beta_top_joined_default_csv": str(
                OUT_DIR / "finite_beta_top_joined_default.csv"
            ),
            "finite_beta_problemlike_top_csv": str(
                OUT_DIR / "finite_beta_problemlike_top.csv"
            ),
        },
    }
    (OUT_DIR / "finite_beta_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=as_builtin),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True, default=as_builtin))


if __name__ == "__main__":
    main()
