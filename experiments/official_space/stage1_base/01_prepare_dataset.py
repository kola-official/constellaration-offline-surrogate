from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from common import (
    OUTPUT_DIR,
    apply_thread_environment,
    ensure_output_dirs,
    git_info,
    load_config,
    parse_args,
    write_json,
)
from constellaration import problems
from constellaration.generative_model import bootstrap_dataset as bd
from label_utils import (
    DERIVED_PROBLEM2_COLUMNS,
    PROBLEM2_LABELS,
    regression_labels_from_frame,
    unique_existing,
)


OFFICIAL_THRESHOLDS = {
    "aspect_ratio": {"op": "<=", "value": 10.0},
    "abs_edge_iota_over_nfp": {"op": ">=", "value": 0.25},
    "log10_qi": {"op": "<=", "value": -4.0},
    "edge_magnetic_mirror_ratio": {"op": "<=", "value": 0.2},
    "max_elongation": {"op": "<=", "value": 5.0},
}

RELAXED_THRESHOLDS = {
    "aspect_ratio": {"op": "<=", "value": 10.0},
    "abs_edge_iota_over_nfp": {"op": ">=", "value": 0.1675},
    "log10_qi": {"op": "<=", "value": -2.68},
    "edge_magnetic_mirror_ratio": {"op": "<=", "value": 0.266},
    "max_elongation": {"op": "<=", "value": 6.65},
}


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def summarize_series(series: pd.Series) -> dict[str, float | int]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {"count": 0}
    return {
        "count": int(clean.shape[0]),
        "min": float(clean.min()),
        "p25": float(clean.quantile(0.25)),
        "median": float(clean.median()),
        "p75": float(clean.quantile(0.75)),
        "max": float(clean.max()),
    }


def add_problem2_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["L_gradB"] = df["minimum_normalized_magnetic_gradient_scale_length"]
    df["abs_edge_iota_over_nfp"] = df[
        "edge_rotational_transform_over_n_field_periods"
    ].abs()
    df["log10_qi"] = np.log10(pd.to_numeric(df["qi"], errors="coerce"))
    df["aspect_ratio_violation"] = (df["aspect_ratio"] - 10.0) / 10.0
    df["iota_violation"] = (0.25 - df["abs_edge_iota_over_nfp"]) / 0.25
    df["log10_qi_violation"] = (df["log10_qi"] - (-4.0)) / 4.0
    df["mirror_violation"] = (df["edge_magnetic_mirror_ratio"] - 0.2) / 0.2
    df["elongation_violation"] = (df["max_elongation"] - 5.0) / 5.0
    violation_cols = [
        "aspect_ratio_violation",
        "iota_violation",
        "log10_qi_violation",
        "mirror_violation",
        "elongation_violation",
    ]
    df["max_normalized_violation"] = df[violation_cols].max(axis=1)
    df["positive_max_normalized_violation"] = df[violation_cols].clip(lower=0.0).max(axis=1)
    df["feasible_under_problem_2"] = df["max_normalized_violation"] <= 1e-2
    df["score_problem_2"] = np.where(
        df["feasible_under_problem_2"], df["L_gradB"] / 20.0, 0.0
    )
    return df


def relaxed_mask(df: pd.DataFrame) -> pd.Series:
    return (
        (df["aspect_ratio"] <= 10.0)
        & (df["abs_edge_iota_over_nfp"] >= 0.1675)
        & (df["log10_qi"] <= -2.68)
        & (df["edge_magnetic_mirror_ratio"] <= 0.266)
        & (df["max_elongation"] <= 6.65)
    )


def make_feature_frame(
    df: pd.DataFrame,
    regression_labels: list[str],
) -> pd.DataFrame:
    print("Unserializing surfaces for Fourier features...")
    df_surface = bd._unserialize_surface(df.copy())
    X = bd._to_X(
        df_surface,
        max_poloidal_mode=bd.MAX_POLOIDAL_MODE,
        max_toroidal_mode=bd.MAX_TOROIDAL_MODE,
    )
    feature_cols = [f"x_{i:03d}" for i in range(X.shape[1])]
    features = pd.DataFrame(X, columns=feature_cols, index=df.index)
    keep_cols = [
        "sample_id",
        "boundary_json_path",
        "boundary.json",
        "method",
        "is_relaxed55",
    ]
    keep_cols.extend(PROBLEM2_LABELS)
    keep_cols.extend(sorted(DERIVED_PROBLEM2_COLUMNS))
    keep_cols.extend(regression_labels)
    keep_cols = unique_existing(keep_cols, df.columns)
    return pd.concat([df[keep_cols].reset_index(drop=True), features.reset_index(drop=True)], axis=1)


def build_strata(df: pd.DataFrame) -> pd.Series:
    method = df.get("method", pd.Series("unknown", index=df.index)).fillna("unknown").astype(str)
    try:
        obj_bin = pd.qcut(df["L_gradB"], q=5, labels=False, duplicates="drop").astype(str)
    except Exception:
        obj_bin = pd.Series("0", index=df.index)
    near_bin = (df["positive_max_normalized_violation"] <= df["positive_max_normalized_violation"].quantile(0.2)).astype(int).astype(str)
    strata = method + "_o" + obj_bin + "_n" + near_bin
    counts = strata.value_counts()
    return strata.where(strata.map(counts) >= 3, other="rare")


def split_dataframe(df: pd.DataFrame, seed: int, test_size: float, validation_size: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    main = df[~df["is_relaxed55"]].copy()
    strata = build_strata(main)
    notes: list[str] = []
    try:
        train_val, test = train_test_split(
            main,
            test_size=test_size,
            random_state=seed,
            stratify=strata,
        )
    except ValueError as exc:
        notes.append(f"first_split_unstratified_fallback: {exc}")
        train_val, test = train_test_split(
            main,
            test_size=test_size,
            random_state=seed,
            stratify=None,
        )
    val_fraction_of_train_val = validation_size / (1.0 - test_size)
    strata_tv = build_strata(train_val)
    try:
        train, validation = train_test_split(
            train_val,
            test_size=val_fraction_of_train_val,
            random_state=seed,
            stratify=strata_tv,
        )
    except ValueError as exc:
        notes.append(f"validation_split_unstratified_fallback: {exc}")
        train, validation = train_test_split(
            train_val,
            test_size=val_fraction_of_train_val,
            random_state=seed,
            stratify=None,
        )
    return train.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True), notes


def make_optimization_validation(validation: pd.DataFrame, max_rows: int | None, seed: int) -> pd.DataFrame:
    top_objective = validation["L_gradB"] >= validation["L_gradB"].quantile(0.8)
    near_feasible = validation["positive_max_normalized_violation"] <= validation[
        "positive_max_normalized_violation"
    ].quantile(0.2)
    opt = validation[top_objective | near_feasible].copy()
    if max_rows and len(opt) > max_rows:
        opt = opt.sample(n=max_rows, random_state=seed)
    return opt.reset_index(drop=True)


def top_record(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {}
    row = df.sort_values(["positive_max_normalized_violation", "L_gradB"], ascending=[True, False]).iloc[0]
    keys = [
        "sample_id",
        "method",
        "L_gradB",
        "aspect_ratio",
        "abs_edge_iota_over_nfp",
        "log10_qi",
        "edge_magnetic_mirror_ratio",
        "max_elongation",
        "positive_max_normalized_violation",
        "max_normalized_violation",
        "score_problem_2",
    ]
    return {key: (row[key].item() if hasattr(row[key], "item") else row[key]) for key in keys}


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    hardware = apply_thread_environment(config)
    ensure_output_dirs()
    seed = int(config.get("seed", 0))
    data_config = config.get("data", {})

    print("Loading HF default dataset for raw count...")
    raw = load_dataset("proxima-fusion/constellaration", "default")["train"]
    raw_count = len(raw)

    print("Loading no-error source dataset via upstream helper...")
    df = bd.load_source_datasets_with_no_errors()
    no_error_count = int(len(df))
    df = df[df["boundary.n_field_periods"] == 3].copy()
    nfp3_count = int(len(df))
    max_rows = data_config.get("max_rows")
    if max_rows:
        df = df.sample(n=min(int(max_rows), len(df)), random_state=seed).copy()

    df = bd._unflatten_metrics_and_concatenate(df)
    df = add_problem2_columns(df)
    df["sample_id"] = [sha1_text(x) for x in df["boundary.json"]]
    df["boundary_json_path"] = ""
    df["is_relaxed55"] = relaxed_mask(df)

    label_mode = str(data_config.get("label_mode", "default_all"))
    min_finite_fraction = float(data_config.get("auxiliary_label_min_finite_fraction", 0.98))
    regression_labels, auxiliary_labels, label_report = regression_labels_from_frame(
        df,
        label_mode=label_mode,
        min_finite_fraction=min_finite_fraction,
    )

    feature_df = make_feature_frame(df, regression_labels)

    boundaries_dir = OUTPUT_DIR / "dataset" / "boundaries"
    boundaries_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for sample_id, boundary_json in zip(feature_df["sample_id"], feature_df["boundary.json"], strict=False):
        path = boundaries_dir / f"{sample_id}.json"
        if not path.exists():
            path.write_text(boundary_json)
        paths.append(str(path))
    feature_df["boundary_json_path"] = paths

    train, validation, test, split_notes = split_dataframe(
        feature_df,
        seed=seed,
        test_size=float(data_config.get("test_size", 0.15)),
        validation_size=float(data_config.get("validation_size", 0.15)),
    )
    opt_val = make_optimization_validation(
        validation,
        max_rows=data_config.get("optimization_validation_max_rows"),
        seed=seed,
    )

    dataset_dir = OUTPUT_DIR / "dataset"
    train.to_parquet(dataset_dir / "train.parquet", index=False)
    validation.to_parquet(dataset_dir / "validation.parquet", index=False)
    test.to_parquet(dataset_dir / "test.parquet", index=False)
    opt_val.to_parquet(dataset_dir / "optimization_validation.parquet", index=False)

    relaxed = feature_df[feature_df["is_relaxed55"]].sort_values("L_gradB", ascending=False)
    relaxed_records = relaxed.to_dict(orient="records")
    with (dataset_dir / "relaxed55.jsonl").open("w") as handle:
        for record in relaxed_records:
            handle.write(
                json.dumps(
                    record,
                    default=lambda value: value.item()
                    if hasattr(value, "item")
                    else str(value),
                )
                + "\n"
            )

    method_coverage = (
        relaxed["method"].value_counts().to_dict() if "method" in relaxed.columns else {}
    )
    relaxed_manifest = {
        "source_file": "notebooks/generative_model_simple_QI.ipynb",
        "source_script": "experiments/official_space/stage1_base/01_prepare_dataset.py",
        "git": git_info(),
        "official_thresholds": OFFICIAL_THRESHOLDS,
        "relaxed_thresholds": RELAXED_THRESHOLDS,
        "relaxation_factor": 0.33,
        "paper_reported_relaxed_feasible_count": 52,
        "current_server_relaxed_feasible_count": int(len(relaxed)),
        "source_method_coverage": method_coverage,
        "note": "relaxed55 is not an official Problem 2 feasible set",
    }
    write_json(dataset_dir / "relaxed55_manifest.json", relaxed_manifest)

    gap = {
        "official_feasible_count": int(feature_df["feasible_under_problem_2"].sum()),
        "relaxed_feasible_count": int(len(relaxed)),
        "best_dataset_by_max_violation": top_record(feature_df),
        "best_relaxed55_by_max_violation": top_record(relaxed),
        "relaxed55_gap_to_official_thresholds": {
            "edge_iota_over_nfp_deficit": summarize_series(0.25 - relaxed["abs_edge_iota_over_nfp"]),
            "log10_qi_excess": summarize_series(relaxed["log10_qi"] - (-4.0)),
            "edge_magnetic_mirror_ratio_excess": summarize_series(relaxed["edge_magnetic_mirror_ratio"] - 0.2),
            "max_elongation_excess": summarize_series(relaxed["max_elongation"] - 5.0),
        },
        "note": "surrogate feasibility prediction is extrapolative because the offline default data contains no official Problem 2 feasible sample",
    }
    write_json(dataset_dir / "feasibility_gap.json", gap)

    split_manifest = {
        "seed": seed,
        "hardware_config": hardware,
        "split_notes": split_notes,
        "split": {
            "train": int(len(train)),
            "validation": int(len(validation)),
            "test": int(len(test)),
            "optimization_validation": int(len(opt_val)),
        },
        "relaxed55_excluded_from_train_validation_test": bool(
            train["is_relaxed55"].sum() == 0
            and validation["is_relaxed55"].sum() == 0
            and test["is_relaxed55"].sum() == 0
        ),
        "relaxed55_usage": "separate_seed_set_for_E1_E3_only",
        "feature_columns": [c for c in feature_df.columns if c.startswith("x_")],
        "problem2_labels": PROBLEM2_LABELS,
        "auxiliary_regression_labels": auxiliary_labels,
        "regression_labels": regression_labels,
        "label_report": label_report,
    }
    write_json(dataset_dir / "split_manifest.json", split_manifest)

    data_check = {
        "hf_default_train_rows": raw_count,
        "no_error_rows": no_error_count,
        "no_error_nfp3_rows": nfp3_count,
        "used_rows": int(len(feature_df)),
        "official_feasible_rows": int(feature_df["feasible_under_problem_2"].sum()),
        "relaxed_feasible_rows": int(len(relaxed)),
        "outputs": {
            "train": str(dataset_dir / "train.parquet"),
            "validation": str(dataset_dir / "validation.parquet"),
            "test": str(dataset_dir / "test.parquet"),
            "optimization_validation": str(dataset_dir / "optimization_validation.parquet"),
            "relaxed55": str(dataset_dir / "relaxed55.jsonl"),
        },
    }
    write_json(OUTPUT_DIR / "run_summary" / "data_check.json", data_check)
    print(json.dumps(data_check, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
