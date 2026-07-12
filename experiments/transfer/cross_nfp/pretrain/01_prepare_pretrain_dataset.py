from __future__ import annotations

import json
from typing import Any

import pandas as pd
from constellaration.generative_model import bootstrap_dataset as bd
from sklearn.model_selection import train_test_split

from common_cross_nfp import (
    DEFAULT_15_LABELS,
    OUTPUT_DIR,
    apply_cli_overrides,
    add_problem2_columns,
    apply_thread_environment,
    ensure_output_dirs,
    load_config,
    parse_args,
    resolve_path,
    sha1_text,
    write_json,
)


def make_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    print("Unserializing non-Nfp3 surfaces for Fourier features...", flush=True)
    df_surface = bd._unserialize_surface(df.copy())
    x = bd._to_X(
        df_surface,
        max_poloidal_mode=bd.MAX_POLOIDAL_MODE,
        max_toroidal_mode=bd.MAX_TOROIDAL_MODE,
    )
    feature_cols = [f"x_{idx:03d}" for idx in range(x.shape[1])]
    features = pd.DataFrame(x, columns=feature_cols, index=df.index)
    keep_cols = [
        "sample_id",
        "boundary.json",
        "method",
        "nfp",
        "boundary.n_field_periods",
        "feasible_under_problem_2",
        "max_normalized_violation",
        "positive_max_normalized_violation",
    ]
    keep_cols.extend(DEFAULT_15_LABELS)
    keep_cols = [column for column in keep_cols if column in df.columns]
    return pd.concat([df[keep_cols].reset_index(drop=True), features.reset_index(drop=True)], axis=1)


def split_pretrain(frame: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    seed = int(config.get("seed", 0))
    validation_size = float(config.get("data", {}).get("pretrain_validation_size", 0.10))
    notes: list[str] = []
    stratify = frame["nfp"].astype(str) if "nfp" in frame.columns else None
    try:
        train, validation = train_test_split(
            frame,
            test_size=validation_size,
            random_state=seed,
            stratify=stratify,
        )
    except ValueError as exc:
        notes.append(f"pretrain_split_unstratified_fallback: {exc}")
        train, validation = train_test_split(
            frame,
            test_size=validation_size,
            random_state=seed,
            stratify=None,
        )
    return train.reset_index(drop=True), validation.reset_index(drop=True), notes


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config = apply_cli_overrides(config, args)
    hardware = apply_thread_environment(config)
    ensure_output_dirs()

    seed = int(config.get("seed", 0))
    data_config = config.get("data", {})
    output_dir = resolve_path(data_config.get("pretrain_dataset_dir"), OUTPUT_DIR / "dataset_pretrain")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading all no-error source rows from constellaration helper...", flush=True)
    source = bd.load_source_datasets_with_no_errors()
    source_rows = int(len(source))
    if "boundary.n_field_periods" not in source.columns:
        raise KeyError("Expected source column boundary.n_field_periods.")

    source["nfp"] = pd.to_numeric(source["boundary.n_field_periods"], errors="coerce")
    nfp3_value = int(data_config.get("nfp3_value", 3))
    pretrain = source[source["nfp"].notna() & (source["nfp"] != nfp3_value)].copy()
    pre_filter_rows = int(len(pretrain))
    max_rows = args.max_rows or data_config.get("pretrain_max_rows")
    if max_rows:
        pretrain = pretrain.sample(n=min(int(max_rows), len(pretrain)), random_state=seed).copy()

    nfp_raw = pretrain["nfp"].to_numpy()
    pretrain = bd._unflatten_metrics_and_concatenate(pretrain)
    if "boundary.n_field_periods" not in pretrain.columns:
        pretrain["boundary.n_field_periods"] = nfp_raw
    pretrain = add_problem2_columns(pretrain)
    pretrain["sample_id"] = [sha1_text(value) for value in pretrain["boundary.json"]]
    pretrain["nfp"] = pd.to_numeric(pretrain["boundary.n_field_periods"], errors="coerce").astype("Int64")

    feature_frame = make_feature_frame(pretrain)
    required = DEFAULT_15_LABELS + ["max_normalized_violation"]
    missing = [column for column in required if column not in feature_frame.columns]
    if missing:
        raise KeyError(f"Pretrain frame is missing required labels: {missing}")
    before_finite = int(len(feature_frame))
    mask = feature_frame[required].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
    feature_frame = feature_frame.loc[mask].reset_index(drop=True)

    train, validation, split_notes = split_pretrain(feature_frame, config)
    train.to_parquet(output_dir / "train.parquet", index=False)
    validation.to_parquet(output_dir / "validation.parquet", index=False)

    manifest = {
        "source_no_error_rows": source_rows,
        "excluded_nfp_value": nfp3_value,
        "non_nfp3_rows_before_max_rows": pre_filter_rows,
        "rows_before_finite_filter": before_finite,
        "rows_after_finite_filter": int(len(feature_frame)),
        "split": {
            "train": int(len(train)),
            "validation": int(len(validation)),
        },
        "nfp_counts": {
            str(key): int(value)
            for key, value in feature_frame["nfp"].value_counts().sort_index().items()
        },
        "labels": DEFAULT_15_LABELS,
        "split_notes": split_notes,
        "hardware_config": hardware,
        "outputs": {
            "train": str(output_dir / "train.parquet"),
            "validation": str(output_dir / "validation.parquet"),
        },
        "scaler_policy": "Geometry x scaler is not computed here; training uses the Nfp=3 train split scaler.",
    }
    write_json(output_dir / "pretrain_manifest.json", manifest)
    write_json(OUTPUT_DIR / "run_summary" / "pretrain_dataset_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
