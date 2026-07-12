from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from candidate_utils import (
    OUTPUT_DIR,
    constraint_penalty,
    constraint_support_metrics,
    dedupe_rank,
    feature_columns,
    metric_dict,
    predict_ensemble,
    prediction_row,
    prediction_value,
    surface_json_from_x,
    train_bounds,
    uncertainty_dict,
    write_boundary,
    write_jsonl,
)
from candidate_utils import load_model_bundle
from common import apply_thread_environment, ensure_output_dirs, load_config, parse_args, write_json


def load_relaxed55() -> pd.DataFrame:
    path = OUTPUT_DIR / "dataset" / "relaxed55.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return pd.DataFrame(records)


def make_record(
    method_id: str,
    source: str,
    seed: int,
    x: np.ndarray,
    boundary_json: str,
    prediction: dict[str, np.ndarray],
    support: dict[str, float],
    score: float,
    boundary_dir: Path,
) -> dict:
    candidate_id, boundary_path = write_boundary(boundary_json, boundary_dir)
    return {
        "run_id": f"{method_id}_seed{seed}",
        "method_id": method_id,
        "candidate_id": candidate_id,
        "source": source,
        "seed": seed,
        "boundary_json_path": boundary_path,
        "predicted_metrics": metric_dict(prediction),
        "predicted_uncertainty": uncertainty_dict(prediction),
        "support_metrics": support,
        "candidate_score": float(score),
        "rank_before_audit": 0,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    hardware = apply_thread_environment(config)
    ensure_output_dirs()
    seed = int(config.get("seed", 0))
    rng = np.random.default_rng(seed)
    per_method_pool = int(config["candidates"]["per_method_pool"])
    boundary_dir = OUTPUT_DIR / "candidates" / "boundaries"

    train = pd.read_parquet(OUTPUT_DIR / "dataset" / "train.parquet")
    validation = pd.read_parquet(OUTPUT_DIR / "dataset" / "validation.parquet")
    relaxed = load_relaxed55()
    x_cols = feature_columns(train)
    bundle = load_model_bundle("cuda:0")

    lower, upper = train_bounds(train, x_cols)

    e0_records = []
    e0_source = pd.concat([train, relaxed], axis=0, ignore_index=True, sort=False)
    e0_source = e0_source.sort_values(
        ["positive_max_normalized_violation", "L_gradB"], ascending=[True, False]
    ).head(max(per_method_pool * 2, per_method_pool))
    for _, row in e0_source.iterrows():
        x = row[x_cols].to_numpy(dtype=np.float32)
        prediction = predict_ensemble(bundle, x)
        support = constraint_support_metrics(prediction)
        score = float(prediction_value(prediction, "L_gradB") - 10.0 * constraint_penalty(prediction))
        boundary_json = str(row["boundary.json"])
        e0_records.append(
            make_record(
                "E0",
                "dataset_static_or_relaxed55",
                seed,
                x,
                boundary_json,
                prediction,
                support,
                score,
                boundary_dir,
            )
        )
    e0_records = dedupe_rank(e0_records, "candidate_score", per_method_pool)
    write_jsonl(OUTPUT_DIR / "candidates" / "e0_dataset_static.jsonl", e0_records)

    relaxed_x = relaxed[x_cols].to_numpy(dtype=np.float32)
    n_components = min(10, relaxed_x.shape[1], max(2, len(relaxed) - 1))
    pca = PCA(n_components=n_components, random_state=seed)
    relaxed_z = pca.fit_transform(relaxed_x)
    n_gmm = min(5, max(2, len(relaxed) // 12))
    gmm = GaussianMixture(n_components=n_gmm, random_state=seed)
    gmm.fit(relaxed_z)
    z_samples, _ = gmm.sample(int(config["candidates"]["relaxed_gmm_samples"]))
    x_samples = pca.inverse_transform(z_samples).astype(np.float32)
    x_samples = np.clip(x_samples, lower, upper)
    # Keep original relaxed seeds in the same local candidate pool.
    x_samples = np.vstack([relaxed_x, x_samples])

    predictions = predict_ensemble(bundle, x_samples, batch_size=int(config["candidates"]["batch_size"]))
    e1_records = []
    for idx, x in enumerate(x_samples):
        row_prediction = prediction_row(predictions, idx)
        support = constraint_support_metrics(row_prediction)
        score = float(
            prediction_value(row_prediction, "L_gradB")
            - 2.0 * prediction_value(row_prediction, "L_gradB", key="std")
            - 10.0 * constraint_penalty(row_prediction)
        )
        boundary_json = (
            str(relaxed.iloc[idx]["boundary.json"]) if idx < len(relaxed) else surface_json_from_x(x)
        )
        e1_records.append(
            make_record(
                "E1",
                "relaxed55_seed" if idx < len(relaxed) else "relaxed55_gmm",
                seed,
                x,
                boundary_json,
                row_prediction,
                support,
                score,
                boundary_dir,
            )
        )
    e1_records = dedupe_rank(e1_records, "candidate_score", per_method_pool)
    write_jsonl(OUTPUT_DIR / "candidates" / "e1_relaxed55_gmm.jsonl", e1_records)

    write_json(
        OUTPUT_DIR / "run_summary" / "candidate_generation_e0_e1.json",
        {
            "hardware_config": hardware,
            "e0_count": len(e0_records),
            "e1_count": len(e1_records),
            "relaxed55_count": int(len(relaxed)),
            "gmm_components": int(n_gmm),
            "pca_components": int(n_components),
        },
    )
    print(f"Wrote E0={len(e0_records)} E1={len(e1_records)} candidates")


if __name__ == "__main__":
    main()
