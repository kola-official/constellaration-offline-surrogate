from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from common_stage3 import (
    OUTPUT_DIR,
    STAGE1_DIR,
    STAGE1_OUTPUT_DIR,
    TrustDistanceModel,
    apply_thread_environment,
    boundary_path_to_x,
    ensure_output_dirs,
    existing_candidate_records,
    feature_columns,
    load_config,
    parse_args,
    write_json,
    write_jsonl,
)

sys.path.insert(0, str(STAGE1_DIR))
from candidate_utils import (  # noqa: E402
    constraint_support_metrics,
    metric_dict,
    predict_ensemble,
    surface_json_from_x,
    train_bounds,
    uncertainty_dict,
    write_boundary,
)


def load_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(STAGE1_OUTPUT_DIR / "dataset" / "train.parquet")
    validation = pd.read_parquet(STAGE1_OUTPUT_DIR / "dataset" / "validation.parquet")
    relaxed = pd.DataFrame(
        [
            json.loads(line)
            for line in (STAGE1_OUTPUT_DIR / "dataset" / "relaxed55.jsonl").read_text().splitlines()
            if line.strip()
        ]
    )
    return train, validation, relaxed


def prediction_record(
    x: np.ndarray,
    bundle: dict[str, Any],
    trust_model: TrustDistanceModel,
    boundary_dir: Path,
    source: str,
    seed: int,
    prior: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prediction = predict_ensemble(bundle, x, batch_size=1)
    support = constraint_support_metrics(prediction)
    trust = trust_model.evaluate(x)
    boundary_json = surface_json_from_x(x)
    candidate_id, boundary_path = write_boundary(boundary_json, boundary_dir)
    record = {
        "run_id": f"TR-stage3_seed{seed}",
        "method_id": "TR-stage3",
        "candidate_id": candidate_id,
        "source": source,
        "seed": seed,
        "boundary_json_path": boundary_path,
        "predicted_metrics": metric_dict(prediction),
        "predicted_uncertainty": uncertainty_dict(prediction),
        "support_metrics": support,
        "trust_metrics": trust,
        "candidate_score": 0.0,
        "rank_before_audit": 0,
    }
    if prior is not None:
        record["prior_stage"] = prior.get("stage")
        record["prior_method_id"] = prior.get("method_id")
        record["prior_source"] = prior.get("source")
        record["boundary_json_path"] = prior.get("boundary_json_path", boundary_path)
    return record


def trust_pass(record: dict[str, Any], cfg: dict[str, Any]) -> bool:
    trust = record["trust_metrics"]
    uncertainty = record["predicted_uncertainty"]
    support = record["support_metrics"]
    return bool(
        trust["train_distance_ratio"] <= 1.0
        and trust["relaxed_distance_ratio"] <= 1.0
        and uncertainty["log10_qi"] <= float(cfg["max_log10_qi_uncertainty"])
        and uncertainty["L_gradB"] <= float(cfg["max_l_gradb_uncertainty"])
        and support["predicted_max_normalized_violation_std"] <= float(cfg["max_mnv_uncertainty"])
    )


def trust_score(record: dict[str, Any], weights: dict[str, float]) -> float:
    support = record["support_metrics"]
    trust = record["trust_metrics"]
    uncertainty = record["predicted_uncertainty"]
    metrics = record["predicted_metrics"]
    loss = 0.0
    loss += float(weights["qi_violation"]) * float(support["predicted_qi_violation"])
    loss += float(weights["positive_violation"]) * float(support["predicted_positive_violation"])
    loss += float(weights["log10_qi_uncertainty"]) * float(uncertainty["log10_qi"])
    loss += float(weights["train_distance_ratio"]) * float(trust["train_distance_ratio"])
    loss += float(weights["relaxed_distance_ratio"]) * float(trust["relaxed_distance_ratio"])
    loss -= float(weights["l_gradb_bonus"]) * float(metrics["L_gradB"])
    return -float(loss)


def relaxed_local_samples(
    relaxed_x: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    sample_count: int,
    sigma_multiplier: float,
    components: int,
    seed: int,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    components = min(components, relaxed_x.shape[1], relaxed_x.shape[0] - 1)
    pca = PCA(n_components=components, random_state=seed)
    z_pool = pca.fit_transform(relaxed_x)
    nn = NearestNeighbors(n_neighbors=2)
    nn.fit(z_pool)
    dist, _ = nn.kneighbors(z_pool)
    sigma = max(float(np.median(dist[:, 1])) * sigma_multiplier, 1e-6)
    lower_z = np.quantile(z_pool, 0.0, axis=0)
    upper_z = np.quantile(z_pool, 1.0, axis=0)
    samples: list[np.ndarray] = []
    for _ in range(sample_count):
        anchor = z_pool[int(rng.integers(0, z_pool.shape[0]))]
        z = np.clip(anchor + rng.normal(0.0, sigma, size=components), lower_z, upper_z)
        x = pca.inverse_transform(z[None, :])[0]
        samples.append(np.clip(x, lower, upper).astype(np.float32))
    return samples


def rank_and_dedupe(records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for record in records:
        cid = record["candidate_id"]
        if cid not in best or record["candidate_score"] > best[cid]["candidate_score"]:
            best[cid] = record
    ranked = sorted(best.values(), key=lambda row: row["candidate_score"], reverse=True)
    for idx, record in enumerate(ranked[:limit]):
        record["rank_before_audit"] = idx + 1
    return ranked[:limit]


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    hardware = apply_thread_environment(config)
    ensure_output_dirs()
    seed = int(config.get("seed", 0))
    cfg = config["trust_region"]
    train, validation, relaxed = load_frames()
    x_cols = feature_columns(train)
    train_x = train[x_cols].to_numpy(dtype=np.float32)
    validation_x = validation[x_cols].to_numpy(dtype=np.float32)
    relaxed_x = relaxed[x_cols].to_numpy(dtype=np.float32)
    lower, upper = train_bounds(train, x_cols)
    trust_model = TrustDistanceModel(
        train_x,
        validation_x,
        relaxed_x,
        train_components=int(cfg["train_pca_components"]),
        relaxed_components=int(cfg["relaxed_pca_components"]),
        train_quantile=float(cfg["train_distance_quantile"]),
        relaxed_quantile=float(cfg["relaxed_distance_quantile"]),
        train_multiplier=float(cfg["train_distance_multiplier"]),
        relaxed_multiplier=float(cfg["relaxed_distance_multiplier"]),
    )
    from candidate_utils import load_model_bundle

    bundle = load_model_bundle("cuda:0")
    boundary_dir = OUTPUT_DIR / "candidates" / "boundaries"
    records: list[dict[str, Any]] = []
    attempted_by_source = Counter()
    rejected_by_source = Counter()

    for prior in existing_candidate_records():
        attempted_by_source[f"prior:{prior.get('method_id')}"] += 1
        try:
            x = boundary_path_to_x(prior["boundary_json_path"])
            record = prediction_record(
                x,
                bundle,
                trust_model,
                boundary_dir,
                source=f"prior_{prior.get('method_id')}",
                seed=seed,
                prior=prior,
            )
        except Exception:
            rejected_by_source[f"prior:{prior.get('method_id')}"] += 1
            continue
        if not trust_pass(record, cfg):
            rejected_by_source[f"prior:{prior.get('method_id')}"] += 1
            continue
        record["candidate_score"] = trust_score(record, cfg["rank_weights"])
        records.append(record)

    for idx, x in enumerate(relaxed_x):
        attempted_by_source["relaxed55_seed"] += 1
        record = prediction_record(
            x,
            bundle,
            trust_model,
            boundary_dir,
            source="relaxed55_seed",
            seed=seed,
        )
        record["relaxed55_index"] = idx
        if not trust_pass(record, cfg):
            rejected_by_source["relaxed55_seed"] += 1
            continue
        record["candidate_score"] = trust_score(record, cfg["rank_weights"])
        records.append(record)

    for idx, x in enumerate(
        relaxed_local_samples(
            relaxed_x,
            lower,
            upper,
            sample_count=int(cfg["local_samples"]),
            sigma_multiplier=float(cfg["local_sigma_multiplier"]),
            components=int(cfg["relaxed_pca_components"]),
            seed=seed,
        )
    ):
        attempted_by_source["relaxed55_local_sample"] += 1
        record = prediction_record(
            x,
            bundle,
            trust_model,
            boundary_dir,
            source="relaxed55_local_sample",
            seed=seed,
        )
        record["local_sample_index"] = idx
        if not trust_pass(record, cfg):
            rejected_by_source["relaxed55_local_sample"] += 1
            continue
        record["candidate_score"] = trust_score(record, cfg["rank_weights"])
        records.append(record)

    ranked = rank_and_dedupe(records, int(cfg["candidate_pool_limit"]))
    write_jsonl(OUTPUT_DIR / "candidates" / "trust_region_candidates.jsonl", ranked)
    write_json(
        OUTPUT_DIR / "run_summary" / "trust_region_generation_stage3.json",
        {
            "status": "complete",
            "hardware_config": hardware,
            "trust_distance_model": trust_model.summary(),
            "trust_config": cfg,
            "attempted_by_source": dict(attempted_by_source),
            "rejected_by_source": dict(rejected_by_source),
            "accepted_before_dedupe": len(records),
            "written": len(ranked),
            "top_sources": dict(Counter(row["source"] for row in ranked[:20])),
            "top_prior_methods": dict(Counter(row.get("prior_method_id", "new") for row in ranked[:20])),
        },
    )
    print(f"Wrote {len(ranked)} trust-region candidates")


if __name__ == "__main__":
    main()
