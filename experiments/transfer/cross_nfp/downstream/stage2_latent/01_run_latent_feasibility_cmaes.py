from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import nevergrad as ng
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from common_stage2 import (
    OUTPUT_DIR,
    STAGE1_DIR,
    STAGE1_OUTPUT_DIR,
    apply_thread_environment,
    ensure_output_dirs,
    load_config,
    parse_args,
    write_json,
)

sys.path.insert(0, str(STAGE1_DIR))
from candidate_utils import (  # noqa: E402
    SupportModel,
    constraint_support_metrics,
    feature_columns,
    load_model_bundle,
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


def fourier_mode_orders(x_dim: int) -> np.ndarray:
    half = x_dim // 2
    max_toroidal_mode = 4
    shape_cols = 2 * max_toroidal_mode + 1
    orders = []
    for flat_index in range(max_toroidal_mode + 1, max_toroidal_mode + 1 + half):
        poloidal_mode = flat_index // shape_cols
        toroidal_mode = abs((flat_index % shape_cols) - max_toroidal_mode)
        orders.append(float(np.sqrt(poloidal_mode**2 + toroidal_mode**2)))
    return np.array(orders + orders, dtype=np.float32)


class GeometryFilter:
    def __init__(self, train_x: np.ndarray, quantile: float, multiplier: float) -> None:
        self.orders = fourier_mode_orders(train_x.shape[1])
        high_mask = self.orders >= 4.0
        total_energy = np.sum(train_x**2, axis=1) + 1e-12
        high_fraction = np.sum(train_x[:, high_mask] ** 2, axis=1) / total_energy
        max_abs = np.max(np.abs(train_x), axis=1)
        self.high_fraction_threshold = float(np.quantile(high_fraction, quantile) * multiplier + 1e-12)
        self.max_abs_threshold = float(np.quantile(max_abs, quantile) * multiplier + 1e-12)

    def evaluate(self, x: np.ndarray) -> tuple[bool, dict[str, float]]:
        x = np.asarray(x, dtype=np.float32)
        if not np.isfinite(x).all():
            return False, {
                "geometry_filter_pass": 0.0,
                "geometry_penalty": 1e6,
                "spectral_high_fraction": float("inf"),
                "max_abs_coefficient": float("inf"),
            }
        high_mask = self.orders >= 4.0
        total_energy = float(np.sum(x**2) + 1e-12)
        high_fraction = float(np.sum(x[high_mask] ** 2) / total_energy)
        max_abs = float(np.max(np.abs(x)))
        high_excess = max(high_fraction / self.high_fraction_threshold - 1.0, 0.0)
        max_abs_excess = max(max_abs / self.max_abs_threshold - 1.0, 0.0)
        penalty = float(max(high_excess, max_abs_excess))
        return penalty <= 0.0, {
            "geometry_filter_pass": 1.0 if penalty <= 0.0 else 0.0,
            "geometry_penalty": penalty,
            "spectral_high_fraction": high_fraction,
            "max_abs_coefficient": max_abs,
        }

    def summary(self) -> dict[str, float]:
        return {
            "spectral_high_fraction_threshold": self.high_fraction_threshold,
            "max_abs_coefficient_threshold": self.max_abs_threshold,
        }


class LatentSearchSpace:
    def __init__(
        self,
        source_x: np.ndarray,
        lower_x: np.ndarray,
        upper_x: np.ndarray,
        n_components: int,
        seed: int,
        diversity_fraction: float,
    ) -> None:
        if source_x.shape[0] < 3:
            raise ValueError("Need at least three samples to build a latent search space")
        n_components = min(n_components, source_x.shape[1], source_x.shape[0] - 1)
        self.pca = PCA(n_components=n_components, random_state=seed)
        self.z_pool = self.pca.fit_transform(source_x)
        self.lower_x = lower_x
        self.upper_x = upper_x
        self.lower_z = np.quantile(self.z_pool, 0.005, axis=0).astype(np.float32)
        self.upper_z = np.quantile(self.z_pool, 0.995, axis=0).astype(np.float32)
        same = self.upper_z <= self.lower_z
        self.upper_z[same] = self.lower_z[same] + 1e-6
        self.diversity_threshold = self._diversity_threshold(diversity_fraction)

    def _diversity_threshold(self, diversity_fraction: float) -> float:
        if self.z_pool.shape[0] < 3:
            return 1e-6
        nn = NearestNeighbors(n_neighbors=2)
        nn.fit(self.z_pool)
        dist, _ = nn.kneighbors(self.z_pool)
        return float(max(np.median(dist[:, 1]) * diversity_fraction, 1e-6))

    def encode(self, x: np.ndarray) -> np.ndarray:
        z = self.pca.transform(np.asarray(x, dtype=np.float32)[None, :])[0]
        return np.clip(z, self.lower_z, self.upper_z).astype(np.float32)

    def decode(self, z: np.ndarray) -> np.ndarray:
        x = self.pca.inverse_transform(np.asarray(z, dtype=np.float32)[None, :])[0]
        return np.clip(x, self.lower_x, self.upper_x).astype(np.float32)

    def summary(self) -> dict[str, Any]:
        return {
            "n_components": int(self.pca.n_components_),
            "explained_variance_ratio_sum": float(np.sum(self.pca.explained_variance_ratio_)),
            "diversity_threshold": self.diversity_threshold,
        }


def load_relaxed55() -> pd.DataFrame:
    path = STAGE1_OUTPUT_DIR / "dataset" / "relaxed55.jsonl"
    return pd.DataFrame(json.loads(line) for line in path.read_text().splitlines() if line.strip())


def make_record(
    method_id: str,
    source: str,
    seed: int,
    x: np.ndarray,
    prediction: dict[str, np.ndarray],
    support: dict[str, float],
    score: float,
    boundary_dir: Path,
) -> dict[str, Any]:
    boundary_json = surface_json_from_x(x)
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


def feasibility_first_score(
    method_id: str,
    prediction: dict[str, np.ndarray],
    support: dict[str, float],
    config: dict[str, float],
) -> float:
    positive_violation = float(support["predicted_positive_violation"])
    qi_violation = float(support["predicted_qi_violation"])
    objective = prediction_value(prediction, "L_gradB")
    score = float(config["alpha_l"]) * objective
    score -= float(config["lambda_max_violation"]) * positive_violation
    score -= float(config["lambda_qi"]) * qi_violation
    score -= float(config["lambda_infeasible"]) * float(support["predicted_infeasible_prob"])
    score -= float(config["lambda_geometry"]) * float(support["geometry_penalty"])
    if method_id == "E3-stage2":
        score -= float(config["lambda_uncertainty"]) * prediction_value(prediction, "L_gradB", key="std")
        score -= float(config["lambda_support"]) * float(support["support_penalty"])
    if positive_violation <= float(config["near_feasible_bonus_threshold"]):
        score += objective
    return float(score)


def rank_diverse(
    records: list[tuple[dict[str, Any], np.ndarray]],
    limit: int,
    min_latent_distance: float,
) -> list[tuple[dict[str, Any], np.ndarray]]:
    best_by_id: dict[str, tuple[dict[str, Any], np.ndarray]] = {}
    for record, z in records:
        candidate_id = record["candidate_id"]
        previous = best_by_id.get(candidate_id)
        if previous is None or record["candidate_score"] > previous[0]["candidate_score"]:
            best_by_id[candidate_id] = (record, z)
    ranked = sorted(best_by_id.values(), key=lambda item: item[0]["candidate_score"], reverse=True)
    selected: list[tuple[dict[str, Any], np.ndarray]] = []
    leftovers: list[tuple[dict[str, Any], np.ndarray]] = []
    for record, z in ranked:
        if all(float(np.linalg.norm(z - selected_z)) >= min_latent_distance for _, selected_z in selected):
            selected.append((record, z))
        else:
            leftovers.append((record, z))
        if len(selected) >= limit:
            break
    if len(selected) < limit:
        selected.extend(leftovers[: limit - len(selected)])
    for idx, (record, _) in enumerate(selected[:limit]):
        record["rank_before_audit"] = idx + 1
    return selected[:limit]


def choose_initial_x(
    method_id: str,
    x_pool: np.ndarray,
    bundle: dict[str, Any],
    support_model: SupportModel | None,
    geometry_filter: GeometryFilter,
    score_config: dict[str, float],
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    prediction = predict_ensemble(bundle, x_pool, batch_size=batch_size)
    support_penalties = (
        support_model.penalty(x_pool)
        if support_model is not None
        else np.zeros(x_pool.shape[0], dtype=np.float32)
    )
    scores = []
    geometry_penalties = []
    for idx in range(x_pool.shape[0]):
        _, geometry_metrics = geometry_filter.evaluate(x_pool[idx])
        support = constraint_support_metrics(prediction, idx)
        support.update(geometry_metrics)
        support["support_penalty"] = float(support_penalties[idx])
        row_prediction = prediction_row(prediction, idx)
        scores.append(feasibility_first_score(method_id, row_prediction, support, score_config))
        geometry_penalties.append(float(geometry_metrics["geometry_penalty"]))
    best_idx = int(np.argmax(scores))
    return x_pool[best_idx].astype(np.float32), {
        "method_id": method_id,
        "pool_size": int(x_pool.shape[0]),
        "selected_index": best_idx,
        "selected_score": float(scores[best_idx]),
        "selected_predicted_L_gradB": prediction_value(prediction, "L_gradB", best_idx),
        "selected_predicted_max_normalized_violation": float(
            prediction["max_normalized_violation"][best_idx]
        ),
        "selected_predicted_infeasible_prob": float(prediction["infeasible_prob"][best_idx]),
        "selected_support_penalty": float(support_penalties[best_idx]),
        "selected_geometry_penalty": float(geometry_penalties[best_idx]),
    }


def optimize(
    method_id: str,
    seed: int,
    budget: int,
    bundle: dict[str, Any],
    support_model: SupportModel | None,
    geometry_filter: GeometryFilter,
    latent_space: LatentSearchSpace,
    boundary_dir: Path,
    per_method_pool: int,
    init_x: np.ndarray,
    init_source: str,
    score_config: dict[str, float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    init_z = latent_space.encode(init_x)
    parametrization = ng.p.Array(init=init_z).set_bounds(latent_space.lower_z, latent_space.upper_z)
    optimizer_cls = ng.optimizers.CMA if method_id == "E3-stage2" else ng.optimizers.NGOpt
    optimizer = optimizer_cls(parametrization=parametrization, budget=budget, num_workers=1)
    optimizer.parametrization.random_state.seed(seed)
    records: list[tuple[dict[str, Any], np.ndarray]] = []
    rejected_by_geometry = 0

    def evaluate_x(x: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, float], float] | None:
        geometry_pass, geometry_metrics = geometry_filter.evaluate(x)
        if not geometry_pass:
            return None
        prediction = predict_ensemble(bundle, x, batch_size=1)
        support = constraint_support_metrics(prediction)
        support.update(geometry_metrics)
        support["support_penalty"] = float(support_model.penalty(x)[0]) if support_model is not None else 0.0
        score = feasibility_first_score(method_id, prediction, support, score_config)
        return prediction, support, score

    init_x_decoded = latent_space.decode(init_z)
    init_eval = evaluate_x(init_x_decoded)
    if init_eval is not None:
        prediction, support, score = init_eval
        records.append(
            (
                make_record(method_id, init_source, seed, init_x_decoded, prediction, support, score, boundary_dir),
                init_z,
            )
        )

    for _ in range(budget):
        candidate = optimizer.ask()
        z = np.asarray(candidate.value, dtype=np.float32)
        x = latent_space.decode(z)
        evaluation = evaluate_x(x)
        if evaluation is None:
            rejected_by_geometry += 1
            optimizer.tell(candidate, 1e6)
            continue
        prediction, support, score = evaluation
        optimizer.tell(candidate, -score)
        if len(records) < per_method_pool * 5 or score > min(item[0]["candidate_score"] for item in records):
            records.append(
                (
                    make_record(
                        method_id,
                        "latent_feasibility_cmaes" if method_id == "E2-stage2" else "latent_conservative_cmaes",
                        seed,
                        x,
                        prediction,
                        support,
                        score,
                        boundary_dir,
                    ),
                    z,
                )
            )
            if len(records) > per_method_pool * 8:
                records = rank_diverse(records, per_method_pool * 4, latent_space.diversity_threshold)

    ranked = rank_diverse(records, per_method_pool, latent_space.diversity_threshold)
    return [record for record, _ in ranked], {
        "accepted_pool_size": len(records),
        "rejected_by_geometry": rejected_by_geometry,
        "latent_space": latent_space.summary(),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    hardware = apply_thread_environment(config)
    ensure_output_dirs()
    seed = int(config.get("seed", 0))
    candidate_config = config["candidates"]
    score_config = candidate_config["feasibility"]
    per_method_pool = int(candidate_config["per_method_pool"])
    budget = int(candidate_config["cmaes_budget"])
    batch_size = int(candidate_config["batch_size"])
    boundary_dir = OUTPUT_DIR / "candidates" / "boundaries"

    train = pd.read_parquet(STAGE1_OUTPUT_DIR / "dataset" / "train.parquet")
    validation = pd.read_parquet(STAGE1_OUTPUT_DIR / "dataset" / "validation.parquet")
    relaxed = load_relaxed55()
    x_cols = feature_columns(train)
    train_x = train[x_cols].to_numpy(dtype=np.float32)
    validation_x = validation[x_cols].to_numpy(dtype=np.float32)
    relaxed_x = relaxed[x_cols].to_numpy(dtype=np.float32)
    lower, upper = train_bounds(train, x_cols)
    bundle = load_model_bundle("cuda:0")
    support_model = SupportModel(train_x, validation_x, n_components=20)
    geometry_filter = GeometryFilter(
        train_x,
        quantile=float(candidate_config["spectral_threshold_quantile"]),
        multiplier=float(candidate_config["spectral_threshold_multiplier"]),
    )

    e2_space = LatentSearchSpace(
        train_x,
        lower,
        upper,
        n_components=int(candidate_config["e2_latent_components"]),
        seed=seed,
        diversity_fraction=float(candidate_config["diversity_fraction"]),
    )
    e2_init, e2_init_summary = choose_initial_x(
        "E2-stage2", train_x, bundle, None, geometry_filter, score_config, batch_size
    )
    e2_init_summary["source"] = "train"

    e3_components = min(int(candidate_config["e3_latent_components"]), relaxed_x.shape[0] - 1)
    e3_space = LatentSearchSpace(
        relaxed_x,
        lower,
        upper,
        n_components=e3_components,
        seed=seed,
        diversity_fraction=float(candidate_config["diversity_fraction"]),
    )
    e3_init, e3_init_summary = choose_initial_x(
        "E3-stage2", relaxed_x, bundle, support_model, geometry_filter, score_config, batch_size
    )
    e3_init_summary["source"] = "relaxed55"

    e2, e2_summary = optimize(
        "E2-stage2",
        seed,
        budget,
        bundle,
        None,
        geometry_filter,
        e2_space,
        boundary_dir,
        per_method_pool,
        e2_init,
        "latent_feasibility_cmaes_init",
        score_config,
    )
    write_jsonl(OUTPUT_DIR / "candidates" / "e2_latent_feasibility_cmaes.jsonl", e2)

    e3, e3_summary = optimize(
        "E3-stage2",
        seed,
        budget,
        bundle,
        support_model,
        geometry_filter,
        e3_space,
        boundary_dir,
        per_method_pool,
        e3_init,
        "latent_conservative_cmaes_relaxed55_init",
        score_config,
    )
    write_jsonl(OUTPUT_DIR / "candidates" / "e3_latent_conservative_cmaes.jsonl", e3)

    write_json(
        OUTPUT_DIR / "run_summary" / "candidate_generation_stage2.json",
        {
            "hardware_config": hardware,
            "stage1_output_dir": str(STAGE1_OUTPUT_DIR),
            "search_revision": "stage2_latent_feasibility_first_no_online_vmec",
            "budget": budget,
            "e2_count": len(e2),
            "e3_count": len(e3),
            "geometry_filter": geometry_filter.summary(),
            "score_config": score_config,
            "initialization": {"E2-stage2": e2_init_summary, "E3-stage2": e3_init_summary},
            "search_summary": {"E2-stage2": e2_summary, "E3-stage2": e3_summary},
        },
    )
    print(f"Wrote E2-stage2={len(e2)} E3-stage2={len(e3)} candidates")


if __name__ == "__main__":
    main()
