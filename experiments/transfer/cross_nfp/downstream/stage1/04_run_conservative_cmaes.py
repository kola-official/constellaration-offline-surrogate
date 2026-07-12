from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import nevergrad as ng
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from candidate_utils import (
    OUTPUT_DIR,
    SupportModel,
    constraint_penalty,
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
from common import apply_thread_environment, ensure_output_dirs, load_config, parse_args, write_json


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
    def __init__(self, train_x: np.ndarray) -> None:
        self.orders = fourier_mode_orders(train_x.shape[1])
        high_mask = self.orders >= 4.0
        total_energy = np.sum(train_x**2, axis=1) + 1e-12
        high_fraction = np.sum(train_x[:, high_mask] ** 2, axis=1) / total_energy
        max_abs = np.max(np.abs(train_x), axis=1)
        self.high_fraction_threshold = float(np.quantile(high_fraction, 0.995) * 1.25 + 1e-12)
        self.max_abs_threshold = float(np.quantile(max_abs, 0.995) * 1.25 + 1e-12)

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
        lower: np.ndarray,
        upper: np.ndarray,
        n_components: int,
        seed: int,
    ) -> None:
        if source_x.shape[0] < 3:
            raise ValueError("Need at least three samples to build a latent search space")
        n_components = min(n_components, source_x.shape[1], source_x.shape[0] - 1)
        self.pca = PCA(n_components=n_components, random_state=seed)
        self.z_pool = self.pca.fit_transform(source_x)
        self.lower_x = lower
        self.upper_x = upper
        self.lower_z = np.quantile(self.z_pool, 0.005, axis=0).astype(np.float32)
        self.upper_z = np.quantile(self.z_pool, 0.995, axis=0).astype(np.float32)
        same = self.upper_z <= self.lower_z
        self.upper_z[same] = self.lower_z[same] + 1e-6
        self.diversity_threshold = self._diversity_threshold()

    def _diversity_threshold(self) -> float:
        if self.z_pool.shape[0] < 3:
            return 1e-6
        nn = NearestNeighbors(n_neighbors=2)
        nn.fit(self.z_pool)
        dist, _ = nn.kneighbors(self.z_pool)
        return float(max(np.median(dist[:, 1]) * 0.25, 1e-6))

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


def feasibility_first_score(
    method_id: str,
    prediction: dict[str, np.ndarray],
    support: dict[str, float],
    support_penalty: float,
    geometry_penalty: float,
) -> float:
    positive_violation = float(support["predicted_positive_violation"])
    qi_violation = float(support["predicted_qi_violation"])
    infeasible_prob = float(support["predicted_infeasible_prob"])
    objective = prediction_value(prediction, "L_gradB")

    # Feasibility is deliberately lexicographic-ish: while every known seed is
    # infeasible, reducing violation should dominate chasing a high L_gradB.
    score = 0.25 * objective - 80.0 * positive_violation - 40.0 * qi_violation
    score -= 2.0 * infeasible_prob
    if positive_violation <= 0.05:
        score += objective
    if method_id == "E3":
        score -= 2.0 * prediction_value(prediction, "L_gradB", key="std")
        score -= 2.0 * support_penalty
        score -= 8.0 * geometry_penalty
    return float(score)


def rank_diverse(
    records: list[tuple[dict[str, Any], np.ndarray]],
    limit: int,
    min_latent_distance: float,
) -> list[dict[str, Any]]:
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
    result = [record for record, _ in selected[:limit]]
    for idx, record in enumerate(result):
        record["rank_before_audit"] = idx + 1
    return result


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
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    init_z = latent_space.encode(init_x)
    parametrization = ng.p.Array(init=init_z).set_bounds(latent_space.lower_z, latent_space.upper_z)
    optimizer_cls = ng.optimizers.CMA if method_id == "E3" else ng.optimizers.NGOpt
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
        support_penalty = float(support_model.penalty(x)[0]) if support_model is not None else 0.0
        support.update(geometry_metrics)
        support["support_penalty"] = support_penalty
        score = feasibility_first_score(
            method_id,
            prediction,
            support,
            support_penalty=support_penalty,
            geometry_penalty=float(geometry_metrics["geometry_penalty"]),
        )
        return prediction, support, score

    init_eval = evaluate_x(latent_space.decode(init_z))
    if init_eval is not None:
        prediction, support, score = init_eval
        records.append(
            (
                make_record(
                    method_id,
                    init_source,
                    seed,
                    latent_space.decode(init_z),
                    prediction,
                    support,
                    score,
                    boundary_dir,
                ),
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
        if len(records) < per_method_pool * 5 or score > min(r[0]["candidate_score"] for r in records):
            records.append(
                (
                    make_record(
                        method_id,
                        "surrogate_only_latent_cmaes" if method_id == "E2" else "conservative_latent_cmaes",
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
                ranked = rank_diverse(records, per_method_pool * 4, latent_space.diversity_threshold)
                keep_ids = {record["candidate_id"] for record in ranked}
                records = [(record, z_value) for record, z_value in records if record["candidate_id"] in keep_ids]

    ranked_records = rank_diverse(records, per_method_pool, latent_space.diversity_threshold)
    summary = {
        "rejected_by_geometry": rejected_by_geometry,
        "accepted_pool_size": len(records),
        "latent_space": latent_space.summary(),
    }
    return ranked_records, summary


def load_relaxed55() -> pd.DataFrame:
    path = OUTPUT_DIR / "dataset" / "relaxed55.jsonl"
    if not path.exists():
        return pd.DataFrame()
    return pd.DataFrame(json.loads(line) for line in path.read_text().splitlines() if line.strip())


def choose_initial_x(
    method_id: str,
    x_pool: np.ndarray,
    bundle: dict[str, Any],
    support_model: SupportModel | None,
    geometry_filter: GeometryFilter,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if x_pool.size == 0:
        raise ValueError(f"{method_id} initial pool is empty")
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
        score = feasibility_first_score(
            method_id,
            prediction_row(prediction, idx),
            support,
            support_penalty=float(support_penalties[idx]),
            geometry_penalty=float(geometry_metrics["geometry_penalty"]),
        )
        scores.append(score)
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


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    hardware = apply_thread_environment(config)
    ensure_output_dirs()
    seed = int(config.get("seed", 0))
    per_method_pool = int(config["candidates"]["per_method_pool"])
    budget = int(config["candidates"]["cmaes_budget"])
    boundary_dir = OUTPUT_DIR / "candidates" / "boundaries"

    train = pd.read_parquet(OUTPUT_DIR / "dataset" / "train.parquet")
    validation = pd.read_parquet(OUTPUT_DIR / "dataset" / "validation.parquet")
    x_cols = feature_columns(train)
    train_x = train[x_cols].to_numpy(dtype=np.float32)
    validation_x = validation[x_cols].to_numpy(dtype=np.float32)
    lower, upper = train_bounds(train, x_cols)
    bundle = load_model_bundle("cuda:0")
    support_model = SupportModel(train_x, validation_x, n_components=20)
    geometry_filter = GeometryFilter(train_x)
    candidate_batch_size = int(config["candidates"]["batch_size"])

    e2_init, e2_init_summary = choose_initial_x(
        "E2", train_x, bundle, None, geometry_filter, candidate_batch_size
    )
    e2_init_summary["source"] = "train"
    e2_space = LatentSearchSpace(train_x, lower, upper, n_components=20, seed=seed)

    relaxed = load_relaxed55()
    if relaxed.empty:
        e3_pool = train_x
        e3_source = "train_fallback"
    else:
        e3_pool = relaxed[x_cols].to_numpy(dtype=np.float32)
        e3_source = "relaxed55"
    e3_init, e3_init_summary = choose_initial_x(
        "E3", e3_pool, bundle, support_model, geometry_filter, candidate_batch_size
    )
    e3_init_summary["source"] = e3_source
    e3_components = min(10, e3_pool.shape[0] - 1)
    e3_space = LatentSearchSpace(e3_pool, lower, upper, n_components=e3_components, seed=seed)

    e2, e2_summary = optimize(
        "E2",
        seed,
        budget,
        bundle,
        None,
        geometry_filter,
        e2_space,
        boundary_dir,
        per_method_pool,
        e2_init,
        "surrogate_only_latent_cmaes_init",
    )
    write_jsonl(OUTPUT_DIR / "candidates" / "e2_surrogate_only_cmaes.jsonl", e2)

    e3, e3_summary = optimize(
        "E3",
        seed,
        budget,
        bundle,
        support_model,
        geometry_filter,
        e3_space,
        boundary_dir,
        per_method_pool,
        e3_init,
        "conservative_latent_cmaes_relaxed55_init",
    )
    write_jsonl(OUTPUT_DIR / "candidates" / "e3_csa_cmaes_full.jsonl", e3)

    write_json(
        OUTPUT_DIR / "run_summary" / "candidate_generation_e2_e3.json",
        {
            "hardware_config": hardware,
            "budget": budget,
            "e2_count": len(e2),
            "e3_count": len(e3),
            "support_threshold": support_model.threshold,
            "geometry_filter": geometry_filter.summary(),
            "search_revision": "latent_pca_feasibility_first_no_online_vmec",
            "initialization": {
                "E2": e2_init_summary,
                "E3": e3_init_summary,
            },
            "search_summary": {
                "E2": e2_summary,
                "E3": e3_summary,
            },
        },
    )
    print(f"Wrote E2={len(e2)} E3={len(e3)} candidates")


if __name__ == "__main__":
    main()
