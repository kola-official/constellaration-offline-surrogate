from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nevergrad as ng
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from common_stage4 import (
    OUTPUT_DIR,
    STAGE1_DIR,
    STAGE1_OUTPUT_DIR,
    actual_positive_violation,
    apply_thread_environment,
    ensure_output_dirs,
    existing_audit_records,
    feature_columns,
    finite_float,
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
    predicted_constraint_violations,
    surface_json_from_x,
    train_bounds,
    uncertainty_dict,
    write_boundary,
)
from candidate_utils import load_model_bundle  # noqa: E402


@dataclass
class SearchSpace:
    name: str
    lower: np.ndarray
    upper: np.ndarray
    pca: PCA | None = None
    x_lower: np.ndarray | None = None
    x_upper: np.ndarray | None = None

    def encode(self, x: np.ndarray) -> np.ndarray:
        if self.pca is None:
            return np.clip(x, self.lower, self.upper).astype(np.float32)
        z = self.pca.transform(np.asarray(x, dtype=np.float32)[None, :])[0]
        return np.clip(z, self.lower, self.upper).astype(np.float32)

    def decode(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float32)
        if self.pca is None:
            return np.clip(value, self.lower, self.upper).astype(np.float32)
        x = self.pca.inverse_transform(value[None, :])[0]
        if self.x_lower is not None and self.x_upper is not None:
            x = np.clip(x, self.x_lower, self.x_upper)
        return x.astype(np.float32)


class TrustDistanceModel:
    def __init__(
        self,
        train_x: np.ndarray,
        validation_x: np.ndarray,
        relaxed_x: np.ndarray,
        train_components: int,
        relaxed_components: int,
        train_quantile: float,
        relaxed_quantile: float,
        train_multiplier: float,
        relaxed_multiplier: float,
    ) -> None:
        train_components = min(train_components, train_x.shape[1], train_x.shape[0] - 1)
        relaxed_components = min(relaxed_components, relaxed_x.shape[1], relaxed_x.shape[0] - 1)
        self.train_pca = PCA(n_components=train_components, random_state=0)
        train_z = self.train_pca.fit_transform(train_x)
        validation_z = self.train_pca.transform(validation_x)
        self.train_nn = NearestNeighbors(n_neighbors=1)
        self.train_nn.fit(train_z)
        validation_dist, _ = self.train_nn.kneighbors(validation_z)
        self.train_threshold = float(np.quantile(validation_dist[:, 0], train_quantile) * train_multiplier)

        self.relaxed_pca = PCA(n_components=relaxed_components, random_state=0)
        relaxed_z = self.relaxed_pca.fit_transform(relaxed_x)
        self.relaxed_nn = NearestNeighbors(n_neighbors=1)
        self.relaxed_nn.fit(relaxed_z)
        relaxed_dist, _ = self.relaxed_nn.kneighbors(relaxed_z)
        positive = relaxed_dist[:, 0][relaxed_dist[:, 0] > 0]
        base = positive if positive.size else np.array([1e-6], dtype=np.float32)
        self.relaxed_threshold = float(max(np.quantile(base, relaxed_quantile) * relaxed_multiplier, 1e-6))

    def evaluate(self, x: np.ndarray) -> dict[str, float]:
        x = np.asarray(x, dtype=np.float32)[None, :]
        train_dist, _ = self.train_nn.kneighbors(self.train_pca.transform(x))
        relaxed_dist, _ = self.relaxed_nn.kneighbors(self.relaxed_pca.transform(x))
        train_d = float(train_dist[0, 0])
        relaxed_d = float(relaxed_dist[0, 0])
        return {
            "train_distance": train_d,
            "train_distance_ratio": train_d / max(self.train_threshold, 1e-12),
            "relaxed_distance": relaxed_d,
            "relaxed_distance_ratio": relaxed_d / max(self.relaxed_threshold, 1e-12),
        }

    def summary(self) -> dict[str, float | int]:
        return {
            "train_components": int(self.train_pca.n_components_),
            "relaxed_components": int(self.relaxed_pca.n_components_),
            "train_threshold": self.train_threshold,
            "relaxed_threshold": self.relaxed_threshold,
        }


class GeometryFilter:
    def __init__(self, train_x: np.ndarray) -> None:
        self.max_abs_threshold = float(np.quantile(np.max(np.abs(train_x), axis=1), 0.995) * 1.25)
        total = np.sum(train_x**2, axis=1) + 1e-12
        high = np.sum(train_x[:, -20:] ** 2, axis=1) / total
        self.high_fraction_threshold = float(np.quantile(high, 0.995) * 1.25 + 1e-12)

    def evaluate(self, x: np.ndarray) -> dict[str, float]:
        x = np.asarray(x, dtype=np.float32)
        if not np.isfinite(x).all():
            return {
                "geometry_filter_pass": 0.0,
                "geometry_penalty": 1e6,
                "max_abs_coefficient": float("inf"),
                "spectral_high_fraction": float("inf"),
            }
        max_abs = float(np.max(np.abs(x)))
        high = float(np.sum(x[-20:] ** 2) / (np.sum(x**2) + 1e-12))
        penalty = max(
            max_abs / max(self.max_abs_threshold, 1e-12) - 1.0,
            high / max(self.high_fraction_threshold, 1e-12) - 1.0,
            0.0,
        )
        return {
            "geometry_filter_pass": 1.0 if penalty <= 0.0 else 0.0,
            "geometry_penalty": float(penalty),
            "max_abs_coefficient": max_abs,
            "spectral_high_fraction": high,
        }

    def summary(self) -> dict[str, float]:
        return {
            "max_abs_threshold": self.max_abs_threshold,
            "high_fraction_threshold": self.high_fraction_threshold,
        }


def load_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(STAGE1_OUTPUT_DIR / "dataset" / "train.parquet")
    validation = pd.read_parquet(STAGE1_OUTPUT_DIR / "dataset" / "validation.parquet")
    relaxed = pd.DataFrame(
        json.loads(line)
        for line in (STAGE1_OUTPUT_DIR / "dataset" / "relaxed55.jsonl").read_text().splitlines()
        if line.strip()
    )
    return train, validation, relaxed


def make_pca_space(name: str, x_pool: np.ndarray, lower_x: np.ndarray, upper_x: np.ndarray, n_components: int) -> SearchSpace:
    n_components = min(n_components, x_pool.shape[1], x_pool.shape[0] - 1)
    pca = PCA(n_components=n_components, random_state=0)
    z_pool = pca.fit_transform(x_pool)
    lower_z = np.quantile(z_pool, 0.005, axis=0).astype(np.float32)
    upper_z = np.quantile(z_pool, 0.995, axis=0).astype(np.float32)
    same = upper_z <= lower_z
    upper_z[same] = lower_z[same] + 1e-6
    return SearchSpace(name=name, lower=lower_z, upper=upper_z, pca=pca, x_lower=lower_x, x_upper=upper_x)


def prediction_payload(
    x: np.ndarray,
    bundle: dict[str, Any],
    trust_model: TrustDistanceModel,
    geometry_filter: GeometryFilter,
) -> dict[str, Any]:
    prediction = predict_ensemble(bundle, x, batch_size=1)
    support = constraint_support_metrics(prediction)
    trust = trust_model.evaluate(x)
    geometry = geometry_filter.evaluate(x)
    metrics = prediction["mean"][0]
    return {
        "prediction": prediction,
        "predicted_metrics": metric_dict(metrics),
        "predicted_uncertainty": uncertainty_dict(prediction["std"][0]),
        "support_metrics": support,
        "trust_metrics": trust,
        "geometry_metrics": geometry,
        "constraint_violations": predicted_constraint_violations(metrics),
    }


def alm_loss(
    payload: dict[str, Any],
    lambdas: np.ndarray,
    cfg: dict[str, Any],
    raw_group: bool,
) -> float:
    metrics = payload["predicted_metrics"]
    g = np.array(
        [
            payload["constraint_violations"]["aspect_ratio"],
            payload["constraint_violations"]["iota"],
            payload["constraint_violations"]["log10_qi"],
            payload["constraint_violations"]["mirror"],
            payload["constraint_violations"]["elongation"],
        ],
        dtype=np.float32,
    )
    gpos = np.maximum(g, 0.0)
    alm = float(np.dot(lambdas, gpos) + 0.5 * float(cfg["rho"]) * np.dot(gpos, gpos))
    loss = -float(cfg["objective_scale"]) * float(metrics["L_gradB"]) + alm
    if not raw_group:
        unc = payload["predicted_uncertainty"]
        trust = payload["trust_metrics"]
        geom = payload["geometry_metrics"]
        loss += float(cfg["uncertainty_weight"]) * (
            finite_float(unc.get("log10_qi")) + 0.25 * finite_float(unc.get("L_gradB"))
        )
        loss += float(cfg["train_distance_weight"]) * max(trust["train_distance_ratio"] - 1.0, 0.0)
        loss += float(cfg["relaxed_distance_weight"]) * max(trust["relaxed_distance_ratio"] - 1.0, 0.0)
        loss += float(cfg["geometry_weight"]) * finite_float(geom["geometry_penalty"], 1e6)
    return float(loss)


def rank_score(record: dict[str, Any], weights: dict[str, float]) -> float:
    support = record["support_metrics"]
    trust = record["trust_metrics"]
    unc = record["predicted_uncertainty"]
    geom = record["geometry_metrics"]
    metrics = record["predicted_metrics"]
    return float(
        float(weights["objective_weight"]) * finite_float(metrics.get("L_gradB"))
        - float(weights["positive_violation_weight"]) * finite_float(support.get("predicted_positive_violation"))
        - float(weights["qi_violation_weight"]) * finite_float(support.get("predicted_qi_violation"))
        - float(weights["uncertainty_weight"]) * finite_float(unc.get("log10_qi"))
        - float(weights["train_distance_weight"]) * max(finite_float(trust.get("train_distance_ratio")) - 1.0, 0.0)
        - float(weights["relaxed_distance_weight"]) * max(finite_float(trust.get("relaxed_distance_ratio")) - 1.0, 0.0)
        - float(weights["geometry_weight"]) * finite_float(geom.get("geometry_penalty"), 1e6)
    )


def make_record(
    method_id: str,
    source: str,
    seed: int,
    x: np.ndarray,
    payload: dict[str, Any],
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
        "predicted_metrics": payload["predicted_metrics"],
        "predicted_uncertainty": payload["predicted_uncertainty"],
        "support_metrics": payload["support_metrics"],
        "trust_metrics": payload["trust_metrics"],
        "geometry_metrics": payload["geometry_metrics"],
        "predicted_constraint_violations": payload["constraint_violations"],
        "candidate_score": 0.0,
        "rank_before_audit": 0,
    }


def choose_initial_x(x_pool: np.ndarray, bundle: dict[str, Any], trust_model: TrustDistanceModel, geometry_filter: GeometryFilter) -> np.ndarray:
    batch = min(4096, len(x_pool))
    best_score = -float("inf")
    best_x = x_pool[0]
    for start in range(0, len(x_pool), batch):
        xs = x_pool[start : start + batch]
        pred = predict_ensemble(bundle, xs, batch_size=batch)
        for idx, x in enumerate(xs):
            payload = {
                "predicted_metrics": metric_dict(pred["mean"][idx]),
                "predicted_uncertainty": uncertainty_dict(pred["std"][idx]),
                "support_metrics": constraint_support_metrics(pred, idx),
                "trust_metrics": trust_model.evaluate(x),
                "geometry_metrics": geometry_filter.evaluate(x),
            }
            score = -payload["support_metrics"]["predicted_positive_violation"] + 0.02 * payload["predicted_metrics"]["L_gradB"]
            if score > best_score:
                best_score = float(score)
                best_x = x
    return best_x.astype(np.float32)


def optimize_group(
    method_id: str,
    space: SearchSpace,
    init_x: np.ndarray,
    bundle: dict[str, Any],
    trust_model: TrustDistanceModel,
    geometry_filter: GeometryFilter,
    cfg: dict[str, Any],
    seed: int,
    boundary_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = np.random.default_rng(seed)
    init_value = space.encode(init_x)
    parametrization = ng.p.Array(init=init_value).set_bounds(space.lower, space.upper)
    optimizer = ng.optimizers.NGOpt(parametrization=parametrization, budget=int(cfg["surrogate_budget_per_seed"]), num_workers=1)
    optimizer.parametrization.random_state.seed(seed)
    lambdas = np.zeros(5, dtype=np.float32)
    best_gpos = np.ones(5, dtype=np.float32) * 10.0
    records: list[dict[str, Any]] = []
    rejected_geometry = 0
    raw_group = method_id == "S4A-1"
    alm_cfg = cfg["alm"]

    for step in range(int(cfg["surrogate_budget_per_seed"])):
        candidate = optimizer.ask()
        x = space.decode(np.asarray(candidate.value, dtype=np.float32))
        payload = prediction_payload(x, bundle, trust_model, geometry_filter)
        if payload["geometry_metrics"]["geometry_filter_pass"] < 1.0 and not raw_group:
            rejected_geometry += 1
            optimizer.tell(candidate, 1e6)
            continue
        loss = alm_loss(payload, lambdas, alm_cfg, raw_group=raw_group)
        optimizer.tell(candidate, loss)
        g = np.array(list(payload["constraint_violations"].values()), dtype=np.float32)
        gpos = np.maximum(g, 0.0)
        if float(np.max(gpos)) < float(np.max(best_gpos)):
            best_gpos = gpos
        if (step + 1) % int(alm_cfg["lambda_update_interval"]) == 0:
            lambdas = np.maximum(0.0, lambdas + float(alm_cfg["rho"]) * best_gpos)
        if len(records) < int(cfg["candidate_pool_per_group"]) * 4 or rng.random() < 0.05:
            record = make_record(method_id, f"{space.name}_surrogate_alm_ngopt", seed, x, payload, boundary_dir)
            record["candidate_score"] = rank_score(record, cfg["ranking"])
            record["alm_loss"] = float(loss)
            records.append(record)

    summary = {
        "method_id": method_id,
        "space": space.name,
        "seed": seed,
        "records_before_dedupe": len(records),
        "rejected_geometry": rejected_geometry,
        "final_lambdas": lambdas.tolist(),
        "best_predicted_gpos": best_gpos.tolist(),
    }
    return records, summary


def dedupe_best(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for record in records:
        cid = record["candidate_id"]
        if cid not in best or record["candidate_score"] > best[cid]["candidate_score"]:
            best[cid] = record
    return sorted(best.values(), key=lambda row: row["candidate_score"], reverse=True)


def attach_prior_risk(records: list[dict[str, Any]], audited: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    audited_by_id = {row.get("candidate_id"): row for row in audited if row.get("candidate_id")}
    for record in records:
        prior = audited_by_id.get(record["candidate_id"])
        record["prior_audit_match"] = bool(prior)
        if prior:
            record["prior_audit_stage"] = prior.get("stage")
            record["prior_vmec_success"] = prior.get("vmec_success")
            record["prior_positive_max_violation"] = actual_positive_violation(prior)
            record["high_risk_surrogate_arbitrage"] = bool(
                actual_positive_violation(prior) is not None
                and actual_positive_violation(prior) >= float(cfg["high_risk_positive_violation"])
            )
        else:
            record["high_risk_surrogate_arbitrage"] = False
    return records


def rank_diverse(records: list[dict[str, Any]], limit: int, min_distance: float = 0.02) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    leftovers: list[dict[str, Any]] = []
    for record in dedupe_best(records):
        x = np.array([record["predicted_metrics"]["L_gradB"], record["support_metrics"]["predicted_positive_violation"]], dtype=np.float32)
        if all(
            np.linalg.norm(
                x - np.array([row["predicted_metrics"]["L_gradB"], row["support_metrics"]["predicted_positive_violation"]], dtype=np.float32)
            )
            >= min_distance
            for row in selected
        ):
            selected.append(record)
        else:
            leftovers.append(record)
        if len(selected) >= limit:
            break
    if len(selected) < limit:
        selected.extend(leftovers[: limit - len(selected)])
    return selected[:limit]


def assign_ranks(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for idx, record in enumerate(records):
        record["rank_before_audit"] = idx + 1
    return records


def random_control(candidate_pool: list[dict[str, Any]], repeats: int, audit_budget: int, seed: int) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    rng = np.random.default_rng(seed)
    pool = dedupe_best(candidate_pool)
    rows = []
    selected_first: list[dict[str, Any]] = []
    for rep in range(repeats):
        if len(pool) <= audit_budget:
            indices = np.arange(len(pool))
        else:
            indices = rng.choice(len(pool), size=audit_budget, replace=False)
        chosen = [pool[int(i)] for i in indices]
        if rep == 0:
            selected_first = [dict(row, method_id="S4A-R", source="random_audit_control") for row in chosen]
        rows.append(
            {
                "repeat": rep,
                "candidate_ids": ",".join(row["candidate_id"] for row in chosen),
                "mean_predicted_positive_violation": float(np.mean([row["support_metrics"]["predicted_positive_violation"] for row in chosen])),
                "best_predicted_positive_violation": float(min(row["support_metrics"]["predicted_positive_violation"] for row in chosen)),
                "mean_candidate_score": float(np.mean([row["candidate_score"] for row in chosen])),
            }
        )
    return assign_ranks(selected_first), pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    hardware = apply_thread_environment(config)
    ensure_output_dirs()
    seed = int(config.get("seed", 0))
    cfg = config["stage4a"]
    train, validation, relaxed = load_frames()
    x_cols = feature_columns(train)
    train_x = train[x_cols].to_numpy(dtype=np.float32)
    validation_x = validation[x_cols].to_numpy(dtype=np.float32)
    relaxed_x = relaxed[x_cols].to_numpy(dtype=np.float32)
    lower, upper = train_bounds(train, x_cols)

    bundle = load_model_bundle("cuda:0")
    trust_model = TrustDistanceModel(
        train_x,
        validation_x,
        relaxed_x,
        int(cfg["train_pca_components"]),
        int(cfg["relaxed_pca_components"]),
        float(cfg["train_distance_quantile"]),
        float(cfg["relaxed_distance_quantile"]),
        float(cfg["train_distance_multiplier"]),
        float(cfg["relaxed_distance_multiplier"]),
    )
    geometry_filter = GeometryFilter(train_x)
    raw_space = SearchSpace("raw_train_box", lower, upper)
    train_space = make_pca_space("train_pca", train_x, lower, upper, int(cfg["train_pca_components"]))
    relaxed_space = make_pca_space("relaxed55_pca", relaxed_x, lower, upper, int(cfg["relaxed_pca_components"]))
    init_train = choose_initial_x(train_x, bundle, trust_model, geometry_filter)
    init_relaxed = choose_initial_x(relaxed_x, bundle, trust_model, geometry_filter)
    boundary_dir = OUTPUT_DIR / "candidates" / "boundaries"

    group_specs = [
        ("S4A-1", raw_space, init_train),
        ("S4A-2", train_space, init_train),
        ("S4A-3", relaxed_space, init_relaxed),
    ]
    all_records: list[dict[str, Any]] = []
    summaries = []
    for method_id, space, init_x in group_specs:
        group_records: list[dict[str, Any]] = []
        for offset in range(int(cfg["seeds_per_group"])):
            records, summary = optimize_group(
                method_id,
                space,
                init_x,
                bundle,
                trust_model,
                geometry_filter,
                cfg,
                seed + offset * 1009,
                boundary_dir,
            )
            group_records.extend(records)
            summaries.append(summary)
        group_records = attach_prior_risk(dedupe_best(group_records), existing_audit_records(), cfg)
        group_records = assign_ranks(group_records[: int(cfg["candidate_pool_per_group"])])
        write_jsonl(OUTPUT_DIR / "candidates" / f"{method_id.lower()}_surrogate_alm_ngopt.jsonl", group_records)
        all_records.extend(group_records)

    audited = existing_audit_records()
    all_records = attach_prior_risk(dedupe_best(all_records), audited, cfg)
    trust_records = [row for row in all_records if row["method_id"] in {"S4A-2", "S4A-3"} and not row["high_risk_surrogate_arbitrage"]]
    diverse = assign_ranks(rank_diverse(trust_records, int(cfg["candidate_pool_per_group"])))
    for row in diverse:
        row["method_id"] = "S4A-4"
        row["source"] = "diverse_trust_ranked_union"
    write_jsonl(OUTPUT_DIR / "candidates" / "s4a-4_surrogate_alm_diverse.jsonl", diverse)

    random_candidates, random_df = random_control(
        all_records,
        repeats=int(cfg["random_repeats"]),
        audit_budget=int(cfg["random_audit_budget"]),
        seed=seed + 4444,
    )
    write_jsonl(OUTPUT_DIR / "candidates" / "s4a-r_random_audit_candidates.jsonl", random_candidates)
    random_df.to_csv(OUTPUT_DIR / "tables" / "s4a_random_audit_control.csv", index=False)

    ranking_rows = []
    for row in all_records + diverse + random_candidates:
        ranking_rows.append(
            {
                "method_id": row["method_id"],
                "candidate_id": row["candidate_id"],
                "rank_before_audit": row.get("rank_before_audit"),
                "candidate_score": row["candidate_score"],
                "predicted_L_gradB": row["predicted_metrics"]["L_gradB"],
                "predicted_positive_violation": row["support_metrics"]["predicted_positive_violation"],
                "predicted_qi_violation": row["support_metrics"]["predicted_qi_violation"],
                "train_distance_ratio": row["trust_metrics"]["train_distance_ratio"],
                "relaxed_distance_ratio": row["trust_metrics"]["relaxed_distance_ratio"],
                "geometry_penalty": row["geometry_metrics"]["geometry_penalty"],
                "high_risk_surrogate_arbitrage": row["high_risk_surrogate_arbitrage"],
                "prior_audit_match": row["prior_audit_match"],
            }
        )
    pd.DataFrame(ranking_rows).to_csv(OUTPUT_DIR / "tables" / "s4a_candidate_ranking.csv", index=False)
    selected_counts = Counter(row["method_id"] for row in ranking_rows)
    summary = {
        "run_id": config.get("run_id"),
        "hardware": hardware,
        "stage1_output_dir": str(STAGE1_OUTPUT_DIR),
        "trust_model": trust_model.summary(),
        "geometry_filter": geometry_filter.summary(),
        "search_summaries": summaries,
        "candidate_counts": dict(selected_counts),
        "outputs": {
            "ranking_csv": str(OUTPUT_DIR / "tables" / "s4a_candidate_ranking.csv"),
            "random_control_csv": str(OUTPUT_DIR / "tables" / "s4a_random_audit_control.csv"),
        },
    }
    write_json(OUTPUT_DIR / "run_summary" / "candidate_generation_stage4a.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
