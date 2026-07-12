from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from common import OUTPUT_DIR, apply_thread_environment, ensure_output_dirs, load_config, parse_args, write_json
from constellaration import forward_model, problems
from constellaration.geometry import surface_rz_fourier


CANDIDATE_FILES = {
    "E0": "e0_dataset_static.jsonl",
    "E1": "e1_relaxed55_gmm.jsonl",
    "E2": "e2_surrogate_only_cmaes.jsonl",
    "E3": "e3_csa_cmaes_full.jsonl",
}


def load_candidates(limit: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for method_id, filename in CANDIDATE_FILES.items():
        path = OUTPUT_DIR / "candidates" / filename
        method_records = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
        method_records = sorted(method_records, key=lambda item: item["rank_before_audit"])
        records.extend(method_records[:limit])
    return records


def load_benchmark_candidates() -> list[dict[str, Any]]:
    records = []
    for method_id, filename in CANDIDATE_FILES.items():
        path = OUTPUT_DIR / "candidates" / filename
        method_records = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
        method_records = sorted(method_records, key=lambda item: item["rank_before_audit"])
        if method_records:
            records.append(method_records[0])
    return records


def metric_payload(metrics: forward_model.ConstellarationMetrics) -> dict[str, Any]:
    return metrics.model_dump()


def evaluate_candidate(record: dict[str, Any]) -> dict[str, Any]:
    start = time.time()
    problem = problems.SimpleToBuildQIStellarator()
    try:
        boundary_json = Path(record["boundary_json_path"]).read_text()
        boundary = surface_rz_fourier.SurfaceRZFourier.model_validate_json(boundary_json)
        settings = forward_model.ConstellarationSettings.default_high_fidelity()
        metrics, _ = forward_model.forward_model(boundary, settings=settings)
        violations = problem._normalized_constraint_violations(metrics)
        is_feasible = problem.is_feasible(metrics)
        score = problem._score(metrics) if is_feasible else 0.0
        objective, minimize = problem.get_objective(metrics)
        return {
            "method_id": record["method_id"],
            "candidate_id": record["candidate_id"],
            "rank_before_audit": record.get("rank_before_audit"),
            "boundary_json_path": record["boundary_json_path"],
            "vmec_success": True,
            "metrics": metric_payload(metrics),
            "constraint_violations": {
                "aspect_ratio": float(violations[0]),
                "iota": float(violations[1]),
                "log10_qi": float(violations[2]),
                "mirror": float(violations[3]),
                "elongation": float(violations[4]),
                "max": float(np.max(violations)),
                "positive_max": float(np.maximum(violations, 0.0).max()),
            },
            "objective": float(objective),
            "minimize_objective": bool(minimize),
            "is_feasible": bool(is_feasible),
            "score": float(score),
            "runtime_seconds": float(time.time() - start),
            "error_type": None,
            "error_message": None,
        }
    except Exception as exc:
        return {
            "method_id": record.get("method_id"),
            "candidate_id": record.get("candidate_id"),
            "rank_before_audit": record.get("rank_before_audit"),
            "boundary_json_path": record.get("boundary_json_path"),
            "vmec_success": False,
            "metrics": {},
            "constraint_violations": {},
            "objective": None,
            "minimize_objective": False,
            "is_feasible": False,
            "score": 0.0,
            "runtime_seconds": float(time.time() - start),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def run_parallel(records: list[dict[str, Any]], workers: int) -> list[dict[str, Any]]:
    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(evaluate_candidate, record) for record in records]
        for future in as_completed(futures):
            result = future.result()
            print(json.dumps({
                "method_id": result["method_id"],
                "candidate_id": result["candidate_id"],
                "success": result["vmec_success"],
                "runtime_seconds": result["runtime_seconds"],
                "score": result["score"],
            }, sort_keys=True))
            results.append(result)
    return sorted(results, key=lambda item: (item["method_id"], item.get("rank_before_audit") or 0))


def summarize_benchmark(results: list[dict[str, Any]], workers: int) -> dict[str, Any]:
    runtimes = [item["runtime_seconds"] for item in results]
    successes = [item["vmec_success"] for item in results]
    return {
        "num_candidates": len(results),
        "num_workers": workers,
        "mean_vmec_seconds": float(np.mean(runtimes)) if runtimes else None,
        "p50_vmec_seconds": float(np.quantile(runtimes, 0.5)) if runtimes else None,
        "p90_vmec_seconds": float(np.quantile(runtimes, 0.9)) if runtimes else None,
        "vmec_success_rate": float(np.mean(successes)) if successes else None,
        "records": results,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    hardware = apply_thread_environment(config)
    ensure_output_dirs()
    workers = int(config.get("vmec_workers", 8))
    audit_budget = args.audit_budget or int(config.get("audit_budget_per_method", 10))

    benchmark_candidates = load_benchmark_candidates()
    benchmark_results = run_parallel(benchmark_candidates, workers=min(workers, len(benchmark_candidates)))
    benchmark = summarize_benchmark(benchmark_results, workers=min(workers, len(benchmark_candidates)))
    write_json(OUTPUT_DIR / "run_summary" / "vmec_benchmark.json", benchmark)

    if args.benchmark_only:
        print("Benchmark-only mode complete")
        return

    audit_records = load_candidates(audit_budget)
    audit_results = run_parallel(audit_records, workers=workers)
    write_jsonl(OUTPUT_DIR / "vmec_audit" / "audit.jsonl", audit_results)
    manifest = {
        "audit_budget_per_method": audit_budget,
        "attempted_total": len(audit_results),
        "workers": workers,
        "hardware_config": hardware,
        "success_by_method": {
            method: int(sum(1 for row in audit_results if row["method_id"] == method and row["vmec_success"]))
            for method in CANDIDATE_FILES
        },
        "attempted_by_method": {
            method: int(sum(1 for row in audit_results if row["method_id"] == method))
            for method in CANDIDATE_FILES
        },
    }
    write_json(OUTPUT_DIR / "vmec_audit" / "audit_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
