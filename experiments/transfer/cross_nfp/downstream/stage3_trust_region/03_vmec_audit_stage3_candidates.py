from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from common_stage3 import (
    OUTPUT_DIR,
    apply_thread_environment,
    ensure_output_dirs,
    existing_audit_records,
    load_config,
    load_jsonl,
    parse_args,
    write_json,
    write_jsonl,
)
from constellaration import forward_model, problems
from constellaration.geometry import surface_rz_fourier


def load_candidates(limit: int) -> list[dict[str, Any]]:
    records = load_jsonl(OUTPUT_DIR / "candidates" / "trust_region_candidates.jsonl")
    records = sorted(records, key=lambda item: item.get("rank_before_audit") or 10**9)
    return records[:limit]


def attach_candidate_fields(result: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    result = dict(result)
    result["method_id"] = "TR-stage3"
    result["candidate_id"] = candidate["candidate_id"]
    result["rank_before_audit"] = candidate.get("rank_before_audit")
    result["source"] = candidate.get("source")
    result["candidate_score"] = candidate.get("candidate_score")
    result["predicted_metrics"] = candidate.get("predicted_metrics", {})
    result["predicted_uncertainty"] = candidate.get("predicted_uncertainty", {})
    result["support_metrics"] = candidate.get("support_metrics", {})
    result["trust_metrics"] = candidate.get("trust_metrics", {})
    result["prior_stage"] = candidate.get("prior_stage")
    result["prior_method_id"] = candidate.get("prior_method_id")
    result["prior_source"] = candidate.get("prior_source")
    return result


def reuse_prior_audit(candidate: dict[str, Any], prior: dict[str, Any]) -> dict[str, Any]:
    result = dict(prior)
    result["reused_prior_audit"] = True
    result["reused_from_stage"] = prior.get("stage")
    result["reused_from_method_id"] = prior.get("method_id")
    result["original_runtime_seconds"] = prior.get("runtime_seconds")
    result["runtime_seconds"] = 0.0
    return attach_candidate_fields(result, candidate)


def evaluate_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    start = time.time()
    problem = problems.SimpleToBuildQIStellarator()
    try:
        boundary_json = Path(candidate["boundary_json_path"]).read_text()
        boundary = surface_rz_fourier.SurfaceRZFourier.model_validate_json(boundary_json)
        settings = forward_model.ConstellarationSettings.default_high_fidelity()
        metrics, _ = forward_model.forward_model(boundary, settings=settings)
        violations = problem._normalized_constraint_violations(metrics)
        is_feasible = problem.is_feasible(metrics)
        score = problem._score(metrics) if is_feasible else 0.0
        objective, minimize = problem.get_objective(metrics)
        result = {
            "boundary_json_path": candidate["boundary_json_path"],
            "vmec_success": True,
            "metrics": metrics.model_dump(),
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
            "reused_prior_audit": False,
        }
        return attach_candidate_fields(result, candidate)
    except Exception as exc:
        result = {
            "boundary_json_path": candidate.get("boundary_json_path"),
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
            "reused_prior_audit": False,
        }
        return attach_candidate_fields(result, candidate)


def run_new_vmec(candidates: list[dict[str, Any]], workers: int) -> list[dict[str, Any]]:
    if not candidates:
        return []
    results = []
    with ProcessPoolExecutor(max_workers=min(workers, len(candidates))) as executor:
        futures = [executor.submit(evaluate_candidate, candidate) for candidate in candidates]
        for future in as_completed(futures):
            result = future.result()
            print(
                json.dumps(
                    {
                        "method_id": result["method_id"],
                        "candidate_id": result["candidate_id"],
                        "success": result["vmec_success"],
                        "runtime_seconds": result["runtime_seconds"],
                        "score": result["score"],
                        "reused_prior_audit": result["reused_prior_audit"],
                    },
                    sort_keys=True,
                )
            )
            results.append(result)
    return results


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    hardware = apply_thread_environment(config)
    ensure_output_dirs()
    workers = int(config.get("vmec_workers", 8))
    audit_budget = args.audit_budget or int(config.get("audit_budget_per_method", 8))
    candidates = load_candidates(audit_budget)
    prior_by_id = {row["candidate_id"]: row for row in existing_audit_records()}
    reused = []
    needs_vmec = []
    for candidate in candidates:
        prior = prior_by_id.get(candidate["candidate_id"])
        if prior is None:
            needs_vmec.append(candidate)
        else:
            reused.append(reuse_prior_audit(candidate, prior))
    new_results = run_new_vmec(needs_vmec, workers=workers)
    audit_results = sorted(
        reused + new_results,
        key=lambda item: item.get("rank_before_audit") or 10**9,
    )
    write_jsonl(OUTPUT_DIR / "vmec_audit" / "audit_stage3.jsonl", audit_results)
    manifest = {
        "audit_budget": audit_budget,
        "attempted_total": len(audit_results),
        "reused_prior_audit_count": int(sum(1 for row in audit_results if row.get("reused_prior_audit"))),
        "new_vmec_count": len(new_results),
        "workers": workers,
        "hardware_config": hardware,
        "success_count": int(sum(1 for row in audit_results if row.get("vmec_success"))),
        "official_feasible_count": int(sum(1 for row in audit_results if row.get("is_feasible"))),
        "source_counts": {
            source: int(sum(1 for row in audit_results if row.get("source") == source))
            for source in sorted({str(row.get("source")) for row in audit_results})
        },
        "prior_method_counts": {
            method: int(sum(1 for row in audit_results if row.get("prior_method_id") == method))
            for method in sorted({str(row.get("prior_method_id")) for row in audit_results})
        },
    }
    write_json(OUTPUT_DIR / "vmec_audit" / "audit_manifest_stage3.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
