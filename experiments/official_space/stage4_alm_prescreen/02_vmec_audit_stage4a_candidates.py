from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from common_stage4 import (
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


CANDIDATE_FILES = {
    "S4A-1": "s4a-1_surrogate_alm_ngopt.jsonl",
    "S4A-2": "s4a-2_surrogate_alm_ngopt.jsonl",
    "S4A-3": "s4a-3_surrogate_alm_ngopt.jsonl",
    "S4A-4": "s4a-4_surrogate_alm_diverse.jsonl",
    "S4A-R": "s4a-r_random_audit_candidates.jsonl",
}


def sort_by_rank(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(records, key=lambda item: item.get("rank_before_audit") or 10**9)


def load_method_candidates(limit: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for method_id, filename in CANDIDATE_FILES.items():
        records = sort_by_rank(load_jsonl(OUTPUT_DIR / "candidates" / filename))
        for record in records[:limit]:
            row = dict(record)
            row["audit_group"] = method_id
            row["audit_role"] = "primary_method_budget"
            selected.append(row)
    return selected


def load_candidate_pool_by_id() -> dict[str, dict[str, Any]]:
    pool: dict[str, dict[str, Any]] = {}
    for filename in [
        "s4a-1_surrogate_alm_ngopt.jsonl",
        "s4a-2_surrogate_alm_ngopt.jsonl",
        "s4a-3_surrogate_alm_ngopt.jsonl",
        "s4a-4_surrogate_alm_diverse.jsonl",
        "s4a-r_random_audit_candidates.jsonl",
    ]:
        for record in load_jsonl(OUTPUT_DIR / "candidates" / filename):
            pool.setdefault(record["candidate_id"], record)
    return pool


def load_random_repeat_auxiliary(primary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    random_csv = OUTPUT_DIR / "tables" / "s4a_random_audit_control.csv"
    if not random_csv.exists():
        return []
    primary_ids = {row["candidate_id"] for row in primary}
    pool = load_candidate_pool_by_id()
    aux: list[dict[str, Any]] = []
    seen: set[str] = set()
    with random_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            for candidate_id in row["candidate_ids"].split(","):
                if not candidate_id or candidate_id in primary_ids or candidate_id in seen:
                    continue
                source = pool.get(candidate_id)
                if source is None:
                    continue
                record = dict(source)
                record["audit_group"] = "S4A-R-aux"
                record["method_id"] = "S4A-R-aux"
                record["source"] = "random_repeat_auxiliary_truth"
                record["audit_role"] = "random_repeat_auxiliary_truth"
                record["rank_before_audit"] = None
                aux.append(record)
                seen.add(candidate_id)
    return aux


def attach_candidate_fields(result: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    result = dict(result)
    result["method_id"] = candidate.get("audit_group", candidate.get("method_id"))
    result["candidate_id"] = candidate["candidate_id"]
    result["rank_before_audit"] = candidate.get("rank_before_audit")
    result["source"] = candidate.get("source")
    result["audit_role"] = candidate.get("audit_role")
    result["candidate_score"] = candidate.get("candidate_score")
    result["predicted_metrics"] = candidate.get("predicted_metrics", {})
    result["predicted_uncertainty"] = candidate.get("predicted_uncertainty", {})
    result["support_metrics"] = candidate.get("support_metrics", {})
    result["trust_metrics"] = candidate.get("trust_metrics", {})
    result["geometry_metrics"] = candidate.get("geometry_metrics", {})
    result["predicted_constraint_violations"] = candidate.get("predicted_constraint_violations", {})
    result["high_risk_surrogate_arbitrage"] = candidate.get("high_risk_surrogate_arbitrage")
    result["prior_audit_match"] = candidate.get("prior_audit_match")
    return result


def reuse_prior_audit(candidate: dict[str, Any], prior: dict[str, Any]) -> dict[str, Any]:
    result = dict(prior)
    result["reused_prior_audit"] = True
    result["reused_within_stage4a"] = False
    result["reused_from_stage"] = prior.get("stage")
    result["reused_from_method_id"] = prior.get("method_id")
    result["original_runtime_seconds"] = prior.get("runtime_seconds")
    result["runtime_seconds"] = 0.0
    result["physical_vmec_call"] = False
    return attach_candidate_fields(result, candidate)


def reuse_stage4a_evaluation(candidate: dict[str, Any], evaluated: dict[str, Any], first_use: bool) -> dict[str, Any]:
    result = dict(evaluated)
    result["reused_prior_audit"] = False
    result["reused_within_stage4a"] = not first_use
    result["original_runtime_seconds"] = evaluated.get("runtime_seconds")
    result["runtime_seconds"] = float(evaluated.get("runtime_seconds", 0.0)) if first_use else 0.0
    result["physical_vmec_call"] = bool(first_use)
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
        return {
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
        }
    except Exception as exc:
        return {
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
        }


def timeout_result(candidate: dict[str, Any], timeout_seconds: int) -> dict[str, Any]:
    return {
        "candidate_id": candidate["candidate_id"],
        "boundary_json_path": candidate.get("boundary_json_path"),
        "vmec_success": False,
        "metrics": {},
        "constraint_violations": {},
        "objective": None,
        "minimize_objective": False,
        "is_feasible": False,
        "score": 0.0,
        "runtime_seconds": float(timeout_seconds),
        "error_type": "TimeoutExpired",
        "error_message": f"VMEC++ audit exceeded {timeout_seconds} seconds",
    }


def run_candidate_subprocess(candidate: dict[str, Any], timeout_seconds: int, tmp_dir: Path, config_path: str) -> dict[str, Any]:
    input_path = tmp_dir / f"{candidate['candidate_id']}.input.json"
    output_path = tmp_dir / f"{candidate['candidate_id']}.output.json"
    input_path.write_text(json.dumps(candidate, sort_keys=True))
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--config",
        config_path,
        "--single-candidate-json",
        str(input_path),
        "--single-output-json",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=False, timeout=timeout_seconds, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return timeout_result(candidate, timeout_seconds)
    if not output_path.exists():
        result = timeout_result(candidate, timeout_seconds)
        result["error_type"] = "MissingWorkerOutput"
        result["error_message"] = "Worker process exited without writing a result JSON"
        return result
    result = json.loads(output_path.read_text())
    result["candidate_id"] = candidate["candidate_id"]
    return result


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def run_new_vmec(candidates: list[dict[str, Any]], workers: int, timeout_seconds: int, config_path: str) -> dict[str, dict[str, Any]]:
    if not candidates:
        return {}
    results: dict[str, dict[str, Any]] = {}
    tmp_dir = OUTPUT_DIR / "vmec_audit" / "stage4a_worker_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = OUTPUT_DIR / "vmec_audit" / "audit_stage4a_unique_checkpoint.jsonl"
    completed_ids = {
        row["candidate_id"]
        for row in load_jsonl(checkpoint_path)
        if row.get("candidate_id") and row.get("vmec_success") is not None
    }
    for row in load_jsonl(checkpoint_path):
        if row.get("candidate_id") in completed_ids:
            results[row["candidate_id"]] = row
    remaining = [candidate for candidate in candidates if candidate["candidate_id"] not in completed_ids]
    if not remaining:
        return results
    with ThreadPoolExecutor(max_workers=min(workers, len(remaining))) as executor:
        futures = {
            executor.submit(run_candidate_subprocess, candidate, timeout_seconds, tmp_dir, config_path): candidate
            for candidate in remaining
        }
        for future in as_completed(futures):
            candidate = futures[future]
            result = future.result()
            result["candidate_id"] = candidate["candidate_id"]
            print(
                json.dumps(
                    {
                        "candidate_id": candidate["candidate_id"],
                        "success": result["vmec_success"],
                        "runtime_seconds": result["runtime_seconds"],
                        "score": result["score"],
                    },
                    sort_keys=True,
                )
            )
            results[candidate["candidate_id"]] = result
            append_jsonl(checkpoint_path, result)
    return results


def main() -> None:
    args = parse_args()
    if args.single_candidate_json:
        candidate = json.loads(Path(args.single_candidate_json).read_text())
        result = evaluate_candidate(candidate)
        if args.single_output_json:
            Path(args.single_output_json).write_text(json.dumps(result, sort_keys=True))
        else:
            print(json.dumps(result, sort_keys=True))
        return

    config = load_config(args.config)
    hardware = apply_thread_environment(config)
    ensure_output_dirs()
    workers = int(config.get("vmec_workers", 8))
    audit_budget = args.audit_budget or int(config.get("audit_budget_per_method", 10))
    timeout_seconds = args.audit_timeout_seconds or int(config.get("vmec_timeout_seconds", 600))

    primary = load_method_candidates(audit_budget)
    auxiliary = [] if args.skip_random_aux else load_random_repeat_auxiliary(primary)
    requested = primary + auxiliary

    prior_by_id = {row["candidate_id"]: row for row in existing_audit_records() if row.get("candidate_id")}
    prior_by_id.update(
        {
            row["candidate_id"]: dict(row, stage="stage4a_existing")
            for row in load_jsonl(OUTPUT_DIR / "vmec_audit" / "audit_stage4a.jsonl")
            if row.get("candidate_id") and row.get("vmec_success") is not None
        }
    )

    reused_prior: dict[int, dict[str, Any]] = {}
    needs_by_id: dict[str, dict[str, Any]] = {}
    for idx, candidate in enumerate(requested):
        prior = prior_by_id.get(candidate["candidate_id"])
        if prior is not None:
            reused_prior[idx] = reuse_prior_audit(candidate, prior)
        else:
            needs_by_id.setdefault(candidate["candidate_id"], candidate)

    new_by_id = run_new_vmec(
        list(needs_by_id.values()),
        workers=workers,
        timeout_seconds=timeout_seconds,
        config_path=args.config,
    )
    first_use_by_id: set[str] = set()
    audit_results: list[dict[str, Any]] = []
    for idx, candidate in enumerate(requested):
        if idx in reused_prior:
            audit_results.append(reused_prior[idx])
            continue
        evaluated = new_by_id[candidate["candidate_id"]]
        first_use = candidate["candidate_id"] not in first_use_by_id
        first_use_by_id.add(candidate["candidate_id"])
        audit_results.append(reuse_stage4a_evaluation(candidate, evaluated, first_use))

    write_jsonl(OUTPUT_DIR / "vmec_audit" / "audit_stage4a.jsonl", audit_results)
    primary_results = [row for row in audit_results if row.get("audit_role") == "primary_method_budget"]
    manifest = {
        "audit_budget_per_method": audit_budget,
        "attempted_primary_slots": len(primary_results),
        "attempted_primary_slots_by_method": {
            method: int(sum(1 for row in primary_results if row.get("method_id") == method))
            for method in CANDIDATE_FILES
        },
        "attempted_slots_include_failures": True,
        "auxiliary_random_truth_slots": len([row for row in audit_results if row.get("audit_role") == "random_repeat_auxiliary_truth"]),
        "unique_requested_candidate_ids": len({row["candidate_id"] for row in requested}),
        "new_physical_vmec_calls": len(new_by_id),
        "reused_prior_audit_count": len(reused_prior),
        "reused_within_stage4a_count": int(sum(1 for row in audit_results if row.get("reused_within_stage4a"))),
        "workers": workers,
        "vmec_timeout_seconds": timeout_seconds,
        "hardware_config": hardware,
        "success_count": int(sum(1 for row in audit_results if row.get("vmec_success"))),
        "success_count_primary": int(sum(1 for row in primary_results if row.get("vmec_success"))),
        "official_feasible_count_primary": int(sum(1 for row in primary_results if row.get("is_feasible"))),
    }
    write_json(OUTPUT_DIR / "vmec_audit" / "audit_manifest_stage4a.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
