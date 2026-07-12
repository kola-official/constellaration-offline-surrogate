from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from common_stage4 import (
    OUTPUT_DIR,
    actual_positive_violation,
    existing_audit_records,
    load_config,
    load_jsonl,
    parse_args,
    write_json,
)


PRIMARY_METHODS = ["S4A-1", "S4A-2", "S4A-3", "S4A-4", "S4A-R"]
PAPER_PROBLEM2_NORM_CONSTR_VIOL = 0.009


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def q(values: list[float], prob: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=float), prob))


def best_positive_violation(rows: list[dict[str, Any]]) -> float | None:
    values = [
        actual_positive_violation(row)
        for row in rows
        if row.get("vmec_success") and actual_positive_violation(row) is not None
    ]
    return min(values) if values else None


def actual_l_gradb(row: dict[str, Any]) -> float | None:
    metrics = row.get("metrics") or {}
    for key in ["L_gradB", "minimum_normalized_magnetic_gradient_scale_length"]:
        value = metrics.get(key)
        if value is not None:
            return float(value)
    return None


def load_random_repeats() -> list[dict[str, Any]]:
    path = OUTPUT_DIR / "tables" / "s4a_random_audit_control.csv"
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def summarize_methods(primary: list[dict[str, Any]], baseline_best: float | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in PRIMARY_METHODS:
        group = [row for row in primary if row.get("method_id") == method]
        best = best_positive_violation(group)
        attempted = len(group)
        efficiency = None
        if baseline_best is not None and best is not None and attempted:
            efficiency = (baseline_best - best) / attempted
        rows.append(
            {
                "method_id": method,
                "attempted_vmec_calls": attempted,
                "attempted_definition": "primary audit slots, includes failed VMEC++ attempts",
                "success_count": int(sum(1 for row in group if row.get("vmec_success"))),
                "failure_count": int(sum(1 for row in group if not row.get("vmec_success"))),
                "feasible_count": int(sum(1 for row in group if row.get("is_feasible"))),
                "best_positive_max_violation": best,
                "best_score": max([float(row.get("score", 0.0) or 0.0) for row in group], default=0.0),
                "best_L_gradB_success": max(
                    [
                        actual_l_gradb(row)
                        for row in group
                        if row.get("vmec_success") and actual_l_gradb(row) is not None
                    ],
                    default=None,
                ),
                "audit_efficiency_vs_stage1_3_best": efficiency,
            }
        )
    return rows


def summarize_random_actual(audit_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_id = {row["candidate_id"]: row for row in audit_rows if row.get("candidate_id")}
    repeat_rows: list[dict[str, Any]] = []
    for repeat in load_random_repeats():
        candidate_ids = [cid for cid in repeat["candidate_ids"].split(",") if cid]
        actuals: list[float] = []
        missing = 0
        failures = 0
        for candidate_id in candidate_ids:
            row = by_id.get(candidate_id)
            if row is None:
                missing += 1
                continue
            value = actual_positive_violation(row)
            if row.get("vmec_success") and value is not None:
                actuals.append(value)
            else:
                failures += 1
        best = min(actuals) if actuals else None
        repeat_rows.append(
            {
                "repeat": repeat.get("repeat"),
                "attempted_vmec_calls": len(candidate_ids),
                "success_count": len(actuals),
                "failure_count": failures,
                "missing_truth_count": missing,
                "best_positive_max_violation": best,
                "mean_positive_max_violation_success_only": float(np.mean(actuals)) if actuals else None,
                "predicted_best_positive_violation": repeat.get("best_predicted_positive_violation"),
                "candidate_ids": repeat.get("candidate_ids"),
            }
        )
    best_values = [
        float(row["best_positive_max_violation"])
        for row in repeat_rows
        if row.get("best_positive_max_violation") not in (None, "")
    ]
    summary = {
        "repeat_count": len(repeat_rows),
        "complete_truth_repeat_count": int(sum(1 for row in repeat_rows if row["missing_truth_count"] == 0)),
        "best_positive_max_violation_median": q(best_values, 0.50),
        "best_positive_max_violation_q05": q(best_values, 0.05),
        "best_positive_max_violation_q95": q(best_values, 0.95),
        "best_positive_max_violation_min": min(best_values) if best_values else None,
        "best_positive_max_violation_max": max(best_values) if best_values else None,
    }
    return repeat_rows, summary


def percentile_against_random(method_best: float | None, random_best_values: list[float]) -> float | None:
    if method_best is None or not random_best_values:
        return None
    return float(np.mean(np.asarray(random_best_values, dtype=float) <= method_best))


def main() -> None:
    args = parse_args()
    _ = load_config(args.config)
    audit_rows = load_jsonl(OUTPUT_DIR / "vmec_audit" / "audit_stage4a.jsonl")
    primary = [row for row in audit_rows if row.get("audit_role") == "primary_method_budget"]
    prior = existing_audit_records()
    baseline_best = best_positive_violation(prior)

    method_rows = summarize_methods(primary, baseline_best)
    random_rows, random_summary = summarize_random_actual(audit_rows)
    random_best_values = [
        float(row["best_positive_max_violation"])
        for row in random_rows
        if row.get("best_positive_max_violation") not in (None, "")
    ]
    for row in method_rows:
        row["percentile_vs_random_actual_control_lower_is_better"] = percentile_against_random(
            row["best_positive_max_violation"],
            random_best_values,
        )

    efficiency_rows = []
    for row in method_rows:
        best = row["best_positive_max_violation"]
        attempted = row["attempted_vmec_calls"]
        eff_random_median = None
        if best is not None and random_summary["best_positive_max_violation_median"] is not None and attempted:
            eff_random_median = (random_summary["best_positive_max_violation_median"] - best) / attempted
        efficiency_rows.append(
            {
                "method_id": row["method_id"],
                "baseline": "Stage1-3 historical best",
                "baseline_best_positive_max_violation": baseline_best,
                "method_best_positive_max_violation": best,
                "attempted_vmec_calls": attempted,
                "audit_efficiency": row["audit_efficiency_vs_stage1_3_best"],
            }
        )
        efficiency_rows.append(
            {
                "method_id": row["method_id"],
                "baseline": "S4A-R random actual median",
                "baseline_best_positive_max_violation": random_summary["best_positive_max_violation_median"],
                "method_best_positive_max_violation": best,
                "attempted_vmec_calls": attempted,
                "audit_efficiency": eff_random_median,
            }
        )

    write_csv(
        OUTPUT_DIR / "tables" / "s4a_audit_summary.csv",
        method_rows,
        [
            "method_id",
            "attempted_vmec_calls",
            "attempted_definition",
            "success_count",
            "failure_count",
            "feasible_count",
            "best_positive_max_violation",
            "best_score",
            "best_L_gradB_success",
            "audit_efficiency_vs_stage1_3_best",
            "percentile_vs_random_actual_control_lower_is_better",
        ],
    )
    write_csv(
        OUTPUT_DIR / "tables" / "s4a_random_actual_control.csv",
        random_rows,
        [
            "repeat",
            "attempted_vmec_calls",
            "success_count",
            "failure_count",
            "missing_truth_count",
            "best_positive_max_violation",
            "mean_positive_max_violation_success_only",
            "predicted_best_positive_violation",
            "candidate_ids",
        ],
    )
    write_csv(
        OUTPUT_DIR / "tables" / "s4a_audit_efficiency.csv",
        efficiency_rows,
        [
            "method_id",
            "baseline",
            "baseline_best_positive_max_violation",
            "method_best_positive_max_violation",
            "attempted_vmec_calls",
            "audit_efficiency",
        ],
    )

    payload = {
        "stage1_3_historical_best_positive_max_violation": baseline_best,
        "paper_problem2_norm_constr_viol_reference": PAPER_PROBLEM2_NORM_CONSTR_VIOL,
        "method_summary": method_rows,
        "random_actual_control_summary": random_summary,
        "notes": {
            "audit_efficiency_formula": "(best_violation_baseline - best_violation_method) / n_vmec_calls",
            "n_vmec_calls_definition": "primary attempted VMEC++ audit slots; failed/invalid attempts are included",
            "auxiliary_random_truth": "extra VMEC++ labels used only to evaluate the repeated S4A-R random-control distribution",
        },
    }
    write_json(OUTPUT_DIR / "run_summary" / "stage4a_analysis_summary.json", payload)
    write_report(payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def fmt(value: Any, digits: int = 6) -> str:
    if value is None or value == "":
        return "NA"
    try:
        return f"{float(value):.{digits}g}"
    except Exception:
        return str(value)


def write_report(payload: dict[str, Any]) -> None:
    lines = [
        "# Stage 4A surrogate-assisted ALM-NGOpt audit report",
        "",
        "## Scope",
        "",
        "This report evaluates Stage 4A only. The surrogate is used for candidate generation and ranking; all reported constraint values in the audit tables are VMEC++ high-fidelity evaluations.",
        "",
        "The headline audit-efficiency metric is defined as",
        "",
        "`audit_efficiency = (best_violation_baseline - best_violation_method) / n_vmec_calls`,",
        "",
        "where `n_vmec_calls` is the number of primary attempted VMEC++ audit slots and includes failed or invalid attempts.",
        "",
        "## Baselines",
        "",
        f"- Stage 1-3 historical best positive max violation: {fmt(payload['stage1_3_historical_best_positive_max_violation'])}",
        f"- Paper ALM-NGOpt Problem 2 reference norm_constr_viol: {fmt(payload['paper_problem2_norm_constr_viol_reference'])}",
        "",
        "## Method Summary",
        "",
        "| method | attempted | success | feasible | best positive max violation | efficiency vs Stage 1-3 | random-control percentile |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["method_summary"]:
        lines.append(
            "| {method_id} | {attempted_vmec_calls} | {success_count} | {feasible_count} | {best} | {eff} | {pct} |".format(
                method_id=row["method_id"],
                attempted_vmec_calls=row["attempted_vmec_calls"],
                success_count=row["success_count"],
                feasible_count=row["feasible_count"],
                best=fmt(row["best_positive_max_violation"]),
                eff=fmt(row["audit_efficiency_vs_stage1_3_best"]),
                pct=fmt(row["percentile_vs_random_actual_control_lower_is_better"]),
            )
        )
    random_summary = payload["random_actual_control_summary"]
    lines.extend(
        [
            "",
            "## Random Control",
            "",
            f"- Repeated random-control draws: {random_summary['repeat_count']}",
            f"- Complete-truth random-control draws: {random_summary['complete_truth_repeat_count']}",
            f"- Best positive max violation median: {fmt(random_summary['best_positive_max_violation_median'])}",
            f"- Best positive max violation 5-95% interval: {fmt(random_summary['best_positive_max_violation_q05'])} to {fmt(random_summary['best_positive_max_violation_q95'])}",
            "",
            "## Interpretation Guardrails",
            "",
            "- A lower violation than Stage 1-3 at the same primary audit budget supports surrogate-assisted prescreening efficiency, not pure surrogate replacement.",
            "- Any feasible or near-feasible claim must be based on the VMEC++ audit rows, not on surrogate predictions.",
            "- Auxiliary random-control VMEC++ labels are reported separately from the primary per-method budget.",
            "",
        ]
    )
    path = OUTPUT_DIR / "run_summary" / "final_report_stage4a.md"
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
