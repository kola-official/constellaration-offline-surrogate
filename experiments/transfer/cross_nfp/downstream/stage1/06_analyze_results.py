from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common import OUTPUT_DIR, ensure_output_dirs, parse_args, read_json, write_json


METHODS = ["E0", "E1", "E2", "E3"]
CONSTRAINT_NAMES = ["aspect_ratio", "elongation", "iota", "log10_qi", "mirror"]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_candidates() -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for path in sorted((OUTPUT_DIR / "candidates").glob("e*.jsonl")):
        for record in load_jsonl(path):
            row = {
                "candidate_id": record.get("candidate_id"),
                "method_id": record.get("method_id"),
                "source": record.get("source"),
                "candidate_score": record.get("candidate_score"),
                "rank_before_audit": record.get("rank_before_audit"),
            }
            for key, value in record.get("predicted_metrics", {}).items():
                row[f"pred_{key}"] = value
            for key, value in record.get("predicted_uncertainty", {}).items():
                row[f"pred_std_{key}"] = value
            for key, value in record.get("support_metrics", {}).items():
                row[f"support_{key}"] = value
            records.append(row)
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).drop_duplicates(["candidate_id", "method_id"], keep="first")


def load_audit() -> pd.DataFrame:
    records = load_jsonl(OUTPUT_DIR / "vmec_audit" / "audit.jsonl")
    if not records:
        raise FileNotFoundError("Missing outputs/vmec_audit/audit.jsonl")
    frame = pd.json_normalize(records, sep=".")
    frame["vmec_success"] = frame["vmec_success"].fillna(False).astype(bool)
    frame["is_feasible"] = frame["is_feasible"].fillna(False).astype(bool)
    frame["runtime_seconds"] = pd.to_numeric(frame["runtime_seconds"], errors="coerce")
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce").fillna(0.0)
    frame["objective"] = pd.to_numeric(frame["objective"], errors="coerce")
    for name in CONSTRAINT_NAMES + ["positive_max", "max"]:
        column = f"constraint_violations.{name}"
        if column not in frame.columns:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def merge_audit_candidates(audit: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return audit.copy()
    merged = audit.merge(
        candidates,
        on=["candidate_id", "method_id"],
        how="left",
        suffixes=("", "_candidate"),
    )
    if "rank_before_audit_candidate" in merged.columns:
        merged["rank_before_audit"] = merged["rank_before_audit"].fillna(
            merged["rank_before_audit_candidate"]
        )
    return merged


def dominant_constraint(row: pd.Series) -> str:
    values = {
        name: row.get(f"constraint_violations.{name}", np.nan) for name in CONSTRAINT_NAMES
    }
    clean = {key: value for key, value in values.items() if pd.notna(value)}
    if not clean:
        return ""
    return max(clean, key=clean.get)


def safe_max(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return float(clean.max()) if not clean.empty else math.nan


def safe_min(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return float(clean.min()) if not clean.empty else math.nan


def safe_median(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return float(clean.median()) if not clean.empty else math.nan


def write_main_tables(merged: pd.DataFrame) -> dict[str, pd.DataFrame]:
    merged = merged.copy()
    merged["dominant_constraint"] = merged.apply(dominant_constraint, axis=1)
    summary_rows = []
    efficiency_rows = []
    for method in METHODS:
        group = merged[merged["method_id"] == method].copy()
        successful = group[group["vmec_success"]]
        feasible = group[group["is_feasible"]]
        dominant = ""
        if not successful.empty:
            counts = Counter(successful["dominant_constraint"])
            dominant = counts.most_common(1)[0][0]
        top_rank = pd.to_numeric(group["rank_before_audit"], errors="coerce").min()
        top_rows = group[pd.to_numeric(group["rank_before_audit"], errors="coerce") == top_rank]
        top_success = bool(top_rows["vmec_success"].any()) if not top_rows.empty else False
        total_runtime = float(group["runtime_seconds"].sum()) if not group.empty else 0.0
        success_count = int(group["vmec_success"].sum())
        feasible_count = int(group["is_feasible"].sum())
        attempted = int(len(group))
        summary_rows.append(
            {
                "method_id": method,
                "attempted": attempted,
                "vmec_success_count": success_count,
                "vmec_success_rate": success_count / attempted if attempted else math.nan,
                "official_feasible_count": feasible_count,
                "official_feasible_rate": feasible_count / attempted if attempted else math.nan,
                "best_official_score": safe_max(group["score"]) if attempted else math.nan,
                "best_success_objective": safe_max(successful["objective"]) if success_count else math.nan,
                "best_positive_max_violation": safe_min(successful["constraint_violations.positive_max"])
                if success_count
                else math.nan,
                "median_positive_max_violation": safe_median(
                    successful["constraint_violations.positive_max"]
                )
                if success_count
                else math.nan,
                "dominant_success_violation": dominant,
                "top_rank_vmec_success": top_success,
                "median_runtime_seconds": safe_median(group["runtime_seconds"]),
                "total_runtime_seconds": total_runtime,
            }
        )
        efficiency_rows.append(
            {
                "method_id": method,
                "attempted": attempted,
                "vmec_success_count": success_count,
                "official_feasible_count": feasible_count,
                "total_runtime_seconds": total_runtime,
                "mean_runtime_seconds": float(group["runtime_seconds"].mean())
                if attempted
                else math.nan,
                "median_runtime_seconds": safe_median(group["runtime_seconds"]),
                "seconds_per_vmec_success": total_runtime / success_count
                if success_count
                else math.nan,
                "seconds_per_official_feasible": total_runtime / feasible_count
                if feasible_count
                else math.nan,
            }
        )
    main = pd.DataFrame(summary_rows)
    efficiency = pd.DataFrame(efficiency_rows)
    constraint_cols = [
        "method_id",
        "candidate_id",
        "source",
        "rank_before_audit",
        "vmec_success",
        "is_feasible",
        "objective",
        "score",
        "dominant_constraint",
        "constraint_violations.positive_max",
        "constraint_violations.aspect_ratio",
        "constraint_violations.elongation",
        "constraint_violations.iota",
        "constraint_violations.log10_qi",
        "constraint_violations.mirror",
        "pred_L_gradB",
        "support_predicted_max_normalized_violation",
        "support_predicted_infeasible_prob",
        "support_support_penalty",
        "runtime_seconds",
        "error_type",
        "error_message",
    ]
    for column in constraint_cols:
        if column not in merged.columns:
            merged[column] = np.nan
    constraint = merged[constraint_cols].sort_values(["method_id", "rank_before_audit"])

    tables_dir = OUTPUT_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    main.to_csv(tables_dir / "main_comparison.csv", index=False)
    efficiency.to_csv(tables_dir / "efficiency.csv", index=False)
    constraint.to_csv(tables_dir / "constraint_breakdown.csv", index=False)
    merged.to_parquet(tables_dir / "audit_with_predictions.parquet", index=False)
    return {"main": main, "efficiency": efficiency, "constraint": constraint, "merged": merged}


def write_surrogate_quality() -> pd.DataFrame:
    metrics_path = OUTPUT_DIR / "models" / "surrogate_ensemble" / "metrics.json"
    metrics = read_json(metrics_path)
    rows: list[dict[str, Any]] = []
    for split, split_metrics in metrics.get("splits", {}).items():
        for label, values in split_metrics.get("regression", {}).items():
            rows.append(
                {
                    "split": split,
                    "family": "regression",
                    "label": label,
                    "mae": values.get("mae"),
                    "rmse": values.get("rmse"),
                    "r2": values.get("r2"),
                    "status": "computed",
                }
            )
        constraint = split_metrics.get("constraint_violation", {})
        if constraint:
            rows.append(
                {
                    "split": split,
                    "family": "constraint_violation",
                    "label": constraint.get("label", "max_normalized_violation"),
                    "mae": constraint.get("mae"),
                    "rmse": constraint.get("rmse"),
                    "r2": constraint.get("r2"),
                    "threshold": constraint.get("threshold"),
                    "status": "computed",
                }
            )
        feasibility = split_metrics.get("feasibility", {})
        rows.append(
            {
                "split": split,
                "family": "feasibility",
                "label": "official_problem_2",
                "brier": feasibility.get("brier"),
                "auroc": feasibility.get("auroc"),
                "auprc": feasibility.get("auprc"),
                "positive_count": feasibility.get("positive_count"),
                "sample_count": feasibility.get("sample_count"),
                "status": feasibility.get("status", "computed"),
            }
        )
        rows.append(
            {
                "split": split,
                "family": "uncertainty",
                "label": "ensemble_mean_std_vs_abs_error",
                "correlation": split_metrics.get("uncertainty_error_correlation"),
                "status": "computed",
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT_DIR / "tables" / "surrogate_quality.csv", index=False)
    return frame


def write_reference_baseline() -> pd.DataFrame:
    path = OUTPUT_DIR / "baselines" / "paper_alm_ngopt_reference.json"
    payload = read_json(path) if path.exists() else {}
    row = {
        "baseline_id": payload.get("baseline_id", "R0_paper_ALM_NGOpt"),
        "problem": payload.get("problem", "SimpleToBuildQIStellarator"),
        "evidence_type": payload.get("evidence_type", "paper_reference"),
        "paper_reported_score": payload.get("paper_reported_score"),
        "rerun_on_gpu_server": payload.get(
            "rerun_on_gpu_server", payload.get("rerun_on_rtx3090", False)
        ),
        "comparison_rule": payload.get("comparison_rule"),
        "reason_not_rerun": payload.get("reason_not_rerun"),
        "source_table_or_section": payload.get("source", {}).get("table_or_section"),
        "source_note": payload.get("source", {}).get("quote_or_note"),
    }
    frame = pd.DataFrame([row])
    frame.to_csv(OUTPUT_DIR / "tables" / "reference_baseline.csv", index=False)
    return frame


def write_figures(merged: pd.DataFrame, main: pd.DataFrame) -> dict[str, str]:
    status: dict[str, str] = {}
    figures_dir = OUTPUT_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        status["matplotlib"] = f"unavailable: {exc}"
        write_json(OUTPUT_DIR / "run_summary" / "figure_status.json", status)
        return status

    colors = {"E0": "#4C78A8", "E1": "#F58518", "E2": "#54A24B", "E3": "#B279A2"}
    method_to_x = {method: idx for idx, method in enumerate(METHODS)}

    fig, ax = plt.subplots(figsize=(7, 4))
    for method, group in merged.groupby("method_id"):
        x = np.full(len(group), method_to_x.get(method, 0), dtype=float)
        jitter = np.linspace(-0.12, 0.12, len(group)) if len(group) > 1 else np.array([0.0])
        y = pd.to_numeric(group["objective"], errors="coerce")
        ax.scatter(
            x + jitter,
            y,
            c=colors.get(method, "#666666"),
            marker="o",
            alpha=0.8,
            label=method,
        )
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels(METHODS)
    ax.set_ylabel("Audited objective L_gradB")
    ax.set_title("Offline VMEC audit objective")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "objective_feasibility.png", dpi=180)
    plt.close(fig)
    status["objective_feasibility.png"] = "written"

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        main["method_id"],
        main["median_positive_max_violation"],
        color=[colors.get(method, "#666666") for method in main["method_id"]],
    )
    ax.set_ylabel("Median positive max violation")
    ax.set_title("Constraint violation after successful VMEC")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "constraint_violation.png", dpi=180)
    plt.close(fig)
    status["constraint_violation.png"] = "written"

    comparable = merged[
        merged["vmec_success"]
        & merged["support_predicted_max_normalized_violation"].notna()
        & merged["constraint_violations.positive_max"].notna()
    ].copy()
    if not comparable.empty:
        fig, ax = plt.subplots(figsize=(5, 5))
        for method, group in comparable.groupby("method_id"):
            ax.scatter(
                group["support_predicted_max_normalized_violation"],
                group["constraint_violations.positive_max"],
                c=colors.get(method, "#666666"),
                alpha=0.8,
                label=method,
            )
        lo = min(
            float(comparable["support_predicted_max_normalized_violation"].min()),
            float(comparable["constraint_violations.positive_max"].min()),
        )
        hi = max(
            float(comparable["support_predicted_max_normalized_violation"].max()),
            float(comparable["constraint_violations.positive_max"].max()),
        )
        ax.plot([lo, hi], [lo, hi], color="#333333", linewidth=1, linestyle="--")
        ax.set_xlabel("Predicted max normalized violation")
        ax.set_ylabel("Audited positive max violation")
        ax.set_title("Surrogate vs offline audit")
        ax.legend(frameon=False)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(figures_dir / "predicted_vs_audited_violation.png", dpi=180)
        plt.close(fig)
        status["predicted_vs_audited_violation.png"] = "written"
    write_json(OUTPUT_DIR / "run_summary" / "figure_status.json", status)
    return status


def metric_lookup(surrogate: pd.DataFrame, split: str, family: str, label: str, field: str) -> Any:
    rows = surrogate[
        (surrogate["split"] == split)
        & (surrogate["family"] == family)
        & (surrogate["label"] == label)
    ]
    if rows.empty or field not in rows.columns:
        return None
    value = rows.iloc[0][field]
    if pd.isna(value):
        return None
    return value


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    text = frame.copy()
    for column in text.columns:
        if pd.api.types.is_float_dtype(text[column]):
            text[column] = text[column].map(lambda value: "" if pd.isna(value) else f"{value:.4g}")
        else:
            text[column] = text[column].map(lambda value: "" if pd.isna(value) else str(value))
    headers = list(text.columns)
    rows = text.values.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def write_final_report(
    main: pd.DataFrame,
    surrogate: pd.DataFrame,
    reference: pd.DataFrame,
    manifest: dict[str, Any],
    figure_status: dict[str, str],
) -> None:
    best_rows = main.sort_values(
        ["best_official_score", "best_success_objective"], ascending=[False, False]
    )
    best_method = best_rows.iloc[0]["method_id"] if not best_rows.empty else ""
    l_r2 = metric_lookup(surrogate, "test", "regression", "L_gradB", "r2")
    l_mae = metric_lookup(surrogate, "test", "regression", "L_gradB", "mae")
    v_r2 = metric_lookup(
        surrogate, "test", "constraint_violation", "max_normalized_violation", "r2"
    )
    v_mae = metric_lookup(
        surrogate, "test", "constraint_violation", "max_normalized_violation", "mae"
    )
    baseline_score = reference.iloc[0].get("paper_reported_score") if not reference.empty else None
    lines = [
        "# Offline Surrogate Experiment Report",
        "",
        "## Scope",
        "",
        "This report summarizes the offline surrogate-assisted pipeline for ConStellaration Problem 2. The ALM-NGOpt baseline is recorded only as a paper reference and was not rerun under the same local compute budget.",
        "",
        "## Surrogate",
        "",
        f"- Test L_gradB: MAE={l_mae:.4f}, R2={l_r2:.4f}." if l_mae is not None else "- Test L_gradB metrics unavailable.",
        f"- Test max_normalized_violation: MAE={v_mae:.4f}, R2={v_r2:.4f}." if v_mae is not None else "- Constraint violation metrics unavailable.",
        "- Official feasibility AUROC/AUPRC is not computed because the held-out offline split has zero official feasible positives.",
        "- The feasibility signal used by E2/E3 is derived from continuous max_normalized_violation regression, not from an all-negative BCE classifier.",
        "",
        "## Audit Summary",
        "",
        f"- Attempted total: {manifest.get('attempted_total')} candidates; budget per method: {manifest.get('audit_budget_per_method')}.",
        f"- VMEC success by method: {manifest.get('success_by_method')}.",
        "- No audited candidate satisfied the official Problem 2 feasibility threshold in this run; all official scores are therefore 0.",
        f"- Best equal-budget method by official score: {best_method} (tie at score 0; inspect objective and violation tables for failure mode).",
        "",
        "## Main Comparison",
        "",
        dataframe_to_markdown(main),
        "",
        "## Baseline Reference",
        "",
        f"- Paper ALM-NGOpt reference score: {baseline_score}. External reference only; budget and evaluation flow differ from this offline run.",
        "",
        "## Failure Mode",
        "",
        "- E0 and E1 are VMEC-stable but remain outside official feasibility; log10_qi is the dominant violation among successful candidates.",
        "- E2 finds high surrogate L_gradB candidates but all top-10 candidates fail VMEC almost immediately, indicating distribution/geometry failure.",
        "- E3 keeps the relaxed55 initialization candidate VMEC-runnable, but subsequent CMA-ES drift candidates fail quickly; this supports keeping relaxed55 as a seed prior and treating post-init search conservatively.",
        "",
        "## Artifacts",
        "",
        "- tables/main_comparison.csv",
        "- tables/constraint_breakdown.csv",
        "- tables/surrogate_quality.csv",
        "- tables/efficiency.csv",
        "- tables/reference_baseline.csv",
        f"- figures: {figure_status}",
    ]
    report_path = OUTPUT_DIR / "run_summary" / "final_report.md"
    report_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    _ = parse_args()
    ensure_output_dirs()
    candidates = load_candidates()
    audit = load_audit()
    merged = merge_audit_candidates(audit, candidates)
    tables = write_main_tables(merged)
    surrogate = write_surrogate_quality()
    reference = write_reference_baseline()
    manifest = read_json(OUTPUT_DIR / "vmec_audit" / "audit_manifest.json")
    figure_status = write_figures(tables["merged"], tables["main"])
    write_final_report(tables["main"], surrogate, reference, manifest, figure_status)
    write_json(
        OUTPUT_DIR / "run_summary" / "analysis_status.json",
        {
            "status": "complete",
            "tables": sorted(path.name for path in (OUTPUT_DIR / "tables").glob("*")),
            "figures": figure_status,
        },
    )
    print("Wrote analysis tables, figures, and final_report.md")


if __name__ == "__main__":
    main()
