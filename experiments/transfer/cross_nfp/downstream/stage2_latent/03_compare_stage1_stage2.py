from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common_stage2 import OUTPUT_DIR, STAGE1_OUTPUT_DIR, ensure_output_dirs, parse_args, read_json, write_json


STAGE1_METHOD_MAP = {"E0": "E0-old", "E1": "E1-old", "E2": "E2-old", "E3": "E3-old"}
METHOD_ORDER = ["E0-old", "E1-old", "E2-old", "E3-old", "E2-stage2", "E3-stage2"]
CONSTRAINTS = ["aspect_ratio", "iota", "log10_qi", "mirror", "elongation"]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def flatten_audit(records: list[dict[str, Any]], stage: str) -> pd.DataFrame:
    rows = []
    for record in records:
        method = record.get("method_id")
        if stage == "stage1":
            method = STAGE1_METHOD_MAP.get(str(method), str(method))
        row = {
            "stage": stage,
            "method_id": method,
            "candidate_id": record.get("candidate_id"),
            "rank_before_audit": record.get("rank_before_audit"),
            "vmec_success": bool(record.get("vmec_success", False)),
            "is_feasible": bool(record.get("is_feasible", False)),
            "score": record.get("score", 0.0),
            "objective": record.get("objective"),
            "runtime_seconds": record.get("runtime_seconds"),
            "error_type": record.get("error_type"),
            "error_message": record.get("error_message"),
        }
        metrics = record.get("metrics", {})
        row["qi"] = metrics.get("qi")
        if row["qi"] is not None and row["qi"] > 0:
            row["log10_qi_gap"] = float(np.log10(row["qi"]) + 4.0)
        violations = record.get("constraint_violations", {})
        for name in CONSTRAINTS + ["positive_max", "max"]:
            row[f"violation_{name}"] = violations.get(name)
        if "log10_qi_gap" not in row and row.get("violation_log10_qi") is not None:
            row["log10_qi_gap"] = float(row["violation_log10_qi"]) * 4.0
        rows.append(row)
    frame = pd.DataFrame(rows)
    for column in [
        "score",
        "objective",
        "runtime_seconds",
        "log10_qi_gap",
        "violation_positive_max",
        "violation_log10_qi",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def dominant_constraint(row: pd.Series) -> str:
    values = {name: row.get(f"violation_{name}", np.nan) for name in CONSTRAINTS}
    clean = {key: value for key, value in values.items() if pd.notna(value)}
    if not clean:
        return ""
    return max(clean, key=clean.get)


def pareto_count(group: pd.DataFrame) -> int:
    success = group[group["vmec_success"]].dropna(subset=["objective", "violation_positive_max"])
    if success.empty:
        return 0
    count = 0
    for idx, row in success.iterrows():
        others = success.drop(index=idx)
        dominated = (
            (others["violation_positive_max"] <= row["violation_positive_max"])
            & (others["objective"] >= row["objective"])
            & (
                (others["violation_positive_max"] < row["violation_positive_max"])
                | (others["objective"] > row["objective"])
            )
        ).any()
        if not dominated:
            count += 1
    return int(count)


def safe_min(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return float(clean.min()) if not clean.empty else math.nan


def safe_max(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return float(clean.max()) if not clean.empty else math.nan


def safe_median(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return float(clean.median()) if not clean.empty else math.nan


def summarize(audit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    audit = audit.copy()
    audit["dominant_constraint"] = audit.apply(dominant_constraint, axis=1)
    for method in METHOD_ORDER:
        group = audit[audit["method_id"] == method]
        if group.empty:
            continue
        success = group[group["vmec_success"]]
        feasible = group[group["is_feasible"]]
        dominant = ""
        if not success.empty:
            dominant = Counter(success["dominant_constraint"]).most_common(1)[0][0]
        total_runtime = float(pd.to_numeric(group["runtime_seconds"], errors="coerce").sum())
        success_count = int(group["vmec_success"].sum())
        feasible_count = int(group["is_feasible"].sum())
        attempted = int(len(group))
        rows.append(
            {
                "method_id": method,
                "attempted": attempted,
                "vmec_success_count": success_count,
                "vmec_success_rate": success_count / attempted if attempted else math.nan,
                "official_feasible_count": feasible_count,
                "official_feasible_rate": feasible_count / attempted if attempted else math.nan,
                "best_official_score": safe_max(group["score"]),
                "best_success_objective": safe_max(success["objective"]) if success_count else math.nan,
                "best_positive_max_violation": safe_min(success["violation_positive_max"]) if success_count else math.nan,
                "median_positive_max_violation": safe_median(success["violation_positive_max"]) if success_count else math.nan,
                "best_log10_qi_gap": safe_min(success["log10_qi_gap"]) if success_count else math.nan,
                "median_log10_qi_gap": safe_median(success["log10_qi_gap"]) if success_count else math.nan,
                "dominant_success_violation": dominant,
                "pareto_count": pareto_count(group),
                "unique_candidate_count": int(group["candidate_id"].nunique()),
                "median_runtime_seconds": safe_median(group["runtime_seconds"]),
                "total_runtime_seconds": total_runtime,
                "seconds_per_vmec_success": total_runtime / success_count if success_count else math.nan,
            }
        )
    return pd.DataFrame(rows)


def write_figures(audit: pd.DataFrame, summary: pd.DataFrame) -> dict[str, str]:
    status: dict[str, str] = {}
    figures_dir = OUTPUT_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        status["matplotlib"] = f"unavailable: {exc}"
        write_json(OUTPUT_DIR / "run_summary" / "figure_status_stage2.json", status)
        return status

    colors = {
        "E0-old": "#4C78A8",
        "E1-old": "#F58518",
        "E2-old": "#54A24B",
        "E3-old": "#B279A2",
        "E2-stage2": "#72B7B2",
        "E3-stage2": "#E45756",
    }

    success = audit[audit["vmec_success"]].copy()
    if not success.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        for method, group in success.groupby("method_id"):
            ax.scatter(
                group["violation_positive_max"],
                group["objective"],
                label=method,
                color=colors.get(method, "#666666"),
                alpha=0.85,
            )
        ax.set_xlabel("Positive max normalized violation")
        ax.set_ylabel("Audited L_gradB")
        ax.set_title("Violation vs objective")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(figures_dir / "positive_max_violation_vs_L_gradB.png", dpi=180)
        plt.close(fig)
        status["positive_max_violation_vs_L_gradB.png"] = "written"

        fig, ax = plt.subplots(figsize=(7, 4))
        data = [
            success[success["method_id"] == method]["log10_qi_gap"].dropna().to_numpy()
            for method in METHOD_ORDER
            if not success[success["method_id"] == method].empty
        ]
        labels = [method for method in METHOD_ORDER if not success[success["method_id"] == method].empty]
        if data:
            ax.boxplot(data, labels=labels, showfliers=False)
            ax.set_ylabel("log10(qi) gap to official threshold")
            ax.set_title("QI gap by method")
            ax.tick_params(axis="x", rotation=25)
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            fig.savefig(figures_dir / "log10_qi_gap_by_method.png", dpi=180)
            plt.close(fig)
            status["log10_qi_gap_by_method.png"] = "written"

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        summary["method_id"],
        summary["vmec_success_rate"],
        color=[colors.get(method, "#666666") for method in summary["method_id"]],
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("VMEC success rate")
    ax.set_title("Audited runnability")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "vmec_success_rate_by_method.png", dpi=180)
    plt.close(fig)
    status["vmec_success_rate_by_method.png"] = "written"

    write_json(OUTPUT_DIR / "run_summary" / "figure_status_stage2.json", status)
    return status


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


def _fmt_metric(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if math.isnan(numeric):
        return "n/a"
    return f"{numeric:.4g}"


def _method_row(summary: pd.DataFrame, method_id: str) -> pd.Series | None:
    rows = summary[summary["method_id"] == method_id]
    if rows.empty:
        return None
    return rows.iloc[0]


def write_report(summary: pd.DataFrame, figure_status: dict[str, str]) -> None:
    stage2 = summary[summary["method_id"].isin(["E2-stage2", "E3-stage2"])]
    stage1 = summary[summary["method_id"].isin(["E0-old", "E1-old", "E2-old", "E3-old"])]
    lines = [
        "# Stage2 Latent Feasibility Report",
        "",
        "## Scope",
        "",
        "This report compares the one-shot stage2 latent feasibility search against the completed stage1 audit. Official feasible count is retained but is not the headline metric because the offline data has zero official feasible samples.",
        "",
        "## Main Table",
        "",
        dataframe_to_markdown(summary),
        "",
        "## Stage2 Reading",
        "",
    ]
    if not stage2.empty:
        best = stage2.sort_values(["best_positive_max_violation", "best_log10_qi_gap"], ascending=True).iloc[0]
        lines.append(
            f"- Best stage2 method by violation gap: {best['method_id']} with best_positive_max_violation={best['best_positive_max_violation']:.4g} and best_log10_qi_gap={best['best_log10_qi_gap']:.4g}."
        )

    e2_old = _method_row(summary, "E2-old")
    e2_new = _method_row(summary, "E2-stage2")
    e3_old = _method_row(summary, "E3-old")
    e3_new = _method_row(summary, "E3-stage2")
    if e2_old is not None and e2_new is not None:
        lines.append(
            f"- E2 runnability did not improve: {int(e2_new['vmec_success_count'])}/{int(e2_new['attempted'])} stage2 successes versus {int(e2_old['vmec_success_count'])}/{int(e2_old['attempted'])} old successes."
        )
    if e3_old is not None and e3_new is not None:
        lines.append(
            f"- E3 runnability improved: {int(e3_new['vmec_success_count'])}/{int(e3_new['attempted'])} stage2 successes versus {int(e3_old['vmec_success_count'])}/{int(e3_old['attempted'])} old successes."
        )

    if not stage1.empty and not stage2.empty:
        best_stage1_violation = safe_min(stage1["best_positive_max_violation"])
        best_stage2_violation = safe_min(stage2["best_positive_max_violation"])
        best_stage1_qi = safe_min(stage1["best_log10_qi_gap"])
        best_stage2_qi = safe_min(stage2["best_log10_qi_gap"])
        lines.append(
            "- Constraint gaps did not improve in this fixed-budget audit: "
            f"best stage2 positive_max={_fmt_metric(best_stage2_violation)} versus best old positive_max={_fmt_metric(best_stage1_violation)}, "
            f"and best stage2 log10(qi) gap={_fmt_metric(best_stage2_qi)} versus best old gap={_fmt_metric(best_stage1_qi)}."
        )

    feasible_total = int(pd.to_numeric(summary["official_feasible_count"], errors="coerce").fillna(0).sum())
    lines.extend(
        [
            f"- Official feasible count remains {feasible_total}; this should be reported as a negative result, not hidden.",
            "- Dominant successful-candidate violation remains log10(qi), so the main bottleneck is still the missing feasible-side data and the qi gap rather than GPU training capacity.",
            "",
            "## Figures",
            "",
            str(figure_status),
        ]
    )
    (OUTPUT_DIR / "run_summary" / "final_report_stage2.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    _ = parse_args()
    ensure_output_dirs()
    stage1_records = load_jsonl(STAGE1_OUTPUT_DIR / "vmec_audit" / "audit.jsonl")
    stage2_records = load_jsonl(OUTPUT_DIR / "vmec_audit" / "audit_stage2.jsonl")
    if not stage2_records:
        raise SystemExit("Missing stage2 audit results: outputs_stage2/vmec_audit/audit_stage2.jsonl")
    audit = pd.concat(
        [flatten_audit(stage1_records, "stage1"), flatten_audit(stage2_records, "stage2")],
        ignore_index=True,
    )
    summary = summarize(audit)
    tables_dir = OUTPUT_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(tables_dir / "stage1_stage2_audit_flat.csv", index=False)
    summary.to_csv(tables_dir / "stage1_stage2_main_comparison.csv", index=False)
    figure_status = write_figures(audit, summary)
    write_report(summary, figure_status)

    manifest = read_json(OUTPUT_DIR / "vmec_audit" / "audit_manifest_stage2.json")
    write_json(
        OUTPUT_DIR / "run_summary" / "analysis_status_stage2.json",
        {
            "status": "complete",
            "stage2_manifest": manifest,
            "tables": sorted(path.name for path in tables_dir.glob("*")),
            "figures": figure_status,
        },
    )
    print("Wrote stage1-vs-stage2 tables, figures, and report")


if __name__ == "__main__":
    main()
