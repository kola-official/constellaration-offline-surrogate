from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from common_stage3 import OUTPUT_DIR, actual_log10_qi, ensure_output_dirs, load_config, load_jsonl, parse_args, read_json, write_json


def _fmt(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    if math.isnan(numeric):
        return ""
    return f"{numeric:.4g}"


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    text = frame.copy()
    for column in text.columns:
        if pd.api.types.is_float_dtype(text[column]) or pd.api.types.is_integer_dtype(text[column]):
            text[column] = text[column].map(_fmt)
        else:
            text[column] = text[column].map(lambda value: "" if pd.isna(value) else str(value))
    headers = list(text.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in text.values.tolist():
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def flatten_stage3_audit(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for record in records:
        actual = actual_log10_qi(record.get("metrics", {}))
        pred_log10_qi = record.get("predicted_metrics", {}).get("log10_qi")
        row = {
            "method_id": record.get("method_id"),
            "candidate_id": record.get("candidate_id"),
            "rank_before_audit": record.get("rank_before_audit"),
            "source": record.get("source"),
            "prior_method_id": record.get("prior_method_id"),
            "reused_prior_audit": bool(record.get("reused_prior_audit")),
            "vmec_success": bool(record.get("vmec_success")),
            "is_feasible": bool(record.get("is_feasible")),
            "predicted_log10_qi": pred_log10_qi,
            "predicted_log10_qi_gap": float(pred_log10_qi) + 4.0 if pred_log10_qi is not None else np.nan,
            "predicted_qi_violation": record.get("support_metrics", {}).get("predicted_qi_violation"),
            "predicted_positive_violation": record.get("support_metrics", {}).get("predicted_positive_violation"),
            "predicted_log10_qi_uncertainty": record.get("predicted_uncertainty", {}).get("log10_qi"),
            "train_distance_ratio": record.get("trust_metrics", {}).get("train_distance_ratio"),
            "relaxed_distance_ratio": record.get("trust_metrics", {}).get("relaxed_distance_ratio"),
            "actual_L_gradB": record.get("objective"),
            "actual_log10_qi": actual,
            "actual_log10_qi_gap": actual + 4.0 if actual is not None else np.nan,
            "actual_positive_max_violation": record.get("constraint_violations", {}).get("positive_max"),
            "actual_qi_violation": record.get("constraint_violations", {}).get("log10_qi"),
            "runtime_seconds": record.get("runtime_seconds"),
            "original_runtime_seconds": record.get("original_runtime_seconds"),
            "error_type": record.get("error_type"),
        }
        if row["actual_log10_qi"] is not None and pred_log10_qi is not None:
            row["log10_qi_error_actual_minus_predicted"] = (
                float(row["actual_log10_qi"]) - float(pred_log10_qi)
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("rank_before_audit")


def summarize_stage3(audit: pd.DataFrame) -> pd.DataFrame:
    success = audit[audit["vmec_success"]]
    return pd.DataFrame(
        [
            {
                "method_id": "TR-stage3",
                "attempted": int(audit.shape[0]),
                "vmec_success_count": int(success.shape[0]),
                "vmec_success_rate": float(success.shape[0] / audit.shape[0]) if audit.shape[0] else np.nan,
                "official_feasible_count": int(audit["is_feasible"].sum()) if "is_feasible" in audit else 0,
                "reused_prior_audit_count": int(audit["reused_prior_audit"].sum())
                if "reused_prior_audit" in audit
                else 0,
                "new_vmec_count": int((~audit["reused_prior_audit"]).sum())
                if "reused_prior_audit" in audit
                else int(audit.shape[0]),
                "best_actual_positive_max_violation": float(success["actual_positive_max_violation"].min())
                if not success.empty
                else np.nan,
                "best_actual_log10_qi_gap": float(success["actual_log10_qi_gap"].min())
                if not success.empty
                else np.nan,
                "median_actual_log10_qi_gap": float(success["actual_log10_qi_gap"].median())
                if not success.empty
                else np.nan,
                "best_actual_L_gradB": float(success["actual_L_gradB"].max()) if not success.empty else np.nan,
            }
        ]
    )


def write_figures(audit: pd.DataFrame) -> dict[str, str]:
    status: dict[str, str] = {}
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"matplotlib": f"unavailable: {exc}"}
    figures_dir = OUTPUT_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not audit.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        counts = Counter(audit["source"].fillna("unknown"))
        ax.bar(list(counts.keys()), list(counts.values()), color="#4C78A8")
        ax.set_ylabel("Audited candidates")
        ax.set_title("Stage3 trust-region source mix")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(figures_dir / "trust_region_source_mix.png", dpi=180)
        plt.close(fig)
        status["trust_region_source_mix.png"] = "written"

    success = audit[audit["vmec_success"]].dropna(subset=["predicted_log10_qi", "actual_log10_qi"])
    if not success.empty:
        fig, ax = plt.subplots(figsize=(5.5, 4.8))
        ax.scatter(success["predicted_log10_qi"], success["actual_log10_qi"], color="#E45756", alpha=0.85)
        lo = min(success["predicted_log10_qi"].min(), success["actual_log10_qi"].min()) - 0.1
        hi = max(success["predicted_log10_qi"].max(), success["actual_log10_qi"].max()) + 0.1
        ax.plot([lo, hi], [lo, hi], color="#444444", lw=1, ls="--")
        ax.axvline(-4.0, color="#999999", lw=1, ls=":")
        ax.axhline(-4.0, color="#999999", lw=1, ls=":")
        ax.set_xlabel("Surrogate predicted log10(qi)")
        ax.set_ylabel("VMEC measured log10(qi)")
        ax.set_title("Trust-region predicted vs measured QI")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(figures_dir / "trust_region_predicted_vs_actual_log10_qi.png", dpi=180)
        plt.close(fig)
        status["trust_region_predicted_vs_actual_log10_qi.png"] = "written"

    return status


def write_report(
    diagnostic_summary: pd.DataFrame,
    trust_summary: pd.DataFrame,
    audit_top: pd.DataFrame,
    generation: dict[str, Any],
    manifest: dict[str, Any],
    figure_status: dict[str, str],
) -> None:
    stage2_e3 = diagnostic_summary[diagnostic_summary["method_id"] == "E3-stage2"]
    old = diagnostic_summary[diagnostic_summary["method_id"].isin(["E0-old", "E1-old", "E3-old"])]
    lines = [
        "# Stage3 Surrogate Arbitrage Report",
        "",
        "## Scope",
        "",
        "Stage3 is an independent diagnostic and trust-region ablation. Stage1 and Stage2 outputs are read-only inputs; all new artifacts are written under outputs_stage3.",
        "",
        "## Diagnostic Summary",
        "",
        dataframe_to_markdown(diagnostic_summary),
        "",
        "## Trust-Region Audit Summary",
        "",
        dataframe_to_markdown(trust_summary),
        "",
        "## Top Audited Trust-Region Candidates",
        "",
        dataframe_to_markdown(audit_top),
        "",
        "## Reading",
        "",
    ]
    if not stage2_e3.empty:
        row = stage2_e3.iloc[0]
        lines.append(
            f"- E3-stage2 median surrogate optimism on successful VMEC candidates is {row['median_log10_qi_error_actual_minus_predicted']:.4g} log10(qi) units, measured as actual minus predicted."
        )
    if not old.empty and not trust_summary.empty:
        old_best = float(old["best_actual_log10_qi_gap"].min())
        trust_best = float(trust_summary.iloc[0]["best_actual_log10_qi_gap"])
        old_best_positive = float(old["best_actual_positive_max_violation"].min())
        trust_best_positive = float(trust_summary.iloc[0]["best_actual_positive_max_violation"])
        lines.append(
            f"- Best old audited log10(qi) gap is {old_best:.4g}; best Stage3 trust-region gap is {trust_best:.4g}."
        )
        lines.append(
            f"- Best old audited positive max violation is {old_best_positive:.4g}; best Stage3 trust-region positive max violation is {trust_best_positive:.4g}."
        )
    lines.append(
        f"- Trust-region candidate generation accepted {generation.get('accepted_before_dedupe')} candidates before dedupe and wrote {generation.get('written')} ranked candidates."
    )
    lines.append(
        f"- Audit reused {manifest.get('reused_prior_audit_count')} prior VMEC records and ran {manifest.get('new_vmec_count')} new VMEC evaluations."
    )
    if manifest.get("official_feasible_count", 0) == 0:
        lines.append("- Official feasible count remains 0 in the Stage3 trust-region audit.")
    top_prior = generation.get("top_prior_methods", {})
    if top_prior:
        lines.append(f"- Top trust-region candidates by prior method: {top_prior}.")
    lines.append(
        "- Interpretation: the trust-region ablation rejects the Stage2 optimized blind spot and collapses to E0-like database candidates. It improves audited log10(qi) gap within that trusted pool, but it does not beat the old best overall positive max violation and does not produce an official feasible candidate."
    )
    lines.extend(["", "## Figures", "", str(figure_status)])
    (OUTPUT_DIR / "run_summary" / "final_report_stage3.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    _ = load_config(args.config)
    ensure_output_dirs()
    tables_dir = OUTPUT_DIR / "tables"
    diagnostic_summary = pd.read_csv(tables_dir / "stage3_surrogate_arbitrage_summary.csv")
    audit_records = load_jsonl(OUTPUT_DIR / "vmec_audit" / "audit_stage3.jsonl")
    if not audit_records:
        raise SystemExit("Missing Stage3 audit results")
    audit = flatten_stage3_audit(audit_records)
    trust_summary = summarize_stage3(audit)
    audit_top = audit[
        [
            "rank_before_audit",
            "source",
            "prior_method_id",
            "reused_prior_audit",
            "vmec_success",
            "predicted_log10_qi",
            "actual_log10_qi_gap",
            "actual_positive_max_violation",
            "train_distance_ratio",
            "relaxed_distance_ratio",
        ]
    ].copy()
    audit.to_csv(tables_dir / "stage3_trust_region_audit_flat.csv", index=False)
    trust_summary.to_csv(tables_dir / "stage3_trust_region_summary.csv", index=False)
    audit_top.to_csv(tables_dir / "stage3_trust_region_top_audit.csv", index=False)
    figure_status = write_figures(audit)
    generation = read_json(OUTPUT_DIR / "run_summary" / "trust_region_generation_stage3.json")
    manifest = read_json(OUTPUT_DIR / "vmec_audit" / "audit_manifest_stage3.json")
    write_report(diagnostic_summary, trust_summary, audit_top, generation, manifest, figure_status)
    write_json(
        OUTPUT_DIR / "run_summary" / "analysis_status_stage3.json",
        {
            "status": "complete",
            "tables": sorted(path.name for path in tables_dir.glob("stage3_*")),
            "figures": figure_status,
            "manifest": manifest,
        },
    )
    print("Wrote Stage3 analysis report")


if __name__ == "__main__":
    main()
