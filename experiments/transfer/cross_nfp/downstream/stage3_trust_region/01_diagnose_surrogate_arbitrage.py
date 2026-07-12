from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common_stage3 import (
    OUTPUT_DIR,
    STAGE1_OUTPUT_DIR,
    TrustDistanceModel,
    actual_log10_qi,
    apply_thread_environment,
    boundary_path_to_x,
    ensure_output_dirs,
    existing_audit_records,
    existing_candidate_records,
    feature_columns,
    load_config,
    parse_args,
    write_json,
)


METHOD_COLORS = {
    "E0-old": "#4C78A8",
    "E1-old": "#F58518",
    "E2-old": "#54A24B",
    "E3-old": "#B279A2",
    "E2-stage2": "#72B7B2",
    "E3-stage2": "#E45756",
}


def _metric(record: dict[str, Any], *path: str) -> float | None:
    value: Any = record
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _log10_gap(value: float | None) -> float | None:
    if value is None or math.isnan(value):
        return None
    return value + 4.0


def load_dataset_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(STAGE1_OUTPUT_DIR / "dataset" / "train.parquet")
    validation = pd.read_parquet(STAGE1_OUTPUT_DIR / "dataset" / "validation.parquet")
    relaxed = pd.DataFrame(
        [
            __import__("json").loads(line)
            for line in (STAGE1_OUTPUT_DIR / "dataset" / "relaxed55.jsonl").read_text().splitlines()
            if line.strip()
        ]
    )
    return train, validation, relaxed


def offline_best_log10_qi() -> float:
    values: list[float] = []
    for name in ["train", "validation", "test", "optimization_validation"]:
        path = STAGE1_OUTPUT_DIR / "dataset" / f"{name}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path, columns=["log10_qi"])
        values.extend(pd.to_numeric(frame["log10_qi"], errors="coerce").dropna().tolist())
    return float(np.min(values)) if values else float("nan")


def build_join_table(config: dict[str, Any]) -> pd.DataFrame:
    train, validation, relaxed = load_dataset_frames()
    x_cols = feature_columns(train)
    trust_cfg = config["trust_region"]
    trust_model = TrustDistanceModel(
        train[x_cols].to_numpy(dtype=np.float32),
        validation[x_cols].to_numpy(dtype=np.float32),
        relaxed[x_cols].to_numpy(dtype=np.float32),
        train_components=int(trust_cfg["train_pca_components"]),
        relaxed_components=int(trust_cfg["relaxed_pca_components"]),
        train_quantile=float(trust_cfg["train_distance_quantile"]),
        relaxed_quantile=float(trust_cfg["relaxed_distance_quantile"]),
        train_multiplier=float(trust_cfg["train_distance_multiplier"]),
        relaxed_multiplier=float(trust_cfg["relaxed_distance_multiplier"]),
    )

    candidate_by_id = {row["candidate_id"]: row for row in existing_candidate_records()}
    rows: list[dict[str, Any]] = []
    for audit in existing_audit_records():
        candidate = candidate_by_id.get(audit.get("candidate_id"))
        if candidate is None:
            continue
        pred_log10_qi = _metric(candidate, "predicted_metrics", "log10_qi")
        actual = actual_log10_qi(audit.get("metrics", {}))
        row = {
            "stage": audit.get("stage"),
            "method_id": audit.get("method_id"),
            "candidate_id": audit.get("candidate_id"),
            "rank_before_audit": audit.get("rank_before_audit"),
            "source": candidate.get("source"),
            "vmec_success": bool(audit.get("vmec_success", False)),
            "is_feasible": bool(audit.get("is_feasible", False)),
            "runtime_seconds": audit.get("runtime_seconds"),
            "error_type": audit.get("error_type"),
            "predicted_L_gradB": _metric(candidate, "predicted_metrics", "L_gradB"),
            "predicted_log10_qi": pred_log10_qi,
            "predicted_log10_qi_gap": _log10_gap(pred_log10_qi),
            "predicted_log10_qi_uncertainty": _metric(candidate, "predicted_uncertainty", "log10_qi"),
            "predicted_L_gradB_uncertainty": _metric(candidate, "predicted_uncertainty", "L_gradB"),
            "predicted_positive_violation": _metric(candidate, "support_metrics", "predicted_positive_violation"),
            "predicted_qi_violation": _metric(candidate, "support_metrics", "predicted_qi_violation"),
            "predicted_mnv": _metric(candidate, "support_metrics", "predicted_max_normalized_violation"),
            "predicted_mnv_uncertainty": _metric(
                candidate, "support_metrics", "predicted_max_normalized_violation_std"
            ),
            "actual_L_gradB": audit.get("objective"),
            "actual_log10_qi": actual,
            "actual_log10_qi_gap": _log10_gap(actual),
            "actual_positive_max_violation": _metric(audit, "constraint_violations", "positive_max"),
            "actual_qi_violation": _metric(audit, "constraint_violations", "log10_qi"),
        }
        if row["actual_log10_qi"] is not None and row["predicted_log10_qi"] is not None:
            row["log10_qi_error_actual_minus_predicted"] = (
                float(row["actual_log10_qi"]) - float(row["predicted_log10_qi"])
            )
            row["log10_qi_gap_error_actual_minus_predicted"] = (
                float(row["actual_log10_qi_gap"]) - float(row["predicted_log10_qi_gap"])
            )
        try:
            row.update(trust_model.evaluate(boundary_path_to_x(candidate["boundary_json_path"])))
        except Exception as exc:
            row["trust_distance_error"] = str(exc)
        rows.append(row)

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["stage", "method_id", "rank_before_audit"]).reset_index(drop=True)
    write_json(
        OUTPUT_DIR / "run_summary" / "trust_distance_model_stage3.json",
        trust_model.summary(),
    )
    return frame


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method_id, group in frame.groupby("method_id"):
        success = group[group["vmec_success"]].copy()
        rows.append(
            {
                "method_id": method_id,
                "audited": int(len(group)),
                "vmec_success_count": int(success.shape[0]),
                "best_actual_positive_max_violation": float(success["actual_positive_max_violation"].min())
                if not success.empty
                else np.nan,
                "best_actual_log10_qi_gap": float(success["actual_log10_qi_gap"].min())
                if not success.empty
                else np.nan,
                "median_predicted_log10_qi": float(success["predicted_log10_qi"].median())
                if not success.empty
                else np.nan,
                "median_actual_log10_qi": float(success["actual_log10_qi"].median())
                if not success.empty
                else np.nan,
                "median_log10_qi_error_actual_minus_predicted": float(
                    success["log10_qi_error_actual_minus_predicted"].median()
                )
                if not success.empty
                else np.nan,
                "median_train_distance_ratio": float(success["train_distance_ratio"].median())
                if not success.empty and "train_distance_ratio" in success
                else np.nan,
                "median_relaxed_distance_ratio": float(success["relaxed_distance_ratio"].median())
                if not success.empty and "relaxed_distance_ratio" in success
                else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("method_id")


def write_figures(frame: pd.DataFrame, offline_best: float) -> dict[str, str]:
    status: dict[str, str] = {}
    figures_dir = OUTPUT_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        status["matplotlib"] = f"unavailable: {exc}"
        return status

    success = frame[frame["vmec_success"]].dropna(subset=["predicted_log10_qi", "actual_log10_qi"])
    if not success.empty:
        fig, ax = plt.subplots(figsize=(6.6, 5.4))
        for method_id, group in success.groupby("method_id"):
            ax.scatter(
                group["predicted_log10_qi"],
                group["actual_log10_qi"],
                label=method_id,
                color=METHOD_COLORS.get(method_id, "#666666"),
                alpha=0.85,
                s=48,
            )
        lo = float(min(success["predicted_log10_qi"].min(), success["actual_log10_qi"].min(), offline_best)) - 0.1
        hi = float(max(success["predicted_log10_qi"].max(), success["actual_log10_qi"].max())) + 0.1
        ax.plot([lo, hi], [lo, hi], color="#444444", lw=1, ls="--", label="ideal")
        if np.isfinite(offline_best):
            ax.axvline(offline_best, color="#222222", lw=1, ls=":", label="offline best")
        ax.axvline(-4.0, color="#999999", lw=1, ls="-.", label="official threshold")
        ax.axhline(-4.0, color="#999999", lw=1, ls="-.")
        ax.set_xlabel("Surrogate predicted log10(qi)")
        ax.set_ylabel("VMEC measured log10(qi)")
        ax.set_title("Predicted vs measured QI")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(figures_dir / "predicted_vs_actual_log10_qi.png", dpi=180)
        plt.close(fig)
        status["predicted_vs_actual_log10_qi.png"] = "written"

        fig, ax = plt.subplots(figsize=(7, 4.5))
        for method_id, group in success.groupby("method_id"):
            ax.scatter(
                group["rank_before_audit"],
                group["actual_log10_qi_gap"],
                label=method_id,
                color=METHOD_COLORS.get(method_id, "#666666"),
                alpha=0.85,
            )
        ax.set_xlabel("Rank before audit")
        ax.set_ylabel("VMEC log10(qi) gap")
        ax.set_title("Audited QI gap by rank")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(figures_dir / "actual_qi_gap_by_rank.png", dpi=180)
        plt.close(fig)
        status["actual_qi_gap_by_rank.png"] = "written"

        fig, ax = plt.subplots(figsize=(7, 4.5))
        x_col = "predicted_log10_qi_uncertainty"
        y_col = "log10_qi_gap_error_actual_minus_predicted"
        for method_id, group in success.groupby("method_id"):
            ax.scatter(
                group[x_col],
                group[y_col],
                label=method_id,
                color=METHOD_COLORS.get(method_id, "#666666"),
                alpha=0.85,
            )
        ax.axhline(0, color="#444444", lw=1, ls="--")
        ax.set_xlabel("Surrogate log10(qi) ensemble std")
        ax.set_ylabel("Actual minus predicted log10(qi) gap")
        ax.set_title("Surrogate optimism vs uncertainty")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(figures_dir / "qi_error_vs_uncertainty.png", dpi=180)
        plt.close(fig)
        status["qi_error_vs_uncertainty.png"] = "written"

        if "train_distance_ratio" in success:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            for method_id, group in success.groupby("method_id"):
                ax.scatter(
                    group["train_distance_ratio"],
                    group[y_col],
                    label=method_id,
                    color=METHOD_COLORS.get(method_id, "#666666"),
                    alpha=0.85,
                )
            ax.axhline(0, color="#444444", lw=1, ls="--")
            ax.axvline(1, color="#999999", lw=1, ls=":")
            ax.set_xlabel("Train nearest-neighbor distance ratio")
            ax.set_ylabel("Actual minus predicted log10(qi) gap")
            ax.set_title("Surrogate optimism vs trust distance")
            ax.grid(alpha=0.25)
            ax.legend(frameon=False, fontsize=8)
            fig.tight_layout()
            fig.savefig(figures_dir / "qi_error_vs_train_distance.png", dpi=180)
            plt.close(fig)
            status["qi_error_vs_train_distance.png"] = "written"

    return status


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    hardware = apply_thread_environment(config)
    ensure_output_dirs()
    frame = build_join_table(config)
    summary = summarize(frame)
    tables_dir = OUTPUT_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(tables_dir / "stage3_prediction_audit_join.csv", index=False)
    summary.to_csv(tables_dir / "stage3_surrogate_arbitrage_summary.csv", index=False)
    offline_best = offline_best_log10_qi()
    figure_status = write_figures(frame, offline_best)
    write_json(
        OUTPUT_DIR / "run_summary" / "diagnostics_stage3.json",
        {
            "status": "complete",
            "hardware_config": hardware,
            "offline_best_log10_qi": offline_best,
            "rows": int(frame.shape[0]),
            "successful_rows": int(frame["vmec_success"].sum()) if not frame.empty else 0,
            "figures": figure_status,
        },
    )
    print(f"Wrote diagnostics for {frame.shape[0]} audited candidates")


if __name__ == "__main__":
    main()
