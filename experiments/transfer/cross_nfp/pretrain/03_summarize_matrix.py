from __future__ import annotations

import json
from pathlib import Path

from common_cross_nfp import OUTPUT_DIR, ensure_output_dirs, load_config, parse_args, write_json


def metric_at(metrics: dict, split: str, group: str, label: str | None, field: str) -> float | None:
    try:
        if label is None:
            return metrics["splits"][split][group][field]
        return metrics["splits"][split][group][label][field]
    except KeyError:
        return None


def load_metrics(model_name: str) -> dict | None:
    path = OUTPUT_DIR / "models" / model_name / "metrics.json"
    if not path.exists():
        return None
    with path.open("r") as handle:
        return json.load(handle)


def fmt(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.6g}"


def with_suffix(name: str, suffix: str) -> str:
    return f"{name}{suffix}" if suffix else name


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_output_dirs()
    suffix = str(args.run_suffix or "")
    names = [
        with_suffix(str(config.get("baseline", {}).get("run_name", "baseline_random_15metric_nfp")), suffix),
        with_suffix(str(config.get("pretrain", {}).get("run_name", "pretrain_90k_15metric")), suffix),
        with_suffix(str(config.get("finetune", {}).get("low_lr_run_name", "finetune_low_lr_15metric")), suffix),
        with_suffix(str(config.get("finetune", {}).get("default_lr_run_name", "finetune_default_lr_15metric")), suffix),
    ]
    rows = []
    for name in names:
        metrics = load_metrics(name)
        if metrics is None:
            continue
        row = {
            "run": name,
            "stage": metrics.get("stage", ""),
            "test_log10_qi_mae": metric_at(metrics, "test", "regression", "log10_qi", "mae"),
            "opt_log10_qi_mae": metric_at(metrics, "optimization_validation", "regression", "log10_qi", "mae"),
            "test_log10_qi_optimistic_gap": metric_at(
                metrics, "test", "regression", "log10_qi", "optimistic_gap_true_minus_pred"
            ),
            "opt_log10_qi_optimistic_gap": metric_at(
                metrics,
                "optimization_validation",
                "regression",
                "log10_qi",
                "optimistic_gap_true_minus_pred",
            ),
            "test_violation_mae": metric_at(metrics, "test", "constraint_violation", None, "mae"),
            "opt_violation_mae": metric_at(metrics, "optimization_validation", "constraint_violation", None, "mae"),
        }
        rows.append(row)

    output_stem = f"pretrain_finetune_comparison{suffix}"
    output = OUTPUT_DIR / "run_summary" / f"{output_stem}.md"
    lines = [
        "# Cross-Nfp Pretrain/Finetune Comparison",
        "",
        "| run | stage | test log10_qi MAE | opt log10_qi MAE | test log10_qi optimistic gap | opt log10_qi optimistic gap | test violation MAE | opt violation MAE |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {run} | {stage} | {test_log10_qi_mae} | {opt_log10_qi_mae} | "
            "{test_log10_qi_optimistic_gap} | {opt_log10_qi_optimistic_gap} | "
            "{test_violation_mae} | {opt_violation_mae} |".format(
                run=row["run"],
                stage=row["stage"],
                test_log10_qi_mae=fmt(row["test_log10_qi_mae"]),
                opt_log10_qi_mae=fmt(row["opt_log10_qi_mae"]),
                test_log10_qi_optimistic_gap=fmt(row["test_log10_qi_optimistic_gap"]),
                opt_log10_qi_optimistic_gap=fmt(row["opt_log10_qi_optimistic_gap"]),
                test_violation_mae=fmt(row["test_violation_mae"]),
                opt_violation_mae=fmt(row["opt_violation_mae"]),
            )
        )
    lines.append("")
    lines.append("Positive optimistic gap means the model predicted lower log10_qi or violation than the audited value.")
    output.write_text("\n".join(lines) + "\n")
    write_json(OUTPUT_DIR / "run_summary" / f"{output_stem}.json", rows)
    print(output)


if __name__ == "__main__":
    main()
