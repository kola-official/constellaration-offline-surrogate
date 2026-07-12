from __future__ import annotations

import json
import math
import statistics
from typing import Any

from common_cross_nfp import OUTPUT_DIR, ensure_output_dirs, load_config, parse_args, write_json


KEY_METRICS = [
    "test_log10_qi_mae",
    "opt_log10_qi_mae",
    "test_log10_qi_optimistic_gap",
    "opt_log10_qi_optimistic_gap",
    "test_violation_mae",
    "opt_violation_mae",
]


def parse_seeds(config: dict[str, Any], value: str | None) -> list[int]:
    if value:
        return [int(item) for item in value.replace(",", " ").split()]
    return [int(seed) for seed in config.get("multiseed", {}).get("seeds", [0, 1, 2])]


def strip_seed_suffix(run: str, seed: int) -> str:
    suffix = f"_seed{seed}"
    if run.endswith(suffix):
        return run[: -len(suffix)]
    return run


def load_seed_rows(seed: int) -> list[dict[str, Any]]:
    path = OUTPUT_DIR / "run_summary" / f"pretrain_finetune_comparison_seed{seed}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing seed summary: {path}")
    with path.open("r") as handle:
        rows = json.load(handle)
    for row in rows:
        row["seed"] = seed
        row["base_run"] = strip_seed_suffix(str(row["run"]), seed)
    return rows


def mean_std(values: list[float]) -> dict[str, float | int]:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return {"count": 0, "mean": float("nan"), "std": float("nan")}
    return {
        "count": len(finite),
        "mean": float(statistics.mean(finite)),
        "std": float(statistics.stdev(finite)) if len(finite) >= 2 else 0.0,
    }


def improvement(metric: str, baseline: dict[str, Any], candidate: dict[str, Any]) -> float | None:
    base = baseline.get(metric)
    cand = candidate.get(metric)
    if base is None or cand is None:
        return None
    if "optimistic_gap" in metric:
        return abs(float(base)) - abs(float(cand))
    return float(base) - float(cand)


def fmt(summary: dict[str, float | int]) -> str:
    if int(summary["count"]) == 0:
        return "NA"
    return f"{float(summary['mean']):.6g} +/- {float(summary['std']):.3g}"


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_output_dirs()
    seeds = parse_seeds(config, args.seeds)
    all_rows: list[dict[str, Any]] = []
    for seed in seeds:
        all_rows.extend(load_seed_rows(seed))

    baseline_name = str(config.get("baseline", {}).get("run_name", "baseline_random_15metric_nfp"))
    finetune_names = [
        str(config.get("finetune", {}).get("low_lr_run_name", "finetune_low_lr_15metric")),
        str(config.get("finetune", {}).get("default_lr_run_name", "finetune_default_lr_15metric")),
    ]

    by_seed_run = {
        (int(row["seed"]), str(row["base_run"])): row
        for row in all_rows
    }
    aggregate: dict[str, Any] = {
        "seeds": seeds,
        "metric_summary": {},
        "improvement_vs_baseline": {},
    }

    base_runs = sorted({str(row["base_run"]) for row in all_rows})
    for run in base_runs:
        aggregate["metric_summary"][run] = {
            metric: mean_std([row.get(metric) for row in all_rows if row["base_run"] == run])
            for metric in KEY_METRICS
        }

    for run in finetune_names:
        per_seed = []
        for seed in seeds:
            baseline = by_seed_run.get((seed, baseline_name))
            candidate = by_seed_run.get((seed, run))
            if baseline is None or candidate is None:
                continue
            per_seed.append(
                {
                    "seed": seed,
                    **{
                        metric: improvement(metric, baseline, candidate)
                        for metric in KEY_METRICS
                    },
                }
            )
        aggregate["improvement_vs_baseline"][run] = {
            "per_seed": per_seed,
            "summary": {
                metric: mean_std([row.get(metric) for row in per_seed])
                for metric in KEY_METRICS
            },
        }

    gate_counts = {}
    for run, payload in aggregate["improvement_vs_baseline"].items():
        summary = payload["summary"]
        improved = [
            metric
            for metric in ["opt_log10_qi_mae", "opt_log10_qi_optimistic_gap", "opt_violation_mae"]
            if summary[metric]["count"] == len(seeds) and float(summary[metric]["mean"]) > 0.0
        ]
        gate_counts[run] = improved
    best_run = max(gate_counts, key=lambda name: len(gate_counts[name])) if gate_counts else None
    recommendation = "stop"
    if best_run and len(gate_counts[best_run]) >= 2:
        recommendation = "continue"
    elif best_run and len(gate_counts[best_run]) == 1:
        recommendation = "borderline"
    aggregate["gate"] = {
        "recommendation": recommendation,
        "best_run": best_run,
        "key_improved_by_run": gate_counts,
        "rule": "Continue only if mean improvement across all seeds is positive for at least two key opt-val metrics.",
    }

    output_json = OUTPUT_DIR / "run_summary" / "pretrain_finetune_multiseed_summary.json"
    write_json(output_json, aggregate)

    lines = [
        "# Cross-Nfp Multiseed Summary",
        "",
        f"Seeds: {', '.join(str(seed) for seed in seeds)}",
        "",
        f"Recommendation: `{recommendation}`",
        "",
        "## Metrics",
        "",
        "| run | opt log10_qi MAE | opt log10_qi optimistic gap | opt violation MAE |",
        "| --- | ---: | ---: | ---: |",
    ]
    for run in base_runs:
        metrics = aggregate["metric_summary"][run]
        lines.append(
            f"| {run} | {fmt(metrics['opt_log10_qi_mae'])} | "
            f"{fmt(metrics['opt_log10_qi_optimistic_gap'])} | {fmt(metrics['opt_violation_mae'])} |"
        )
    lines.extend(
        [
            "",
            "## Improvements Vs Baseline",
            "",
            "| run | opt log10_qi MAE delta | opt log10_qi optimistic gap delta | opt violation MAE delta |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for run, payload in aggregate["improvement_vs_baseline"].items():
        summary = payload["summary"]
        lines.append(
            f"| {run} | {fmt(summary['opt_log10_qi_mae'])} | "
            f"{fmt(summary['opt_log10_qi_optimistic_gap'])} | {fmt(summary['opt_violation_mae'])} |"
        )
    output_md = OUTPUT_DIR / "run_summary" / "pretrain_finetune_multiseed_summary.md"
    output_md.write_text("\n".join(lines) + "\n")
    print(output_md)


if __name__ == "__main__":
    main()
