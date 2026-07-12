from __future__ import annotations

import json
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


def load_rows_from_path(path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing comparison summary: {path}")
    with path.open("r") as handle:
        return json.load(handle)


def row_by_name(rows: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for row in rows:
        if row.get("run") == name:
            return row
    return None


def improvement(metric: str, baseline: dict[str, Any], candidate: dict[str, Any]) -> float | None:
    base = baseline.get(metric)
    cand = candidate.get(metric)
    if base is None or cand is None:
        return None
    if "optimistic_gap" in metric:
        return abs(float(base)) - abs(float(cand))
    return float(base) - float(cand)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_output_dirs()
    suffix = str(args.run_suffix or "")
    comparison_path = OUTPUT_DIR / "run_summary" / f"pretrain_finetune_comparison{suffix}.json"
    rows = load_rows_from_path(comparison_path)
    baseline_name = f"{config.get('baseline', {}).get('run_name', 'baseline_random_15metric_nfp')}{suffix}"
    finetune_names = [
        f"{config.get('finetune', {}).get('low_lr_run_name', 'finetune_low_lr_15metric')}{suffix}",
        f"{config.get('finetune', {}).get('default_lr_run_name', 'finetune_default_lr_15metric')}{suffix}",
    ]
    baseline = row_by_name(rows, baseline_name)
    if baseline is None:
        raise SystemExit(f"Missing baseline row: {baseline_name}")

    candidates = []
    for name in finetune_names:
        row = row_by_name(rows, name)
        if row is None:
            continue
        deltas = {
            metric: improvement(metric, baseline, row)
            for metric in KEY_METRICS
        }
        key_improved = [
            metric
            for metric in ["opt_log10_qi_mae", "opt_log10_qi_optimistic_gap", "opt_violation_mae"]
            if deltas.get(metric) is not None and float(deltas[metric]) > 0.0
        ]
        candidates.append(
            {
                "run": name,
                "stage": row.get("stage"),
                "metrics": row,
                "improvements_vs_baseline": deltas,
                "key_improved": key_improved,
                "key_improved_count": len(key_improved),
            }
        )

    best = None
    if candidates:
        best = sorted(
            candidates,
            key=lambda item: (
                item["key_improved_count"],
                item["improvements_vs_baseline"].get("opt_log10_qi_mae") or float("-inf"),
            ),
            reverse=True,
        )[0]

    recommendation = "stop"
    reason = "No finetune run improved opt log10_qi MAE, optimistic gap, or opt violation MAE together."
    if best and best["key_improved_count"] >= 2:
        recommendation = "continue"
        reason = (
            f"{best['run']} improved {best['key_improved_count']} key metrics; "
            "repeat with more seeds before claiming transfer."
        )
    elif best and best["key_improved_count"] == 1:
        recommendation = "borderline"
        reason = f"{best['run']} improved only one key metric; treat as weak evidence."

    gate = {
        "baseline": baseline_name,
        "candidates": candidates,
        "best_candidate": best["run"] if best else None,
        "recommendation": recommendation,
        "reason": reason,
        "key_metrics": KEY_METRICS,
        "note": "Optimistic-gap improvement is computed on absolute gap; positive deltas are better.",
    }
    output_stem = f"pretrain_finetune_gate{suffix}"
    write_json(OUTPUT_DIR / "run_summary" / f"{output_stem}.json", gate)

    md_path = OUTPUT_DIR / "run_summary" / f"{output_stem}.md"
    lines = [
        "# Cross-Nfp Gate",
        "",
        f"Recommendation: `{recommendation}`",
        "",
        reason,
        "",
        "| run | key improved | opt log10_qi MAE delta | opt optimistic gap delta | opt violation MAE delta |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for item in candidates:
        deltas = item["improvements_vs_baseline"]
        lines.append(
            "| {run} | {count} | {qi} | {gap} | {viol} |".format(
                run=item["run"],
                count=item["key_improved_count"],
                qi=deltas.get("opt_log10_qi_mae"),
                gap=deltas.get("opt_log10_qi_optimistic_gap"),
                viol=deltas.get("opt_violation_mae"),
            )
        )
    md_path.write_text("\n".join(lines) + "\n")
    print(json.dumps(gate, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
