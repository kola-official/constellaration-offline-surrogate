from __future__ import annotations

import json
from pathlib import Path
from typing import Any


EXPERIMENT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = EXPERIMENT_DIR / "outputs_cross_nfp"


EXPECTED_RUNS = {
    "smoke": [
        "smoke_baseline_random_15metric_nfp",
        "smoke_pretrain_15metric",
        "smoke_finetune_low_lr_15metric",
        "smoke_finetune_default_lr_15metric",
    ],
    "full": [
        "baseline_random_15metric_nfp",
        "pretrain_90k_15metric",
        "finetune_low_lr_15metric",
        "finetune_default_lr_15metric",
    ],
}


def read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    with path.open("r") as handle:
        return json.load(handle)


def tail(path: Path, lines: int = 8) -> list[str]:
    if not path.exists():
        return []
    text = path.read_text(errors="replace").splitlines()
    return text[-lines:]


def log_status(log_dir: Path) -> list[dict[str, Any]]:
    if not log_dir.exists():
        return []
    rows = []
    for path in sorted(log_dir.glob("*.log")):
        recent = tail(path, lines=5)
        rows.append(
            {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "tail": recent,
            }
        )
    return rows


def marker_status(marker_dir: Path) -> list[str]:
    if not marker_dir.exists():
        return []
    return [path.stem for path in sorted(marker_dir.glob("*.done"))]


def model_status(names: list[str]) -> list[dict[str, Any]]:
    rows = []
    for name in names:
        model_dir = OUTPUT_DIR / "models" / name
        metrics_path = model_dir / "metrics.json"
        metrics = read_json(metrics_path)
        row: dict[str, Any] = {
            "run": name,
            "metrics_exists": metrics_path.exists(),
            "member_count": len(list(model_dir.glob("member_*/model.pt"))) if model_dir.exists() else 0,
        }
        if metrics:
            row["stage"] = metrics.get("stage")
            row["members_recorded"] = metrics.get("members")
            try:
                row["test_log10_qi_mae"] = metrics["splits"]["test"]["regression"]["log10_qi"]["mae"]
            except KeyError:
                pass
            try:
                row["opt_log10_qi_mae"] = metrics["splits"]["optimization_validation"]["regression"]["log10_qi"]["mae"]
            except KeyError:
                pass
        rows.append(row)
    return rows


def seed_run_names(seed: int) -> list[str]:
    suffix = f"_seed{seed}"
    return [
        f"baseline_random_15metric_nfp{suffix}",
        f"pretrain_90k_15metric{suffix}",
        f"finetune_low_lr_15metric{suffix}",
        f"finetune_default_lr_15metric{suffix}",
    ]


def main() -> None:
    multiseed_names = []
    for seed in [0, 1, 2]:
        multiseed_names.extend(seed_run_names(seed))
    report = {
        "output_dir": str(OUTPUT_DIR),
        "markers": {
            "smoke": marker_status(OUTPUT_DIR / "run_markers" / "smoke"),
            "full": marker_status(OUTPUT_DIR / "run_markers" / "full"),
            "multiseed": marker_status(OUTPUT_DIR / "run_markers" / "multiseed"),
        },
        "logs": {
            "smoke": log_status(OUTPUT_DIR / "run_logs_smoke"),
            "full": log_status(OUTPUT_DIR / "run_logs"),
            "multiseed": log_status(OUTPUT_DIR / "run_logs_multiseed"),
        },
        "models": {
            "smoke": model_status(EXPECTED_RUNS["smoke"]),
            "full": model_status(EXPECTED_RUNS["full"]),
            "multiseed": model_status(multiseed_names),
        },
        "summaries": {
            "comparison": str(OUTPUT_DIR / "run_summary" / "pretrain_finetune_comparison.md"),
            "gate": str(OUTPUT_DIR / "run_summary" / "pretrain_finetune_gate.md"),
            "multiseed": str(OUTPUT_DIR / "run_summary" / "pretrain_finetune_multiseed_summary.md"),
            "comparison_exists": (OUTPUT_DIR / "run_summary" / "pretrain_finetune_comparison.md").exists(),
            "gate_exists": (OUTPUT_DIR / "run_summary" / "pretrain_finetune_gate.md").exists(),
            "multiseed_exists": (OUTPUT_DIR / "run_summary" / "pretrain_finetune_multiseed_summary.md").exists(),
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
