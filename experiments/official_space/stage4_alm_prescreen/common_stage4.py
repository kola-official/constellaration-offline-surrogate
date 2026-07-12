from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml


EXPERIMENT_DIR = Path(__file__).resolve().parent


def _find_repo_root(start: Path) -> Path:
    markers = ("requirements.txt", "LICENSE", "CITATION.cff")
    for candidate in [start, *start.parents]:
        if all((candidate / name).is_file() for name in markers):
            return candidate
    raise RuntimeError(f"Could not locate repository root from {start}")


REPO_ROOT = _find_repo_root(EXPERIMENT_DIR)
STAGE1_DIR = EXPERIMENT_DIR.parent / "stage1_base"
STAGE2_DIR = EXPERIMENT_DIR.parent / "stage2_latent"
STAGE3_DIR = EXPERIMENT_DIR.parent / "stage3_trust_region"
STAGE1_OUTPUT_DIR = STAGE1_DIR / "outputs"
STAGE2_OUTPUT_DIR = STAGE2_DIR / "outputs_stage2"
STAGE3_OUTPUT_DIR = STAGE3_DIR / "outputs_stage3"
OUTPUT_DIR = EXPERIMENT_DIR / "outputs_stage4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage4a_prescreen.yaml")
    parser.add_argument("--audit-budget", type=int, default=None)
    parser.add_argument("--audit-timeout-seconds", type=int, default=None)
    parser.add_argument("--single-candidate-json", default=None)
    parser.add_argument("--single-output-json", default=None)
    parser.add_argument("--skip-random-aux", action="store_true")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = EXPERIMENT_DIR / config_path
    with config_path.open("r") as handle:
        return yaml.safe_load(handle)


def hardware_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get(
        "hardware",
        {
            "cpu_workers": 8,
            "dataloader_workers": 0,
            "torch_num_threads": 4,
            "gpu_devices": [0, 1],
            "parallel_ensemble": True,
        },
    )


def apply_thread_environment(config: dict[str, Any]) -> dict[str, Any]:
    hw = hardware_config(config)
    workers = int(hw.get("cpu_workers", 8))
    torch_threads = int(hw.get("torch_num_threads", workers))
    os.environ.setdefault("OMP_NUM_THREADS", str(workers))
    os.environ.setdefault("MKL_NUM_THREADS", str(workers))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(workers))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(workers))
    os.environ.setdefault(
        "CUDA_VISIBLE_DEVICES",
        ",".join(str(device) for device in hw.get("gpu_devices", [0, 1])),
    )
    try:
        import torch

        torch.set_num_threads(torch_threads)
    except Exception:
        pass
    return {
        "cpu_workers": workers,
        "torch_num_threads": torch_threads,
        "dataloader_workers": int(hw.get("dataloader_workers", 0)),
        "gpu_devices": list(hw.get("gpu_devices", [0, 1])),
        "parallel_ensemble": bool(hw.get("parallel_ensemble", True)),
    }


def ensure_output_dirs() -> None:
    for name in [
        "candidates",
        "vmec_audit",
        "calibration",
        "models",
        "tables",
        "figures",
        "run_summary",
    ]:
        (OUTPUT_DIR / name).mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "candidates" / "boundaries").mkdir(parents=True, exist_ok=True)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def feature_columns(frame: Any) -> list[str]:
    return [column for column in frame.columns if str(column).startswith("x_")]


def actual_positive_violation(row: dict[str, Any]) -> float | None:
    violations = row.get("constraint_violations") or {}
    value = violations.get("positive_max")
    return float(value) if value is not None else None


def existing_audit_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in load_jsonl(STAGE1_OUTPUT_DIR / "vmec_audit" / "audit.jsonl"):
        row = dict(row)
        row["stage"] = "stage1"
        row["stage_method_id"] = row.get("method_id")
        records.append(row)
    for row in load_jsonl(STAGE2_OUTPUT_DIR / "vmec_audit" / "audit_stage2.jsonl"):
        row = dict(row)
        row["stage"] = "stage2"
        row["stage_method_id"] = row.get("method_id")
        records.append(row)
    for row in load_jsonl(STAGE3_OUTPUT_DIR / "vmec_audit" / "audit_stage3.jsonl"):
        row = dict(row)
        row["stage"] = "stage3"
        row["stage_method_id"] = row.get("method_id")
        records.append(row)
    return records


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        return out if np.isfinite(out) else default
    except Exception:
        return default
