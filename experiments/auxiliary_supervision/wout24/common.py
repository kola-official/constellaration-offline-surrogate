from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml


EXPERIMENT_DIR = Path(__file__).resolve().parent


def _find_repo_root(start: Path) -> Path:
    markers = ("requirements.txt", "LICENSE", "CITATION.cff")
    for candidate in [start, *start.parents]:
        if all((candidate / name).is_file() for name in markers):
            return candidate
    raise RuntimeError(f"Could not locate repository root from {start}")


REPO_ROOT = _find_repo_root(EXPERIMENT_DIR)
STAGE1_DIR = REPO_ROOT / "experiments" / "official_space" / "stage1_base"
OUTPUT_DIR = EXPERIMENT_DIR / "outputs_wout24"

if str(STAGE1_DIR) not in sys.path:
    sys.path.append(str(STAGE1_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/quick_wout24.yaml")
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--audit-budget", type=int, default=None)
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
        "dataset",
        "models",
        "candidates",
        "vmec_audit",
        "baselines",
        "tables",
        "figures",
        "run_summary",
    ]:
        (OUTPUT_DIR / name).mkdir(parents=True, exist_ok=True)


def dataset_dir_from_config(config: dict[str, Any]) -> Path:
    value = config.get("data", {}).get("stage1_dataset_dir")
    path = Path(value) if value else STAGE1_DIR / "outputs" / "dataset"
    if not path.is_absolute():
        path = (EXPERIMENT_DIR / path).resolve()
    return path


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_json(path: str | Path) -> Any:
    with Path(path).open("r") as handle:
        return json.load(handle)
