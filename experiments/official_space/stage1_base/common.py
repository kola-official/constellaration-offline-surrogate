from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import yaml


EXPERIMENT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = EXPERIMENT_DIR / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/quick.yaml")
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
            "dataloader_workers": 4,
            "torch_num_threads": 8,
            "gpu_devices": [0, 1],
            "parallel_ensemble": True,
        },
    )


def apply_thread_environment(config: dict[str, Any]) -> dict[str, Any]:
    hw = hardware_config(config)
    workers = int(hw.get("cpu_workers", 8))
    torch_threads = int(hw.get("torch_num_threads", workers))
    # Set defaults only when the user/session has not already made a stronger choice.
    os.environ.setdefault("OMP_NUM_THREADS", str(workers))
    os.environ.setdefault("MKL_NUM_THREADS", str(workers))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(workers))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(workers))
    os.environ.setdefault(
        "CUDA_VISIBLE_DEVICES",
        ",".join(str(device) for device in hw.get("gpu_devices", [0, 1])),
    )
    return {
        "cpu_workers": workers,
        "torch_num_threads": torch_threads,
        "dataloader_workers": int(hw.get("dataloader_workers", 4)),
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


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_json(path: str | Path) -> Any:
    with Path(path).open("r") as handle:
        return json.load(handle)


def run_text(cmd: list[str], cwd: str | Path | None = None) -> str:
    try:
        return subprocess.check_output(
            cmd,
            cwd=str(cwd) if cwd is not None else None,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def repo_root() -> Path:
    """Walk up from this experiment directory until repo markers are found."""
    markers = ("requirements.txt", "LICENSE", "CITATION.cff")
    for candidate in [EXPERIMENT_DIR, *EXPERIMENT_DIR.parents]:
        if all((candidate / name).is_file() for name in markers):
            return candidate
    raise RuntimeError(f"Could not locate repository root from {EXPERIMENT_DIR}")


def git_info() -> dict[str, str]:
    root = repo_root()
    return {
        "branch": run_text(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root),
        "commit": run_text(["git", "rev-parse", "--short", "HEAD"], cwd=root),
        "status_short": run_text(["git", "status", "--short"], cwd=root),
        "remote": run_text(["git", "remote", "get-url", "origin"], cwd=root),
    }


def active_environment() -> dict[str, str]:
    return {
        "python": run_text(["python", "--version"]),
        "executable": run_text(["python", "-c", "import sys; print(sys.executable)"]),
        "conda_prefix": os.environ.get("CONDA_PREFIX", ""),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
    }
