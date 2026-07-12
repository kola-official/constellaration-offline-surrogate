from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


EXPERIMENT_DIR = Path(__file__).resolve().parent


def _find_repo_root(start: Path) -> Path:
    markers = ("requirements.txt", "LICENSE", "CITATION.cff")
    for candidate in [start, *start.parents]:
        if all((candidate / name).is_file() for name in markers):
            return candidate
    raise RuntimeError(f"Could not locate repository root from {start}")


REPO_ROOT = _find_repo_root(EXPERIMENT_DIR)
STAGE1_OUTPUT_DIR = (
    REPO_ROOT / "experiments" / "official_space" / "stage1_base" / "outputs"
)
OUTPUT_DIR = EXPERIMENT_DIR / "outputs_cross_nfp"

PROBLEM2_LABELS = [
    "L_gradB",
    "aspect_ratio",
    "abs_edge_iota_over_nfp",
    "log10_qi",
    "edge_magnetic_mirror_ratio",
    "max_elongation",
]

DEFAULT_AUX_LABELS = [
    "aspect_ratio_over_edge_rotational_transform",
    "average_triangularity",
    "axis_magnetic_mirror_ratio",
    "axis_rotational_transform_over_n_field_periods",
    "edge_rotational_transform_over_n_field_periods",
    "flux_compression_in_regions_of_bad_curvature",
    "minimum_normalized_magnetic_gradient_scale_length",
    "qi",
    "vacuum_well",
]

DEFAULT_15_LABELS = PROBLEM2_LABELS + DEFAULT_AUX_LABELS
CONSTRAINT_LABEL = "max_normalized_violation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/quick_cross_nfp.yaml")
    parser.add_argument("--pretrain-dataset-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def run_text(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def import_report(names: list[str]) -> dict[str, dict[str, str | bool]]:
    report = {}
    for name in names:
        try:
            module = importlib.import_module(name)
            report[name] = {
                "ok": True,
                "version": str(getattr(module, "__version__", "")),
            }
        except Exception as exc:
            report[name] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
    return report


def load_config(path: str | Path) -> tuple[dict[str, Any], str | None]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = EXPERIMENT_DIR / config_path
    try:
        import yaml

        with config_path.open("r") as handle:
            return yaml.safe_load(handle), None
    except Exception as exc:
        return {}, f"{type(exc).__name__}: {exc}"


def resolve_path(value: str | Path | None, default: Path) -> Path:
    path = Path(value) if value else default
    if not path.is_absolute():
        path = (EXPERIMENT_DIR / path).resolve()
    return path


def apply_thread_environment(config: dict[str, Any]) -> dict[str, Any]:
    hw = config.get(
        "hardware",
        {
            "cpu_workers": 8,
            "dataloader_workers": 0,
            "torch_num_threads": 4,
            "gpu_devices": [0, 1],
            "parallel_ensemble": True,
        },
    )
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
    return {
        "cpu_workers": workers,
        "torch_num_threads": torch_threads,
        "dataloader_workers": int(hw.get("dataloader_workers", 0)),
        "gpu_devices": list(hw.get("gpu_devices", [0, 1])),
        "parallel_ensemble": bool(hw.get("parallel_ensemble", True)),
    }


def check_stage1_dataset(config: dict[str, Any]) -> dict[str, Any]:
    import pandas as pd

    dataset_dir = resolve_path(
        config.get("data", {}).get("nfp3_dataset_dir"),
        STAGE1_OUTPUT_DIR / "dataset",
    )
    report: dict[str, Any] = {"dataset_dir": str(dataset_dir), "splits": {}}
    required_splits = ["train", "validation", "test", "optimization_validation"]
    for split in required_splits:
        path = dataset_dir / f"{split}.parquet"
        split_report: dict[str, Any] = {"path": str(path), "exists": path.exists()}
        if path.exists():
            frame = pd.read_parquet(path)
            x_cols = [str(column) for column in frame.columns if str(column).startswith("x_")]
            missing_labels = [label for label in DEFAULT_15_LABELS if label not in frame.columns]
            split_report.update(
                {
                    "rows": int(len(frame)),
                    "feature_count": len(x_cols),
                    "has_constraint_label": CONSTRAINT_LABEL in frame.columns,
                    "missing_default_15_labels": missing_labels,
                    "has_feasible_label": "feasible_under_problem_2" in frame.columns,
                }
            )
        report["splits"][split] = split_report
    return report


def main() -> None:
    args = parse_args()
    config, config_error = load_config(args.config)
    if args.seed is not None:
        config["seed"] = int(args.seed)
    if args.pretrain_dataset_dir:
        config.setdefault("data", {})["pretrain_dataset_dir"] = str(args.pretrain_dataset_dir)
    hardware = apply_thread_environment(config)

    required_imports = ["yaml", "numpy", "pandas", "pyarrow", "sklearn", "torch", "constellaration"]
    optional_imports = ["datasets"]
    imports = import_report(required_imports + optional_imports)

    torch_report: dict[str, Any] = {}
    try:
        import torch

        torch.set_num_threads(int(hardware["torch_num_threads"]))
        torch_report = {
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": str(torch.version.cuda),
            "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ]
            if torch.cuda.is_available()
            else [],
        }
    except Exception as exc:
        torch_report = {"error": f"{type(exc).__name__}: {exc}"}

    dataset_report: dict[str, Any]
    try:
        dataset_report = check_stage1_dataset(config)
    except Exception as exc:
        dataset_report = {"error": f"{type(exc).__name__}: {exc}"}

    report = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "config_path": str(args.config),
        "config_error": config_error,
        "hardware_config": hardware,
        "imports": imports,
        "torch": torch_report,
        "nvidia_smi": run_text(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader",
            ]
        ),
        "stage1_dataset": dataset_report,
    }
    output = OUTPUT_DIR / "run_summary" / "environment_check.json"
    write_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)

    if config_error:
        raise SystemExit(f"Could not load config: {config_error}")
    failed_imports = [name for name in required_imports if not imports[name].get("ok")]
    if failed_imports:
        raise SystemExit(f"Missing required imports: {failed_imports}")
    if not torch_report.get("cuda_available"):
        raise SystemExit("torch.cuda.is_available() is false; expected RTX3090 CUDA environment.")
    if "error" in dataset_report:
        raise SystemExit(f"Could not inspect Nfp=3 dataset: {dataset_report['error']}")
    for split, item in dataset_report.get("splits", {}).items():
        if not item.get("exists"):
            raise SystemExit(f"Missing Nfp=3 dataset split: {split}")
        if item.get("missing_default_15_labels"):
            raise SystemExit(f"{split} is missing labels: {item['missing_default_15_labels']}")
        if int(item.get("feature_count", 0)) <= 0:
            raise SystemExit(f"{split} has no x_* feature columns.")
        if not item.get("has_constraint_label"):
            raise SystemExit(f"{split} is missing {CONSTRAINT_LABEL}.")


if __name__ == "__main__":
    main()
