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
STAGE1_OUTPUT_DIR = STAGE1_DIR / "outputs"
OUTPUT_DIR = EXPERIMENT_DIR / "outputs_wout24"

if str(STAGE1_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE1_DIR))


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

CONSTRAINT_LABEL = "max_normalized_violation"

WOUT24_LABELS = [
    "wout_bmnc_pca_00",
    "wout_bmnc_pca_01",
    "wout_bmnc_pca_02",
    "wout_bmnc_pca_03",
    "wout_bsupvmnc_edge_axis_l2",
    "wout_bmnc_edge_axis_l2",
    "wout_bsubvmnc_mean",
    "wout_bsupvmnc_mean",
    "wout_rmnc_pca_00",
    "wout_rmnc_pca_01",
    "wout_zmns_pca_00",
    "wout_zmns_pca_01",
    "wout_lmns_pca_00",
    "wout_lmns_pca_01",
    "wout_iota_profile_pca_00",
    "wout_iota_profile_pca_01",
    "wout_iota_profile_slope_inner",
    "wout_iota_profile_curvature_rms",
    "wout_DMerc_min",
    "wout_DMerc_q25",
    "wout_Dgeod_q25",
    "wout_Dgeod_q75",
    "wout_jcuru_q75",
    "wout_bvco_std",
]


def parse_args(default_config: str = "configs/quick_wout24.yaml") -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--filtered-parts-dir",
        default=None,
        help="Local directory containing filtered VMEC++ wout Parquet parts.",
    )
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
    for name in ["dataset", "models", "tables", "figures", "run_summary"]:
        (OUTPUT_DIR / name).mkdir(parents=True, exist_ok=True)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def feature_columns(frame: Any) -> list[str]:
    return [str(column) for column in frame.columns if str(column).startswith("x_")]
