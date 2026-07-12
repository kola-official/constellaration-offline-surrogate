from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
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
OUTPUT_DIR = EXPERIMENT_DIR / "outputs_cross_nfp"

if str(STAGE1_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE1_DIR))

from label_utils import CONSTRAINT_LABEL, PROBLEM2_LABELS, regression_label_weights  # noqa: E402


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

DEFAULT_15_LABELS = list(PROBLEM2_LABELS) + DEFAULT_AUX_LABELS


def parse_args(default_config: str = "configs/quick_cross_nfp.yaml") -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--stage", choices=["pretrain", "baseline", "finetune"], default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--pretrain-model-dir", default=None)
    parser.add_argument("--pretrain-dataset-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-suffix", default="")
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--load-heads", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.seed is not None:
        config["seed"] = int(args.seed)
    if args.pretrain_dataset_dir:
        config.setdefault("data", {})["pretrain_dataset_dir"] = str(args.pretrain_dataset_dir)
    return config


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = EXPERIMENT_DIR / config_path
    with config_path.open("r") as handle:
        return yaml.safe_load(handle)


def resolve_path(value: str | Path | None, default: Path) -> Path:
    path = Path(value) if value else default
    if not path.is_absolute():
        path = (EXPERIMENT_DIR / path).resolve()
    return path


def ensure_output_dirs() -> None:
    for name in ["dataset_pretrain", "models", "tables", "run_summary"]:
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


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def add_problem2_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["L_gradB"] = df["minimum_normalized_magnetic_gradient_scale_length"]
    df["abs_edge_iota_over_nfp"] = df[
        "edge_rotational_transform_over_n_field_periods"
    ].abs()
    df["log10_qi"] = np.log10(pd.to_numeric(df["qi"], errors="coerce"))
    df["aspect_ratio_violation"] = (df["aspect_ratio"] - 10.0) / 10.0
    df["iota_violation"] = (0.25 - df["abs_edge_iota_over_nfp"]) / 0.25
    df["log10_qi_violation"] = (df["log10_qi"] - (-4.0)) / 4.0
    df["mirror_violation"] = (df["edge_magnetic_mirror_ratio"] - 0.2) / 0.2
    df["elongation_violation"] = (df["max_elongation"] - 5.0) / 5.0
    violation_cols = [
        "aspect_ratio_violation",
        "iota_violation",
        "log10_qi_violation",
        "mirror_violation",
        "elongation_violation",
    ]
    df["max_normalized_violation"] = df[violation_cols].max(axis=1)
    df["positive_max_normalized_violation"] = df[violation_cols].clip(lower=0.0).max(axis=1)
    df["feasible_under_problem_2"] = df["max_normalized_violation"] <= 1e-2
    df["score_problem_2"] = np.where(
        df["feasible_under_problem_2"], df["L_gradB"] / 20.0, 0.0
    )
    return df


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [str(column) for column in frame.columns if str(column).startswith("x_")]


def finite_mask(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    values = frame[columns].to_numpy(dtype=np.float32)
    return np.isfinite(values).all(axis=1)


def nfp_values(frame: pd.DataFrame, default_nfp: int = 3) -> np.ndarray:
    for column in ["nfp", "boundary.n_field_periods", "n_field_periods"]:
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce").fillna(default_nfp).to_numpy(dtype=np.float32)
    return np.full(len(frame), float(default_nfp), dtype=np.float32)


def append_nfp_condition(
    x: np.ndarray,
    frame: pd.DataFrame,
    config: dict[str, Any],
    default_nfp: int = 3,
) -> np.ndarray:
    feature_config = config.get("features", {})
    if not bool(feature_config.get("include_nfp", True)):
        return x.astype(np.float32)
    scale = float(feature_config.get("nfp_scale", 3.0))
    if scale <= 0:
        raise ValueError("features.nfp_scale must be positive.")
    condition = (nfp_values(frame, default_nfp=default_nfp) / scale).reshape(-1, 1)
    return np.concatenate([x, condition.astype(np.float32)], axis=1).astype(np.float32)


def conditioned_feature_columns(x_cols: list[str], config: dict[str, Any]) -> list[str]:
    if bool(config.get("features", {}).get("include_nfp", True)):
        scale = float(config.get("features", {}).get("nfp_scale", 3.0))
        suffix = int(scale) if scale.is_integer() else scale
        return x_cols + [f"nfp_over_{suffix}"]
    return x_cols


def default_label_weights(config: dict[str, Any]) -> np.ndarray:
    return regression_label_weights(DEFAULT_15_LABELS, config.get("surrogate", {}))
