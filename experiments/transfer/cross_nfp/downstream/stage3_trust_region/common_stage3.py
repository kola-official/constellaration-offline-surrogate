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
STAGE1_DIR = EXPERIMENT_DIR.parent / "stage1"
STAGE2_DIR = EXPERIMENT_DIR.parent / "stage2_latent"
STAGE1_OUTPUT_DIR = STAGE1_DIR / "outputs"
STAGE2_OUTPUT_DIR = STAGE2_DIR / "outputs_stage2"
OUTPUT_DIR = EXPERIMENT_DIR / "outputs_stage3"


STAGE1_CANDIDATE_FILES = {
    "E0-old": "e0_dataset_static.jsonl",
    "E1-old": "e1_relaxed55_gmm.jsonl",
    "E2-old": "e2_surrogate_only_cmaes.jsonl",
    "E3-old": "e3_csa_cmaes_full.jsonl",
}

STAGE2_CANDIDATE_FILES = {
    "E2-stage2": "e2_latent_feasibility_cmaes.jsonl",
    "E3-stage2": "e3_latent_conservative_cmaes.jsonl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/quick_stage3.yaml")
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
    for name in ["candidates", "vmec_audit", "tables", "figures", "run_summary"]:
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


def boundary_json_to_x(boundary_json: str) -> np.ndarray:
    from constellaration.generative_model import bootstrap_dataset as bd
    from constellaration.geometry import surface_rz_fourier

    boundary = surface_rz_fourier.SurfaceRZFourier.model_validate_json(boundary_json)
    surface = surface_rz_fourier.set_max_mode_numbers(
        boundary,
        max_poloidal_mode=bd.MAX_POLOIDAL_MODE,
        max_toroidal_mode=bd.MAX_TOROIDAL_MODE,
    )
    return np.concatenate(
        [
            surface.r_cos.ravel()[surface.max_toroidal_mode + 1 :],
            surface.z_sin.ravel()[surface.max_toroidal_mode + 1 :],
        ]
    ).astype(np.float32)


def boundary_path_to_x(path: str | Path) -> np.ndarray:
    return boundary_json_to_x(Path(path).read_text())


def actual_log10_qi(metrics: dict[str, Any]) -> float | None:
    qi = metrics.get("qi")
    if qi is None or float(qi) <= 0.0:
        return None
    return float(np.log10(float(qi)))


def normalize_stage1_method(method_id: str) -> str:
    return {
        "E0": "E0-old",
        "E1": "E1-old",
        "E2": "E2-old",
        "E3": "E3-old",
    }.get(method_id, method_id)


def existing_candidate_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for method_id, filename in STAGE1_CANDIDATE_FILES.items():
        path = STAGE1_OUTPUT_DIR / "candidates" / filename
        for row in load_jsonl(path):
            row = dict(row)
            row["stage"] = "stage1"
            row["original_method_id"] = row.get("method_id")
            row["method_id"] = method_id
            records.append(row)
    for method_id, filename in STAGE2_CANDIDATE_FILES.items():
        path = STAGE2_OUTPUT_DIR / "candidates" / filename
        for row in load_jsonl(path):
            row = dict(row)
            row["stage"] = "stage2"
            row["original_method_id"] = row.get("method_id")
            row["method_id"] = method_id
            records.append(row)
    return records


def existing_audit_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in load_jsonl(STAGE1_OUTPUT_DIR / "vmec_audit" / "audit.jsonl"):
        row = dict(row)
        row["stage"] = "stage1"
        row["method_id"] = normalize_stage1_method(str(row.get("method_id")))
        records.append(row)
    for row in load_jsonl(STAGE2_OUTPUT_DIR / "vmec_audit" / "audit_stage2.jsonl"):
        row = dict(row)
        row["stage"] = "stage2"
        records.append(row)
    return records


class TrustDistanceModel:
    def __init__(
        self,
        train_x: np.ndarray,
        validation_x: np.ndarray,
        relaxed_x: np.ndarray,
        train_components: int,
        relaxed_components: int,
        train_quantile: float,
        relaxed_quantile: float,
        train_multiplier: float,
        relaxed_multiplier: float,
    ) -> None:
        from sklearn.decomposition import PCA
        from sklearn.neighbors import NearestNeighbors

        train_components = min(train_components, train_x.shape[1], train_x.shape[0] - 1)
        relaxed_components = min(relaxed_components, relaxed_x.shape[1], relaxed_x.shape[0] - 1)
        self.train_pca = PCA(n_components=train_components, random_state=0)
        train_z = self.train_pca.fit_transform(train_x)
        val_z = self.train_pca.transform(validation_x)
        self.train_nn = NearestNeighbors(n_neighbors=1)
        self.train_nn.fit(train_z)
        val_dist, _ = self.train_nn.kneighbors(val_z)
        self.train_threshold = float(np.quantile(val_dist[:, 0], train_quantile) * train_multiplier)

        self.relaxed_pca = PCA(n_components=relaxed_components, random_state=0)
        relaxed_z = self.relaxed_pca.fit_transform(relaxed_x)
        self.relaxed_nn = NearestNeighbors(n_neighbors=1)
        self.relaxed_nn.fit(relaxed_z)
        relaxed_internal = NearestNeighbors(n_neighbors=2)
        relaxed_internal.fit(relaxed_z)
        relaxed_dist, _ = relaxed_internal.kneighbors(relaxed_z)
        self.relaxed_threshold = float(
            np.quantile(relaxed_dist[:, 1], relaxed_quantile) * relaxed_multiplier
        )
        self.train_explained_variance = float(np.sum(self.train_pca.explained_variance_ratio_))
        self.relaxed_explained_variance = float(np.sum(self.relaxed_pca.explained_variance_ratio_))

    def evaluate(self, x: np.ndarray) -> dict[str, float]:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        train_z = self.train_pca.transform(x)
        train_dist, _ = self.train_nn.kneighbors(train_z)
        relaxed_z = self.relaxed_pca.transform(x)
        relaxed_dist, _ = self.relaxed_nn.kneighbors(relaxed_z)
        train_distance = float(train_dist[0, 0])
        relaxed_distance = float(relaxed_dist[0, 0])
        return {
            "train_nn_distance": train_distance,
            "train_distance_threshold": self.train_threshold,
            "train_distance_ratio": train_distance / max(self.train_threshold, 1e-12),
            "relaxed_nn_distance": relaxed_distance,
            "relaxed_distance_threshold": self.relaxed_threshold,
            "relaxed_distance_ratio": relaxed_distance / max(self.relaxed_threshold, 1e-12),
        }

    def summary(self) -> dict[str, float]:
        return {
            "train_threshold": self.train_threshold,
            "relaxed_threshold": self.relaxed_threshold,
            "train_explained_variance": self.train_explained_variance,
            "relaxed_explained_variance": self.relaxed_explained_variance,
        }
