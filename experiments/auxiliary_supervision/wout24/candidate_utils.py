from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from torch import nn

from common import OUTPUT_DIR
from constellaration.generative_model import bootstrap_dataset as bd
from label_utils import CONSTRAINT_LABEL, PROBLEM2_LABELS, label_to_index


DEFAULT_FEASIBILITY_THRESHOLD = 1e-2
DEFAULT_FEASIBILITY_TEMPERATURE = 0.05


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, width),
            nn.LayerNorm(width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.LayerNorm(width),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class GroupedMultiTaskMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        problem2_dim: int,
        default_aux_dim: int,
        wout_dim: int,
        width: int,
        blocks: int,
        dropout: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Linear(input_dim, width),
            nn.LayerNorm(width),
            nn.SiLU(),
        ]
        for _ in range(blocks):
            layers.append(ResidualBlock(width, dropout))
        self.backbone = nn.Sequential(*layers)
        self.problem2_head = nn.Linear(width, problem2_dim)
        self.default_aux_head = nn.Linear(width, default_aux_dim)
        self.wout_head = nn.Linear(width, wout_dim)
        self.constraint_head = nn.Linear(width, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.backbone(x)
        return {
            "problem2": self.problem2_head(h),
            "default_aux": self.default_aux_head(h),
            "wout": self.wout_head(h),
            "constraint": self.constraint_head(h).squeeze(-1),
        }


def violation_to_infeasible_prob(
    values: np.ndarray,
    threshold: float = DEFAULT_FEASIBILITY_THRESHOLD,
    temperature: float = DEFAULT_FEASIBILITY_TEMPERATURE,
) -> np.ndarray:
    logits = (np.asarray(values, dtype=np.float32) - threshold) / temperature
    logits = np.clip(logits, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-logits))


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if column.startswith("x_")]


def load_model_bundle(device: str = "cuda:0") -> dict[str, Any]:
    model_dir = OUTPUT_DIR / "models" / "wout24_multitask"
    scalers = np.load(model_dir / "scalers.npz", allow_pickle=True)
    x_mean = scalers["x_mean"].astype(np.float32)
    x_std = scalers["x_std"].astype(np.float32)
    y_mean = scalers["problem2_mean"].astype(np.float32)
    y_std = scalers["problem2_std"].astype(np.float32)
    violation_mean = float(np.asarray(scalers["violation_mean"]).item())
    violation_std = float(np.asarray(scalers["violation_std"]).item())
    feasibility_threshold = DEFAULT_FEASIBILITY_THRESHOLD
    feasibility_temperature = DEFAULT_FEASIBILITY_TEMPERATURE
    labels = [str(x) for x in scalers["problem2_labels"]]
    default_aux_labels = [str(x) for x in scalers["default_aux_labels"]]
    wout_labels = [str(x) for x in scalers["wout_labels"]]
    feature_cols = [str(x) for x in scalers["feature_columns"]]
    label_index = label_to_index(labels)

    torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
    models = []
    for path in sorted(model_dir.glob("member_*/model.pt")):
        checkpoint = torch.load(path, map_location=torch_device)
        config = checkpoint["config"]
        model = GroupedMultiTaskMLP(
            input_dim=len(feature_cols),
            problem2_dim=len(labels),
            default_aux_dim=len(default_aux_labels),
            wout_dim=len(wout_labels),
            width=int(config["width"]),
            blocks=int(config["blocks"]),
            dropout=float(config["dropout"]),
        ).to(torch_device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        models.append(model)
    if not models:
        raise RuntimeError("No surrogate ensemble checkpoints found.")
    return {
        "models": models,
        "device": torch_device,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "violation_mean": violation_mean,
        "violation_std": violation_std,
        "feasibility_threshold": feasibility_threshold,
        "feasibility_temperature": feasibility_temperature,
        "labels": labels,
        "default_aux_labels": default_aux_labels,
        "wout_labels": wout_labels,
        "label_to_index": label_index,
        "feature_cols": feature_cols,
    }


def predict_ensemble(bundle: dict[str, Any], x_raw: np.ndarray, batch_size: int = 8192) -> dict[str, Any]:
    x_raw = np.asarray(x_raw, dtype=np.float32)
    if x_raw.ndim == 1:
        x_raw = x_raw[None, :]
    x_scaled = (x_raw - bundle["x_mean"]) / bundle["x_std"]
    tensor = torch.from_numpy(x_scaled.astype(np.float32)).to(bundle["device"])
    member_y = []
    member_v = []
    with torch.no_grad():
        for model in bundle["models"]:
            ys = []
            vs = []
            for start in range(0, tensor.shape[0], batch_size):
                pred = model(tensor[start : start + batch_size])
                ys.append(pred["problem2"].cpu().numpy())
                vs.append(pred["constraint"].cpu().numpy())
            member_y.append(np.vstack(ys) * bundle["y_std"] + bundle["y_mean"])
            member_v.append(np.concatenate(vs) * bundle["violation_std"] + bundle["violation_mean"])
    pred_stack = np.stack(member_y, axis=0)
    violation_stack = np.stack(member_v, axis=0)
    violation_mean = violation_stack.mean(axis=0)
    infeasible_prob = violation_to_infeasible_prob(
        violation_mean,
        threshold=float(bundle["feasibility_threshold"]),
        temperature=float(bundle["feasibility_temperature"]),
    )
    return {
        "mean": pred_stack.mean(axis=0),
        "std": pred_stack.std(axis=0),
        "max_normalized_violation": violation_mean,
        "max_normalized_violation_std": violation_stack.std(axis=0),
        "infeasible_prob": infeasible_prob,
        "feasible_prob": 1.0 - infeasible_prob,
        "labels": bundle["labels"],
        "label_to_index": bundle["label_to_index"],
    }


def prediction_row(prediction: dict[str, Any], row_index: int) -> dict[str, Any]:
    row = {}
    for key, value in prediction.items():
        if isinstance(value, np.ndarray) and value.shape[:1] == prediction["mean"].shape[:1]:
            row[key] = value[row_index : row_index + 1]
        else:
            row[key] = value
    return row


def prediction_value(
    prediction: dict[str, Any],
    label: str,
    row_index: int = 0,
    key: str = "mean",
) -> float:
    index = prediction["label_to_index"][label]
    return float(prediction[key][row_index, index])


def metric_dict(
    prediction_or_values: dict[str, Any] | np.ndarray,
    row_index: int = 0,
    labels: list[str] | None = None,
    include_auxiliary: bool = False,
) -> dict[str, float]:
    if isinstance(prediction_or_values, dict):
        prediction = prediction_or_values
        values = prediction["mean"][row_index]
        labels = [str(label) for label in prediction["labels"]]
    else:
        values = np.asarray(prediction_or_values)
        labels = list(labels or PROBLEM2_LABELS)
    label_index = label_to_index(labels)
    selected = labels if include_auxiliary else [label for label in PROBLEM2_LABELS if label in label_index]
    return {label: float(values[label_index[label]]) for label in selected}


def uncertainty_dict(
    prediction_or_values: dict[str, Any] | np.ndarray,
    row_index: int = 0,
    labels: list[str] | None = None,
    include_auxiliary: bool = False,
) -> dict[str, float]:
    if isinstance(prediction_or_values, dict):
        prediction = prediction_or_values
        values = prediction["std"][row_index]
        labels = [str(label) for label in prediction["labels"]]
    else:
        values = np.asarray(prediction_or_values)
        labels = list(labels or PROBLEM2_LABELS)
    label_index = label_to_index(labels)
    selected = labels if include_auxiliary else [label for label in PROBLEM2_LABELS if label in label_index]
    return {label: float(values[label_index[label]]) for label in selected}


def predicted_positive_violation(
    prediction_or_values: dict[str, Any] | np.ndarray,
    row_index: int = 0,
    labels: list[str] | None = None,
) -> float:
    violations = predicted_constraint_violations(prediction_or_values, row_index, labels)
    return float(max(max(value, 0.0) for value in violations.values()))


def predicted_constraint_violations(
    prediction_or_values: dict[str, Any] | np.ndarray,
    row_index: int = 0,
    labels: list[str] | None = None,
) -> dict[str, float]:
    if isinstance(prediction_or_values, dict):
        prediction = prediction_or_values
        aspect = prediction_value(prediction, "aspect_ratio", row_index)
        iota = prediction_value(prediction, "abs_edge_iota_over_nfp", row_index)
        log10_qi = prediction_value(prediction, "log10_qi", row_index)
        mirror = prediction_value(prediction, "edge_magnetic_mirror_ratio", row_index)
        elong = prediction_value(prediction, "max_elongation", row_index)
    else:
        values = np.asarray(prediction_or_values)
        index = label_to_index(list(labels or PROBLEM2_LABELS))
        aspect = float(values[index["aspect_ratio"]])
        iota = float(values[index["abs_edge_iota_over_nfp"]])
        log10_qi = float(values[index["log10_qi"]])
        mirror = float(values[index["edge_magnetic_mirror_ratio"]])
        elong = float(values[index["max_elongation"]])
    return {
        "aspect_ratio": float((aspect - 10.0) / 10.0),
        "iota": float((0.25 - abs(iota)) / 0.25),
        "log10_qi": float((log10_qi - (-4.0)) / 4.0),
        "mirror": float((mirror - 0.2) / 0.2),
        "elongation": float((elong - 5.0) / 5.0),
    }


def constraint_penalty(prediction: dict[str, np.ndarray], row_index: int = 0) -> float:
    metric_penalty = predicted_positive_violation(prediction, row_index)
    predicted_mnv = float(prediction["max_normalized_violation"][row_index])
    head_penalty = max(predicted_mnv - DEFAULT_FEASIBILITY_THRESHOLD, 0.0)
    return float(max(metric_penalty, head_penalty))


def constraint_support_metrics(prediction: dict[str, np.ndarray], row_index: int = 0) -> dict[str, float]:
    predicted_mnv = float(prediction["max_normalized_violation"][row_index])
    violations = predicted_constraint_violations(prediction, row_index)
    return {
        "predicted_positive_violation": constraint_penalty(prediction, row_index),
        "predicted_qi_violation": max(violations["log10_qi"], 0.0),
        "predicted_aspect_ratio_violation": max(violations["aspect_ratio"], 0.0),
        "predicted_iota_violation": max(violations["iota"], 0.0),
        "predicted_mirror_violation": max(violations["mirror"], 0.0),
        "predicted_elongation_violation": max(violations["elongation"], 0.0),
        "predicted_max_normalized_violation": predicted_mnv,
        "predicted_max_normalized_violation_std": float(prediction["max_normalized_violation_std"][row_index]),
        "predicted_infeasible_prob": float(prediction["infeasible_prob"][row_index]),
    }


def surface_json_from_x(x: np.ndarray) -> str:
    surface = bd._x_to_surface(
        np.asarray(x, dtype=float),
        max_poloidal_mode=bd.MAX_POLOIDAL_MODE,
        max_toroidal_mode=bd.MAX_TOROIDAL_MODE,
        n_field_periods=bd.N_FIELD_PERIODS,
    )
    return surface.model_dump_json()


def write_boundary(boundary_json: str, directory: Path) -> tuple[str, str]:
    directory.mkdir(parents=True, exist_ok=True)
    candidate_id = sha1_text(boundary_json)
    path = directory / f"{candidate_id}.json"
    if not path.exists():
        path.write_text(boundary_json)
    return candidate_id, str(path)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def train_bounds(train: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x = train[feature_cols].to_numpy(dtype=np.float32)
    lower = np.quantile(x, 0.005, axis=0).astype(np.float32)
    upper = np.quantile(x, 0.995, axis=0).astype(np.float32)
    same = upper <= lower
    upper[same] = lower[same] + 1e-6
    return lower, upper


class SupportModel:
    def __init__(self, train_x: np.ndarray, validation_x: np.ndarray, n_components: int = 20) -> None:
        n_components = min(n_components, train_x.shape[1], max(2, train_x.shape[0] - 1))
        self.pca = PCA(n_components=n_components, random_state=0)
        train_z = self.pca.fit_transform(train_x)
        validation_z = self.pca.transform(validation_x)
        self.nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
        self.nn.fit(train_z)
        val_dist, _ = self.nn.kneighbors(validation_z)
        self.threshold = float(np.quantile(val_dist[:, 0], 0.9))

    def penalty(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        z = self.pca.transform(x)
        dist, _ = self.nn.kneighbors(z)
        return np.maximum(dist[:, 0] - self.threshold, 0.0)


def dedupe_rank(records: list[dict[str, Any]], score_key: str, limit: int) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for record in records:
        cid = record["candidate_id"]
        if cid not in best or record[score_key] > best[cid][score_key]:
            best[cid] = record
    ranked = sorted(best.values(), key=lambda item: item[score_key], reverse=True)
    for idx, record in enumerate(ranked[:limit]):
        record["rank_before_audit"] = idx + 1
    return ranked[:limit]
