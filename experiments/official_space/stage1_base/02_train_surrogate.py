from __future__ import annotations

import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn import metrics as sk_metrics
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from common import (
    OUTPUT_DIR,
    apply_thread_environment,
    ensure_output_dirs,
    load_config,
    parse_args,
    read_json,
    write_json,
)
from label_utils import CONSTRAINT_LABEL, PROBLEM2_LABELS, regression_label_weights, unique_existing


FEASIBILITY_THRESHOLD = 1e-2
FEASIBILITY_TEMPERATURE = 0.05


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


class MultiTaskMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, width: int, blocks: int, dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Linear(input_dim, width),
            nn.LayerNorm(width),
            nn.SiLU(),
        ]
        for _ in range(blocks):
            layers.append(ResidualBlock(width, dropout))
        self.backbone = nn.Sequential(*layers)
        self.regression_head = nn.Linear(width, output_dim)
        self.constraint_head = nn.Linear(width, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        return self.regression_head(h), self.constraint_head(h).squeeze(-1)


def violation_to_infeasible_prob(values: np.ndarray) -> np.ndarray:
    logits = (values - FEASIBILITY_THRESHOLD) / FEASIBILITY_TEMPERATURE
    logits = np.clip(logits, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-logits))


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if column.startswith("x_")]


def finite_mask(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    values = df[columns].to_numpy(dtype=np.float32)
    return np.isfinite(values).all(axis=1)


def load_regression_labels(dataset_dir: Path, train_frame: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    configured = config.get("surrogate", {}).get("regression_labels")
    if configured and configured != "auto":
        labels = [str(label) for label in configured]
    else:
        manifest_path = dataset_dir / "split_manifest.json"
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            labels = [str(label) for label in manifest.get("regression_labels", [])]
        else:
            labels = PROBLEM2_LABELS
    labels = unique_existing(labels, train_frame.columns)
    missing_problem2 = [label for label in PROBLEM2_LABELS if label not in labels and label in train_frame.columns]
    labels = missing_problem2 + labels
    if not labels:
        raise ValueError("No regression labels were found in the training parquet.")
    return labels


def weighted_huber_loss(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    loss = F.huber_loss(pred, target, reduction="none")
    return (loss * weights).sum(dim=1).mean() / weights.sum().clamp_min(1e-8)


def load_arrays(config: dict[str, Any]) -> dict[str, Any]:
    dataset_dir = OUTPUT_DIR / "dataset"
    frames = {
        name: pd.read_parquet(dataset_dir / f"{name}.parquet")
        for name in ["train", "validation", "test", "optimization_validation"]
    }
    x_cols = feature_columns(frames["train"])
    regression_labels = load_regression_labels(dataset_dir, frames["train"], config)
    required = x_cols + regression_labels + [CONSTRAINT_LABEL, "feasible_under_problem_2"]
    for name, frame in list(frames.items()):
        mask = finite_mask(frame, required)
        frames[name] = frame.loc[mask].reset_index(drop=True)

    x_mean = frames["train"][x_cols].to_numpy(dtype=np.float32).mean(axis=0)
    x_std = frames["train"][x_cols].to_numpy(dtype=np.float32).std(axis=0)
    x_std = np.where(x_std < 1e-8, 1.0, x_std).astype(np.float32)

    y_mean = frames["train"][regression_labels].to_numpy(dtype=np.float32).mean(axis=0)
    y_std = frames["train"][regression_labels].to_numpy(dtype=np.float32).std(axis=0)
    y_std = np.where(y_std < 1e-8, 1.0, y_std).astype(np.float32)
    violation_mean = float(frames["train"][CONSTRAINT_LABEL].to_numpy(dtype=np.float32).mean())
    violation_std = float(frames["train"][CONSTRAINT_LABEL].to_numpy(dtype=np.float32).std())
    if violation_std < 1e-8:
        violation_std = 1.0

    arrays: dict[str, Any] = {
        "frames": frames,
        "x_cols": x_cols,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "violation_mean": violation_mean,
        "violation_std": violation_std,
        "labels": regression_labels,
        "label_weights": regression_label_weights(regression_labels, config.get("surrogate", {})),
    }
    for name, frame in frames.items():
        x = frame[x_cols].to_numpy(dtype=np.float32)
        y = frame[regression_labels].to_numpy(dtype=np.float32)
        violation_raw = frame[CONSTRAINT_LABEL].to_numpy(dtype=np.float32)
        feasible = frame["feasible_under_problem_2"].astype(float).to_numpy(dtype=np.float32)
        arrays[name] = {
            "x": ((x - x_mean) / x_std).astype(np.float32),
            "y": ((y - y_mean) / y_std).astype(np.float32),
            "y_raw": y.astype(np.float32),
            "violation": ((violation_raw - violation_mean) / violation_std).astype(np.float32),
            "violation_raw": violation_raw.astype(np.float32),
            "feasible": feasible,
            "sample_id": frame["sample_id"].astype(str).to_numpy()
            if "sample_id" in frame.columns
            else np.array([str(i) for i in range(len(frame))]),
        }
    return arrays


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    violation: np.ndarray,
    batch_size: int,
    shuffle: bool,
    workers: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(x),
        torch.from_numpy(y),
        torch.from_numpy(violation),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )


def train_one_member(
    member_id: int,
    device_index: int,
    config: dict[str, Any],
    arrays: dict[str, Any],
    output_dir: str,
) -> dict[str, Any]:
    torch_threads = int(config["hardware"].get("torch_num_threads", 8))
    torch.set_num_threads(torch_threads)
    seed = int(config.get("seed", 0)) + member_id * 1009
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{device_index}")
    else:
        device = torch.device("cpu")

    surrogate_config = config["surrogate"]
    model = MultiTaskMLP(
        input_dim=arrays["train"]["x"].shape[1],
        output_dim=len(arrays["labels"]),
        width=int(surrogate_config["width"]),
        blocks=int(surrogate_config["blocks"]),
        dropout=float(surrogate_config["dropout"]),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(surrogate_config["learning_rate"]),
        weight_decay=float(surrogate_config["weight_decay"]),
    )
    constraint_loss = nn.HuberLoss()
    label_weights = torch.from_numpy(arrays["label_weights"]).to(device)

    batch_size = int(surrogate_config["batch_size"])
    workers = int(config["hardware"].get("dataloader_workers", 4))
    train_loader = make_loader(
        arrays["train"]["x"],
        arrays["train"]["y"],
        arrays["train"]["violation"],
        batch_size=batch_size,
        shuffle=True,
        workers=workers,
    )
    validation_loader = make_loader(
        arrays["validation"]["x"],
        arrays["validation"]["y"],
        arrays["validation"]["violation"],
        batch_size=batch_size,
        shuffle=False,
        workers=workers,
    )

    best_validation = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    patience = int(surrogate_config["patience"])
    max_epochs = int(surrogate_config["max_epochs"])

    for epoch in range(max_epochs):
        model.train()
        for xb, yb, vb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            vb = vb.to(device, non_blocking=True)
            pred_y, pred_v = model(xb)
            loss = weighted_huber_loss(pred_y, yb, label_weights)
            loss = loss + 0.2 * constraint_loss(pred_v, vb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        model.eval()
        losses = []
        with torch.no_grad():
            for xb, yb, vb in validation_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                vb = vb.to(device, non_blocking=True)
                pred_y, pred_v = model(xb)
                loss = weighted_huber_loss(pred_y, yb, label_weights) + 0.2 * constraint_loss(pred_v, vb)
                losses.append(float(loss.detach().cpu()))
        validation_loss = float(np.mean(losses))
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    member_dir = Path(output_dir) / f"member_{member_id:02d}"
    member_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "member_id": member_id,
            "seed": seed,
            "device_index": device_index,
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation,
            "config": surrogate_config,
            "regression_labels": arrays["labels"],
        },
        member_dir / "model.pt",
    )
    return {
        "member_id": member_id,
        "seed": seed,
        "device": str(device),
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation,
    }


def predict_member(model_path: Path, arrays: dict[str, Any], split: str, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    checkpoint = torch.load(model_path, map_location=device)
    surrogate_config = checkpoint["config"]
    model = MultiTaskMLP(
        input_dim=arrays[split]["x"].shape[1],
        output_dim=len(arrays["labels"]),
        width=int(surrogate_config["width"]),
        blocks=int(surrogate_config["blocks"]),
        dropout=float(surrogate_config["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    x = torch.from_numpy(arrays[split]["x"]).to(device)
    preds_y: list[np.ndarray] = []
    preds_v: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, x.shape[0], 8192):
            pred_y, pred_v = model(x[start : start + 8192])
            preds_y.append(pred_y.cpu().numpy())
            preds_v.append(pred_v.cpu().numpy())
    return np.vstack(preds_y), np.concatenate(preds_v)


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
) -> dict[str, dict[str, float]]:
    result = {}
    for idx, label in enumerate(labels):
        true = y_true[:, idx]
        pred = y_pred[:, idx]
        result[label] = {
            "mae": float(sk_metrics.mean_absolute_error(true, pred)),
            "rmse": float(np.sqrt(sk_metrics.mean_squared_error(true, pred))),
            "r2": float(sk_metrics.r2_score(true, pred)),
        }
    return result


def write_predictions(
    arrays: dict[str, Any],
    split: str,
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
    violation_mean: np.ndarray,
    violation_std: np.ndarray,
    infeasible_prob: np.ndarray,
    output_path: Path,
) -> None:
    frame = pd.DataFrame({"sample_id": arrays[split]["sample_id"]})
    y_true = arrays[split]["y_raw"]
    for idx, label in enumerate(arrays["labels"]):
        frame[f"true_{label}"] = y_true[:, idx]
        frame[f"pred_{label}"] = pred_mean[:, idx]
        frame[f"std_{label}"] = pred_std[:, idx]
    frame["true_feasible"] = arrays[split]["feasible"]
    frame[f"true_{CONSTRAINT_LABEL}"] = arrays[split]["violation_raw"]
    frame[f"pred_{CONSTRAINT_LABEL}"] = violation_mean
    frame[f"std_{CONSTRAINT_LABEL}"] = violation_std
    frame["pred_infeasible_prob"] = infeasible_prob
    frame["pred_feasible_prob"] = 1.0 - infeasible_prob
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)


def ensemble_evaluate(arrays: dict[str, Any], model_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_paths = sorted(model_dir.glob("member_*/model.pt"))
    metrics: dict[str, Any] = {"members": len(model_paths), "splits": {}}
    y_mean = arrays["y_mean"]
    y_std = arrays["y_std"]
    violation_mean_scale = arrays["violation_mean"]
    violation_std_scale = arrays["violation_std"]
    for split in ["test", "optimization_validation"]:
        member_y = []
        member_v = []
        for path in model_paths:
            pred_y_z, pred_v_z = predict_member(path, arrays, split, device)
            member_y.append(pred_y_z * y_std + y_mean)
            member_v.append(pred_v_z * violation_std_scale + violation_mean_scale)
        pred_stack = np.stack(member_y, axis=0)
        violation_stack = np.stack(member_v, axis=0)
        pred_mean = pred_stack.mean(axis=0)
        pred_std = pred_stack.std(axis=0)
        violation_pred_mean = violation_stack.mean(axis=0)
        violation_pred_std = violation_stack.std(axis=0)
        infeasible_prob = violation_to_infeasible_prob(violation_pred_mean)
        feasible_prob = 1.0 - infeasible_prob
        y_true = arrays[split]["y_raw"]
        violation_true = arrays[split]["violation_raw"]
        split_metrics: dict[str, Any] = {
            "regression": regression_metrics(y_true, pred_mean, arrays["labels"]),
            "constraint_violation": {
                "label": CONSTRAINT_LABEL,
                "mae": float(sk_metrics.mean_absolute_error(violation_true, violation_pred_mean)),
                "rmse": float(np.sqrt(sk_metrics.mean_squared_error(violation_true, violation_pred_mean))),
                "r2": float(sk_metrics.r2_score(violation_true, violation_pred_mean)),
                "threshold": FEASIBILITY_THRESHOLD,
                "infeasible_probability_temperature": FEASIBILITY_TEMPERATURE,
            },
            "feasibility": {},
        }
        feasible_true = arrays[split]["feasible"]
        if len(np.unique(feasible_true)) >= 2:
            split_metrics["feasibility"] = {
                "auroc": float(sk_metrics.roc_auc_score(feasible_true, feasible_prob)),
                "auprc": float(sk_metrics.average_precision_score(feasible_true, feasible_prob)),
                "brier": float(sk_metrics.brier_score_loss(feasible_true, feasible_prob)),
            }
        else:
            split_metrics["feasibility"] = {
                "status": "not_computed_single_class",
                "positive_count": int(feasible_true.sum()),
                "sample_count": int(feasible_true.shape[0]),
                "brier": float(sk_metrics.brier_score_loss(feasible_true, feasible_prob)),
            }
        abs_error = np.mean(np.abs(y_true - pred_mean), axis=1)
        avg_std = np.mean(pred_std, axis=1)
        if np.std(avg_std) > 1e-12 and np.std(abs_error) > 1e-12:
            corr = float(np.corrcoef(avg_std, abs_error)[0, 1])
        else:
            corr = float("nan")
        split_metrics["uncertainty_error_correlation"] = corr
        metrics["splits"][split] = split_metrics
        write_predictions(
            arrays,
            split,
            pred_mean,
            pred_std,
            violation_pred_mean,
            violation_pred_std,
            infeasible_prob,
            model_dir / f"predictions_{split}.parquet",
        )
    return metrics


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    hardware = apply_thread_environment(config)
    config["hardware"] = hardware
    ensure_output_dirs()

    model_dir = OUTPUT_DIR / "models" / "surrogate_ensemble"
    model_dir.mkdir(parents=True, exist_ok=True)

    arrays = load_arrays(config)
    np.savez(
        model_dir / "scalers.npz",
        x_mean=arrays["x_mean"],
        x_std=arrays["x_std"],
        y_mean=arrays["y_mean"],
        y_std=arrays["y_std"],
        violation_mean=np.array(arrays["violation_mean"], dtype=np.float32),
        violation_std=np.array(arrays["violation_std"], dtype=np.float32),
        violation_label=np.array(CONSTRAINT_LABEL),
        feasibility_threshold=np.array(FEASIBILITY_THRESHOLD, dtype=np.float32),
        feasibility_temperature=np.array(FEASIBILITY_TEMPERATURE, dtype=np.float32),
        labels=np.array(arrays["labels"]),
        label_weights=arrays["label_weights"],
        feature_columns=np.array(arrays["x_cols"]),
    )

    gpu_devices = hardware.get("gpu_devices", [0])
    ensemble_members = int(config["surrogate"]["ensemble_members"])
    output_dir = str(model_dir)
    member_results = []
    max_workers = len(gpu_devices) if hardware.get("parallel_ensemble", True) else 1
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for member_id in range(ensemble_members):
            device_index = int(gpu_devices[member_id % len(gpu_devices)])
            futures.append(
                executor.submit(
                    train_one_member,
                    member_id,
                    device_index,
                    config,
                    {key: value for key, value in arrays.items() if key != "frames"},
                    output_dir,
                )
            )
        for future in as_completed(futures):
            member_results.append(future.result())
            print(json.dumps(member_results[-1], sort_keys=True))

    metrics = ensemble_evaluate(arrays, model_dir, config)
    metrics["member_results"] = sorted(member_results, key=lambda item: item["member_id"])
    metrics["hardware_config"] = hardware
    metrics["regression_labels"] = arrays["labels"]
    metrics["problem2_labels"] = PROBLEM2_LABELS
    metrics["label_weights"] = {
        label: float(weight)
        for label, weight in zip(arrays["labels"], arrays["label_weights"], strict=False)
    }
    metrics["constraint_label"] = CONSTRAINT_LABEL
    metrics["class_imbalance_note"] = (
        "official feasible labels are all zero in current Nfp=3 default data; "
        "candidate feasibility risk is derived from the continuous max_normalized_violation regressor"
    )
    write_json(model_dir / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
