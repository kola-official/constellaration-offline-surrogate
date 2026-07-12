from __future__ import annotations

import copy
import json
import math
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

from common_cross_nfp import (
    CONSTRAINT_LABEL,
    DEFAULT_15_LABELS,
    OUTPUT_DIR,
    STAGE1_OUTPUT_DIR,
    apply_cli_overrides,
    append_nfp_condition,
    apply_thread_environment,
    conditioned_feature_columns,
    default_label_weights,
    ensure_output_dirs,
    feature_columns,
    finite_mask,
    load_config,
    parse_args,
    resolve_path,
    write_json,
)


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


def stage_overrides(config: dict[str, Any], stage: str, learning_rate: float | None) -> dict[str, Any]:
    merged = copy.deepcopy(config)
    surrogate = merged["surrogate"]
    if stage == "pretrain":
        pretrain = merged.get("pretrain", {})
        if "max_epochs" in pretrain:
            surrogate["max_epochs"] = pretrain["max_epochs"]
        if "patience" in pretrain:
            surrogate["patience"] = pretrain["patience"]
    if learning_rate is not None:
        surrogate["learning_rate"] = learning_rate
    return merged


def x_scaler_from_nfp3(config: dict[str, Any], x_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    dataset_dir = resolve_path(
        config.get("data", {}).get("nfp3_dataset_dir"),
        STAGE1_OUTPUT_DIR / "dataset",
    )
    train = pd.read_parquet(dataset_dir / "train.parquet")
    missing = [column for column in x_cols if column not in train.columns]
    if missing:
        raise KeyError(f"Nfp=3 train split is missing feature columns: {missing[:5]}")
    x_train = train[x_cols].to_numpy(dtype=np.float32)
    x_mean = x_train.mean(axis=0).astype(np.float32)
    x_std = x_train.std(axis=0).astype(np.float32)
    x_std = np.where(x_std < 1e-8, 1.0, x_std).astype(np.float32)
    return x_mean, x_std


def load_frames(config: dict[str, Any], stage: str) -> dict[str, pd.DataFrame]:
    data_config = config.get("data", {})
    if stage == "pretrain":
        dataset_dir = resolve_path(data_config.get("pretrain_dataset_dir"), OUTPUT_DIR / "dataset_pretrain")
        split_names = ["train", "validation"]
    else:
        dataset_dir = resolve_path(
            data_config.get("nfp3_dataset_dir"),
            STAGE1_OUTPUT_DIR / "dataset",
        )
        split_names = ["train", "validation", "test", "optimization_validation"]
    frames = {name: pd.read_parquet(dataset_dir / f"{name}.parquet") for name in split_names}
    max_rows = data_config.get("max_rows_per_split")
    if max_rows:
        limit = int(max_rows)
        seed = int(config.get("seed", 0))
        for split, frame in list(frames.items()):
            if len(frame) > limit:
                frames[split] = frame.sample(n=limit, random_state=seed).reset_index(drop=True)
    return frames


def group_scaler(frame: pd.DataFrame, labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    values = frame[labels].to_numpy(dtype=np.float32)
    mean = np.nanmean(values, axis=0).astype(np.float32)
    std = np.nanstd(values, axis=0).astype(np.float32)
    mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
    std = np.where((np.isfinite(std)) & (std >= 1e-8), std, 1.0).astype(np.float32)
    return mean, std


def load_arrays(config: dict[str, Any], stage: str) -> dict[str, Any]:
    frames = load_frames(config, stage)
    x_cols = feature_columns(frames["train"])
    labels = [label for label in DEFAULT_15_LABELS if label in frames["train"].columns]
    if labels != DEFAULT_15_LABELS:
        missing = [label for label in DEFAULT_15_LABELS if label not in labels]
        raise KeyError(f"Training data is missing default 15 labels: {missing}")
    required = x_cols + labels + [CONSTRAINT_LABEL]
    if "feasible_under_problem_2" in frames["train"].columns:
        required.append("feasible_under_problem_2")
    for split, frame in list(frames.items()):
        split_required = [column for column in required if column in frame.columns]
        keep = finite_mask(frame, split_required)
        frames[split] = frame.loc[keep].reset_index(drop=True)

    x_mean, x_std = x_scaler_from_nfp3(config, x_cols)
    y_mean, y_std = group_scaler(frames["train"], labels)
    violation_values = frames["train"][CONSTRAINT_LABEL].to_numpy(dtype=np.float32)
    violation_mean = float(np.nanmean(violation_values))
    violation_std = float(np.nanstd(violation_values))
    if not math.isfinite(violation_std) or violation_std < 1e-8:
        violation_std = 1.0

    arrays: dict[str, Any] = {
        "frames": frames,
        "x_cols": x_cols,
        "feature_columns": conditioned_feature_columns(x_cols, config),
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "violation_mean": violation_mean,
        "violation_std": violation_std,
        "labels": labels,
        "label_weights": default_label_weights(config),
    }

    nfp3_value = int(config.get("data", {}).get("nfp3_value", 3))
    for split, frame in frames.items():
        x_raw = frame[x_cols].to_numpy(dtype=np.float32)
        x_scaled = ((x_raw - x_mean.reshape(1, -1)) / x_std.reshape(1, -1)).astype(np.float32)
        x_conditioned = append_nfp_condition(x_scaled, frame, config, default_nfp=nfp3_value)
        y_raw = frame[labels].to_numpy(dtype=np.float32)
        violation_raw = frame[CONSTRAINT_LABEL].to_numpy(dtype=np.float32)
        arrays[split] = {
            "x": x_conditioned,
            "y": ((y_raw - y_mean.reshape(1, -1)) / y_std.reshape(1, -1)).astype(np.float32),
            "y_raw": y_raw.astype(np.float32),
            "violation": ((violation_raw - violation_mean) / violation_std).astype(np.float32),
            "violation_raw": violation_raw.astype(np.float32),
            "sample_id": frame["sample_id"].astype(str).to_numpy()
            if "sample_id" in frame.columns
            else np.array([str(idx) for idx in range(len(frame))]),
            "feasible": frame["feasible_under_problem_2"].astype(float).to_numpy(dtype=np.float32)
            if "feasible_under_problem_2" in frame.columns
            else np.zeros(len(frame), dtype=np.float32),
        }
    return arrays


def make_loader(arrays: dict[str, Any], split: str, batch_size: int, shuffle: bool, workers: int) -> DataLoader:
    data = arrays[split]
    return DataLoader(
        TensorDataset(
            torch.from_numpy(data["x"]),
            torch.from_numpy(data["y"]),
            torch.from_numpy(data["violation"]),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )


def weighted_huber_loss(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    loss = F.huber_loss(pred, target, reduction="none")
    return (loss * weights.reshape(1, -1)).sum(dim=1).mean() / weights.sum().clamp_min(1e-8)


def build_model(arrays: dict[str, Any], config: dict[str, Any], device: torch.device) -> MultiTaskMLP:
    surrogate = config["surrogate"]
    return MultiTaskMLP(
        input_dim=arrays["train"]["x"].shape[1],
        output_dim=len(arrays["labels"]),
        width=int(surrogate["width"]),
        blocks=int(surrogate["blocks"]),
        dropout=float(surrogate["dropout"]),
    ).to(device)


def load_pretrained_state(
    model: nn.Module,
    pretrain_model_dir: Path | None,
    member_id: int,
    load_heads: bool,
) -> dict[str, Any]:
    if pretrain_model_dir is None:
        return {"loaded": False, "reason": "no_pretrain_model_dir"}
    path = pretrain_model_dir / f"member_{member_id:02d}" / "model.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing pretrain checkpoint for member {member_id}: {path}")
    checkpoint = torch.load(path, map_location="cpu")
    source_state = checkpoint["model_state"]
    target_state = model.state_dict()
    loaded_keys = []
    skipped_keys = []
    for key, value in source_state.items():
        allowed = key.startswith("backbone.")
        if load_heads:
            allowed = allowed or key.startswith("regression_head.") or key.startswith("constraint_head.")
        if allowed and key in target_state and target_state[key].shape == value.shape:
            target_state[key] = value
            loaded_keys.append(key)
        elif allowed:
            skipped_keys.append(key)
    model.load_state_dict(target_state)
    return {
        "loaded": True,
        "path": str(path),
        "load_heads": load_heads,
        "loaded_key_count": len(loaded_keys),
        "skipped_keys": skipped_keys,
    }


def train_one_member(
    member_id: int,
    device_index: int,
    config: dict[str, Any],
    arrays: dict[str, Any],
    output_dir: str,
    pretrain_model_dir: str | None,
    load_heads: bool,
) -> dict[str, Any]:
    torch.set_num_threads(int(config["hardware"].get("torch_num_threads", 4)))
    seed = int(config.get("seed", 0)) + member_id * 1009
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(f"cuda:{device_index}" if torch.cuda.is_available() else "cpu")
    model = build_model(arrays, config, device)
    init_report = load_pretrained_state(
        model,
        Path(pretrain_model_dir) if pretrain_model_dir else None,
        member_id,
        load_heads=load_heads,
    )
    model.to(device)
    surrogate = config["surrogate"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(surrogate["learning_rate"]),
        weight_decay=float(surrogate["weight_decay"]),
    )
    constraint_loss_weight = float(surrogate.get("constraint_loss_weight", 0.2))
    constraint_loss = nn.HuberLoss()
    label_weights = torch.from_numpy(arrays["label_weights"]).to(device)
    batch_size = int(surrogate["batch_size"])
    workers = int(config["hardware"].get("dataloader_workers", 0))
    train_loader = make_loader(arrays, "train", batch_size, True, workers)
    validation_loader = make_loader(arrays, "validation", batch_size, False, workers)

    best_validation = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    patience = int(surrogate["patience"])
    max_epochs = int(surrogate["max_epochs"])
    for epoch in range(max_epochs):
        model.train()
        for xb, yb, vb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            vb = vb.to(device, non_blocking=True)
            pred_y, pred_v = model(xb)
            loss = weighted_huber_loss(pred_y, yb, label_weights)
            loss = loss + constraint_loss_weight * constraint_loss(pred_v, vb)
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
                loss = weighted_huber_loss(pred_y, yb, label_weights)
                loss = loss + constraint_loss_weight * constraint_loss(pred_v, vb)
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
            "config": config["surrogate"],
            "regression_labels": arrays["labels"],
            "feature_columns": arrays["feature_columns"],
            "init_report": init_report,
        },
        member_dir / "model.pt",
    )
    return {
        "member_id": member_id,
        "seed": seed,
        "device": str(device),
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation,
        "init_report": init_report,
    }


def predict_member(model_path: Path, arrays: dict[str, Any], split: str, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    checkpoint = torch.load(model_path, map_location=device)
    config = {"surrogate": checkpoint["config"]}
    model = build_model(arrays, config, device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    x = torch.from_numpy(arrays[split]["x"]).to(device)
    pred_y_parts: list[np.ndarray] = []
    pred_v_parts: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, x.shape[0], 8192):
            pred_y, pred_v = model(x[start : start + 8192])
            pred_y_parts.append(pred_y.cpu().numpy())
            pred_v_parts.append(pred_v.cpu().numpy())
    return np.vstack(pred_y_parts), np.concatenate(pred_v_parts)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict[str, dict[str, float]]:
    result = {}
    for idx, label in enumerate(labels):
        true = y_true[:, idx]
        pred = y_pred[:, idx]
        pred_minus_true = pred - true
        result[label] = {
            "mae": float(sk_metrics.mean_absolute_error(true, pred)),
            "rmse": float(np.sqrt(sk_metrics.mean_squared_error(true, pred))),
            "r2": float(sk_metrics.r2_score(true, pred)),
            "bias_pred_minus_true": float(np.mean(pred_minus_true)),
            "optimistic_gap_true_minus_pred": float(np.mean(true - pred)),
        }
    return result


def violation_to_infeasible_prob(values: np.ndarray) -> np.ndarray:
    logits = (values - FEASIBILITY_THRESHOLD) / FEASIBILITY_TEMPERATURE
    logits = np.clip(logits, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-logits))


def write_predictions(
    arrays: dict[str, Any],
    split: str,
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
    violation_mean: np.ndarray,
    violation_std: np.ndarray,
    output_path: Path,
) -> None:
    frame = pd.DataFrame({"sample_id": arrays[split]["sample_id"]})
    y_true = arrays[split]["y_raw"]
    for idx, label in enumerate(arrays["labels"]):
        frame[f"true_{label}"] = y_true[:, idx]
        frame[f"pred_{label}"] = pred_mean[:, idx]
        frame[f"std_{label}"] = pred_std[:, idx]
    frame[f"true_{CONSTRAINT_LABEL}"] = arrays[split]["violation_raw"]
    frame[f"pred_{CONSTRAINT_LABEL}"] = violation_mean
    frame[f"std_{CONSTRAINT_LABEL}"] = violation_std
    frame["pred_infeasible_prob"] = violation_to_infeasible_prob(violation_mean)
    frame["pred_feasible_prob"] = 1.0 - frame["pred_infeasible_prob"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)


def ensemble_evaluate(arrays: dict[str, Any], model_dir: Path) -> dict[str, Any]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_paths = sorted(model_dir.glob("member_*/model.pt"))
    splits = [split for split in ["validation", "test", "optimization_validation"] if split in arrays]
    metrics: dict[str, Any] = {"members": len(model_paths), "splits": {}}
    for split in splits:
        member_y = []
        member_v = []
        for path in model_paths:
            pred_y_z, pred_v_z = predict_member(path, arrays, split, device)
            member_y.append(pred_y_z * arrays["y_std"].reshape(1, -1) + arrays["y_mean"].reshape(1, -1))
            member_v.append(pred_v_z * arrays["violation_std"] + arrays["violation_mean"])
        pred_stack = np.stack(member_y, axis=0)
        violation_stack = np.stack(member_v, axis=0)
        pred_mean = pred_stack.mean(axis=0)
        pred_std = pred_stack.std(axis=0)
        violation_pred_mean = violation_stack.mean(axis=0)
        violation_pred_std = violation_stack.std(axis=0)
        violation_true = arrays[split]["violation_raw"]
        split_metrics = {
            "regression": regression_metrics(arrays[split]["y_raw"], pred_mean, arrays["labels"]),
            "constraint_violation": {
                "label": CONSTRAINT_LABEL,
                "mae": float(sk_metrics.mean_absolute_error(violation_true, violation_pred_mean)),
                "rmse": float(np.sqrt(sk_metrics.mean_squared_error(violation_true, violation_pred_mean))),
                "r2": float(sk_metrics.r2_score(violation_true, violation_pred_mean)),
                "bias_pred_minus_true": float(np.mean(violation_pred_mean - violation_true)),
                "optimistic_gap_true_minus_pred": float(np.mean(violation_true - violation_pred_mean)),
                "threshold": FEASIBILITY_THRESHOLD,
                "infeasible_probability_temperature": FEASIBILITY_TEMPERATURE,
            },
        }
        metrics["splits"][split] = split_metrics
        write_predictions(
            arrays,
            split,
            pred_mean,
            pred_std,
            violation_pred_mean,
            violation_pred_std,
            model_dir / f"predictions_{split}.parquet",
        )
    return metrics


def model_dir_for_run(config: dict[str, Any], stage: str, run_name: str | None) -> Path:
    if run_name:
        name = run_name
    elif stage == "pretrain":
        name = str(config.get("pretrain", {}).get("run_name", "pretrain_90k_15metric"))
    elif stage == "baseline":
        name = str(config.get("baseline", {}).get("run_name", "baseline_random_15metric_nfp"))
    else:
        name = str(config.get("finetune", {}).get("low_lr_run_name", "finetune_low_lr_15metric"))
    return OUTPUT_DIR / "models" / name


def default_pretrain_dir(config: dict[str, Any]) -> Path:
    run_name = str(config.get("finetune", {}).get("pretrain_run_name", "pretrain_90k_15metric"))
    return OUTPUT_DIR / "models" / run_name


def main() -> None:
    args = parse_args()
    if args.stage is None:
        raise SystemExit("--stage is required: pretrain, baseline, or finetune.")
    config = load_config(args.config)
    config = apply_cli_overrides(config, args)
    config = stage_overrides(config, args.stage, args.learning_rate)
    if args.max_rows:
        config.setdefault("data", {})["max_rows_per_split"] = int(args.max_rows)
    hardware = apply_thread_environment(config)
    config["hardware"] = hardware
    ensure_output_dirs()

    model_dir = model_dir_for_run(config, args.stage, args.run_name)
    model_dir.mkdir(parents=True, exist_ok=True)
    arrays = load_arrays(config, args.stage)
    np.savez(
        model_dir / "scalers.npz",
        x_mean=arrays["x_mean"],
        x_std=arrays["x_std"],
        y_mean=arrays["y_mean"],
        y_std=arrays["y_std"],
        violation_mean=np.array(arrays["violation_mean"], dtype=np.float32),
        violation_std=np.array(arrays["violation_std"], dtype=np.float32),
        labels=np.array(arrays["labels"]),
        label_weights=arrays["label_weights"],
        feature_columns=np.array(arrays["feature_columns"]),
        geometry_feature_columns=np.array(arrays["x_cols"]),
        nfp_condition_included=np.array(bool(config.get("features", {}).get("include_nfp", True))),
        nfp_scale=np.array(float(config.get("features", {}).get("nfp_scale", 3.0)), dtype=np.float32),
        x_scaler_source=np.array("nfp3_train"),
    )

    pretrain_model_dir = None
    load_heads = False
    if args.stage == "finetune":
        pretrain_model_dir = str(
            resolve_path(args.pretrain_model_dir, default_pretrain_dir(config))
        )
        load_heads = bool(args.load_heads or config.get("finetune", {}).get("load_heads", False))

    gpu_devices = hardware.get("gpu_devices", [0])
    ensemble_members = int(config["surrogate"]["ensemble_members"])
    max_workers = len(gpu_devices) if hardware.get("parallel_ensemble", True) else 1
    arrays_for_workers = {key: value for key, value in arrays.items() if key != "frames"}
    member_results = []
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
                    arrays_for_workers,
                    str(model_dir),
                    pretrain_model_dir,
                    load_heads,
                )
            )
        for future in as_completed(futures):
            result = future.result()
            member_results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)

    metrics = ensemble_evaluate(arrays, model_dir)
    metrics["stage"] = args.stage
    metrics["model_dir"] = str(model_dir)
    metrics["member_results"] = sorted(member_results, key=lambda item: item["member_id"])
    metrics["hardware_config"] = hardware
    metrics["regression_labels"] = arrays["labels"]
    metrics["label_weights"] = {
        label: float(weight)
        for label, weight in zip(arrays["labels"], arrays["label_weights"], strict=False)
    }
    metrics["feature_policy"] = {
        "feature_columns": arrays["feature_columns"],
        "geometry_feature_count": len(arrays["x_cols"]),
        "x_scaler_source": "nfp3_train",
        "nfp_condition_included": bool(config.get("features", {}).get("include_nfp", True)),
        "nfp_scale": float(config.get("features", {}).get("nfp_scale", 3.0)),
    }
    metrics["finetune_init"] = {
        "pretrain_model_dir": pretrain_model_dir,
        "load_heads": load_heads,
    }
    write_json(model_dir / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
