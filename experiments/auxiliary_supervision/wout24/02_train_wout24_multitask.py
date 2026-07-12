from __future__ import annotations

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

from common_wout24 import (
    CONSTRAINT_LABEL,
    DEFAULT_AUX_LABELS,
    OUTPUT_DIR,
    PROBLEM2_LABELS,
    STAGE1_OUTPUT_DIR,
    WOUT24_LABELS,
    apply_thread_environment,
    ensure_output_dirs,
    feature_columns,
    load_config,
    parse_args,
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


def finite_mask(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    values = frame[columns].to_numpy(dtype=np.float32)
    return np.isfinite(values).all(axis=1)


def resolve_path(value: str | None, default: Path) -> Path:
    path = Path(value) if value else default
    if not path.is_absolute():
        path = (Path(__file__).resolve().parent / path).resolve()
    return path


def group_scaler(frame: pd.DataFrame, labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    values = frame[labels].to_numpy(dtype=np.float32)
    mean = np.nanmean(values, axis=0).astype(np.float32)
    std = np.nanstd(values, axis=0).astype(np.float32)
    mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
    std = np.where((np.isfinite(std)) & (std >= 1e-8), std, 1.0).astype(np.float32)
    return mean, std


def scaled_group(frame: pd.DataFrame, labels: list[str], mean: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = frame[labels].to_numpy(dtype=np.float32)
    mask = np.isfinite(raw).astype(np.float32)
    filled = np.where(mask > 0.0, raw, mean.reshape(1, -1)).astype(np.float32)
    scaled = ((filled - mean.reshape(1, -1)) / std.reshape(1, -1)).astype(np.float32)
    return scaled, raw.astype(np.float32), mask.astype(np.float32)


def label_weights(labels: list[str], config: dict[str, Any]) -> np.ndarray:
    weights = np.ones(len(labels), dtype=np.float32)
    for name, value in config.get("surrogate", {}).get("label_weights", {}).items():
        if name in labels:
            weights[labels.index(name)] = float(value)
    return weights


def load_frames(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    dataset_dir = resolve_path(
        config.get("data", {}).get("stage1_dataset_dir"),
        STAGE1_OUTPUT_DIR / "dataset",
    )
    wout_labels_path = resolve_path(
        config.get("data", {}).get("wout24_labels_path"),
        OUTPUT_DIR / "dataset" / "wout24_labels.parquet",
    )
    sample_map_path = resolve_path(
        config.get("data", {}).get("sample_wout_map_path"),
        OUTPUT_DIR / "dataset" / "sample_wout_map.parquet",
    )
    if not wout_labels_path.exists():
        raise FileNotFoundError(f"Missing {wout_labels_path}; run 01_build_wout24_labels.py first.")
    if not sample_map_path.exists():
        raise FileNotFoundError(f"Missing {sample_map_path}; run 01_build_wout24_labels.py first.")

    wout_labels = pd.read_parquet(wout_labels_path)
    sample_map = pd.read_parquet(sample_map_path)[["sample_id", "wout_id"]].drop_duplicates("sample_id")
    frames: dict[str, pd.DataFrame] = {}
    for split in ["train", "validation", "test", "optimization_validation"]:
        frame = pd.read_parquet(dataset_dir / f"{split}.parquet")
        frame = frame.merge(sample_map, on="sample_id", how="left")
        frame = frame.merge(wout_labels, on="wout_id", how="left")
        frames[split] = frame
    max_rows = config.get("data", {}).get("max_rows")
    if max_rows:
        limit = int(max_rows)
        seed = int(config.get("seed", 0))
        for split, frame in list(frames.items()):
            if len(frame) > limit:
                frames[split] = frame.sample(n=limit, random_state=seed).reset_index(drop=True)
    return frames


def load_arrays(config: dict[str, Any]) -> dict[str, Any]:
    frames = load_frames(config)
    x_cols = feature_columns(frames["train"])
    required_complete = x_cols + PROBLEM2_LABELS + DEFAULT_AUX_LABELS + [CONSTRAINT_LABEL, "feasible_under_problem_2"]
    for split, frame in list(frames.items()):
        keep = finite_mask(frame, required_complete)
        frames[split] = frame.loc[keep].reset_index(drop=True)

    train = frames["train"]
    x_train = train[x_cols].to_numpy(dtype=np.float32)
    x_mean = x_train.mean(axis=0).astype(np.float32)
    x_std = x_train.std(axis=0).astype(np.float32)
    x_std = np.where(x_std < 1e-8, 1.0, x_std).astype(np.float32)

    p2_mean, p2_std = group_scaler(train, PROBLEM2_LABELS)
    aux_mean, aux_std = group_scaler(train, DEFAULT_AUX_LABELS)
    wout_mean, wout_std = group_scaler(train, WOUT24_LABELS)
    violation_mean = float(train[CONSTRAINT_LABEL].to_numpy(dtype=np.float32).mean())
    violation_std = float(train[CONSTRAINT_LABEL].to_numpy(dtype=np.float32).std())
    if not math.isfinite(violation_std) or violation_std < 1e-8:
        violation_std = 1.0

    arrays: dict[str, Any] = {
        "frames": frames,
        "x_cols": x_cols,
        "x_mean": x_mean,
        "x_std": x_std,
        "problem2_labels": list(PROBLEM2_LABELS),
        "default_aux_labels": list(DEFAULT_AUX_LABELS),
        "wout_labels": list(WOUT24_LABELS),
        "problem2_mean": p2_mean,
        "problem2_std": p2_std,
        "default_aux_mean": aux_mean,
        "default_aux_std": aux_std,
        "wout_mean": wout_mean,
        "wout_std": wout_std,
        "violation_mean": violation_mean,
        "violation_std": violation_std,
        "problem2_weights": label_weights(PROBLEM2_LABELS, config),
        "default_aux_weights": np.ones(len(DEFAULT_AUX_LABELS), dtype=np.float32),
        "wout_weights": np.ones(len(WOUT24_LABELS), dtype=np.float32),
    }

    for split, frame in frames.items():
        x = frame[x_cols].to_numpy(dtype=np.float32)
        p2, p2_raw, p2_mask = scaled_group(frame, PROBLEM2_LABELS, p2_mean, p2_std)
        aux, aux_raw, aux_mask = scaled_group(frame, DEFAULT_AUX_LABELS, aux_mean, aux_std)
        wout, wout_raw, wout_mask = scaled_group(frame, WOUT24_LABELS, wout_mean, wout_std)
        violation_raw = frame[CONSTRAINT_LABEL].to_numpy(dtype=np.float32)
        arrays[split] = {
            "x": ((x - x_mean) / x_std).astype(np.float32),
            "problem2": p2,
            "problem2_raw": p2_raw,
            "problem2_mask": p2_mask,
            "default_aux": aux,
            "default_aux_raw": aux_raw,
            "default_aux_mask": aux_mask,
            "wout": wout,
            "wout_raw": wout_raw,
            "wout_mask": wout_mask,
            "violation": ((violation_raw - violation_mean) / violation_std).astype(np.float32),
            "violation_raw": violation_raw.astype(np.float32),
            "feasible": frame["feasible_under_problem_2"].astype(float).to_numpy(dtype=np.float32),
            "sample_id": frame["sample_id"].astype(str).to_numpy(),
            "wout_id": frame["wout_id"].astype(str).to_numpy() if "wout_id" in frame.columns else np.array([""] * len(frame)),
        }
    return arrays


def make_loader(arrays: dict[str, Any], split: str, batch_size: int, shuffle: bool, workers: int) -> DataLoader:
    data = arrays[split]
    tensors = [
        torch.from_numpy(data["x"]),
        torch.from_numpy(data["problem2"]),
        torch.from_numpy(data["problem2_mask"]),
        torch.from_numpy(data["default_aux"]),
        torch.from_numpy(data["default_aux_mask"]),
        torch.from_numpy(data["wout"]),
        torch.from_numpy(data["wout_mask"]),
        torch.from_numpy(data["violation"]),
    ]
    return DataLoader(
        TensorDataset(*tensors),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )


def masked_huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    loss = F.huber_loss(pred, target, reduction="none")
    weighted_mask = mask * weights.reshape(1, -1)
    denom = weighted_mask.sum().clamp_min(1e-8)
    return (loss * weighted_mask).sum() / denom


def batch_loss(
    pred: dict[str, torch.Tensor],
    batch: tuple[torch.Tensor, ...],
    weights: dict[str, torch.Tensor],
    config: dict[str, Any],
) -> torch.Tensor:
    _, p2, p2_mask, aux, aux_mask, wout, wout_mask, violation = batch
    surrogate = config["surrogate"]
    loss = float(surrogate.get("problem2_loss_weight", 1.0)) * masked_huber_loss(
        pred["problem2"], p2, p2_mask, weights["problem2"]
    )
    loss = loss + float(surrogate.get("default_aux_loss_weight", 0.25)) * masked_huber_loss(
        pred["default_aux"], aux, aux_mask, weights["default_aux"]
    )
    loss = loss + float(surrogate.get("wout_loss_weight", 0.15)) * masked_huber_loss(
        pred["wout"], wout, wout_mask, weights["wout"]
    )
    loss = loss + float(surrogate.get("constraint_loss_weight", 0.2)) * F.huber_loss(
        pred["constraint"], violation
    )
    return loss


def build_model(arrays: dict[str, Any], config: dict[str, Any], device: torch.device) -> GroupedMultiTaskMLP:
    surrogate = config["surrogate"]
    return GroupedMultiTaskMLP(
        input_dim=arrays["train"]["x"].shape[1],
        problem2_dim=len(arrays["problem2_labels"]),
        default_aux_dim=len(arrays["default_aux_labels"]),
        wout_dim=len(arrays["wout_labels"]),
        width=int(surrogate["width"]),
        blocks=int(surrogate["blocks"]),
        dropout=float(surrogate["dropout"]),
    ).to(device)


def train_one_member(
    member_id: int,
    device_index: int,
    config: dict[str, Any],
    arrays: dict[str, Any],
    output_dir: str,
) -> dict[str, Any]:
    torch.set_num_threads(int(config["hardware"].get("torch_num_threads", 4)))
    seed = int(config.get("seed", 0)) + member_id * 1009
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(f"cuda:{device_index}" if torch.cuda.is_available() else "cpu")
    model = build_model(arrays, config, device)
    surrogate = config["surrogate"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(surrogate["learning_rate"]),
        weight_decay=float(surrogate["weight_decay"]),
    )
    weights = {
        "problem2": torch.from_numpy(arrays["problem2_weights"]).to(device),
        "default_aux": torch.from_numpy(arrays["default_aux_weights"]).to(device),
        "wout": torch.from_numpy(arrays["wout_weights"]).to(device),
    }
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
        for raw_batch in train_loader:
            batch = tuple(t.to(device, non_blocking=True) for t in raw_batch)
            pred = model(batch[0])
            loss = batch_loss(pred, batch, weights, config)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        losses = []
        with torch.no_grad():
            for raw_batch in validation_loader:
                batch = tuple(t.to(device, non_blocking=True) for t in raw_batch)
                losses.append(float(batch_loss(model(batch[0]), batch, weights, config).detach().cpu()))
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
            "config": surrogate,
            "problem2_labels": arrays["problem2_labels"],
            "default_aux_labels": arrays["default_aux_labels"],
            "wout_labels": arrays["wout_labels"],
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


def predict_member(model_path: Path, arrays: dict[str, Any], split: str, device: torch.device) -> dict[str, np.ndarray]:
    checkpoint = torch.load(model_path, map_location=device)
    config = {"surrogate": checkpoint["config"]}
    model = build_model(arrays, config, device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    x = torch.from_numpy(arrays[split]["x"]).to(device)
    outputs: dict[str, list[np.ndarray]] = {"problem2": [], "default_aux": [], "wout": [], "constraint": []}
    with torch.no_grad():
        for start in range(0, x.shape[0], 8192):
            pred = model(x[start : start + 8192])
            for key in outputs:
                outputs[key].append(pred[key].cpu().numpy())
    return {
        "problem2": np.vstack(outputs["problem2"]),
        "default_aux": np.vstack(outputs["default_aux"]),
        "wout": np.vstack(outputs["wout"]),
        "constraint": np.concatenate(outputs["constraint"]),
    }


def unscale(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return values * std.reshape(1, -1) + mean.reshape(1, -1)


def regression_metrics_masked(
    true: np.ndarray,
    pred: np.ndarray,
    mask: np.ndarray,
    labels: list[str],
) -> dict[str, dict[str, float | int | str]]:
    result: dict[str, dict[str, float | int | str]] = {}
    for idx, label in enumerate(labels):
        keep = mask[:, idx] > 0.0
        if int(keep.sum()) < 3:
            result[label] = {"status": "not_enough_finite", "count": int(keep.sum())}
            continue
        y_true = true[keep, idx]
        y_pred = pred[keep, idx]
        result[label] = {
            "count": int(keep.sum()),
            "mae": float(sk_metrics.mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(sk_metrics.mean_squared_error(y_true, y_pred))),
            "r2": float(sk_metrics.r2_score(y_true, y_pred)),
        }
    return result


def violation_to_infeasible_prob(values: np.ndarray) -> np.ndarray:
    logits = (values - FEASIBILITY_THRESHOLD) / FEASIBILITY_TEMPERATURE
    logits = np.clip(logits, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-logits))


def write_predictions(
    arrays: dict[str, Any],
    split: str,
    predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    violation_mean: np.ndarray,
    violation_std: np.ndarray,
    output_path: Path,
) -> None:
    frame = pd.DataFrame({"sample_id": arrays[split]["sample_id"], "wout_id": arrays[split]["wout_id"]})
    for group, labels in [
        ("problem2", arrays["problem2_labels"]),
        ("default_aux", arrays["default_aux_labels"]),
        ("wout", arrays["wout_labels"]),
    ]:
        pred_mean, pred_std = predictions[group]
        raw = arrays[split][f"{group}_raw"]
        mask = arrays[split][f"{group}_mask"]
        for idx, label in enumerate(labels):
            frame[f"true_{label}"] = raw[:, idx]
            frame[f"mask_{label}"] = mask[:, idx]
            frame[f"pred_{label}"] = pred_mean[:, idx]
            frame[f"std_{label}"] = pred_std[:, idx]
    frame[f"true_{CONSTRAINT_LABEL}"] = arrays[split]["violation_raw"]
    frame[f"pred_{CONSTRAINT_LABEL}"] = violation_mean
    frame[f"std_{CONSTRAINT_LABEL}"] = violation_std
    frame["pred_infeasible_prob"] = violation_to_infeasible_prob(violation_mean)
    frame["pred_feasible_prob"] = 1.0 - frame["pred_infeasible_prob"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)


def ensemble_evaluate(arrays: dict[str, Any], model_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_paths = sorted(model_dir.glob("member_*/model.pt"))
    metrics: dict[str, Any] = {"members": len(model_paths), "splits": {}}
    for split in ["test", "optimization_validation"]:
        stacks: dict[str, list[np.ndarray]] = {"problem2": [], "default_aux": [], "wout": [], "constraint": []}
        for path in model_paths:
            pred = predict_member(path, arrays, split, device)
            stacks["problem2"].append(unscale(pred["problem2"], arrays["problem2_mean"], arrays["problem2_std"]))
            stacks["default_aux"].append(unscale(pred["default_aux"], arrays["default_aux_mean"], arrays["default_aux_std"]))
            stacks["wout"].append(unscale(pred["wout"], arrays["wout_mean"], arrays["wout_std"]))
            stacks["constraint"].append(pred["constraint"] * arrays["violation_std"] + arrays["violation_mean"])
        pred_mean_std: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for group in ["problem2", "default_aux", "wout"]:
            stack = np.stack(stacks[group], axis=0)
            pred_mean_std[group] = (stack.mean(axis=0), stack.std(axis=0))
        violation_stack = np.stack(stacks["constraint"], axis=0)
        violation_mean = violation_stack.mean(axis=0)
        violation_std = violation_stack.std(axis=0)
        split_metrics = {
            "problem2": regression_metrics_masked(
                arrays[split]["problem2_raw"],
                pred_mean_std["problem2"][0],
                arrays[split]["problem2_mask"],
                arrays["problem2_labels"],
            ),
            "default_aux": regression_metrics_masked(
                arrays[split]["default_aux_raw"],
                pred_mean_std["default_aux"][0],
                arrays[split]["default_aux_mask"],
                arrays["default_aux_labels"],
            ),
            "wout": regression_metrics_masked(
                arrays[split]["wout_raw"],
                pred_mean_std["wout"][0],
                arrays[split]["wout_mask"],
                arrays["wout_labels"],
            ),
            "constraint_violation": {
                "label": CONSTRAINT_LABEL,
                "mae": float(sk_metrics.mean_absolute_error(arrays[split]["violation_raw"], violation_mean)),
                "rmse": float(np.sqrt(sk_metrics.mean_squared_error(arrays[split]["violation_raw"], violation_mean))),
                "r2": float(sk_metrics.r2_score(arrays[split]["violation_raw"], violation_mean)),
                "threshold": FEASIBILITY_THRESHOLD,
                "infeasible_probability_temperature": FEASIBILITY_TEMPERATURE,
            },
            "wout_mask_coverage": {
                label: float(arrays[split]["wout_mask"][:, idx].mean())
                for idx, label in enumerate(arrays["wout_labels"])
            },
        }
        metrics["splits"][split] = split_metrics
        write_predictions(
            arrays,
            split,
            pred_mean_std,
            violation_mean,
            violation_std,
            model_dir / f"predictions_{split}.parquet",
        )
    return metrics


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    hardware = apply_thread_environment(config)
    config["hardware"] = hardware
    ensure_output_dirs()
    model_dir = OUTPUT_DIR / "models" / "wout24_multitask"
    model_dir.mkdir(parents=True, exist_ok=True)
    arrays = load_arrays(config)
    np.savez(
        model_dir / "scalers.npz",
        x_mean=arrays["x_mean"],
        x_std=arrays["x_std"],
        problem2_mean=arrays["problem2_mean"],
        problem2_std=arrays["problem2_std"],
        default_aux_mean=arrays["default_aux_mean"],
        default_aux_std=arrays["default_aux_std"],
        wout_mean=arrays["wout_mean"],
        wout_std=arrays["wout_std"],
        violation_mean=np.array(arrays["violation_mean"], dtype=np.float32),
        violation_std=np.array(arrays["violation_std"], dtype=np.float32),
        problem2_labels=np.array(arrays["problem2_labels"]),
        default_aux_labels=np.array(arrays["default_aux_labels"]),
        wout_labels=np.array(arrays["wout_labels"]),
        feature_columns=np.array(arrays["x_cols"]),
    )
    split_summary = {
        split: {
            "rows": int(arrays[split]["x"].shape[0]),
            "wout_rows_any_label": int((arrays[split]["wout_mask"].sum(axis=1) > 0).sum()),
            "wout_label_mean_coverage": float(arrays[split]["wout_mask"].mean()),
        }
        for split in ["train", "validation", "test", "optimization_validation"]
    }
    write_json(OUTPUT_DIR / "run_summary" / "wout24_training_data_summary.json", split_summary)

    gpu_devices = hardware.get("gpu_devices", [0])
    ensemble_members = int(config["surrogate"]["ensemble_members"])
    member_results = []
    max_workers = len(gpu_devices) if hardware.get("parallel_ensemble", True) else 1
    arrays_for_workers = {key: value for key, value in arrays.items() if key != "frames"}
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
                )
            )
        for future in as_completed(futures):
            member_results.append(future.result())
            print(json.dumps(member_results[-1], sort_keys=True), flush=True)
    metrics = ensemble_evaluate(arrays, model_dir, config)
    metrics["member_results"] = sorted(member_results, key=lambda item: item["member_id"])
    metrics["hardware_config"] = hardware
    metrics["split_summary"] = split_summary
    metrics["problem2_labels"] = arrays["problem2_labels"]
    metrics["default_aux_labels"] = arrays["default_aux_labels"]
    metrics["wout_labels"] = arrays["wout_labels"]
    metrics["model_note"] = (
        "Shared encoder with separate problem2/default_aux/wout heads and label-wise masked wout loss. "
        "Online scoring should consume only problem2 labels plus the constraint head."
    )
    write_json(model_dir / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
