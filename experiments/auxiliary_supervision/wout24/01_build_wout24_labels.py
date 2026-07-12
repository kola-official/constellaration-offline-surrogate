from __future__ import annotations

import hashlib
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from datasets import load_dataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from common_wout24 import (
    OUTPUT_DIR,
    STAGE1_OUTPUT_DIR,
    WOUT24_LABELS,
    apply_thread_environment,
    ensure_output_dirs,
    load_config,
    parse_args,
    write_json,
)


PCA_SPECS = {
    "bmnc": {"field": "bmnc", "mode": "nyq", "components": 4, "prefix": "wout_bmnc_pca"},
    "rmnc": {"field": "rmnc", "mode": "mnmax", "components": 2, "prefix": "wout_rmnc_pca"},
    "zmns": {"field": "zmns", "mode": "mnmax", "components": 2, "prefix": "wout_zmns_pca"},
    "lmns": {"field": "lmns", "mode": "mnmax", "components": 2, "prefix": "wout_lmns_pca"},
    "iota_profile": {"field": "iota_full", "mode": "profile", "components": 2, "prefix": "wout_iota_profile_pca"},
}

_WORKER_PCA_MODELS: dict[str, Any] | None = None
_WORKER_CONFIG: dict[str, Any] | None = None


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def finite_float(value: Any) -> float:
    if value is None or isinstance(value, bool):
        return float("nan")
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def finite_array(value: Any) -> np.ndarray:
    if not isinstance(value, list) or not value:
        return np.array([], dtype=np.float32)
    try:
        arr = np.asarray(value, dtype=np.float32)
    except Exception:
        return np.array([], dtype=np.float32)
    return arr


def finite_1d(value: Any) -> np.ndarray:
    arr = finite_array(value).reshape(-1)
    return arr[np.isfinite(arr)]


def quantile(value: Any, q: float) -> float:
    arr = finite_1d(value)
    if arr.size == 0:
        return float("nan")
    return float(np.quantile(arr, q))


def mean(value: Any) -> float:
    arr = finite_1d(value)
    return float(arr.mean()) if arr.size else float("nan")


def std(value: Any) -> float:
    arr = finite_1d(value)
    return float(arr.std()) if arr.size else float("nan")


def min_value(value: Any) -> float:
    arr = finite_1d(value)
    return float(arr.min()) if arr.size else float("nan")


def edge_axis_l2(value: Any) -> float:
    arr = finite_array(value)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return float("nan")
    diff = arr[-1] - arr[0]
    if not np.isfinite(diff).any():
        return float("nan")
    return float(np.sqrt(np.nanmean(diff * diff)))


def iota_shape_vector(value: Any) -> np.ndarray:
    arr = finite_1d(value)
    if arr.size < 4:
        return np.full(99, np.nan, dtype=np.float32)
    x = np.linspace(0.0, 1.0, arr.size, dtype=np.float32)
    line = arr[0] + (arr[-1] - arr[0]) * x
    residual = arr - line
    return resample_1d(residual, 99)


def resample_1d(arr: np.ndarray, size: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return np.full(size, np.nan, dtype=np.float32)
    if arr.size == size:
        return arr.astype(np.float32)
    src = np.linspace(0.0, 1.0, arr.size)
    dst = np.linspace(0.0, 1.0, size)
    return np.interp(dst, src, arr).astype(np.float32)


def iota_slope_inner(value: Any) -> float:
    arr = finite_1d(value)
    if arr.size < 8:
        return float("nan")
    idx = max(1, int(round(0.25 * (arr.size - 1))))
    return float(arr[idx] - arr[0])


def iota_curvature_rms(value: Any) -> float:
    arr = finite_1d(value)
    if arr.size < 5:
        return float("nan")
    curv = np.diff(arr, n=2)
    return float(np.sqrt(np.mean(curv * curv)))


def low_mode_mask(obj: dict[str, Any], mode: str, abs_m_max: int, abs_n_over_nfp_max: int) -> np.ndarray:
    if mode == "nyq":
        xm = finite_1d(obj.get("xm_nyq"))
        xn = finite_1d(obj.get("xn_nyq"))
    else:
        xm = finite_1d(obj.get("xm"))
        xn = finite_1d(obj.get("xn"))
    nfp = abs(finite_float(obj.get("nfp")))
    if not math.isfinite(nfp) or nfp < 1.0:
        nfp = 1.0
    n = min(xm.size, xn.size)
    if n == 0:
        return np.array([], dtype=bool)
    xm = xm[:n]
    xn = xn[:n]
    return (np.abs(xm) <= abs_m_max) & (np.abs(xn / nfp) <= abs_n_over_nfp_max)


def pca_vector(obj: dict[str, Any], spec: dict[str, Any], config: dict[str, Any]) -> np.ndarray:
    if spec["mode"] == "profile":
        return iota_shape_vector(obj.get(spec["field"]))
    arr = finite_array(obj.get(spec["field"]))
    if arr.ndim != 2:
        return np.array([], dtype=np.float32)
    mask = low_mode_mask(
        obj,
        str(spec["mode"]),
        int(config["wout"].get("pca_low_mode_abs_m_max", 4)),
        int(config["wout"].get("pca_low_mode_abs_n_over_nfp_max", 4)),
    )
    if mask.size == 0:
        return np.array([], dtype=np.float32)
    width = min(arr.shape[1], mask.size)
    selected = arr[:, :width][:, mask[:width]]
    return selected.astype(np.float32).reshape(-1)


def sorted_part_paths(parts_dir: Path) -> list[Path]:
    def key(path: Path) -> tuple[int, str]:
        try:
            return int(path.stem.split(".")[1]), path.name
        except Exception:
            return 10**9, path.name

    return sorted(parts_dir.glob("*.parquet"), key=key)


def iter_wout_json(parts: list[Path]):
    for path in parts:
        table = pq.read_table(path, columns=["id", "json"])
        ids = table["id"].to_pylist()
        payloads = table["json"].to_pylist()
        for wout_id, payload in zip(ids, payloads, strict=False):
            yield str(wout_id), json.loads(payload)


def pca_sample_parts(parts: list[Path], max_rows: int) -> list[Path]:
    if not parts:
        return []
    approx_rows_per_part = 32
    count = min(len(parts), max(32, math.ceil(max_rows / approx_rows_per_part)))
    indices = np.linspace(0, len(parts) - 1, count, dtype=int)
    return [parts[int(idx)] for idx in sorted(set(indices))]


def fit_pca_models(parts: list[Path], config: dict[str, Any]) -> dict[str, Any]:
    max_rows = int(config["wout"].get("pca_fit_max_rows", 8000))
    sample_parts = pca_sample_parts(parts, max_rows)
    buffers: dict[str, list[np.ndarray]] = {name: [] for name in PCA_SPECS}
    started = time.time()
    rows_seen = 0
    for wout_id, obj in iter_wout_json(sample_parts):
        rows_seen += 1
        for name, spec in PCA_SPECS.items():
            if len(buffers[name]) >= max_rows:
                continue
            vector = pca_vector(obj, spec, config)
            if vector.size and np.isfinite(vector).all():
                buffers[name].append(vector)
        if rows_seen % 1000 == 0:
            print(
                json.dumps(
                    {
                        "stage": "pca_sample",
                        "rows_seen": rows_seen,
                        "elapsed_seconds": round(time.time() - started, 1),
                        "buffers": {key: len(value) for key, value in buffers.items()},
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    models: dict[str, Any] = {}
    summary: dict[str, Any] = {"sample_parts": len(sample_parts), "rows_seen": rows_seen, "models": {}}
    for name, spec in PCA_SPECS.items():
        if not buffers[name]:
            raise RuntimeError(f"No PCA samples collected for {name}")
        matrix = np.vstack(buffers[name]).astype(np.float32)
        components = min(int(spec["components"]), matrix.shape[0], matrix.shape[1])
        scaler = StandardScaler()
        scaled = scaler.fit_transform(matrix)
        pca = PCA(n_components=components, random_state=int(config.get("seed", 0)))
        pca.fit(scaled)
        models[name] = {"scaler": scaler, "pca": pca, "spec": spec}
        summary["models"][name] = {
            "samples": int(matrix.shape[0]),
            "input_dim": int(matrix.shape[1]),
            "components": int(components),
            "explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
        }
    write_json(OUTPUT_DIR / "run_summary" / "wout24_pca_fit_summary.json", summary)
    joblib.dump(models, OUTPUT_DIR / "dataset" / "wout24_pca_models.joblib")
    return models


def load_or_fit_pca_models(parts: list[Path], config: dict[str, Any]) -> dict[str, Any]:
    pca_path = OUTPUT_DIR / "dataset" / "wout24_pca_models.joblib"
    reuse = bool(config["wout"].get("reuse_pca_models", True))
    if reuse and pca_path.exists():
        print(
            json.dumps(
                {"stage": "pca_model_reuse", "path": str(pca_path)},
                sort_keys=True,
            ),
            flush=True,
        )
        return joblib.load(pca_path)
    return fit_pca_models(parts, config)


def transform_pca(name: str, obj: dict[str, Any], model: dict[str, Any], config: dict[str, Any]) -> dict[str, float]:
    spec = model["spec"]
    vector = pca_vector(obj, spec, config)
    prefix = str(spec["prefix"])
    result = {f"{prefix}_{idx:02d}": float("nan") for idx in range(int(spec["components"]))}
    if vector.size != int(model["scaler"].mean_.shape[0]) or not np.isfinite(vector).all():
        return result
    scaled = model["scaler"].transform(vector.reshape(1, -1))
    values = model["pca"].transform(scaled).reshape(-1)
    for idx, value in enumerate(values):
        result[f"{prefix}_{idx:02d}"] = float(value)
    return result


def direct_wout_features(obj: dict[str, Any]) -> dict[str, float]:
    return {
        "wout_bsupvmnc_edge_axis_l2": edge_axis_l2(obj.get("bsupvmnc")),
        "wout_bmnc_edge_axis_l2": edge_axis_l2(obj.get("bmnc")),
        "wout_bsubvmnc_mean": mean(obj.get("bsubvmnc")),
        "wout_bsupvmnc_mean": mean(obj.get("bsupvmnc")),
        "wout_iota_profile_slope_inner": iota_slope_inner(obj.get("iota_full")),
        "wout_iota_profile_curvature_rms": iota_curvature_rms(obj.get("iota_full")),
        "wout_DMerc_min": min_value(obj.get("DMerc")),
        "wout_DMerc_q25": quantile(obj.get("DMerc"), 0.25),
        "wout_Dgeod_q25": quantile(obj.get("Dgeod"), 0.25),
        "wout_Dgeod_q75": quantile(obj.get("Dgeod"), 0.75),
        "wout_jcuru_q75": quantile(obj.get("jcuru"), 0.75),
        "wout_bvco_std": std(obj.get("bvco")),
    }


def transform_wout_record(
    wout_id: str,
    obj: dict[str, Any],
    pca_models: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, float | str]:
    record: dict[str, float | str] = {"wout_id": wout_id}
    for name, model in pca_models.items():
        record.update(transform_pca(name, obj, model, config))
    record.update(direct_wout_features(obj))
    return record


def init_transform_worker(pca_models: dict[str, Any], config: dict[str, Any]) -> None:
    global _WORKER_PCA_MODELS, _WORKER_CONFIG
    _WORKER_PCA_MODELS = pca_models
    _WORKER_CONFIG = config


def transform_part_worker(path_text: str) -> tuple[str, list[dict[str, float | str]]]:
    if _WORKER_PCA_MODELS is None or _WORKER_CONFIG is None:
        raise RuntimeError("Transform worker was not initialized")
    rows = [
        transform_wout_record(wout_id, obj, _WORKER_PCA_MODELS, _WORKER_CONFIG)
        for wout_id, obj in iter_wout_json([Path(path_text)])
    ]
    return path_text, rows


def transform_wout_serial(
    parts: list[Path],
    pca_models: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, float | str]]:
    rows = []
    started = time.time()
    for index, (wout_id, obj) in enumerate(iter_wout_json(parts), start=1):
        rows.append(transform_wout_record(wout_id, obj, pca_models, config))
        if index % 1000 == 0:
            print(
                json.dumps(
                    {
                        "stage": "wout24_transform",
                        "rows": index,
                        "elapsed_seconds": round(time.time() - started, 1),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return rows


def transform_wout_parallel(
    parts: list[Path],
    pca_models: dict[str, Any],
    config: dict[str, Any],
    workers: int,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    started = time.time()
    rows_done = 0
    parts_done = 0
    next_report = 1000
    print(
        json.dumps(
            {
                "stage": "wout24_transform_start",
                "workers": workers,
                "parts": len(parts),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=init_transform_worker,
        initargs=(pca_models, config),
    ) as executor:
        futures = {executor.submit(transform_part_worker, str(path)): path for path in parts}
        for future in as_completed(futures):
            path = futures[future]
            try:
                _, part_rows = future.result()
            except Exception as exc:
                raise RuntimeError(f"Failed to transform {path}") from exc
            rows.extend(part_rows)
            rows_done += len(part_rows)
            parts_done += 1
            if rows_done >= next_report or parts_done == len(parts):
                print(
                    json.dumps(
                        {
                            "stage": "wout24_transform",
                            "workers": workers,
                            "rows": rows_done,
                            "parts_done": parts_done,
                            "parts_total": len(parts),
                            "elapsed_seconds": round(time.time() - started, 1),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                while next_report <= rows_done:
                    next_report += 1000
    return rows


def build_wout_label_table(parts: list[Path], pca_models: dict[str, Any], config: dict[str, Any]) -> pd.DataFrame:
    configured_workers = config["wout"].get(
        "transform_workers",
        config.get("hardware", {}).get("cpu_workers", 1),
    )
    workers = max(1, min(int(configured_workers), len(parts)))
    if workers > 1:
        rows = transform_wout_parallel(parts, pca_models, config, workers)
    else:
        rows = transform_wout_serial(parts, pca_models, config)
    frame = pd.DataFrame(rows)
    missing = [label for label in WOUT24_LABELS if label not in frame.columns]
    if missing:
        raise RuntimeError(f"Missing expected wout24 labels: {missing}")
    return frame[["wout_id"] + WOUT24_LABELS].sort_values("wout_id").reset_index(drop=True)


def stage1_sample_ids(dataset_dir: Path) -> pd.DataFrame:
    rows = []
    for split in ["train", "validation", "test", "optimization_validation"]:
        path = dataset_dir / f"{split}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path, columns=["sample_id"])
        rows.extend({"sample_id": str(value), "split": split} for value in frame["sample_id"])
    return pd.DataFrame(rows).drop_duplicates()


def build_sample_wout_map(dataset_dir: Path, config: dict[str, Any]) -> pd.DataFrame:
    ids = stage1_sample_ids(dataset_dir)
    wanted = set(ids["sample_id"].astype(str))
    rows = []
    dataset = load_dataset("proxima-fusion/constellaration", "default", split="train")
    for row in dataset:
        boundary_json = row.get("boundary.json")
        wout_id = row.get("misc.vmecpp_wout_id")
        if not boundary_json or not wout_id:
            continue
        sample_id = sha1_text(str(boundary_json))
        if sample_id in wanted:
            rows.append({"sample_id": sample_id, "wout_id": str(wout_id)})
    mapped = ids.merge(pd.DataFrame(rows), on="sample_id", how="left")
    return mapped


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    hardware = apply_thread_environment(config)
    ensure_output_dirs()

    dataset_dir = Path(config.get("data", {}).get("stage1_dataset_dir") or STAGE1_OUTPUT_DIR / "dataset")
    if not dataset_dir.is_absolute():
        dataset_dir = (Path(__file__).resolve().parent / dataset_dir).resolve()
    configured_parts_dir = args.filtered_parts_dir or config.get("wout", {}).get("filtered_parts_dir")
    if not configured_parts_dir:
        raise ValueError(
            "A local wout parts directory is required. Pass --filtered-parts-dir "
            "or set wout.filtered_parts_dir in a private local config."
        )
    parts_dir = Path(configured_parts_dir).expanduser().resolve()
    parts = sorted_part_paths(parts_dir)
    if not parts:
        raise FileNotFoundError(f"No parquet parts found in {parts_dir}")

    print(json.dumps({"parts": len(parts), "parts_dir": str(parts_dir)}, sort_keys=True), flush=True)
    pca_models = load_or_fit_pca_models(parts, config)
    labels = build_wout_label_table(parts, pca_models, config)
    labels_path = OUTPUT_DIR / "dataset" / "wout24_labels.parquet"
    labels.to_parquet(labels_path, index=False)

    sample_map = build_sample_wout_map(dataset_dir, config)
    sample_map_path = OUTPUT_DIR / "dataset" / "sample_wout_map.parquet"
    sample_map.to_parquet(sample_map_path, index=False)

    finite_counts = {
        label: int(np.isfinite(pd.to_numeric(labels[label], errors="coerce")).sum())
        for label in WOUT24_LABELS
    }
    summary = {
        "hardware": hardware,
        "dataset_dir": str(dataset_dir),
        "parts_dir": str(parts_dir),
        "wout_label_rows": int(len(labels)),
        "sample_map_rows": int(len(sample_map)),
        "sample_map_with_wout_id": int(sample_map["wout_id"].notna().sum()),
        "sample_map_with_wout24_labels": int(sample_map["wout_id"].isin(set(labels["wout_id"])).sum()),
        "wout24_labels": WOUT24_LABELS,
        "finite_counts": finite_counts,
        "outputs": {
            "wout24_labels": str(labels_path),
            "sample_wout_map": str(sample_map_path),
            "pca_models": str(OUTPUT_DIR / "dataset" / "wout24_pca_models.joblib"),
        },
    }
    write_json(OUTPUT_DIR / "run_summary" / "wout24_label_build_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
