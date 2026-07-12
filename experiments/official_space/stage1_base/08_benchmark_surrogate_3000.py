from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from candidate_utils import (
    OUTPUT_DIR,
    SupportModel,
    constraint_support_metrics,
    feature_columns,
    load_model_bundle,
    predict_ensemble,
    prediction_value,
    train_bounds,
)
from common import apply_thread_environment, ensure_output_dirs, load_config, write_json


EXPERIMENT_DIR = Path(__file__).resolve().parent


def load_stage1_generation_module():
    path = EXPERIMENT_DIR / "04_run_conservative_cmaes.py"
    spec = importlib.util.spec_from_file_location("stage1_generation", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark 3000 surrogate evaluations and surrogate-driven candidate generation."
    )
    parser.add_argument("--config", default="configs/quick.yaml")
    parser.add_argument("--budget", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Only benchmark batched and single-sample surrogate forward passes.",
    )
    return parser.parse_args()


def sync_cuda(bundle: dict[str, Any]) -> None:
    try:
        import torch

        if bundle["device"].type == "cuda":
            torch.cuda.synchronize(bundle["device"])
    except Exception:
        pass


def reset_peak_memory(bundle: dict[str, Any]) -> None:
    try:
        import torch

        if bundle["device"].type == "cuda":
            torch.cuda.reset_peak_memory_stats(bundle["device"])
    except Exception:
        pass


def gpu_memory_snapshot(bundle: dict[str, Any]) -> dict[str, Any]:
    try:
        import torch

        if bundle["device"].type != "cuda":
            return {}
        index = bundle["device"].index if bundle["device"].index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        return {
            "device_index": int(index),
            "device_name": props.name,
            "total_memory_gib": props.total_memory / 1024**3,
            "max_memory_allocated_mib": torch.cuda.max_memory_allocated(index) / 1024**2,
            "max_memory_reserved_mib": torch.cuda.max_memory_reserved(index) / 1024**2,
            "memory_allocated_mib": torch.cuda.memory_allocated(index) / 1024**2,
            "memory_reserved_mib": torch.cuda.memory_reserved(index) / 1024**2,
        }
    except Exception as exc:
        return {"error": str(exc)}


def nvidia_smi_snapshot() -> str:
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def summarize_times(times: list[float], n: int) -> dict[str, Any]:
    arr = np.asarray(times, dtype=np.float64)
    return {
        "repeats": len(times),
        "total_seconds_mean": float(arr.mean()),
        "total_seconds_median": float(np.median(arr)),
        "total_seconds_min": float(arr.min()),
        "total_seconds_max": float(arr.max()),
        "evals_per_second_mean": float(n / arr.mean()),
        "milliseconds_per_eval_mean": float(arr.mean() * 1000.0 / n),
        "milliseconds_per_eval_median": float(np.median(arr) * 1000.0 / n),
        "all_total_seconds": [float(x) for x in times],
    }


def benchmark_batched_forward(
    bundle: dict[str, Any],
    x: np.ndarray,
    *,
    repeats: int,
    warmup: int,
    batch_size: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        predict_ensemble(bundle, x, batch_size=batch_size)
    sync_cuda(bundle)
    reset_peak_memory(bundle)
    times = []
    for _ in range(repeats):
        sync_cuda(bundle)
        start = time.perf_counter()
        prediction = predict_ensemble(bundle, x, batch_size=batch_size)
        sync_cuda(bundle)
        times.append(time.perf_counter() - start)
    payload = summarize_times(times, x.shape[0])
    payload["batch_size"] = batch_size
    payload["ensemble_members"] = len(bundle["models"])
    payload["output_checksum"] = float(np.mean(prediction["mean"]))
    payload["gpu_memory"] = gpu_memory_snapshot(bundle)
    return payload


def benchmark_single_forward_loop(
    bundle: dict[str, Any],
    x: np.ndarray,
    *,
    repeats: int,
    warmup: int,
) -> dict[str, Any]:
    subset = x
    for _ in range(warmup):
        for row in subset[: min(16, subset.shape[0])]:
            predict_ensemble(bundle, row, batch_size=1)
    sync_cuda(bundle)
    reset_peak_memory(bundle)
    times = []
    checksum = 0.0
    for _ in range(repeats):
        sync_cuda(bundle)
        start = time.perf_counter()
        for row in subset:
            prediction = predict_ensemble(bundle, row, batch_size=1)
            checksum += prediction_value(prediction, "L_gradB")
        sync_cuda(bundle)
        times.append(time.perf_counter() - start)
    payload = summarize_times(times, subset.shape[0])
    payload["batch_size"] = 1
    payload["ensemble_members"] = len(bundle["models"])
    payload["output_checksum"] = checksum / max(1, repeats * subset.shape[0])
    payload["gpu_memory"] = gpu_memory_snapshot(bundle)
    return payload


def benchmark_generation_loop(
    *,
    method_id: str,
    seed: int,
    budget: int,
    bundle: dict[str, Any],
    train_x: np.ndarray,
    validation_x: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    relaxed_x: np.ndarray | None,
) -> dict[str, Any]:
    import nevergrad as ng

    stage1 = load_stage1_generation_module()
    geometry_filter = stage1.GeometryFilter(train_x)
    support_model = SupportModel(train_x, validation_x, n_components=20) if method_id == "E3" else None
    source_x = train_x if method_id == "E2" or relaxed_x is None or relaxed_x.size == 0 else relaxed_x
    n_components = 20 if method_id == "E2" else min(10, source_x.shape[0] - 1)
    latent_space = stage1.LatentSearchSpace(
        source_x,
        lower,
        upper,
        n_components=n_components,
        seed=seed,
    )
    init_x = source_x[0]
    init_z = latent_space.encode(init_x)
    parametrization = ng.p.Array(init=init_z).set_bounds(latent_space.lower_z, latent_space.upper_z)
    optimizer_cls = ng.optimizers.CMA if method_id == "E3" else ng.optimizers.NGOpt
    optimizer = optimizer_cls(parametrization=parametrization, budget=budget, num_workers=1)
    optimizer.parametrization.random_state.seed(seed)

    accepted = 0
    rejected_by_geometry = 0
    score_checksum = 0.0
    sync_cuda(bundle)
    reset_peak_memory(bundle)
    start = time.perf_counter()
    for _ in range(budget):
        candidate = optimizer.ask()
        z = np.asarray(candidate.value, dtype=np.float32)
        x = latent_space.decode(z)
        geometry_pass, geometry_metrics = geometry_filter.evaluate(x)
        if not geometry_pass:
            rejected_by_geometry += 1
            optimizer.tell(candidate, 1e6)
            continue
        prediction = predict_ensemble(bundle, x, batch_size=1)
        support = constraint_support_metrics(prediction)
        support_penalty = float(support_model.penalty(x)[0]) if support_model is not None else 0.0
        support.update(geometry_metrics)
        support["support_penalty"] = support_penalty
        score = stage1.feasibility_first_score(
            method_id,
            prediction,
            support,
            support_penalty=support_penalty,
            geometry_penalty=float(geometry_metrics["geometry_penalty"]),
        )
        optimizer.tell(candidate, -score)
        accepted += 1
        score_checksum += float(score)
    sync_cuda(bundle)
    elapsed = time.perf_counter() - start
    return {
        "method_id": method_id,
        "budget": budget,
        "accepted_evaluations": accepted,
        "rejected_by_geometry": rejected_by_geometry,
        "total_seconds": elapsed,
        "evals_per_second_attempted": float(budget / elapsed),
        "evals_per_second_accepted": float(accepted / elapsed) if accepted else None,
        "milliseconds_per_attempted_eval": float(elapsed * 1000.0 / budget),
        "milliseconds_per_accepted_eval": float(elapsed * 1000.0 / accepted) if accepted else None,
        "latent_space": latent_space.summary(),
        "geometry_filter": geometry_filter.summary(),
        "score_checksum": score_checksum,
        "gpu_memory": gpu_memory_snapshot(bundle),
    }


def load_relaxed_x(x_cols: list[str]) -> np.ndarray | None:
    path = OUTPUT_DIR / "dataset" / "relaxed55.jsonl"
    if not path.exists():
        return None
    frame = pd.DataFrame(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    if frame.empty:
        return None
    return frame[x_cols].to_numpy(dtype=np.float32)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    hardware = apply_thread_environment(config)
    ensure_output_dirs()

    import torch

    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    setup_start = time.perf_counter()
    train = pd.read_parquet(OUTPUT_DIR / "dataset" / "train.parquet")
    validation = pd.read_parquet(OUTPUT_DIR / "dataset" / "validation.parquet")
    x_cols = feature_columns(train)
    train_x = train[x_cols].to_numpy(dtype=np.float32)
    validation_x = validation[x_cols].to_numpy(dtype=np.float32)
    lower, upper = train_bounds(train, x_cols)
    relaxed_x = load_relaxed_x(x_cols)
    bundle = load_model_bundle(args.device)
    sample_x = train_x[: args.budget].copy()
    setup_seconds = time.perf_counter() - setup_start

    result: dict[str, Any] = {
        "started_at": started,
        "benchmark": "surrogate_3000_evaluation",
        "budget": args.budget,
        "config_path": args.config,
        "hardware_config": hardware,
        "runtime_environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "torch_version": torch.__version__,
            "torch_cuda_available": bool(torch.cuda.is_available()),
            "torch_cuda_version": getattr(torch.version, "cuda", None),
            "inference_device": str(bundle["device"]),
            "nvidia_smi": nvidia_smi_snapshot(),
        },
        "model": {
            "ensemble_members": len(bundle["models"]),
            "input_dim": int(sample_x.shape[1]),
            "labels": bundle["labels"],
            "note": "Candidate generation loads all ensemble members onto one inference device by default.",
        },
        "data": {
            "train_rows": int(train_x.shape[0]),
            "validation_rows": int(validation_x.shape[0]),
            "relaxed_rows": int(relaxed_x.shape[0]) if relaxed_x is not None else 0,
        },
        "setup_seconds": setup_seconds,
    }

    result["batched_forward_3000"] = benchmark_batched_forward(
        bundle,
        sample_x,
        repeats=args.repeats,
        warmup=args.warmup,
        batch_size=args.batch_size,
    )
    result["single_forward_loop_3000"] = benchmark_single_forward_loop(
        bundle,
        sample_x,
        repeats=max(1, min(args.repeats, 3)),
        warmup=args.warmup,
    )
    if not args.skip_generation:
        result["sequential_generation_3000"] = {
            "E2": benchmark_generation_loop(
                method_id="E2",
                seed=int(config.get("seed", 0)),
                budget=args.budget,
                bundle=bundle,
                train_x=train_x,
                validation_x=validation_x,
                lower=lower,
                upper=upper,
                relaxed_x=relaxed_x,
            ),
            "E3": benchmark_generation_loop(
                method_id="E3",
                seed=int(config.get("seed", 0)),
                budget=args.budget,
                bundle=bundle,
                train_x=train_x,
                validation_x=validation_x,
                lower=lower,
                upper=upper,
                relaxed_x=relaxed_x,
            ),
        }

    usage = resource.getrusage(resource.RUSAGE_SELF)
    result["process_resource_usage"] = {
        "max_rss_kib": int(usage.ru_maxrss),
        "user_cpu_seconds": float(usage.ru_utime),
        "system_cpu_seconds": float(usage.ru_stime),
    }
    result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    out = Path(args.out) if args.out else OUTPUT_DIR / "run_summary" / "surrogate_3000_benchmark.json"
    write_json(out, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
