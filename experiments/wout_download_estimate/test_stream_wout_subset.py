from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import sys
import time
from pathlib import Path
from typing import Any


REPO_ID = "proxima-fusion/constellaration"
DEFAULT_CONFIG = "default"
WOUT_CONFIG = "vmecpp_wout"
TARGET_IDS_JSON = "target_wout_ids_nfp3_no_error.json"


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def json_size(row: dict[str, Any]) -> int:
    try:
        return len(json.dumps(row, separators=(",", ":"), default=str).encode("utf-8"))
    except Exception:
        return 0


def read_network_rx_bytes() -> int | None:
    path = Path("/proc/net/dev")
    if not path.exists():
        return None
    total = 0
    try:
        for line in path.read_text().splitlines()[2:]:
            if ":" not in line:
                continue
            iface, payload = line.split(":", 1)
            iface = iface.strip()
            if iface == "lo":
                continue
            fields = payload.split()
            if fields:
                total += int(fields[0])
        return total
    except Exception:
        return None


def nested_get(row: dict[str, Any], flat_key: str, parent_key: str, child_key: str) -> Any:
    if flat_key in row:
        return row[flat_key]
    parent = row.get(parent_key)
    if isinstance(parent, dict):
        return parent.get(child_key)
    return None


def detect_wout_id(row: dict[str, Any]) -> str | None:
    candidates = [
        "id",
        "vmecpp_wout_id",
        "vmecpp_wout.id",
        "wout.id",
        "plasma_config_id",
    ]
    for key in candidates:
        value = row.get(key)
        if value:
            return str(value)
    for parent_key in ("vmecpp_wout", "wout", "misc"):
        parent = row.get(parent_key)
        if isinstance(parent, dict):
            for child_key in ("id", "vmecpp_wout_id"):
                value = parent.get(child_key)
                if value:
                    return str(value)
    return None


def load_or_build_target_ids(args: argparse.Namespace) -> tuple[set[str], dict[str, Any]]:
    cache_path = Path(args.output_dir) / TARGET_IDS_JSON
    if args.reuse_target_ids and cache_path.exists():
        payload = json.loads(cache_path.read_text())
        return set(payload["target_wout_ids"]), {
            "source": "cache",
            "cache_path": str(cache_path),
            "target_wout_ids": len(payload["target_wout_ids"]),
            "target_default_rows": payload.get("target_default_rows"),
        }

    from datasets import load_dataset

    started = time.time()
    ds = load_dataset(
        REPO_ID,
        DEFAULT_CONFIG,
        split="train",
        cache_dir=args.cache_dir,
        streaming=args.default_streaming,
    )
    target_ids: set[str] = set()
    scanned = 0
    target_rows = 0
    missing_wout_id = 0
    for row in ds:
        scanned += 1
        if args.default_max_rows and scanned > args.default_max_rows:
            break
        nfp = nested_get(row, "boundary.n_field_periods", "boundary", "n_field_periods")
        has_error = nested_get(
            row,
            "misc.has_neurips_2025_forward_model_error",
            "misc",
            "has_neurips_2025_forward_model_error",
        )
        wout_id = nested_get(row, "misc.vmecpp_wout_id", "misc", "vmecpp_wout_id")
        if int(nfp or -1) == 3 and has_error is False:
            target_rows += 1
            if wout_id:
                target_ids.add(str(wout_id))
            else:
                missing_wout_id += 1

    payload = {
        "created_at": now(),
        "repo_id": REPO_ID,
        "default_config": DEFAULT_CONFIG,
        "default_rows_scanned": scanned,
        "target_default_rows": target_rows,
        "target_wout_ids": sorted(target_ids),
        "missing_wout_id": missing_wout_id,
        "seconds": time.time() - started,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return target_ids, {
        "source": "built_from_default",
        "cache_path": str(cache_path),
        "default_rows_scanned": scanned,
        "target_default_rows": target_rows,
        "target_wout_ids": len(target_ids),
        "missing_wout_id": missing_wout_id,
        "seconds": time.time() - started,
    }


def make_wout_stream(args: argparse.Namespace, worker_id: int, num_workers: int):
    from datasets import load_dataset

    ds = load_dataset(
        REPO_ID,
        WOUT_CONFIG,
        split="train",
        cache_dir=args.cache_dir,
        streaming=True,
    )
    shard_applied = False
    if num_workers > 1:
        if hasattr(ds, "shard"):
            ds = ds.shard(num_shards=num_workers, index=worker_id)
            shard_applied = True
        else:
            raise RuntimeError(
                "This datasets version does not support IterableDataset.shard; "
                "rerun with --workers 1."
            )
    return ds, shard_applied


def worker_scan(
    worker_id: int,
    num_workers: int,
    target_ids: set[str],
    args_dict: dict[str, Any],
    result_queue: mp.Queue,
) -> None:
    args = argparse.Namespace(**args_dict)
    started = time.time()
    rx0 = read_network_rx_bytes()
    stats = {
        "worker_id": worker_id,
        "rows_scanned": 0,
        "hits": 0,
        "unknown_id_rows": 0,
        "payload_bytes_seen": 0,
        "first_keys": None,
        "first_id": None,
        "hit_examples": [],
        "error": None,
        "shard_applied": False,
    }
    try:
        ds, shard_applied = make_wout_stream(args, worker_id, num_workers)
        stats["shard_applied"] = shard_applied
        per_worker_limit = args.max_wout_rows_per_worker
        deadline = started + args.seconds_per_worker if args.seconds_per_worker else None
        output_path = None
        output_handle = None
        if args.save_hits:
            output_path = Path(args.output_dir) / f"wout_hits_worker{worker_id:02d}.jsonl"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_handle = output_path.open("w")
            stats["output_path"] = str(output_path)
        try:
            for row in ds:
                stats["rows_scanned"] += 1
                if stats["first_keys"] is None:
                    stats["first_keys"] = sorted(row.keys())
                    stats["first_id"] = detect_wout_id(row)
                wout_id = detect_wout_id(row)
                if not wout_id:
                    stats["unknown_id_rows"] += 1
                if args.measure_payload_bytes:
                    stats["payload_bytes_seen"] += json_size(row)
                if wout_id in target_ids:
                    stats["hits"] += 1
                    if len(stats["hit_examples"]) < args.max_hit_examples:
                        stats["hit_examples"].append(
                            {
                                "wout_id": wout_id,
                                "row_index_in_worker_stream": stats["rows_scanned"],
                                "keys": sorted(row.keys()),
                                "approx_json_bytes": json_size(row),
                            }
                        )
                    if output_handle is not None:
                        output_handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
                if stats["rows_scanned"] % args.progress_every == 0:
                    result_queue.put({"type": "progress", **stats, "elapsed": time.time() - started})
                if per_worker_limit and stats["rows_scanned"] >= per_worker_limit:
                    break
                if deadline and time.time() >= deadline:
                    break
        finally:
            if output_handle is not None:
                output_handle.close()
    except Exception as exc:
        stats["error"] = f"{type(exc).__name__}: {exc}"
    rx1 = read_network_rx_bytes()
    stats["elapsed"] = time.time() - started
    stats["network_rx_bytes"] = (rx1 - rx0) if rx0 is not None and rx1 is not None else None
    result_queue.put({"type": "done", **stats})


def drain_progress(result_queue: mp.Queue, expected_done: int) -> list[dict[str, Any]]:
    done: list[dict[str, Any]] = []
    last_print = 0.0
    while len(done) < expected_done:
        try:
            item = result_queue.get(timeout=2.0)
        except queue.Empty:
            continue
        if item["type"] == "progress":
            if time.time() - last_print >= 5:
                print(
                    json.dumps(
                        {
                            "event": "progress",
                            "worker_id": item["worker_id"],
                            "rows_scanned": item["rows_scanned"],
                            "hits": item["hits"],
                            "elapsed": round(item["elapsed"], 2),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                last_print = time.time()
        elif item["type"] == "done":
            done.append(item)
            print(
                json.dumps(
                    {
                        "event": "worker_done",
                        "worker_id": item["worker_id"],
                        "rows_scanned": item["rows_scanned"],
                        "hits": item["hits"],
                        "elapsed": round(item["elapsed"], 2),
                        "error": item["error"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return done


def summarize(args: argparse.Namespace, target_meta: dict[str, Any], worker_results: list[dict[str, Any]]) -> dict[str, Any]:
    rows = sum(int(item["rows_scanned"]) for item in worker_results)
    hits = sum(int(item["hits"]) for item in worker_results)
    elapsed = max((float(item["elapsed"]) for item in worker_results), default=0.0)
    payload_bytes = sum(int(item.get("payload_bytes_seen") or 0) for item in worker_results)
    network_bytes_values = [item.get("network_rx_bytes") for item in worker_results]
    network_bytes = None
    if all(value is not None for value in network_bytes_values):
        # Multiple processes observe the same host-level counter, so use max, not sum.
        network_bytes = max(int(value) for value in network_bytes_values)
    rows_per_second = rows / elapsed if elapsed > 0 else None
    hit_rate = hits / rows if rows else None
    expected_rows_for_all_targets = None
    expected_seconds_for_all_targets = None
    target_count = int(target_meta.get("target_wout_ids") or 0)
    if hit_rate and rows_per_second:
        expected_rows_for_all_targets = target_count / hit_rate
        expected_seconds_for_all_targets = expected_rows_for_all_targets / rows_per_second
    return {
        "created_at": now(),
        "repo_id": REPO_ID,
        "wout_config": WOUT_CONFIG,
        "args": vars(args),
        "target_meta": target_meta,
        "totals": {
            "rows_scanned": rows,
            "hits": hits,
            "elapsed_wall_seconds": elapsed,
            "rows_per_second": rows_per_second,
            "hit_rate": hit_rate,
            "payload_bytes_seen": payload_bytes,
            "payload_mb_per_second": (payload_bytes / 1_000_000 / elapsed) if elapsed > 0 else None,
            "network_rx_bytes": network_bytes,
            "network_mb_per_second": (network_bytes / 1_000_000 / elapsed)
            if network_bytes is not None and elapsed > 0
            else None,
            "expected_rows_for_all_targets_at_observed_hit_rate": expected_rows_for_all_targets,
            "expected_seconds_for_all_targets_at_observed_rate": expected_seconds_for_all_targets,
        },
        "worker_results": worker_results,
        "schema_guess": {
            "first_keys_by_worker": [
                {"worker_id": item["worker_id"], "first_keys": item.get("first_keys")}
                for item in worker_results
            ],
            "first_ids_by_worker": [
                {"worker_id": item["worker_id"], "first_id": item.get("first_id")}
                for item in worker_results
            ],
        },
        "notes": [
            "This is a streaming scan test. It avoids full local cache, but network reads may still touch many shards.",
            "If multiple workers report shard_applied=true, the streaming dataset was split across workers.",
            "network_rx_bytes is a host-level Linux counter and should be treated as approximate.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output-dir", default="experiments/wout_download_estimate/outputs")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-wout-rows-per-worker", type=int, default=200)
    parser.add_argument("--seconds-per-worker", type=float, default=0.0)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--default-max-rows", type=int, default=0)
    parser.add_argument("--default-streaming", action="store_true")
    parser.add_argument("--reuse-target-ids", action="store_true")
    parser.add_argument("--save-hits", action="store_true")
    parser.add_argument("--measure-payload-bytes", action="store_true")
    parser.add_argument("--max-hit-examples", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(json.dumps({"event": "start", "time": now(), "args": vars(args)}, sort_keys=True), flush=True)
    try:
        target_ids, target_meta = load_or_build_target_ids(args)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "event": "target_id_load_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        raise
    print(json.dumps({"event": "target_ids_ready", **target_meta}, sort_keys=True), flush=True)
    if not target_ids:
        raise RuntimeError("No target wout ids were found; cannot test streaming filter.")

    ctx = mp.get_context("spawn")
    result_queue: mp.Queue = ctx.Queue()
    args_dict = vars(args)
    workers = max(1, int(args.workers))
    processes = [
        ctx.Process(target=worker_scan, args=(worker_id, workers, target_ids, args_dict, result_queue))
        for worker_id in range(workers)
    ]
    for process in processes:
        process.start()
    worker_results = drain_progress(result_queue, expected_done=workers)
    for process in processes:
        process.join()
    summary = summarize(args, target_meta, worker_results)
    summary_path = Path(args.output_dir) / "stream_wout_subset_test_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str))
    print(json.dumps({"event": "summary_written", "path": str(summary_path)}, sort_keys=True), flush=True)
    print(json.dumps(summary["totals"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
