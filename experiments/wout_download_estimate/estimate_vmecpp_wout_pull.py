from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from typing import Any


REPO_ID = "proxima-fusion/constellaration"
WOUT_PATH = "vmecpp_wout"
DEFAULT_CONFIG = "default"
DEFAULT_KNOWN_WOUT_SIZE_GB = 767.0
DEFAULT_KNOWN_RAW_ROWS = 182_222
DEFAULT_KNOWN_TARGET_ROWS = 68_191


def gb_to_bytes(value: float) -> int:
    return int(value * 1_000_000_000)


def format_bytes(num_bytes: float | None) -> str:
    if num_bytes is None:
        return "unknown"
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if abs(value) < 1000.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1000.0
    return f"{value:.2f} TB"


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f} s"
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    if seconds < 86400:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 86400:.1f} d"


@dataclass
class HubTreeSummary:
    total_bytes: int | None
    file_count: int | None
    source: str
    error: str | None = None


def read_hub_tree(timeout_seconds: float) -> HubTreeSummary:
    try:
        from huggingface_hub import HfApi
    except Exception as exc:
        return HubTreeSummary(
            total_bytes=None,
            file_count=None,
            source="unavailable",
            error=f"cannot import huggingface_hub: {type(exc).__name__}: {exc}",
        )

    started = time.time()
    try:
        api = HfApi()
        files = []
        for item in api.list_repo_tree(
            repo_id=REPO_ID,
            path_in_repo=WOUT_PATH,
            repo_type="dataset",
            recursive=False,
            expand=True,
        ):
            if time.time() - started > timeout_seconds:
                raise TimeoutError(f"Hub tree query exceeded {timeout_seconds:.1f}s")
            size = getattr(item, "size", None)
            path = getattr(item, "path", "")
            if size is not None and path.endswith(".parquet"):
                files.append((path, int(size)))
        return HubTreeSummary(
            total_bytes=sum(size for _, size in files),
            file_count=len(files),
            source="huggingface_hub",
        )
    except Exception as exc:
        return HubTreeSummary(
            total_bytes=None,
            file_count=None,
            source="failed",
            error=f"{type(exc).__name__}: {exc}",
        )


@dataclass
class DefaultScanSummary:
    raw_rows: int
    mapped_wout_rows: int | None
    unique_wout_ids: int | None
    target_rows: int
    unique_target_wout_ids: int | None
    source: str
    error: str | None = None


def row_get(row: dict[str, Any], flat_key: str, nested_key: str, child_key: str) -> Any:
    if flat_key in row:
        return row[flat_key]
    parent = row.get(nested_key)
    if isinstance(parent, dict):
        return parent.get(child_key)
    return None


def scan_default_dataset(cache_dir: str | None, timeout_seconds: float) -> DefaultScanSummary:
    try:
        from datasets import load_dataset
    except Exception as exc:
        return DefaultScanSummary(
            raw_rows=DEFAULT_KNOWN_RAW_ROWS,
            mapped_wout_rows=None,
            unique_wout_ids=None,
            target_rows=DEFAULT_KNOWN_TARGET_ROWS,
            unique_target_wout_ids=None,
            source="known_counts_fallback",
            error=f"cannot import datasets: {type(exc).__name__}: {exc}",
        )

    started = time.time()
    try:
        dataset = load_dataset(
            REPO_ID,
            DEFAULT_CONFIG,
            split="train",
            cache_dir=cache_dir,
        )
        raw_rows = len(dataset)
        all_ids: set[str] = set()
        target_ids: set[str] = set()
        target_rows = 0
        mapped_rows = 0
        for row in dataset:
            if time.time() - started > timeout_seconds:
                raise TimeoutError(f"default scan exceeded {timeout_seconds:.1f}s")
            wout_id = row_get(row, "misc.vmecpp_wout_id", "misc", "vmecpp_wout_id")
            nfp = row_get(row, "boundary.n_field_periods", "boundary", "n_field_periods")
            has_error = row_get(
                row,
                "misc.has_neurips_2025_forward_model_error",
                "misc",
                "has_neurips_2025_forward_model_error",
            )
            if wout_id:
                mapped_rows += 1
                all_ids.add(str(wout_id))
            if int(nfp or -1) == 3 and has_error is False:
                target_rows += 1
                if wout_id:
                    target_ids.add(str(wout_id))
        return DefaultScanSummary(
            raw_rows=raw_rows,
            mapped_wout_rows=mapped_rows,
            unique_wout_ids=len(all_ids),
            target_rows=target_rows,
            unique_target_wout_ids=len(target_ids),
            source="datasets_scan",
        )
    except Exception as exc:
        return DefaultScanSummary(
            raw_rows=DEFAULT_KNOWN_RAW_ROWS,
            mapped_wout_rows=None,
            unique_wout_ids=None,
            target_rows=DEFAULT_KNOWN_TARGET_ROWS,
            unique_target_wout_ids=None,
            source="known_counts_fallback",
            error=f"{type(exc).__name__}: {exc}",
        )


def build_estimate(
    hub: HubTreeSummary,
    default: DefaultScanSummary,
    known_full_size_gb: float,
    speeds_mbps: list[float],
) -> dict[str, Any]:
    full_bytes = hub.total_bytes or gb_to_bytes(known_full_size_gb)
    full_size_source = hub.source if hub.total_bytes else "known_size_fallback"
    target_ids = default.unique_target_wout_ids or default.target_rows
    total_ids = default.unique_wout_ids or default.mapped_wout_rows or default.raw_rows
    subset_fraction = target_ids / total_ids
    subset_raw_wout_bytes = full_bytes * subset_fraction

    expected_file_touch_fraction = None
    if hub.file_count and hub.file_count > 0:
        rows_per_file = max(default.raw_rows / hub.file_count, 1.0)
        row_fraction = default.target_rows / default.raw_rows
        expected_file_touch_fraction = 1.0 - math.pow(1.0 - row_fraction, rows_per_file)

    time_estimates = []
    for mbps in speeds_mbps:
        bytes_per_second = mbps * 1_000_000 / 8.0
        seconds_full = full_bytes / bytes_per_second
        seconds_subset_payload = subset_raw_wout_bytes / bytes_per_second
        time_estimates.append(
            {
                "network_mbps": mbps,
                "full_wout_download": format_duration(seconds_full),
                "linear_subset_payload_only": format_duration(seconds_subset_payload),
            }
        )

    return {
        "repo_id": REPO_ID,
        "wout_path": WOUT_PATH,
        "hub_tree": hub.__dict__,
        "default_scan": default.__dict__,
        "full_wout_size": {
            "bytes": full_bytes,
            "human": format_bytes(full_bytes),
            "source": full_size_source,
        },
        "subset_estimate": {
            "target_rows_or_ids": target_ids,
            "total_rows_or_ids": total_ids,
            "fraction": subset_fraction,
            "raw_wout_payload_bytes": int(subset_raw_wout_bytes),
            "raw_wout_payload_human": format_bytes(subset_raw_wout_bytes),
            "expected_file_touch_fraction_if_uniform": expected_file_touch_fraction,
        },
        "disk_budget": {
            "minimum_if_streaming_and_saving_subset_only": format_bytes(subset_raw_wout_bytes),
            "practical_if_download_full_then_save_subset_copy": format_bytes(
                full_bytes + subset_raw_wout_bytes
            ),
            "safe_scratch_budget_for_hf_cache_plus_processing": format_bytes(
                full_bytes * 1.5 + subset_raw_wout_bytes
            ),
            "full_wout_parquet_cache_only": format_bytes(full_bytes),
        },
        "time_estimates": time_estimates,
        "notes": [
            "The linear subset payload is only the final retained raw wout data estimate.",
            "If the target rows are spread across most parquet shards, extraction still needs reading nearly the full vmecpp_wout directory.",
            "Saving derived low-dimensional wout features can be much smaller than saving raw wout JSON.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--hub-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--default-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--known-full-size-gb", type=float, default=DEFAULT_KNOWN_WOUT_SIZE_GB)
    parser.add_argument(
        "--speeds-mbps",
        type=float,
        nargs="+",
        default=[10.0, 20.0, 50.0, 100.0, 200.0],
    )
    args = parser.parse_args()

    hub = read_hub_tree(args.hub_timeout_seconds)
    default = scan_default_dataset(args.cache_dir, args.default_timeout_seconds)
    estimate = build_estimate(hub, default, args.known_full_size_gb, args.speeds_mbps)
    print(json.dumps(estimate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
