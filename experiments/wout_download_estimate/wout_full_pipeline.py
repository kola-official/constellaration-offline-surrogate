from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ID = "proxima-fusion/constellaration"
WOUT_DIR = "vmecpp_wout"
DEFAULT_ENDPOINT = "https://hf-mirror.com"


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def load_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    tmp.replace(path)


def load_target_ids(path: Path) -> set[str]:
    payload = load_json(path)
    if isinstance(payload, list):
        values = payload
    elif isinstance(payload, dict):
        for key in ("target_wout_ids", "ids", "target_ids"):
            value = payload.get(key)
            if isinstance(value, list):
                values = value
                break
        else:
            raise ValueError(f"cannot find target id list in {path}")
    else:
        raise ValueError(f"unsupported target id JSON in {path}: {type(payload).__name__}")
    return {str(value) for value in values}


@dataclass(frozen=True)
class RemoteFile:
    path: str
    size: int


def read_tree_files(tree_json: Path, include_id_map: bool = True) -> list[RemoteFile]:
    files = []
    for item in load_json(tree_json):
        path = str(item.get("path", ""))
        if item.get("type") != "file":
            continue
        if not path.startswith(f"{WOUT_DIR}/") or not path.endswith(".parquet"):
            continue
        if not include_id_map and path.endswith("id_to_file_map.parquet"):
            continue
        files.append(RemoteFile(path=path, size=int(item.get("size") or 0)))
    files.sort(key=lambda item: item.path)
    return files


def local_path(download_dir: Path, remote_path: str) -> Path:
    return download_dir / remote_path


def file_size(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return None


def is_complete(path: Path, expected_size: int) -> bool:
    return file_size(path) == expected_size


def download_one(
    item: RemoteFile,
    download_dir: Path,
    endpoint: str,
    repo_id: str,
    curl_path: str,
    retries: int,
) -> dict[str, Any]:
    final_path = local_path(download_dir, item.path)
    partial_path = final_path.with_name(final_path.name + ".part")
    final_path.parent.mkdir(parents=True, exist_ok=True)

    if is_complete(final_path, item.size):
        return {
            "path": item.path,
            "status": "skipped_complete",
            "bytes": item.size,
            "seconds": 0.0,
        }

    if final_path.exists() and file_size(final_path) != item.size:
        if partial_path.exists() and file_size(partial_path):
            # Keep the larger partial file for resume.
            if (file_size(final_path) or 0) > (file_size(partial_path) or 0):
                final_path.replace(partial_path)
            else:
                final_path.unlink()
        else:
            final_path.replace(partial_path)

    url = f"{endpoint.rstrip('/')}/datasets/{repo_id}/resolve/main/{item.path}"
    started = time.time()
    cmd = [
        curl_path,
        "--fail",
        "--location",
        "--show-error",
        "--silent",
        "--retry",
        str(retries),
        "--retry-all-errors",
        "--retry-delay",
        "5",
        "--connect-timeout",
        "60",
        "--continue-at",
        "-",
        "--output",
        str(partial_path),
        url,
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True)
    seconds = time.time() - started
    if proc.returncode != 0:
        return {
            "path": item.path,
            "status": "failed",
            "returncode": proc.returncode,
            "stderr": proc.stderr[-4000:],
            "stdout": proc.stdout[-1000:],
            "partial_bytes": file_size(partial_path),
            "expected_bytes": item.size,
            "seconds": seconds,
        }

    actual = file_size(partial_path)
    if actual != item.size:
        return {
            "path": item.path,
            "status": "size_mismatch",
            "partial_bytes": actual,
            "expected_bytes": item.size,
            "seconds": seconds,
        }

    partial_path.replace(final_path)
    return {
        "path": item.path,
        "status": "downloaded",
        "bytes": item.size,
        "seconds": seconds,
    }


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def summarize_download(files: list[RemoteFile], download_dir: Path) -> dict[str, Any]:
    complete = 0
    partial = 0
    missing = 0
    complete_bytes = 0
    partial_bytes = 0
    bad_size: list[dict[str, Any]] = []
    for item in files:
        final_path = local_path(download_dir, item.path)
        partial_path = final_path.with_name(final_path.name + ".part")
        size = file_size(final_path)
        if size == item.size:
            complete += 1
            complete_bytes += item.size
        elif size is None:
            part_size = file_size(partial_path)
            if part_size is None:
                missing += 1
            else:
                partial += 1
                partial_bytes += part_size
        else:
            partial += 1
            partial_bytes += size
            bad_size.append({"path": item.path, "size": size, "expected": item.size})
    return {
        "created_at": now(),
        "download_dir": str(download_dir),
        "files_total": len(files),
        "files_complete": complete,
        "files_partial_or_bad": partial,
        "files_missing": missing,
        "bytes_expected": sum(item.size for item in files),
        "bytes_complete": complete_bytes,
        "bytes_partial": partial_bytes,
        "bad_size_examples": bad_size[:20],
    }


def command_download(args: argparse.Namespace) -> int:
    files = read_tree_files(args.tree_json, include_id_map=True)
    args.download_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.download_dir / "download_events.jsonl"
    status_path = args.download_dir / "download_status.json"

    dump_json(
        status_path,
        {
            **summarize_download(files, args.download_dir),
            "status": "starting",
            "workers": args.workers,
            "endpoint": args.endpoint,
        },
    )

    print(f"[{now()}] download start files={len(files)} workers={args.workers}", flush=True)
    failures: list[dict[str, Any]] = []
    started = time.time()
    completed_jobs = 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_file = {
            executor.submit(
                download_one,
                item,
                args.download_dir,
                args.endpoint,
                args.repo_id,
                args.curl_path,
                args.retries,
            ): item
            for item in files
        }
        for future in cf.as_completed(future_to_file):
            completed_jobs += 1
            result = future.result()
            append_jsonl(log_path, {"created_at": now(), **result})
            status = result.get("status")
            if status not in {"downloaded", "skipped_complete"}:
                failures.append(result)
            if completed_jobs % args.progress_every == 0 or failures:
                summary = summarize_download(files, args.download_dir)
                dump_json(
                    status_path,
                    {
                        **summary,
                        "status": "running",
                        "workers": args.workers,
                        "endpoint": args.endpoint,
                        "jobs_seen": completed_jobs,
                        "failures": len(failures),
                        "failure_examples": failures[-10:],
                        "elapsed_seconds": time.time() - started,
                    },
                )
                print(
                    f"[{now()}] jobs={completed_jobs}/{len(files)} "
                    f"complete={summary['files_complete']} "
                    f"partial={summary['files_partial_or_bad']} "
                    f"missing={summary['files_missing']} failures={len(failures)}",
                    flush=True,
                )

    summary = summarize_download(files, args.download_dir)
    ok = summary["files_complete"] == summary["files_total"] and not failures
    dump_json(
        status_path,
        {
            **summary,
            "status": "complete" if ok else "incomplete",
            "workers": args.workers,
            "endpoint": args.endpoint,
            "failures": len(failures),
            "failure_examples": failures[-20:],
            "elapsed_seconds": time.time() - started,
        },
    )
    print(f"[{now()}] download {'complete' if ok else 'incomplete'}", flush=True)
    return 0 if ok else 2


def detect_id_column(schema_names: list[str]) -> str:
    for name in ("id", "vmecpp_wout_id", "wout_id", "plasma_config_id"):
        if name in schema_names:
            return name
    raise ValueError(f"cannot detect id column in schema: {schema_names}")


def filter_one_part(
    part_path: Path,
    output_dir: Path,
    target_ids: set[str],
    compression: str,
) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    started = time.time()
    table = pq.read_table(part_path)
    id_col = detect_id_column(table.schema.names)
    ids = table[id_col]
    mask = pc.is_in(ids, value_set=pa.array(sorted(target_ids), type=pa.string()))
    filtered = table.filter(mask)
    hit_count = filtered.num_rows
    out_path = output_dir / part_path.name
    seen_ids: list[str] = []
    if hit_count:
        output_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(filtered, out_path, compression=compression)
        seen_ids = [str(value) for value in filtered[id_col].to_pylist()]
    elif out_path.exists():
        out_path.unlink()
    return {
        "part": part_path.name,
        "rows": table.num_rows,
        "hits": hit_count,
        "id_column": id_col,
        "output_path": str(out_path) if hit_count else None,
        "output_bytes": file_size(out_path) if hit_count else 0,
        "seen_ids": seen_ids,
        "seconds": time.time() - started,
    }


def command_filter(args: argparse.Namespace) -> int:
    files = [
        item
        for item in read_tree_files(args.tree_json, include_id_map=False)
        if item.path.endswith(".parquet")
    ]
    status = summarize_download(files, args.download_dir)
    if args.require_complete and status["files_complete"] != status["files_total"]:
        dump_json(args.output_dir / "filter_summary.json", {**status, "status": "download_incomplete"})
        print(
            f"[{now()}] refusing to filter: complete={status['files_complete']} "
            f"total={status['files_total']}",
            flush=True,
        )
        return 3

    target_ids = load_target_ids(args.target_ids_json)
    part_paths = [local_path(args.download_dir, item.path) for item in files]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    events_path = args.output_dir / "filter_events.jsonl"
    summary_path = args.output_dir / "filter_summary.json"
    if events_path.exists() and not args.append_events:
        events_path.unlink()

    started = time.time()
    seen_ids: set[str] = set()
    total_rows = 0
    total_hits = 0
    outputs = 0
    errors: list[dict[str, Any]] = []
    print(
        f"[{now()}] filter start parts={len(part_paths)} targets={len(target_ids)} "
        f"workers={args.workers}",
        flush=True,
    )
    with cf.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_path = {
            executor.submit(
                filter_one_part,
                path,
                args.output_dir / "parts",
                target_ids,
                args.compression,
            ): path
            for path in part_paths
        }
        for index, future in enumerate(cf.as_completed(future_to_path), start=1):
            path = future_to_path[future]
            try:
                result = future.result()
                seen_ids.update(result.pop("seen_ids"))
                total_rows += int(result["rows"])
                total_hits += int(result["hits"])
                outputs += 1 if result.get("output_path") else 0
                append_jsonl(events_path, {"created_at": now(), **result})
            except Exception as exc:
                err = {"part": path.name, "error": f"{type(exc).__name__}: {exc}"}
                errors.append(err)
                append_jsonl(events_path, {"created_at": now(), **err})
            if index % args.progress_every == 0 or errors:
                missing_count = len(target_ids - seen_ids)
                dump_json(
                    summary_path,
                    {
                        "created_at": now(),
                        "status": "running",
                        "parts_seen": index,
                        "parts_total": len(part_paths),
                        "target_ids": len(target_ids),
                        "matched_ids_so_far": len(seen_ids),
                        "missing_ids_so_far": missing_count,
                        "rows_scanned_so_far": total_rows,
                        "hits_so_far": total_hits,
                        "output_part_files_so_far": outputs,
                        "errors": len(errors),
                        "error_examples": errors[-10:],
                        "elapsed_seconds": time.time() - started,
                    },
                )
                print(
                    f"[{now()}] filter parts={index}/{len(part_paths)} "
                    f"matched={len(seen_ids)} missing={missing_count} errors={len(errors)}",
                    flush=True,
                )

    missing_ids = sorted(target_ids - seen_ids)
    dump_json(args.output_dir / "missing_target_wout_ids.json", missing_ids)
    summary = {
        "created_at": now(),
        "status": "complete" if not errors else "complete_with_errors",
        "download_dir": str(args.download_dir),
        "output_dir": str(args.output_dir),
        "parts_total": len(part_paths),
        "target_ids": len(target_ids),
        "matched_ids": len(seen_ids),
        "missing_ids": len(missing_ids),
        "rows_scanned": total_rows,
        "hits": total_hits,
        "output_part_files": outputs,
        "errors": len(errors),
        "error_examples": errors[-20:],
        "elapsed_seconds": time.time() - started,
    }
    dump_json(summary_path, summary)
    print(
        f"[{now()}] filter {summary['status']} matched={len(seen_ids)} "
        f"missing={len(missing_ids)} hits={total_hits}",
        flush=True,
    )
    return 0 if not errors else 4


def command_status(args: argparse.Namespace) -> int:
    files = read_tree_files(args.tree_json, include_id_map=True)
    dump_json(args.download_dir / "download_status.json", summarize_download(files, args.download_dir))
    print(json.dumps(summarize_download(files, args.download_dir), indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", DEFAULT_ENDPOINT))
    subparsers = parser.add_subparsers(dest="command", required=True)

    common_download = argparse.ArgumentParser(add_help=False)
    common_download.add_argument("--tree-json", type=Path, required=True)
    common_download.add_argument("--download-dir", type=Path, required=True)

    download = subparsers.add_parser("download", parents=[common_download])
    download.add_argument("--workers", type=int, default=16)
    download.add_argument("--curl-path", default=shutil.which("curl") or "curl")
    download.add_argument("--retries", type=int, default=20)
    download.add_argument("--progress-every", type=int, default=20)
    download.set_defaults(func=command_download)

    filt = subparsers.add_parser("filter", parents=[common_download])
    filt.add_argument("--target-ids-json", type=Path, required=True)
    filt.add_argument("--output-dir", type=Path, required=True)
    filt.add_argument("--workers", type=int, default=4)
    filt.add_argument("--compression", default="zstd")
    filt.add_argument("--progress-every", type=int, default=25)
    filt.add_argument("--require-complete", action=argparse.BooleanOptionalAction, default=True)
    filt.add_argument("--append-events", action="store_true")
    filt.set_defaults(func=command_filter)

    status = subparsers.add_parser("status", parents=[common_download])
    status.set_defaults(func=command_status)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
