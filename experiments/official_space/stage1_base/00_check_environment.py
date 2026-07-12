from __future__ import annotations

import importlib
import json
import platform
import subprocess
from pathlib import Path

from common import (
    OUTPUT_DIR,
    active_environment,
    apply_thread_environment,
    ensure_output_dirs,
    git_info,
    load_config,
    parse_args,
    run_text,
    write_json,
)


REQUIRED_IMPORTS = [
    "torch",
    "constellaration",
    "vmecpp",
    "simsopt",
    "datasets",
    "sklearn",
    "nevergrad",
    "pyarrow",
    "yaml",
]


def check_import(name: str) -> dict[str, str | bool]:
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "unknown")
        return {"ok": True, "version": str(version)}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    hardware = apply_thread_environment(config)
    ensure_output_dirs()
    imports = {name: check_import(name) for name in REQUIRED_IMPORTS}

    cuda = {"available": False}
    if imports.get("torch", {}).get("ok"):
        import torch

        cuda = {
            "available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
            "devices": [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ],
            "torch_cuda": str(getattr(torch.version, "cuda", "")),
        }

    payload = {
        "platform": platform.platform(),
        "environment": active_environment(),
        "hardware_config": hardware,
        "git": git_info(),
        "imports": imports,
        "cuda": cuda,
        "nvidia_smi": run_text(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,driver_version",
                "--format=csv,noheader",
            ]
        ),
        "cpu": run_text(["bash", "-lc", "lscpu | egrep 'Model name|CPU\\(s\\)|Core\\(s\\)|Thread\\(s\\)|Socket\\(s\\)'"]),
        "memory": run_text(["free", "-h"]),
        "disk": run_text(["df", "-h", "."]),
    }

    write_json(OUTPUT_DIR / "run_summary" / "import_check.json", payload)
    freeze = run_text(["python", "-m", "pip", "freeze"])
    env_txt = {
        "payload": payload,
        "pip_freeze": freeze.splitlines() if freeze else [],
    }
    with (OUTPUT_DIR / "run_summary" / "env.txt").open("w") as handle:
        handle.write(json.dumps(env_txt, indent=2, sort_keys=True))
        handle.write("\n")

    missing = [name for name, result in imports.items() if not result.get("ok")]
    if missing:
        raise SystemExit(f"Gate A failed; missing imports: {missing}")
    if not cuda.get("available"):
        raise SystemExit("Gate A failed; torch is importable but CUDA is unavailable")

    print("Gate A passed")


if __name__ == "__main__":
    main()
