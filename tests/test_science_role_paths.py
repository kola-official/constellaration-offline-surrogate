"""Layout and path-wiring checks for the science-role experiment rename.

These tests drive the shipped experiment common modules (with a lightweight
``yaml`` stub so import works without a full ML stack) and assert peer-stage
directories resolve under the new names.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _install_yaml_stub() -> None:
    if "yaml" in sys.modules:
        return
    stub = ModuleType("yaml")
    stub.safe_load = lambda _stream: {}  # type: ignore[attr-defined]
    sys.modules["yaml"] = stub


def _load_module(rel_path: str, name: str):
    _install_yaml_stub()
    path = ROOT / rel_path
    # Stub heavy optional deps used only by some commons at import time.
    for dep in ("numpy", "pandas"):
        if dep not in sys.modules:
            sys.modules[dep] = ModuleType(dep)
    # pretrain common imports label_utils from stage1 via sys.path insert after
    # STAGE1_DIR is set; ensure stage1 path is importable before load.
    stage1 = ROOT / "experiments" / "official_space" / "stage1_base"
    if str(stage1) not in sys.path:
        sys.path.insert(0, str(stage1))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


OFFICIAL = [
    "experiments/official_space/stage1_base",
    "experiments/official_space/stage2_latent",
    "experiments/official_space/stage3_trust_region",
    "experiments/official_space/stage4_alm_prescreen",
]
AUX = ["experiments/auxiliary_supervision/wout24"]
TRANSFER = [
    "experiments/transfer/cross_nfp/pretrain",
    "experiments/transfer/cross_nfp/downstream/stage1",
    "experiments/transfer/cross_nfp/downstream/stage2_latent",
    "experiments/transfer/cross_nfp/downstream/stage3_trust_region",
    "experiments/transfer/cross_nfp/downstream/stage4_alm_prescreen",
]


@pytest.mark.parametrize("rel", OFFICIAL + AUX + TRANSFER)
def test_science_role_experiment_dirs_exist(rel: str) -> None:
    path = ROOT / rel
    assert path.is_dir(), f"missing experiment dir {rel}"
    assert not rel.startswith("rtx3090_"), "live path must not use machine prefix"


def test_no_top_level_machine_prefixed_experiment_dirs() -> None:
    live = [
        p.name
        for p in ROOT.iterdir()
        if p.is_dir() and p.name.startswith("rtx3090_")
    ]
    assert live == [], f"top-level machine-prefixed experiment dirs remain: {live}"


def test_stage1_repo_root_discovery() -> None:
    mod = _load_module(
        "experiments/official_space/stage1_base/common.py",
        "stage1_common_under_test",
    )
    assert mod.repo_root() == ROOT


def test_official_stage2_peer_wiring() -> None:
    mod = _load_module(
        "experiments/official_space/stage2_latent/common_stage2.py",
        "official_stage2_common_under_test",
    )
    assert mod.REPO_ROOT == ROOT
    assert mod.STAGE1_DIR == ROOT / "experiments" / "official_space" / "stage1_base"
    assert mod.STAGE1_DIR.is_dir()
    assert "rtx3090" not in str(mod.STAGE1_DIR)


def test_official_stage4_peer_wiring() -> None:
    mod = _load_module(
        "experiments/official_space/stage4_alm_prescreen/common_stage4.py",
        "official_stage4_common_under_test",
    )
    assert mod.STAGE1_DIR.name == "stage1_base"
    assert mod.STAGE2_DIR.name == "stage2_latent"
    assert mod.STAGE3_DIR.name == "stage3_trust_region"
    assert mod.STAGE1_DIR.is_dir()
    assert mod.STAGE2_DIR.is_dir()
    assert mod.STAGE3_DIR.is_dir()


def test_transfer_downstream_stage2_peer_wiring() -> None:
    mod = _load_module(
        "experiments/transfer/cross_nfp/downstream/stage2_latent/common_stage2.py",
        "transfer_stage2_common_under_test",
    )
    assert mod.STAGE1_DIR == (
        ROOT / "experiments" / "transfer" / "cross_nfp" / "downstream" / "stage1"
    )
    assert mod.STAGE1_DIR.is_dir()


def test_wout24_cross_track_stage1() -> None:
    mod = _load_module(
        "experiments/auxiliary_supervision/wout24/common.py",
        "wout24_common_under_test",
    )
    assert mod.STAGE1_DIR == ROOT / "experiments" / "official_space" / "stage1_base"
    assert mod.REPO_ROOT == ROOT


def test_renamed_runners_exist() -> None:
    runners = [
        "experiments/transfer/cross_nfp/pretrain/04_run_full_matrix.sh",
        "experiments/transfer/cross_nfp/pretrain/RUNBOOK.md",
        "experiments/wout_download_estimate/run_full_wout_pipeline.sh",
        "experiments/wout_download_estimate/run_stream_test.sh",
        "experiments/official_space/stage2_latent/01_run_latent_feasibility_cmaes.py",
        "experiments/official_space/stage4_alm_prescreen/01_run_surrogate_alm_ngopt.py",
    ]
    for rel in runners:
        assert (ROOT / rel).is_file(), rel
    # Hardware-tagged names must not remain as live entry points.
    retired = [
        "experiments/transfer/cross_nfp/pretrain/04_run_rtx3090_matrix.sh",
        "experiments/transfer/cross_nfp/pretrain/RUNBOOK_RTX3090.md",
        "experiments/wout_download_estimate/run_full_wout_pipeline_rtx3090.sh",
    ]
    for rel in retired:
        assert not (ROOT / rel).exists(), f"retired hardware name still present: {rel}"


def test_readme_documents_science_role_paths() -> None:
    readme = (ROOT / "README.md").read_text()
    assert "experiments/official_space/stage1_base" in readme
    assert "cd experiments/official_space/stage1_base" in readme
    assert "cd rtx3090_simple_qi" not in readme
