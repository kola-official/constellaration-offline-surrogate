#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

if [ -z "${PYTHON_BIN:-}" ] && command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate constellaration-py310 || true
fi
PYTHON_BIN="${PYTHON_BIN:-python}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export HUGGINGFACE_HUB_ENDPOINT="${HUGGINGFACE_HUB_ENDPOINT:-$HF_ENDPOINT}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

"$PYTHON_BIN" experiments/wout_download_estimate/test_stream_wout_subset.py \
  --cache-dir "$HF_DATASETS_CACHE" \
  --output-dir experiments/wout_download_estimate/outputs \
  --workers "${WOUT_TEST_WORKERS:-4}" \
  --max-wout-rows-per-worker "${WOUT_TEST_ROWS_PER_WORKER:-200}" \
  --progress-every 25 \
  --measure-payload-bytes
