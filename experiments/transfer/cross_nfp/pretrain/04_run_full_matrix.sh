#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-configs/quick_cross_nfp.yaml}"
LOG_DIR="${LOG_DIR:-outputs_cross_nfp/run_logs}"
MARKER_DIR="${MARKER_DIR:-outputs_cross_nfp/run_markers/full}"
FORCE="${FORCE:-0}"
mkdir -p "${LOG_DIR}"
mkdir -p "${MARKER_DIR}"

run_step() {
  local name="$1"
  shift
  local marker="${MARKER_DIR}/${name}.done"
  if [[ "${FORCE}" != "1" && -f "${marker}" ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] SKIP ${name} (${marker} exists)"
    return 0
  fi
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] START ${name}"
  "$@" 2>&1 | tee "${LOG_DIR}/${name}.log"
  touch "${marker}"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE ${name}"
}

run_step 00_check_environment "${PYTHON_BIN}" 00_check_environment.py --config "${CONFIG}"

run_step 01_prepare_pretrain_dataset "${PYTHON_BIN}" 01_prepare_pretrain_dataset.py --config "${CONFIG}"

run_step 02_baseline_random_15metric_nfp "${PYTHON_BIN}" 02_train_15metric.py \
  --config "${CONFIG}" \
  --stage baseline \
  --run-name baseline_random_15metric_nfp

run_step 03_pretrain_90k_15metric "${PYTHON_BIN}" 02_train_15metric.py \
  --config "${CONFIG}" \
  --stage pretrain \
  --run-name pretrain_90k_15metric

run_step 04_finetune_low_lr_15metric "${PYTHON_BIN}" 02_train_15metric.py \
  --config "${CONFIG}" \
  --stage finetune \
  --run-name finetune_low_lr_15metric \
  --learning-rate 0.0003 \
  --pretrain-model-dir outputs_cross_nfp/models/pretrain_90k_15metric

run_step 05_finetune_default_lr_15metric "${PYTHON_BIN}" 02_train_15metric.py \
  --config "${CONFIG}" \
  --stage finetune \
  --run-name finetune_default_lr_15metric \
  --learning-rate 0.001 \
  --pretrain-model-dir outputs_cross_nfp/models/pretrain_90k_15metric

run_step 06_summarize_matrix "${PYTHON_BIN}" 03_summarize_matrix.py --config "${CONFIG}"
run_step 07_gate_results "${PYTHON_BIN}" 06_gate_results.py --config "${CONFIG}"
