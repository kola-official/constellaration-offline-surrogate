#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-configs/quick_cross_nfp.yaml}"
SEEDS="${SEEDS:-0 1 2}"
LOG_DIR="${LOG_DIR:-outputs_cross_nfp/run_logs_multiseed}"
MARKER_DIR="${MARKER_DIR:-outputs_cross_nfp/run_markers/multiseed}"
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

for seed in ${SEEDS}; do
  suffix="_seed${seed}"
  pretrain_dataset_dir="outputs_cross_nfp/dataset_pretrain${suffix}"
  pretrain_model_dir="outputs_cross_nfp/models/pretrain_90k_15metric${suffix}"

  run_step "seed${seed}_01_prepare_pretrain_dataset" \
    "${PYTHON_BIN}" 01_prepare_pretrain_dataset.py \
      --config "${CONFIG}" \
      --seed "${seed}" \
      --pretrain-dataset-dir "${pretrain_dataset_dir}"

  run_step "seed${seed}_02_baseline_random_15metric_nfp" \
    "${PYTHON_BIN}" 02_train_15metric.py \
      --config "${CONFIG}" \
      --seed "${seed}" \
      --pretrain-dataset-dir "${pretrain_dataset_dir}" \
      --stage baseline \
      --run-name "baseline_random_15metric_nfp${suffix}"

  run_step "seed${seed}_03_pretrain_90k_15metric" \
    "${PYTHON_BIN}" 02_train_15metric.py \
      --config "${CONFIG}" \
      --seed "${seed}" \
      --pretrain-dataset-dir "${pretrain_dataset_dir}" \
      --stage pretrain \
      --run-name "pretrain_90k_15metric${suffix}"

  run_step "seed${seed}_04_finetune_low_lr_15metric" \
    "${PYTHON_BIN}" 02_train_15metric.py \
      --config "${CONFIG}" \
      --seed "${seed}" \
      --pretrain-dataset-dir "${pretrain_dataset_dir}" \
      --stage finetune \
      --run-name "finetune_low_lr_15metric${suffix}" \
      --learning-rate 0.0003 \
      --pretrain-model-dir "${pretrain_model_dir}"

  run_step "seed${seed}_05_finetune_default_lr_15metric" \
    "${PYTHON_BIN}" 02_train_15metric.py \
      --config "${CONFIG}" \
      --seed "${seed}" \
      --pretrain-dataset-dir "${pretrain_dataset_dir}" \
      --stage finetune \
      --run-name "finetune_default_lr_15metric${suffix}" \
      --learning-rate 0.001 \
      --pretrain-model-dir "${pretrain_model_dir}"

  run_step "seed${seed}_06_summarize_matrix" \
    "${PYTHON_BIN}" 03_summarize_matrix.py \
      --config "${CONFIG}" \
      --run-suffix "${suffix}"

  run_step "seed${seed}_07_gate_results" \
    "${PYTHON_BIN}" 06_gate_results.py \
      --config "${CONFIG}" \
      --run-suffix "${suffix}"
done

seeds_csv="$(echo "${SEEDS}" | tr ' ' ',')"
run_step 99_aggregate_multiseed "${PYTHON_BIN}" 09_aggregate_multiseed.py --config "${CONFIG}" --seeds "${seeds_csv}"
