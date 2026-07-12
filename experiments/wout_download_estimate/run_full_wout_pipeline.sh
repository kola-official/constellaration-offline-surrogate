#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="${BASE:-${SCRIPT_DIR}}"
OUTPUTS="$BASE/outputs"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT="$BASE/wout_full_pipeline.py"
TREE_JSON="$OUTPUTS/mirror_probe/wout_tree_all.json"
TARGET_IDS_JSON="$OUTPUTS/target_wout_ids_nfp3_no_error.json"
DOWNLOAD_DIR="$OUTPUTS/vmecpp_wout_full"
FILTER_DIR="$OUTPUTS/vmecpp_wout_filtered_68191"
LOG_DIR="$OUTPUTS/full_wout_pipeline_logs"
DOWNLOAD_WORKERS="${DOWNLOAD_WORKERS:-16}"
FILTER_WORKERS="${FILTER_WORKERS:-4}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

mkdir -p "$LOG_DIR"

echo "started_at=$(date '+%Y-%m-%d %H:%M:%S')"
echo "host=$(hostname)"
echo "python=$PYTHON_BIN"
echo "base=$BASE"
echo "download_workers=$DOWNLOAD_WORKERS"
echo "filter_workers=$FILTER_WORKERS"
echo "download_dir=$DOWNLOAD_DIR"
echo "filter_dir=$FILTER_DIR"

"$PYTHON_BIN" "$SCRIPT" \
  --endpoint "$HF_ENDPOINT" \
  download \
  --tree-json "$TREE_JSON" \
  --download-dir "$DOWNLOAD_DIR" \
  --workers "$DOWNLOAD_WORKERS" \
  --progress-every 10

"$PYTHON_BIN" "$SCRIPT" \
  --endpoint "$HF_ENDPOINT" \
  filter \
  --tree-json "$TREE_JSON" \
  --download-dir "$DOWNLOAD_DIR" \
  --target-ids-json "$TARGET_IDS_JSON" \
  --output-dir "$FILTER_DIR" \
  --workers "$FILTER_WORKERS" \
  --progress-every 10 \
  --require-complete

echo "finished_at=$(date '+%Y-%m-%d %H:%M:%S')"
