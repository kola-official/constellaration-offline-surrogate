#!/usr/bin/env bash
set -euo pipefail

# Bundle this pretrain experiment from the repository root.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

stamp="$(date '+%Y%m%d_%H%M%S')"
rel_path="experiments/transfer/cross_nfp/pretrain"
bundle="cross_nfp_pretrain_bundle_${stamp}.tar.gz"

tar \
  --exclude "${rel_path}/outputs_cross_nfp" \
  --exclude "${rel_path}/__pycache__" \
  --exclude "${rel_path}/.DS_Store" \
  -czf "${bundle}" \
  "${rel_path}"

shasum -a 256 "${bundle}" > "${bundle}.sha256"

echo "${PWD}/${bundle}"
echo "${PWD}/${bundle}.sha256"
