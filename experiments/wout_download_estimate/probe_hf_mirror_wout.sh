#!/usr/bin/env bash
set -euo pipefail

ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
REPO="proxima-fusion/constellaration"
WOUT_DIR="vmecpp_wout"
OUT_DIR="${OUT_DIR:-experiments/wout_download_estimate/outputs/mirror_probe}"
RANGE_BYTES="${RANGE_BYTES:-1048576}"

mkdir -p "$OUT_DIR"

echo "endpoint=$ENDPOINT"
echo "output_dir=$OUT_DIR"

curl_probe() {
  local url="$1"
  local output="$2"
  curl -L -sS --max-time 60 \
    -w "HTTP_CODE=%{http_code} SIZE_DOWNLOAD=%{size_download} TIME_TOTAL=%{time_total} SPEED_DOWNLOAD=%{speed_download}\n" \
    "$url" -o "$output"
}

echo "[1/5] API repository probe"
curl_probe "$ENDPOINT/api/datasets/$REPO" "$OUT_DIR/repo_api.json"

echo "[2/5] API tree first page probe"
curl_probe "$ENDPOINT/api/datasets/$REPO/tree/main/$WOUT_DIR?recursive=false&expand=true" "$OUT_DIR/wout_tree_page1.json"

echo "[3/5] API tree full pagination"
python3 - "$ENDPOINT" "$REPO" "$WOUT_DIR" "$OUT_DIR/wout_tree_all.json" <<'PY'
import json
import os
import re
import subprocess
import sys
import tempfile
import time

endpoint, repo, wout_dir, output_path = sys.argv[1:5]
url = f"{endpoint}/api/datasets/{repo}/tree/main/{wout_dir}?recursive=false&expand=true"
files = []
pages = 0
started = time.time()
while url:
    pages += 1
    header_path = tempfile.NamedTemporaryFile(delete=False).name
    body_path = tempfile.NamedTemporaryFile(delete=False).name
    try:
        subprocess.check_call(
            ["curl", "-L", "-sS", "--max-time", "60", "-D", header_path, url, "-o", body_path]
        )
        headers = open(header_path, "r", errors="ignore").read()
        body = open(body_path, "rb").read()
    finally:
        for path in (header_path, body_path):
            try:
                os.unlink(path)
            except OSError:
                pass
    data = json.loads(body.decode("utf-8"))
    if not isinstance(data, list):
        raise RuntimeError(f"unexpected response page {pages}: {data!r}")
    files.extend([item for item in data if item.get("type") == "file"])
    links = re.findall(r'<([^>]+)>; rel="next"', headers)
    url = links[-1].replace("https://huggingface.co/", endpoint + "/") if links else None
    print(json.dumps({"page": pages, "page_items": len(data), "files_so_far": len(files), "next": bool(url)}), flush=True)

summary = {
    "pages": pages,
    "files": len(files),
    "total_size_bytes": sum(int(item.get("size") or 0) for item in files),
    "total_size_gb": sum(int(item.get("size") or 0) for item in files) / 1e9,
    "elapsed_seconds": time.time() - started,
}
with open(output_path, "w") as handle:
    json.dump(files, handle)
print("TREE_SUMMARY=" + json.dumps(summary, sort_keys=True))
PY

echo "[4/5] Download tiny id_to_file_map"
curl_probe "$ENDPOINT/datasets/$REPO/resolve/main/$WOUT_DIR/id_to_file_map.parquet" "$OUT_DIR/id_to_file_map.parquet"
ls -lh "$OUT_DIR/id_to_file_map.parquet"

echo "[5/5] HEAD and range probes"
FIRST_PART="$(python3 - "$OUT_DIR/wout_tree_all.json" <<'PY'
import json
import sys
files = json.load(open(sys.argv[1]))
for item in files:
    path = item.get("path", "")
    if path.endswith(".parquet") and not path.endswith("id_to_file_map.parquet"):
        print(path)
        break
PY
)"
echo "first_part=$FIRST_PART"
curl -L -sS -I --max-time 60 "$ENDPOINT/datasets/$REPO/resolve/main/$FIRST_PART" | sed -n '1,80p'
curl -L -sS --max-time 90 -r "0-$((RANGE_BYTES - 1))" \
  -w "RANGE_HTTP=%{http_code} RANGE_BYTES=%{size_download} RANGE_TIME=%{time_total} RANGE_SPEED=%{speed_download}\n" \
  "$ENDPOINT/datasets/$REPO/resolve/main/$FIRST_PART" \
  -o "$OUT_DIR/range_probe.bin"
rm -f "$OUT_DIR/range_probe.bin"

echo "probe complete"
