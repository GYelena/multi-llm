#!/usr/bin/env bash
# 把检索模型 A（默认 BAAI/bge-base-en-v1.5）下载到本地目录，供离线训练。
#
# 用法:
#   ./training/download_model_A.sh
#   OUT=/data/models/bge-base-en-v1.5 ./training/download_model_A.sh
#
# 若 huggingface.co 超时，先试镜像:
#   export HF_ENDPOINT=https://hf-mirror.com
#   ./training/download_model_A.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="${REPO:-BAAI/bge-base-en-v1.5}"
OUT="${OUT:-${ROOT}/models/bge-base-en-v1.5}"

mkdir -p "${OUT}"

if command -v huggingface-cli &>/dev/null; then
  echo "[download_model_A] huggingface-cli download ${REPO} -> ${OUT}"
  huggingface-cli download "${REPO}" --local-dir "${OUT}" --local-dir-use-symlinks False
else
  echo "[download_model_A] huggingface-cli not found; using Python snapshot_download"
  export REPO OUT
  python3 -c "
from huggingface_hub import snapshot_download
import os
out = os.environ['OUT']
repo = os.environ['REPO']
path = snapshot_download(repo_id=repo, local_dir=out, local_dir_use_symlinks=False)
print('downloaded to', path)
"
fi

echo "[download_model_A] done. Train with:"
echo "  BASE_MODEL=${OUT} ./training/run_retrieval_A.sh"
