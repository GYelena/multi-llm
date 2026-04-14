#!/usr/bin/env bash
# Expert A：RAG **检索**（bi-encoder 对比学习），数据为 A_retrieval.jsonl。
# 英文语料默认：bge-base-en-v1.5（可改 small / e5）
#   export BASE_MODEL=BAAI/bge-small-en-v1.5
#   export BASE_MODEL=intfloat/e5-base-v2
# 大模型（如 DeepSeek-R1）仅当你刻意要统一底座时再设 BASE_MODEL=/path/to/DeepSeek-R1
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

BASE_MODEL="${BASE_MODEL:-BAAI/bge-base-en-v1.5}"
DATA="${DATA:-${ROOT}/Data/Processed_data/A_retrieval.jsonl}"
OUT="${OUT:-${ROOT}/outputs/retriever_A}"
GPU="${GPU:-0}"

export CUDA_VISIBLE_DEVICES="${GPU}"
echo "[run_retrieval_A] BASE_MODEL=${BASE_MODEL}"
echo "[run_retrieval_A] DATA=${DATA} OUT=${OUT}"

EXTRA=()
if [[ -n "${MAX_SAMPLES:-}" ]]; then EXTRA+=(--max-samples "${MAX_SAMPLES}"); fi
# 进程内用 cuda:0（与 CUDA_VISIBLE_DEVICES 选中的物理卡对应）
DEVICE="${DEVICE:-0}"

exec python3 "${ROOT}/training/train_retrieval_biencoder.py" \
  --model "${BASE_MODEL}" \
  --data "${DATA}" \
  --output "${OUT}" \
  --device "${DEVICE}" \
  "${EXTRA[@]}" \
  "$@"
