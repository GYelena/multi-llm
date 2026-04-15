#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/muti-llm"
SCRIPT="${ROOT}/build_retriever_a_faiss_index.py"
LOG_DIR="${ROOT}/outputs/retriever_A/logs"
mkdir -p "${LOG_DIR}"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="${LOG_DIR}/build_index_${TS}.log"
PID_PATH="${LOG_DIR}/build_index_latest.pid"

GPU="${GPU:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/outputs/retriever_A/faiss_index_all_model_a}"

CMD=(
  python3 "${SCRIPT}"
  --gpu "${GPU}"
  --include-qa-corpora
  --include-qa-pairs
  --output-dir "${OUTPUT_DIR}"
)

# passthrough any extra args
if [[ "$#" -gt 0 ]]; then
  CMD+=("$@")
fi

echo "[run] starting in background..."
echo "[run] log: ${LOG_PATH}"
echo "[run] output_dir: ${OUTPUT_DIR}"
echo "[run] gpu: ${GPU}"

nohup "${CMD[@]}" >"${LOG_PATH}" 2>&1 &
PID=$!
echo "${PID}" >"${PID_PATH}"

echo "[ok] pid=${PID}"
echo "[ok] tail log: tail -f \"${LOG_PATH}\""
echo "[ok] stop: kill ${PID}"
