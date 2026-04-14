#!/usr/bin/env bash
# 在 **后台** 跑检索训练，绑定环境变量里的 GPU（默认第一张：GPU=0）。
# 日志: training/logs/retrieval_A_<timestamp>.log
# 断点: 默认自动从 output/checkpoint_latest/training_state.pt 续训（传 --no-auto-resume 可强制从头）
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

LOG_DIR="${LOG_DIR:-${ROOT}/training/logs}"
mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG_DIR}/retrieval_A_${TS}.log"

GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="${GPU}"

echo "[run_retrieval_A_bg] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[run_retrieval_A_bg] log -> ${LOG}"

nohup "${ROOT}/training/run_retrieval_A.sh" "$@" >>"${LOG}" 2>&1 &
echo $! >"${LOG_DIR}/retrieval_A_${TS}.pid"
echo "[run_retrieval_A_bg] pid=$(cat "${LOG_DIR}/retrieval_A_${TS}.pid")"
echo "[run_retrieval_A_bg] tail -f ${LOG}"
