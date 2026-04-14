#!/usr/bin/env bash
# 在 **后台** 跑 LoRA C（C_sft），默认绑定 **第三张卡**（GPU=2，即 CUDA 下标 2）。
# 日志: training/logs/lora_C_<timestamp>.log
# 断点: 与 train_lora_sft 一致 — 同 OUT 目录下存在 checkpoint-* 时自动续训；传 --no-auto-resume 强制从头。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

LOG_DIR="${LOG_DIR:-${ROOT}/training/logs}"
mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG_DIR}/lora_C_${TS}.log"

GPU="${GPU:-2}"
export CUDA_VISIBLE_DEVICES="${GPU}"

echo "[run_lora_C_bg] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[run_lora_C_bg] log -> ${LOG}"

nohup "${ROOT}/training/run_lora_C.sh" "$@" >>"${LOG}" 2>&1 &
echo $! >"${LOG_DIR}/lora_C_${TS}.pid"
echo "[run_lora_C_bg] pid=$(cat "${LOG_DIR}/lora_C_${TS}.pid")"
echo "[run_lora_C_bg] tail -f ${LOG}"
