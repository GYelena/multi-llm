#!/usr/bin/env bash
# LoRA SFT for expert C (C_sft.jsonl). Single GPU.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

# 必填：与线上一致的因果 LM 底座（本地路径或 HF id）
: "${BASE_MODEL:?Set BASE_MODEL to your causal LM path, e.g. /root/autodl-tmp/muti-llm/DeepSeek-R1-Distill-Qwen-7B}"

OUT="${OUT:-${ROOT}/outputs/lora_C}"
DATA="${DATA:-${ROOT}/Data/Processed_data/C_sft.jsonl}"
GPU="${GPU:-0}"

# 大 JSONL：先冒烟再长跑。例：MAX_STEPS=5000 或 MAX_SAMPLES=2000
# 速度优先（32GB）：默认 batch=1、accum=8；显存仍紧张可设 USE_QLORA=1 或减小 MAX_SEQ_LENGTH
MAX_STEPS="${MAX_STEPS:-}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-}"

EXTRA=()
EXTRA+=(--per-device-train-batch-size "${PER_DEVICE_BATCH}" --gradient-accumulation-steps "${GRAD_ACCUM}")
if [[ -n "${MAX_STEPS}" ]]; then EXTRA+=(--max-steps "${MAX_STEPS}"); fi
if [[ -n "${MAX_SAMPLES}" ]]; then EXTRA+=(--max-samples "${MAX_SAMPLES}"); fi
if [[ -n "${MAX_SEQ_LENGTH}" ]]; then EXTRA+=(--max-seq-length "${MAX_SEQ_LENGTH}"); fi
if [[ "${USE_QLORA:-0}" == "1" ]]; then EXTRA+=(--qlora-4bit); fi

echo "[run_lora_C] CUDA_VISIBLE_DEVICES=${GPU} BASE_MODEL=${BASE_MODEL}"
export CUDA_VISIBLE_DEVICES="${GPU}"
exec python3 "${ROOT}/training/train_lora_sft.py" \
  --model "${BASE_MODEL}" \
  --data "${DATA}" \
  --output "${OUT}" \
  "${EXTRA[@]}" \
  "$@"
