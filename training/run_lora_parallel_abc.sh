#!/usr/bin/env bash
# 三张卡并行：各跑一个进程，分别训 A_generation / B_sft / C_sft（各绑一张 GPU）。
# 要求：三张 32GB 能各自装下一个底座 + LoRA；OOM 时减小 batch、开 --qlora-4bit，或改为分时跑。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

: "${BASE_MODEL:?Set BASE_MODEL to your causal LM path}"

OUT_A="${OUT_A:-${ROOT}/outputs/lora_A}"
OUT_B="${OUT_B:-${ROOT}/outputs/lora_B}"
OUT_C="${OUT_C:-${ROOT}/outputs/lora_C}"

DATA_A="${DATA_A:-${ROOT}/Data/Processed_data/A_generation.jsonl}"
DATA_B="${DATA_B:-${ROOT}/Data/Processed_data/B_sft.jsonl}"
DATA_C="${DATA_C:-${ROOT}/Data/Processed_data/C_sft.jsonl}"

# 每路默认用少量步数/样本做「能跑通」的演示；正式训练请改大或去掉 --max-samples/--max-steps
MAX_STEPS_B="${MAX_STEPS_B:-3000}"
MAX_SAMPLES_AC="${MAX_SAMPLES_AC:-3000}"

echo "[parallel] BASE_MODEL=${BASE_MODEL}"
echo "[parallel] A GPU0 -> ${OUT_A}  (max-samples ${MAX_SAMPLES_AC})"
echo "[parallel] B GPU1 -> ${OUT_B}  (max-steps ${MAX_STEPS_B})"
echo "[parallel] C GPU2 -> ${OUT_C}  (max-samples ${MAX_SAMPLES_AC})"

CUDA_VISIBLE_DEVICES=0 python3 "${ROOT}/training/train_lora_sft.py" \
  --model "${BASE_MODEL}" \
  --data "${DATA_A}" \
  --output "${OUT_A}" \
  --max-samples "${MAX_SAMPLES_AC}" \
  --no-streaming \
  "$@" &

CUDA_VISIBLE_DEVICES=1 python3 "${ROOT}/training/train_lora_sft.py" \
  --model "${BASE_MODEL}" \
  --data "${DATA_B}" \
  --output "${OUT_B}" \
  --max-steps "${MAX_STEPS_B}" \
  "$@" &

CUDA_VISIBLE_DEVICES=2 python3 "${ROOT}/training/train_lora_sft.py" \
  --model "${BASE_MODEL}" \
  --data "${DATA_C}" \
  --output "${OUT_C}" \
  --max-samples "${MAX_SAMPLES_AC}" \
  --no-streaming \
  "$@" &

wait
echo "[parallel] all jobs finished."
