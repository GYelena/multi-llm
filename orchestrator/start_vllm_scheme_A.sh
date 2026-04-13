#!/usr/bin/env bash
# Scheme A: one vLLM OpenAI server with tensor parallel across 2 GPUs.
# Run this in a dedicated terminal (foreground). Then use orchestrator with
# SINGLE_VLLM_URL=http://127.0.0.1:${PORT} or ./start_demo.sh --mode full --single-vllm-url ...
#
# If workers crash with: AssertionError: duplicate template name (torch inductor),
# this script defaults to mitigations compatible with PyTorch 2.9.x + vLLM 0.16.x:
#   - TORCHDYNAMO_DISABLE=1  (disables @torch.compile side effects during imports)
#   - --enforce-eager        (disable cudagraphs; slower but more stable)
# Override: TORCHDYNAMO_DISABLE=0 VLLM_ENFORCE_EAGER=0 ./orchestrator/start_vllm_scheme_A.sh
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/muti-llm/DeepSeek-R1}"
PORT="${PORT:-8000}"
SERVED_NAME="${SERVED_MODEL_NAME:-deepseek-r1}"
MAX_LEN="${MAX_MODEL_LEN:-8192}"
GPU_UTIL="${GPU_MEMORY_UTILIZATION:-0.92}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

# Mitigate torch.compile / inductor crashes in vLLM worker subprocesses (e.g. deep_gemm import).
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"

VLLM_EXTRA=()
if [[ "${VLLM_ENFORCE_EAGER:-1}" == "1" ]]; then
  VLLM_EXTRA+=(--enforce-eager)
fi

echo "[start_vllm_scheme_A] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[start_vllm_scheme_A] model=${MODEL_PATH} port=${PORT} tp=2 served=${SERVED_NAME}"
echo "[start_vllm_scheme_A] TORCHDYNAMO_DISABLE=${TORCHDYNAMO_DISABLE} VLLM_ENFORCE_EAGER=${VLLM_ENFORCE_EAGER:-1}"

exec python3 -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --tensor-parallel-size 2 \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --served-model-name "${SERVED_NAME}" \
  --dtype bfloat16 \
  --max-model-len "${MAX_LEN}" \
  --gpu-memory-utilization "${GPU_UTIL}" \
  "${VLLM_EXTRA[@]}"
