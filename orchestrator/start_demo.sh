#!/usr/bin/env bash
set -euo pipefail

# One-command runner for orchestrator demo.
# Modes:
#   local   -> planner=mock, backend=mock
#   experts -> planner=mock, backend=openai
#   full    -> planner=openai, backend=openai (recommended)
#
# Scheme A (2x GPU, one vLLM TP=2): start ./orchestrator/start_vllm_scheme_A.sh
# then:  ./orchestrator/start_demo.sh --mode full --single-vllm-url http://127.0.0.1:8000

MODE="full"
QUERY="who invented relativity"
MAX_STEPS="12"
MODEL_NAME="${MODEL_NAME:-deepseek-r1}"
 A_MODEL_NAME="${A_MODEL_NAME:-${MODEL_NAME}}"
 B_MODEL_NAME="${B_MODEL_NAME:-${MODEL_NAME}}"
 C_MODEL_NAME="${C_MODEL_NAME:-${MODEL_NAME}}"
PLANNER_MODEL_NAME="${PLANNER_MODEL_NAME:-deepseek-r1}"
PLANNER_BASE_URL="${PLANNER_BASE_URL:-http://127.0.0.1:8000}"
A_BASE_URL="${A_BASE_URL:-http://127.0.0.1:8001}"
B_BASE_URL="${B_BASE_URL:-http://127.0.0.1:8002}"
C_BASE_URL="${C_BASE_URL:-http://127.0.0.1:8003}"
A_BACKEND="${A_BACKEND:-openai}"
A_RETRIEVER_CHECKPOINT="${A_RETRIEVER_CHECKPOINT:-/root/autodl-tmp/muti-llm/outputs/retriever_A/checkpoint-164000}"
A_RETRIEVER_BASE_MODEL="${A_RETRIEVER_BASE_MODEL:-/root/autodl-tmp/muti-llm/models/bge-base-en-v1.5}"
A_RETRIEVER_INDEX_DIR="${A_RETRIEVER_INDEX_DIR:-/root/autodl-tmp/muti-llm/outputs/retriever_A/faiss_index}"
A_RETRIEVER_TOP_K="${A_RETRIEVER_TOP_K:-5}"
API_KEY="${OPENAI_API_KEY:-dummy}"
# Long enough for R1-style experts over HTTP (override with EXPERT_TIMEOUT_SECONDS=60 for fast-fail).
EXPERT_TIMEOUT_SECONDS="${EXPERT_TIMEOUT_SECONDS:-300}"
TRACE_PATH="${TRACE_PATH:-}"
RUN_TRACE_DIR="${RUN_TRACE_DIR:-}"
DISABLE_PER_RUN_TRACE="${DISABLE_PER_RUN_TRACE:-0}"
SINGLE_VLLM_URL="${SINGLE_VLLM_URL:-}"

usage() {
  cat <<'EOF'
Usage:
  ./orchestrator/start_demo.sh [options]

Options:
  --mode <local|experts|full>   Run mode (default: full)
  --query "<text>"              Query for orchestrator
  --max-steps <int>             Max controller steps (default: 12)
  --single-vllm-url <url>       Planner + A/B/C all use this OpenAI base (Scheme A)
  -h, --help                    Show this help

Examples:
  ./orchestrator/start_demo.sh --mode local
  ./orchestrator/start_demo.sh --mode experts --query "请先检索再推理"
  ./orchestrator/start_demo.sh --mode full --query "请分解任务并执行"
  ./orchestrator/start_demo.sh --mode full --single-vllm-url http://127.0.0.1:8000 --query "请分解任务并执行"

Env (optional):
  TRACE_PATH=/path/to/global_trace.jsonl
  RUN_TRACE_DIR=/path/to/per_run_dir
  DISABLE_PER_RUN_TRACE=1
  SINGLE_VLLM_URL=http://127.0.0.1:8000   (same as --single-vllm-url; Scheme A)
  A_BACKEND=openai|local_retriever        (use local_retriever for true local RAG A)
  A_RETRIEVER_CHECKPOINT=/path/to/checkpoint
  A_RETRIEVER_BASE_MODEL=/path/to/bge-base
  A_RETRIEVER_INDEX_DIR=/path/to/faiss_index_dir
  A_RETRIEVER_TOP_K=5
  A_MODEL_NAME=your-rag-model-name         (expert A model id on its local endpoint)
  B_MODEL_NAME=your-lora-b-model-name      (expert B model id on its local endpoint)
  C_MODEL_NAME=your-lora-c-model-name      (expert C model id on its local endpoint)
  EXPERT_TIMEOUT_SECONDS=300             (default 300; expert A/B/C HTTP read timeout)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --query)
      QUERY="${2:-}"
      shift 2
      ;;
    --max-steps)
      MAX_STEPS="${2:-}"
      shift 2
      ;;
    --single-vllm-url)
      SINGLE_VLLM_URL="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ -n "${SINGLE_VLLM_URL}" ]]; then
  PLANNER_BASE_URL="${SINGLE_VLLM_URL}"
  A_BASE_URL="${SINGLE_VLLM_URL}"
  B_BASE_URL="${SINGLE_VLLM_URL}"
  C_BASE_URL="${SINGLE_VLLM_URL}"
  echo "[start_demo] Scheme A: planner + experts -> ${SINGLE_VLLM_URL}"
fi

TRACE_EXTRA=()
if [[ -n "${TRACE_PATH}" ]]; then
  TRACE_EXTRA+=(--trace-path "${TRACE_PATH}")
fi
if [[ -n "${RUN_TRACE_DIR}" ]]; then
  TRACE_EXTRA+=(--run-trace-dir "${RUN_TRACE_DIR}")
fi
if [[ "${DISABLE_PER_RUN_TRACE}" == "1" ]]; then
  TRACE_EXTRA+=(--disable-per-run-trace)
fi

COMMON_ARGS=(
  --query "${QUERY}"
  --max-steps "${MAX_STEPS}"
  "${TRACE_EXTRA[@]}"
)

if [[ "${MODE}" == "local" ]]; then
  echo "[start_demo] mode=local (planner=mock, experts=mock)"
  python3 -m orchestrator.cli \
    "${COMMON_ARGS[@]}" \
    --planner-backend mock \
    --backend mock
elif [[ "${MODE}" == "experts" ]]; then
  echo "[start_demo] mode=experts (planner=mock, experts=openai) expert_timeout=${EXPERT_TIMEOUT_SECONDS}s"
  python3 -m orchestrator.cli \
    "${COMMON_ARGS[@]}" \
    --planner-backend mock \
    --backend openai \
    --a-backend "${A_BACKEND}" \
    --a-retriever-checkpoint "${A_RETRIEVER_CHECKPOINT}" \
    --a-retriever-base-model "${A_RETRIEVER_BASE_MODEL}" \
    --a-retriever-index-dir "${A_RETRIEVER_INDEX_DIR}" \
    --a-retriever-top-k "${A_RETRIEVER_TOP_K}" \
    --model-name "${MODEL_NAME}" \
    --a-model-name "${A_MODEL_NAME}" \
    --b-model-name "${B_MODEL_NAME}" \
    --c-model-name "${C_MODEL_NAME}" \
    --a-base-url "${A_BASE_URL}" \
    --b-base-url "${B_BASE_URL}" \
    --c-base-url "${C_BASE_URL}" \
    --api-key "${API_KEY}" \
    --expert-timeout-seconds "${EXPERT_TIMEOUT_SECONDS}" \
    --healthcheck-before-run
elif [[ "${MODE}" == "full" ]]; then
  echo "[start_demo] mode=full (planner=openai, experts=openai) expert_timeout=${EXPERT_TIMEOUT_SECONDS}s"
  python3 -m orchestrator.cli \
    "${COMMON_ARGS[@]}" \
    --planner-backend openai \
    --planner-base-url "${PLANNER_BASE_URL}" \
    --planner-model-name "${PLANNER_MODEL_NAME}" \
    --backend openai \
    --a-backend "${A_BACKEND}" \
    --a-retriever-checkpoint "${A_RETRIEVER_CHECKPOINT}" \
    --a-retriever-base-model "${A_RETRIEVER_BASE_MODEL}" \
    --a-retriever-index-dir "${A_RETRIEVER_INDEX_DIR}" \
    --a-retriever-top-k "${A_RETRIEVER_TOP_K}" \
    --model-name "${MODEL_NAME}" \
    --a-model-name "${A_MODEL_NAME}" \
    --b-model-name "${B_MODEL_NAME}" \
    --c-model-name "${C_MODEL_NAME}" \
    --a-base-url "${A_BASE_URL}" \
    --b-base-url "${B_BASE_URL}" \
    --c-base-url "${C_BASE_URL}" \
    --api-key "${API_KEY}" \
    --expert-timeout-seconds "${EXPERT_TIMEOUT_SECONDS}" \
    --healthcheck-before-run
else
  echo "Unsupported mode: ${MODE}" >&2
  usage
  exit 1
fi
