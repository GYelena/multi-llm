#!/usr/bin/env bash
# Scheme A batch benchmark: one vLLM URL + long expert HTTP timeout (avoids R1 false timeouts).
# Override any value via env before calling.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

INPUT="${INPUT:-orchestrator/queries.math_finance_common.jsonl}"
OUTPUT="${OUTPUT:-Data/Processed_data/orchestrator_metrics_scheme_A.jsonl}"
SINGLE_VLLM_URL="${SINGLE_VLLM_URL:-http://127.0.0.1:8000}"
EXPERT_TIMEOUT_SECONDS="${EXPERT_TIMEOUT_SECONDS:-300}"
MODEL_NAME="${MODEL_NAME:-deepseek-r1}"
PLANNER_MODEL_NAME="${PLANNER_MODEL_NAME:-deepseek-r1}"
API_KEY="${OPENAI_API_KEY:-dummy}"

usage() {
  cat <<'EOF'
Usage:
  ./orchestrator/run_benchmark_scheme_A.sh

Runs orchestrator.benchmark with fixed Scheme A defaults (override via env):

  INPUT                 JSONL path (default: orchestrator/queries.math_finance_common.jsonl)
  OUTPUT                metrics JSONL path (default: Data/Processed_data/orchestrator_metrics_scheme_A.jsonl)
  SINGLE_VLLM_URL       planner + A/B/C base (default: http://127.0.0.1:8000)
  EXPERT_TIMEOUT_SECONDS  HTTP read timeout for experts (default: 300)
  MODEL_NAME / PLANNER_MODEL_NAME / OPENAI_API_KEY

Example:
  SINGLE_VLLM_URL=http://127.0.0.1:8000 ./orchestrator/run_benchmark_scheme_A.sh
  INPUT=orchestrator/queries.example.jsonl OUTPUT=/tmp/m.jsonl ./orchestrator/run_benchmark_scheme_A.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

echo "[run_benchmark_scheme_A] SINGLE_VLLM_URL=${SINGLE_VLLM_URL} EXPERT_TIMEOUT_SECONDS=${EXPERT_TIMEOUT_SECONDS}"
echo "[run_benchmark_scheme_A] INPUT=${INPUT} OUTPUT=${OUTPUT}"

python3 -m orchestrator.benchmark \
  --input "${INPUT}" \
  --output "${OUTPUT}" \
  --healthcheck-before-run \
  --single-vllm-url "${SINGLE_VLLM_URL}" \
  --expert-timeout-seconds "${EXPERT_TIMEOUT_SECONDS}" \
  --planner-backend openai \
  --planner-model-name "${PLANNER_MODEL_NAME}" \
  --backend openai \
  --model-name "${MODEL_NAME}" \
  --api-key "${API_KEY}"

echo "[run_benchmark_scheme_A] done. Summarize with:"
echo "  python3 -m orchestrator.metrics_report --input ${OUTPUT}"
