# Orchestrator Quick Start

This guide helps you run the reactive orchestrator in three stages:
- local dry run (no model services),
- real expert services (A/B/C),
- real central planner + real experts.

## 1) What This CLI Does

Run from project root:

```bash
cd /root/autodl-tmp/muti-llm
python3 -m orchestrator.cli --query "your question"
```

The CLI has two independent switches:
- `--planner-backend`: who creates the initial DAG plan
  - `rule`: deterministic rules
  - `mock`: built-in mock JSON planner
  - `openai`: central planner model via OpenAI-compatible API
- `--backend`: who executes expert nodes (A/B/C)
  - `mock`: built-in mock experts
  - `openai`: real expert model services via OpenAI-compatible API

## 2) Trace layout (global + per-run)

By default:

- Global trace appends to `--trace-path`.
- Per-run traces also append to `<trace-dir>/orchestrator_runs/<runId>.jsonl`.

Disable per-run files:

```bash
python3 -m orchestrator.cli --query "hello" --planner-backend mock --backend mock --disable-per-run-trace
```

Each successful run emits `run_summary` with latency, reconstruct stats, and trace paths.

## 3) Stage A: Local Debug (No Services Required)

```bash
cd /root/autodl-tmp/muti-llm
python3 -m orchestrator.cli \
  --query "who invented relativity" \
  --planner-backend mock \
  --backend mock
```

Expected:
- You get JSON output with `finalAnswer`.
- `Data/Processed_data/orchestrator_trace.jsonl` is appended with events.

## 4) Batch benchmark (metrics JSONL)

```bash
cd /root/autodl-tmp/muti-llm
python3 -m orchestrator.benchmark \
  --input orchestrator/queries.example.jsonl \
  --output Data/Processed_data/orchestrator_metrics.jsonl \
  --planner-backend mock \
  --backend mock
```

Scheme A (one vLLM, same base URL for planner + A/B/C):

```bash
python3 -m orchestrator.benchmark \
  --input orchestrator/queries.example.jsonl \
  --output Data/Processed_data/orchestrator_metrics.jsonl \
  --healthcheck-before-run \
  --single-vllm-url http://127.0.0.1:8000 \
  --expert-timeout-seconds 300 \
  --planner-backend openai \
  --planner-model-name deepseek-r1 \
  --backend openai \
  --model-name deepseek-r1 \
  --api-key dummy
```

Fixed defaults for daily Scheme A batch runs (override with env vars such as `INPUT`, `OUTPUT`, `SINGLE_VLLM_URL`, `EXPERT_TIMEOUT_SECONDS`):

```bash
cd /root/autodl-tmp/muti-llm
./orchestrator/run_benchmark_scheme_A.sh
```

## 4.1) Summarize metrics (mean / P95)

```bash
cd /root/autodl-tmp/muti-llm
python3 -m orchestrator.metrics_report \
  --input Data/Processed_data/orchestrator_metrics.jsonl
```

## 5) Stage B: Real Experts, Mock Planner

Use this to validate A/B/C serving first.

```bash
cd /root/autodl-tmp/muti-llm
python3 -m orchestrator.cli \
  --query "请先检索事实再给出结论" \
  --planner-backend mock \
  --backend openai \
  --model-name deepseek-r1 \
  --a-base-url http://127.0.0.1:8001 \
  --b-base-url http://127.0.0.1:8002 \
  --c-base-url http://127.0.0.1:8003
```

## 6) Stage C: Real Planner + Real Experts

Use this when your central planner service is available.

```bash
cd /root/autodl-tmp/muti-llm
python3 -m orchestrator.cli \
  --query "请分解任务并执行" \
  --planner-backend openai \
  --planner-base-url http://127.0.0.1:8000 \
  --planner-model-name deepseek-r1 \
  --backend openai \
  --model-name deepseek-r1 \
  --a-base-url http://127.0.0.1:8001 \
  --b-base-url http://127.0.0.1:8002 \
  --c-base-url http://127.0.0.1:8003
```

## 6.1) Scheme A: 2 GPUs, single vLLM (TP=2), one URL for planner + experts

Terminal 1 (blocking):

```bash
chmod +x ./orchestrator/start_vllm_scheme_A.sh
./orchestrator/start_vllm_scheme_A.sh
```

If workers fail with `AssertionError: duplicate template name` under `torch/_inductor`, the script defaults to `TORCHDYNAMO_DISABLE=1` and `--enforce-eager` for PyTorch 2.9.x + vLLM 0.16.x. Re-run after pulling the latest `start_vllm_scheme_A.sh`. If it still fails, align PyTorch and vLLM versions (e.g. PyTorch 2.6/2.7 LTS or a matching vLLM wheel).

Terminal 2:

```bash
./orchestrator/start_demo.sh --mode full \
  --single-vllm-url http://127.0.0.1:8000 \
  --query "decompose and execute"
```

## 7) Useful Runtime Controls

- `--max-steps 12`: max controller loop steps
- `--t-risk 0.70`: reconstruct risk threshold
- `--t-uncertainty 0.60`: reconstruct uncertainty threshold
- `--max-reconstruct-times 3`: max reconstruct rounds
- `--max-patch-ops 2`: max patches per reconstruct step
- `--reconstruct-budget-ratio 0.25`: reconstruct budget cap relative to `max-steps`
- `--trace-path /path/to/file.jsonl`: custom trace output path

## 8) Quick Troubleshooting

- `planner HTTP error`:
  - Check `--planner-base-url`.
  - Confirm planner server exposes `/v1/chat/completions`.
- `No adapter registered for expert ...`:
  - Check `taskType/expert` output from planner JSON.
  - Current experts are `A`, `B`, `C`.
- Empty or invalid planner JSON:
  - Keep `--planner-backend rule` or `mock` temporarily.
  - Verify central planner prompt enforces strict JSON only.
- All nodes remain pending:
  - Inspect DAG dependencies in trace for cycles or missing node ids.
- Expert call errors (`http_...`, `call_exception`):
  - Check expert port and model service health.
  - Verify `--model-name` matches served model.

## 9) Trace Events to Inspect

The trace JSONL includes these event types:
- `run_start`
- `controller_plan`
- `planner_fallback` (when planner JSON fails and rule planner is used)
- `controller_step`
- `expert_call`
- `expert_result`
- `controller_eval`
- `controller_reconstruct` (when triggered)
- `controller_reconstruct_skipped` (when blocked by caps)
- `final_answer`
- `run_summary`

Start with `controller_plan` and `controller_eval` when debugging reconstruct behavior.
