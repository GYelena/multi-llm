# Orchestrator 快速上手（中文）

本指南给你一个可执行的三阶段流程：
- 本地空跑（不依赖模型服务）；
- 专家 A/B/C 接真实服务；
- 中心规划器 + 专家都接真实服务。

## 1) 先理解两个开关

运行入口：

```bash
cd /root/autodl-tmp/muti-llm
python3 -m orchestrator.cli --query "你的问题"
```

两个核心参数互相独立：

- `--planner-backend`：谁负责生成初始 DAG
  - `rule`：规则规划（确定性）
  - `mock`：内置 mock 规划（本地调试）
  - `openai`：走 OpenAI 兼容接口，让中心规划模型返回 JSON DAG

- `--backend`：谁执行专家节点（A/B/C）
  - `mock`：内置 mock 专家
  - `openai`：调用真实专家服务（A/B/C 的 OpenAI 兼容端点）

## 2) 阶段 A：本地调试（不需要起服务）

```bash
cd /root/autodl-tmp/muti-llm
python3 -m orchestrator.cli \
  --query "who invented relativity" \
  --planner-backend mock \
  --backend mock
```

预期：
- 终端输出 JSON，包含 `finalAnswer`。
- `Data/Processed_data/orchestrator_trace.jsonl` 增加事件记录。

## 3) 阶段 B：专家接真实服务，规划先用 mock

用于先验证 A/B/C 的服务可用性。

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

## 4) 阶段 C：中心规划器 + 专家都接真实服务

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

## 4.1) 方案 A：双卡 RTX 5090 ×2，单 vLLM（TP=2）共用端口

适合「同一基座、一张逻辑集群」：**一个** OpenAI 兼容服务（`tensor-parallel-size 2`），编排器里把 **中心 + A + B + C** 的 base URL **都指向同一地址**（例如 `http://127.0.0.1:8000`）。

**终端 1：起 vLLM（前台，占满 GPU0+1）**

```bash
chmod +x ./orchestrator/start_vllm_scheme_A.sh
./orchestrator/start_vllm_scheme_A.sh
```

若启动时报错 **`AssertionError: duplicate template name`**（出现在 `torch/_inductor/...`，vLLM worker 导入 `@torch.compile` 相关模块时）：  
脚本已默认设置 **`TORCHDYNAMO_DISABLE=1`** 并加上 **`--enforce-eager`**。更新脚本后请重新执行上述命令。  
若仍失败，再考虑：**把 PyTorch 降到 2.6/2.7 LTS** 或 **升级 vLLM/torch 到互相匹配的版本**（当前日志里也有 torch/torchao 版本提示）。

可用环境变量覆盖默认值：

- `MODEL_PATH`：模型目录（默认 `.../DeepSeek-R1`）
- `PORT`：监听端口（默认 `8000`）
- `SERVED_MODEL_NAME`：对外名（默认 `deepseek-r1`，需与 orchestrator 的 `--model-name` / `--planner-model-name` 一致）
- `CUDA_VISIBLE_DEVICES`（默认 `0,1`）
- `MAX_MODEL_LEN`、`GPU_MEMORY_UTILIZATION`
- `TORCHDYNAMO_DISABLE`（默认 `1`；设为 `0` 可恢复 torch.compile 行为，一般不推荐）
- `VLLM_ENFORCE_EAGER`（默认 `1` 即加 `--enforce-eager`；设为 `0` 可尝试关闭以提速，但可能再触发 inductor 问题）

**终端 2：跑编排（一键）**

```bash
./orchestrator/start_demo.sh --mode full \
  --single-vllm-url http://127.0.0.1:8000 \
  --query "请分解任务并执行"
```

或等价环境变量：`export SINGLE_VLLM_URL=http://127.0.0.1:8000` 后再 `./orchestrator/start_demo.sh --mode full`。

## 5) Trace 固化（全局 + 每次 run 单独文件）

默认行为（无需额外参数）：

- 全局 trace 仍写入 `--trace-path`（默认 `Data/Processed_data/orchestrator_trace.jsonl`）。
- 同目录下自动创建 `orchestrator_runs/`，并为每次运行写入单独文件：  
  `orchestrator_runs/<runId>.jsonl`  
  便于按请求回放完整事件链。

关闭单独文件：

```bash
python3 -m orchestrator.cli --query "hello" --planner-backend mock --backend mock --disable-per-run-trace
```

自定义 per-run 目录：

```bash
python3 -m orchestrator.cli --query "hello" --planner-backend mock --backend mock --run-trace-dir /tmp/my_runs
```

每次运行结束会额外写入 `run_summary` 事件（耗时、专家调用次数、重规划统计、trace 路径）。

## 6) 启动前端口检查（推荐）

你可以先检查中心与专家端口是否连通：

```bash
cd /root/autodl-tmp/muti-llm
python3 -m orchestrator.healthcheck \
  --planner-url http://127.0.0.1:8000 \
  --a-url http://127.0.0.1:8001 \
  --b-url http://127.0.0.1:8002 \
  --c-url http://127.0.0.1:8003
```

如果只想检查专家，跳过 planner：

```bash
python3 -m orchestrator.healthcheck --skip-planner
```

如果希望在主命令里自动检查，追加：

```bash
--healthcheck-before-run
```

## 7) 批量评测（metrics JSONL）

准备一个 JSONL（每行一个 JSON），字段：

- `query`（必填）
- `id`（可选，用于对齐你自己的题库）

示例：`orchestrator/queries.example.jsonl`

运行：

```bash
cd /root/autodl-tmp/muti-llm
python3 -m orchestrator.benchmark \
  --input orchestrator/queries.example.jsonl \
  --output Data/Processed_data/orchestrator_metrics.jsonl \
  --planner-backend mock \
  --backend mock
```

方案 A（单 vLLM，planner 与 A/B/C 同一 base URL）可加 `--single-vllm-url`（也可用环境变量 `SINGLE_VLLM_URL`）：

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

日常固定跑法（已写死默认 `SINGLE_VLLM_URL`、`EXPERT_TIMEOUT_SECONDS=300` 等，可用环境变量覆盖）：

```bash
cd /root/autodl-tmp/muti-llm
./orchestrator/run_benchmark_scheme_A.sh
```

输出每行一条指标：`wallMs`、`reconstructRounds`、`successHeuristic`、`traceRunPath` 等。

## 7.1 汇总 metrics（均值 / P95）

对上面生成的 `orchestrator_metrics.jsonl` 做汇总：

```bash
cd /root/autodl-tmp/muti-llm
python3 -m orchestrator.metrics_report \
  --input Data/Processed_data/orchestrator_metrics.jsonl
```

写入文件：

```bash
python3 -m orchestrator.metrics_report \
  --input Data/Processed_data/orchestrator_metrics.jsonl \
  --output Data/Processed_data/orchestrator_metrics_summary.json
```

## 8) 一键脚本（推荐）

你也可以直接用一键脚本，减少手动参数输入：

```bash
cd /root/autodl-tmp/muti-llm
./orchestrator/start_demo.sh --mode local
```

三种模式：

- `--mode local`：`planner=mock` + `backend=mock`
- `--mode experts`：`planner=mock` + `backend=openai`
- `--mode full`：`planner=openai` + `backend=openai`

示例：

```bash
./orchestrator/start_demo.sh --mode experts --query "请先检索再推理"
./orchestrator/start_demo.sh --mode full --query "请分解任务并执行"
```

## 9) 常用调参

- `--max-steps 12`：控制器最大循环步数
- `--t-risk 0.70`：重规划风险阈值
- `--t-uncertainty 0.60`：不确定性阈值
- `--max-reconstruct-times 3`：最多重规划轮数
- `--max-patch-ops 2`：每轮最多 patch 数
- `--reconstruct-budget-ratio 0.25`：重规划预算占比
- `--trace-path /path/to/trace.jsonl`：自定义 trace 输出路径

## 10) 常见问题排查

- `planner HTTP error`：
  - 检查 `--planner-base-url`
  - 确认服务暴露 `/v1/chat/completions`
- 专家调用异常（`http_...` / `call_exception`）：
  - 检查端口和服务进程
  - 检查 `--model-name` 是否与服务端一致
- 规划器返回非 JSON：
  - 先改成 `--planner-backend mock` 或 `rule`
  - 调整中心模型提示词，强制输出纯 JSON

## 11) Trace 里重点看什么

关键事件：
- `run_start`
- `controller_plan`
- `controller_step`
- `expert_call`
- `expert_result`
- `controller_eval`
- `controller_reconstruct` / `controller_reconstruct_skipped`
- `final_answer`
- `run_summary`

排查重规划问题时，优先看 `controller_eval` 和 `controller_reconstruct`。
