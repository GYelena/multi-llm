# Planner 48h 执行指南（SFT + DPO）

这份指南对应当前仓库已落地的最小闭环脚本，目标是在 48 小时内完成：
- `planner_sft_seed.jsonl` 生成
- `planner_dpo_candidates.jsonl` + `planner_dpo_pairs.jsonl` 生成
- 首轮 SFT 与首轮 DPO 训练

## 0) 环境变量（你需要提供）

```bash
export DMX_BASE_URL="https://www.dmxapi.cn/v1"
export DMX_API_KEY="你的key"
export DMX_MODEL="glm-5.1-cc"
```

预算（默认内置）：
- 输入单价：`6.32 CNY / 1M token`
- 输出单价：`22.12 CNY / 1M token`
- 日预算：`50 CNY`

如价格变化，请通过脚本参数覆盖。

## 1) 先构建“真实 query 池”（推荐）

> 不要用 `orchestrator/data_train_examples/planner.example.jsonl`（测试文件）。

```bash
python3 training/planner/build_planner_query_pool.py \
  --metrics "/root/autodl-tmp/muti-llm/Data/Processed_data/orchestrator_metrics_math_fin_common.jsonl" \
  --trace "/root/autodl-tmp/muti-llm/Data/Processed_data/orchestrator_trace.jsonl" \
  --output "/root/autodl-tmp/muti-llm/Data/Processed_data/planner_query_pool.jsonl" \
  --use-model-a \
  --model-a-root "/root/autodl-tmp/muti-llm/Data/Model_A" \
  --english-only \
  --max-total 1200 \
  --target-failure-ratio 0.3
```

输出：
- `Data/Processed_data/planner_query_pool.jsonl`

## 2) 生成 SFT 种子数据（含预算闸门 + 机检）

```bash
python3 training/planner/generate_planner_sft_seed.py \
  --input "/root/autodl-tmp/muti-llm/Data/Processed_data/planner_query_pool.jsonl" \
  --output "/root/autodl-tmp/muti-llm/Data/Processed_data/planner_sft_seed.jsonl" \
  --max-samples 600 \
  --mode all \
  --daily-budget-cny 50
```

输出：
- `Data/Processed_data/planner_sft_seed.jsonl`
- `Data/Processed_data/planner_sft_seed.stats.json`

## 3) 用现有 SFT 训练脚本训练 planner（首轮）

```bash
CUDA_VISIBLE_DEVICES=0 python3 training/train_lora_sft.py \
  --model "/root/autodl-tmp/muti-llm/models" \
  --data "/root/autodl-tmp/muti-llm/Data/Processed_data/planner_sft_seed.jsonl" \
  --output "/root/autodl-tmp/muti-llm/outputs/planner_sft" \
  --max-steps 800 \
  --max-seq-length 2048 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 2e-4 \
  --format chat
```

可先 smoke：
- 把 `--max-steps` 改为 `100`

## 4) 采样 DPO 候选（硬规则评分）

```bash
python3 training/planner/generate_planner_dpo_candidates.py \
  --input "/root/autodl-tmp/muti-llm/Data/Processed_data/planner_query_pool.jsonl" \
  --output "/root/autodl-tmp/muti-llm/Data/Processed_data/planner_dpo_candidates.jsonl" \
  --kind all \
  --samples-per-query 3 \
  --max-queries 300 \
  --temperature-list "0.0,0.1,0.2"
```

输出：
- `Data/Processed_data/planner_dpo_candidates.jsonl`

## 5) 构建 DPO 对（可选启用模型裁决）

纯硬规则（更省预算）：

```bash
python3 training/planner/build_planner_dpo_pairs.py \
  --input "/root/autodl-tmp/muti-llm/Data/Processed_data/planner_dpo_candidates.jsonl" \
  --output "/root/autodl-tmp/muti-llm/Data/Processed_data/planner_dpo_pairs.jsonl" \
  --max-pairs 2000 \
  --no-use-judge
```

近分样本用模型裁决（质量更高，成本更高）：

```bash
python3 training/planner/build_planner_dpo_pairs.py \
  --input "/root/autodl-tmp/muti-llm/Data/Processed_data/planner_dpo_candidates.jsonl" \
  --output "/root/autodl-tmp/muti-llm/Data/Processed_data/planner_dpo_pairs.jsonl" \
  --max-pairs 2000 \
  --use-judge \
  --min-hard-gap 0.08
```

输出：
- `Data/Processed_data/planner_dpo_pairs.jsonl`
- `Data/Processed_data/planner_dpo_pairs.stats.json`

## 6) 首轮 DPO 训练

```bash
CUDA_VISIBLE_DEVICES=0 python3 training/planner/train_planner_dpo.py \
  --model "/root/autodl-tmp/muti-llm/models" \
  --data "/root/autodl-tmp/muti-llm/Data/Processed_data/planner_dpo_pairs.jsonl" \
  --output "/root/autodl-tmp/muti-llm/outputs/planner_dpo" \
  --max-samples 2000 \
  --max-length 2048 \
  --max-prompt-length 1400 \
  --max-steps 600 \
  --beta 0.1 \
  --learning-rate 5e-6
```

## 7) 最小验收（48h）

建议至少看四个指标：
- JSON 解析成功率
- schema/DAG 校验通过率
- `planner_subgraph_fallback` 占比
- “重试后成功”占比

若 DPO 后格式指标明显退化（如下降 > 2%），回退到 SFT checkpoint 并降低 DPO 学习率/步数。

## 8) 当前脚本清单

- `training/planner/build_planner_query_pool.py`
- `training/planner/generate_planner_sft_seed.py`
- `training/planner/generate_planner_dpo_candidates.py`
- `training/planner/build_planner_dpo_pairs.py`
- `training/planner/train_planner_dpo.py`

