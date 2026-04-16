# ABC 模型微调：数据预处理、训练方式、数据规模与当前结果

本文基于当前仓库代码与已产出文件整理，覆盖三个专家模型（A/B/C）在本项目中的数据处理与训练现状。

## 0. 总览（当前快照）

- 预处理脚本：`preprocess_multimodel_data.py`
- 训练脚本：
  - A 检索：`training/train_retrieval_biencoder.py`
  - A 生成 + B/C：`training/train_lora_sft.py`
- 预处理输出目录：`Data/Processed_data`
- 已有训练输出目录：`outputs/retriever_A`、`outputs/lora_B_full`、`outputs/lora_C_full`

当前预处理数据量（JSONL 行数）：

- `A_retrieval.jsonl`：673,143
- `A_generation.jsonl`：353,120
- `B_sft.jsonl`：1,345,431
- `C_sft.jsonl`：681,324

---

## 1. A 模型（检索 + 生成）

### 1.1 A 的数据预处理

### A_retrieval（检索训练数据）

由 `build_a_retrieval()` 生成，主要流程：

1. 读取 MS MARCO（v2.1 + v1.1）训练 parquet；
2. 从样本中抽取：
   - `query`
   - `positive_passages`（根据 `is_selected`）
   - `negative_passages`（负例采样，默认 `neg_count=5`）
3. 读取 BEIR HotpotQA / FiQA 的 `queries + corpus + qrels`，构建正负样本对；
4. 输出统一格式 JSONL（`task_type=retrieval`）。

字段形态（示意）：`query / positive_passages / negative_passages / source`

数据来源分布（当前文件）：

- `ms_marco`：582,643
- `beir_hotpotqa`：85,000
- `beir_fiqa`：5,500
- 合计：673,143

### A_generation（RAG 生成训练数据）

由 `build_a_generation()` 生成，主要流程：

1. HotpotQA（fullwiki + distractor）：
   - 构造 `retrieved_passages`
   - 抽取 `supporting_facts` 作为 `citations`
2. NQ Open：
   - 构造问答对（无检索段落）
3. TriviaQA nocontext：
   - 使用 question + alias answer

输出结构（示意）：

- `instruction`
- `input: { query, retrieved_passages }`
- `output: { answer, citations, cannot_answer }`
- `source`

数据来源分布（当前文件）：

- `trivia_qa_rc_nocontext`：156,328
- `hotpot_qa`：105,257
- `nq_open`：91,535
- 合计：353,120

### 1.2 A 的微调方式

本项目中 A 有两条线：

1. **A 检索模型（主线）**
   - 脚本：`training/train_retrieval_biencoder.py`
   - 结构：共享编码器（bi-encoder）+ mean pooling + L2 normalize
   - 损失：in-batch InfoNCE（query-passages 双向交叉熵）
   - 推荐底座：`BAAI/bge-base-en-v1.5`（脚本默认）
   - 启动脚本：`training/run_retrieval_A.sh`

2. **A 生成模型（可选，与 B/C 同一 LoRA SFT 框架）**
   - 脚本：`training/train_lora_sft.py`
   - 数据：`A_generation.jsonl`
   - 适合做“检索后回答生成”的补充实验

### 1.3 A 的当前结果（仓库已产出）

- 检索训练 checkpoint：`outputs/retriever_A/checkpoint-162000`、`checkpoint-164000`
- 最新 checkpoint 步数：**164,000**
- 已构建 FAISS 索引：
  - `outputs/retriever_A/faiss_index/meta.json`：`doc_count=300`（小索引）
  - `outputs/retriever_A/faiss_index_all_model_a/meta.json`：`doc_count=16,653,680`（全量）
- 全量索引构建信息：
  - `index_type=IndexFlatIP`
  - `embedding_dim=768`
  - `build_sec=5301.43`
  - `resumed=true`（断点续构建）

---

## 2. B 模型（推理专家）

### 2.1 B 的数据预处理

由 `build_b_sft()` 生成，统一转成 SFT 文本对：

- `instruction`
- `input`
- `output`
- `meta`（含 `source/domain`）

核心清洗逻辑：

- 过滤空 `input/output`
- 多轮对话数据（OpenThoughts、Agentar）提取首个 user + assistant 对
- 保留来源标签，便于后续分析

数据来源分布（当前 `B_sft.jsonl`）：

- `numinamath_1_5`：896,012
- `reasoning_v1_20m_sample`：125,244
- `openthoughts_114k`：113,957
- `openmathreasoning`：111,150
- `agentar_deepfinance_100k`：99,068
- 合计：1,345,431

### 2.2 B 的微调方式

- 脚本：`training/train_lora_sft.py`
- 启动脚本：`training/run_lora_B.sh`
- 训练范式：因果语言模型 LoRA SFT（TRL `SFTTrainer`）
- 默认关键设置（脚本默认）：
  - LoRA：`r=32`、`alpha=64`、`dropout=0.05`
  - 目标模块：`q/k/v/o + gate/up/down proj`
  - 序列长度：`2048`
  - batch=1，grad_accum=8
  - 默认 bf16，支持可选 QLoRA 4bit
  - 支持自动续训（checkpoint 恢复）

### 2.3 B 的当前结果（仓库已产出）

- 输出目录：`outputs/lora_B_full`
- 最新可见 checkpoint：`checkpoint-82500`
- `trainer_state.json` 进度：
  - `global_step=82500`
  - `max_steps=168179`
  - `epoch=0.4905`
  - 最近训练 loss（log_history 末尾）：**0.3874**
- LoRA 配置显示底座：`/root/autodl-tmp/muti-llm/DeepSeek-R1`

---

## 3. C 模型（写作专家）

### 3.1 C 的数据预处理

由 `build_c_sft()` 生成，数据格式与 B 一致（instruction/input/output/meta）：

数据来源分布（当前 `C_sft.jsonl`）：

- `cnn_dailymail_3_0_0`：311,971
- `trivia_qa_rc_nocontext`：156,328
- `openthoughts_114k`：113,957
- `agentar_deepfinance_100k`：99,068
- 合计：681,324

### 3.2 C 的微调方式

- 脚本：`training/train_lora_sft.py`
- 启动脚本：`training/run_lora_C.sh`
- 训练机制与 B 相同（LoRA SFT，同一套超参数风格与续训机制）

### 3.3 C 的当前结果（仓库已产出）

- 输出目录：`outputs/lora_C_full`
- 最新可见 checkpoint：`checkpoint-46500`
- `trainer_state.json` 进度：
  - `global_step=46500`
  - `max_steps=85166`
  - `epoch=0.5460`
  - 最近训练 loss（log_history 末尾）：**0.4491**
- LoRA 配置显示底座：`/root/autodl-tmp/muti-llm/DeepSeek-R1`

---

## 4. 结论（面向当前项目）

- A/B/C 已形成“检索-推理-写作”分工的数据与训练闭环；
- A 的检索训练与全量索引产物已经落地，具备 RAG 召回基础；
- B/C 已进入中期训练阶段（checkpoint 持续增长，loss 可跟踪）；
- 当前仓库里“结果”以训练进度与中间 checkpoint 为主，若需要对外展示效果，建议补充统一评测脚本与指标（如 EM/F1、ROUGE、领域任务准确率）。

