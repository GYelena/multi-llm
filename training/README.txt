训练脚本分工
============

1) Expert A — RAG **检索**（双塔 / bi-encoder）
   - 数据: Data/Processed_data/A_retrieval.jsonl（query + positive_passages + negative_passages）
   - 脚本: training/train_retrieval_biencoder.py
   - 依赖: pip install -r training/requirements-retrieval.txt
   - 模型下到本地（离线训练）: `./training/download_model_A.sh`
     默认保存到 `models/bge-base-en-v1.5`；若官网超时: `export HF_ENDPOINT=https://hf-mirror.com` 后再执行。
     训练时: `BASE_MODEL=/root/autodl-tmp/muti-llm/models/bge-base-en-v1.5 ./training/run_retrieval_A.sh`
   - **推荐基座（小、专为向量设计）**：
     - **仅英文语料（默认）**: `BAAI/bge-base-en-v1.5`；更省显存可用 `BAAI/bge-small-en-v1.5`；备选 `intfloat/e5-base-v2`
     - 中文为主: `BAAI/bge-small-zh-v1.5`
     - 多语: `BAAI/bge-m3`
     - 首次运行会从 HuggingFace 下载；也可先下载到本地再设 `BASE_MODEL=/path`
   - 启动: `./training/run_retrieval_A.sh`（默认英文 bge-base-en-v1.5）
   - GPU：默认绑定可见的第 0 号卡（`CUDA_VISIBLE_DEVICES=0` + `--device 0`）；换卡如 `GPU=1`。
   - 后台：`./training/run_retrieval_A_bg.sh`，日志 `training/logs/retrieval_A_*.log`，同目录 `.pid`。
   - 断点续训：存在 `outputs/.../checkpoint_latest/training_state.pt` 时默认自动接着训；`--no-auto-resume` 强制从头；`--resume /path/to/training_state.pt` 指定快照。Ctrl+C 会先写 checkpoint_latest 再退出。
   - 说明: 用 AutoModel 编码 + in-batch InfoNCE。大因果模型（如 DeepSeek-R1）仅作可选底座；检索与 B/C 生成解耦是常规做法。

2) Expert A — **生成**（RAG 读入 passages 再写答案，因果 LM LoRA）
   - 数据: A_generation.jsonl
   - 脚本: training/train_lora_sft.py（与 B/C 同套）

3) Expert B / C — 因果 LM LoRA
   - 数据: B_sft.jsonl / C_sft.jsonl
   - 脚本: training/train_lora_sft.py
   - 依赖: pip install -r training/requirements-lora.txt

单卡训 B:
  export BASE_MODEL=/root/autodl-tmp/muti-llm/DeepSeek-R1
  ./training/run_lora_B.sh

单卡训检索 A:
  ./training/run_retrieval_A.sh
  可选: MAX_SAMPLES=5000 ./training/run_retrieval_A.sh --max-samples 5000

三卡并行（**若**仍用「三模型都是因果 LM」演示）:
  ./training/run_lora_parallel_abc.sh
  其中 GPU0 是 A_generation，不是 A_retrieval。若要三卡同时训「检索 A + B + C」，请自行：
  GPU0: run_retrieval_A.sh
  GPU1: run_lora_B.sh
  GPU2: 训 C（train_lora_sft.py --data .../C_sft.jsonl）

注意:
  - 超大 A_retrieval 全量读入会占内存；可加 --max-samples 分段跑。
  - 若单卡 32GB 装不下因果 LM 底座，LoRA 脚本可加 --qlora-4bit（需 bitsandbytes）。
