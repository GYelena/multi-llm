#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import List

# 默认检索查询语句
DEFAULT_QUERY = "Which planet in our solar system is known as the Red Planet?"

# 默认候选文本列表
DEFAULT_CANDIDATES = [
    "Mars is often called the Red Planet because of iron oxide on its surface.",
    "Jupiter is the largest planet in the solar system.",
    "Photosynthesis converts light energy into chemical energy in plants.",
    "Saturn has prominent ring systems made of ice and rock fragments.",
]

def parse_args() -> argparse.Namespace:
    """
    解析命令行参数，支持指定检索器检查点、基础模型、GPU、查询内容、候选集以及Top-k等设置。
    """
    parser = argparse.ArgumentParser(description="Run Retriever-A checkpoint inference (embedding retrieval).")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/root/autodl-tmp/muti-llm/outputs/retriever_A/checkpoint-164000",
        help="Retriever checkpoint path (preferred).",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="/root/autodl-tmp/muti-llm/models/bge-base-en-v1.5",
        help="Fallback base model path when checkpoint loading fails.",
    )
    parser.add_argument("--gpu", type=str, default="0", help="CUDA visible device id(s), e.g. '0'")
    parser.add_argument("--query", type=str, default=DEFAULT_QUERY, help="English retrieval query.")
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Candidate passage text (repeat this arg multiple times).",
    )
    parser.add_argument(
        "--candidates-file",
        type=str,
        default="",
        help="Optional path to JSON/JSONL/TXT candidates file.",
    )
    parser.add_argument("--top-k", type=int, default=4, help="Top-k results to show.")
    return parser.parse_args()

def load_candidates(args: argparse.Namespace) -> List[str]:
    """
    加载候选文本，可来自多次传入的--candidate参数、文件、或默认内置。
    支持json/jsonl/txt三种格式自动解析。
    """
    if args.candidate:
        # 若命令行指定了候选文本，优先使用
        return [c.strip() for c in args.candidate if c.strip()]
    if not args.candidates_file:
        # 若未指定文件，使用默认候选集
        return list(DEFAULT_CANDIDATES)

    path = Path(args.candidates_file)
    if not path.exists():
        raise FileNotFoundError(f"Candidates file not found: {path}")

    # 处理.json格式，期望为字符串列表
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("JSON candidates file must contain a string list")
        return [str(x).strip() for x in data if str(x).strip()]

    rows: List[str] = []
    text = path.read_text(encoding="utf-8")
    # 处理.jsonl格式，每行一个对象，优先取"text"字段
    if path.suffix.lower() == ".jsonl":
        for i, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict) and "text" in obj:
                val = str(obj["text"]).strip()
            else:
                val = str(obj).strip()
            if val:
                rows.append(val)
        return rows

    # 默认当做txt格式，每行一个非空候选
    for line in text.splitlines():
        line = line.strip()
        if line:
            rows.append(line)
    return rows

def main() -> None:
    """
    主程序入口：加载参数和候选集，加载模型/分词器（支持断点和基础模型），
    计算Embedding检索得分，并输出排序结果和元数据。
    """
    args = parse_args()
    # 预先设置CUDA设备（必须在导入torch前）
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # 导入深度学习相关依赖
    import torch
    from transformers import AutoModel, AutoTokenizer

    t0 = time.time()  # 记录起始时间
    candidates = load_candidates(args)  # 加载候选集
    if not candidates:
        raise ValueError("No candidates provided")  # 候选集为空则报错

    def try_load(path: str):
        """
        加载指定路径下的分词器和模型，自动分配设备和精度
        """
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModel.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="auto")
        return tok, model

    used_source = args.checkpoint  # 初始默认用checkpoint
    try:
        # 优先尝试加载检索器微调权重
        tokenizer, model = try_load(args.checkpoint)
    except Exception as e:
        print(f"[warn] checkpoint load failed, fallback to base model: {e}")
        # 加载失败则回退到基础模型
        tokenizer, model = try_load(args.base_model)
        used_source = args.base_model

    model.eval()  # 设为推理模式

    # 按BGE风格给query和passage加前缀提示
    query_text = "Represent this sentence for searching relevant passages: " + args.query
    passage_texts = ["Represent this sentence for retrieval: " + p for p in candidates]

    @torch.no_grad()
    def embed(texts: List[str]):
        """
        对输入文本列表批量编码并提取归一化的向量Embedding
        """
        # 分词并转为张量（自动填充裁剪，最大长度512）
        batch = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(model.device)
        out = model(**batch)  # 前向推理
        emb = out.last_hidden_state[:, 0]  # 取每句第一个Token的隐状态（CLS池化变体）
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)  # L2标准化
        return emb

    # 查询和候选均转为embedding
    q_emb = embed([query_text])      # (1, d)
    p_emb = embed(passage_texts)     # (n, d)

    # 内积得到相关性分数
    scores = (q_emb @ p_emb.T).squeeze(0).detach().cpu().tolist()  # (n,)
    # 按分数从高到低排序
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

    top_k = max(1, min(args.top_k, len(ranked)))  # top-k阈值自适应上限

    print("\n===== QUERY =====")
    print(args.query)
    print("\n===== TOP RESULTS =====")
    # 依次输出排名、分数、文本
    for rank, (idx, score) in enumerate(ranked[:top_k], 1):
        print(
            {
                "rank": rank,
                "score": round(float(score), 6),
                "text": candidates[idx],
            }
        )
    print("\n===== META =====")
    # 输出检索元数据（权重来源、候选数、设备、耗时等）
    print(
        {
            "checkpoint": args.checkpoint,
            "base_model": args.base_model,
            "used_source": used_source,
            "candidate_count": len(candidates),
            "visible_gpus": args.gpu,
            "device": str(model.device),
            "elapsed_sec": round(time.time() - t0, 2),
        }
    )

if __name__ == "__main__":
    main()
