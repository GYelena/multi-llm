#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import signal
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

import numpy as np
import pyarrow.parquet as pq

def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。
    支持指定模型检查点、基础模型、RAG数据集目录、输出目录、GPU编号、批量大小、最大长度、最多处理文档数、额外数据源等。
    """
    parser = argparse.ArgumentParser(
        description="为Retriever-A从rag_english_v2语料库构建FAISS索引"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/root/autodl-tmp/muti-llm/outputs/retriever_A/checkpoint-164000",
        help="Retriever-A 检索器微调权重路径（优先优先加载）",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="/root/autodl-tmp/muti-llm/models/bge-base-en-v1.5",
        help="加载checkpoint失败时的基础模型路径",
    )
    parser.add_argument(
        "--rag-root",
        type=str,
        default="/root/autodl-tmp/muti-llm/Data/Model_A/rag_english_v2",
        help="RAG语料根目录（存放parquet文件）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/root/autodl-tmp/muti-llm/outputs/retriever_A/faiss_index",
        help="输出faiss索引和元数据目录",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="GPU编号，例如'0'。会设置CUDA_VISIBLE_DEVICES"
    )
    parser.add_argument("--batch-size", type=int, default=64, help="编码批处理大小")
    parser.add_argument("--max-length", type=int, default=512, help="tokenizer最大长度")
    parser.add_argument(
        "--max-docs",
        type=int,
        default=0,
        help="最多处理文档数，0表示全部处理，仅用于快速测试"
    )
    parser.add_argument(
        "--extra-corpus-glob",
        action="append",
        default=[],
        help="额外的parquet文件通配符（可重复）"
    )
    parser.add_argument(
        "--include-ms-marco",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否并入 ms_marco passage 语料（默认开启）",
    )
    parser.add_argument(
        "--include-hotpot-context",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否并入 hotpot_qa context 语料（默认开启）",
    )
    parser.add_argument(
        "--include-qa-pairs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否并入 NQ/Trivia 的 QA 对（默认关闭，噪声较大）",
    )
    parser.add_argument(
        "--include-qa-corpora",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否并入 commonsense_qa/mmlu/agentar（默认关闭，可显著扩库但可能带噪声）",
    )
    parser.add_argument(
        "--checkpoint-every-docs",
        type=int,
        default=50000,
        help="每处理多少文档写一次中间索引与meta（0表示只在最后写）",
    )
    return parser.parse_args()

def default_corpus_parquets(rag_root: Path) -> List[Path]:
    """
    默认收集RAG数据集下常用的parquet文件路径。
    只包含已知的beir_hotpotqa和beir_fiqa子集，与训练脚本保持一致。
    """
    patterns = [
        rag_root / "beir_hotpotqa" / "corpus" / "*.parquet",
        rag_root / "beir_fiqa" / "corpus" / "*.parquet",
    ]
    files: List[Path] = []
    for pattern in patterns:
        files.extend(sorted(pattern.parent.glob(pattern.name)))
    return files

def iter_docs_from_beir(parquet_files: Iterable[Path]) -> Iterator[Dict[str, str]]:
    for fp in parquet_files:
        pf = pq.ParquetFile(str(fp))
        source = str(fp)
        for batch in pf.iter_batches(batch_size=2048):
            data = batch.to_pydict()
            ids = data.get("_id", [])
            titles = data.get("title", [])
            texts = data.get("text", [])
            for doc_id, title, text in zip(ids, titles, texts):
                did = str(doc_id).strip()
                t = str(title or "").strip()
                body = str(text or "").strip()
                if not did or not body:
                    continue
                yield {
                    "doc_id": f"beir:{did}",
                    "title": t,
                    "text": body,
                    "source": source,
                }


def iter_docs_from_ms_marco(parquet_files: Iterable[Path]) -> Iterator[Dict[str, str]]:
    for fp in parquet_files:
        pf = pq.ParquetFile(str(fp))
        source = str(fp)
        for batch in pf.iter_batches(batch_size=1024):
            data = batch.to_pydict()
            qids = data.get("query_id", [])
            passages_col = data.get("passages", [])
            for qid, passages in zip(qids, passages_col):
                if not isinstance(passages, dict):
                    continue
                p_texts = passages.get("passage_text", [])
                if not isinstance(p_texts, list):
                    continue
                qid_s = str(qid).strip()
                for i, txt in enumerate(p_texts):
                    body = str(txt or "").strip()
                    if not body:
                        continue
                    yield {
                        "doc_id": f"msmarco:{qid_s}:p{i}",
                        "title": "",
                        "text": body,
                        "source": source,
                    }


def iter_docs_from_hotpot_context(parquet_files: Iterable[Path]) -> Iterator[Dict[str, str]]:
    for fp in parquet_files:
        pf = pq.ParquetFile(str(fp))
        source = str(fp)
        for batch in pf.iter_batches(batch_size=1024):
            data = batch.to_pydict()
            qids = data.get("id", [])
            contexts = data.get("context", [])
            for qid, ctx in zip(qids, contexts):
                if not isinstance(ctx, dict):
                    continue
                titles = ctx.get("title", [])
                sents_group = ctx.get("sentences", [])
                if not isinstance(titles, list) or not isinstance(sents_group, list):
                    continue
                for i, (title, sents) in enumerate(zip(titles, sents_group)):
                    title_s = str(title or "").strip()
                    if isinstance(sents, list):
                        body = " ".join(str(x).strip() for x in sents if str(x).strip())
                    else:
                        body = str(sents or "").strip()
                    if not body:
                        continue
                    yield {
                        "doc_id": f"hotpot:{qid}:ctx{i}",
                        "title": title_s,
                        "text": body,
                        "source": source,
                    }


def iter_docs_from_qa_pairs(parquet_files: Iterable[Path], prefix: str) -> Iterator[Dict[str, str]]:
    for fp in parquet_files:
        pf = pq.ParquetFile(str(fp))
        source = str(fp)
        for batch in pf.iter_batches(batch_size=1024):
            data = batch.to_pydict()
            questions = data.get("question", [])
            answers = data.get("answer", [])
            ids = data.get("question_id", [])
            for i, (q, a) in enumerate(zip(questions, answers)):
                q_text = str(q or "").strip()
                if not q_text:
                    continue
                a_text = ""
                if isinstance(a, list):
                    a_text = str(a[0]).strip() if a else ""
                elif isinstance(a, dict):
                    val = a.get("value", "")
                    if isinstance(val, list):
                        a_text = str(val[0]).strip() if val else ""
                    else:
                        a_text = str(val or "").strip()
                else:
                    a_text = str(a or "").strip()
                if not a_text:
                    continue
                qid = ""
                if isinstance(ids, list) and i < len(ids):
                    qid = str(ids[i]).strip()
                if not qid:
                    qid = f"{Path(source).stem}:{i}"
                yield {
                    "doc_id": f"{prefix}:{qid}",
                    "title": q_text,
                    "text": a_text,
                    "source": source,
                }


def _strip_reasoning_tags(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def _choices_to_lines(choices) -> List[str]:
    if isinstance(choices, dict):
        labels = choices.get("label", [])
        texts = choices.get("text", [])
        lines = []
        for i, t in enumerate(texts):
            lab = str(labels[i]).strip() if i < len(labels) else str(i)
            lines.append(f"{lab}. {str(t).strip()}")
        return lines
    if isinstance(choices, list):
        return [f"{i}. {str(t).strip()}" for i, t in enumerate(choices)]
    return []


def iter_docs_from_qa_corpora(model_a_root: Path) -> Iterator[Dict[str, str]]:
    try:
        from datasets import load_from_disk
    except Exception as e:
        raise RuntimeError(
            "include_qa_corpora requires datasets package; please install huggingface datasets."
        ) from e

    # commonsense_qa_full
    csqa_root = model_a_root / "commonsense_qa_full"
    if csqa_root.exists():
        ds = load_from_disk(str(csqa_root))
        for split_name in ds.keys():
            for row in ds[split_name]:
                qid = str(row.get("id", "")).strip() or f"{split_name}:{hash(str(row))}"
                question = str(row.get("question", "")).strip()
                if not question:
                    continue
                lines = _choices_to_lines(row.get("choices"))
                answer_key = str(row.get("answerKey", "")).strip()
                answer_text = ""
                choices = row.get("choices")
                if isinstance(choices, dict):
                    labels = [str(x).strip() for x in choices.get("label", [])]
                    texts = [str(x).strip() for x in choices.get("text", [])]
                    if answer_key in labels:
                        idx = labels.index(answer_key)
                        if idx < len(texts):
                            answer_text = texts[idx]
                body_parts = []
                if lines:
                    body_parts.append("Choices:\n" + "\n".join(lines))
                if answer_key or answer_text:
                    body_parts.append(f"Gold answer: {answer_key} {answer_text}".strip())
                body = "\n\n".join(body_parts).strip()
                if not body:
                    continue
                yield {
                    "doc_id": f"csqa:{split_name}:{qid}",
                    "title": question,
                    "text": body,
                    "source": str(csqa_root),
                }

    # mmlu_all_full
    mmlu_root = model_a_root / "mmlu_all_full"
    if mmlu_root.exists():
        ds = load_from_disk(str(mmlu_root))
        for split_name in ds.keys():
            for i, row in enumerate(ds[split_name]):
                question = str(row.get("question", "")).strip()
                if not question:
                    continue
                subject = str(row.get("subject", "")).strip()
                choices = row.get("choices", [])
                lines = _choices_to_lines(choices)
                answer_idx_raw = row.get("answer", "")
                answer_idx = -1
                try:
                    answer_idx = int(answer_idx_raw)
                except Exception:
                    answer_idx = -1
                answer_text = ""
                if isinstance(choices, list) and 0 <= answer_idx < len(choices):
                    answer_text = str(choices[answer_idx]).strip()
                body_parts = []
                if subject:
                    body_parts.append(f"Subject: {subject}")
                if lines:
                    body_parts.append("Choices:\n" + "\n".join(lines))
                if answer_idx >= 0:
                    body_parts.append(f"Gold answer: {answer_idx}. {answer_text}".strip())
                body = "\n\n".join(body_parts).strip()
                if not body:
                    continue
                yield {
                    "doc_id": f"mmlu:{split_name}:{i}",
                    "title": question,
                    "text": body,
                    "source": str(mmlu_root),
                }

    # agentar_deepfinance_100k_full
    agent_root = model_a_root / "agentar_deepfinance_100k_full"
    if agent_root.exists():
        ds = load_from_disk(str(agent_root))
        for split_name in ds.keys():
            for row in ds[split_name]:
                rid = str(row.get("id", "")).strip() or f"{split_name}:{hash(str(row))}"
                messages = row.get("messages", [])
                if not isinstance(messages, list):
                    continue
                user_msg = ""
                assistant_msg = ""
                for m in messages:
                    if not isinstance(m, dict):
                        continue
                    role = str(m.get("role", "")).upper().strip()
                    content = str(m.get("content", "")).strip()
                    if not content:
                        continue
                    if not user_msg and role in {"HUMAN", "USER"}:
                        user_msg = content
                    elif not assistant_msg and role in {"ASSISTANT", "AI"}:
                        assistant_msg = _strip_reasoning_tags(content)
                    if user_msg and assistant_msg:
                        break
                if not user_msg or not assistant_msg:
                    continue
                yield {
                    "doc_id": f"agentar:{split_name}:{rid}",
                    "title": user_msg,
                    "text": assistant_msg,
                    "source": str(agent_root),
                }


def collect_parquet_sources(args: argparse.Namespace, rag_root: Path) -> Dict[str, List[Path]]:
    sources: Dict[str, List[Path]] = {"beir": default_corpus_parquets(rag_root)}

    if args.include_ms_marco:
        ms_patterns = [
            str(rag_root / "ms_marco" / "v2.1" / "*.parquet"),
            str(rag_root / "ms_marco" / "v1.1" / "*.parquet"),
        ]
        ms_files: List[Path] = []
        for p in ms_patterns:
            ms_files.extend(Path(x).resolve() for x in sorted(glob.glob(p)))
        sources["ms_marco"] = sorted(set(ms_files))

    if args.include_hotpot_context:
        hp_patterns = [
            str(rag_root / "hotpot_qa" / "fullwiki" / "*.parquet"),
            str(rag_root / "hotpot_qa" / "distractor" / "*.parquet"),
        ]
        hp_files: List[Path] = []
        for p in hp_patterns:
            hp_files.extend(Path(x).resolve() for x in sorted(glob.glob(p)))
        sources["hotpot_context"] = sorted(set(hp_files))

    if args.include_qa_pairs:
        nq_files = [Path(x).resolve() for x in sorted(glob.glob(str(rag_root / "nq_open" / "nq_open" / "*.parquet")))]
        model_a_root = rag_root.parent
        trivia_files = [
            Path(x).resolve()
            for x in sorted(glob.glob(str(model_a_root / "trivia_qa_rc_nocontext_files" / "rc.nocontext" / "*.parquet")))
        ]
        sources["nq_qa"] = nq_files
        sources["trivia_qa"] = trivia_files

    extra_files: List[Path] = []
    for extra in args.extra_corpus_glob:
        extra_files.extend(Path(p).resolve() for p in sorted(glob.glob(extra)))
    if extra_files:
        sources["extra"] = sorted(set(extra_files))
    return sources

def mean_pool(last_hidden_state, attention_mask):
    """
    对Transformer模型的输出做均值池化。
    仅统计有效token（由attention_mask指定）的向量均值作为sentence embedding。
    """
    mask = attention_mask.unsqueeze(-1).float()  # 转换mask形状并转float
    summed = (last_hidden_state * mask).sum(dim=1)   # 有效token向量求和
    denom = mask.sum(dim=1).clamp(min=1e-9)          # 有效token数
    return summed / denom                            # 求均值


def build_doc_iterators(args: argparse.Namespace, rag_root: Path) -> List[Tuple[str, Iterator[Dict[str, str]]]]:
    sources = collect_parquet_sources(args, rag_root)
    iterators: List[Tuple[str, Iterator[Dict[str, str]]]] = [
        ("beir", iter_docs_from_beir(sources.get("beir", []))),
        ("extra", iter_docs_from_beir(sources.get("extra", []))),
    ]
    if args.include_ms_marco:
        iterators.append(("ms_marco", iter_docs_from_ms_marco(sources.get("ms_marco", []))))
    if args.include_hotpot_context:
        iterators.append(("hotpot_context", iter_docs_from_hotpot_context(sources.get("hotpot_context", []))))
    if args.include_qa_pairs:
        iterators.append(("nq_qa", iter_docs_from_qa_pairs(sources.get("nq_qa", []), prefix="nq")))
        iterators.append(("trivia_qa", iter_docs_from_qa_pairs(sources.get("trivia_qa", []), prefix="trivia")))
    if args.include_qa_corpora:
        iterators.append(("qa_corpora", iter_docs_from_qa_corpora(rag_root.parent)))
    return iterators


def collect_source_file_stats(args: argparse.Namespace, rag_root: Path) -> Tuple[Dict[str, List[Path]], List[Path]]:
    sources = collect_parquet_sources(args, rag_root)
    all_parquet_files = sorted({p for files in sources.values() for p in files})
    return sources, all_parquet_files


def main() -> None:
    """
    主程序入口：
      1. 解析参数，设置环境变量
      2. 自动导入faiss、torch、transformers等依赖
      3. 收集语料parquet文件
      4. 加载检索器模型（优先断点权重）
      5. 读取并唯一化文档，组batch编码
      6. 计算每个文档embedding并标准化
      7. 用FAISS索引向量
      8. 将索引、文档与meta保存到输出目录
    """
    args = parse_args()
    # Ensure logs are flushed line-by-line in nohup/background runs.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)
    # 设置环境变量，确保后续torch等在对应GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    try:
        import faiss  # faiss可用CPU/GPU
    except Exception as e:
        raise RuntimeError(
            "缺少faiss依赖，请先安装：pip install faiss-cpu 或 faiss-gpu"
        ) from e

    # 延迟导入，确保环境变量已生效
    import torch
    from transformers import AutoModel, AutoTokenizer

    t0 = time.time()
    rag_root = Path(args.rag_root).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sources, all_parquet_files = collect_source_file_stats(args, rag_root)
    if not all_parquet_files:
        raise FileNotFoundError(f"No corpus parquet files found under {rag_root}")

    print(f"[info] found {len(all_parquet_files)} parquet corpus file(s)")
    for source_name, files in sources.items():
        if not files:
            continue
        print(f"  [{source_name}] {len(files)} files")

    def try_load(path: str):
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModel.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="auto")
        return tok, model

    used_source = args.checkpoint
    try:
        tokenizer, model = try_load(args.checkpoint)
    except Exception as e:
        print(f"[warn] checkpoint load failed, fallback to base model: {e}")
        tokenizer, model = try_load(args.base_model)
        used_source = args.base_model

    model.eval()

    seen_ids: set[str] = set()
    source_doc_counts: Dict[str, int] = {}
    iterators = build_doc_iterators(args, rag_root)
    batch_size = max(1, args.batch_size)
    index_path = out_dir / "index.faiss"
    docs_path = out_dir / "docs.jsonl"
    meta_path = out_dir / "meta.json"
    index = None
    total_docs = 0
    encoded_batches = 0
    pending_docs: List[Dict[str, str]] = []
    pending_texts: List[str] = []
    stop_requested = False

    def build_meta(final: bool) -> Dict[str, object]:
        return {
            "doc_count": total_docs,
            "embedding_dim": index.d if index is not None else None,
            "index_type": type(index).__name__ if index is not None else "",
            "checkpoint": args.checkpoint,
            "base_model": args.base_model,
            "used_source": used_source,
            "rag_root": str(rag_root),
            "parquet_files": [str(p) for p in all_parquet_files],
            "source_doc_counts": source_doc_counts,
            "include_ms_marco": bool(args.include_ms_marco),
            "include_hotpot_context": bool(args.include_hotpot_context),
            "include_qa_pairs": bool(args.include_qa_pairs),
            "include_qa_corpora": bool(args.include_qa_corpora),
            "checkpoint_every_docs": int(args.checkpoint_every_docs),
            "is_final": final,
            "build_sec": round(time.time() - t0, 2),
        }

    def write_checkpoint(final: bool) -> None:
        if index is None or total_docs == 0:
            return
        faiss.write_index(index, str(index_path))
        meta_path.write_text(json.dumps(build_meta(final=final), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        marker = "final" if final else "intermediate"
        print(f"[info] wrote {marker} checkpoint at {total_docs} docs")

    def _on_signal(signum, _frame):
        nonlocal stop_requested
        stop_requested = True
        print(f"[warn] signal={signum} received, will stop after current batch and persist checkpoint")

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    with docs_path.open("w", encoding="utf-8") as docs_writer:
        with torch.no_grad():
            def flush_pending() -> None:
                nonlocal index, total_docs, encoded_batches, pending_docs, pending_texts
                if not pending_docs:
                    return
                batch = tokenizer(
                    pending_texts,
                    padding=True,
                    truncation=True,
                    max_length=args.max_length,
                    return_tensors="pt",
                ).to(model.device)
                out = model(**batch)
                embs = mean_pool(out.last_hidden_state, batch["attention_mask"])
                embs = torch.nn.functional.normalize(embs, p=2, dim=1)
                mat = embs.detach().cpu().float().numpy().astype("float32")
                if index is None:
                    index = faiss.IndexFlatIP(int(mat.shape[1]))
                index.add(mat)
                for d in pending_docs:
                    docs_writer.write(json.dumps(d, ensure_ascii=False) + "\n")
                total_docs += len(pending_docs)
                encoded_batches += 1
                if encoded_batches % 20 == 0:
                    print(f"[info] encoded {total_docs} docs")
                if args.checkpoint_every_docs > 0 and total_docs % args.checkpoint_every_docs == 0:
                    docs_writer.flush()
                    write_checkpoint(final=False)
                pending_docs = []
                pending_texts = []

            for source_name, it in iterators:
                for row in it:
                    if stop_requested:
                        break
                    if row["doc_id"] in seen_ids:
                        continue
                    seen_ids.add(row["doc_id"])
                    source_doc_counts[source_name] = source_doc_counts.get(source_name, 0) + 1
                    if row["title"]:
                        text = row["title"] + "\n" + row["text"]
                    else:
                        text = row["text"]
                    pending_docs.append(row)
                    pending_texts.append("Represent this sentence for retrieval: " + text)
                    if len(pending_docs) >= batch_size:
                        flush_pending()
                    if args.max_docs > 0 and (total_docs + len(pending_docs)) >= args.max_docs:
                        flush_pending()
                        break
                if stop_requested:
                    break
                if args.max_docs > 0 and total_docs >= args.max_docs:
                    break
            flush_pending()
            docs_writer.flush()

    for name, cnt in sorted(source_doc_counts.items()):
        print(f"[info] source={name} docs={cnt}")

    if index is None or total_docs == 0:
        raise RuntimeError("No documents loaded from corpus files.")

    print(f"[info] loaded docs: {total_docs}")
    write_checkpoint(final=True)

    print("\n===== DONE =====")
    print(
        json.dumps(
            {
                "index": str(index_path),
                "docs": str(docs_path),
                "meta": str(meta_path),
                "stopped_early": stop_requested,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
