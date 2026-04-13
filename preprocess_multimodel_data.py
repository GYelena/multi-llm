#!/usr/bin/env python3
"""
Preprocess local datasets for multi-model training.

Outputs:
  - A_retrieval.jsonl     (Model A retriever/reranker)
  - A_generation.jsonl    (Model A RAG generator)
  - B_sft.jsonl           (Model B LoRA SFT)
  - C_sft.jsonl           (Model C LoRA SFT)

This script is designed for the local folder layout:
  /root/autodl-tmp/muti-llm/Data/Model_A
  /root/autodl-tmp/muti-llm/Data/Model_B
  /root/autodl-tmp/muti-llm/Data/Model_C

"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pyarrow.parquet as pq
from datasets import Dataset, DatasetDict, IterableDataset, load_dataset, load_from_disk


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(msg, flush=True)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def iter_dataset_rows(obj: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(obj, (Dataset, IterableDataset)):
        for ex in obj:
            yield ex
        return
    if isinstance(obj, DatasetDict):
        for split in obj.keys():
            for ex in obj[split]:
                yield ex
        return
    raise TypeError(f"Unsupported dataset object type: {type(obj)}")


def load_local_disk(path: Path) -> Any:
    return load_from_disk(str(path))


def load_local_parquet_stream(pattern: str) -> Iterable[Dict[str, Any]]:
    files = sorted(glob.glob(pattern))
    if not files:
        return iter(())
    ds = load_dataset("parquet", data_files=files, split="train", streaming=True)
    return iter_dataset_rows(ds)


def first_non_empty_str(values: List[Any]) -> str:
    for v in values:
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def parse_agentar_messages(messages: List[Dict[str, Any]]) -> Tuple[str, str]:
    user_msg = ""
    assistant_msg = ""
    for m in messages:
        role = str(m.get("role", "")).upper()
        content = str(m.get("content", "")).strip()
        if not content:
            continue
        if not user_msg and role == "HUMAN":
            user_msg = content
        elif not assistant_msg and role == "ASSISTANT":
            assistant_msg = content
        if user_msg and assistant_msg:
            break
    return user_msg, assistant_msg


def parse_openthoughts_conversations(conversations: List[Dict[str, Any]]) -> Tuple[str, str]:
    user_msg = ""
    assistant_msg = ""
    for turn in conversations:
        role = str(turn.get("from", "")).lower()
        val = str(turn.get("value", "")).strip()
        if not val:
            continue
        if not user_msg and role == "user":
            user_msg = val
        elif user_msg and not assistant_msg and role == "assistant":
            assistant_msg = val
            break
    return user_msg, assistant_msg


def build_beir_retrieval_rows(
    queries_parquet: Path,
    corpus_parquet: Path,
    qrels_tsv: Path,
    source_name: str,
    neg_count: int,
    max_queries: Optional[int],
    seed: int,
    log_every: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    log(f"[A_retrieval][{source_name}] loading queries: {queries_parquet}")

    # query_id -> query text
    query_text_map: Dict[str, str] = {}
    qf = pq.ParquetFile(str(queries_parquet))
    q_count = 0
    for batch in qf.iter_batches(batch_size=4096):
        data = batch.to_pydict()
        ids = data.get("_id", [])
        titles = data.get("title", [])
        texts = data.get("text", [])
        for qid, title, text in zip(ids, titles, texts):
            qid_s = str(qid)
            qtxt = first_non_empty_str([text, title])
            if qtxt:
                query_text_map[qid_s] = qtxt
                q_count += 1
    log(f"[A_retrieval][{source_name}] loaded {q_count} queries")

    # qrels: query_id -> positive doc ids
    positives_by_query: Dict[str, List[str]] = {}
    needed_doc_ids: set[str] = set()
    log(f"[A_retrieval][{source_name}] loading qrels: {qrels_tsv}")
    with qrels_tsv.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f, delimiter="\t")
        qrels_count = 0
        for row in reader:
            qid = str(row.get("query-id", "")).strip()
            did = str(row.get("corpus-id", "")).strip()
            score_raw = row.get("score", "1")
            if not qid or not did:
                continue
            try:
                score = float(score_raw)
            except ValueError:
                score = 0.0
            if score <= 0:
                continue
            positives_by_query.setdefault(qid, []).append(did)
            needed_doc_ids.add(did)
            qrels_count += 1
    log(f"[A_retrieval][{source_name}] loaded {qrels_count} positive qrels pairs")

    # Build doc map only for needed positive IDs.
    doc_map: Dict[str, Dict[str, str]] = {}
    log(f"[A_retrieval][{source_name}] loading corpus: {corpus_parquet}")
    cf = pq.ParquetFile(str(corpus_parquet))
    for batch in cf.iter_batches(batch_size=8192):
        data = batch.to_pydict()
        ids = data.get("_id", [])
        titles = data.get("title", [])
        texts = data.get("text", [])
        for did, title, text in zip(ids, titles, texts):
            did_s = str(did)
            if did_s not in needed_doc_ids:
                continue
            doc_map[did_s] = {
                "doc_id": did_s,
                "title": str(title or ""),
                "text": str(text or ""),
            }
    log(f"[A_retrieval][{source_name}] indexed {len(doc_map)} positive docs")

    candidate_neg_ids = list(doc_map.keys())
    rows: List[Dict[str, Any]] = []

    for qid, pos_ids in positives_by_query.items():
        qtxt = query_text_map.get(qid, "")
        if not qtxt:
            continue

        positives = [doc_map[pid] for pid in pos_ids if pid in doc_map]
        if not positives:
            continue

        pos_id_set = {p["doc_id"] for p in positives}
        neg_pool = [x for x in candidate_neg_ids if x not in pos_id_set]
        if neg_pool:
            sampled_neg_ids = rng.sample(neg_pool, k=min(neg_count, len(neg_pool)))
            negatives = [doc_map[nid] for nid in sampled_neg_ids]
        else:
            negatives = []

        rows.append(
            {
                "id": f"{source_name}:{qid}",
                "task_type": "retrieval",
                "query": qtxt,
                "positive_passages": positives,
                "negative_passages": negatives,
                "source": source_name,
            }
        )
        if max_queries is not None and len(rows) >= max_queries:
            break
        if log_every and len(rows) % log_every == 0:
            log(f"[A_retrieval][{source_name}] built rows: {len(rows)}")

    log(f"[A_retrieval][{source_name}] done, rows={len(rows)}")
    return rows


def build_a_retrieval(args: argparse.Namespace) -> int:
    model_a = Path(args.model_a_dir)
    rag = model_a / "rag_english_v2"
    out_path = Path(args.output_dir) / "A_retrieval.jsonl"

    rows: List[Dict[str, Any]] = []
    neg_count = args.neg_count
    max_ms = args.max_a_ms_marco
    max_beir = args.max_a_beir

    log("[A_retrieval] start")
    # 1) MS MARCO
    ms_patterns = [
        str(rag / "ms_marco" / "v2.1" / "train-*.parquet"),
        str(rag / "ms_marco" / "v1.1" / "train-*.parquet"),
    ]
    ms_files: List[str] = []
    for p in ms_patterns:
        ms_files.extend(sorted(glob.glob(p)))
    if ms_files:
        log(f"[A_retrieval][ms_marco] loading {len(ms_files)} parquet files")
        ms_ds = load_dataset("parquet", data_files=ms_files, split="train", streaming=True)
        count = 0
        for ex in ms_ds:
            query = str(ex.get("query", "")).strip()
            passages = ex.get("passages", {})
            p_texts = passages.get("passage_text", []) if isinstance(passages, dict) else []
            p_selected = passages.get("is_selected", []) if isinstance(passages, dict) else []

            if not query or not p_texts:
                continue

            positives = []
            negatives = []
            qid = str(ex.get("query_id", f"ms_{count}"))
            for i, txt in enumerate(p_texts):
                text = str(txt or "").strip()
                if not text:
                    continue
                flag = False
                if i < len(p_selected):
                    v = p_selected[i]
                    flag = bool(v) and str(v) not in {"0", "False", "false"}
                item = {"doc_id": f"{qid}_p{i}", "title": "", "text": text}
                if flag:
                    positives.append(item)
                else:
                    negatives.append(item)
            if not positives:
                continue
            negatives = negatives[:neg_count]

            rows.append(
                {
                    "id": f"ms_marco:{qid}",
                    "task_type": "retrieval",
                    "query": query,
                    "positive_passages": positives,
                    "negative_passages": negatives,
                    "source": "ms_marco",
                }
            )
            count += 1
            if max_ms is not None and count >= max_ms:
                break
            if args.log_every and count % args.log_every == 0:
                log(f"[A_retrieval][ms_marco] built rows: {count}")
        log(f"[A_retrieval][ms_marco] done, rows={count}")

    # 2) BEIR HotpotQA
    hotpot_rows = build_beir_retrieval_rows(
        queries_parquet=rag / "beir_hotpotqa" / "queries" / "queries-00000-of-00001.parquet",
        corpus_parquet=rag / "beir_hotpotqa" / "corpus" / "corpus-00000-of-00001.parquet",
        qrels_tsv=rag / "beir_hotpotqa_qrels" / "train.tsv",
        source_name="beir_hotpotqa",
        neg_count=neg_count,
        max_queries=max_beir,
        seed=args.seed,
        log_every=args.log_every,
    )
    rows.extend(hotpot_rows)

    # 3) BEIR FiQA
    fiqa_rows = build_beir_retrieval_rows(
        queries_parquet=rag / "beir_fiqa" / "queries" / "queries-00000-of-00001.parquet",
        corpus_parquet=rag / "beir_fiqa" / "corpus" / "corpus-00000-of-00001.parquet",
        qrels_tsv=rag / "beir_fiqa_qrels" / "train.tsv",
        source_name="beir_fiqa",
        neg_count=neg_count,
        max_queries=max_beir,
        seed=args.seed,
        log_every=args.log_every,
    )
    rows.extend(fiqa_rows)

    ensure_dir(out_path.parent)
    written = write_jsonl(out_path, rows)
    log(f"[A_retrieval] wrote {written} rows -> {out_path}")
    return written


def extract_hotpot_citations(context: Dict[str, Any], supporting: Dict[str, Any]) -> List[Dict[str, str]]:
    citations: List[Dict[str, str]] = []
    titles = context.get("title", []) if isinstance(context, dict) else []
    sentences = context.get("sentences", []) if isinstance(context, dict) else []

    sf_titles = supporting.get("title", []) if isinstance(supporting, dict) else []
    sf_sent_ids = supporting.get("sent_id", []) if isinstance(supporting, dict) else []
    for t, sid in zip(sf_titles, sf_sent_ids):
        try:
            sid_int = int(sid)
        except Exception:
            continue
        matched_quote = ""
        matched_doc_id = str(t)
        for doc_title, doc_sents in zip(titles, sentences):
            if str(doc_title) != str(t):
                continue
            if isinstance(doc_sents, list) and 0 <= sid_int < len(doc_sents):
                matched_quote = str(doc_sents[sid_int]).strip()
            break
        citations.append({"doc_id": matched_doc_id, "quote": matched_quote})
    return citations


def build_a_generation(args: argparse.Namespace) -> int:
    model_a = Path(args.model_a_dir)
    rag = model_a / "rag_english_v2"
    out_path = Path(args.output_dir) / "A_generation.jsonl"
    rows: List[Dict[str, Any]] = []

    log("[A_generation] start")
    # HotpotQA (fullwiki + distractor)
    hp_files = sorted(glob.glob(str(rag / "hotpot_qa" / "fullwiki" / "*.parquet")))
    hp_files += sorted(glob.glob(str(rag / "hotpot_qa" / "distractor" / "*.parquet")))
    hp_seen: set[str] = set()
    if hp_files:
        log(f"[A_generation][hotpot_qa] loading {len(hp_files)} parquet files")
        hp_ds = load_dataset("parquet", data_files=hp_files, split="train", streaming=True)
        count = 0
        for ex in hp_ds:
            ex_id = str(ex.get("id", ""))
            if ex_id and ex_id in hp_seen:
                continue
            if ex_id:
                hp_seen.add(ex_id)

            question = str(ex.get("question", "")).strip()
            answer = str(ex.get("answer", "")).strip()
            if not question or not answer:
                continue

            context = ex.get("context", {})
            passages = []
            if isinstance(context, dict):
                titles = context.get("title", [])
                sents = context.get("sentences", [])
                for i, (t, ss) in enumerate(zip(titles, sents)):
                    if isinstance(ss, list):
                        text = " ".join(str(x).strip() for x in ss if str(x).strip())
                    else:
                        text = str(ss or "").strip()
                    passages.append({"doc_id": str(t) or f"doc_{i}", "title": str(t or ""), "text": text})

            citations = extract_hotpot_citations(context, ex.get("supporting_facts", {}))
            rows.append(
                {
                    "id": f"hotpot_qa:{ex_id or count}",
                    "task_type": "rag_generation",
                    "instruction": "Answer the question using only provided passages and cite evidence.",
                    "input": {"query": question, "retrieved_passages": passages},
                    "output": {"answer": answer, "citations": citations, "cannot_answer": False},
                    "source": "hotpot_qa",
                }
            )
            count += 1
            if args.max_a_hotpot is not None and count >= args.max_a_hotpot:
                break
            if args.log_every and count % args.log_every == 0:
                log(f"[A_generation][hotpot_qa] built rows: {count}")
        log(f"[A_generation][hotpot_qa] done, rows={count}")

    # NQ Open (no passages in raw data)
    nq_files = sorted(glob.glob(str(rag / "nq_open" / "nq_open" / "*.parquet")))
    if nq_files:
        log(f"[A_generation][nq_open] loading {len(nq_files)} parquet files")
        nq_ds = load_dataset("parquet", data_files=nq_files, split="train", streaming=True)
        count = 0
        for ex in nq_ds:
            q = str(ex.get("question", "")).strip()
            ans = ex.get("answer", [])
            ans_text = ""
            if isinstance(ans, list) and ans:
                ans_text = str(ans[0]).strip()
            elif isinstance(ans, str):
                ans_text = ans.strip()
            if not q or not ans_text:
                continue
            rows.append(
                {
                    "id": f"nq_open:{count}",
                    "task_type": "rag_generation",
                    "instruction": "Answer the question concisely.",
                    "input": {"query": q, "retrieved_passages": []},
                    "output": {"answer": ans_text, "citations": [], "cannot_answer": False},
                    "source": "nq_open",
                }
            )
            count += 1
            if args.max_a_nq is not None and count >= args.max_a_nq:
                break
            if args.log_every and count % args.log_every == 0:
                log(f"[A_generation][nq_open] built rows: {count}")
        log(f"[A_generation][nq_open] done, rows={count}")

    # TriviaQA nocontext
    tr_files = sorted(glob.glob(str(model_a / "trivia_qa_rc_nocontext_files" / "rc.nocontext" / "*.parquet")))
    if tr_files:
        log(f"[A_generation][trivia_qa] loading {len(tr_files)} parquet files")
        tr_ds = load_dataset("parquet", data_files=tr_files, split="train", streaming=True)
        count = 0
        for ex in tr_ds:
            q = str(ex.get("question", "")).strip()
            answer_obj = ex.get("answer", {})
            aliases = answer_obj.get("aliases", []) if isinstance(answer_obj, dict) else []
            ans_text = str(aliases[0]).strip() if aliases else ""
            if not q or not ans_text:
                continue
            rows.append(
                {
                    "id": f"trivia_qa:{ex.get('question_id', count)}",
                    "task_type": "rag_generation",
                    "instruction": "Answer the question concisely.",
                    "input": {"query": q, "retrieved_passages": []},
                    "output": {"answer": ans_text, "citations": [], "cannot_answer": False},
                    "source": "trivia_qa_rc_nocontext",
                }
            )
            count += 1
            if args.max_a_trivia is not None and count >= args.max_a_trivia:
                break
            if args.log_every and count % args.log_every == 0:
                log(f"[A_generation][trivia_qa] built rows: {count}")
        log(f"[A_generation][trivia_qa] done, rows={count}")

    ensure_dir(out_path.parent)
    written = write_jsonl(out_path, rows)
    log(f"[A_generation] wrote {written} rows -> {out_path}")
    return written


def build_b_sft(args: argparse.Namespace) -> int:
    model_b = Path(args.model_b_dir)
    out_path = Path(args.output_dir) / "B_sft.jsonl"
    rows: List[Dict[str, Any]] = []

    log("[B_sft] start")
    # NuminaMath
    log("[B_sft][numina] loading dataset from disk")
    numina = load_local_disk(model_b / "numinamath_1_5_full")
    count = 0
    for ex in iter_dataset_rows(numina):
        problem = str(ex.get("problem", "")).strip()
        solution = str(ex.get("solution", "")).strip()
        if not problem or not solution:
            continue
        rows.append(
            {
                "id": f"numina:{count}",
                "task_type": "reasoning",
                "instruction": "Solve the problem step by step and provide the final answer.",
                "input": problem,
                "output": solution,
                "meta": {"source": "numinamath_1_5", "domain": "math", "difficulty": "unknown"},
            }
        )
        count += 1
        if args.max_b_numina is not None and count >= args.max_b_numina:
            break
        if args.log_every and count % args.log_every == 0:
            log(f"[B_sft][numina] built rows: {count}")
    log(f"[B_sft][numina] done, rows={count}")

    # OpenMathReasoning sample
    omr_files = sorted(glob.glob(str(model_b / "openmathreasoning_cot_sample_files" / "data" / "*.parquet")))
    if omr_files:
        log(f"[B_sft][openmath] loading {len(omr_files)} parquet files")
        omr = load_dataset("parquet", data_files=omr_files, split="train", streaming=True)
        count = 0
        for ex in omr:
            problem = str(ex.get("problem", "")).strip()
            solution = str(ex.get("generated_solution", "")).strip()
            if not problem or not solution:
                continue
            rows.append(
                {
                    "id": f"openmath:{count}",
                    "task_type": "reasoning",
                    "instruction": "Solve the problem step by step and provide the final answer.",
                    "input": problem,
                    "output": solution,
                    "meta": {"source": "openmathreasoning", "domain": "math", "difficulty": "unknown"},
                }
            )
            count += 1
            if args.max_b_openmath is not None and count >= args.max_b_openmath:
                break
            if args.log_every and count % args.log_every == 0:
                log(f"[B_sft][openmath] built rows: {count}")
        log(f"[B_sft][openmath] done, rows={count}")

    # reasoning-v1 sample
    r1_files = sorted(glob.glob(str(model_b / "reasoning_v1_20m_sample_files" / "data" / "*.parquet")))
    if r1_files:
        log(f"[B_sft][reasoning_v1] loading {len(r1_files)} parquet files")
        r1 = load_dataset("parquet", data_files=r1_files, split="train", streaming=True)
        count = 0
        for ex in r1:
            prompt = str(ex.get("prompt", "")).strip()
            response = str(ex.get("response", "")).strip()
            if not prompt or not response:
                continue
            rows.append(
                {
                    "id": f"reasoning_v1:{count}",
                    "task_type": "reasoning",
                    "instruction": "Reason carefully and answer the question.",
                    "input": prompt,
                    "output": response,
                    "meta": {"source": "reasoning_v1_20m_sample", "domain": "general", "difficulty": "unknown"},
                }
            )
            count += 1
            if args.max_b_reasoning_v1 is not None and count >= args.max_b_reasoning_v1:
                break
            if args.log_every and count % args.log_every == 0:
                log(f"[B_sft][reasoning_v1] built rows: {count}")
        log(f"[B_sft][reasoning_v1] done, rows={count}")

    # OpenThoughts
    ot_files = sorted(glob.glob(str(model_b / "openthoughts_114k_default_files" / "data" / "*.parquet")))
    if ot_files:
        log(f"[B_sft][openthoughts] loading {len(ot_files)} parquet files")
        ot = load_dataset("parquet", data_files=ot_files, split="train", streaming=True)
        count = 0
        for ex in ot:
            user_msg, assistant_msg = parse_openthoughts_conversations(ex.get("conversations", []))
            if not user_msg or not assistant_msg:
                continue
            rows.append(
                {
                    "id": f"openthoughts:{count}",
                    "task_type": "reasoning",
                    "instruction": "Solve the user problem with clear reasoning.",
                    "input": user_msg,
                    "output": assistant_msg,
                    "meta": {"source": "openthoughts_114k", "domain": "general", "difficulty": "unknown"},
                }
            )
            count += 1
            if args.max_b_openthoughts is not None and count >= args.max_b_openthoughts:
                break
            if args.log_every and count % args.log_every == 0:
                log(f"[B_sft][openthoughts] built rows: {count}")
        log(f"[B_sft][openthoughts] done, rows={count}")

    # Agentar finance
    log("[B_sft][agentar] loading dataset from disk")
    agentar = load_local_disk(model_b / "agentar_deepfinance_100k_full")
    count = 0
    for ex in iter_dataset_rows(agentar):
        user_msg, assistant_msg = parse_agentar_messages(ex.get("messages", []))
        if not user_msg or not assistant_msg:
            continue
        rows.append(
            {
                "id": f"agentar:{ex.get('id', count)}",
                "task_type": "reasoning",
                "instruction": "Solve the financial question with clear reasoning.",
                "input": user_msg,
                "output": assistant_msg,
                "meta": {"source": "agentar_deepfinance_100k", "domain": "finance", "difficulty": "unknown"},
            }
        )
        count += 1
        if args.max_b_agentar is not None and count >= args.max_b_agentar:
            break
        if args.log_every and count % args.log_every == 0:
            log(f"[B_sft][agentar] built rows: {count}")
    log(f"[B_sft][agentar] done, rows={count}")

    ensure_dir(out_path.parent)
    written = write_jsonl(out_path, rows)
    log(f"[B_sft] wrote {written} rows -> {out_path}")
    return written


def build_c_sft(args: argparse.Namespace) -> int:
    model_c = Path(args.model_c_dir)
    out_path = Path(args.output_dir) / "C_sft.jsonl"
    rows: List[Dict[str, Any]] = []

    log("[C_sft] start")
    # CNN/DailyMail
    log("[C_sft][cnn_dailymail] loading dataset from disk")
    cnndm = load_local_disk(model_c / "cnn_dailymail_3_0_0_full")
    count = 0
    for ex in iter_dataset_rows(cnndm):
        article = str(ex.get("article", "")).strip()
        highlights = str(ex.get("highlights", "")).strip()
        if not article or not highlights:
            continue
        rows.append(
            {
                "id": f"cnn_dailymail:{ex.get('id', count)}",
                "task_type": "writing",
                "instruction": "Summarize the input article into a concise, coherent paragraph.",
                "input": article,
                "output": highlights,
                "meta": {"source": "cnn_dailymail_3_0_0", "domain": "general", "difficulty": "unknown"},
            }
        )
        count += 1
        if args.max_c_cnn is not None and count >= args.max_c_cnn:
            break
        if args.log_every and count % args.log_every == 0:
            log(f"[C_sft][cnn_dailymail] built rows: {count}")
    log(f"[C_sft][cnn_dailymail] done, rows={count}")

    # OpenThoughts (for structured writing / response style)
    ot_files = sorted(glob.glob(str(model_c / "openthoughts_114k_default_files" / "data" / "*.parquet")))
    if ot_files:
        log(f"[C_sft][openthoughts] loading {len(ot_files)} parquet files")
        ot = load_dataset("parquet", data_files=ot_files, split="train", streaming=True)
        count = 0
        for ex in ot:
            user_msg, assistant_msg = parse_openthoughts_conversations(ex.get("conversations", []))
            if not user_msg or not assistant_msg:
                continue
            rows.append(
                {
                    "id": f"openthoughts_c:{count}",
                    "task_type": "writing",
                    "instruction": "Provide a clear, coherent final response to the user request.",
                    "input": user_msg,
                    "output": assistant_msg,
                    "meta": {"source": "openthoughts_114k", "domain": "general", "difficulty": "unknown"},
                }
            )
            count += 1
            if args.max_c_openthoughts is not None and count >= args.max_c_openthoughts:
                break
            if args.log_every and count % args.log_every == 0:
                log(f"[C_sft][openthoughts] built rows: {count}")
        log(f"[C_sft][openthoughts] done, rows={count}")

    # Agentar
    log("[C_sft][agentar] loading dataset from disk")
    agentar = load_local_disk(model_c / "agentar_deepfinance_100k_full")
    count = 0
    for ex in iter_dataset_rows(agentar):
        user_msg, assistant_msg = parse_agentar_messages(ex.get("messages", []))
        if not user_msg or not assistant_msg:
            continue
        rows.append(
            {
                "id": f"agentar_c:{ex.get('id', count)}",
                "task_type": "writing",
                "instruction": "Write a professional and coherent response for the financial query.",
                "input": user_msg,
                "output": assistant_msg,
                "meta": {"source": "agentar_deepfinance_100k", "domain": "finance", "difficulty": "unknown"},
            }
        )
        count += 1
        if args.max_c_agentar is not None and count >= args.max_c_agentar:
            break
        if args.log_every and count % args.log_every == 0:
            log(f"[C_sft][agentar] built rows: {count}")
    log(f"[C_sft][agentar] done, rows={count}")

    # TriviaQA nocontext
    tr_files = sorted(glob.glob(str(model_c / "trivia_qa_rc_nocontext_files" / "rc.nocontext" / "*.parquet")))
    if tr_files:
        log(f"[C_sft][trivia_qa] loading {len(tr_files)} parquet files")
        tr = load_dataset("parquet", data_files=tr_files, split="train", streaming=True)
        count = 0
        for ex in tr:
            q = str(ex.get("question", "")).strip()
            answer_obj = ex.get("answer", {})
            aliases = answer_obj.get("aliases", []) if isinstance(answer_obj, dict) else []
            ans_text = str(aliases[0]).strip() if aliases else ""
            if not q or not ans_text:
                continue
            rows.append(
                {
                    "id": f"trivia_c:{ex.get('question_id', count)}",
                    "task_type": "writing",
                    "instruction": "Provide a concise and fluent answer to the question.",
                    "input": q,
                    "output": ans_text,
                    "meta": {"source": "trivia_qa_rc_nocontext", "domain": "general", "difficulty": "unknown"},
                }
            )
            count += 1
            if args.max_c_trivia is not None and count >= args.max_c_trivia:
                break
            if args.log_every and count % args.log_every == 0:
                log(f"[C_sft][trivia_qa] built rows: {count}")
        log(f"[C_sft][trivia_qa] done, rows={count}")

    ensure_dir(out_path.parent)
    written = write_jsonl(out_path, rows)
    log(f"[C_sft] wrote {written} rows -> {out_path}")
    return written


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Preprocess datasets for A/B/C models.")
    p.add_argument("--model-a-dir", default="/root/autodl-tmp/muti-llm/Data/Model_A")
    p.add_argument("--model-b-dir", default="/root/autodl-tmp/muti-llm/Data/Model_B")
    p.add_argument("--model-c-dir", default="/root/autodl-tmp/muti-llm/Data/Model_C")
    p.add_argument("--output-dir", default="/root/autodl-tmp/muti-llm/Data/Processed_data")
    p.add_argument(
        "--target",
        choices=["all", "A_retrieval", "A_generation", "B_sft", "C_sft"],
        default="all",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--neg-count", type=int, default=5, help="Negatives per query for retrieval rows.")

    # Optional sampling limits
    p.add_argument("--max-a-ms-marco", type=int, default=None)
    p.add_argument("--max-a-beir", type=int, default=None)
    p.add_argument("--max-a-hotpot", type=int, default=None)
    p.add_argument("--max-a-nq", type=int, default=None)
    p.add_argument("--max-a-trivia", type=int, default=None)

    p.add_argument("--max-b-numina", type=int, default=None)
    p.add_argument("--max-b-openmath", type=int, default=None)
    p.add_argument("--max-b-reasoning-v1", type=int, default=None)
    p.add_argument("--max-b-openthoughts", type=int, default=None)
    p.add_argument("--max-b-agentar", type=int, default=None)

    p.add_argument("--max-c-cnn", type=int, default=None)
    p.add_argument("--max-c-openthoughts", type=int, default=None)
    p.add_argument("--max-c-agentar", type=int, default=None)
    p.add_argument("--max-c-trivia", type=int, default=None)
    p.add_argument(
        "--log-every",
        type=int,
        default=50000,
        help="Print progress every N accepted rows in long loops. Set 0 to disable.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)
    log("[main] preprocessing started")
    log(f"[main] target={args.target}, output_dir={out_dir}")
    if args.log_every:
        log(f"[main] progress logging enabled: every {args.log_every} rows")
    else:
        log("[main] progress logging disabled")

    results: Dict[str, int] = {}
    if args.target in ("all", "A_retrieval"):
        results["A_retrieval"] = build_a_retrieval(args)
    if args.target in ("all", "A_generation"):
        results["A_generation"] = build_a_generation(args)
    if args.target in ("all", "B_sft"):
        results["B_sft"] = build_b_sft(args)
    if args.target in ("all", "C_sft"):
        results["C_sft"] = build_c_sft(args)

    print("Preprocessing done.")
    for k, v in results.items():
        print(f"{k}: {v} rows")


if __name__ == "__main__":
    main()

