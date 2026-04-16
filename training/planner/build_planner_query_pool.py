#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build planner query pool from real datasets/logs")
    p.add_argument(
        "--metrics",
        type=str,
        default="/root/autodl-tmp/muti-llm/Data/Processed_data/orchestrator_metrics_math_fin_common.jsonl",
        help="Metrics JSONL from orchestrator benchmark runs.",
    )
    p.add_argument(
        "--trace",
        type=str,
        default="/root/autodl-tmp/muti-llm/Data/Processed_data/orchestrator_trace.jsonl",
        help="Global trace JSONL, used as extra query source.",
    )
    p.add_argument(
        "--output",
        type=str,
        default="/root/autodl-tmp/muti-llm/Data/Processed_data/planner_query_pool.jsonl",
    )
    p.add_argument("--max-total", type=int, default=1200)
    p.add_argument("--target-failure-ratio", type=float, default=0.30)
    p.add_argument(
        "--model-a-root",
        type=str,
        default="/root/autodl-tmp/muti-llm/Data/Model_A",
        help="Root path that contains mmlu_all_full / commonsense_qa_full / agentar_deepfinance_100k_full",
    )
    p.add_argument(
        "--use-model-a",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Extract real queries from Model_A datasets",
    )
    p.add_argument("--max-math", type=int, default=600, help="Max math queries from MMLU")
    p.add_argument("--max-finance", type=int, default=600, help="Max finance queries from Agentar/MMLU")
    p.add_argument("--max-common", type=int, default=600, help="Max commonsense queries from CSQA")
    p.add_argument(
        "--english-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep mostly ASCII queries to avoid Chinese pool contamination",
    )
    p.add_argument(
        "--extra-jsonl",
        action="append",
        default=[],
        help="Extra JSONL files to extract query-like instruction text (repeatable).",
    )
    p.add_argument("--max-extra", type=int, default=1000, help="Max rows to take from extra JSONL sources.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _normalize_query(q: str) -> str:
    return " ".join(str(q).split())


def _mostly_ascii(text: str, min_ratio: float = 0.90) -> bool:
    if not text:
        return False
    ascii_count = sum(1 for ch in text if ord(ch) < 128)
    return ascii_count / max(1, len(text)) >= min_ratio


def _collect_from_metrics(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    success: List[Dict[str, Any]] = []
    failure: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for r in rows:
        q = _normalize_query(str(r.get("query", "")).strip())
        if not q or q in seen:
            continue
        seen.add(q)
        item = {
            "id": str(r.get("benchmarkId", "")) or f"m_{len(seen):06d}",
            "query": q,
            "source": "metrics",
            "runId": str(r.get("runId", "")),
            "successHeuristic": bool(r.get("successHeuristic", False)),
            "reconstructRounds": int(r.get("reconstructRounds", 0) or 0),
        }
        if not item["successHeuristic"] or item["reconstructRounds"] > 0:
            failure.append(item)
        else:
            success.append(item)
    return success, failure


def _collect_from_trace(rows: List[Dict[str, Any]], seen_queries: Set[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        payload = r.get("payload", {})
        if not isinstance(payload, dict):
            continue
        q = _normalize_query(str(payload.get("query", "")).strip())
        if not q or q in seen_queries:
            continue
        seen_queries.add(q)
        run_id = str(payload.get("runId", ""))
        out.append(
            {
                "id": f"t_{len(seen_queries):06d}",
                "query": q,
                "source": "trace",
                "runId": run_id,
                "successHeuristic": True,
                "reconstructRounds": 0,
            }
        )
    return out


def _synthesize_failure_queries(base_rows: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if n <= 0 or not base_rows:
        return out
    for i in range(n):
        b = base_rows[i % len(base_rows)]
        q = str(b["query"])
        q2 = q if "__fail_b__" in q else f"{q} __fail_b__"
        out.append(
            {
                "id": f"syn_fail_{i:05d}",
                "query": q2,
                "source": "synthetic_fail",
                "runId": "",
                "successHeuristic": False,
                "reconstructRounds": 1,
            }
        )
    return out


def _collect_from_extra_jsonl(paths: List[str], seen_queries: Set[str], max_extra: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if max_extra <= 0:
        return out
    for raw_path in paths:
        path = Path(raw_path).resolve()
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if len(out) >= max_extra:
                    return out
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                q = ""
                if isinstance(obj.get("query"), str):
                    q = obj["query"]
                elif isinstance(obj.get("instruction"), str):
                    q = obj["instruction"]
                qn = _normalize_query(q.strip())
                if not qn or qn in seen_queries:
                    continue
                seen_queries.add(qn)
                out.append(
                    {
                        "id": f"x_{len(seen_queries):06d}",
                        "query": qn,
                        "source": "extra_jsonl",
                        "runId": "",
                        "successHeuristic": True,
                        "reconstructRounds": 0,
                    }
                )
    return out


def _collect_from_model_a(
    model_a_root: Path,
    seen_queries: Set[str],
    max_math: int,
    max_finance: int,
    max_common: int,
    english_only: bool,
) -> List[Dict[str, Any]]:
    try:
        from datasets import load_from_disk
    except Exception:
        return []

    out: List[Dict[str, Any]] = []
    added_math = 0
    added_fin = 0
    added_common = 0

    def add_query(q: str, src: str, prefix: str) -> None:
        nonlocal out, seen_queries
        qn = _normalize_query(q.strip())
        if not qn:
            return
        if english_only and not _mostly_ascii(qn):
            return
        if qn in seen_queries:
            return
        seen_queries.add(qn)
        out.append(
            {
                "id": f"{prefix}_{len(seen_queries):06d}",
                "query": qn,
                "source": src,
                "runId": "",
                "successHeuristic": True,
                "reconstructRounds": 0,
            }
        )

    # Math from MMLU
    mmlu_root = model_a_root / "mmlu_all_full"
    if mmlu_root.exists() and max_math > 0:
        ds = load_from_disk(str(mmlu_root))
        for split in ds.keys():
            for row in ds[split]:
                if added_math >= max_math:
                    break
                subj = str(row.get("subject", "")).lower()
                if not any(k in subj for k in ["math", "algebra", "geometry", "calculus", "statistics"]):
                    continue
                q = str(row.get("question", "")).strip()
                before = len(out)
                add_query(q, "model_a:mmlu_math", "math")
                if len(out) > before:
                    added_math += 1
            if added_math >= max_math:
                break

    # Common sense from CSQA
    csqa_root = model_a_root / "commonsense_qa_full"
    if csqa_root.exists() and max_common > 0:
        ds = load_from_disk(str(csqa_root))
        for split in ds.keys():
            for row in ds[split]:
                if added_common >= max_common:
                    break
                q = str(row.get("question", "")).strip()
                before = len(out)
                add_query(q, "model_a:commonsense_qa", "common")
                if len(out) > before:
                    added_common += 1
            if added_common >= max_common:
                break

    # Finance from agentar (user messages) and MMLU economics/business subjects
    agent_root = model_a_root / "agentar_deepfinance_100k_full"
    if agent_root.exists() and max_finance > 0:
        ds = load_from_disk(str(agent_root))
        for split in ds.keys():
            for row in ds[split]:
                if added_fin >= max_finance:
                    break
                msgs = row.get("messages", [])
                if not isinstance(msgs, list):
                    continue
                user_q = ""
                for m in msgs:
                    if not isinstance(m, dict):
                        continue
                    role = str(m.get("role", "")).upper().strip()
                    content = str(m.get("content", "")).strip()
                    if role in {"USER", "HUMAN"} and content:
                        user_q = content
                        break
                if not user_q:
                    continue
                before = len(out)
                add_query(user_q, "model_a:agentar_finance", "fin")
                if len(out) > before:
                    added_fin += 1
            if added_fin >= max_finance:
                break

    if mmlu_root.exists() and added_fin < max_finance:
        ds = load_from_disk(str(mmlu_root))
        for split in ds.keys():
            for row in ds[split]:
                if added_fin >= max_finance:
                    break
                subj = str(row.get("subject", "")).lower()
                if not any(k in subj for k in ["econom", "business", "finance", "account"]):
                    continue
                q = str(row.get("question", "")).strip()
                before = len(out)
                add_query(q, "model_a:mmlu_finance", "fin")
                if len(out) > before:
                    added_fin += 1
            if added_fin >= max_finance:
                break

    return out


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    metrics_rows = _read_jsonl(Path(args.metrics).resolve())
    trace_rows = _read_jsonl(Path(args.trace).resolve())

    success, failure = _collect_from_metrics(metrics_rows)
    seen = {x["query"] for x in success} | {x["query"] for x in failure}
    if args.use_model_a:
        success.extend(
            _collect_from_model_a(
                model_a_root=Path(args.model_a_root).resolve(),
                seen_queries=seen,
                max_math=args.max_math,
                max_finance=args.max_finance,
                max_common=args.max_common,
                english_only=args.english_only,
            )
        )
    success.extend(_collect_from_trace(trace_rows, seen))
    success.extend(_collect_from_extra_jsonl(args.extra_jsonl, seen, args.max_extra))

    random.shuffle(success)
    random.shuffle(failure)

    max_total = max(10, int(args.max_total))
    # Keep the final ratio close to target, but never force synthetic failures
    # beyond what available successful samples can support.
    target_ratio = max(0.0, min(0.95, float(args.target_failure_ratio)))
    target_fail = int(max_total * target_ratio)
    max_fail_supported_by_success = int(len(success) * target_ratio / max(1e-9, (1.0 - target_ratio)))
    target_fail = min(target_fail, max_fail_supported_by_success + len(failure))
    picked_failure = failure[:target_fail]

    remain = max_total - len(picked_failure)
    picked_success = success[: max(0, remain)]
    # Optionally synthesize a small number of failure-like prompts so that
    # failure ratio tracks the target under current success pool size.
    if target_ratio > 0 and picked_success:
        fail_target_from_success = int(len(picked_success) * target_ratio / max(1e-9, (1.0 - target_ratio)))
        synth_n = max(0, fail_target_from_success - len(picked_failure))
        if synth_n > 0:
            picked_failure.extend(_synthesize_failure_queries(picked_success if picked_success else failure, synth_n))

    rows = picked_failure + picked_success
    random.shuffle(rows)
    if len(rows) > max_total:
        rows = rows[:max_total]

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"id": r["id"], "query": r["query"]}, ensure_ascii=False) + "\n")

    stats = {
        "output": str(out_path),
        "total": len(rows),
        "failureLike": sum(1 for x in rows if "__fail_b__" in x["query"]),
        "fromMetricsSuccess": len(success),
        "fromMetricsFailure": len(failure),
        "targetFailureRatio": args.target_failure_ratio,
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
