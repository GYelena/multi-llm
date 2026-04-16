#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class OpenAICompatJudge:
    def __init__(self, base_url: str, api_key: str, model: str, timeout_seconds: int) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds

    def decide(self, prompt_obj: Dict[str, Any], cand_a: Dict[str, Any], cand_b: Dict[str, Any]) -> Tuple[str, str]:
        system_prompt = (
            "You are a strict planner judge. Compare candidate A and B for the same planner prompt. "
            "Prefer the more executable, lower-risk, and lower-cost candidate. "
            "Return ONLY JSON: {\"preferred\":\"A|B\",\"reason\":\"...\"}."
        )
        user_payload = {
            "prompt": prompt_obj,
            "candidateA": cand_a,
            "candidateB": cand_b,
            "criteria": [
                "json validity and schema consistency",
                "dependency correctness",
                "cost and simplicity",
                "expected utility",
            ],
        }
        text = self._chat(system_prompt, json.dumps(user_payload, ensure_ascii=False), temperature=0.0)
        parsed = self._extract_json(text)
        preferred = str(parsed.get("preferred", "")).strip().upper()
        if preferred not in {"A", "B"}:
            raise ValueError(f"judge returned invalid preferred={preferred}")
        reason = str(parsed.get("reason", "")).strip()
        return preferred, reason

    def _chat(self, system_prompt: str, user_content: str, temperature: float) -> str:
        url = f"{self.base_url}/v1/chat/completions"
        body = {
            "model": self.model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "tool_choice": "none",
            "response_format": {"type": "json_object"},
        }
        req = urllib.request.Request(
            url,
            method="POST",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            payload = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"judge HTTP {e.code}: {payload}") from e
        choices = data.get("choices", [])
        if not choices:
            return ""
        return str(choices[0].get("message", {}).get("content", "")).strip()

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            s = text.find("{")
            e = text.rfind("}")
            if s < 0 or e < 0 or e <= s:
                raise ValueError("no json object found in judge output")
            return json.loads(text[s : e + 1])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build planner DPO pairs from sampled candidates")
    p.add_argument(
        "--input",
        type=str,
        default="/root/autodl-tmp/muti-llm/Data/Processed_data/planner_dpo_candidates.jsonl",
    )
    p.add_argument(
        "--output",
        type=str,
        default="/root/autodl-tmp/muti-llm/Data/Processed_data/planner_dpo_pairs.jsonl",
    )
    p.add_argument("--max-pairs", type=int, default=2000)
    p.add_argument(
        "--min-hard-gap",
        type=float,
        default=0.08,
        help="If top and runner-up score gap < this threshold, optionally use judge model.",
    )
    p.add_argument("--require-kinds", type=str, default="", help="Optional comma list: plan,patch,subgraph")
    p.add_argument("--prefer-hard-reject-as-negative", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use-judge", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--timeout-seconds", type=int, default=90)
    p.add_argument("--base-url", type=str, default=os.getenv("DMX_BASE_URL", "https://www.dmxapi.cn/v1"))
    p.add_argument("--api-key", type=str, default=os.getenv("DMX_API_KEY", ""))
    p.add_argument("--model", type=str, default=os.getenv("DMX_MODEL", "glm-5.1-cc"))
    return p.parse_args()


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _group_by_context(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        cid = str(r.get("contextId", "")).strip()
        if not cid:
            continue
        grouped.setdefault(cid, []).append(r)
    return grouped


def _sort_candidates(cands: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        cands,
        key=lambda x: (
            bool(x.get("hardReject", False)),
            -float(x.get("hardScore", -1000.0)),
        ),
    )


def _pick_pair(
    cands: List[Dict[str, Any]],
    prefer_hard_reject_as_negative: bool,
) -> Optional[Tuple[Dict[str, Any], Dict[str, Any], float]]:
    if len(cands) < 2:
        return None
    ranked = _sort_candidates(cands)
    chosen = ranked[0]

    rejected: Optional[Dict[str, Any]] = None
    if prefer_hard_reject_as_negative:
        for c in reversed(ranked):
            if bool(c.get("hardReject", False)):
                rejected = c
                break
    if rejected is None:
        rejected = ranked[-1]

    chosen_score = float(chosen.get("hardScore", -1000.0))
    rejected_score = float(rejected.get("hardScore", -1000.0))
    return chosen, rejected, chosen_score - rejected_score


def _to_json_text(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def main() -> None:
    args = parse_args()
    in_path = Path(args.input).resolve()
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path = out_path.with_suffix(".stats.json")

    rows = _load_rows(in_path)
    grouped = _group_by_context(rows)
    allowed_kinds = {x.strip() for x in args.require_kinds.split(",") if x.strip()}

    judge: Optional[OpenAICompatJudge] = None
    if args.use_judge:
        if not args.api_key:
            raise ValueError("use-judge enabled but missing API key")
        judge = OpenAICompatJudge(args.base_url, args.api_key, args.model, args.timeout_seconds)

    written = 0
    skipped = 0
    judge_used = 0
    with out_path.open("w", encoding="utf-8") as out_f:
        for context_id, cands in grouped.items():
            if written >= args.max_pairs:
                break
            if len(cands) < 2:
                skipped += 1
                continue

            kind = str(cands[0].get("kind", ""))
            if allowed_kinds and kind not in allowed_kinds:
                skipped += 1
                continue

            picked = _pick_pair(cands, prefer_hard_reject_as_negative=args.prefer_hard_reject_as_negative)
            if picked is None:
                skipped += 1
                continue
            chosen, rejected, score_gap = picked

            judge_reason = ""
            use_judge_now = judge is not None and score_gap < args.min_hard_gap
            if use_judge_now:
                try:
                    preferred, judge_reason = judge.decide(
                        prompt_obj=chosen.get("prompt", {}),
                        cand_a=chosen.get("candidate", {}),
                        cand_b=rejected.get("candidate", {}),
                    )
                    judge_used += 1
                    if preferred == "B":
                        chosen, rejected = rejected, chosen
                except Exception as e:  # noqa: BLE001
                    judge_reason = f"judge_error:{e}"

            prompt_text = _to_json_text(chosen.get("prompt", {}))
            chosen_text = _to_json_text(chosen.get("candidate", {}))
            rejected_text = _to_json_text(rejected.get("candidate", {}))
            row = {
                "id": f"dpo:{context_id}:{written}",
                "prompt": prompt_text,
                "chosen": chosen_text,
                "rejected": rejected_text,
                "meta": {
                    "contextId": context_id,
                    "kind": kind,
                    "chosenCandidateId": chosen.get("id"),
                    "rejectedCandidateId": rejected.get("id"),
                    "chosenHardScore": chosen.get("hardScore"),
                    "rejectedHardScore": rejected.get("hardScore"),
                    "scoreGap": score_gap,
                    "judgeUsed": use_judge_now,
                    "judgeReason": judge_reason,
                },
            }
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    stats = {
        "input": str(in_path),
        "output": str(out_path),
        "contexts": len(grouped),
        "pairs": written,
        "skipped": skipped,
        "judgeUsedCount": judge_used,
        "judgeEnabled": args.use_judge,
    }
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
