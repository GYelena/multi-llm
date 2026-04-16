#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from orchestrator.planner import (
    OpenAIJsonPlanner,
    RulePlanner,
    parse_dag_plan,
    parse_reconstruct_patches,
    parse_subgraph_replacement,
)
from orchestrator.protocol import NodeState, NodeStatus, states_to_json


@dataclass
class BudgetMeter:
    daily_budget_cny: float
    price_input_cny_per_1m: float
    price_output_cny_per_1m: float
    spent_cny: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def add_usage(self, usage: Dict[str, Any]) -> None:
        p = int(usage.get("prompt_tokens", 0) or 0)
        c = int(usage.get("completion_tokens", 0) or 0)
        self.prompt_tokens += p
        self.completion_tokens += c
        self.spent_cny += (p / 1_000_000.0) * self.price_input_cny_per_1m
        self.spent_cny += (c / 1_000_000.0) * self.price_output_cny_per_1m

    def can_continue(self) -> bool:
        return self.spent_cny < self.daily_budget_cny


class OpenAICompatClient:
    def __init__(self, base_url: str, api_key: str, model: str, timeout_seconds: int) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds

    def chat_json(
        self,
        *,
        system_prompt: str,
        user_content: str,
        temperature: float,
        prefer_response_format_json: bool = True,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        body = {
            "model": self.model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "tool_choice": "none",
        }
        attempts: List[Dict[str, Any]] = []
        if prefer_response_format_json:
            attempts.append({**body, "response_format": {"type": "json_object"}})
        attempts.append(body)

        last_err: Exception | None = None
        for req_body in attempts:
            try:
                data = self._post_chat(req_body)
                text = self._extract_text(data)
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError:
                    parsed = self._extract_embedded_json(text)
                return parsed, data.get("usage", {})
            except Exception as e:  # noqa: BLE001
                last_err = e
                continue
        raise RuntimeError(f"chat_json failed: {last_err}")

    def _post_chat(self, body: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}/v1/chat/completions"
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
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            payload = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"HTTP {e.code}: {payload}") from e

    @staticmethod
    def _extract_text(data: Dict[str, Any]) -> str:
        choices = data.get("choices", [])
        if not choices:
            return ""
        msg = choices[0].get("message", {})
        return str(msg.get("content", "")).strip()

    @staticmethod
    def _extract_embedded_json(text: str) -> Dict[str, Any]:
        s = text.find("{")
        e = text.rfind("}")
        if s < 0 or e < 0 or e <= s:
            raise ValueError("no json object found in model content")
        return json.loads(text[s : e + 1])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate planner SFT seed dataset via teacher API")
    p.add_argument(
        "--input",
        type=str,
        default="/root/autodl-tmp/muti-llm/orchestrator/queries.math_finance_common.jsonl",
        help="Seed task JSONL with at least {id?, query, dag?}",
    )
    p.add_argument(
        "--output",
        type=str,
        default="/root/autodl-tmp/muti-llm/Data/Processed_data/planner_sft_seed.jsonl",
    )
    p.add_argument("--max-samples", type=int, default=300)
    p.add_argument("--mode", choices=["all", "plan", "patch", "subgraph"], default="all")
    p.add_argument("--max-patch-ops", type=int, default=2)
    p.add_argument("--temperature-plan", type=float, default=0.2)
    p.add_argument("--temperature-reconstruct", type=float, default=0.1)
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--sleep-seconds", type=float, default=0.0)
    p.add_argument("--timeout-seconds", type=int, default=90)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--base-url", type=str, default=os.getenv("DMX_BASE_URL", "https://www.dmxapi.cn/v1"))
    p.add_argument("--api-key", type=str, default=os.getenv("DMX_API_KEY", ""))
    p.add_argument("--model", type=str, default=os.getenv("DMX_MODEL", "glm-5.1-cc"))

    p.add_argument("--daily-budget-cny", type=float, default=50.0)
    p.add_argument("--price-input-cny-per-1m", type=float, default=6.32)
    p.add_argument("--price-output-cny-per-1m", type=float, default=22.12)
    return p.parse_args()


def load_seed_tasks(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            query = str(obj.get("query", "")).strip()
            if not query:
                continue
            rows.append(obj)
    if not rows:
        raise ValueError(f"no valid query rows in {path}")
    return rows


def choose_failed_root(dag_dict: Dict[str, Any]) -> str:
    nodes = dag_dict.get("nodes", [])
    if not nodes:
        return "T1"
    for n in nodes:
        deps = n.get("dependencies", [])
        if isinstance(deps, list) and len(deps) > 0:
            return str(n.get("nodeId", "T1"))
    return str(nodes[0].get("nodeId", "T1"))


def collect_descendants(dag_dict: Dict[str, Any], root: str) -> List[str]:
    children: Dict[str, List[str]] = {}
    for n in dag_dict.get("nodes", []):
        nid = str(n.get("nodeId", ""))
        for dep in n.get("dependencies", []) or []:
            children.setdefault(str(dep), []).append(nid)
    queue = list(children.get(root, []))
    seen = set(queue)
    out: List[str] = []
    while queue:
        cur = queue.pop(0)
        out.append(cur)
        for ch in children.get(cur, []):
            if ch in seen:
                continue
            seen.add(ch)
            queue.append(ch)
    return out


def build_reconstruct_context(
    query: str,
    dag_dict: Dict[str, Any],
    max_patch_ops: int,
) -> Tuple[Dict[str, Any], str, List[str]]:
    root = choose_failed_root(dag_dict)
    descendants = collect_descendants(dag_dict, root)
    failed_scope = [root] + [x for x in descendants if x != root]

    node_states: Dict[str, NodeState] = {}
    artifacts_summary: Dict[str, Dict[str, Any]] = {}
    failed_node_payloads: Dict[str, Dict[str, Any]] = {}
    for n in dag_dict.get("nodes", []):
        nid = str(n.get("nodeId", ""))
        if nid == root:
            st = NodeState(
                node_id=nid,
                status=NodeStatus.FAILED,
                confidence=0.0,
                risk_score=1.0,
                uncertainty=1.0,
                error_code="invalid_json",
            )
            failed_node_payloads[nid] = {
                "errorCode": "invalid_json",
                "payloadPreview": '{"error":"failed to parse structured output"}',
                "payloadKeys": ["error"],
            }
            artifacts_summary[nid] = {
                "keys": ["error"],
                "preview": '{"error":"failed to parse structured output"}',
            }
        elif nid in descendants:
            st = NodeState(
                node_id=nid,
                status=NodeStatus.PENDING,
                confidence=0.0,
                risk_score=0.4,
                uncertainty=0.8,
                error_code=None,
            )
            artifacts_summary[nid] = {
                "keys": [],
                "preview": "{}",
            }
        else:
            st = NodeState(
                node_id=nid,
                status=NodeStatus.DONE,
                confidence=0.85,
                risk_score=0.15,
                uncertainty=0.15,
                error_code=None,
            )
            artifacts_summary[nid] = {
                "keys": ["summary"],
                "preview": '{"summary":"upstream context available"}',
            }
        node_states[nid] = st

    states_json = states_to_json(node_states)
    payload = {
        "query": query,
        "maxPatchOps": max_patch_ops,
        "dag": dag_dict,
        "states": states_json,
        "artifactsSummary": artifacts_summary,
        "failedNodePayloads": failed_node_payloads,
    }
    return payload, root, failed_scope


def normalize_patches(patches: List[Any]) -> Dict[str, Any]:
    out: List[Dict[str, Any]] = []
    for p in patches:
        item: Dict[str, Any] = {
            "op": p.op.value,
            "targetNode": p.target_node,
            "reason": p.reason,
            "expectedGain": p.expected_gain,
            "costImpact": p.cost_impact,
        }
        if p.new_node is not None:
            item["newNode"] = p.new_node.to_dict()
        out.append(item)
    return {"patches": out}


def normalize_subgraph(plan: Any) -> Dict[str, Any]:
    return {
        "replacement": {
            "replaceRootNode": plan.replace_root_node,
            "removeNodeIds": plan.remove_node_ids,
            "newNodes": [n.to_dict() for n in plan.new_nodes],
            "bridgeDependencies": plan.bridge_dependencies,
            "reason": plan.reason,
            "expectedGain": plan.expected_gain,
            "costImpact": plan.cost_impact,
        }
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    if not args.api_key:
        raise ValueError("missing API key. Set --api-key or DMX_API_KEY")

    in_path = Path(args.input).resolve()
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path = out_path.with_suffix(".stats.json")

    tasks = load_seed_tasks(in_path)
    client = OpenAICompatClient(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        timeout_seconds=args.timeout_seconds,
    )
    meter = BudgetMeter(
        daily_budget_cny=args.daily_budget_cny,
        price_input_cny_per_1m=args.price_input_cny_per_1m,
        price_output_cny_per_1m=args.price_output_cny_per_1m,
    )

    generated = 0
    failures = 0
    by_kind: Dict[str, int] = {"plan": 0, "patch": 0, "subgraph": 0}
    kinds = ["plan", "patch", "subgraph"] if args.mode == "all" else [args.mode]

    rule_planner = RulePlanner()
    started_at = datetime.now().isoformat()
    with out_path.open("w", encoding="utf-8") as out_f:
        for idx in range(args.max_samples):
            if not meter.can_continue():
                print(f"[budget] stop early at spent={meter.spent_cny:.4f} CNY")
                break

            seed = tasks[idx % len(tasks)]
            query = str(seed["query"])
            sample_id = str(seed.get("id", f"seed_{idx:05d}"))
            kind = kinds[idx % len(kinds)]

            if isinstance(seed.get("dag"), dict):
                try:
                    dag = parse_dag_plan(seed["dag"])
                except Exception:  # noqa: BLE001
                    dag = rule_planner.plan(query)
            else:
                dag = rule_planner.plan(query)
            dag_dict = {"nodes": [n.to_dict() for n in dag.nodes]}

            ok = False
            for attempt in range(1, args.max_retries + 2):
                try:
                    if kind == "plan":
                        parsed, usage = client.chat_json(
                            system_prompt=OpenAIJsonPlanner._system_prompt(),
                            user_content=query,
                            temperature=args.temperature_plan,
                            prefer_response_format_json=True,
                        )
                        meter.add_usage(usage)
                        parsed_dag = parse_dag_plan(parsed)
                        output_json = {"nodes": [n.to_dict() for n in parsed_dag.nodes]}
                        instruction = "Generate a DAG planner JSON only."
                        model_input: Dict[str, Any] = {"query": query}
                    elif kind == "patch":
                        payload, _, _ = build_reconstruct_context(query=query, dag_dict=dag_dict, max_patch_ops=args.max_patch_ops)
                        parsed, usage = client.chat_json(
                            system_prompt=OpenAIJsonPlanner._reconstruct_system_prompt(),
                            user_content=json.dumps(payload, ensure_ascii=False),
                            temperature=args.temperature_reconstruct,
                            prefer_response_format_json=True,
                        )
                        meter.add_usage(usage)
                        patches = parse_reconstruct_patches(parsed, dag=dag, max_patch_ops=args.max_patch_ops)
                        if not patches:
                            raise ValueError("empty valid patches")
                        output_json = normalize_patches(patches)
                        instruction = "Generate reconstruct patch JSON only."
                        model_input = payload
                    else:
                        payload, root, failed_scope = build_reconstruct_context(
                            query=query,
                            dag_dict=dag_dict,
                            max_patch_ops=args.max_patch_ops,
                        )
                        selected = set(failed_scope)
                        payload["failedSubgraphNodes"] = [n for n in dag_dict["nodes"] if n["nodeId"] in selected]
                        payload["failedSubgraphStates"] = {k: v for k, v in payload["states"].items() if k in selected}
                        payload["failedSubgraphArtifacts"] = {
                            k: v for k, v in payload["artifactsSummary"].items() if k in selected
                        }
                        parsed, usage = client.chat_json(
                            system_prompt=OpenAIJsonPlanner._subgraph_reconstruct_system_prompt(),
                            user_content=json.dumps(payload, ensure_ascii=False),
                            temperature=args.temperature_reconstruct,
                            prefer_response_format_json=True,
                        )
                        meter.add_usage(usage)
                        plan = parse_subgraph_replacement(parsed, dag=dag, max_patch_ops=args.max_patch_ops)
                        if plan is None:
                            raise ValueError("invalid subgraph replacement")
                        if plan.replace_root_node != root:
                            raise ValueError("replaceRootNode mismatch with failed root")
                        output_json = normalize_subgraph(plan)
                        instruction = "Generate subgraph replacement JSON only."
                        model_input = payload

                    row = {
                        "id": f"{sample_id}:{kind}:{idx}",
                        "instruction": instruction,
                        "input": model_input,
                        "output": output_json,
                        "meta": {
                            "taskKind": kind,
                            "teacherModel": args.model,
                            "query": query,
                        },
                    }
                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out_f.flush()
                    generated += 1
                    by_kind[kind] += 1
                    ok = True
                    break
                except Exception as e:  # noqa: BLE001
                    if attempt > args.max_retries:
                        failures += 1
                        print(f"[warn] failed {sample_id}:{kind} after retries: {e}")
                    else:
                        time.sleep(0.3 * attempt)

            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

            if generated > 0 and generated % 20 == 0:
                print(
                    f"[progress] generated={generated} failures={failures} "
                    f"spent_cny={meter.spent_cny:.4f} prompt={meter.prompt_tokens} completion={meter.completion_tokens}"
                )

            if not ok and not meter.can_continue():
                break

    stats = {
        "startedAt": started_at,
        "finishedAt": datetime.now().isoformat(),
        "input": str(in_path),
        "output": str(out_path),
        "model": args.model,
        "baseUrl": args.base_url,
        "generated": generated,
        "failures": failures,
        "byKind": by_kind,
        "budget": {
            "dailyBudgetCny": args.daily_budget_cny,
            "spentCny": round(meter.spent_cny, 6),
            "promptTokens": meter.prompt_tokens,
            "completionTokens": meter.completion_tokens,
            "priceInputCnyPer1M": args.price_input_cny_per_1m,
            "priceOutputCnyPer1M": args.price_output_cny_per_1m,
        },
    }
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
