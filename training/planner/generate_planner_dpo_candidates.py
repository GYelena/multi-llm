#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
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


class OpenAICompatClient:
    def __init__(self, base_url: str, api_key: str, model: str, timeout_seconds: int) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds

    def chat_text(
        self,
        *,
        system_prompt: str,
        user_content: str,
        temperature: float,
        prefer_response_format_json: bool = True,
    ) -> Tuple[str, Dict[str, Any]]:
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
                return text, data.get("usage", {})
            except Exception as e:  # noqa: BLE001
                last_err = e
                continue
        raise RuntimeError(f"chat_text failed: {last_err}")

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
    def extract_json(text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            s = text.find("{")
            e = text.rfind("}")
            if s < 0 or e < 0 or e <= s:
                raise ValueError("no json object found in content")
            return json.loads(text[s : e + 1])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sample planner candidates for DPO pairing")
    p.add_argument(
        "--input",
        type=str,
        default="/root/autodl-tmp/muti-llm/orchestrator/queries.math_finance_common.jsonl",
        help="JSONL rows with query (and optional dag).",
    )
    p.add_argument(
        "--output",
        type=str,
        default="/root/autodl-tmp/muti-llm/Data/Processed_data/planner_dpo_candidates.jsonl",
    )
    p.add_argument("--kind", choices=["all", "plan", "patch", "subgraph"], default="all")
    p.add_argument("--samples-per-query", type=int, default=3)
    p.add_argument("--max-queries", type=int, default=200)
    p.add_argument("--max-patch-ops", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--timeout-seconds", type=int, default=90)
    p.add_argument("--sleep-seconds", type=float, default=0.0)
    p.add_argument(
        "--temperature-list",
        type=str,
        default="0.0,0.1,0.2",
        help="Comma-separated temperatures sampled per query.",
    )

    p.add_argument("--base-url", type=str, default=os.getenv("DMX_BASE_URL", "https://www.dmxapi.cn/v1"))
    p.add_argument("--api-key", type=str, default=os.getenv("DMX_API_KEY", ""))
    p.add_argument("--model", type=str, default=os.getenv("DMX_MODEL", "glm-5.1-cc"))
    return p.parse_args()


def load_queries(path: Path, max_queries: int) -> List[Dict[str, Any]]:
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
            if len(rows) >= max_queries:
                break
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


def build_reconstruct_context(query: str, dag_dict: Dict[str, Any], max_patch_ops: int) -> Tuple[Dict[str, Any], str, List[str]]:
    root = choose_failed_root(dag_dict)
    descendants = collect_descendants(dag_dict, root)
    failed_scope = [root] + [x for x in descendants if x != root]

    states: Dict[str, NodeState] = {}
    artifacts_summary: Dict[str, Dict[str, Any]] = {}
    failed_node_payloads: Dict[str, Dict[str, Any]] = {}
    for n in dag_dict.get("nodes", []):
        nid = str(n.get("nodeId", ""))
        if nid == root:
            states[nid] = NodeState(
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
            artifacts_summary[nid] = {"keys": ["error"], "preview": '{"error":"failed to parse structured output"}'}
        elif nid in descendants:
            states[nid] = NodeState(
                node_id=nid,
                status=NodeStatus.PENDING,
                confidence=0.0,
                risk_score=0.5,
                uncertainty=0.8,
                error_code=None,
            )
            artifacts_summary[nid] = {"keys": [], "preview": "{}"}
        else:
            states[nid] = NodeState(
                node_id=nid,
                status=NodeStatus.DONE,
                confidence=0.85,
                risk_score=0.15,
                uncertainty=0.15,
                error_code=None,
            )
            artifacts_summary[nid] = {"keys": ["summary"], "preview": '{"summary":"upstream ok"}'}

    payload = {
        "query": query,
        "maxPatchOps": max_patch_ops,
        "dag": dag_dict,
        "states": states_to_json(states),
        "artifactsSummary": artifacts_summary,
        "failedNodePayloads": failed_node_payloads,
    }
    return payload, root, failed_scope


def make_context_id(query: str, kind: str) -> str:
    h = hashlib.sha1(f"{kind}::{query}".encode("utf-8")).hexdigest()[:16]
    return f"{kind}:{h}"


def hard_score_for_plan(plan_nodes: int, has_write: bool, has_reason: bool) -> float:
    score = 1.0
    if has_write:
        score += 0.2
    if has_reason:
        score += 0.1
    score -= 0.03 * float(max(0, plan_nodes - 2))
    return score


def hard_score_for_patch(patch_count: int, cost_sum: float, has_modify_or_add: bool) -> float:
    score = 1.0
    if has_modify_or_add:
        score += 0.15
    score -= 0.4 * float(cost_sum)
    score -= 0.03 * float(max(0, patch_count - 1))
    return score


def hard_score_for_subgraph(node_count: int, remove_count: int, cost: float) -> float:
    score = 1.2
    score -= 0.35 * float(cost)
    score -= 0.02 * float(max(0, node_count + remove_count - 3))
    return score


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    if not args.api_key:
        raise ValueError("missing API key. Set --api-key or DMX_API_KEY")

    in_path = Path(args.input).resolve()
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_queries(in_path, args.max_queries)
    if not rows:
        raise ValueError(f"no valid query rows in {in_path}")

    temps = [float(x.strip()) for x in args.temperature_list.split(",") if x.strip()]
    if not temps:
        temps = [0.1]

    kinds = ["plan", "patch", "subgraph"] if args.kind == "all" else [args.kind]
    client = OpenAICompatClient(args.base_url, args.api_key, args.model, args.timeout_seconds)
    rule_planner = RulePlanner()

    written = 0
    invalid = 0
    with out_path.open("w", encoding="utf-8") as out_f:
        for i, row in enumerate(rows):
            query = str(row["query"])
            qid = str(row.get("id", f"q{i:05d}"))
            if isinstance(row.get("dag"), dict):
                try:
                    dag = parse_dag_plan(row["dag"])
                except Exception:  # noqa: BLE001
                    dag = rule_planner.plan(query)
            else:
                dag = rule_planner.plan(query)
            dag_dict = {"nodes": [n.to_dict() for n in dag.nodes]}

            for kind in kinds:
                context_id = make_context_id(query, kind)
                for sidx in range(args.samples_per_query):
                    temp = temps[sidx % len(temps)]
                    if kind == "plan":
                        system_prompt = OpenAIJsonPlanner._system_prompt()
                        user_content = query
                        prompt_obj: Dict[str, Any] = {"query": query, "kind": "plan"}
                    elif kind == "patch":
                        payload, _, _ = build_reconstruct_context(query, dag_dict, args.max_patch_ops)
                        system_prompt = OpenAIJsonPlanner._reconstruct_system_prompt()
                        user_content = json.dumps(payload, ensure_ascii=False)
                        prompt_obj = payload
                    else:
                        payload, _, failed_scope = build_reconstruct_context(query, dag_dict, args.max_patch_ops)
                        selected = set(failed_scope)
                        payload["failedSubgraphNodes"] = [n for n in dag_dict["nodes"] if n["nodeId"] in selected]
                        payload["failedSubgraphStates"] = {k: v for k, v in payload["states"].items() if k in selected}
                        payload["failedSubgraphArtifacts"] = {
                            k: v for k, v in payload["artifactsSummary"].items() if k in selected
                        }
                        system_prompt = OpenAIJsonPlanner._subgraph_reconstruct_system_prompt()
                        user_content = json.dumps(payload, ensure_ascii=False)
                        prompt_obj = payload

                    text = ""
                    usage: Dict[str, Any] = {}
                    hard_reject = False
                    hard_score = -1000.0
                    hard_signals: Dict[str, Any] = {
                        "jsonValid": False,
                        "schemaValid": False,
                    }
                    candidate_obj: Dict[str, Any]
                    try:
                        text, usage = client.chat_text(
                            system_prompt=system_prompt,
                            user_content=user_content,
                            temperature=temp,
                            prefer_response_format_json=True,
                        )
                        parsed = client.extract_json(text)
                        hard_signals["jsonValid"] = True

                        if kind == "plan":
                            parsed_plan = parse_dag_plan(parsed)
                            nodes = [n.to_dict() for n in parsed_plan.nodes]
                            has_write = any(n["taskType"] == "write" for n in nodes)
                            has_reason = any(n["taskType"] == "reason" for n in nodes)
                            hard_score = hard_score_for_plan(len(nodes), has_write=has_write, has_reason=has_reason)
                            hard_signals.update(
                                {
                                    "schemaValid": True,
                                    "nodeCount": len(nodes),
                                    "hasWriteNode": has_write,
                                    "hasReasonNode": has_reason,
                                }
                            )
                            candidate_obj = {"nodes": nodes}
                        elif kind == "patch":
                            patches = parse_reconstruct_patches(parsed, dag=dag, max_patch_ops=args.max_patch_ops)
                            if not patches:
                                raise ValueError("empty valid patches")
                            cost_sum = sum(float(p.cost_impact) for p in patches)
                            has_modify_or_add = any(p.op.value in ("modify", "add") for p in patches)
                            hard_score = hard_score_for_patch(
                                patch_count=len(patches),
                                cost_sum=cost_sum,
                                has_modify_or_add=has_modify_or_add,
                            )
                            hard_signals.update(
                                {
                                    "schemaValid": True,
                                    "patchCount": len(patches),
                                    "costSum": cost_sum,
                                    "hasModifyOrAdd": has_modify_or_add,
                                }
                            )
                            candidate_obj = {
                                "patches": [
                                    {
                                        "op": p.op.value,
                                        "targetNode": p.target_node,
                                        "reason": p.reason,
                                        "expectedGain": p.expected_gain,
                                        "costImpact": p.cost_impact,
                                        "newNode": p.new_node.to_dict() if p.new_node else None,
                                    }
                                    for p in patches
                                ]
                            }
                        else:
                            plan = parse_subgraph_replacement(parsed, dag=dag, max_patch_ops=args.max_patch_ops)
                            if plan is None:
                                raise ValueError("invalid subgraph replacement")
                            hard_score = hard_score_for_subgraph(
                                node_count=len(plan.new_nodes),
                                remove_count=len(plan.remove_node_ids),
                                cost=float(plan.cost_impact),
                            )
                            hard_signals.update(
                                {
                                    "schemaValid": True,
                                    "newNodeCount": len(plan.new_nodes),
                                    "removeNodeCount": len(plan.remove_node_ids),
                                    "costImpact": plan.cost_impact,
                                }
                            )
                            candidate_obj = {
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
                    except Exception as e:  # noqa: BLE001
                        hard_reject = True
                        invalid += 1
                        candidate_obj = {"rawText": text[:5000], "error": str(e)}
                        hard_signals["error"] = str(e)

                    out_row = {
                        "id": f"{qid}:{kind}:{sidx}",
                        "contextId": context_id,
                        "kind": kind,
                        "prompt": prompt_obj,
                        "candidate": candidate_obj,
                        "hardReject": hard_reject,
                        "hardScore": hard_score,
                        "hardSignals": hard_signals,
                        "meta": {
                            "query": query,
                            "temperature": temp,
                            "model": args.model,
                            "usage": usage,
                        },
                    }
                    out_f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                    written += 1

                    if args.sleep_seconds > 0:
                        time.sleep(args.sleep_seconds)

            if (i + 1) % 20 == 0:
                print(f"[progress] queries={i+1} candidates={written} invalid={invalid}")

    print(
        json.dumps(
            {
                "input": str(in_path),
                "output": str(out_path),
                "queries": len(rows),
                "candidates": written,
                "invalid": invalid,
                "model": args.model,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
