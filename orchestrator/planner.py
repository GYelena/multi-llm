from __future__ import annotations

import json
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List

from .protocol import (
    Budget,
    DagPlan,
    ExpertName,
    NodeState,
    PatchOp,
    ReconstructPatch,
    SubgraphReplacementPlan,
    TaskNode,
    TaskType,
    states_to_json,
    validate_subgraph_replacement_against_dag,
)


class BasePlanner(ABC):
    """Planner interface that outputs a validated DAG plan."""

    @abstractmethod
    def plan(self, query: str) -> DagPlan:
        raise NotImplementedError

    def propose_reconstruct_patches(
        self,
        query: str,
        dag: DagPlan,
        states: Dict[str, NodeState],
        max_patch_ops: int,
        artifacts_summary: Dict[str, Dict[str, Any]],
        failed_node_payloads: Dict[str, Dict[str, Any]],
    ) -> List[ReconstructPatch]:
        return []

    def propose_subgraph_replacement(
        self,
        query: str,
        dag: DagPlan,
        states: Dict[str, NodeState],
        max_patch_ops: int,
        artifacts_summary: Dict[str, Dict[str, Any]],
        failed_node_payloads: Dict[str, Dict[str, Any]],
        failed_subgraph_nodes: List[Dict[str, Any]],
        failed_subgraph_states: Dict[str, Dict[str, Any]],
        failed_subgraph_artifacts: Dict[str, Dict[str, Any]],
    ) -> SubgraphReplacementPlan | None:
        return None


@dataclass
class RulePlanner(BasePlanner):
    """Heuristic fallback planner."""

    def plan(self, query: str) -> DagPlan:
        lower = query.lower()
        needs_fact = any(k in lower for k in ["who", "when", "where", "fact", "source"])

        nodes: List[TaskNode] = []
        if needs_fact:
            nodes.append(TaskNode(node_id="T1", task_type=TaskType.RETRIEVE, expert=ExpertName.A))
            nodes.append(
                TaskNode(
                    node_id="T2",
                    task_type=TaskType.REASON,
                    expert=ExpertName.B,
                    dependencies=["T1"],
                    input_refs=["T1"],
                )
            )
            nodes.append(
                TaskNode(
                    node_id="T3",
                    task_type=TaskType.WRITE,
                    expert=ExpertName.C,
                    dependencies=["T2"],
                    input_refs=["T1", "T2"],
                )
            )
        else:
            nodes.append(TaskNode(node_id="T1", task_type=TaskType.REASON, expert=ExpertName.B))
            nodes.append(
                TaskNode(
                    node_id="T2",
                    task_type=TaskType.WRITE,
                    expert=ExpertName.C,
                    dependencies=["T1"],
                    input_refs=["T1"],
                )
            )

        dag = DagPlan(nodes=nodes)
        dag.validate()
        return dag


@dataclass
class MockJsonPlanner(BasePlanner):
    """Returns synthetic JSON-like planning behavior for local testing."""

    def plan(self, query: str) -> DagPlan:
        lower = query.lower()
        if "verify" in lower:
            data = {
                "nodes": [
                    {
                        "nodeId": "T1",
                        "taskType": "reason",
                        "expert": "B",
                        "dependencies": [],
                        "inputRefs": [],
                        "budget": {"maxTokens": 1200, "maxSeconds": 45},
                    },
                    {
                        "nodeId": "T2",
                        "taskType": "verify",
                        "expert": "B",
                        "dependencies": ["T1"],
                        "inputRefs": ["T1"],
                        "budget": {"maxTokens": 800, "maxSeconds": 30},
                    },
                    {
                        "nodeId": "T3",
                        "taskType": "write",
                        "expert": "C",
                        "dependencies": ["T2"],
                        "inputRefs": ["T1", "T2"],
                        "budget": {"maxTokens": 900, "maxSeconds": 30},
                    },
                ]
            }
        else:
            data = {
                "nodes": [
                    {
                        "nodeId": "T1",
                        "taskType": "retrieve",
                        "expert": "A",
                        "dependencies": [],
                        "inputRefs": [],
                        "budget": {"maxTokens": 900, "maxSeconds": 30},
                    },
                    {
                        "nodeId": "T2",
                        "taskType": "reason",
                        "expert": "B",
                        "dependencies": ["T1"],
                        "inputRefs": ["T1"],
                        "budget": {"maxTokens": 1100, "maxSeconds": 45},
                    },
                    {
                        "nodeId": "T3",
                        "taskType": "write",
                        "expert": "C",
                        "dependencies": ["T2"],
                        "inputRefs": ["T1", "T2"],
                        "budget": {"maxTokens": 900, "maxSeconds": 30},
                    },
                ]
            }
        return parse_dag_plan(data)


@dataclass
class OpenAIJsonPlanner(BasePlanner):
    """Uses an OpenAI-compatible endpoint to generate DAG JSON."""

    base_url: str
    model: str
    api_key: str = "dummy"
    timeout_seconds: int = 60
    prefer_response_format_json: bool = True

    def plan(self, query: str) -> DagPlan:
        text = self._chat_once(
            system_prompt=self._system_prompt(),
            user_content=query,
            temperature=0.2,
            expect_json=True,
        )
        if not text:
            raise RuntimeError("planner returned empty content")

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = self._try_extract_embedded_json(text)
        return parse_dag_plan(parsed)

    def propose_reconstruct_patches(
        self,
        query: str,
        dag: DagPlan,
        states: Dict[str, NodeState],
        max_patch_ops: int,
        artifacts_summary: Dict[str, Dict[str, Any]],
        failed_node_payloads: Dict[str, Dict[str, Any]],
    ) -> List[ReconstructPatch]:
        base_payload = {
            "query": query,
            "maxPatchOps": max_patch_ops,
            "dag": {"nodes": [n.to_dict() for n in dag.nodes]},
            "states": states_to_json(states),
            "artifactsSummary": artifacts_summary,
            "failedNodePayloads": failed_node_payloads,
        }
        planner_context_summary = ""
        try:
            planner_context_summary = self._summarize_reconstruct_context(base_payload)
        except Exception:
            planner_context_summary = ""
        payload = {
            **base_payload,
            "plannerContextSummary": planner_context_summary,
        }
        text = self._chat_once(
            system_prompt=self._reconstruct_system_prompt(),
            user_content=json.dumps(payload, ensure_ascii=False),
            temperature=0.1,
            expect_json=True,
        )
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = self._try_extract_embedded_json(text)
        return parse_reconstruct_patches(parsed, dag=dag, max_patch_ops=max_patch_ops)

    def propose_subgraph_replacement(
        self,
        query: str,
        dag: DagPlan,
        states: Dict[str, NodeState],
        max_patch_ops: int,
        artifacts_summary: Dict[str, Dict[str, Any]],
        failed_node_payloads: Dict[str, Dict[str, Any]],
        failed_subgraph_nodes: List[Dict[str, Any]],
        failed_subgraph_states: Dict[str, Dict[str, Any]],
        failed_subgraph_artifacts: Dict[str, Dict[str, Any]],
    ) -> SubgraphReplacementPlan | None:
        base_payload = {
            "query": query,
            "maxPatchOps": max_patch_ops,
            "dag": {"nodes": [n.to_dict() for n in dag.nodes]},
            "states": states_to_json(states),
            "artifactsSummary": artifacts_summary,
            "failedNodePayloads": failed_node_payloads,
            "failedSubgraphNodes": failed_subgraph_nodes,
            "failedSubgraphStates": failed_subgraph_states,
            "failedSubgraphArtifacts": failed_subgraph_artifacts,
        }
        planner_context_summary = ""
        try:
            planner_context_summary = self._summarize_reconstruct_context(base_payload)
        except Exception:
            planner_context_summary = ""
        payload = {
            **base_payload,
            "plannerContextSummary": planner_context_summary,
        }
        text = self._chat_once(
            system_prompt=self._subgraph_reconstruct_system_prompt(),
            user_content=json.dumps(payload, ensure_ascii=False),
            temperature=0.1,
            expect_json=True,
        )
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = self._try_extract_embedded_json(text)
        return parse_subgraph_replacement(parsed, dag=dag, max_patch_ops=max_patch_ops)

    def _chat_once(self, system_prompt: str, user_content: str, temperature: float, expect_json: bool = False) -> str:
        url = f"{self.base_url.rstrip('/')}/v1/chat/completions"
        base_body = {
            "model": self.model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            # Explicitly avoid tool-call paths for planner robustness.
            "tool_choice": "none",
        }
        attempts: List[Dict[str, Any]] = []
        if expect_json and self.prefer_response_format_json:
            attempts.append(
                {
                    **base_body,
                    "response_format": {"type": "json_object"},
                }
            )
        attempts.append(base_body)

        last_error: Exception | None = None
        for i, body in enumerate(attempts):
            req = urllib.request.Request(
                url,
                method="POST",
                data=json.dumps(body).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                return self._extract_text(data)
            except urllib.error.HTTPError as e:
                last_error = e
                # Some backends reject response_format/tool_choice in chat payload.
                # If the first attempt fails, fall back to plain payload once.
                if i < len(attempts) - 1:
                    continue
                raise RuntimeError(f"planner HTTP error: {e.code}") from e
            except Exception as e:
                last_error = e
                if i < len(attempts) - 1:
                    continue
                raise RuntimeError("planner request failed") from e

        if last_error is not None:
            raise RuntimeError("planner request failed") from last_error
        raise RuntimeError("planner request failed")

    def _summarize_reconstruct_context(self, payload: Dict[str, Any]) -> str:
        text = self._chat_once(
            system_prompt=self._reconstruct_summary_prompt(),
            user_content=json.dumps(payload, ensure_ascii=False),
            temperature=0.0,
            expect_json=False,
        )
        return text.strip()[:2000]

    @staticmethod
    def _extract_text(data: Dict[str, Any]) -> str:
        choices = data.get("choices", [])
        if not choices:
            return ""
        msg = choices[0].get("message", {})
        return str(msg.get("content", "")).strip()

    @staticmethod
    def _try_extract_embedded_json(text: str) -> Dict[str, Any]:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < 0 or end <= start:
            raise ValueError("planner output does not contain JSON object")
        return json.loads(text[start : end + 1])

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You are a DAG planner. Return ONLY valid JSON with schema: "
            '{"nodes":[{"nodeId":"T1","taskType":"retrieve|reason|write|verify",'
            '"expert":"A|B|C","dependencies":["..."],"inputRefs":["..."],'
            '"budget":{"maxTokens":int,"maxSeconds":int}}]}. '
            "Do not include markdown or explanations. Keep dependencies acyclic."
        )

    @staticmethod
    def _reconstruct_system_prompt() -> str:
        return (
            "You are a reconstruct planner for a DAG executor. "
            "Return ONLY valid JSON with schema: "
            '{"patches":[{"op":"add|remove|modify","targetNode":"T1","reason":"...","expectedGain":0.0,'
            '"costImpact":0.0,"newNode":{"nodeId":"T1_r1","taskType":"retrieve|reason|write|verify",'
            '"expert":"A|B|C","dependencies":["..."],"inputRefs":["..."],'
            '"budget":{"maxTokens":int,"maxSeconds":int}}}]}. '
            "Rules: for remove, omit newNode. for modify/add, include valid newNode. "
            "Respect maxPatchOps and keep dependencies acyclic. "
            "Use plannerContextSummary plus states/artifactsSummary/failedNodePayloads for content-aware patching."
        )

    @staticmethod
    def _reconstruct_summary_prompt() -> str:
        return (
            "You are a context summarizer for reconstruct planning. "
            "Given DAG nodes, node states, artifacts summary and failed-node payload snippets, "
            "write a concise operational summary for a planner model. "
            "Output plain text only in <= 12 lines focusing on failure causes, likely fixes, and dependency impact."
        )

    @staticmethod
    def _subgraph_reconstruct_system_prompt() -> str:
        return (
            "You are a subgraph reconstruct planner for a DAG executor. "
            "Return ONLY valid JSON with schema: "
            '{"replacement":{"replaceRootNode":"T2","removeNodeIds":["T2","T3"],"newNodes":['
            '{"nodeId":"T2_new","taskType":"retrieve|reason|write|verify","expert":"A|B|C",'
            '"dependencies":["..."],"inputRefs":["..."],"budget":{"maxTokens":int,"maxSeconds":int}}],'
            '"bridgeDependencies":{"T2_new":["T1"]},"reason":"...","expectedGain":0.0,"costImpact":0.0}}. '
            "Rules: removeNodeIds should normally cover failed root and descendants, newNodes must be acyclic, "
            "all dependencies must reference either retained nodes or newNodes, and plan must respect maxPatchOps "
            "by minimizing equivalent operations."
        )


def parse_dag_plan(data: Dict[str, Any]) -> DagPlan:
    if "nodes" not in data or not isinstance(data["nodes"], list):
        raise ValueError("planner JSON must contain a list field 'nodes'")

    nodes: List[TaskNode] = []
    for raw in data["nodes"]:
        nodes.append(_parse_task_node(raw))

    dag = DagPlan(nodes=nodes)
    dag.validate()
    return dag


def parse_reconstruct_patches(data: Dict[str, Any], dag: DagPlan, max_patch_ops: int) -> List[ReconstructPatch]:
    raw_patches = data.get("patches", [])
    if not isinstance(raw_patches, list):
        raise ValueError("reconstruct JSON must contain list field 'patches'")

    existing_ids = {n.node_id for n in dag.nodes}
    patches: List[ReconstructPatch] = []
    for raw in raw_patches:
        if len(patches) >= max_patch_ops:
            break
        if not isinstance(raw, dict):
            continue
        try:
            op = PatchOp(str(raw.get("op", "")).strip())
        except Exception:
            continue
        target = str(raw.get("targetNode", "")).strip()
        if not target:
            continue
        if op in (PatchOp.REMOVE, PatchOp.MODIFY) and target not in existing_ids:
            continue
        reason = str(raw.get("reason", "")).strip() or "planner suggested reconstruct patch"
        expected_gain = _to_float(raw.get("expectedGain"), default=0.3)
        cost_impact = _to_float(raw.get("costImpact"), default=0.15)

        new_node = None
        if op in (PatchOp.ADD, PatchOp.MODIFY):
            raw_node = raw.get("newNode")
            if not isinstance(raw_node, dict):
                continue
            try:
                new_node = _parse_task_node(raw_node, fallback_node_id=target)
            except Exception:
                continue
            if op == PatchOp.ADD and new_node.node_id in existing_ids:
                continue

        patches.append(
            ReconstructPatch(
                op=op,
                target_node=target,
                reason=reason,
                expected_gain=expected_gain,
                cost_impact=cost_impact,
                new_node=new_node,
            )
        )
    return patches


def parse_subgraph_replacement(
    data: Dict[str, Any], dag: DagPlan, max_patch_ops: int
) -> SubgraphReplacementPlan | None:
    raw = data.get("replacement")
    if not isinstance(raw, dict):
        return None

    replace_root_node = str(raw.get("replaceRootNode", "")).strip()
    remove_node_ids = [str(x).strip() for x in raw.get("removeNodeIds", []) if str(x).strip()]
    raw_new_nodes = raw.get("newNodes", [])
    if not isinstance(raw_new_nodes, list):
        return None
    if not replace_root_node or not remove_node_ids or not raw_new_nodes:
        return None

    new_nodes: List[TaskNode] = []
    for item in raw_new_nodes:
        if not isinstance(item, dict):
            return None
        try:
            new_nodes.append(_parse_task_node(item))
        except Exception:
            return None

    bridge_raw = raw.get("bridgeDependencies", {}) or {}
    bridge_dependencies: Dict[str, List[str]] = {}
    if not isinstance(bridge_raw, dict):
        return None
    for nid, deps in bridge_raw.items():
        if not isinstance(deps, list):
            return None
        clean_deps = [str(d).strip() for d in deps if str(d).strip()]
        bridge_dependencies[str(nid).strip()] = clean_deps

    plan = SubgraphReplacementPlan(
        replace_root_node=replace_root_node,
        remove_node_ids=remove_node_ids,
        new_nodes=new_nodes,
        bridge_dependencies=bridge_dependencies,
        reason=str(raw.get("reason", "")).strip() or "planner suggested subgraph replacement",
        expected_gain=_to_float(raw.get("expectedGain"), default=0.5),
        cost_impact=_to_float(raw.get("costImpact"), default=0.2),
    )
    validate_subgraph_replacement_against_dag(plan, dag)

    equivalent_ops = max(1, len(plan.remove_node_ids) + len(plan.new_nodes))
    if equivalent_ops > max_patch_ops:
        return None
    return plan


def _parse_task_node(raw: Dict[str, Any], fallback_node_id: str = "") -> TaskNode:
    node_id = str(raw.get("nodeId", "")).strip() or fallback_node_id
    task_type = TaskType(str(raw.get("taskType", "")).strip())
    expert = ExpertName(str(raw.get("expert", "")).strip())
    dependencies = [str(x) for x in raw.get("dependencies", [])]
    input_refs = [str(x) for x in raw.get("inputRefs", [])]
    budget_raw = raw.get("budget", {}) or {}
    budget = Budget(
        max_tokens=int(budget_raw.get("maxTokens", 1024)),
        max_seconds=int(budget_raw.get("maxSeconds", 30)),
    )
    return TaskNode(
        node_id=node_id,
        task_type=task_type,
        expert=expert,
        dependencies=dependencies,
        input_refs=input_refs,
        budget=budget,
    )


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default
