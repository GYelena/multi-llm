from __future__ import annotations

import json
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List

from .protocol import Budget, DagPlan, ExpertName, TaskNode, TaskType


class BasePlanner(ABC):
    """Planner interface that outputs a validated DAG plan."""

    @abstractmethod
    def plan(self, query: str) -> DagPlan:
        raise NotImplementedError


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

    def plan(self, query: str) -> DagPlan:
        url = f"{self.base_url.rstrip('/')}/v1/chat/completions"
        body = {
            "model": self.model,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": query},
            ],
        }
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
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"planner HTTP error: {e.code}") from e
        except Exception as e:
            raise RuntimeError("planner request failed") from e

        text = self._extract_text(data)
        if not text:
            raise RuntimeError("planner returned empty content")

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = self._try_extract_embedded_json(text)
        return parse_dag_plan(parsed)

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


def parse_dag_plan(data: Dict[str, Any]) -> DagPlan:
    if "nodes" not in data or not isinstance(data["nodes"], list):
        raise ValueError("planner JSON must contain a list field 'nodes'")

    nodes: List[TaskNode] = []
    for raw in data["nodes"]:
        node_id = str(raw.get("nodeId", "")).strip()
        task_type = TaskType(str(raw.get("taskType", "")).strip())
        expert = ExpertName(str(raw.get("expert", "")).strip())
        dependencies = [str(x) for x in raw.get("dependencies", [])]
        input_refs = [str(x) for x in raw.get("inputRefs", [])]
        budget_raw = raw.get("budget", {}) or {}
        budget = Budget(
            max_tokens=int(budget_raw.get("maxTokens", 1024)),
            max_seconds=int(budget_raw.get("maxSeconds", 30)),
        )
        nodes.append(
            TaskNode(
                node_id=node_id,
                task_type=task_type,
                expert=expert,
                dependencies=dependencies,
                input_refs=input_refs,
                budget=budget,
            )
        )

    dag = DagPlan(nodes=nodes)
    dag.validate()
    return dag
