from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class TaskType(str, Enum):
    RETRIEVE = "retrieve"
    REASON = "reason"
    WRITE = "write"
    VERIFY = "verify"


class ExpertName(str, Enum):
    A = "A"
    B = "B"
    C = "C"


class NodeStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


class PatchOp(str, Enum):
    ADD = "add"
    REMOVE = "remove"
    MODIFY = "modify"


@dataclass
class Budget:
    max_tokens: int = 1024
    max_seconds: int = 30

    def validate(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError("Budget.max_tokens must be > 0")
        if self.max_seconds <= 0:
            raise ValueError("Budget.max_seconds must be > 0")


@dataclass
class TaskNode:
    node_id: str
    task_type: TaskType
    expert: ExpertName
    dependencies: List[str] = field(default_factory=list)
    input_refs: List[str] = field(default_factory=list)
    budget: Budget = field(default_factory=Budget)

    def validate(self) -> None:
        if not self.node_id.strip():
            raise ValueError("TaskNode.node_id must be non-empty")
        self.budget.validate()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodeId": self.node_id,
            "taskType": self.task_type.value,
            "expert": self.expert.value,
            "dependencies": self.dependencies,
            "inputRefs": self.input_refs,
            "budget": {
                "maxTokens": self.budget.max_tokens,
                "maxSeconds": self.budget.max_seconds,
            },
        }


@dataclass
class DagPlan:
    nodes: List[TaskNode]

    def rewrite_dependency_refs(self, old_id: str, new_id: str) -> None:
        """Replace references to old_id with new_id in all nodes (deps + input_refs)."""
        if old_id == new_id:
            return
        for node in self.nodes:
            node.dependencies = [new_id if d == old_id else d for d in node.dependencies]
            node.input_refs = [new_id if r == old_id else r for r in node.input_refs]

    def validate(self) -> None:
        if not self.nodes:
            raise ValueError("DagPlan must contain at least one node")
        ids = {n.node_id for n in self.nodes}
        if len(ids) != len(self.nodes):
            raise ValueError("DagPlan contains duplicate node_id values")
        for node in self.nodes:
            node.validate()
            for dep in node.dependencies:
                if dep not in ids:
                    raise ValueError(f"Unknown dependency '{dep}' in node '{node.node_id}'")


@dataclass
class ExpertRequest:
    run_id: str
    node: TaskNode
    query: str
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExpertResponse:
    node_id: str
    summary: str
    confidence: float
    payload: Dict[str, Any] = field(default_factory=dict)
    error_code: Optional[str] = None
    next_hint: Optional[str] = None

    def validate(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("ExpertResponse.confidence must be in [0,1]")


@dataclass
class NodeState:
    node_id: str
    status: NodeStatus = NodeStatus.PENDING
    confidence: float = 0.0
    risk_score: float = 0.0
    uncertainty: float = 1.0
    artifact_ref: str = ""
    error_code: Optional[str] = None

    def validate(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("NodeState.confidence must be in [0,1]")
        if not 0.0 <= self.risk_score <= 1.0:
            raise ValueError("NodeState.risk_score must be in [0,1]")
        if not 0.0 <= self.uncertainty <= 1.0:
            raise ValueError("NodeState.uncertainty must be in [0,1]")


@dataclass
class ReconstructPatch:
    op: PatchOp
    target_node: str
    reason: str
    expected_gain: float
    cost_impact: float
    new_node: Optional[TaskNode] = None


def states_to_json(states: Dict[str, NodeState]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for node_id, state in states.items():
        result[node_id] = {
            "status": state.status.value,
            "confidence": state.confidence,
            "riskScore": state.risk_score,
            "uncertainty": state.uncertainty,
            "artifactRef": state.artifact_ref,
            "errorCode": state.error_code,
        }
    return result
