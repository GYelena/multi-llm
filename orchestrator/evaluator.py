from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .protocol import NodeState, NodeStatus


@dataclass
class ReconstructDecision:
    should_reconstruct: bool
    reason: str
    risk_score: float
    uncertainty: float


@dataclass
class EvalConfig:
    t_risk: float = 0.70
    t_uncertainty: float = 0.60
    cooldown_steps: int = 2
    max_reconstruct_times: int = 3


class StateEvaluator:
    """Computes runtime quality signals and reconstruction decisions."""

    def __init__(self, config: EvalConfig) -> None:
        self.config = config

    def aggregate_risk(self, states: Dict[str, NodeState]) -> float:
        if not states:
            return 0.0
        failed = sum(1 for s in states.values() if s.status == NodeStatus.FAILED)
        done = [s for s in states.values() if s.status == NodeStatus.DONE]
        avg_conf_done = sum(s.confidence for s in done) / len(done) if done else 0.0
        failure_ratio = failed / len(states)
        risk = 0.65 * failure_ratio + 0.35 * (1.0 - avg_conf_done)
        risk = max(0.0, min(1.0, risk))
        if failed > 0:
            # Any failure should be high-risk for reconstruction gating (avoid dilution by PENDING nodes).
            risk = max(risk, 0.76)
        return risk

    def aggregate_uncertainty(self, states: Dict[str, NodeState]) -> float:
        if not states:
            return 1.0
        avg = sum(s.uncertainty for s in states.values()) / len(states)
        return max(0.0, min(1.0, avg))

    def decide_reconstruct(
        self,
        states: Dict[str, NodeState],
        current_step: int,
        last_reconstruct_step: int,
        reconstruct_times: int,
    ) -> ReconstructDecision:
        risk = self.aggregate_risk(states)
        uncertainty = self.aggregate_uncertainty(states)

        if reconstruct_times >= self.config.max_reconstruct_times:
            return ReconstructDecision(False, "reconstruct budget exhausted", risk, uncertainty)

        if current_step - last_reconstruct_step < self.config.cooldown_steps:
            return ReconstructDecision(False, "in reconstruct cooldown", risk, uncertainty)

        if risk >= self.config.t_risk and uncertainty >= self.config.t_uncertainty:
            return ReconstructDecision(True, "high risk and high uncertainty", risk, uncertainty)

        return ReconstructDecision(False, "threshold not met", risk, uncertainty)
