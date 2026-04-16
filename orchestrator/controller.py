from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from .evaluator import EvalConfig, StateEvaluator
from .experts import ExpertRegistry
from .planner import BasePlanner, RulePlanner
from .protocol import (
    Budget,
    DagPlan,
    ExpertName,
    ExpertRequest,
    ExpertResponse,
    NodeState,
    NodeStatus,
    PatchOp,
    ReconstructPatch,
    SubgraphReplacementPlan,
    TaskNode,
    TaskType,
    states_to_json,
    validate_subgraph_replacement_against_dag,
)
from .trace import TraceLogger


@dataclass
class OrchestratorConfig:
    max_steps: int = 12
    t_risk: float = 0.70
    t_uncertainty: float = 0.60
    cooldown_steps: int = 2
    max_reconstruct_times: int = 3
    """Upper bound on RECONSTRUCT rounds (also capped by reconstruct_budget_ratio)."""
    max_patch_ops_per_round: int = 2
    """Max patch operations applied in one reconstruct step."""
    reconstruct_budget_ratio: float = 0.25
    """Fraction of max_steps used as additional cap on reconstruct count."""
    subgraph_planner_max_attempts: int = 2
    """Max planner attempts for one subgraph proposal before fallback."""


class OrchestratorController:
    """Reactive DAG controller for multi-expert execution."""

    def __init__(
        self,
        config: OrchestratorConfig,
        registry: ExpertRegistry,
        tracer: TraceLogger,
        planner: BasePlanner | None = None,
    ) -> None:
        self.config = config
        self.registry = registry
        self.tracer = tracer
        self.planner = planner or RulePlanner()
        budget_cap = max(1, int(config.max_steps * max(0.0, min(1.0, config.reconstruct_budget_ratio))))
        effective_max_reconstruct = min(config.max_reconstruct_times, budget_cap)
        self.evaluator = StateEvaluator(
            EvalConfig(
                t_risk=config.t_risk,
                t_uncertainty=config.t_uncertainty,
                cooldown_steps=config.cooldown_steps,
                max_reconstruct_times=effective_max_reconstruct,
            )
        )

    def run(self, query: str) -> Dict[str, Any]:
        t_wall0 = time.perf_counter()
        run_id = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S_%f")
        self.tracer.log_event(
            "run_start",
            {
                "runId": run_id,
                "query": query,
                "maxSteps": self.config.max_steps,
                "tRisk": self.config.t_risk,
                "tUncertainty": self.config.t_uncertainty,
                "maxReconstructTimes": self.config.max_reconstruct_times,
                "maxPatchOpsPerRound": self.config.max_patch_ops_per_round,
                "reconstructBudgetRatio": self.config.reconstruct_budget_ratio,
            },
        )

        dag = self.build_initial_plan(query, run_id)
        dag.validate()

        states = {node.node_id: NodeState(node_id=node.node_id) for node in dag.nodes}
        artifacts: Dict[str, Dict[str, Any]] = {}

        reconstruct_rounds = 0
        patches_applied_total = 0
        last_reconstruct_step = -999
        reconstruct_cost_sum = 0.0
        subgraph_reconstruct_count = 0
        expert_call_count = 0
        last_step = 0

        self.tracer.log_event(
            "controller_plan",
            {
                "runId": run_id,
                "query": query,
                "nodes": [n.to_dict() for n in dag.nodes],
                "effectiveMaxReconstruct": self.evaluator.config.max_reconstruct_times,
                "maxPatchOpsPerRound": self.config.max_patch_ops_per_round,
            },
        )

        for step in range(1, self.config.max_steps + 1):
            last_step = step
            ready = self._ready_nodes(dag, states)
            if not ready:
                break

            self.tracer.log_event(
                "controller_step",
                {
                    "runId": run_id,
                    "step": step,
                    "readyNodes": [n.node_id for n in ready],
                },
            )

            for node in ready:
                resp = self._dispatch_node(run_id, node, query, artifacts)
                expert_call_count += 1
                self._update_state(states[node.node_id], resp)
                artifacts[node.node_id] = resp.payload

            decision = self.evaluator.decide_reconstruct(
                states=states,
                current_step=step,
                last_reconstruct_step=last_reconstruct_step,
                reconstruct_times=reconstruct_rounds,
            )

            self.tracer.log_event(
                "controller_eval",
                {
                    "runId": run_id,
                    "step": step,
                    "riskScore": decision.risk_score,
                    "uncertainty": decision.uncertainty,
                    "reason": decision.reason,
                    "shouldReconstruct": decision.should_reconstruct,
                    "states": states_to_json(states),
                },
            )

            if decision.should_reconstruct:
                subgraph_plan = self._build_reconstruct_subgraph_plan(
                    query=query,
                    run_id=run_id,
                    step=step,
                    dag=dag,
                    states=states,
                    artifacts=artifacts,
                )
                if subgraph_plan is not None:
                    equivalent_ops = self._estimate_subgraph_equivalent_ops(subgraph_plan)
                    if equivalent_ops > self.config.max_patch_ops_per_round:
                        self.tracer.log_event(
                            "controller_subgraph_reconstruct_skipped",
                            {
                                "runId": run_id,
                                "step": step,
                                "reason": "equivalent op cap exceeded",
                                "equivalentOps": equivalent_ops,
                                "maxPatchOpsPerRound": self.config.max_patch_ops_per_round,
                            },
                        )
                    elif reconstruct_cost_sum + subgraph_plan.cost_impact > self._reconstruct_cost_cap():
                        self.tracer.log_event(
                            "controller_subgraph_reconstruct_skipped",
                            {
                                "runId": run_id,
                                "step": step,
                                "reason": "reconstruct cost cap exceeded",
                                "reconstructCostSum": reconstruct_cost_sum,
                                "planCost": subgraph_plan.cost_impact,
                            },
                        )
                    else:
                        self._apply_subgraph_replacement(dag=dag, states=states, plan=subgraph_plan)
                        reconstruct_cost_sum += subgraph_plan.cost_impact
                        reconstruct_rounds += 1
                        subgraph_reconstruct_count += 1
                        patches_applied_total += equivalent_ops
                        last_reconstruct_step = step
                        self.tracer.log_event(
                            "controller_subgraph_reconstruct",
                            {
                                "runId": run_id,
                                "step": step,
                                "replaceRootNode": subgraph_plan.replace_root_node,
                                "removeNodeIds": subgraph_plan.remove_node_ids,
                                "newNodes": [n.to_dict() for n in subgraph_plan.new_nodes],
                                "bridgeDependencies": subgraph_plan.bridge_dependencies,
                                "reason": subgraph_plan.reason,
                                "expectedGain": subgraph_plan.expected_gain,
                                "costImpact": subgraph_plan.cost_impact,
                                "equivalentOps": equivalent_ops,
                                "reconstructCostSum": reconstruct_cost_sum,
                            },
                        )
                        if self._all_done(dag, states):
                            break
                        continue

                patches = self._build_reconstruct_patches(
                    query=query,
                    run_id=run_id,
                    step=step,
                    dag=dag,
                    states=states,
                    artifacts=artifacts,
                )
                applied = 0
                for patch in patches[: self.config.max_patch_ops_per_round]:
                    if reconstruct_cost_sum + patch.cost_impact > self._reconstruct_cost_cap():
                        self.tracer.log_event(
                            "controller_reconstruct_skipped",
                            {
                                "runId": run_id,
                                "step": step,
                                "reason": "reconstruct cost cap exceeded",
                                "reconstructCostSum": reconstruct_cost_sum,
                                "patchCost": patch.cost_impact,
                            },
                        )
                        break
                    self._apply_patch(dag, states, patch)
                    reconstruct_cost_sum += patch.cost_impact
                    applied += 1
                    patches_applied_total += 1
                    self.tracer.log_event(
                        "controller_reconstruct",
                        {
                            "runId": run_id,
                            "step": step,
                            "patch": {
                                "op": patch.op.value,
                                "targetNode": patch.target_node,
                                "reason": patch.reason,
                                "expectedGain": patch.expected_gain,
                                "costImpact": patch.cost_impact,
                                "newNode": patch.new_node.to_dict() if patch.new_node else None,
                            },
                            "reconstructCostSum": reconstruct_cost_sum,
                        },
                    )
                if applied > 0:
                    reconstruct_rounds += 1
                    last_reconstruct_step = step
                if applied == 0 and patches:
                    self.tracer.log_event(
                        "controller_reconstruct_skipped",
                        {
                            "runId": run_id,
                            "step": step,
                            "reason": "no patch applied under caps",
                            "patchCount": len(patches),
                        },
                    )

            if self._all_done(dag, states):
                break

        final_answer = self._synthesize_final_answer(query, artifacts)
        duration_ms = int((time.perf_counter() - t_wall0) * 1000)
        failed_nodes = [nid for nid, st in states.items() if st.status == NodeStatus.FAILED]
        done_nodes = [nid for nid, st in states.items() if st.status == NodeStatus.DONE]

        self.tracer.log_event(
            "final_answer",
            {
                "runId": run_id,
                "answer": final_answer,
                "states": states_to_json(states),
                "reconstructRounds": reconstruct_rounds,
                "patchesAppliedTotal": patches_applied_total,
                "reconstructCostSum": reconstruct_cost_sum,
                "subgraphReconstructCount": subgraph_reconstruct_count,
                "durationMs": duration_ms,
                "expertCallCount": expert_call_count,
                "controllerSteps": last_step,
            },
        )

        run_trace_path: str | None = None
        if self.tracer.run_trace_dir is not None:
            run_trace_path = str((Path(self.tracer.run_trace_dir) / f"{run_id}.jsonl").resolve())

        self.tracer.log_event(
            "run_summary",
            {
                "runId": run_id,
                "query": query,
                "durationMs": duration_ms,
                "expertCallCount": expert_call_count,
                "controllerSteps": last_step,
                "nodeCount": len(states),
                "doneNodeCount": len(done_nodes),
                "failedNodeCount": len(failed_nodes),
                "failedNodes": failed_nodes,
                "reconstructRounds": reconstruct_rounds,
                "patchesAppliedTotal": patches_applied_total,
                "reconstructCostSum": reconstruct_cost_sum,
                "subgraphReconstructCount": subgraph_reconstruct_count,
                "traceGlobalPath": str(self.tracer.path.resolve()),
                "traceRunPath": run_trace_path,
            },
        )

        return {
            "runId": run_id,
            "finalAnswer": final_answer,
            "states": states_to_json(states),
            "reconstructRounds": reconstruct_rounds,
            "patchesAppliedTotal": patches_applied_total,
            "reconstructCostSum": reconstruct_cost_sum,
            "subgraphReconstructCount": subgraph_reconstruct_count,
            "durationMs": duration_ms,
            "expertCallCount": expert_call_count,
            "controllerSteps": last_step,
            "failedNodes": failed_nodes,
            "traceGlobalPath": str(self.tracer.path.resolve()),
            "traceRunPath": run_trace_path,
        }

    def _reconstruct_cost_cap(self) -> float:
        return float(max(1, int(self.config.max_steps * max(0.0, min(1.0, self.config.reconstruct_budget_ratio)))))

    def build_initial_plan(self, query: str, run_id: str) -> DagPlan:
        try:
            dag = self.planner.plan(query)
            dag.validate()
            return dag
        except Exception as e:
            # Fall back to deterministic rule planner when planner output is invalid/unavailable.
            self.tracer.log_event(
                "planner_fallback",
                {
                    "runId": run_id,
                    "reason": str(e),
                    "planner": self.planner.__class__.__name__,
                },
            )
            return RulePlanner().plan(query)

    def _dispatch_node(
        self,
        run_id: str,
        node: TaskNode,
        query: str,
        artifacts: Dict[str, Dict[str, Any]],
    ) -> ExpertResponse:
        context = {ref: artifacts.get(ref, {}) for ref in node.input_refs}
        request = ExpertRequest(run_id=run_id, node=node, query=query, context=context)
        adapter = self.registry.get(node.expert.value)

        self.tracer.log_event(
            "expert_call",
            {
                "runId": run_id,
                "nodeId": node.node_id,
                "expert": node.expert.value,
                "taskType": node.task_type.value,
                "inputRefs": node.input_refs,
            },
        )

        response = adapter.run(request)
        response.validate()

        self.tracer.log_event(
            "expert_result",
            {
                "runId": run_id,
                "nodeId": node.node_id,
                "summary": response.summary,
                "confidence": response.confidence,
                "errorCode": response.error_code,
            },
        )
        return response

    @staticmethod
    def _update_state(state: NodeState, response: ExpertResponse) -> None:
        if response.error_code:
            state.status = NodeStatus.FAILED
            state.confidence = 0.0
            state.risk_score = 1.0
            state.uncertainty = 1.0
            state.error_code = response.error_code
        else:
            state.status = NodeStatus.DONE
            state.confidence = response.confidence
            state.risk_score = max(0.0, 1.0 - response.confidence)
            state.uncertainty = max(0.0, 1.0 - response.confidence)
            state.artifact_ref = response.node_id
        state.validate()

    @staticmethod
    def _ready_nodes(dag: DagPlan, states: Dict[str, NodeState]) -> List[TaskNode]:
        ready: List[TaskNode] = []
        for node in dag.nodes:
            state = states[node.node_id]
            if state.status != NodeStatus.PENDING:
                continue
            if all(states[dep].status == NodeStatus.DONE for dep in node.dependencies):
                ready.append(node)
        return ready

    @staticmethod
    def _all_done(dag: DagPlan, states: Dict[str, NodeState]) -> bool:
        return all(states[n.node_id].status in (NodeStatus.DONE, NodeStatus.SKIPPED) for n in dag.nodes)

    def _build_reconstruct_patches(
        self,
        query: str,
        run_id: str,
        step: int,
        dag: DagPlan,
        states: Dict[str, NodeState],
        artifacts: Dict[str, Dict[str, Any]],
    ) -> List[ReconstructPatch]:
        planner_patches: List[ReconstructPatch] = []
        try:
            planner_patches = self.planner.propose_reconstruct_patches(
                query=query,
                dag=dag,
                states=states,
                max_patch_ops=self.config.max_patch_ops_per_round,
                artifacts_summary=self._build_artifacts_summary(artifacts),
                failed_node_payloads=self._build_failed_node_payloads(states, artifacts),
            )
        except Exception as e:
            self.tracer.log_event(
                "planner_reconstruct_fallback",
                {
                    "runId": run_id,
                    "step": step,
                    "reason": str(e),
                    "planner": self.planner.__class__.__name__,
                },
            )

        accepted = [p for p in planner_patches if self._patch_applicable(patch=p, dag=dag)]
        if accepted:
            self.tracer.log_event(
                "planner_reconstruct_proposal",
                {
                    "runId": run_id,
                    "step": step,
                    "source": "planner",
                    "patches": [
                        {
                            "op": p.op.value,
                            "targetNode": p.target_node,
                            "reason": p.reason,
                            "expectedGain": p.expected_gain,
                            "costImpact": p.cost_impact,
                            "newNode": p.new_node.to_dict() if p.new_node else None,
                        }
                        for p in accepted
                    ],
                },
            )
            return accepted

        fallback = self._build_rule_reconstruct_patches(dag=dag, states=states)
        if fallback:
            self.tracer.log_event(
                "planner_reconstruct_proposal",
                {
                    "runId": run_id,
                    "step": step,
                    "source": "rule_fallback",
                    "patches": [
                        {
                            "op": p.op.value,
                            "targetNode": p.target_node,
                            "reason": p.reason,
                            "expectedGain": p.expected_gain,
                            "costImpact": p.cost_impact,
                            "newNode": p.new_node.to_dict() if p.new_node else None,
                        }
                        for p in fallback
                    ],
                },
            )
        return fallback

    def _build_reconstruct_subgraph_plan(
        self,
        query: str,
        run_id: str,
        step: int,
        dag: DagPlan,
        states: Dict[str, NodeState],
        artifacts: Dict[str, Dict[str, Any]],
    ) -> SubgraphReplacementPlan | None:
        context = self._build_failed_subgraph_context(dag=dag, states=states, artifacts=artifacts)
        if not context["failedSubgraphNodeIds"]:
            return None
        max_attempts = max(1, int(self.config.subgraph_planner_max_attempts))
        for attempt in range(1, max_attempts + 1):
            try:
                plan = self.planner.propose_subgraph_replacement(
                    query=query,
                    dag=dag,
                    states=states,
                    max_patch_ops=self.config.max_patch_ops_per_round,
                    artifacts_summary=self._build_artifacts_summary(artifacts),
                    failed_node_payloads=self._build_failed_node_payloads(states, artifacts),
                    failed_subgraph_nodes=context["failedSubgraphNodes"],
                    failed_subgraph_states=context["failedSubgraphStates"],
                    failed_subgraph_artifacts=context["failedSubgraphArtifacts"],
                )
            except Exception as e:
                self.tracer.log_event(
                    "planner_subgraph_fallback",
                    {
                        "runId": run_id,
                        "step": step,
                        "reason": str(e),
                        "planner": self.planner.__class__.__name__,
                        "attempt": attempt,
                        "maxAttempts": max_attempts,
                        "willRetry": attempt < max_attempts,
                    },
                )
                continue

            if plan is None:
                self.tracer.log_event(
                    "planner_subgraph_fallback",
                    {
                        "runId": run_id,
                        "step": step,
                        "reason": "empty subgraph proposal",
                        "planner": self.planner.__class__.__name__,
                        "attempt": attempt,
                        "maxAttempts": max_attempts,
                        "willRetry": attempt < max_attempts,
                    },
                )
                continue

            try:
                self._validate_subgraph_replacement(
                    dag=dag, plan=plan, allowed_node_ids=set(context["failedSubgraphNodeIds"])
                )
            except Exception as e:
                self.tracer.log_event(
                    "planner_subgraph_fallback",
                    {
                        "runId": run_id,
                        "step": step,
                        "reason": f"invalid subgraph proposal: {e}",
                        "planner": self.planner.__class__.__name__,
                        "attempt": attempt,
                        "maxAttempts": max_attempts,
                        "willRetry": attempt < max_attempts,
                    },
                )
                continue

            self.tracer.log_event(
                "planner_subgraph_proposal",
                {
                    "runId": run_id,
                    "step": step,
                    "replaceRootNode": plan.replace_root_node,
                    "removeNodeIds": plan.remove_node_ids,
                    "newNodeCount": len(plan.new_nodes),
                    "bridgeDependencies": plan.bridge_dependencies,
                    "reason": plan.reason,
                    "expectedGain": plan.expected_gain,
                    "costImpact": plan.cost_impact,
                    "failedSubgraphNodeIds": context["failedSubgraphNodeIds"],
                    "attempt": attempt,
                    "maxAttempts": max_attempts,
                },
            )
            return plan
        return None

    @staticmethod
    def _build_artifacts_summary(artifacts: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        summary: Dict[str, Dict[str, Any]] = {}
        for node_id, payload in artifacts.items():
            if not isinstance(payload, dict):
                summary[node_id] = {"preview": OrchestratorController._preview_text(payload, head=300, tail=240)}
                continue
            keys = sorted(payload.keys())
            summary[node_id] = {
                "keys": keys,
                "preview": OrchestratorController._preview_text(payload, head=500, tail=300),
            }
        return summary

    @staticmethod
    def _build_failed_node_payloads(
        states: Dict[str, NodeState], artifacts: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        failed_payloads: Dict[str, Dict[str, Any]] = {}
        for node_id, state in states.items():
            if state.status != NodeStatus.FAILED:
                continue
            payload = artifacts.get(node_id, {})
            failed_payloads[node_id] = {
                "errorCode": state.error_code,
                "payloadPreview": OrchestratorController._preview_text(payload, head=1200, tail=600),
                "payloadKeys": sorted(payload.keys()) if isinstance(payload, dict) else [],
            }
        return failed_payloads

    @staticmethod
    def _preview_text(payload: Any, head: int, tail: int) -> str:
        text = str(payload)
        if len(text) <= head + tail + 20:
            return text
        return f"{text[:head]}\n...<truncated {len(text) - head - tail} chars>...\n{text[-tail:]}"

    @staticmethod
    def _collect_descendants(dag: DagPlan, root_node_id: str) -> List[str]:
        children: Dict[str, List[str]] = {}
        for node in dag.nodes:
            for dep in node.dependencies:
                children.setdefault(dep, []).append(node.node_id)
        result: List[str] = []
        queue = list(children.get(root_node_id, []))
        seen = set(queue)
        while queue:
            curr = queue.pop(0)
            result.append(curr)
            for child in children.get(curr, []):
                if child in seen:
                    continue
                seen.add(child)
                queue.append(child)
        return result

    def _build_failed_subgraph_context(
        self, dag: DagPlan, states: Dict[str, NodeState], artifacts: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        failed_nodes = [node for node in dag.nodes if states[node.node_id].status == NodeStatus.FAILED]
        if not failed_nodes:
            return {
                "failedSubgraphNodeIds": [],
                "failedSubgraphNodes": [],
                "failedSubgraphStates": {},
                "failedSubgraphArtifacts": {},
            }

        root = failed_nodes[0]
        descendant_ids = self._collect_descendants(dag, root.node_id)
        ordered_ids = [root.node_id] + [nid for nid in descendant_ids if nid != root.node_id]
        selected = set(ordered_ids)
        failed_subgraph_nodes = [node.to_dict() for node in dag.nodes if node.node_id in selected]
        failed_subgraph_states = {
            node_id: {
                "status": states[node_id].status.value,
                "confidence": states[node_id].confidence,
                "riskScore": states[node_id].risk_score,
                "uncertainty": states[node_id].uncertainty,
                "artifactRef": states[node_id].artifact_ref,
                "errorCode": states[node_id].error_code,
            }
            for node_id in ordered_ids
            if node_id in states
        }
        failed_subgraph_artifacts = {
            node_id: {
                "keys": sorted(artifacts.get(node_id, {}).keys()) if isinstance(artifacts.get(node_id), dict) else [],
                "preview": self._preview_text(artifacts.get(node_id, {}), head=600, tail=300),
            }
            for node_id in ordered_ids
        }
        return {
            "failedSubgraphNodeIds": ordered_ids,
            "failedSubgraphNodes": failed_subgraph_nodes,
            "failedSubgraphStates": failed_subgraph_states,
            "failedSubgraphArtifacts": failed_subgraph_artifacts,
        }

    @staticmethod
    def _estimate_subgraph_equivalent_ops(plan: SubgraphReplacementPlan) -> int:
        return max(1, len(plan.remove_node_ids) + len(plan.new_nodes))

    @staticmethod
    def _validate_subgraph_replacement(
        dag: DagPlan, plan: SubgraphReplacementPlan, allowed_node_ids: set[str]
    ) -> None:
        validate_subgraph_replacement_against_dag(plan, dag)
        remove_set = set(plan.remove_node_ids)
        if not remove_set.issubset(allowed_node_ids):
            raise ValueError("removeNodeIds must be limited to failed root + descendants")
        if plan.replace_root_node not in remove_set:
            raise ValueError("replaceRootNode must be included in removeNodeIds")

        retained_nodes = [node for node in dag.nodes if node.node_id not in remove_set]
        for node in retained_nodes:
            leaked = [dep for dep in node.dependencies if dep in remove_set]
            if leaked:
                raise ValueError(f"retained node '{node.node_id}' depends on removed node(s) {leaked}")

        retained_ids = {node.node_id for node in retained_nodes}
        new_ids = {node.node_id for node in plan.new_nodes}
        all_candidate_ids = retained_ids | new_ids

        for new_node in plan.new_nodes:
            merged_deps = list(new_node.dependencies)
            for dep in plan.bridge_dependencies.get(new_node.node_id, []):
                if dep not in merged_deps:
                    merged_deps.append(dep)
            for dep in merged_deps:
                if dep not in all_candidate_ids:
                    raise ValueError(f"new node '{new_node.node_id}' references unknown dependency '{dep}'")

        candidate_nodes = list(retained_nodes)
        for new_node in plan.new_nodes:
            extra_deps = plan.bridge_dependencies.get(new_node.node_id, [])
            deps = list(new_node.dependencies)
            refs = list(new_node.input_refs)
            for dep in extra_deps:
                if dep not in deps:
                    deps.append(dep)
                if dep not in refs:
                    refs.append(dep)
            candidate_nodes.append(
                TaskNode(
                    node_id=new_node.node_id,
                    task_type=new_node.task_type,
                    expert=new_node.expert,
                    dependencies=deps,
                    input_refs=refs,
                    budget=new_node.budget,
                )
            )
        DagPlan(nodes=candidate_nodes).validate()

    @staticmethod
    def _patch_applicable(patch: ReconstructPatch, dag: DagPlan) -> bool:
        existing_ids = {n.node_id for n in dag.nodes}
        if patch.op == PatchOp.ADD:
            return patch.new_node is not None and patch.new_node.node_id not in existing_ids
        if patch.op == PatchOp.MODIFY:
            return patch.new_node is not None and patch.target_node in existing_ids
        if patch.op == PatchOp.REMOVE:
            return patch.target_node in existing_ids
        return False

    def _build_rule_reconstruct_patches(self, dag: DagPlan, states: Dict[str, NodeState]) -> List[ReconstructPatch]:
        failed = [n for n in dag.nodes if states[n.node_id].status == NodeStatus.FAILED]
        if not failed:
            return []
        patches: List[ReconstructPatch] = []
        for target in failed[: self.config.max_patch_ops_per_round]:
            error_code = states[target.node_id].error_code or ""

            # 1) Prefer in-place MODIFY for richer reconstruct behavior.
            modify_patch = self._build_modify_patch(target, error_code)
            if modify_patch is not None:
                patches.append(modify_patch)
                continue

            # 2) For terminal failed retry leaf nodes, prune via REMOVE.
            if target.node_id.endswith("_retry") and self._is_leaf_node(dag, target.node_id):
                patches.append(
                    ReconstructPatch(
                        op=PatchOp.REMOVE,
                        target_node=target.node_id,
                        reason="remove terminal failed retry leaf node",
                        expected_gain=0.1,
                        cost_impact=0.05,
                    )
                )
                continue

            # 3) Fallback to ADD retry node.
            new_id = f"{target.node_id}_retry"
            if any(n.node_id == new_id for n in dag.nodes):
                continue
            new_node = TaskNode(
                node_id=new_id,
                task_type=target.task_type,
                expert=target.expert,
                dependencies=list(target.dependencies),
                input_refs=list(target.input_refs),
                budget=target.budget,
            )
            patches.append(
                ReconstructPatch(
                    op=PatchOp.ADD,
                    target_node=target.node_id,
                    reason="retry failed node with same expert",
                    expected_gain=0.4,
                    cost_impact=0.2,
                    new_node=new_node,
                )
            )
        return patches

    @staticmethod
    def _is_leaf_node(dag: DagPlan, node_id: str) -> bool:
        return all(node_id not in node.dependencies for node in dag.nodes if node.node_id != node_id)

    def _build_modify_patch(self, target: TaskNode, error_code: str) -> ReconstructPatch | None:
        lowered = error_code.lower()
        transient_errors = {
            "mock_b_failure",
            "network_timeout",
            "rate_limited",
            "upstream_server_error",
            "call_exception",
            "network_error",
            "connection_refused",
            "dns_resolution_failed",
            "repair_network_timeout",
            "repair_rate_limited",
            "repair_upstream_server_error",
            "repair_call_exception",
        }
        schema_errors = {
            "invalid_json",
            "invalid_schema",
            "repair_invalid_json",
            "repair_invalid_schema",
        }

        next_expert = target.expert
        next_budget = Budget(max_tokens=target.budget.max_tokens, max_seconds=target.budget.max_seconds)
        changed = False
        reason_parts: List[str] = []

        if lowered in transient_errors and (next_budget.max_tokens < 4096 or next_budget.max_seconds < 120):
            next_budget.max_tokens = min(4096, max(next_budget.max_tokens + 256, int(next_budget.max_tokens * 1.5)))
            next_budget.max_seconds = min(120, max(next_budget.max_seconds + 10, int(next_budget.max_seconds * 1.5)))
            changed = True
            reason_parts.append("increase node budget for transient runtime error")

        if lowered in schema_errors and target.task_type in (TaskType.REASON, TaskType.VERIFY):
            if target.expert != ExpertName.C:
                next_expert = ExpertName.C
                changed = True
                reason_parts.append("switch to schema-stable expert for structured output")

        if not changed:
            return None

        modified = TaskNode(
            node_id=target.node_id,
            task_type=target.task_type,
            expert=next_expert,
            dependencies=list(target.dependencies),
            input_refs=list(target.input_refs),
            budget=next_budget,
        )
        return ReconstructPatch(
            op=PatchOp.MODIFY,
            target_node=target.node_id,
            reason="; ".join(reason_parts),
            expected_gain=0.45,
            cost_impact=0.12,
            new_node=modified,
        )

    @staticmethod
    def _apply_patch(dag: DagPlan, states: Dict[str, NodeState], patch: ReconstructPatch) -> None:
        if patch.op == PatchOp.ADD and patch.new_node is not None:
            existing_ids = {n.node_id for n in dag.nodes}
            if patch.new_node.node_id in existing_ids:
                return
            old_id = patch.target_node
            if old_id in states:
                states[old_id].status = NodeStatus.SKIPPED
                states[old_id].error_code = states[old_id].error_code or "superseded_by_retry"
            dag.rewrite_dependency_refs(old_id, patch.new_node.node_id)
            dag.nodes.append(patch.new_node)
            states[patch.new_node.node_id] = NodeState(node_id=patch.new_node.node_id)
            dag.validate()
            return

        if patch.op == PatchOp.MODIFY and patch.new_node is not None:
            idx = next((i for i, node in enumerate(dag.nodes) if node.node_id == patch.target_node), -1)
            if idx < 0:
                return
            original = dag.nodes[idx]
            replacement = patch.new_node
            dag.nodes[idx] = replacement
            if original.node_id != replacement.node_id:
                dag.rewrite_dependency_refs(original.node_id, replacement.node_id)
                state = states.pop(original.node_id, None)
                states[replacement.node_id] = state or NodeState(node_id=replacement.node_id)
            state = states.get(replacement.node_id)
            if state is None:
                states[replacement.node_id] = NodeState(node_id=replacement.node_id)
            else:
                state.status = NodeStatus.PENDING
                state.confidence = 0.0
                state.risk_score = 0.0
                state.uncertainty = 1.0
                state.artifact_ref = ""
                state.error_code = None
            dag.validate()
            return

        if patch.op == PatchOp.REMOVE:
            target = next((node for node in dag.nodes if node.node_id == patch.target_node), None)
            if target is None:
                return
            for node in dag.nodes:
                if node.node_id == target.node_id:
                    continue
                if target.node_id in node.dependencies:
                    new_deps: List[str] = []
                    for dep in node.dependencies:
                        if dep == target.node_id:
                            for inherited in target.dependencies:
                                if inherited not in new_deps:
                                    new_deps.append(inherited)
                        elif dep not in new_deps:
                            new_deps.append(dep)
                    node.dependencies = new_deps
                node.input_refs = [ref for ref in node.input_refs if ref != target.node_id]
            dag.nodes = [node for node in dag.nodes if node.node_id != target.node_id]
            states.pop(target.node_id, None)
            if not dag.nodes:
                return
            dag.validate()

    @staticmethod
    def _apply_subgraph_replacement(
        dag: DagPlan, states: Dict[str, NodeState], plan: SubgraphReplacementPlan
    ) -> None:
        remove_set = set(plan.remove_node_ids)
        retained_nodes = [node for node in dag.nodes if node.node_id not in remove_set]
        for node in retained_nodes:
            if any(dep in remove_set for dep in node.dependencies):
                raise ValueError(f"cannot remove nodes still required by retained node '{node.node_id}'")

        for node_id in remove_set:
            if node_id in states:
                states[node_id].status = NodeStatus.SKIPPED
                states[node_id].artifact_ref = ""
                states[node_id].error_code = states[node_id].error_code or "subgraph_replaced"

        inserted_nodes: List[TaskNode] = []
        for node in plan.new_nodes:
            deps = list(node.dependencies)
            refs = list(node.input_refs)
            for dep in plan.bridge_dependencies.get(node.node_id, []):
                if dep not in deps:
                    deps.append(dep)
                if dep not in refs:
                    refs.append(dep)
            inserted_nodes.append(
                TaskNode(
                    node_id=node.node_id,
                    task_type=node.task_type,
                    expert=node.expert,
                    dependencies=deps,
                    input_refs=refs,
                    budget=node.budget,
                )
            )
            states[node.node_id] = NodeState(node_id=node.node_id)

        dag.nodes = retained_nodes + inserted_nodes
        dag.validate()

    @staticmethod
    def _synthesize_final_answer(query: str, artifacts: Dict[str, Dict[str, Any]]) -> str:
        if not artifacts:
            return f"No expert output produced for query: {query}"
        ordered = sorted(artifacts.items(), key=lambda x: x[0])
        chunks = [f"[{node_id}] {payload}" for node_id, payload in ordered]
        return "Final synthesized response based on expert artifacts\n" + "\n".join(chunks)
