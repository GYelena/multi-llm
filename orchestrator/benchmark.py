from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from .healthcheck import run_healthcheck
from .stack import add_shared_orchestrator_args, apply_single_vllm_url, create_controller


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch-run orchestrator and write metrics JSONL")
    p.add_argument(
        "--input",
        type=str,
        required=True,
        help="JSONL file: each line {\"id\": optional, \"query\": \"...\"}",
    )
    p.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output metrics JSONL path",
    )
    p.add_argument(
        "--healthcheck-before-run",
        action="store_true",
        help="Check planner/A/B/C endpoints once before the batch",
    )
    add_shared_orchestrator_args(p)
    return p.parse_args()


def _load_queries(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {e}") from e
            if "query" not in obj:
                raise ValueError(f"Missing 'query' at {path}:{line_no}")
            rows.append(obj)
    return rows


def _success_heuristic(states: Dict[str, Any]) -> bool:
    for st in states.values():
        if not isinstance(st, dict):
            return False
        if st.get("status") == "failed":
            return False
    return True


def main() -> None:
    args = parse_args()
    apply_single_vllm_url(args)
    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.healthcheck_before_run:
        health_results = run_healthcheck(
            planner_url=args.planner_base_url,
            a_url=args.a_base_url,
            b_url=args.b_base_url,
            c_url=args.c_base_url,
            skip_planner=(args.planner_backend != "openai"),
            probe_model=args.model_name,
        )
        failed = [r for r in health_results if not r.ok]
        for r in health_results:
            flag = "OK" if r.ok else "FAIL"
            print(f"[healthcheck:{flag}] {r.name} {r.url} -> {r.status}")
        if failed:
            raise SystemExit("healthcheck failed: at least one required endpoint is unavailable")

    queries = _load_queries(in_path)
    controller = create_controller(args)

    with out_path.open("w", encoding="utf-8") as out_f:
        for row in queries:
            qid = row.get("id", "")
            query = str(row["query"])
            t0 = time.perf_counter()
            result = controller.run(query)
            wall_ms = int((time.perf_counter() - t0) * 1000)

            metric: Dict[str, Any] = {
                "benchmarkId": qid,
                "query": query,
                "runId": result.get("runId"),
                "wallMs": wall_ms,
                "durationMs": result.get("durationMs"),
                "expertCallCount": result.get("expertCallCount"),
                "controllerSteps": result.get("controllerSteps"),
                "reconstructRounds": result.get("reconstructRounds"),
                "patchesAppliedTotal": result.get("patchesAppliedTotal"),
                "reconstructCostSum": result.get("reconstructCostSum"),
                "failedNodes": result.get("failedNodes", []),
                "successHeuristic": _success_heuristic(result.get("states", {})),
                "traceGlobalPath": result.get("traceGlobalPath"),
                "traceRunPath": result.get("traceRunPath"),
            }
            out_f.write(json.dumps(metric, ensure_ascii=False) + "\n")
            out_f.flush()
            print(json.dumps(metric, ensure_ascii=False))


if __name__ == "__main__":
    main()
