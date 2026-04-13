from __future__ import annotations

import argparse
import json
from pathlib import Path

from .healthcheck import run_healthcheck
from .stack import add_shared_orchestrator_args, apply_single_vllm_url, create_controller


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reactive multi-expert orchestrator CLI")
    parser.add_argument("--query", type=str, required=True, help="User query to process")
    add_shared_orchestrator_args(parser)
    parser.add_argument(
        "--healthcheck-before-run",
        action="store_true",
        help="Check planner/A/B/C endpoints before orchestrator run",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_single_vllm_url(args)

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

    controller = create_controller(args)
    result = controller.run(args.query)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
