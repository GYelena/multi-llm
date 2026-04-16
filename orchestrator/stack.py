from __future__ import annotations

import argparse
import os
from pathlib import Path

from .controller import OrchestratorConfig, OrchestratorController
from .experts import build_mock_registry, build_openai_registry
from .planner import MockJsonPlanner, OpenAIJsonPlanner, RulePlanner
from .trace import TraceLogger


DEFAULT_PLANNER_BASE_MODEL = "/root/autodl-tmp/muti-llm/models"


def add_shared_orchestrator_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-steps", type=int, default=12, help="Maximum orchestration steps")
    parser.add_argument(
        "--trace-path",
        type=str,
        default="/root/autodl-tmp/muti-llm/Data/Processed_data/orchestrator_trace.jsonl",
        help="Global JSONL trace output path",
    )
    parser.add_argument(
        "--run-trace-dir",
        type=str,
        default="",
        help="Directory for per-run trace files (default: <trace-dir>/orchestrator_runs/)",
    )
    parser.add_argument(
        "--disable-per-run-trace",
        action="store_true",
        help="Do not write per-run trace files",
    )
    parser.add_argument(
        "--backend",
        choices=["mock", "openai"],
        default="openai",
        help="Expert backend type",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="deepseek-r1",
        help="Fallback model identifier for OpenAI-compatible expert backends",
    )
    parser.add_argument(
        "--a-model-name",
        type=str,
        default=os.getenv("EXPERT_A_MODEL_NAME", ""),
        help="Model identifier for expert A backend (fallbacks to --model-name)",
    )
    parser.add_argument(
        "--b-model-name",
        type=str,
        default=os.getenv("EXPERT_B_MODEL_NAME", ""),
        help="Model identifier for expert B backend (fallbacks to --model-name)",
    )
    parser.add_argument(
        "--c-model-name",
        type=str,
        default=os.getenv("EXPERT_C_MODEL_NAME", ""),
        help="Model identifier for expert C backend (fallbacks to --model-name)",
    )
    parser.add_argument("--a-base-url", type=str, default=os.getenv("EXPERT_A_BASE_URL", "http://127.0.0.1:8001"))
    parser.add_argument("--b-base-url", type=str, default=os.getenv("EXPERT_B_BASE_URL", "http://127.0.0.1:8002"))
    parser.add_argument("--c-base-url", type=str, default=os.getenv("EXPERT_C_BASE_URL", "http://127.0.0.1:8003"))
    parser.add_argument(
        "--a-backend",
        choices=["openai", "local_retriever"],
        default=os.getenv("EXPERT_A_BACKEND", "openai"),
        help="Expert A backend mode",
    )
    parser.add_argument(
        "--a-retriever-checkpoint",
        type=str,
        default=os.getenv("EXPERT_A_RETRIEVER_CHECKPOINT", "/root/autodl-tmp/muti-llm/outputs/retriever_A/checkpoint-164000"),
        help="Local retriever checkpoint path for expert A",
    )
    parser.add_argument(
        "--a-retriever-base-model",
        type=str,
        default=os.getenv("EXPERT_A_RETRIEVER_BASE_MODEL", "/root/autodl-tmp/muti-llm/models/bge-base-en-v1.5"),
        help="Local retriever base model path for expert A",
    )
    parser.add_argument(
        "--a-retriever-index-dir",
        type=str,
        default=os.getenv("EXPERT_A_RETRIEVER_INDEX_DIR", "/root/autodl-tmp/muti-llm/outputs/retriever_A/faiss_index"),
        help="Local retriever FAISS directory for expert A",
    )
    parser.add_argument(
        "--a-retriever-top-k",
        type=int,
        default=int(os.getenv("EXPERT_A_RETRIEVER_TOP_K", "5")),
        help="Top-k retrieval size for local retriever expert A",
    )
    parser.add_argument("--api-key", type=str, default=os.getenv("OPENAI_API_KEY", "dummy"))
    parser.add_argument(
        "--expert-timeout-seconds",
        type=int,
        default=int(os.getenv("ORCHESTRATOR_EXPERT_TIMEOUT_SECONDS", "60")),
        help="HTTP read timeout for expert A/B/C OpenAI-compatible calls (raise for slow R1)",
    )
    parser.add_argument(
        "--planner-backend",
        choices=["rule", "mock", "openai"],
        default="openai",
        help="Planner backend for initial DAG generation",
    )
    parser.add_argument(
        "--planner-base-url",
        type=str,
        default=os.getenv("CENTRAL_PLANNER_BASE_URL", "http://127.0.0.1:8000"),
        help="OpenAI-compatible planner endpoint",
    )
    parser.add_argument(
        "--planner-model-name",
        type=str,
        default=os.getenv("CENTRAL_PLANNER_MODEL", DEFAULT_PLANNER_BASE_MODEL),
        help="Planner model name when planner-backend=openai",
    )
    parser.add_argument("--t-risk", type=float, default=0.70, help="Reconstruct risk threshold")
    parser.add_argument("--t-uncertainty", type=float, default=0.60, help="Reconstruct uncertainty threshold")
    parser.add_argument("--max-reconstruct-times", type=int, default=3, help="Max reconstruct rounds")
    parser.add_argument("--max-patch-ops", type=int, default=2, help="Max patch ops per reconstruct step")
    parser.add_argument(
        "--reconstruct-budget-ratio",
        type=float,
        default=0.25,
        help="Reconstruct budget ratio relative to max-steps",
    )
    parser.add_argument(
        "--single-vllm-url",
        type=str,
        default=os.getenv("SINGLE_VLLM_URL", ""),
        help="Scheme A: set planner-base-url and a/b/c-base-url to this same OpenAI-compatible base",
    )


def apply_single_vllm_url(args: argparse.Namespace) -> None:
    """If --single-vllm-url is set, point planner and all experts to one vLLM server."""
    raw = getattr(args, "single_vllm_url", None)
    if raw is None:
        return
    url = str(raw).strip().rstrip("/")
    if not url:
        return
    args.planner_base_url = url
    args.a_base_url = url
    args.b_base_url = url
    args.c_base_url = url


def create_tracer(args: argparse.Namespace) -> TraceLogger:
    run_trace_dir: Path | None = None
    if not getattr(args, "disable_per_run_trace", False):
        if getattr(args, "run_trace_dir", ""):
            run_trace_dir = Path(str(args.run_trace_dir))
        else:
            run_trace_dir = Path(args.trace_path).resolve().parent / "orchestrator_runs"
    return TraceLogger(Path(args.trace_path), run_trace_dir=run_trace_dir)


def create_controller(args: argparse.Namespace) -> OrchestratorController:
    if args.backend == "mock":
        registry = build_mock_registry()
    else:
        registry = build_openai_registry(
            model_name=args.model_name,
            a_base_url=args.a_base_url,
            b_base_url=args.b_base_url,
            c_base_url=args.c_base_url,
            a_backend=args.a_backend,
            a_retriever_checkpoint=args.a_retriever_checkpoint,
            a_retriever_base_model=args.a_retriever_base_model,
            a_retriever_index_dir=args.a_retriever_index_dir,
            a_retriever_top_k=args.a_retriever_top_k,
            a_model_name=args.a_model_name,
            b_model_name=args.b_model_name,
            c_model_name=args.c_model_name,
            api_key=args.api_key,
            timeout_seconds=max(1, int(args.expert_timeout_seconds)),
        )

    if args.planner_backend == "rule":
        planner = RulePlanner()
    elif args.planner_backend == "mock":
        planner = MockJsonPlanner()
    else:
        planner = OpenAIJsonPlanner(
            base_url=args.planner_base_url,
            model=args.planner_model_name,
            api_key=args.api_key,
        )

    tracer = create_tracer(args)
    config = OrchestratorConfig(
        max_steps=args.max_steps,
        t_risk=args.t_risk,
        t_uncertainty=args.t_uncertainty,
        max_reconstruct_times=args.max_reconstruct_times,
        max_patch_ops_per_round=args.max_patch_ops,
        reconstruct_budget_ratio=args.reconstruct_budget_ratio,
    )
    return OrchestratorController(config=config, registry=registry, tracer=tracer, planner=planner)
