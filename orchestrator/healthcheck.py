from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import List


@dataclass
class EndpointResult:
    name: str
    url: str
    ok: bool
    status: str
    detail: str


def check_chat_completions(
    endpoint_url: str,
    timeout_seconds: int = 4,
    probe_model: str | None = None,
) -> EndpointResult:
    endpoint_url = endpoint_url.rstrip("/")
    target = f"{endpoint_url}/v1/chat/completions"
    model = probe_model or os.getenv("HEALTHCHECK_MODEL", "deepseek-r1")
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
    }
    req = urllib.request.Request(
        target,
        method="POST",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer dummy",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            code = resp.getcode()
            return EndpointResult(
                name="",
                url=endpoint_url,
                ok=(200 <= code < 500),
                status=f"HTTP {code}",
                detail="reachable",
            )
    except urllib.error.HTTPError as e:
        # 404/405 usually means wrong service or wrong path (not OpenAI-compatible vLLM here).
        if e.code in (404, 405):
            return EndpointResult(
                name="",
                url=endpoint_url,
                ok=False,
                status=f"HTTP {e.code}",
                detail="path not found; expect POST /v1/chat/completions on this base URL (is vLLM running?)",
            )
        # Other 4xx often means auth/model validation, but endpoint exists.
        ok = 400 <= e.code < 500
        return EndpointResult(
            name="",
            url=endpoint_url,
            ok=ok,
            status=f"HTTP {e.code}",
            detail="reachable but request rejected (often OK: wrong model name in probe)",
        )
    except Exception as e:
        return EndpointResult(
            name="",
            url=endpoint_url,
            ok=False,
            status="ERROR",
            detail=str(e),
        )


def run_healthcheck(
    planner_url: str,
    a_url: str,
    b_url: str,
    c_url: str,
    skip_planner: bool = False,
    probe_model: str | None = None,
) -> List[EndpointResult]:
    checks = []
    if not skip_planner:
        checks.append(("planner", planner_url))
    checks.extend([("expert_a", a_url), ("expert_b", b_url), ("expert_c", c_url)])

    results: List[EndpointResult] = []
    for name, url in checks:
        r = check_chat_completions(url, probe_model=probe_model)
        r.name = name
        results.append(r)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check planner/expert OpenAI-compatible endpoints")
    parser.add_argument("--planner-url", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--a-url", type=str, default="http://127.0.0.1:8001")
    parser.add_argument("--b-url", type=str, default="http://127.0.0.1:8002")
    parser.add_argument("--c-url", type=str, default="http://127.0.0.1:8003")
    parser.add_argument("--skip-planner", action="store_true")
    parser.add_argument(
        "--probe-model",
        type=str,
        default=os.getenv("HEALTHCHECK_MODEL", "deepseek-r1"),
        help="Model name for minimal chat probe (must match --served-model-name on vLLM)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_healthcheck(
        planner_url=args.planner_url,
        a_url=args.a_url,
        b_url=args.b_url,
        c_url=args.c_url,
        skip_planner=args.skip_planner,
        probe_model=args.probe_model,
    )

    all_ok = True
    for r in results:
        flag = "OK" if r.ok else "FAIL"
        print(f"[{flag}] {r.name}: {r.url} -> {r.status} ({r.detail})")
        if not r.ok:
            all_ok = False

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
