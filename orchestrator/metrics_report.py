from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List


def _percentile(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    k = (len(sorted_vals) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_vals[int(k)])
    return float(sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f))


def _load_metrics(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {e}") from e
    return rows


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"count": 0, "error": "no rows"}

    ok = sum(1 for r in rows if r.get("successHeuristic") is True)
    wall = sorted(float(r.get("wallMs") or 0) for r in rows)
    dur = sorted(float(r.get("durationMs") or 0) for r in rows)
    recon = [float(r.get("reconstructRounds") or 0) for r in rows]
    patches = [float(r.get("patchesAppliedTotal") or 0) for r in rows]
    experts = [float(r.get("expertCallCount") or 0) for r in rows]
    steps = [float(r.get("controllerSteps") or 0) for r in rows]

    def mean(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "count": n,
        "successRate": ok / n,
        "successCount": ok,
        "wallMs": {
            "mean": mean(wall),
            "p50": _percentile(wall, 0.50),
            "p95": _percentile(wall, 0.95),
            "max": max(wall) if wall else 0.0,
        },
        "durationMs": {
            "mean": mean(dur),
            "p50": _percentile(dur, 0.50),
            "p95": _percentile(dur, 0.95),
            "max": max(dur) if dur else 0.0,
        },
        "reconstructRounds": {"mean": mean(recon), "max": max(recon) if recon else 0.0},
        "patchesAppliedTotal": {"mean": mean(patches), "max": max(patches) if patches else 0.0},
        "expertCallCount": {"mean": mean(experts), "max": max(experts) if experts else 0.0},
        "controllerSteps": {"mean": mean(steps), "max": max(steps) if steps else 0.0},
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize orchestrator benchmark metrics JSONL")
    p.add_argument("--input", type=str, required=True, help="Path to metrics JSONL from orchestrator.benchmark")
    p.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional path to write summary JSON (stdout if empty)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.input)
    rows = _load_metrics(path)
    summary = summarize(rows)
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    if summary.get("count") == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
