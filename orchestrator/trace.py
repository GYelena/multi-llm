from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class TraceLogger:
    """Append-only JSONL trace. Optionally mirrors each runId to a dedicated file."""

    path: Path
    run_trace_dir: Optional[Path] = None

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.run_trace_dir is not None:
            self.run_trace_dir.mkdir(parents=True, exist_ok=True)

    def log_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "eventType": event_type,
            "payload": payload,
        }
        line = json.dumps(row, ensure_ascii=False) + "\n"
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line)
            f.flush()

        run_id = payload.get("runId")
        if self.run_trace_dir and isinstance(run_id, str) and run_id.strip():
            per_path = self.run_trace_dir / f"{run_id.strip()}.jsonl"
            with per_path.open("a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
