from __future__ import annotations

import json
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

from .protocol import ExpertRequest, ExpertResponse


class BaseExpertAdapter(ABC):
    """Common expert adapter interface."""

    @abstractmethod
    def run(self, request: ExpertRequest) -> ExpertResponse:
        raise NotImplementedError


@dataclass
class MockExpertAdapter(BaseExpertAdapter):
    expert_name: str
    _fail_b_once: bool = False

    def run(self, request: ExpertRequest) -> ExpertResponse:
        query = request.query.lower()
        if (
            self.expert_name == "B"
            and "__fail_b__" in query
            and not request.node.node_id.endswith("_retry")
            and not self._fail_b_once
        ):
            self._fail_b_once = True
            return ExpertResponse(
                node_id=request.node.node_id,
                summary="mock failure for reconstruct demo",
                confidence=0.0,
                payload={},
                error_code="mock_b_failure",
            )
        if self.expert_name == "A":
            payload = {
                "claims": [f"Retrieved factual hints for: {request.query}"],
                "sourceRefs": ["mock://source/1"],
                "citationConfidence": 0.75,
            }
            return ExpertResponse(
                node_id=request.node.node_id,
                summary="fact retrieval complete",
                confidence=0.75,
                payload=payload,
            )
        if self.expert_name == "B":
            payload = {
                "reasoningSteps": [
                    "Parse problem",
                    "Identify constraints",
                    "Produce answer candidate",
                ],
                "verifications": ["mock-check:passed"],
                "checkResult": "passed",
            }
            base_conf = 0.72
            if "hard" in query or "prove" in query:
                base_conf = 0.62
            return ExpertResponse(
                node_id=request.node.node_id,
                summary="reasoning complete",
                confidence=base_conf,
                payload=payload,
            )

        payload = {
            "draft": "Concise final response synthesized from available evidence.",
            "fidelityReport": "All statements grounded in context artifacts.",
            "unsupportedStatements": [],
        }
        return ExpertResponse(
            node_id=request.node.node_id,
            summary="writing complete",
            confidence=0.80,
            payload=payload,
        )


@dataclass
class OpenAIExpertAdapter(BaseExpertAdapter):
    expert_name: str
    base_url: str
    model: str
    system_prompt: str
    timeout_seconds: int = 60
    api_key: str = "dummy"

    def run(self, request: ExpertRequest) -> ExpertResponse:
        url = f"{self.base_url.rstrip('/')}/v1/chat/completions"
        user_content = self._build_user_content(request)
        body = {
            "model": self.model,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_content},
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
            return ExpertResponse(
                node_id=request.node.node_id,
                summary="remote expert call failed",
                confidence=0.0,
                payload={},
                error_code=f"http_{e.code}",
            )
        except Exception:
            return ExpertResponse(
                node_id=request.node.node_id,
                summary="remote expert call exception",
                confidence=0.0,
                payload={},
                error_code="call_exception",
            )

        text = self._extract_text(data)
        confidence = 0.70 if text else 0.0
        return ExpertResponse(
            node_id=request.node.node_id,
            summary=f"expert {self.expert_name} completed",
            confidence=confidence,
            payload={"text": text, "raw": data},
            error_code=None if text else "empty_response",
        )

    def _build_user_content(self, request: ExpertRequest) -> str:
        return json.dumps(
            {
                "nodeId": request.node.node_id,
                "taskType": request.node.task_type.value,
                "query": request.query,
                "context": request.context,
            },
            ensure_ascii=False,
        )

    @staticmethod
    def _extract_text(data: Dict[str, object]) -> str:
        try:
            choices = data.get("choices", [])  # type: ignore[assignment]
            if not choices:
                return ""
            msg = choices[0].get("message", {})
            return str(msg.get("content", "")).strip()
        except Exception:
            return ""


class ExpertRegistry:
    def __init__(self, adapters: Dict[str, BaseExpertAdapter]) -> None:
        self.adapters = adapters

    def get(self, expert_name: str) -> BaseExpertAdapter:
        if expert_name not in self.adapters:
            raise KeyError(f"No adapter registered for expert '{expert_name}'")
        return self.adapters[expert_name]


def build_mock_registry() -> ExpertRegistry:
    return ExpertRegistry(
        {
            "A": MockExpertAdapter("A"),
            "B": MockExpertAdapter("B"),
            "C": MockExpertAdapter("C"),
        }
    )


def build_openai_registry(
    model_name: str,
    a_base_url: str,
    b_base_url: str,
    c_base_url: str,
    api_key: Optional[str] = None,
    *,
    timeout_seconds: int = 60,
) -> ExpertRegistry:
    shared_key = api_key or "dummy"
    return ExpertRegistry(
        {
            "A": OpenAIExpertAdapter(
                expert_name="A",
                base_url=a_base_url,
                model=model_name,
                api_key=shared_key,
                timeout_seconds=timeout_seconds,
                system_prompt=(
                    "You are Expert A (factual retrieval and grounding). "
                    "Return concise evidence-oriented results."
                ),
            ),
            "B": OpenAIExpertAdapter(
                expert_name="B",
                base_url=b_base_url,
                model=model_name,
                api_key=shared_key,
                timeout_seconds=timeout_seconds,
                system_prompt=(
                    "You are Expert B (reasoning). "
                    "Return structured step-by-step reasoning."
                ),
            ),
            "C": OpenAIExpertAdapter(
                expert_name="C",
                base_url=c_base_url,
                model=model_name,
                api_key=shared_key,
                timeout_seconds=timeout_seconds,
                system_prompt=(
                    "You are Expert C (writing). "
                    "Produce clear and faithful final responses from context."
                ),
            ),
        }
    )
