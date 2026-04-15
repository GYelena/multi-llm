from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

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
                raw_body = resp.read().decode("utf-8", errors="replace")
                try:
                    data = json.loads(raw_body)
                except json.JSONDecodeError:
                    return ExpertResponse(
                        node_id=request.node.node_id,
                        summary="remote expert returned invalid JSON payload",
                        confidence=0.0,
                        payload={"rawBody": raw_body[:2000]},
                        error_code="invalid_upstream_json",
                    )
        except urllib.error.HTTPError as e:
            return ExpertResponse(
                node_id=request.node.node_id,
                summary="remote expert call failed",
                confidence=0.0,
                payload={},
                error_code=self._classify_http_error(e.code),
            )
        except urllib.error.URLError as e:
            reason = str(getattr(e, "reason", "")).lower()
            if "timed out" in reason:
                code = "network_timeout"
            elif "refused" in reason:
                code = "connection_refused"
            elif "name or service not known" in reason or "temporary failure in name resolution" in reason:
                code = "dns_resolution_failed"
            else:
                code = "network_error"
            return ExpertResponse(
                node_id=request.node.node_id,
                summary="remote expert network error",
                confidence=0.0,
                payload={"reason": str(getattr(e, "reason", ""))},
                error_code=code,
            )
        except (TimeoutError, socket.timeout):
            return ExpertResponse(
                node_id=request.node.node_id,
                summary="remote expert timeout",
                confidence=0.0,
                payload={},
                error_code="network_timeout",
            )
        except Exception as e:
            return ExpertResponse(
                node_id=request.node.node_id,
                summary="remote expert call exception",
                confidence=0.0,
                payload={"exception": str(e)},
                error_code="call_exception",
            )

        text = self._extract_text(data)
        if not text:
            return ExpertResponse(
                node_id=request.node.node_id,
                summary=f"expert {self.expert_name} returned empty content",
                confidence=0.0,
                payload={"raw": data},
                error_code="empty_response",
            )

        payload, parse_error = self._parse_structured_payload(text)
        if parse_error is not None:
            repaired_payload, repair_error = self._repair_structured_payload(
                url=url,
                request=request,
                raw_text=text,
            )
            if repair_error is None and repaired_payload is not None:
                confidence = self._pick_confidence(repaired_payload)
                return ExpertResponse(
                    node_id=request.node.node_id,
                    summary=f"expert {self.expert_name} completed after repair_once",
                    confidence=confidence,
                    payload=repaired_payload,
                    error_code=None,
                )
            return ExpertResponse(
                node_id=request.node.node_id,
                summary=f"expert {self.expert_name} output is not valid structured JSON",
                confidence=0.0,
                payload={"rawText": text, "raw": data},
                error_code=repair_error or parse_error,
            )

        confidence = self._pick_confidence(payload)
        return ExpertResponse(
            node_id=request.node.node_id,
            summary=f"expert {self.expert_name} completed",
            confidence=confidence,
            payload=payload,
            error_code=None,
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

    @staticmethod
    def _classify_http_error(code: int) -> str:
        if code == 401:
            return "unauthorized"
        if code == 403:
            return "forbidden"
        if code == 404:
            return "endpoint_not_found"
        if code == 408:
            return "network_timeout"
        if code == 422:
            return "invalid_request"
        if code == 429:
            return "rate_limited"
        if 500 <= code < 600:
            return "upstream_server_error"
        return f"http_{code}"

    def _parse_structured_payload(self, text: str) -> Tuple[Dict[str, Any], Optional[str]]:
        raw = self._parse_json_like_text(text)
        if raw is None:
            return {}, "invalid_json"
        if not isinstance(raw, dict):
            return {}, "invalid_schema"
        if self.expert_name == "A":
            if not isinstance(raw.get("claims"), list):
                return {}, "invalid_schema"
            if not isinstance(raw.get("sourceRefs"), list):
                return {}, "invalid_schema"
            citation_conf = raw.get("citationConfidence")
            if not isinstance(citation_conf, (int, float)):
                return {}, "invalid_schema"
            raw["citationConfidence"] = max(0.0, min(1.0, float(citation_conf)))
            raw.setdefault("evidences", [])
        elif self.expert_name == "B":
            if not isinstance(raw.get("reasoningSteps"), list):
                return {}, "invalid_schema"
            if not isinstance(raw.get("verifications"), list):
                return {}, "invalid_schema"
            if not isinstance(raw.get("checkResult"), str):
                return {}, "invalid_schema"
        elif self.expert_name == "C":
            if not isinstance(raw.get("draft"), str):
                return {}, "invalid_schema"
            if not isinstance(raw.get("fidelityReport"), str):
                return {}, "invalid_schema"
            if not isinstance(raw.get("unsupportedStatements"), list):
                return {}, "invalid_schema"
        else:
            return {}, "invalid_schema"
        return raw, None

    @staticmethod
    def _parse_json_like_text(text: str) -> Optional[Any]:
        stripped = text.strip()
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass
        if stripped.startswith("```"):
            first_nl = stripped.find("\n")
            last_fence = stripped.rfind("```")
            if first_nl != -1 and last_fence > first_nl:
                inner = stripped[first_nl + 1 : last_fence].strip()
                try:
                    return json.loads(inner)
                except json.JSONDecodeError:
                    pass
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(stripped[start : end + 1])
            except json.JSONDecodeError:
                return None
        return None

    def _pick_confidence(self, payload: Dict[str, Any]) -> float:
        raw_conf = payload.get("confidence")
        if isinstance(raw_conf, (int, float)):
            return max(0.0, min(1.0, float(raw_conf)))
        defaults = {"A": 0.72, "B": 0.70, "C": 0.78}
        if self.expert_name == "A":
            citation_conf = payload.get("citationConfidence")
            if isinstance(citation_conf, (int, float)):
                return max(0.0, min(1.0, float(citation_conf)))
        return defaults.get(self.expert_name, 0.70)

    def _repair_structured_payload(
        self,
        url: str,
        request: ExpertRequest,
        raw_text: str,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        body = {
            "model": self.model,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": self._repair_system_prompt()},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "expert": self.expert_name,
                            "nodeId": request.node.node_id,
                            "taskType": request.node.task_type.value,
                            "invalidOutput": raw_text,
                            "targetSchemaHint": self._schema_hint(),
                        },
                        ensure_ascii=False,
                    ),
                },
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
                raw_body = resp.read().decode("utf-8", errors="replace")
                try:
                    data = json.loads(raw_body)
                except json.JSONDecodeError:
                    return None, "repair_invalid_upstream_json"
        except urllib.error.HTTPError as e:
            return None, f"repair_{self._classify_http_error(e.code)}"
        except urllib.error.URLError as e:
            reason = str(getattr(e, "reason", "")).lower()
            if "timed out" in reason:
                code = "network_timeout"
            elif "refused" in reason:
                code = "connection_refused"
            elif "name or service not known" in reason or "temporary failure in name resolution" in reason:
                code = "dns_resolution_failed"
            else:
                code = "network_error"
            return None, f"repair_{code}"
        except (TimeoutError, socket.timeout):
            return None, "repair_network_timeout"
        except Exception:
            return None, "repair_call_exception"

        text = self._extract_text(data)
        if not text:
            return None, "repair_empty_response"
        payload, parse_error = self._parse_structured_payload(text)
        if parse_error is not None:
            return None, f"repair_{parse_error}"
        return payload, None

    @staticmethod
    def _repair_system_prompt() -> str:
        return (
            "You are a JSON repair assistant. "
            "Rewrite the given invalid model output into VALID JSON that strictly matches target schema hint. "
            "Return ONLY JSON. Do not include markdown or explanations."
        )

    def _schema_hint(self) -> str:
        if self.expert_name == "A":
            return (
                '{"claims": ["..."], "evidences": ["..."], "sourceRefs": ["..."], '
                '"citationConfidence": 0.0, "confidence": 0.0}'
            )
        if self.expert_name == "B":
            return (
                '{"reasoningSteps": ["..."], "verifications": ["..."], '
                '"checkResult": "passed|failed", "confidence": 0.0}'
            )
        return (
            '{"draft": "...", "fidelityReport": "...", '
            '"unsupportedStatements": ["..."], "confidence": 0.0}'
        )


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
                    "Return ONLY valid JSON with keys: claims (list[str]), evidences (list[str]), "
                    "sourceRefs (list[str]), citationConfidence (float in [0,1]), confidence (optional float in [0,1])."
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
                    "Return ONLY valid JSON with keys: reasoningSteps (list[str]), verifications (list[str]), "
                    "checkResult (str), confidence (optional float in [0,1])."
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
                    "Return ONLY valid JSON with keys: draft (str), fidelityReport (str), "
                    "unsupportedStatements (list[str]), confidence (optional float in [0,1])."
                ),
            ),
        }
    )
