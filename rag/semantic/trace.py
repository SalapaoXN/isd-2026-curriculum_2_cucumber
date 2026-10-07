"""JSON-safe semantic trace model for observability and evaluation.

The trace answers, per question: did the LLM misunderstand it, did the
resolver choose the wrong entity, was scope incomplete, was the plan wrong,
was the evidence wrong, or did the answerer misstate a correct result?
No API keys, secrets, or chain-of-thought are ever recorded.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from rag.semantic.prompts import (
    ANSWERER_PROMPT_VERSION,
    SEMANTIC_INTERPRETER_PROMPT_VERSION,
)


@dataclass(slots=True)
class StageTiming:
    interpreter_ms: float = 0.0
    resolution_ms: float = 0.0
    execution_ms: float = 0.0
    answerer_ms: float = 0.0
    total_ms: float = 0.0


@dataclass(slots=True)
class SemanticTrace:
    question: str = ""
    mode: str = "semantic"
    interpreter_prompt_version: str = SEMANTIC_INTERPRETER_PROMPT_VERSION
    answerer_prompt_version: str = ANSWERER_PROMPT_VERSION
    interpreter_raw: str = ""
    semantic_intent: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)
    context_merge: dict[str, Any] = field(default_factory=dict)
    resolved_intent: dict[str, Any] = field(default_factory=dict)
    query_plan: dict[str, Any] = field(default_factory=dict)
    verified_summary: dict[str, Any] = field(default_factory=dict)
    failure_category: str = "NONE"
    failure_reason: str | None = None
    timing: StageTiming = field(default_factory=StageTiming)
    llm_request_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["timing"] = asdict(self.timing)
        return payload


__all__ = ["SemanticTrace", "StageTiming"]
