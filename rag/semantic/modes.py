"""Legacy/shadow/semantic mode switch and API adaptation.

- legacy: frozen 0764d3e behavior; the semantic pipeline never runs.
- shadow: legacy produces the user-visible answer; the semantic pipeline
  additionally runs for comparison/trace collection. Semantic output must
  NOT affect the final answer, claims, provenance, or returned context —
  it is recorded through internal logging only and can never raise.
- semantic: the semantic pipeline is the production path.

Default mode is legacy. The semantic path is never the default here.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

_logger = logging.getLogger("cucumber.semantic_shadow")

QA_MODE_LEGACY = "legacy"
QA_MODE_SHADOW = "shadow"
QA_MODE_SEMANTIC = "semantic"

QA_MODES = frozenset({QA_MODE_LEGACY, QA_MODE_SHADOW, QA_MODE_SEMANTIC})

QA_MODE_ENV_VAR = "CUCUMBER_QA_MODE"
DEFAULT_QA_MODE = QA_MODE_LEGACY


def active_qa_mode() -> str:
    """Return the configured QA mode, defaulting safely to legacy."""
    raw = os.environ.get(QA_MODE_ENV_VAR, "")
    mode = raw.strip().casefold() if isinstance(raw, str) else ""
    if mode in QA_MODES:
        return mode
    if mode:
        _logger.warning("unknown %s=%r; using legacy", QA_MODE_ENV_VAR, raw)
    return DEFAULT_QA_MODE


def semantic_ask_response(
    db_path: str | Path,
    question: str,
    conversation_context: dict[str, Any] | None,
    *,
    home_program: str | None = None,
    clarification_resolution: dict[str, Any] | None = None,
    clarification_resolutions: list[dict[str, Any]] | None = None,
    interpret_callable: Callable[..., str] | None,
    answer_callable: Callable[..., str] | None = None,
    sql_callable: Callable[..., str] | None = None,
) -> dict[str, Any]:
    """Run the semantic pipeline and adapt it to the AskResponse shape."""
    from rag.semantic.pipeline import semantic_answer

    outcome = semantic_answer(
        db_path,
        question,
        conversation_context if isinstance(conversation_context, dict) else None,
        home_program=home_program,
        **({"clarification_resolution": clarification_resolution}
           if clarification_resolution is not None else {}),
        **({"clarification_resolutions": clarification_resolutions}
           if clarification_resolutions is not None else {}),
        interpret_callable=interpret_callable,
        answer_callable=answer_callable,
        sql_callable=sql_callable,
    )
    result = outcome.result
    provenance: list[Any] = []
    for reference in result.provenance:
        # Frozen provenance entries are MappingProxyType, not plain dicts.
        if isinstance(reference, Mapping):
            provenance.append(dict(reference))
    status = result.status
    action: str | None = None
    if status != "answer":
        action = "insufficient_evidence" if status == "insufficient_evidence" else status
    summary = getattr(getattr(outcome, "trace", None), "verified_summary", {})
    if status == "clarify_program" and summary.get("scope_dimension") in {"program", "catalog", "plan", "year", "semester", "comparison_operation"}:
        status = "clarification_required"
        action = summary["scope_dimension"] + "_required"
    response = {
        "question": question,
        "answer": result.final_answer,
        "status": status,
        "action": action,
        "route": "semantic",
        "provenance": provenance,
        "next_context": outcome.next_context,
        "comparison": None,
    }
    if (
        action in {"plan_required", "catalog_required"}
        and (summary.get("operand") in {"left", "right"}
             or (action == "catalog_required" and summary.get("operand") is None))
        and isinstance(summary.get("program"), str)
    ):
        response["clarification_target"] = {
            "dimension": summary["scope_dimension"], "program": summary["program"],
            "operand": summary.get("operand"),
        }
    return response


def run_shadow_comparison(
    db_path: str | Path,
    question: str,
    conversation_context: dict[str, Any] | None,
    legacy_response: dict[str, Any],
    *,
    interpret_callable: Callable[..., str] | None,
    answer_callable: Callable[..., str] | None = None,
    sql_callable: Callable[..., str] | None = None,
) -> None:
    """Run the semantic pipeline beside legacy output; log only, never raise.

    The legacy response dict is never inspected for facts and never
    mutated: only its status string is copied into the comparison record.
    """
    try:
        from rag.semantic.pipeline import semantic_answer

        outcome = semantic_answer(
            db_path,
            question,
            conversation_context if isinstance(conversation_context, dict) else None,
            interpret_callable=interpret_callable,
            answer_callable=answer_callable,
            sql_callable=sql_callable,
        )
        legacy_status = None
        if isinstance(legacy_response, dict):
            status = legacy_response.get("status")
            legacy_status = status if isinstance(status, str) else None
        record = {
            "mode": "shadow",
            "question": question if isinstance(question, str) else "",
            "legacy_status": legacy_status,
            "semantic_status": outcome.result.status,
            "semantic_claim_count": len(outcome.result.claims),
            "semantic_provenance_count": len(outcome.result.provenance),
            "failure_category": outcome.trace.failure_category,
            "llm_request_count": outcome.trace.llm_request_count,
            "trace": outcome.trace.to_dict(),
        }
        _logger.info("semantic shadow comparison: %s", json.dumps(record, ensure_ascii=False, default=str)[:8000])
    except Exception as error:
        _logger.warning("semantic shadow run failed safely: %s", type(error).__name__)


__all__ = [
    "DEFAULT_QA_MODE",
    "QA_MODES",
    "QA_MODE_ENV_VAR",
    "QA_MODE_LEGACY",
    "QA_MODE_SEMANTIC",
    "QA_MODE_SHADOW",
    "active_qa_mode",
    "run_shadow_comparison",
    "semantic_ask_response",
]
