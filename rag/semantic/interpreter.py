"""Single-call semantic interpreter: language in, structured intent out.

Exactly one provider call per turn. Output is strictly validated against
the closed schema; malformed output, unknown enums, unknown fields, or
multiple interpretations are rejected. No retries. Provider exceptions
propagate unchanged so the pipeline can fail closed with a typed reason.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from rag.semantic.prompts import (
    SEMANTIC_INTERPRETER_PROMPT_VERSION,
    build_semantic_interpreter_prompt,
)
from rag.semantic.schema import (
    AGGREGATION_FUNCTIONS,
    COMPARISON_OPERAND_KEYS,
    COMPARISON_OPERATIONS,
    FILTER_FIELDS,
    FILTER_OPERATORS,
    GROUP_DIMENSIONS,
    MAX_HINT_LEN,
    MAX_ORDINAL,
    MAX_TEXT_LEN,
    MEASURES,
    POLICY_TOPICS,
    RANK_DIRECTIONS,
    RELATIONS,
    REQUESTED_FIELDS,
    SUBJECTS,
    TARGET_KINDS,
    TASKS,
    AggregationSpec,
    ComparisonSpec,
    RankingSpec,
    ScopeMention,
    SemanticFilter,
    SemanticIntent,
    SemanticSchemaError,
    SemanticTarget,
)

MAX_PAYLOAD_LEN = 8192

_INTENT_FIELDS = frozenset(
    {
        "task",
        "subject",
        "relation",
        "target",
        "scope",
        "filters",
        "aggregation",
        "ranking",
        "comparison",
        "requested_fields",
        "clarification",
        "policy_topic",
        "observed_value",
    }
)

_TARGET_FIELDS = frozenset({"kind", "raw_text", "normalized_hint", "ordinal"})

_SCOPE_FIELDS = frozenset({"program", "catalog", "plan", "plan_hint", "year", "semester"})

_FILTER_FIELDS = frozenset({"field", "operator", "value"})

_AGGREGATION_FIELDS = frozenset({"function", "measure", "group_by"})

_RANKING_FIELDS = frozenset({"metric", "direction", "limit"})

_COMPARISON_FIELDS = frozenset({"left", "right", "measure", "operation"})


def _require_text(value: Any, field: str, *, allow_none: bool) -> str | None:
    if value is None:
        if allow_none:
            return None
        raise SemanticSchemaError(f"{field} must be a string")
    if not isinstance(value, str):
        raise SemanticSchemaError(f"{field} must be a string or null")
    text = value.strip()
    if not text:
        if allow_none:
            return None
        raise SemanticSchemaError(f"{field} must be a non-empty string")
    if len(text) > MAX_TEXT_LEN:
        raise SemanticSchemaError(f"{field} exceeds {MAX_TEXT_LEN} characters")
    return text


def _require_int(value: Any, field: str, *, allow_none: bool) -> int | None:
    if value is None:
        if allow_none:
            return None
        raise SemanticSchemaError(f"{field} must be an integer")
    if isinstance(value, bool) or not isinstance(value, int):
        raise SemanticSchemaError(f"{field} must be an integer or null")
    return value


def _parse_target(data: Any) -> SemanticTarget:
    if not isinstance(data, dict) or set(data) != _TARGET_FIELDS:
        raise SemanticSchemaError("target has an invalid schema")
    kind = data["kind"]
    if kind not in TARGET_KINDS:
        raise SemanticSchemaError(f"unknown target kind: {kind!r}")
    raw_text = _require_text(data["raw_text"], "target.raw_text", allow_none=True)
    hint = _require_text(
        data["normalized_hint"], "target.normalized_hint", allow_none=True
    )
    if hint is not None and len(hint) > MAX_HINT_LEN:
        raise SemanticSchemaError("normalized_hint is too long")
    ordinal = _require_int(data["ordinal"], "target.ordinal", allow_none=True)
    if ordinal is not None and not 1 <= ordinal <= MAX_ORDINAL:
        raise SemanticSchemaError("target.ordinal out of range")
    if kind == "result_ordinal" and ordinal is None:
        raise SemanticSchemaError("result_ordinal requires an ordinal")
    if kind != "result_ordinal" and ordinal is not None:
        raise SemanticSchemaError("ordinal requires the result_ordinal kind")
    if kind == "literal" and raw_text is None:
        raise SemanticSchemaError("literal target requires raw_text")
    if kind in {"current_course", "previous_result_set"} and raw_text is None:
        raise SemanticSchemaError(f"{kind} target requires raw_text")
    return SemanticTarget(
        kind=kind, raw_text=raw_text, normalized_hint=hint, ordinal=ordinal
    )


def _parse_scope(data: Any) -> ScopeMention:
    if not isinstance(data, dict):
        raise SemanticSchemaError("scope has an invalid schema")
    if set(data) - _SCOPE_FIELDS:
        raise SemanticSchemaError("scope has an invalid schema")
    for required in ("program", "catalog", "plan", "year", "semester"):
        if required not in data:
            raise SemanticSchemaError("scope has an invalid schema")
    year = _require_int(data["year"], "scope.year", allow_none=True)
    semester = _require_int(data["semester"], "scope.semester", allow_none=True)
    if year is not None and not 1 <= year <= 5:
        raise SemanticSchemaError("scope.year out of range")
    if semester is not None and not 1 <= semester <= 2:
        raise SemanticSchemaError("scope.semester out of range")
    plan_hint = data.get("plan_hint")
    if plan_hint is not None:
        if not isinstance(plan_hint, str) or not plan_hint.strip() or len(plan_hint) > 32:
            raise SemanticSchemaError("scope.plan_hint must be bounded text or null")
        plan_hint = plan_hint.strip()
    return ScopeMention(
        program=_require_text(data["program"], "scope.program", allow_none=True),
        catalog=_require_text(data["catalog"], "scope.catalog", allow_none=True),
        plan=_require_text(data["plan"], "scope.plan", allow_none=True),
        plan_hint=plan_hint,
        year=year,
        semester=semester,
    )


def _parse_filter(data: Any) -> SemanticFilter:
    if not isinstance(data, dict) or set(data) != _FILTER_FIELDS:
        raise SemanticSchemaError("filter has an invalid schema")
    field_name = data["field"]
    operator = data["operator"]
    if field_name not in FILTER_FIELDS:
        raise SemanticSchemaError(f"unknown filter field: {field_name!r}")
    if operator not in FILTER_OPERATORS:
        raise SemanticSchemaError(f"unknown filter operator: {operator!r}")
    value = data["value"]
    if operator == "between":
        if (
            not isinstance(value, (list, tuple))
            or len(value) != 2
            or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value)
        ):
            raise SemanticSchemaError("between requires two numbers")
        value = (value[0], value[1])
    elif operator in {"related_to", "contains"}:
        if not isinstance(value, str) or not value.strip() or len(value) > MAX_TEXT_LEN:
            raise SemanticSchemaError(f"{operator} requires bounded text")
    elif field_name == "has_prerequisite":
        if not isinstance(value, bool):
            raise SemanticSchemaError("has_prerequisite requires a boolean")
    elif field_name == "credits":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SemanticSchemaError("credits filter requires a number")
    elif field_name in {"year", "semester"}:
        if isinstance(value, bool) or not isinstance(value, int):
            raise SemanticSchemaError(f"{field_name} filter requires an integer")
    elif not isinstance(value, str) or not value.strip() or len(value) > MAX_TEXT_LEN:
        raise SemanticSchemaError(f"{field_name} filter requires bounded text")
    return SemanticFilter(field=field_name, operator=operator, value=value)


def _parse_aggregation(data: Any) -> AggregationSpec | None:
    if data is None:
        return None
    if not isinstance(data, dict) or set(data) != _AGGREGATION_FIELDS:
        raise SemanticSchemaError("aggregation has an invalid schema")
    function = data["function"]
    measure = data["measure"]
    group_by = data["group_by"]
    if function not in AGGREGATION_FUNCTIONS:
        raise SemanticSchemaError(f"unknown aggregation: {function!r}")
    if measure not in MEASURES:
        raise SemanticSchemaError(f"unknown measure: {measure!r}")
    if not isinstance(group_by, list) or any(
        item not in GROUP_DIMENSIONS for item in group_by
    ):
        raise SemanticSchemaError("unknown group_by dimension")
    return AggregationSpec(
        function=function, measure=measure, group_by=tuple(group_by)
    )


def _parse_ranking(data: Any) -> RankingSpec | None:
    if data is None:
        return None
    if not isinstance(data, dict) or set(data) != _RANKING_FIELDS:
        raise SemanticSchemaError("ranking has an invalid schema")
    metric = data["metric"]
    direction = data["direction"]
    limit = data["limit"]
    if metric not in MEASURES:
        raise SemanticSchemaError(f"unknown ranking metric: {metric!r}")
    if direction not in RANK_DIRECTIONS:
        raise SemanticSchemaError(f"unknown ranking direction: {direction!r}")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 20:
        raise SemanticSchemaError("ranking limit out of range")
    return RankingSpec(metric=metric, direction=direction, limit=limit)


def _parse_scope_side(data: Any, field: str) -> tuple[tuple[str, Any], ...]:
    if not isinstance(data, dict):
        raise SemanticSchemaError(f"{field} must be an object")
    items: list[tuple[str, Any]] = []
    for key, value in data.items():
        if key not in COMPARISON_OPERAND_KEYS:
            raise SemanticSchemaError(f"unknown comparison scope key: {key!r}")
        if value is None:
            continue
        if key in {"year", "semester"}:
            if isinstance(value, bool) or not isinstance(value, int):
                raise SemanticSchemaError(f"{field}.{key} must be an integer")
        elif key == "plan_hint":
            if not isinstance(value, str) or not value.strip() or len(value) > 32:
                raise SemanticSchemaError(f"{field}.plan_hint must be bounded text")
            value = value.strip()
        elif not isinstance(value, str) or not value.strip() or len(value) > MAX_TEXT_LEN:
            raise SemanticSchemaError(f"{field}.{key} must be bounded text")
        items.append((key, value))
    if not items:
        raise SemanticSchemaError(f"{field} must be non-empty")
    return tuple(items)


def _parse_comparison(data: Any) -> ComparisonSpec | None:
    if data is None:
        return None
    if not isinstance(data, dict) or set(data) != _COMPARISON_FIELDS:
        raise SemanticSchemaError("comparison has an invalid schema")
    measure = data["measure"]
    if measure not in MEASURES:
        raise SemanticSchemaError(f"unknown comparison measure: {measure!r}")
    operation = data["operation"]
    if operation is not None and operation not in COMPARISON_OPERATIONS:
        raise SemanticSchemaError(f"unknown comparison operation: {operation!r}")
    return ComparisonSpec(
        left=_parse_scope_side(data["left"], "comparison.left"),
        right=_parse_scope_side(data["right"], "comparison.right"),
        measure=measure,
        operation=operation,
    )


_WHOLE_JSON_FENCE_RE = re.compile(
    r"\A```(?:(?i:json))?[ \t]*\r?\n(?P<body>[\s\S]*?)\r?\n```\Z"
)


def _normalize_json_transport(payload: str) -> str:
    """Unwrap exactly one whole-payload Markdown JSON fence, if present.

    This is transport-only normalization. Raw JSON passes unchanged after
    outer whitespace trimming. Fences are accepted only when they enclose
    the entire payload; prose, multiple blocks, and nested/extra fences are
    rejected rather than searched or extracted.
    """
    text = payload.strip()
    if not text.startswith("```"):
        return text
    match = _WHOLE_JSON_FENCE_RE.fullmatch(text)
    if match is None:
        raise SemanticSchemaError("malformed whole-payload JSON fence")
    body = match.group("body").strip()
    if not body or "```" in body:
        raise SemanticSchemaError("invalid or multiple fenced payload blocks")
    return body


def parse_semantic_intent_payload(payload: str) -> SemanticIntent:
    """Validate one raw interpreter payload into a SemanticIntent."""
    if not isinstance(payload, str) or not payload.strip() or len(payload) > MAX_PAYLOAD_LEN:
        raise SemanticSchemaError("intent payload must be bounded text")
    normalized_payload = _normalize_json_transport(payload)
    try:
        data = json.loads(normalized_payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise SemanticSchemaError(f"malformed intent JSON: {error}") from None
    if not isinstance(data, dict) or set(data) != _INTENT_FIELDS:
        raise SemanticSchemaError("intent payload has an invalid schema")
    task = data["task"]
    subject = data["subject"]
    relation = data["relation"]
    if task not in TASKS:
        raise SemanticSchemaError(f"unknown task: {task!r}")
    if subject not in SUBJECTS:
        raise SemanticSchemaError(f"unknown subject: {subject!r}")
    if relation is not None and relation not in RELATIONS:
        raise SemanticSchemaError(f"unknown relation: {relation!r}")
    filters = data["filters"]
    if not isinstance(filters, list):
        raise SemanticSchemaError("filters must be a list")
    requested = data["requested_fields"]
    if not isinstance(requested, list) or any(item not in REQUESTED_FIELDS for item in requested):
        raise SemanticSchemaError("unknown requested field")
    clarification = _require_text(data["clarification"], "clarification", allow_none=True)
    policy_topic = data["policy_topic"]
    if policy_topic is not None and policy_topic not in POLICY_TOPICS:
        raise SemanticSchemaError(f"unknown policy topic: {policy_topic!r}")
    observed = _require_text(data["observed_value"], "observed_value", allow_none=True)
    if observed is not None and len(observed) > 32:
        raise SemanticSchemaError("observed_value is too long")
    return SemanticIntent(
        task=task,
        subject=subject,
        relation=relation,
        target=_parse_target(data["target"]),
        scope=_parse_scope(data["scope"]),
        filters=tuple(_parse_filter(item) for item in filters),
        aggregation=_parse_aggregation(data["aggregation"]),
        ranking=_parse_ranking(data["ranking"]),
        comparison=_parse_comparison(data["comparison"]),
        requested_fields=tuple(requested),
        clarification=clarification,
        policy_topic=policy_topic,
        observed_value=observed,
    )


def interpret_semantic_intent(
    question: str,
    model_callable: Callable[..., str],
    *,
    prompt_version: str = SEMANTIC_INTERPRETER_PROMPT_VERSION,
) -> tuple[SemanticIntent, str]:
    """Run one interpretation call; return (intent, prompt_version used)."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if not callable(model_callable):
        raise TypeError("model_callable must be callable")
    _ = prompt_version
    prompt = build_semantic_interpreter_prompt(question)
    output = model_callable(prompt)
    if not isinstance(output, str):
        raise TypeError("model output must be a string")
    return parse_semantic_intent_payload(output), SEMANTIC_INTERPRETER_PROMPT_VERSION


__all__ = [
    "MAX_PAYLOAD_LEN",
    "SemanticSchemaError",
    "interpret_semantic_intent",
    "parse_semantic_intent_payload",
]
