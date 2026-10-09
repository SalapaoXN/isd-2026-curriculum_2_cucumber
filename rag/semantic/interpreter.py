"""Single-call semantic interpreter: language in, structured intent out.

Exactly one provider call per turn. Output is strictly validated against
the closed schema; malformed output, unknown enums, unknown fields, or
multiple interpretations are rejected. No retries. Provider exceptions
propagate unchanged so the pipeline can fail closed with a typed reason.
"""

from __future__ import annotations

import json
import inspect
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
    MAX_COURSE_SET_MEMBERS,
    MAX_ORDINAL,
    MAX_TEXT_LEN,
    MEASURES,
    COMPARISON_MEASURES,
    PLAN_SELECTORS,
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
    LiteralCourseReference,
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


def _nullable(schema: dict[str, Any]) -> dict[str, Any]:
    return {"anyOf": [schema, {"type": "null"}]}


def _object_schema(
    properties: dict[str, Any], required: tuple[str, ...] = ()
) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        schema["required"] = list(required)
    return schema


def semantic_intent_json_schema() -> dict[str, Any]:
    """Build transport JSON Schema from the same closed enums/field sets as parsing.

    This constrains generated JSON shape and vocabulary at the provider boundary.
    ``parse_semantic_intent_payload`` remains authoritative and enforces the
    additional cross-field and conditional rules that JSON Schema cannot
    represent in this contract.
    """
    from rag.semantic.schema import (
        AGGREGATION_FUNCTIONS,
        COMPARISON_OPERATIONS,
        FILTER_FIELDS,
        FILTER_OPERATORS,
        GROUP_DIMENSIONS,
        MEASURES,
        POLICY_TOPICS,
        RANK_DIRECTIONS,
        RELATIONS,
        REQUESTED_FIELDS,
        SUBJECTS,
        TARGET_KINDS,
        TASKS,
    )

    def enum_schema(values: frozenset[str]) -> dict[str, Any]:
        return {"type": "string", "enum": sorted(values)}

    string_value = {"type": "string", "maxLength": MAX_TEXT_LEN}
    year_value = {"type": "integer", "minimum": 1, "maximum": 5}
    semester_value = {"type": "integer", "minimum": 1, "maximum": 2}
    scope_properties = {
        "program": _nullable(string_value),
        "catalog": _nullable(string_value),
        "plan": _nullable(string_value),
        "plan_hint": _nullable({"type": "string", "maxLength": 32}),
        "year": _nullable(year_value),
        "semester": _nullable(semester_value),
    }
    operand_properties = {
        "course": _nullable(string_value),
        "program": _nullable(string_value),
        "catalog": _nullable(string_value),
        "plan": _nullable(string_value),
        "plan_hint": _nullable({"type": "string", "maxLength": 32}),
        "year": _nullable(year_value),
        "semester": _nullable(semester_value),
    }
    comparison_properties = {
        "left": {
            **_object_schema(operand_properties),
        },
        "right": {
            **_object_schema(operand_properties),
        },
        "measure": enum_schema(COMPARISON_MEASURES),
        "operation": _nullable(enum_schema(COMPARISON_OPERATIONS)),
        "plan_selector": _nullable(enum_schema(PLAN_SELECTORS)),
    }
    filter_value = {
        "anyOf": [
            string_value,
            {"type": "number"},
            {"type": "boolean"},
            {
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "items": {"type": "number"},
            },
        ]
    }
    properties = {
        "task": enum_schema(TASKS),
        "subject": enum_schema(SUBJECTS),
        "relation": _nullable(enum_schema(RELATIONS)),
        "target": _object_schema(
            {
                "kind": enum_schema(TARGET_KINDS),
                "raw_text": _nullable(string_value),
                "normalized_hint": _nullable(
                    {"type": "string", "maxLength": MAX_HINT_LEN}
                ),
                "ordinal": _nullable(
                    {"type": "integer", "minimum": 1, "maximum": MAX_ORDINAL}
                ),
                "members": {
                    "type": "array", "maxItems": MAX_COURSE_SET_MEMBERS,
                    "items": _object_schema({
                        "raw_text": {**string_value, "minLength": 1},
                        "normalized_hint": _nullable({"type": "string", "maxLength": MAX_HINT_LEN}),
                    }, ("raw_text", "normalized_hint")),
                },
            },
            tuple(sorted(_TARGET_FIELDS)),
        ),
        "scope": _object_schema(
            scope_properties,
            tuple(sorted(_SCOPE_FIELDS - {"plan_hint"})),
        ),
        "filters": {
            "type": "array",
            "items": _object_schema(
                {
                    "field": enum_schema(FILTER_FIELDS),
                    "operator": enum_schema(FILTER_OPERATORS),
                    "value": filter_value,
                },
                tuple(sorted(_FILTER_FIELDS)),
            ),
        },
        "aggregation": _nullable(
            _object_schema(
                {
                    "function": enum_schema(AGGREGATION_FUNCTIONS),
                    "measure": enum_schema(MEASURES),
                    "group_by": {
                        "type": "array",
                        "items": enum_schema(GROUP_DIMENSIONS),
                    },
                },
                tuple(sorted(_AGGREGATION_FIELDS)),
            )
        ),
        "ranking": _nullable(
            _object_schema(
                {
                    "metric": enum_schema(MEASURES),
                    "direction": enum_schema(RANK_DIRECTIONS),
                    "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                },
                tuple(sorted(_RANKING_FIELDS)),
            )
        ),
        "comparison": _nullable(
            _object_schema(
                comparison_properties,
                tuple(sorted(_COMPARISON_FIELDS)),
            )
        ),
        "requested_fields": {
            "type": "array",
            "items": enum_schema(REQUESTED_FIELDS),
        },
        "clarification": _nullable(string_value),
        "policy_topic": _nullable(enum_schema(POLICY_TOPICS)),
        "observed_value": _nullable({"type": "string", "maxLength": 32}),
    }
    properties["target"]["anyOf"] = [
        {
            "properties": {
                "kind": {"enum": ["literal_set"]},
                "raw_text": {"type": "null"},
                "normalized_hint": {"type": "null"},
                "ordinal": {"type": "null"},
                "members": {"minItems": 2},
            },
            "required": ["members"],
        },
        {
            "properties": {
                "kind": {"enum": sorted(TARGET_KINDS - {"literal_set"})},
                "members": {"maxItems": 0},
            },
        },
    ]
    return _object_schema(properties, tuple(sorted(_INTENT_FIELDS)))


def _supports_keyword(callable_object: Callable[..., Any], keyword: str) -> bool:
    """Inspect callable capability without probing it with a model request."""
    try:
        parameters = inspect.signature(callable_object).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        or (
            parameter.name == keyword
            and parameter.kind != inspect.Parameter.POSITIONAL_ONLY
        )
        for parameter in parameters
    )


def _structured_output_kwargs(
    callable_object: Callable[..., Any],
) -> dict[str, Any]:
    """Return supported JSON transport options, or empty for simple callables."""
    if not _supports_keyword(callable_object, "response_mime_type"):
        return {}
    options: dict[str, Any] = {"response_mime_type": "application/json"}
    if _supports_keyword(callable_object, "response_json_schema"):
        options["response_json_schema"] = semantic_intent_json_schema()
    return options


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


def _parse_clarification(value: Any) -> str | None:
    # Auxiliary explanation is not semantic authority. Keep its stored text
    # bounded without rejecting an otherwise valid intent; semantic fields
    # retain _require_text's hard length/type checks.
    if isinstance(value, str):
        value = value.strip()[:MAX_TEXT_LEN]
    return _require_text(value, "clarification", allow_none=True)


def _require_int(value: Any, field: str, *, allow_none: bool) -> int | None:
    if value is None:
        if allow_none:
            return None
        raise SemanticSchemaError(f"{field} must be an integer")
    if isinstance(value, bool) or not isinstance(value, int):
        raise SemanticSchemaError(f"{field} must be an integer or null")
    return value


def _parse_target(data: Any) -> SemanticTarget:
    if (
        not isinstance(data, dict) or set(data) - (_TARGET_FIELDS | {"members"})
        or not _TARGET_FIELDS.issubset(data)
    ):
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
    member_data = data.get("members", [])
    if not isinstance(member_data, list) or len(member_data) > MAX_COURSE_SET_MEMBERS:
        raise SemanticSchemaError("target.members must be a bounded list")
    members = []
    for member in member_data:
        if not isinstance(member, dict) or set(member) != {"raw_text", "normalized_hint"}:
            raise SemanticSchemaError("target member has an invalid schema")
        members.append(LiteralCourseReference(
            raw_text=_require_text(member["raw_text"], "member.raw_text", allow_none=False),
            normalized_hint=_require_text(member["normalized_hint"], "member.normalized_hint", allow_none=True),
        ))
    if kind == "literal_set":
        if len(members) < 2 or any(data[key] is not None for key in ("raw_text", "normalized_hint", "ordinal")):
            raise SemanticSchemaError("literal_set requires 2–20 members and null singleton fields")
    elif members:
        raise SemanticSchemaError("members require literal_set")
    return SemanticTarget(
        kind=kind, raw_text=raw_text, normalized_hint=hint, ordinal=ordinal,
        members=tuple(members),
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


def _parse_scope_side(data: Any, field: str, *, allow_empty: bool = False) -> tuple[tuple[str, Any], ...]:
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
    if not items and not allow_empty:
        raise SemanticSchemaError(f"{field} must be non-empty")
    return tuple(items)


def _parse_comparison(data: Any) -> ComparisonSpec | None:
    if data is None:
        return None
    if (not isinstance(data, dict) or not _COMPARISON_FIELDS.issubset(data)
            or set(data) - (_COMPARISON_FIELDS | {"plan_selector"})):
        raise SemanticSchemaError("comparison has an invalid schema")
    measure = data["measure"]
    if measure not in COMPARISON_MEASURES:
        raise SemanticSchemaError(f"unknown comparison measure: {measure!r}")
    operation = data["operation"]
    if operation is not None and operation not in COMPARISON_OPERATIONS:
        raise SemanticSchemaError(f"unknown comparison operation: {operation!r}")
    selector = data.get("plan_selector")
    if selector is not None and (not isinstance(selector, str) or selector not in PLAN_SELECTORS):
        raise SemanticSchemaError(f"unknown plan selector: {selector!r}")
    return ComparisonSpec(
        left=_parse_scope_side(data["left"], "comparison.left", allow_empty=selector == "available_plans"),
        right=_parse_scope_side(data["right"], "comparison.right", allow_empty=selector == "available_plans"),
        measure=measure,
        operation=operation,
        plan_selector=selector,
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
    clarification = _parse_clarification(data["clarification"])
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
    canonical_program_codes: tuple[str, ...] = (),
    canonical_category_labels: tuple[str, ...] = (),
    canonical_plan_keys: tuple[str, ...] = (),
    last_normal_operation: dict[str, Any] | None = None,
) -> tuple[SemanticIntent, str]:
    """Run one interpretation call; return (intent, prompt_version used)."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if not callable(model_callable):
        raise TypeError("model_callable must be callable")
    _ = prompt_version
    prompt = build_semantic_interpreter_prompt(
        question,
        canonical_program_codes=canonical_program_codes,
        canonical_category_labels=canonical_category_labels,
        canonical_plan_keys=canonical_plan_keys,
        last_normal_operation=last_normal_operation,
    )
    structured_options = _structured_output_kwargs(model_callable)
    output = model_callable(prompt, **structured_options)
    if not isinstance(output, str):
        raise TypeError("model output must be a string")
    return parse_semantic_intent_payload(output), SEMANTIC_INTERPRETER_PROMPT_VERSION


__all__ = [
    "MAX_PAYLOAD_LEN",
    "SemanticSchemaError",
    "interpret_semantic_intent",
    "parse_semantic_intent_payload",
]
