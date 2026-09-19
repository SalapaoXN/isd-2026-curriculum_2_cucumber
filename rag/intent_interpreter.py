"""Standalone v1 Intent Interpreter contract and validation boundary.

This module is a pure, model-agnostic typed boundary for future LLM intent
interpretation. It is NOT wired into QA routing. No SQL, no retrieval, no
retries, and no imports from ``rag.qa`` exist here. The only model contact
is :func:`interpret_question_intent`, which makes exactly one caller-provided
model call to obtain a JSON proposal and validates it.

Core semantic: model output is PROPOSED interpretation only. It is never
authoritative curriculum evidence. Authoritative scope is always supplied by
the caller (deterministic parse / conversation context) and checked by
:func:`validate_execution_scope`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


INTENTS = frozenset(
    {
        "topic_course_search",
        "course_description",
        "placement_query",
        "prerequisite_query",
        "similarity_query",
        "course_comparison",
        "plan_comparison",
        "program_discovery",
        "workload_judgement",
        "preference_recommendation_evidence",
    }
)

#: Evidence-level fact names a proposal may request. There is deliberately no
#: entry for recommendations, rankings, difficulty verdicts, or answer text:
#: conclusions have nowhere to live in this schema.
REQUESTED_FACTS = frozenset(
    {
        "course_list",
        "course_description",
        "placement",
        "prerequisite",
        "similarity",
        "course_comparison",
        "plan_comparison",
        "program_identity",
        "workload_evidence",
        "preference_evidence",
    }
)

#: Judgement dimensions that may be requested as evidence. Anything else
#: (e.g. "difficulty", "recommendation", "best") is unsupported and invalid.
JUDGEMENT_DIMENSIONS = frozenset({"workload", "preference"})

#: Intent -> required judgement_dimension. Judgement intents request evidence
#: only; non-judgement intents must carry no dimension.
INTENT_DIMENSION = {
    "workload_judgement": "workload",
    "preference_recommendation_evidence": "preference",
}

#: Intents exempt from the authoritative-program requirement. Only
#: program_discovery may be valid without program context.
PROGRAM_OPTIONAL_INTENTS = frozenset({"program_discovery"})

#: Payload keys that must never appear in model output. They describe SQL,
#: database identity, computed facts, provenance, or answer text, none of
#: which the model is authoritative for. (Any other unknown key is rejected
#: separately by the unknown-field rule; these get an explicit error.)
FORBIDDEN_FIELDS = frozenset(
    {
        "sql",
        "generated_sql",
        "query_sql",
        "course_id",
        "course_ids",
        "placement_id",
        "placement_ids",
        "plan_id",
        "plan_ids",
        "catalog_id",
        "catalog_ids",
        "credits",
        "credit_units",
        "credit_total",
        "total_credits",
        "prerequisites",
        "prerequisite_facts",
        "placements",
        "placement_facts",
        "provenance",
        "sources",
        "source_pages",
        "answer",
        "answer_text",
        "final_answer",
        "conclusion",
        "recommendation",
    }
)

#: Accepted top-level payload keys. The `proposed_*` prefix marks every model
#: value as a proposal, never an authoritative fact.
KNOWN_FIELDS = frozenset(
    {
        "intent",
        "proposed_program",
        "proposed_plans",
        "proposed_years",
        "proposed_semesters",
        "course_codes",
        "topic",
        "requested_facts",
        "judgement_dimension",
        "unresolved",
    }
)

# Small documented bounds. Oversized strings/collections are rejected.
MAX_PAYLOAD_LEN = 4096
MAX_PROGRAM_LEN = 16
MAX_TOPIC_LEN = 64
MAX_PLANS = 4
MAX_YEARS = 4
MAX_SEMESTERS = 2
MAX_COURSE_CODES = 4
MAX_REQUESTED_FACTS = 6
MAX_UNRESOLVED = 8
MAX_UNRESOLVED_ITEM_LEN = 64

MIN_YEAR = 1
MAX_YEAR = 4
MIN_SEMESTER = 1
MAX_SEMESTER = 2

_COURSE_CODE_RE = re.compile(r"^[0-9]{8}$")

#: Machine-readable eligibility reasons returned by validate_execution_scope.
ELIGIBILITY_REASONS = frozenset(
    {
        "ok",
        "unresolved_items_present",
        "program_conflict",
        "program_unsubstantiated",
        "program_missing",
        "plan_conflict",
        "year_out_of_range",
        "year_conflict",
        "semester_out_of_range",
        "semester_conflict",
        "course_code_not_allowed",
    }
)


class IntentValidationError(ValueError):
    """Raised when a model intent payload fails shape validation."""


@dataclass(frozen=True, slots=True)
class IntentInterpretation:
    """Immutable PROPOSED interpretation of one user request.

    Every `proposed_*` value is a model suggestion only. Authoritative
    curriculum scope is supplied separately to :func:`validate_execution_scope`
    and never taken from this object.
    """

    intent: str
    proposed_program: str | None
    proposed_plans: tuple[str, ...]
    proposed_years: tuple[int, ...]
    proposed_semesters: tuple[int, ...]
    course_codes: tuple[str, ...]
    topic: str | None
    requested_facts: tuple[str, ...]
    judgement_dimension: str | None
    unresolved: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.intent not in INTENTS:
            raise IntentValidationError(
                f"unknown intent: {self.intent!r}"
            )
        for field_name, values in (
            ("proposed_plans", self.proposed_plans),
            ("proposed_years", self.proposed_years),
            ("proposed_semesters", self.proposed_semesters),
            ("course_codes", self.course_codes),
            ("requested_facts", self.requested_facts),
            ("unresolved", self.unresolved),
        ):
            if not isinstance(values, tuple):
                raise IntentValidationError(
                    f"{field_name} must be a tuple"
                )
        expected_dimension = INTENT_DIMENSION.get(self.intent)
        if self.judgement_dimension != expected_dimension:
            raise IntentValidationError(
                f"intent {self.intent!r} requires judgement_dimension "
                f"{expected_dimension!r}, got {self.judgement_dimension!r}"
            )


@dataclass(frozen=True, slots=True)
class ExecutionEligibility:
    """Immutable result of scope-conflict validation."""

    eligible: bool
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.eligible, bool):
            raise TypeError("eligible must be a bool")
        if self.reason not in ELIGIBILITY_REASONS:
            raise ValueError(f"unknown eligibility reason: {self.reason!r}")


def _optional_text(value: Any, field: str, max_len: int) -> str | None:
    """Normalize an optional string: missing/None/blank -> None."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise IntentValidationError(f"{field} must be a string or null")
    text = value.strip()
    if not text:
        return None
    if len(text) > max_len:
        raise IntentValidationError(f"{field} exceeds {max_len} characters")
    return text


def _string_sequence(
    value: Any,
    field: str,
    max_items: int,
    max_item_len: int,
    *,
    allow_values: frozenset[str] | None = None,
    code_format: bool = False,
) -> tuple[str, ...]:
    """Validate a string list field into a normalized tuple.

    Duplicates are rejected, never silently deduped.
    """
    if value is None:
        return ()
    if not isinstance(value, list):
        raise IntentValidationError(f"{field} must be a list or null")
    if len(value) > max_items:
        raise IntentValidationError(
            f"{field} exceeds {max_items} items"
        )
    items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise IntentValidationError(f"{field} items must be strings")
        text = item if code_format else item.strip()
        if not text:
            raise IntentValidationError(f"{field} items must be non-empty")
        if len(text) > max_item_len:
            raise IntentValidationError(
                f"{field} items exceed {max_item_len} characters"
            )
        if code_format and _COURSE_CODE_RE.match(text) is None:
            raise IntentValidationError(
                f"{field} items must be exact 8-digit course codes"
            )
        if allow_values is not None and text not in allow_values:
            raise IntentValidationError(
                f"{field} contains unknown value: {text!r}"
            )
        items.append(text)
    if len(set(items)) != len(items):
        raise IntentValidationError(f"{field} contains duplicate values")
    return tuple(items)


def _int_sequence(
    value: Any, field: str, max_items: int
) -> tuple[int, ...]:
    """Validate an integer list field into a normalized tuple.

    Exact ``int`` type is required (bools rejected). Duplicates rejected.
    """
    if value is None:
        return ()
    if not isinstance(value, list):
        raise IntentValidationError(f"{field} must be a list or null")
    if len(value) > max_items:
        raise IntentValidationError(
            f"{field} exceeds {max_items} items"
        )
    for item in value:
        if type(item) is not int:
            raise IntentValidationError(f"{field} items must be integers")
    if len(set(value)) != len(value):
        raise IntentValidationError(f"{field} contains duplicate values")
    return tuple(value)


def parse_intent_payload(payload: str) -> IntentInterpretation:
    """Strictly validate a future model JSON payload.

    Returns a validated :class:`IntentInterpretation` whose sequences are
    tuples and whose empty optionals are normalized to ``None``/``()``.
    Any malformed shape raises :class:`IntentValidationError`. A non-empty
    ``unresolved`` list is preserved as-is; it marks the interpretation as
    unusable for execution (see :func:`validate_execution_scope`).
    """
    if not isinstance(payload, str):
        raise IntentValidationError("payload must be a JSON string")
    if len(payload) > MAX_PAYLOAD_LEN:
        raise IntentValidationError(
            f"payload exceeds {MAX_PAYLOAD_LEN} characters"
        )
    try:
        data = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise IntentValidationError(f"malformed JSON: {error}") from None
    if not isinstance(data, dict):
        raise IntentValidationError("top level must be a JSON object")
    forbidden = FORBIDDEN_FIELDS.intersection(data)
    if forbidden:
        raise IntentValidationError(
            "forbidden model-output fields: "
            + ", ".join(sorted(forbidden))
        )
    unknown = set(data).difference(KNOWN_FIELDS)
    if unknown:
        raise IntentValidationError(
            "unknown fields: " + ", ".join(sorted(unknown))
        )

    intent = data.get("intent")
    if not isinstance(intent, str) or intent not in INTENTS:
        raise IntentValidationError(f"unknown intent: {intent!r}")

    proposed_program = _optional_text(
        data.get("proposed_program"), "proposed_program", MAX_PROGRAM_LEN
    )
    proposed_plans = _string_sequence(
        data.get("proposed_plans"), "proposed_plans", MAX_PLANS, MAX_PROGRAM_LEN
    )
    proposed_years = _int_sequence(
        data.get("proposed_years"), "proposed_years", MAX_YEARS
    )
    proposed_semesters = _int_sequence(
        data.get("proposed_semesters"), "proposed_semesters", MAX_SEMESTERS
    )
    course_codes = _string_sequence(
        data.get("course_codes"),
        "course_codes",
        MAX_COURSE_CODES,
        MAX_PROGRAM_LEN,
        code_format=True,
    )
    topic = _optional_text(data.get("topic"), "topic", MAX_TOPIC_LEN)
    requested_facts = _string_sequence(
        data.get("requested_facts"),
        "requested_facts",
        MAX_REQUESTED_FACTS,
        MAX_TOPIC_LEN,
        allow_values=REQUESTED_FACTS,
    )
    judgement_dimension = _optional_text(
        data.get("judgement_dimension"),
        "judgement_dimension",
        MAX_TOPIC_LEN,
    )
    if judgement_dimension is not None:
        if judgement_dimension not in JUDGEMENT_DIMENSIONS:
            raise IntentValidationError(
                f"unsupported judgement_dimension: {judgement_dimension!r}"
            )
    unresolved = _string_sequence(
        data.get("unresolved"),
        "unresolved",
        MAX_UNRESOLVED,
        MAX_UNRESOLVED_ITEM_LEN,
    )

    return IntentInterpretation(
        intent=intent,
        proposed_program=proposed_program,
        proposed_plans=proposed_plans,
        proposed_years=proposed_years,
        proposed_semesters=proposed_semesters,
        course_codes=course_codes,
        topic=topic,
        requested_facts=requested_facts,
        judgement_dimension=judgement_dimension,
        unresolved=unresolved,
    )


def _normalize_authoritative_text(value: Any, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string or None")
    text = value.strip()
    return text or None


def _normalize_authoritative_items(value: Any, field: str) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{field} must be a sequence or None")
    try:
        return tuple(value)
    except TypeError as error:
        raise TypeError(f"{field} must be a sequence or None") from error


def validate_execution_scope(
    interpretation: IntentInterpretation,
    *,
    program: str | None = None,
    allowed_plans: Any = (),
    allowed_years: Any = (),
    allowed_semesters: Any = (),
    allowed_course_codes: Any = (),
) -> ExecutionEligibility:
    """Pure deterministic scope-conflict check against caller authority.

    The interpreted program may match the authoritative ``program`` but may
    never override it; proposed plans/years/semesters must be subsets of the
    caller-provided allowed scopes; every proposed course code must be an
    exact 8-digit code present in ``allowed_course_codes`` (codes actually
    seen in user text / deterministic parse, so no program is ever inferred
    from a course code). Any unknown/conflicting proposal, or a non-empty
    ``unresolved`` list, yields ``eligible=False``. ``program_discovery`` is
    exempt from the program rules; every other program-scoped intent must not
    invent a missing program. Validation only: no routing or clarifying here.
    """
    if not isinstance(interpretation, IntentInterpretation):
        raise TypeError("interpretation must be an IntentInterpretation")

    if interpretation.unresolved:
        return ExecutionEligibility(False, "unresolved_items_present")

    authoritative_program = _normalize_authoritative_text(program, "program")
    plans = _normalize_authoritative_items(allowed_plans, "allowed_plans")
    years = _normalize_authoritative_items(allowed_years, "allowed_years")
    semesters = _normalize_authoritative_items(
        allowed_semesters, "allowed_semesters"
    )
    codes = _normalize_authoritative_items(
        allowed_course_codes, "allowed_course_codes"
    )

    if interpretation.intent not in PROGRAM_OPTIONAL_INTENTS:
        proposed = interpretation.proposed_program
        if authoritative_program is None:
            if proposed is None:
                return ExecutionEligibility(False, "program_missing")
            return ExecutionEligibility(False, "program_unsubstantiated")
        if proposed is not None and proposed.casefold() != (
            authoritative_program.casefold()
        ):
            return ExecutionEligibility(False, "program_conflict")

    for plan in interpretation.proposed_plans:
        if plan not in plans:
            return ExecutionEligibility(False, "plan_conflict")
    for year in interpretation.proposed_years:
        if year < MIN_YEAR or year > MAX_YEAR:
            return ExecutionEligibility(False, "year_out_of_range")
        if year not in years:
            return ExecutionEligibility(False, "year_conflict")
    for semester in interpretation.proposed_semesters:
        if semester < MIN_SEMESTER or semester > MAX_SEMESTER:
            return ExecutionEligibility(False, "semester_out_of_range")
        if semester not in semesters:
            return ExecutionEligibility(False, "semester_conflict")
    for code in interpretation.course_codes:
        if code not in codes:
            return ExecutionEligibility(False, "course_code_not_allowed")

    return ExecutionEligibility(True, "ok")


def build_intent_prompt(question: str) -> str:
    """Build the bounded one-shot interpreter prompt for a question.

    The prompt states the interpreter role, the exact JSON wire format, the
    bounded vocabularies (taken from this module's constants), and the
    no-inference extraction rules. It contains no database schema and no
    curriculum facts. Pure string construction; no model contact.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    intents = ", ".join(sorted(INTENTS))
    facts = ", ".join(sorted(REQUESTED_FACTS))
    dimensions = ", ".join(sorted(JUDGEMENT_DIMENSIONS))
    return "\n".join(
        (
            "ROLE: You interpret Thai university curriculum questions into a "
            "bounded JSON proposal. You are an interpreter only. "
            "You do not answer the question.",
            "OUTPUT: Return exactly one JSON object and nothing else. "
            "No Markdown. No code fences. No explanation.",
            "Use exactly these keys: intent, proposed_program, proposed_plans, "
            "proposed_years, proposed_semesters, course_codes, topic, "
            "requested_facts, judgement_dimension, unresolved.",
            f"Allowed intent values: {intents}.",
            f"Allowed requested_facts values: {facts}.",
            "Allowed judgement_dimension values: "
            f"{dimensions} (or null when the intent is not a judgement).",
            "Extraction rules: record only information actually stated or "
            "reasonably paraphrased in the user question. "
            "Never infer a program from course-code prefixes. "
            "Never infer a missing program from uniqueness. "
            "Never invent course codes, plans, years, semesters, "
            "prerequisites, credits, placements, facts, or provenance. "
            "Leave missing information as null / [] or list it in unresolved. "
            "Do not output database IDs, SQL, answer text, or provenance. "
            "Do not conclude recommendations, superiority, difficulty, or answers.",
            "Course-code rule: every course_codes element is ONLY a bare exact "
            "8-digit ASCII course code. Never include program labels such as "
            "IT, DSBA, BIT, AIT, or GENED, never include words such as "
            "วิชา or the English word course, and never include spaces, "
            "punctuation, or explanatory text. Program and course code are "
            "separate semantic fields: the program label belongs in "
            "proposed_program, never inside course_codes. "
            "Question \"IT 06016414 เรียนช่วงไหน\": CORRECT \"proposed_program\": \"IT\", "
            "\"course_codes\": [\"06016414\"]; "
            "INCORRECT \"course_codes\": [\"IT 06016414\"]; "
            "INCORRECT \"course_codes\": [\"วิชา 06016414\"]. "
            "Question \"IT 06016414 กับ 06016420 ต่างกันยังไง\": CORRECT \"proposed_program\": \"IT\", "
            "\"course_codes\": [\"06016414\", \"06016420\"].",
            "program_discovery may leave proposed_program null.",
            "workload_judgement and preference_recommendation_evidence only "
            "identify the requested evidence dimension; they never conclude "
            "whether something is heavy, easy, good, better, or recommended.",
            "Preference rule: for intent preference_recommendation_evidence, "
            "judgement_dimension MUST be preference, topic MUST carry the "
            "user stated area of interest when present, and requested_facts "
            "MUST be exactly [\"course_list\"]. Do NOT emit "
            "[\"preference_evidence\"], and do NOT add recommendation, "
            "ranking, best, or difficulty facts. The model only requests "
            "grounded candidate-course evidence; recommendation happens later "
            "from retrieved evidence. Example question \"IT ปี 3 ถ้าอยากปูทางไป data มีตัวไหนที่ควรจับตาไว้\": CORRECT "
            "\"intent\": \"preference_recommendation_evidence\", "
            "\"proposed_program\": \"IT\", \"proposed_years\": [3], "
            "\"topic\": \"data\", \"requested_facts\": [\"course_list\"], "
            "\"judgement_dimension\": \"preference\", \"unresolved\": []. "
            "WRONG \"requested_facts\": [\"preference_evidence\"].",
            "USER QUESTION:",
            question.strip(),
        )
    )


def interpret_question_intent(
    question: str,
    model_callable: Callable[[str], str],
) -> IntentInterpretation:
    """Convert one question into a validated IntentInterpretation.

    Makes exactly one call to ``model_callable`` with the bounded prompt,
    then passes the raw model text directly through
    :func:`parse_intent_payload`. No retry, no repair, no fallback model, no
    SQL generation, and no answer polishing. Malformed output propagates
    :class:`IntentValidationError`; a model exception propagates unchanged.
    Execution-scope validation is deliberately NOT performed here; it belongs
    to the later routing/compiler layer.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if not callable(model_callable):
        raise TypeError("model_callable must be callable")
    output = model_callable(build_intent_prompt(question))
    if not isinstance(output, str):
        raise TypeError("model output must be a string")
    return parse_intent_payload(output)


__all__ = [
    "INTENTS",
    "REQUESTED_FACTS",
    "JUDGEMENT_DIMENSIONS",
    "PROGRAM_OPTIONAL_INTENTS",
    "FORBIDDEN_FIELDS",
    "KNOWN_FIELDS",
    "ELIGIBILITY_REASONS",
    "MAX_PAYLOAD_LEN",
    "MAX_PROGRAM_LEN",
    "MAX_TOPIC_LEN",
    "MAX_PLANS",
    "MAX_YEARS",
    "MAX_SEMESTERS",
    "MAX_COURSE_CODES",
    "MAX_REQUESTED_FACTS",
    "MAX_UNRESOLVED",
    "MAX_UNRESOLVED_ITEM_LEN",
    "MIN_YEAR",
    "MAX_YEAR",
    "MIN_SEMESTER",
    "MAX_SEMESTER",
    "IntentValidationError",
    "IntentInterpretation",
    "ExecutionEligibility",
    "parse_intent_payload",
    "validate_execution_scope",
    "build_intent_prompt",
    "interpret_question_intent",
]
