"""Top-level curriculum QA over one relational and vector database."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
import re
import sqlite3
from typing import Any

from rag.aggregation import (
    ComparisonAggregation,
    ComponentAggregation,
    CourseSetAggregation,
    EarliestAggregation,
    PlanComparisonAggregation,
    PlanComparisonInput,
    aggregate_plan_comparison,
    aggregate_course_set,
    aggregate_earliest,
    aggregate_required_load,
    aggregate_sum_credits,
    compare_aggregates,
)
from rag.evidence_executor import (
    DirectPrerequisiteBurden,
    DirectPrerequisiteRequirement,
    EvidenceBundle,
    EvidenceExecutionResult,
    execute_evidence_plan,
    execute_exact_similarity_from_bundle,
)
from rag.evidence_planner import (
    EvidencePlan,
    EvidenceRequest,
    StructuralScope,
    build_structural_scope,
    plan_evidence,
)
from rag.grounded_answer import (
    GroundedClaim,
    compose_grounded_answer,
)
from rag.intent_compiler import IntentCompilerError, compile_intent_to_query_spec
from rag.intent_interpreter import (
    IntentValidationError,
    interpret_question_intent,
)
from rag.intent_gate import (
    CountShadowResult,
    IntentShadowResult,
    _count_candidate as _is_count_shadow_candidate,
    _placement_candidate as _placement_shadow_candidate,
    authoritative_scope as _intent_authoritative_scope,
    count_shadow_comparison as _count_shadow_comparison,
    run_count_shadow,
    run_exact_course_shadow,
    run_placement_shadow,
)
from rag.judgement import (
    JudgementEvidence,
    evaluate_preference,
    evaluate_quantity,
    evaluate_workload,
)
from rag.policy.answer import answer_policy_query
from rag.policy.query import PolicyQuery, parse_policy_question
from rag.policy.routing import adapt_policy_answer, route_policy_question
from rag.query_spec import detect_surface_operations, parse_query_spec
from rag.retrieval.retrieve import (
    ConstrainedTopicRetrievalResult,
    SimilarityEvidence,
)
from rag.answer import render_grounded_answer
from rag.resolution import (
    QueryContext,
    ResolutionOutcome,
    has_answerable_target_or_scope,
    resolve_query_spec,
)
from rag.structured.fallback import (
    GroundedCourseCreditResult,
    GroundedCourseListResult,
    GroundedPlacementResult,
    StructuredFallbackScope,
    ground_course_credit,
    ground_course_list,
    ground_placement,
    run_structured_fallback,
)
from rag.structured.queries import (
    catalog_keys_for_program,
    edition_catalog_keys_for_program,
    exact_course_candidates,
    prerequisite_state,
)


_STRUCTURED_FALLBACK_OPERATIONS = frozenset(
    {"list", "count", "existence", "sum_credits", "placement", "earliest", "prerequisite"}
)
_STRUCTURED_FALLBACK_CUE = re.compile(
    r"วิชา|หลักสูตร|ลงเรียน|ลงทะเบียน|หน่วยกิต|เครดิต|วิชาบังคับ|"
    r"ปี|เทอม|ภาคเรียน|course|semester|year|credits?|prerequisite",
    re.IGNORECASE,
)
_CATEGORY_FILTER_RESIDUE = re.compile(
    r"ศึกษาทั่วไป|(?<![A-Za-z0-9_])(?:gen\s*ed|electives?)(?![A-Za-z0-9_])|วิชาเลือก",
    re.IGNORECASE,
)
_REQUIREMENT_FILTER_RESIDUE = re.compile(r"วิชาบังคับ(?!\s*ก่อน)", re.IGNORECASE)
_CREDIT_UNIT_FILTER_RESIDUE = re.compile(
    r"(?<!\d)\d+(?:\.\d+)?\s*หน่วยกิต", re.IGNORECASE
)
_LIST_FILTER_FALLBACK_CUE = re.compile(
    r"มีวิชา(?:[^?\n]{0,80})?(?:อะไร|ไหน)(?:บ้าง)?|"
    r"ลงเรียนวิชา[^?\n]{0,80}(?:อะไร|ไหน)(?:บ้าง)?",
    re.IGNORECASE,
)
_PLACEMENT_FALLBACK_CUE = re.compile(
    r"สามารถลงได้[^?\n]{0,50}(?:ช่วงไหน|ตอนไหน|ปีไหน|เทอมไหน)|"
    r"(?:อยู่ช่วงไหน|เรียนตอนไหน|ลงตอนไหน|ลงเรียนช่วงไหน|ลงเรียนตอนไหน|เรียนปีไหน|เทอมไหน)",
    re.IGNORECASE,
)
_COURSE_CREDIT_FALLBACK_CUE = re.compile(
    r"กี่\s*หน่วย(?:กิต)?(?:อะ|นะ|ครับ|คะ)?(?![ก-๙A-Za-z0-9_])|"
    r"กี่\s*เครดิต(?![ก-๙A-Za-z0-9_])|"
    r"(?<![A-Za-z0-9_])credits?(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_COURSE_CREDIT_EXCLUDE_CUE = re.compile(
    r"ลงทะเบียน|รวม|ทั้งหมด|ทุกวิชา|หมวด|วิชาบังคับ|วิชาเลือก|มีวิชา|กี่รายวิชา",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class StructuredParseCompleteness:
    """Conservative eligibility result for the future SQL fallback."""

    classification: str
    missing_filters: tuple[str, ...] = ()
    program: str | None = None
    plans: tuple[str, ...] = ()
    years: tuple[int, ...] = ()
    semesters: tuple[int, ...] = ()
    course_codes: tuple[str, ...] = ()
    course_name: str | None = None


def _classify_structured_parse_completeness(
    spec: Any,
    resolution: ResolutionOutcome,
    context: QueryContext | None = None,
) -> StructuredParseCompleteness:
    """Classify only bounded structured residue; never infer missing facts."""
    program = getattr(spec, "program", None) or getattr(context, "program", None)
    common = {
        "program": program,
        "plans": tuple(getattr(spec, "plans", ())),
        "years": tuple(getattr(spec, "years", ())),
        "semesters": tuple(getattr(spec, "semesters", ())),
        "course_codes": tuple(getattr(spec, "course_codes", ())),
        "course_name": getattr(spec, "course_name", None),
    }
    not_eligible = StructuredParseCompleteness(
        "not_eligible",
        **common,
    )
    if getattr(resolution, "action", None) != "answer":
        return not_eligible
    if not isinstance(program, str) or not program.strip():
        return not_eligible
    if getattr(spec, "judgement", None) == "unsupported":
        return not_eligible

    question = getattr(spec, "normalized_question", "")
    missing_filters: list[str] = []
    if (
        getattr(spec, "category", None) is None
        and _CATEGORY_FILTER_RESIDUE.search(question)
    ):
        missing_filters.append("category")
    if _REQUIREMENT_FILTER_RESIDUE.search(question):
        missing_filters.append("requirement_type")
    if _CREDIT_UNIT_FILTER_RESIDUE.search(question) and getattr(
        spec, "credit_units", None
    ) is None:
        # A captured integral predicate fully specifies the credit axis;
        # unparseable residue (e.g. non-integral) keeps the partial path.
        missing_filters.append("credit_units")

    if missing_filters:
        return StructuredParseCompleteness(
            "partial",
            missing_filters=tuple(missing_filters),
            **common,
        )

    operations = tuple(getattr(spec, "operations", ()))
    if getattr(spec, "credit_units", None) is not None and operations != ("list",):
        # H23-R1: deterministic credit filtering supports only ("list",).
        # Count/existence/sum/placement/compare + predicate stay fail-closed
        # before planner/fallback; no sum+filter or other combos. Empty
        # missing avoids the filtered-credit fallback seam (which requires
        # non-empty missing) so this fails closed with zero model calls.
        return StructuredParseCompleteness(
            "partial",
            missing_filters=(),
            **common,
        )
    if (
        getattr(spec, "category", None) is not None
        and "sum_credits" in operations
        and (
            len(tuple(getattr(spec, "years", ()))) != 1
            or len(tuple(getattr(spec, "semesters", ()))) != 1
        )
    ):
        # H27-B: category-aware sums are supported only for an explicit
        # single term. Year-only / semester-only / program-only totals would
        # need a default academic scope that changes the meaning of "total",
        # so they stay fail-closed before planner/fallback with zero model
        # calls. Empty missing keeps the filtered-credit fallback seam False.
        return StructuredParseCompleteness(
            "partial",
            missing_filters=(),
            **common,
        )
    if any(operation not in _STRUCTURED_FALLBACK_OPERATIONS for operation in operations):
        return not_eligible
    if getattr(spec, "topic", None) is not None:
        return not_eligible

    if operations:
        return StructuredParseCompleteness("complete", **common)

    has_scope = bool(
        common["plans"]
        or common["years"]
        or common["semesters"]
        or common["course_codes"]
        or common["course_name"]
    )
    if has_scope and (
        _STRUCTURED_FALLBACK_CUE.search(question)
        or _PLACEMENT_FALLBACK_CUE.search(question)
        or _COURSE_CREDIT_FALLBACK_CUE.search(question)
    ):
        return StructuredParseCompleteness(
            "unrecognized_structured",
            **common,
        )
    return not_eligible


_FOLLOWUP_SCOPE_CUE = re.compile(
    r"(?:แล้ว\s*(?:ปี|เทอม|ภาค)|ล่ะ|อีกที)"
)
_FOLLOWUP_COURSE_CUE = re.compile(r"กี่หน่วยกิต|กี่หน่วย|prerequisite|วิชาบังคับก่อน", re.I)
_FOLLOWUP_PLACEMENT_CUE = re.compile(r"เรียนตอนไหน|เรียนช่วงไหน|จัดไว้ปีไหน|เทอมอะไร", re.I)


def _merge_conversation_context(spec: Any, context: QueryContext | None) -> Any:
    """Fill only missing structural fields from explicit caller context."""
    if context is None:
        return spec

    updates: dict[str, Any] = {}
    for field_name in ("program", "category"):
        if getattr(spec, field_name, None) is None:
            value = getattr(context, field_name, None)
            if value is not None:
                updates[field_name] = value
    if not getattr(spec, "plans", ()) and context.plan is not None:
        updates["plans"] = (context.plan,)
    if not getattr(spec, "years", ()) and context.years:
        updates["years"] = context.years
    if not getattr(spec, "semesters", ()) and context.semesters:
        updates["semesters"] = context.semesters
    if not getattr(spec, "course_codes", ()) and context.course_code is not None:
        updates["course_codes"] = (context.course_code,)

    operations = tuple(getattr(spec, "operations", ()))
    if not operations and getattr(spec, "topic", None) is None:
        question = getattr(spec, "normalized_question", "")
        current_exact_reference = bool(
            getattr(spec, "course_codes", ())
            or getattr(spec, "course_name", None) is not None
        )
        merged_program = getattr(spec, "program", None) or context.program
        if (
            not current_exact_reference
            and isinstance(merged_program, str)
            and bool(merged_program.strip())
            and context.operations == ("list",)
            and context.course_code is None
            and "prerequisite" in detect_surface_operations(question)
        ):
            operations = ("prerequisite",)
        elif context.course_code is not None and _FOLLOWUP_COURSE_CUE.search(question):
            if re.search(r"กี่หน่วย", question, re.I):
                operations = ("sum_credits",)
            elif re.search(r"prerequisite|วิชาบังคับก่อน", question, re.I):
                operations = ("existence", "prerequisite")
        elif context.course_code is not None and _FOLLOWUP_PLACEMENT_CUE.search(question):
            operations = ("placement",)
        elif context.operations and _FOLLOWUP_SCOPE_CUE.search(question):
            operations = context.operations
    if operations:
        updates["operations"] = operations
    return replace(spec, **updates) if updates else spec


def _next_conversation_context(
    spec: Any,
    resolution: ResolutionOutcome,
    result: Any,
    catalog_key: str | None = None,
) -> QueryContext | None:
    """Derive structural references only from an authoritative current turn."""
    if getattr(result, "status", None) != "answer" or resolution.action != "answer":
        return None
    if getattr(spec, "topic", None) is not None:
        return None
    program = getattr(spec, "program", None) or resolution.resolved_program
    if not isinstance(program, str) or not program.strip():
        return None
    references = resolution.course_references
    course_code = None
    if len(references) == 1 and len(references[0].candidates) == 1:
        candidate = references[0].candidates[0]
        course_code = candidate.get("course_code")
        if not isinstance(course_code, str) or not course_code.strip():
            course_code = None
    plans = tuple(getattr(spec, "plans", ())) or tuple(resolution.resolved_plans)
    return QueryContext(
        program=program,
        catalog_key=catalog_key,
        plan=plans[0] if len(plans) == 1 else None,
        years=tuple(getattr(spec, "years", ())),
        semesters=tuple(getattr(spec, "semesters", ())),
        category=getattr(spec, "category", None),
        course_code=course_code,
        operations=tuple(getattr(spec, "operations", ())),
    )


def _is_course_list_fallback_candidate(
    spec: Any,
    completeness: StructuredParseCompleteness,
) -> bool:
    """Allow only bounded course-list/filter residue into the SQL seam."""
    if getattr(spec, "topic", None) is not None:
        return False
    operations = tuple(getattr(spec, "operations", ()))
    if getattr(spec, "credit_units", None) is not None:
        # A parsed credit predicate normally stays on the deterministic path.
        # When the requirement type is still unresolved, allow only the
        # bounded list fallback to interpret the combined filter.
        return (
            completeness.classification == "partial"
            and "requirement_type" in completeness.missing_filters
            and operations == ("list",)
        )
    if operations:
        if operations in {("count",), ("existence",)}:
            return True
        return "list" in operations and set(operations) <= {
            "list",
            "sum_credits",
        }
    return bool(
        completeness.classification == "unrecognized_structured"
        and _LIST_FILTER_FALLBACK_CUE.search(
            getattr(spec, "normalized_question", "")
        )
    )


def _is_placement_fallback_candidate(
    spec: Any,
    completeness: StructuredParseCompleteness,
) -> bool:
    """Allow only bounded placement wording into the placement SQL seam."""
    if getattr(spec, "credit_units", None) is not None:
        # H32-B (H23-R1): same credit-bearing fail-closed rule as the
        # course-list seam above; placement + credit must never run an
        # unfiltered placement selector.
        return False
    if completeness.missing_filters:
        return False
    operations = tuple(getattr(spec, "operations", ()))
    if operations:
        return operations == ("placement",)
    return bool(
        completeness.classification == "unrecognized_structured"
        and _PLACEMENT_FALLBACK_CUE.search(
            getattr(spec, "normalized_question", "")
        )
    )


def _run_placement_shadow(
    question: str,
    spec: Any,
    completeness: StructuredParseCompleteness,
    resolution: ResolutionOutcome,
    context: QueryContext | None,
    intent_model_callable: Callable[[str], str] | None,
) -> IntentShadowResult:
    return run_placement_shadow(
        question,
        spec,
        completeness,
        resolution,
        context,
        intent_model_callable,
        interpret_callable=interpret_question_intent,
        compile_callable=compile_intent_to_query_spec,
    )


def _run_count_shadow(
    question: str,
    spec: Any,
    completeness: StructuredParseCompleteness,
    resolution: ResolutionOutcome,
    context: QueryContext | None,
    intent_model_callable: Callable[[str], str] | None,
) -> CountShadowResult:
    return run_count_shadow(
        question,
        spec,
        completeness,
        resolution,
        context,
        intent_model_callable,
        interpret_callable=interpret_question_intent,
        compile_callable=compile_intent_to_query_spec,
    )


def _is_whole_program_total(spec: Any, program: Any) -> bool:
    """Whether an unscoped program credit sum must use requirement evidence.

    A bare program (optionally plan-qualified) credit total has exactly one
    canonical authority: the program requirement record. Placement-row sums
    inflate it (duplicate placements, electives, alternatives), so they must
    never serve it. Any narrowing axis (course, term, category, topic,
    predicate, previous-set reference) keeps the existing evidence path.
    """
    if not isinstance(program, str) or not program.strip():
        return False
    if tuple(getattr(spec, "operations", ())) != ("sum_credits",):
        return False
    if tuple(getattr(spec, "course_codes", ())) or (
        getattr(spec, "course_name", None) is not None
    ):
        return False
    if (
        tuple(getattr(spec, "years", ()))
        or tuple(getattr(spec, "semesters", ()))
        or getattr(spec, "category", None) is not None
        or getattr(spec, "credit_units", None) is not None
        or getattr(spec, "topic", None) is not None
        or tuple(getattr(spec, "group_by", ()))
        or getattr(spec, "judgement", None) not in (None, "none")
    ):
        return False
    if bool(getattr(spec, "references_previous_result_set", False)):
        return False
    if getattr(spec, "result_ordinal", None) is not None:
        return False
    return True


def _is_course_credit_fallback_candidate(
    spec: Any,
    completeness: StructuredParseCompleteness,
    resolution: ResolutionOutcome,
) -> bool:
    """Allow only one exact logical course into the credit SQL seam."""
    if completeness.missing_filters or tuple(getattr(spec, "operations", ())):
        return False

    question = getattr(spec, "normalized_question", "")
    if not _COURSE_CREDIT_FALLBACK_CUE.search(question):
        return False
    if _COURSE_CREDIT_EXCLUDE_CUE.search(question):
        return False

    course_codes: list[str] = []
    for code in completeness.course_codes:
        if isinstance(code, str) and code.strip() and code not in course_codes:
            course_codes.append(code)
    for reference in resolution.course_references:
        for candidate in reference.candidates:
            code = candidate.get("course_code")
            if isinstance(code, str) and code.strip() and code not in course_codes:
                course_codes.append(code)

    return len(resolution.course_references) == 1 and len(course_codes) == 1


def _should_use_intent_interpreter(
    spec: Any,
    completeness: StructuredParseCompleteness,
    resolution: ResolutionOutcome,
    context: QueryContext | None = None,
) -> bool:
    """Allow bounded operation-free and preference evidence interpretation."""
    if getattr(resolution, "action", None) != "answer":
        return False
    if completeness.classification not in {
        "unrecognized_structured",
        "not_eligible",
    }:
        return False
    operations = tuple(getattr(spec, "operations", ()))
    bounded_preference_operations = {
        ("list",),
        ("prerequisite",),
        ("list", "prerequisite"),
    }
    operation_bearing_preference = (
        getattr(spec, "judgement", None) == "preference"
        and getattr(spec, "topic", None) is not None
        and not getattr(spec, "course_codes", ())
        and getattr(spec, "course_name", None) is None
        and operations in bounded_preference_operations
    )
    if (operations and not operation_bearing_preference) or getattr(
        spec, "judgement", None
    ) == "unsupported":
        return False

    program = getattr(spec, "program", None) or getattr(context, "program", None)
    if not isinstance(program, str) or not program.strip():
        return False

    # These are already bounded deterministic fields.  They make the input a
    # plausible structured long-tail question without adding a new phrase
    # catalog or allowing arbitrary prose to invoke a model.
    return bool(
        getattr(spec, "topic", None)
        or getattr(spec, "course_codes", ())
        or getattr(spec, "course_name", None)
        or getattr(spec, "plans", ())
        or getattr(spec, "years", ())
        or getattr(spec, "semesters", ())
        or getattr(spec, "judgement", None) in {"workload", "preference"}
    )


_PROGRAM_LABEL_WORDS = frozenset({"ait", "bit", "dsba", "gened", "it"})

_EXPLICIT_PROGRAM_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:AIT|BIT|DSBA|IT)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)

_DEICTIC_REFERENCE_RE = re.compile(r"(?:ตัวนี้|วิชานี้|อันนี้)")

# NL-2/NL-2B: operation shapes eligible for exact-course title recovery.
# sum_credits/placement/existence fail closed deterministically when no
# exact course target is present (bare sum/existence guards, target guard),
# so admitting them cannot change any already-answerable deterministic path:
# post-guard the candidate is unreachable by construction for those shapes.
# describe is admitted because a targetless, topicless describe carries no
# executable relational meaning (description evidence requires an exact
# target) and fails closed today; it may only gain a validated literal
# title, never lose an answer. Shapes that can already answer without a
# title (prerequisite/list collections) stay strictly authoritative.
_TITLE_RECOVERY_OPERATION_SHAPES = frozenset(
    {
        ("sum_credits",),
        ("placement",),
        ("existence",),
        ("describe",),
    }
)

# NL-2B: only a heuristic describe operation may yield to a validated
# linguistic interpretation. Precise relational operations (sum_credits,
# placement, existence, prerequisite, identity, ...) remain authoritative
# even without a parsed target, per the NL-2 conflict contract.
_LINGUISTIC_OVERRIDE_BASES = frozenset({("describe",)})

# NL-2B: proposal operations the linguistic override may install. Bounded
# exact-course relations only; collections, comparisons, judgements, and
# discovery can never ride the override.
_LINGUISTIC_OVERRIDE_OPERATIONS = frozenset(
    {
        "describe",
        "prerequisite",
        "placement",
        "sum_credits",
        "existence",
        "identity",
    }
)


def _compile_exact_course_language_recovery(
    base_spec: Any,
    structure: Any,
    context: QueryContext | None,
) -> Any:
    """Compile a query-structure proposal with linguistic-operation authority.

    The existing strict compiler stays the only compilation path: this
    helper first tries it unchanged. Only when strict compilation rejects
    with IntentCompilerError AND every narrow precondition holds does it
    retry against an operation-cleared base, treating the heuristic
    deterministic operation as linguistic signal rather than factual
    authority. Preconditions: base carries exactly ("describe",) with NO
    exact course target, no scope/filter/judgement fields, an authoritative
    program is present, the proposal carries exactly one allowed
    exact-course operation with a null predicate and a literal title span.
    Anything else re-raises the strict error. The cleared retry still runs
    the full strict validation (span, scope, facts) inside the compiler.
    """
    try:
        return compile_intent_to_query_spec(base_spec, structure)
    except IntentCompilerError:
        pass
    proposed_operations = tuple(getattr(structure, "operations", ()))
    if (
        tuple(getattr(base_spec, "operations", ()))
        not in _LINGUISTIC_OVERRIDE_BASES
        or len(proposed_operations) != 1
        or proposed_operations[0] not in _LINGUISTIC_OVERRIDE_OPERATIONS
        or getattr(structure, "predicate", None) is not None
        or getattr(structure, "course_name_span", None) is None
        or tuple(getattr(base_spec, "course_codes", ()))
        or getattr(base_spec, "course_name", None) is not None
        or getattr(base_spec, "topic", None) is not None
        or getattr(base_spec, "category", None) is not None
        or getattr(base_spec, "credit_units", None) is not None
        or getattr(base_spec, "judgement", None) not in (None, "none")
        or tuple(getattr(base_spec, "plans", ()))
        or tuple(getattr(base_spec, "years", ()))
        or tuple(getattr(base_spec, "semesters", ()))
        or tuple(getattr(base_spec, "group_by", ()))
        or bool(getattr(base_spec, "references_previous_result_set", False))
        or getattr(base_spec, "result_ordinal", None) is not None
        or (
            isinstance(getattr(context, "course_code", None), str)
            and getattr(context, "course_code").strip()
        )
        or not (getattr(base_spec, "program", None) or getattr(context, "program", None))
    ):
        raise
    return compile_intent_to_query_spec(
        replace(base_spec, operations=()),
        structure,
    )


def _explicit_program_mentions(question: str) -> set[str]:
    """Return the distinct program labels named literally in the question."""
    if not isinstance(question, str):
        return set()
    return {
        match.group(0).upper()
        for match in _EXPLICIT_PROGRAM_RE.finditer(question)
    }


def _has_non_program_ascii_word(question: str) -> bool:
    """Return True when the question carries a non-program ASCII token.

    Generic structural signal reused by the exact-course gates: a literal
    course title such as "Calculus 2" leaves an ASCII alphabetic token that
    is not a program label. Pure-Thai questions (nicknames included) never
    qualify, so no alias inference can enter through this seam.
    """
    if not isinstance(question, str):
        return False
    return any(
        token.strip(".,?!:;()[]{}").isascii()
        and token.strip(".,?!:;()[]{}").isalpha()
        and token.strip(".,?!:;()[]{}").casefold() not in _PROGRAM_LABEL_WORDS
        for token in question.split()
    )


def _should_use_query_structure_interpreter(
    spec: Any,
    resolution: ResolutionOutcome,
    question: str,
    completeness: StructuredParseCompleteness,
    context: QueryContext | None = None,
) -> bool:
    """Admit only bounded incomplete shapes with no model-owned scope."""
    if getattr(resolution, "action", None) not in {"answer", "clarify_program"}:
        return False
    if (
        getattr(spec, "topic", None) is not None
        or getattr(spec, "judgement", None) not in {None, "none"}
        or getattr(spec, "credit_units", None) is not None
    ):
        return False
    surfaced = set(detect_surface_operations(question))
    clarified_collection = bool(
        getattr(resolution, "action", None) in {"answer", "clarify_program"}
        and tuple(getattr(spec, "operations", ())) == ()
        and getattr(spec, "program", None) is None
        and not tuple(getattr(spec, "plans", ()))
        and not tuple(getattr(spec, "years", ()))
        and not tuple(getattr(spec, "semesters", ()))
        and not tuple(getattr(spec, "course_codes", ()))
        and getattr(spec, "course_name", None) is None
        and getattr(spec, "topic", None) is None
        and {"list", "prerequisite"}.issubset(surfaced)
    )
    if clarified_collection:
        return True

    if (
        tuple(getattr(spec, "course_codes", ()))
        or getattr(spec, "course_name", None) is not None
        or getattr(spec, "category", None) is not None
    ):
        return False
    has_scope = bool(
        getattr(spec, "program", None)
        or getattr(context, "program", None)
        or tuple(getattr(spec, "plans", ()))
        or tuple(getattr(spec, "years", ()))
        or tuple(getattr(spec, "semesters", ()))
        or tuple(getattr(spec, "course_codes", ()))
        or getattr(spec, "course_name", None) is not None
    )
    has_non_program_ascii_word = _has_non_program_ascii_word(question)
    scoped_prerequisite_candidate = bool(
        getattr(resolution, "action", None) == "answer"
        and completeness.classification
        in {"not_eligible", "unrecognized_structured"}
        and has_scope
        and "prerequisite" in surfaced
    )
    unscoped_exact_course_candidate = bool(
        getattr(resolution, "action", None) in {"answer", "clarify_program"}
        and completeness.classification
        in {"not_eligible", "unrecognized_structured"}
        and tuple(getattr(spec, "operations", ())) == ()
        and getattr(spec, "program", None) is None
        and not tuple(getattr(spec, "course_codes", ()))
        and getattr(spec, "course_name", None) is None
        and getattr(spec, "topic", None) is None
        and getattr(spec, "category", None) is None
        and getattr(spec, "credit_units", None) is None
        and surfaced.intersection(
            {"prerequisite", "placement", "sum_credits", "describe", "existence"}
        )
        and has_non_program_ascii_word
    )
    flexible_exact_course_candidate = bool(
        getattr(resolution, "action", None) == "answer"
        and completeness.classification
        in {"not_eligible", "unrecognized_structured"}
        and (getattr(spec, "program", None) or getattr(context, "program", None))
        and not tuple(getattr(spec, "plans", ()))
        and not tuple(getattr(spec, "years", ()))
        and not tuple(getattr(spec, "semesters", ()))
        and not tuple(getattr(spec, "operations", ()))
        and has_non_program_ascii_word
    )
    context_course_code = getattr(context, "course_code", None)
    operation_bearing_exact_course_candidate = bool(
        getattr(resolution, "action", None) == "answer"
        and completeness.classification
        in {"not_eligible", "unrecognized_structured", "complete"}
        and tuple(getattr(spec, "operations", ()))
        in _TITLE_RECOVERY_OPERATION_SHAPES
        and not tuple(getattr(spec, "course_codes", ()))
        and getattr(spec, "course_name", None) is None
        and getattr(spec, "category", None) is None
        and not tuple(getattr(spec, "plans", ()))
        and not tuple(getattr(spec, "years", ()))
        and not tuple(getattr(spec, "semesters", ()))
        and not tuple(getattr(spec, "group_by", ()))
        and not bool(getattr(spec, "references_previous_result_set", False))
        and getattr(spec, "result_ordinal", None) is None
        and not (
            isinstance(context_course_code, str) and context_course_code.strip()
        )
        and (getattr(spec, "program", None) or getattr(context, "program", None))
        and len(_explicit_program_mentions(question)) <= 1
        and _has_non_program_ascii_word(question)
        and _DEICTIC_REFERENCE_RE.search(question) is None
    )
    return (
        scoped_prerequisite_candidate
        or unscoped_exact_course_candidate
        or flexible_exact_course_candidate
        or operation_bearing_exact_course_candidate
    )


def _should_use_course_list_interpreter(
    spec: Any,
    completeness: StructuredParseCompleteness,
    resolution: ResolutionOutcome,
    context: QueryContext | None = None,
) -> bool:
    """Admit only operation-free scoped shapes for LIST recovery.

    Partial classifications (credit/category/requirement residue) never
    enter: unsupported filters such as credit values stay fail-closed
    with zero model calls. The only accepted `not_eligible` shape is a
    category-only request whose category is already deterministically
    parsed: with no plan/year/semester axis there is nothing else the
    classification could be missing.
    """
    if getattr(resolution, "action", None) != "answer":
        return False
    if completeness.classification == "not_eligible":
        if (
            getattr(spec, "category", None) is None
            or tuple(getattr(spec, "plans", ())) != ()
            or tuple(getattr(spec, "years", ())) != ()
            or tuple(getattr(spec, "semesters", ())) != ()
        ):
            return False
    elif completeness.classification != "unrecognized_structured":
        return False
    if tuple(getattr(spec, "operations", ())) != ():
        return False
    if getattr(spec, "topic", None) is not None:
        return False
    if getattr(spec, "judgement", None) not in (None, "none"):
        return False
    if tuple(getattr(spec, "course_codes", ())) or (
        getattr(spec, "course_name", None) is not None
    ):
        return False
    program = getattr(spec, "program", None) or getattr(context, "program", None)
    if not isinstance(program, str) or not program.strip():
        return False
    return bool(
        tuple(getattr(spec, "plans", ()))
        or tuple(getattr(spec, "years", ()))
        or tuple(getattr(spec, "semesters", ()))
        or getattr(spec, "category", None) is not None
        or getattr(context, "plan", None) is not None
        or tuple(getattr(context, "years", ()))
        or tuple(getattr(context, "semesters", ()))
        or getattr(context, "category", None) is not None
    )


def _run_course_list_interpreter(
    db_path: str | Path,
    question: str,
    spec: Any,
    context: QueryContext | None,
    intent_model_callable: Callable[[str], str] | None,
) -> tuple[Any, Any, Any] | None:
    """Recover only ("list",) for an eligible shape; None fails closed."""
    if not callable(intent_model_callable):
        return None
    try:
        interpretation = interpret_question_intent(
            question,
            intent_model_callable,
        )
    except Exception:
        return None
    if (
        interpretation.intent != "course_list_query"
        or tuple(interpretation.requested_facts) != ("course_list",)
    ):
        return None
    try:
        compiled_spec = compile_intent_to_query_spec(
            spec,
            interpretation,
            **_intent_authoritative_scope(spec, context),
        )
    except (TypeError, ValueError):
        return None
    if tuple(compiled_spec.operations) != ("list",):
        return None
    try:
        compiled_resolution = resolve_query_spec(
            compiled_spec,
            db_path,
            context=context,
        )
    except (FileNotFoundError, OSError, TypeError, ValueError, KeyError):
        return None
    if compiled_resolution.action != "answer":
        return None
    compiled_completeness = _classify_structured_parse_completeness(
        compiled_spec,
        compiled_resolution,
        context,
    )
    if compiled_completeness.classification != "complete":
        return None
    return compiled_spec, compiled_resolution, compiled_completeness


def _intent_failure_result(question: str) -> dict[str, Any]:
    grounded = compose_grounded_answer(
        resolution_status="insufficient_evidence",
    )
    return {
        "route": None,
        "result": render_grounded_answer(
            grounded,
            answer_model_callable=None,
            question=question,
        ),
    }


def _fallback_scope(
    completeness: StructuredParseCompleteness,
    resolution: ResolutionOutcome,
    *,
    placement_code_identity: bool = False,
    catalog_key: str | None = None,
) -> StructuredFallbackScope | None:
    """Build fallback scope only from deterministic parser/resolver state."""
    program = completeness.program or resolution.resolved_program
    if not isinstance(program, str) or not program.strip():
        return None

    plans = tuple(resolution.resolved_plans or completeness.plans)
    course_ids: list[int] = []
    course_codes: list[str] = list(completeness.course_codes)
    for reference in resolution.course_references:
        for candidate in reference.candidates:
            candidate_id = candidate.get("course_id")
            if isinstance(candidate_id, int) and not isinstance(candidate_id, bool):
                if candidate_id not in course_ids:
                    course_ids.append(candidate_id)
            candidate_code = candidate.get("course_code")
            if isinstance(candidate_code, str) and candidate_code.strip():
                if candidate_code not in course_codes:
                    course_codes.append(candidate_code)

    # A resolved course_id is catalog-local.  For placement fallback, an
    # exact course-code reference remains authoritative across all applicable
    # plan/catalog partitions unless no exact code was resolved.  Do not let
    # one resolver candidate narrow an otherwise unconstrained placement set.
    if placement_code_identity and course_codes:
        course_ids = []

    try:
        return StructuredFallbackScope(
            program=program,
            plans=plans,
            years=completeness.years,
            semesters=completeness.semesters,
            course_ids=tuple(course_ids),
            course_codes=tuple(course_codes),
            catalog_key=catalog_key,
        )
    except (TypeError, ValueError):
        return None


def _fallback_course_list_claim(
    grounded: GroundedCourseListResult,
    scope: StructuredFallbackScope,
    *,
    operation: str = "list",
) -> GroundedClaim:
    """Adapt canonical course-set facts to the requested relation claim."""
    if operation not in {"list", "count", "existence"}:
        raise ValueError(f"unsupported fallback relation operation: {operation!r}")

    effective_scope = StructuralScope(
        program=scope.program,
        plans=scope.plans,
        years=scope.years,
        semesters=scope.semesters,
        catalog_key=scope.catalog_key,
    )
    if grounded.status == "insufficient_evidence":
        return GroundedClaim(
            claim_id=f"fallback_{operation}",
            operation=operation,
            effective_scope=effective_scope,
            status="insufficient_evidence",
        )

    evidence_complete = grounded.status in {"complete", "valid_empty"}
    try:
        aggregate = aggregate_course_set(
            grounded.records,
            evidence_complete=evidence_complete,
        )
    except (TypeError, ValueError, OverflowError):
        return GroundedClaim(
            claim_id=f"fallback_{operation}",
            operation=operation,
            effective_scope=effective_scope,
            status="insufficient_evidence",
        )

    value = {
        "list": aggregate.courses,
        "count": aggregate.count,
        "existence": aggregate.exists,
    }[operation]

    return GroundedClaim(
        claim_id=f"fallback_{operation}",
        operation=operation,
        effective_scope=effective_scope,
        status=aggregate.status,
        value=value,
        evidence=aggregate,
        provenance=_provenance_from_records(aggregate.courses),
    )


def _course_list_fallback_operation(spec: Any) -> str | None:
    """Return the relation operation selected by deterministic parsing only."""
    operations = tuple(getattr(spec, "operations", ()))
    if operations == ("count",):
        return "count"
    if operations == ("existence",):
        return "existence"
    if "list" in operations and set(operations) <= {"list", "sum_credits"}:
        return "list"
    if not operations:
        return "list"
    return None


def _is_filtered_sum_credits_fallback_candidate(
    spec: Any,
    completeness: StructuredParseCompleteness,
) -> bool:
    """Allow only bounded partial credit filters into the list selector."""
    return (
        completeness.classification == "partial"
        and tuple(getattr(spec, "operations", ())) == ("sum_credits",)
        and bool(completeness.missing_filters)
        and set(completeness.missing_filters)
        <= {"category", "requirement_type", "credit_units"}
        and getattr(spec, "topic", None) is None
    )


def _filtered_credit_plan(
    spec: Any,
    resolution: ResolutionOutcome,
    selected_targets: tuple[Mapping[str, Any], ...],
    *,
    catalog_key: str | None = None,
) -> EvidencePlan:
    """Build only the filtered credit request after selector grounding."""
    scope = replace(
        build_structural_scope(spec, resolution, catalog_key=catalog_key),
        course_targets=selected_targets,
    )
    request = EvidenceRequest(
        request_id="credit_facts",
        kind="credit_facts",
        scope=scope,
        course_targets=selected_targets,
        provenance_required=True,
    )
    return EvidencePlan(scope=scope, requests=(request,), group_by=scope.group_by)


def _filtered_credit_status_claim(
    scope: StructuralScope,
    status: str,
) -> GroundedClaim:
    """Represent selector-empty/failure states without executing broad credit facts."""
    if status == "valid_empty":
        aggregate = ComponentAggregation(
            operation="sum_credits",
            status="valid_empty",
            value=0,
        )
        return GroundedClaim(
            claim_id="fallback_sum_credits",
            operation="sum_credits",
            effective_scope=scope,
            status="valid_empty",
            value=0,
            evidence=aggregate,
        )
    return GroundedClaim(
        claim_id="fallback_sum_credits",
        operation="sum_credits",
        effective_scope=scope,
        status="insufficient_evidence",
    )


def _fallback_placement_claim(
    grounded: GroundedPlacementResult,
    scope: StructuredFallbackScope,
) -> GroundedClaim:
    """Adapt canonical fallback placement records to the typed claim shape."""
    effective_scope = StructuralScope(
        program=scope.program,
        plans=scope.plans,
        years=scope.years,
        semesters=scope.semesters,
        catalog_key=scope.catalog_key,
    )
    if grounded.status == "insufficient_evidence":
        return GroundedClaim(
            claim_id="fallback_placement",
            operation="placement",
            effective_scope=effective_scope,
            status="insufficient_evidence",
        )
    if grounded.status == "valid_empty":
        return GroundedClaim(
            claim_id="fallback_placement",
            operation="placement",
            effective_scope=effective_scope,
            status="valid_empty",
        )
    return GroundedClaim(
        claim_id="fallback_placement",
        operation="placement",
        effective_scope=effective_scope,
        status="complete",
        value=grounded.records,
        evidence=grounded.records,
        provenance=_provenance_from_records(grounded.records),
    )


def _fallback_course_credit_claim(
    grounded: GroundedCourseCreditResult,
    scope: StructuredFallbackScope,
) -> GroundedClaim:
    """Adapt canonical fallback credit facts to the existing credit claim shape."""
    effective_scope = StructuralScope(
        program=scope.program,
        plans=scope.plans,
        years=scope.years,
        semesters=scope.semesters,
        catalog_key=scope.catalog_key,
        course_targets=tuple(
            {"course_code": course_code} for course_code in scope.course_codes
        ),
    )
    if grounded.status == "insufficient_evidence":
        return GroundedClaim(
            claim_id="fallback_course_credit",
            operation="sum_credits",
            effective_scope=effective_scope,
            status="insufficient_evidence",
        )
    if grounded.status == "valid_empty":
        aggregate = ComponentAggregation(
            operation="sum_credits",
            status="valid_empty",
            value=0,
        )
    else:
        try:
            aggregate = ComponentAggregation(
                operation="sum_credits",
                status="complete",
                value=grounded.credit_units,
                components=grounded.records,
            )
        except (TypeError, ValueError, OverflowError):
            return GroundedClaim(
                claim_id="fallback_course_credit",
                operation="sum_credits",
                effective_scope=effective_scope,
                status="insufficient_evidence",
            )
    return GroundedClaim(
        claim_id="fallback_course_credit",
        operation="sum_credits",
        effective_scope=effective_scope,
        status=aggregate.status,
        value=aggregate.value,
        evidence=aggregate,
        provenance=grounded.provenance,
    )


def _execution_results(
    bundle: EvidenceBundle,
    kind: str,
) -> tuple[EvidenceExecutionResult, ...]:
    """Select typed executor results without changing executor ordering."""
    if not isinstance(bundle, EvidenceBundle) or not isinstance(kind, str):
        return ()
    return tuple(result for result in bundle.results if result.kind == kind)


def _effective_scope_partition(scope: Any) -> dict[str, Any] | None:
    """Represent one effective scope as the partition metadata expected downstream."""
    if scope is None or not all(
        hasattr(scope, field)
        for field in (
            "program",
            "plans",
            "years",
            "semesters",
            "category",
            "group_by",
        )
    ):
        return None
    return {
        "program": scope.program,
        "plans": tuple(scope.plans),
        "years": tuple(scope.years),
        "semesters": tuple(scope.semesters),
        "category": scope.category,
        "group_by": tuple(scope.group_by),
    }


def _attach_effective_scope(
    records: Iterable[Mapping[str, Any]],
    scope: Any,
) -> tuple[Mapping[str, Any], ...] | None:
    """Copy payload records and attach, without overwriting, their effective partition."""
    partition = _effective_scope_partition(scope)
    if partition is None:
        return None
    attached: list[Mapping[str, Any]] = []
    try:
        for record in records:
            if not isinstance(record, Mapping):
                return None
            copied = dict(record)
            if "partition" not in copied:
                copied["partition"] = partition
            elif not isinstance(copied["partition"], Mapping):
                return None
            attached.append(copied)
    except (TypeError, ValueError):
        return None
    return tuple(attached)


def _payload_records(
    result: EvidenceExecutionResult,
    field: str | None = None,
) -> tuple[Mapping[str, Any], ...] | None:
    """Extract mapping records from one result and attach its concrete scope."""
    payload = result.payload
    if field is not None:
        if not isinstance(payload, Mapping):
            return None
        payload = payload.get(field, ())
    if isinstance(payload, Mapping) or isinstance(payload, (str, bytes)):
        return None
    try:
        records = tuple(payload)
    except TypeError:
        return None
    return _attach_effective_scope(records, result.effective_scope)


def _course_set_aggregate(
    result: EvidenceExecutionResult,
) -> CourseSetAggregation | None:
    """Adapt one course-set result to the frozen list/count/existence aggregate."""
    if result.status == "valid_empty":
        return aggregate_course_set((), evidence_complete=True)
    if result.kind == "topic_matches":
        if result.status != "complete":
            return aggregate_course_set((), evidence_complete=False)
        records = _topic_candidates(result)
    else:
        records = _payload_records(result, "courses")
    if records is None:
        return None
    try:
        return aggregate_course_set(
            records,
            evidence_complete=result.status == "complete",
        )
    except (TypeError, ValueError, OverflowError):
        return None


def _cached_course_aggregate(
    result: EvidenceExecutionResult,
    cache: dict[int, CourseSetAggregation | None],
) -> CourseSetAggregation | None:
    key = id(result)
    if key not in cache:
        cache[key] = _course_set_aggregate(result)
    return cache[key]


def _credit_aggregate(
    result: EvidenceExecutionResult,
) -> ComponentAggregation | None:
    """Adapt one credit result using authoritative counted-credit components."""
    if result.status == "valid_empty":
        return aggregate_sum_credits((), evidence_complete=True)
    records = _payload_records(result, "components")
    if records is None:
        return None
    try:
        return aggregate_sum_credits(
            records,
            evidence_complete=result.status == "complete",
        )
    except (TypeError, ValueError, OverflowError):
        return None


def _provenance_from_records(records: Iterable[Mapping[str, Any]]) -> tuple[Any, ...]:
    """Return first-seen component provenance without deriving new references."""
    result: list[Any] = []
    for record in records:
        provenance = record.get("provenance", ())
        if isinstance(provenance, Mapping) or isinstance(provenance, str):
            provenance = (provenance,)
        if not isinstance(provenance, (list, tuple)):
            return ()
        for reference in provenance:
            if not isinstance(reference, (Mapping, str)):
                return ()
            if reference not in result:
                result.append(reference)
    return tuple(result)


def _provenance_from_value(value: Any) -> tuple[Any, ...]:
    """Read provenance from supported typed evidence without inventing it."""
    if isinstance(value, CourseSetAggregation):
        return _provenance_from_records(value.courses)
    if isinstance(value, ComponentAggregation):
        return _provenance_from_records(value.components)
    if isinstance(value, EarliestAggregation):
        return tuple(
            reference
            for partition in value.partitions
            for reference in partition.provenance
        )
    if isinstance(value, ComparisonAggregation):
        return _provenance_from_value(value.left) + _provenance_from_value(value.right)
    if isinstance(value, PlanComparisonAggregation):
        return tuple(value.provenance)
    if isinstance(value, JudgementEvidence):
        return tuple(value.provenance)
    if isinstance(value, Mapping):
        return _provenance_from_records((value,))
    if isinstance(value, (list, tuple)) and all(isinstance(item, Mapping) for item in value):
        return _provenance_from_records(value)
    return ()


def _claim(
    operation: str,
    result: EvidenceExecutionResult,
    *,
    value: Any = None,
    evidence: Any = None,
    provenance: tuple[Any, ...] = (),
    kind: str = "deterministic_fact",
    status: str | None = None,
    effective_scope: Any = None,
) -> GroundedClaim:
    """Build one immutable claim while redacting failed evidence."""
    claim_status = status or result.status
    if claim_status == "complete" and result.planned_request.provenance_required and not provenance:
        claim_status = "insufficient_evidence"
    if claim_status == "insufficient_evidence":
        value = None
        evidence = None
        provenance = ()
    return GroundedClaim(
        claim_id="pending",
        operation=operation,
        effective_scope=(
            result.effective_scope if effective_scope is None else effective_scope
        ),
        status=claim_status,
        kind=kind,
        value=value,
        evidence=evidence,
        provenance=provenance,
    )


def _relation_results(
    query_spec: Any,
    bundle: EvidenceBundle,
) -> tuple[EvidenceExecutionResult, ...]:
    """Choose topic matches for topic operations and course sets otherwise."""
    if getattr(query_spec, "topic", None) is not None:
        topic_results = _execution_results(bundle, "topic_matches")
        if topic_results:
            return topic_results
    return _execution_results(bundle, "course_set")


def _topic_candidates(
    result: EvidenceExecutionResult,
) -> tuple[Mapping[str, Any], ...] | None:
    payload = result.payload
    if not isinstance(payload, ConstrainedTopicRetrievalResult):
        return None
    if result.status != "complete":
        return ()
    candidates = _attach_effective_scope(
        payload.scored_candidates,
        result.effective_scope,
    )
    return candidates


def _preference_options_with_prerequisites(
    topic_result: EvidenceExecutionResult,
    prerequisite_results: tuple[EvidenceExecutionResult, ...],
) -> tuple[Mapping[str, Any], ...] | None:
    """Join one topic result with its topic-dependent burden result."""
    candidates = _topic_candidates(topic_result)
    if candidates is None or topic_result.status != "complete":
        return None
    matching = tuple(
        result
        for result in prerequisite_results
        if result.kind == "prerequisite_facts"
        and result.planned_request.depends_on == (topic_result.request_id,)
        and result.effective_scope == topic_result.effective_scope
    )
    if len(matching) != 1 or matching[0].status != "complete":
        return None
    payload = matching[0].payload
    if not isinstance(payload, (list, tuple)):
        return None
    burdens = tuple(payload)
    if any(not isinstance(burden, DirectPrerequisiteBurden) for burden in burdens):
        return None

    burdens_by_identity: dict[tuple[Any, Any, Any], list[DirectPrerequisiteBurden]] = {}
    for burden in burdens:
        identity = (burden.program, burden.course_code, burden.course_id)
        burdens_by_identity.setdefault(identity, []).append(burden)

    enriched: list[Mapping[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            return None
        identity = (
            candidate.get("program"),
            candidate.get("course_code"),
            candidate.get("course_id"),
        )
        matching_burdens = burdens_by_identity.get(identity)
        if not matching_burdens:
            return None
        burden = matching_burdens.pop(0)
        option = dict(candidate)
        option["direct_prerequisite_burden"] = burden
        enriched.append(option)

    if any(burdens_by_identity.values()):
        return None
    return tuple(enriched)


def _scope_prerequisite_dependency(
    result: EvidenceExecutionResult,
    course_results: tuple[EvidenceExecutionResult, ...],
) -> bool:
    request = result.planned_request
    return (
        result.kind == "prerequisite_facts"
        and len(request.depends_on) == 1
        and any(
            course_result.request_id == request.depends_on[0]
            for course_result in course_results
        )
    )


def _prerequisite_burden_identity(
    burden: DirectPrerequisiteBurden,
) -> tuple[str, str, int] | None:
    if (
        not isinstance(burden.program, str)
        or not burden.program.strip()
        or not isinstance(burden.course_code, str)
        or not burden.course_code.strip()
        or isinstance(burden.course_id, bool)
        or not isinstance(burden.course_id, int)
    ):
        return None
    return burden.program.strip(), burden.course_code.strip(), burden.course_id


def _classify_prerequisite_burden(
    burden: DirectPrerequisiteBurden,
) -> str | None:
    if not isinstance(burden, DirectPrerequisiteBurden):
        return None
    if burden.status != "complete" or not burden.provenance:
        return None
    groups = burden.ordered_requirement_groups
    if not isinstance(groups, tuple) or any(
        not isinstance(group, DirectPrerequisiteRequirement) or not group.provenance
        for group in groups
    ):
        return None
    required_count = sum(group.kind == "required_course" for group in groups)
    alternative_groups = tuple(
        group for group in groups if group.kind == "alternative_group"
    )
    if any(group.kind not in {"required_course", "alternative_group"} for group in groups):
        return None
    if burden.required_course_count != required_count:
        return None
    if burden.alternative_group_count != len(alternative_groups):
        return None
    if burden.alternative_member_counts != tuple(
        len(group.alternative_members) for group in alternative_groups
    ):
        return None
    if burden.required_course_count or burden.alternative_group_count:
        return "required"
    if groups:
        return None
    return "explicit_none"


def _course_row_identity(
    row: Mapping[str, Any],
    *,
    fallback_program: str | None = None,
) -> tuple[str, str, int] | None:
    program = row.get("program") or fallback_program
    course_code = row.get("course_code")
    course_id = row.get("course_id")
    if (
        not isinstance(program, str)
        or not program.strip()
        or not isinstance(course_code, str)
        or not course_code.strip()
        or isinstance(course_id, bool)
        or not isinstance(course_id, int)
    ):
        return None
    return program.strip(), course_code.strip(), course_id


def _merged_scope_prerequisite_provenance(
    course_rows: tuple[Mapping[str, Any], ...],
    burdens: tuple[DirectPrerequisiteBurden, ...],
) -> tuple[Any, ...] | None:
    course_provenance: list[Any] = []
    for row in course_rows:
        if not _provenance_from_records((row,)):
            return None
        for reference in _provenance_from_records((row,)):
            if reference not in course_provenance:
                course_provenance.append(reference)
    burden_provenance: list[Any] = []
    for burden in burdens:
        if not burden.provenance:
            return None
        for reference in burden.provenance:
            if reference not in burden_provenance:
                burden_provenance.append(reference)
    merged = course_provenance[:]
    for reference in burden_provenance:
        if reference not in merged:
            merged.append(reference)
    return tuple(merged)


def _claim_for_scope_prerequisite(
    prerequisite_result: EvidenceExecutionResult,
    course_results: tuple[EvidenceExecutionResult, ...],
) -> GroundedClaim:
    dependency_id = prerequisite_result.planned_request.depends_on
    matching_courses = tuple(
        result
        for result in course_results
        if dependency_id == (result.request_id,)
        and result.effective_scope == prerequisite_result.effective_scope
    )
    if len(matching_courses) != 1:
        return _claim("list", prerequisite_result, status="insufficient_evidence")
    course_result = matching_courses[0]
    if course_result.status not in {"complete", "valid_empty"}:
        return _claim("list", course_result, status="insufficient_evidence")
    if prerequisite_result.planned_request.positive_prerequisite_collection:
        if prerequisite_result.status == "valid_empty":
            return _claim(
                "list",
                prerequisite_result,
                value=(),
                evidence=(),
                status="valid_empty",
            )
        if prerequisite_result.status != "complete":
            return _claim("list", prerequisite_result, status="insufficient_evidence")
        records = _payload_records(prerequisite_result)
        provenance = _provenance_from_records(records or ())
        if records is None or not records or not provenance:
            return _claim("list", prerequisite_result, status="insufficient_evidence")
        if any(
            record.get("prerequisite_collection_incomplete") is not False
            for record in records
        ):
            return _claim("list", prerequisite_result, status="insufficient_evidence")
        return _claim(
            "list",
            prerequisite_result,
            value=records,
            evidence=records,
            provenance=provenance,
        )
    if prerequisite_result.status != "complete":
        if prerequisite_result.status == "valid_empty":
            payload = prerequisite_result.payload
            if isinstance(payload, (list, tuple)) and not payload:
                pass
            else:
                return _claim("list", course_result, status="insufficient_evidence")
        else:
            return _claim("list", course_result, status="insufficient_evidence")

    course_rows = _payload_records(course_result, "courses")
    if course_rows is None:
        return _claim("list", course_result, status="insufficient_evidence")
    raw_burdens = prerequisite_result.payload
    if not isinstance(raw_burdens, (list, tuple)):
        return _claim("list", course_result, status="insufficient_evidence")
    burdens = tuple(raw_burdens)
    burdens_by_identity: dict[tuple[str, str, int], DirectPrerequisiteBurden] = {}
    for burden in burdens:
        if not isinstance(burden, DirectPrerequisiteBurden):
            return _claim("list", course_result, status="insufficient_evidence")
        identity = _prerequisite_burden_identity(burden)
        if identity is None or identity in burdens_by_identity:
            return _claim("list", course_result, status="insufficient_evidence")
        if _classify_prerequisite_burden(burden) is None:
            return _claim("list", course_result, status="insufficient_evidence")
        burdens_by_identity[identity] = burden

    filtered_rows: list[Mapping[str, Any]] = []
    used_burden_identities: set[tuple[str, str, int]] = set()
    logical_identities: set[tuple[Any, ...]] = set()
    for row in course_rows:
        is_alternative = (
            row.get("is_alternative") is True
            or row.get("alternative_group_id") is not None
        )
        if is_alternative:
            program = row.get("program")
            group_id = row.get("alternative_group_id")
            members = row.get("alternative_courses")
            if (
                not isinstance(program, str)
                or not program.strip()
                or group_id is None
                or not isinstance(members, (list, tuple))
                or not members
            ):
                return _claim("list", course_result, status="insufficient_evidence")
            logical_identity = ("alternative_group", program.strip(), group_id)
            if logical_identity in logical_identities:
                return _claim("list", course_result, status="insufficient_evidence")
            logical_identities.add(logical_identity)
            member_states: list[str] = []
            member_identities: set[tuple[str, str, int]] = set()
            for member in members:
                if not isinstance(member, Mapping):
                    return _claim("list", course_result, status="insufficient_evidence")
                identity = _course_row_identity(member, fallback_program=program)
                if identity is None or identity in member_identities:
                    return _claim("list", course_result, status="insufficient_evidence")
                member_identities.add(identity)
                burden = burdens_by_identity.get(identity)
                if burden is None or identity in used_burden_identities:
                    return _claim("list", course_result, status="insufficient_evidence")
                used_burden_identities.add(identity)
                member_states.append(_classify_prerequisite_burden(burden) or "invalid")
            if all(state == "required" for state in member_states):
                filtered_rows.append(row)
            elif not all(state == "explicit_none" for state in member_states):
                return _claim("list", course_result, status="insufficient_evidence")
            continue

        identity = _course_row_identity(row)
        if identity is None or ("course", identity) in logical_identities:
            return _claim("list", course_result, status="insufficient_evidence")
        logical_identities.add(("course", identity))
        burden = burdens_by_identity.get(identity)
        if burden is None or identity in used_burden_identities:
            return _claim("list", course_result, status="insufficient_evidence")
        used_burden_identities.add(identity)
        state = _classify_prerequisite_burden(burden)
        if state == "required":
            filtered_rows.append(row)
        elif state != "explicit_none":
            return _claim("list", course_result, status="insufficient_evidence")

    if len(used_burden_identities) != len(burdens_by_identity):
        return _claim("list", course_result, status="insufficient_evidence")
    try:
        aggregate = aggregate_course_set(filtered_rows, evidence_complete=True)
    except (TypeError, ValueError, OverflowError):
        return _claim("list", course_result, status="insufficient_evidence")
    provenance = _merged_scope_prerequisite_provenance(course_rows, burdens)
    if provenance is None:
        return _claim("list", course_result, status="insufficient_evidence")
    return _claim(
        "list",
        course_result,
        value=aggregate.courses,
        evidence=aggregate,
        provenance=provenance,
        status=aggregate.status,
    )


def _claim_for_relation_operation(
    operation: str,
    result: EvidenceExecutionResult,
    cache: dict[int, CourseSetAggregation | None],
) -> GroundedClaim | None:
    aggregate = _cached_course_aggregate(result, cache)
    if aggregate is None:
        return _claim(operation, result, status="insufficient_evidence")
    if operation == "list":
        value = aggregate.courses if aggregate.status != "insufficient_evidence" else None
    elif operation == "count":
        value = aggregate.count
    else:
        value = aggregate.exists
    provenance = _provenance_from_records(aggregate.courses)
    return _claim(
        operation,
        result,
        value=value,
        evidence=aggregate,
        provenance=provenance,
        status=aggregate.status,
    )


def _claim_for_credit_operation(
    operation: str,
    result: EvidenceExecutionResult,
) -> GroundedClaim:
    aggregate = _credit_aggregate(result)
    if aggregate is None:
        return _claim(operation, result, status="insufficient_evidence")
    provenance = _provenance_from_records(aggregate.components)
    return _claim(
        operation,
        result,
        value=aggregate.value,
        evidence=aggregate,
        provenance=provenance,
        status=aggregate.status,
    )


def _missing_greatest_credit_claim() -> GroundedClaim:
    """Represent an unsupported or incomplete greatest-credit result."""
    return GroundedClaim(
        claim_id="pending",
        operation="sum_credits",
        status="insufficient_evidence",
    )


def _greatest_credit_claims(
    query_spec: Any,
    bundle: EvidenceBundle,
    results: tuple[EvidenceExecutionResult, ...],
) -> tuple[GroundedClaim, ...] | None:
    """Select maximum concrete-term credit claims independently per plan."""
    operations = set(getattr(query_spec, "operations", ()))
    if (
        operations != {"sum_credits", "compare"}
        or tuple(getattr(query_spec, "group_by", ())) != ("semester",)
        or getattr(query_spec, "years", ())
        or getattr(query_spec, "semesters", ())
        or getattr(query_spec, "course_codes", ())
        or getattr(query_spec, "course_name", None) is not None
    ):
        return None

    if not isinstance(bundle, EvidenceBundle) or not results:
        return (_missing_greatest_credit_claim(),)

    requested_program = getattr(query_spec, "program", None)
    candidates: dict[tuple[str, str], dict[tuple[int, int], ComponentAggregation]] = {}
    representatives: dict[tuple[str, str], EvidenceExecutionResult] = {}
    invalid_plans: set[tuple[str, str]] = set()
    global_invalid = False

    for result in results:
        scope = result.effective_scope
        program = getattr(scope, "program", None)
        plans = tuple(getattr(scope, "plans", ()))
        years = tuple(getattr(scope, "years", ()))
        semesters = tuple(getattr(scope, "semesters", ()))
        if (
            not isinstance(program, str)
            or not program.strip()
            or (isinstance(requested_program, str) and program != requested_program)
            or len(plans) != 1
            or not isinstance(plans[0], str)
            or not plans[0].strip()
            or len(years) != 1
            or isinstance(years[0], bool)
            or not isinstance(years[0], int)
            or len(semesters) != 1
            or isinstance(semesters[0], bool)
            or not isinstance(semesters[0], int)
        ):
            global_invalid = True
            continue

        plan_key = (program, plans[0])
        representatives.setdefault(plan_key, result)
        candidate_key = (years[0], semesters[0])
        if candidate_key in candidates.setdefault(plan_key, {}):
            invalid_plans.add(plan_key)
            continue

        aggregate = _credit_aggregate(result)
        if result.status == "complete":
            valid_status = aggregate is not None and aggregate.status == "complete"
        elif result.status == "valid_empty":
            valid_status = aggregate is not None and aggregate.status == "valid_empty"
        else:
            valid_status = False
        if not valid_status or aggregate is None or aggregate.value is None:
            invalid_plans.add(plan_key)
            continue

        provenance = _provenance_from_records(aggregate.components)
        if result.planned_request.provenance_required and aggregate.status == "complete" and not provenance:
            invalid_plans.add(plan_key)
            continue

        for component in aggregate.components:
            partition = component.get("partition")
            if (
                not isinstance(partition, Mapping)
                or partition.get("program") != program
                or tuple(partition.get("plans", ())) != (plans[0],)
                or tuple(partition.get("years", ())) != years
                or tuple(partition.get("semesters", ())) != semesters
            ):
                invalid_plans.add(plan_key)
                break
        else:
            candidates[plan_key][candidate_key] = aggregate

    if global_invalid or not candidates:
        return (_missing_greatest_credit_claim(),)

    expected: dict[tuple[str, str], set[tuple[int, int]]] = {}
    relation_results = _relation_results(query_spec, bundle)
    if relation_results:
        for result in relation_results:
            scope = result.effective_scope
            program = getattr(scope, "program", None)
            plans = tuple(getattr(scope, "plans", ()))
            years = tuple(getattr(scope, "years", ()))
            semesters = tuple(getattr(scope, "semesters", ()))
            if (
                not isinstance(program, str)
                or len(plans) != 1
                or not isinstance(plans[0], str)
                or not years
                or not semesters
            ):
                global_invalid = True
                continue
            plan_key = (program, plans[0])
            expected.setdefault(plan_key, set()).update(
                (year, semester)
                for year in years
                for semester in semesters
            )
    else:
        scope = getattr(bundle.plan, "scope", None)
        plans = tuple(getattr(scope, "plans", ()))
        years = tuple(getattr(scope, "years", ()))
        semesters = tuple(getattr(scope, "semesters", ()))
        program = getattr(scope, "program", None)
        if plans and years and semesters and isinstance(program, str):
            for plan in plans:
                expected[(program, plan)] = {
                    (year, semester)
                    for year in years
                    for semester in semesters
                }

    if global_invalid:
        return (_missing_greatest_credit_claim(),)

    plan_keys = set(candidates) | set(expected)
    claims: list[GroundedClaim] = []
    for plan_key in sorted(plan_keys):
        plan_candidates = candidates.get(plan_key, {})
        if (
            plan_key in invalid_plans
            or not plan_candidates
            or (
                plan_key in expected
                and set(plan_candidates) != expected[plan_key]
            )
        ):
            representative = representatives.get(plan_key)
            if representative is None:
                claims.append(_missing_greatest_credit_claim())
            else:
                claims.append(
                    _claim(
                        "sum_credits",
                        representative,
                        status="insufficient_evidence",
                        effective_scope=replace(
                            representative.effective_scope,
                            years=(),
                            semesters=(),
                            group_by=("plan",),
                        ),
                    )
                )
            continue

        maximum = max(aggregate.value for aggregate in plan_candidates.values())
        winners = sorted(
            (
                term,
                aggregate,
            )
            for term, aggregate in plan_candidates.items()
            if aggregate.value == maximum
        )
        for term, aggregate in winners:
            representative = next(
                result
                for result in results
                if result.effective_scope.program == plan_key[0]
                and tuple(result.effective_scope.plans) == (plan_key[1],)
                and tuple(result.effective_scope.years) == (term[0],)
                and tuple(result.effective_scope.semesters) == (term[1],)
            )
            claims.append(
                _claim(
                    "sum_credits",
                    representative,
                    value=aggregate.value,
                    evidence=aggregate,
                    provenance=_provenance_from_records(aggregate.components),
                    status=aggregate.status,
                )
            )
    return tuple(claims)


def _coarse_year_credit_claims(
    query_spec: Any,
    results: tuple[EvidenceExecutionResult, ...],
) -> tuple[GroundedClaim, ...] | None:
    """Roll semester credit facts up to one requested year per concrete plan."""
    years = tuple(getattr(query_spec, "years", ()))
    semesters = tuple(getattr(query_spec, "semesters", ()))
    group_by = tuple(getattr(query_spec, "group_by", ()))
    if len(years) != 1 or semesters or "semester" in group_by or not results:
        return None

    requested_year = years[0]
    grouped: dict[
        tuple[str, str, int],
        list[tuple[EvidenceExecutionResult, tuple[Mapping[str, Any], ...] | None]],
    ] = {}
    for result in results:
        scope = result.effective_scope
        program = getattr(scope, "program", None)
        plans = tuple(getattr(scope, "plans", ()))
        result_years = tuple(getattr(scope, "years", ()))
        result_semesters = tuple(getattr(scope, "semesters", ()))
        if (
            not isinstance(program, str)
            or not program.strip()
            or len(plans) != 1
            or not isinstance(plans[0], str)
            or not plans[0].strip()
            or len(result_years) != 1
            or result_years[0] != requested_year
            or len(result_semesters) != 1
        ):
            return None

        records: tuple[Mapping[str, Any], ...] | None
        if result.status == "insufficient_evidence":
            records = None
        elif result.status == "valid_empty":
            records = ()
        elif result.status == "complete":
            records = _payload_records(result, "components")
            if records is None:
                return None
        else:
            return None
        key = (program, plans[0], requested_year)
        grouped.setdefault(key, []).append((result, records))

    claims: list[GroundedClaim] = []
    for (_program, _plan, _year), entries in grouped.items():
        representative, _ = entries[0]
        rollup_scope = replace(
            representative.effective_scope,
            semesters=(),
            group_by=tuple(
                axis
                for axis in representative.effective_scope.group_by
                if axis != "semester"
            ),
        )
        if any(records is None for _, records in entries):
            claims.append(
                _claim(
                    "sum_credits",
                    representative,
                    status="insufficient_evidence",
                    effective_scope=rollup_scope,
                )
            )
            continue

        components = tuple(
            component
            for _, records in entries
            for component in records or ()
        )
        try:
            aggregate = aggregate_sum_credits(components, evidence_complete=True)
        except (TypeError, ValueError, OverflowError):
            claims.append(
                _claim(
                    "sum_credits",
                    representative,
                    status="insufficient_evidence",
                    effective_scope=rollup_scope,
                )
            )
            continue
        claims.append(
            _claim(
                "sum_credits",
                representative,
                value=aggregate.value,
                evidence=aggregate,
                provenance=_provenance_from_records(aggregate.components),
                status=aggregate.status,
                effective_scope=rollup_scope,
            )
        )
    return tuple(claims)


def _claim_for_placement_operation(
    operation: str,
    result: EvidenceExecutionResult,
) -> GroundedClaim:
    records = _payload_records(result, "courses")
    if records is None:
        return _claim(operation, result, status="insufficient_evidence")
    if operation == "earliest":
        try:
            aggregate = aggregate_earliest(
                records,
                evidence_complete=result.status == "complete",
            )
        except (TypeError, ValueError, OverflowError):
            return _claim(operation, result, status="insufficient_evidence")
        return _claim(
            operation,
            result,
            value=aggregate,
            evidence=aggregate,
            provenance=tuple(
                reference
                for partition in aggregate.partitions
                for reference in partition.provenance
            ),
            status=aggregate.status,
        )
    return _claim(
        operation,
        result,
        value=records if result.status == "complete" else None,
        evidence=records if result.status == "complete" else None,
        provenance=_provenance_from_records(records),
    )


def _claim_for_prerequisite_operation(
    operation: str,
    result: EvidenceExecutionResult,
) -> GroundedClaim:
    records = _payload_records(result)
    if records is None:
        return _claim(operation, result, status="insufficient_evidence")
    if (
        result.status == "valid_empty"
        and result.primitive_state == "explicit_none"
        and not result.planned_request.positive_prerequisite_collection
    ):
        provenance = _provenance_from_records(records)
        if (
            not records
            or not provenance
            or any(
                record.get("prerequisite_state") != "explicit_none"
                for record in records
            )
        ):
            return _claim(operation, result, status="insufficient_evidence")
        return _claim(
            operation,
            result,
            value=records,
            evidence=records,
            provenance=provenance,
        )
    return _claim(
        operation,
        result,
        value=records if result.status == "complete" else None,
        evidence=records if result.status == "complete" else None,
        provenance=_provenance_from_records(records),
    )


def _normalized_prerequisite_title(record: Mapping[str, Any]) -> str | None:
    for field in (
        "prerequisite_name_en",
        "prerequisite_name_th",
        "name_en",
        "name_th",
        "course_name",
    ):
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return " ".join(value.casefold().split())
    return None


def _prerequisite_semantic_fingerprint(
    records: Iterable[Mapping[str, Any]],
) -> tuple[tuple[Any, ...], ...] | None:
    fingerprint: list[tuple[Any, ...]] = []
    for record in records:
        if not isinstance(record, Mapping):
            return None
        alternatives = record.get("alternative_courses")
        if isinstance(alternatives, Mapping):
            alternatives = (alternatives,)
        if isinstance(alternatives, (list, tuple)) and alternatives:
            member_titles: list[str] = []
            for member in alternatives:
                if not isinstance(member, Mapping):
                    return None
                title = _normalized_prerequisite_title(member)
                if title is None:
                    return None
                member_titles.append(title)
            group = record.get("alternative_group")
            minimum = record.get("minimum_choices")
            maximum = record.get("maximum_choices")
            if isinstance(group, Mapping):
                minimum = group.get("minimum_choices", minimum)
                maximum = group.get("maximum_choices", maximum)
            fingerprint.append(
                (
                    "alternative",
                    record.get("requirement_type"),
                    minimum,
                    maximum,
                    tuple(member_titles),
                )
            )
            continue
        title = _normalized_prerequisite_title(record)
        if title is None:
            return None
        fingerprint.append(("direct", record.get("requirement_type"), title))
    return tuple(fingerprint)


def _merge_consensus_prerequisites(
    record_sets: tuple[tuple[Mapping[str, Any], ...], ...],
) -> tuple[Mapping[str, Any], ...]:
    if not record_sets:
        return ()
    merged: list[dict[str, Any]] = [dict(record) for record in record_sets[0]]
    for index, record in enumerate(merged):
        references: list[Any] = []
        for records in record_sets:
            provenance = records[index].get("provenance", ())
            if isinstance(provenance, Mapping) or isinstance(provenance, str):
                provenance = (provenance,)
            if isinstance(provenance, (list, tuple)):
                for reference in provenance:
                    if reference not in references:
                        references.append(reference)
        record["provenance"] = tuple(references)
    return tuple(merged)


def _consensus_prerequisite_grounded_answer(
    db_path: str | Path,
    spec: Any,
    resolution: ResolutionOutcome,
    context: QueryContext | None,
) -> Any | None:
    """Answer only unanimous, program-free exact prerequisite references."""
    if spec.operations != ("prerequisite",):
        return None
    # A literal course title is an exact entity reference, not permission to
    # answer from whichever program happens to share that title. Keep the
    # program clarification produced by canonical resolution when scope is
    # absent; consensus across curricula would silently choose factual scope.
    if getattr(spec, "course_name", None) is not None:
        return None
    if getattr(spec, "program", None) is not None:
        return None
    if context is not None and context.program is not None:
        return None
    if len(resolution.course_references) != 1:
        return None
    reference = resolution.course_references[0]
    if spec.course_name is not None:
        candidates = exact_course_candidates(
            db_path,
            course_name=spec.course_name,
            exact_title=True,
        )
    else:
        candidates = list(reference.candidates)
    programs = {
        str(candidate.get("program"))
        for candidate in candidates
        if candidate.get("program") not in (None, "")
    }
    if len(programs) < 2:
        return None

    states: list[Mapping[str, Any]] = []
    fingerprints: list[tuple[tuple[Any, ...], ...] | None] = []
    for candidate in candidates:
        course_id = candidate.get("course_id")
        if isinstance(course_id, bool) or not isinstance(course_id, int):
            return None
        state = prerequisite_state(db_path, course_id)
        states.append(state)
        if state.get("state") == "explicit_none":
            fingerprints.append((('explicit_none',),))
        elif state.get("state") == "required":
            fingerprints.append(
                _prerequisite_semantic_fingerprint(state.get("records", ()))
            )
        else:
            fingerprints.append(None)
    if not fingerprints or any(fingerprint is None for fingerprint in fingerprints):
        return None
    first = fingerprints[0]
    if any(fingerprint != first for fingerprint in fingerprints[1:]):
        return None
    if first == (('explicit_none',),):
        merged_provenance: list[Any] = []
        for state in states:
            for reference in state.get("provenance", ()):
                if reference not in merged_provenance:
                    merged_provenance.append(reference)
        merged = (
            {
                "prerequisite_state": "explicit_none",
                "prerequisite_text": states[0].get("prerequisite_text"),
                "provenance": tuple(merged_provenance),
            },
        )
    else:
        merged = _merge_consensus_prerequisites(
            tuple(tuple(state.get("records", ())) for state in states)
        )
    provenance_values: list[Any] = list(_provenance_from_records(merged))
    for candidate in candidates:
        candidate_provenance = candidate.get("provenance", ())
        if isinstance(candidate_provenance, Mapping) or isinstance(candidate_provenance, str):
            candidate_provenance = (candidate_provenance,)
        if isinstance(candidate_provenance, (list, tuple)):
            for reference in candidate_provenance:
                if reference not in provenance_values:
                    provenance_values.append(reference)
    provenance = tuple(provenance_values)
    claim_status = "valid_empty" if first == (('explicit_none',),) else "complete"
    claim = GroundedClaim(
        claim_id="prerequisite_consensus",
        operation="prerequisite",
        status=claim_status,
        kind="deterministic_fact",
        value=merged,
        evidence=merged,
        provenance=provenance,
    )
    return compose_grounded_answer(composed_claims=(claim,))


def _claim_for_description_operation(
    operation: str,
    result: EvidenceExecutionResult,
) -> GroundedClaim:
    records = _payload_records(result)
    if records is None:
        return _claim(operation, result, status="insufficient_evidence", kind="grounded_summary")
    return _claim(
        operation,
        result,
        value=records if result.status == "complete" else None,
        evidence=records if result.status == "complete" else None,
        provenance=_provenance_from_records(records),
        kind="grounded_summary",
    )


def _claim_for_judgement_operation(
    operation: str,
    result: EvidenceExecutionResult,
    relation_results: tuple[EvidenceExecutionResult, ...],
    credit_results: tuple[EvidenceExecutionResult, ...],
    prerequisite_results: tuple[EvidenceExecutionResult, ...] = (),
    require_prerequisite: bool = False,
) -> GroundedClaim:
    if operation == "preference":
        if result.status != "complete":
            return _claim(operation, result, status=result.status, kind="grounded_summary")
        payload = result.payload
        if not isinstance(payload, ConstrainedTopicRetrievalResult):
            return _claim(operation, result, status="insufficient_evidence", kind="grounded_summary")
        candidates = _topic_candidates(result)
        if candidates is None:
            return _claim(operation, result, status="insufficient_evidence", kind="grounded_summary")
        if require_prerequisite:
            candidates = _preference_options_with_prerequisites(
                result,
                prerequisite_results,
            )
            if candidates is None:
                return _claim(
                    operation,
                    result,
                    status="insufficient_evidence",
                    kind="grounded_summary",
                )
        evidence = evaluate_preference(candidates)
        claim_status = (
            "complete" if evidence.status == "supported" else evidence.status
        )
        return _claim(
            operation,
            result,
            value=evidence,
            evidence=evidence,
            provenance=evidence.provenance,
            kind="grounded_summary",
            status=claim_status,
        )

    if operation == "quantity":
        facts: dict[str, Any] = {}
        aggregate = _course_set_aggregate(result)
        if aggregate is not None:
            facts["count"] = aggregate
        evidence = evaluate_quantity(facts)
    else:
        facts = {}
        course_aggregate = _course_set_aggregate(result)
        if course_aggregate is not None:
            facts["count"] = course_aggregate
            records = (
                _topic_candidates(result)
                if result.kind == "topic_matches"
                else _payload_records(result, "courses")
            )
            if records is not None:
                try:
                    facts["required_load"] = aggregate_required_load(
                        records,
                        evidence_complete=result.status == "complete",
                    )
                except (TypeError, ValueError, OverflowError):
                    pass
        credit_result = next(
            (
                candidate
                for candidate in credit_results
                if candidate.effective_scope == result.effective_scope
            ),
            None,
        )
        if credit_result is not None:
            credit_aggregate = _credit_aggregate(credit_result)
            if credit_aggregate is not None:
                facts["credits"] = credit_aggregate
        evidence = evaluate_workload(facts)
    return _claim(
        operation,
        result,
        value=evidence,
        evidence=evidence,
        provenance=evidence.provenance,
        status=evidence.status,
    )


def _comparison_claims(
    result: EvidenceExecutionResult,
) -> tuple[GroundedClaim, ...]:
    payload = result.payload
    if not isinstance(payload, ComparisonAggregation):
        return ()
    return _comparison_claim_from_payload(
        payload,
        effective_scope=result.effective_scope,
        provenance_required=result.planned_request.provenance_required,
    )


def _comparison_claim_from_payload(
    payload: ComparisonAggregation | PlanComparisonAggregation,
    *,
    effective_scope: Any = None,
    provenance_required: bool = True,
) -> tuple[GroundedClaim, ...]:
    """Compose one comparison payload while preserving its typed operands."""
    provenance = _provenance_from_value(payload)
    status = payload.status
    if status == "complete" and provenance_required and not provenance:
        status = "insufficient_evidence"
    value = payload if status != "insufficient_evidence" else None
    evidence = payload if status != "insufficient_evidence" else None
    retained_provenance = provenance if status != "insufficient_evidence" else ()
    return (
        GroundedClaim(
            claim_id="pending",
            operation="compare",
            effective_scope=effective_scope,
            status=status,
            kind="deterministic_fact",
            value=value,
            evidence=evidence,
            provenance=retained_provenance,
        ),
    )


def _whole_plan_comparison_payload(
    query_spec: Any,
    bundle: EvidenceBundle,
) -> PlanComparisonAggregation | None:
    """Build a typed comparison for exactly two complete plan partitions."""
    operations = tuple(getattr(query_spec, "operations", ()))
    if "compare" not in operations:
        return None
    if tuple(getattr(query_spec, "group_by", ())) != ("plan",):
        return None
    if getattr(query_spec, "course_codes", ()) or getattr(query_spec, "course_name", None):
        return None

    plans = tuple(getattr(query_spec, "plans", ()))
    if len(plans) != 2 or len(set(plans)) != 2:
        return None

    def results_by_plan(
        kind: str,
    ) -> dict[str, tuple[EvidenceExecutionResult, tuple[Mapping[str, Any], ...]]] | None:
        grouped: dict[
            str, tuple[EvidenceExecutionResult, tuple[Mapping[str, Any], ...]]
        ] = {}
        for result in _execution_results(bundle, kind):
            if result.status != "complete":
                return None
            scope = result.effective_scope
            result_plans = tuple(getattr(scope, "plans", ()))
            if len(result_plans) != 1 or result_plans[0] not in plans:
                return None
            plan = result_plans[0]
            if plan in grouped:
                return None
            records = _payload_records(result, "courses")
            if records is None:
                return None
            grouped[plan] = (result, records)
        if set(grouped) != set(plans):
            return None
        return grouped

    course_results = results_by_plan("course_set")
    placement_results = results_by_plan("placement_facts")
    if course_results is None or placement_results is None:
        return None

    inputs: list[PlanComparisonInput] = []
    for plan in plans:
        course_result, _course_records = course_results[plan]
        _placement_result, placement_records = placement_results[plan]
        course_aggregate = _course_set_aggregate(course_result)
        if course_aggregate is None or course_aggregate.status != "complete":
            return None
        inputs.append(
            PlanComparisonInput(
                plan=plan,
                course_set=course_aggregate,
                placements=placement_records,
                placements_complete=True,
            )
        )

    try:
        return aggregate_plan_comparison(inputs[0], inputs[1])
    except (TypeError, ValueError, OverflowError):
        return None


def _earliest_comparison_payload(
    query_spec: Any,
    bundle: EvidenceBundle,
) -> ComparisonAggregation | None:
    """Build the frozen two-plan earliest comparison from placement evidence."""
    operations = set(getattr(query_spec, "operations", ()))
    if not {"placement", "earliest", "compare"}.issubset(operations):
        return None
    if tuple(getattr(query_spec, "group_by", ())) != ("plan",):
        return None
    course_codes = tuple(getattr(query_spec, "course_codes", ()))
    if len(course_codes) != 1 or not isinstance(course_codes[0], str):
        return None
    requested_course_code = course_codes[0].strip()
    if not requested_course_code:
        return None

    grouped_records: dict[str, list[Mapping[str, Any]]] = {}
    grouped_statuses: dict[str, list[str]] = {}
    group_identity: tuple[str, str] | None = None
    placement_results = _execution_results(bundle, "placement_facts")
    if not placement_results:
        return None

    for result in placement_results:
        scope = result.effective_scope
        program = getattr(scope, "program", None)
        plans = tuple(getattr(scope, "plans", ()))
        if not isinstance(program, str) or not program.strip() or len(plans) != 1:
            return None
        plan = plans[0]
        if not isinstance(plan, str) or not plan.strip():
            return None

        request_targets = tuple(
            getattr(result.planned_request, "course_targets", ())
            or getattr(scope, "course_targets", ())
        )
        if len(request_targets) != 1:
            return None
        target_identity = _logical_target_key(request_targets[0])
        if (
            target_identity is None
            or target_identity[0] != program
            or target_identity[1] != requested_course_code
        ):
            return None
        if group_identity is None:
            group_identity = target_identity
        elif target_identity != group_identity:
            return None

        records = _payload_records(result, "courses")
        if records is None or result.status != "complete":
            return None
        for record in records:
            if _logical_target_key(record) != group_identity:
                return None
            record_plan = record.get("plan_key", record.get("plan"))
            if record_plan is not None and record_plan != plan:
                return None
            provenance = record.get("provenance")
            if (
                not isinstance(provenance, (list, tuple))
                or not provenance
                or any(not isinstance(reference, Mapping) for reference in provenance)
            ):
                return None
        grouped_records.setdefault(plan, []).extend(records)
        grouped_statuses.setdefault(plan, []).append(result.status)

    if group_identity is None or len(grouped_records) != 2:
        return None
    operands: dict[str, EarliestAggregation] = {}
    for plan, records in grouped_records.items():
        if not records or any(status != "complete" for status in grouped_statuses[plan]):
            return None
        try:
            aggregate = aggregate_earliest(records, evidence_complete=True)
        except (TypeError, ValueError, OverflowError):
            return None
        if (
            aggregate.status != "complete"
            or aggregate.value is None
            or len(aggregate.partitions) != 1
            or not _provenance_from_value(aggregate)
        ):
            return None
        operands[plan] = aggregate

    plan_scope = getattr(getattr(bundle, "plan", None), "scope", None)
    plan_order = tuple(
        plan
        for plan in getattr(plan_scope, "plans", ())
        if plan in operands
    )
    plan_order += tuple(plan for plan in operands if plan not in plan_order)
    if len(plan_order) != 2:
        return None
    try:
        return compare_aggregates(operands[plan_order[0]], operands[plan_order[1]])
    except (TypeError, ValueError, OverflowError):
        return None


def _placement_comparison_payloads(
    query_spec: Any,
    bundle: EvidenceBundle,
) -> tuple[ComparisonAggregation, ...]:
    """Compare complete exact-course placement operands within safe partitions."""
    operations = set(getattr(query_spec, "operations", ()))
    if (
        "compare" not in operations
        or "placement" not in operations
        or "earliest" in operations
    ):
        return ()

    placement_results = _execution_results(bundle, "placement_facts")
    if not placement_results:
        return ()

    requested_targets: set[tuple[str, str]] = set()
    grouped: dict[tuple[str, str], dict[tuple[str, str], list[Mapping[str, Any]]]] = {}
    for result in placement_results:
        if result.status != "complete":
            return ()
        scope = result.effective_scope
        program = getattr(scope, "program", None)
        plans = tuple(getattr(scope, "plans", ()))
        if not isinstance(program, str) or not program.strip() or len(plans) != 1:
            return ()
        plan = plans[0]
        if not isinstance(plan, str) or not plan.strip():
            return ()

        for target in tuple(
            getattr(result.planned_request, "course_targets", ())
            or getattr(scope, "course_targets", ())
        ):
            target_key = _logical_target_key(target)
            if target_key is not None:
                requested_targets.add(target_key)

        records = _payload_records(result, "courses")
        if not records:
            return ()
        identities = {_logical_target_key(record) for record in records}
        if None in identities or len(identities) != 1:
            return ()
        identity = next(iter(identities))
        if identity[0] != program or identity not in requested_targets:
            return ()
        for record in records:
            record_plan = record.get("plan_key", record.get("plan"))
            if record.get("program") != program or record_plan != plan:
                return ()
        grouped.setdefault((program, plan), {}).setdefault(identity, []).extend(records)

    if not requested_targets or len({program for program, _ in requested_targets}) != 1:
        return ()

    target_count = len(requested_targets)
    requested_plans = tuple(getattr(query_spec, "plans", ()))
    if target_count == 2:
        if any(set(targets) != requested_targets for targets in grouped.values()):
            return ()
        if len(grouped) == 0:
            return ()
    elif target_count == 1:
        if len(requested_plans) != 2 or set(requested_plans) != {
            plan for _, plan in grouped
        }:
            return ()
        if any(set(targets) != requested_targets for targets in grouped.values()):
            return ()
    else:
        return ()

    ordered_targets = tuple(sorted(requested_targets))
    if target_count == 1:
        plans = sorted(grouped)
        if len(plans) != 2:
            return ()
        target = ordered_targets[0]
        left_records = grouped[plans[0]][target]
        right_records = grouped[plans[1]][target]
        try:
            left = aggregate_earliest(left_records, evidence_complete=True)
            right = aggregate_earliest(right_records, evidence_complete=True)
            comparison = compare_aggregates(left, right)
        except (TypeError, ValueError, OverflowError):
            return ()
        if (
            left.status != "complete"
            or right.status != "complete"
            or left.value is None
            or right.value is None
            or len(left.partitions) != 1
            or len(right.partitions) != 1
            or not _provenance_from_value(left)
            or not _provenance_from_value(right)
        ):
            return ()
        return (comparison,)

    payloads: list[ComparisonAggregation] = []
    for target_records in grouped.values():
        aggregates: dict[tuple[str, str], EarliestAggregation] = {}
        for target in ordered_targets:
            try:
                aggregate = aggregate_earliest(
                    target_records[target],
                    evidence_complete=True,
                )
            except (TypeError, ValueError, OverflowError):
                return ()
            if (
                aggregate.status != "complete"
                or aggregate.value is None
                or len(aggregate.partitions) != 1
                or not _provenance_from_value(aggregate)
            ):
                return ()
            aggregates[target] = aggregate
        try:
            payloads.append(
                compare_aggregates(
                    aggregates[ordered_targets[0]],
                    aggregates[ordered_targets[1]],
                )
            )
        except (TypeError, ValueError, OverflowError):
            return ()
    return tuple(payloads)


def _missing_comparison_claim() -> GroundedClaim:
    """Represent an unexecuted comparison without inventing a relation."""
    return GroundedClaim(
        claim_id="pending",
        operation="compare",
        status="insufficient_evidence",
    )


def _similarity_provenance(
    evidence: SimilarityEvidence,
) -> tuple[Any, ...] | None:
    """Validate and retain first-seen provenance from both pair sides."""
    references: list[Any] = []
    for pair in evidence.pairs:
        for side in (pair.left, pair.right):
            provenance = side.get("provenance")
            if not isinstance(provenance, (list, tuple)) or not provenance:
                return None
            for reference in provenance:
                if not isinstance(reference, Mapping):
                    return None
                if reference not in references:
                    references.append(reference)
    return tuple(references)


def _similarity_claim(
    evidence: SimilarityEvidence | None,
    bundle: EvidenceBundle,
    request_ids: tuple[str, str] | None,
    *,
    selected_plan: str | None = None,
) -> GroundedClaim:
    """Wrap one exact similarity result without recomputing any evidence."""
    if not isinstance(evidence, SimilarityEvidence):
        return GroundedClaim(
            claim_id="pending",
            operation="similarity",
            status="insufficient_evidence",
        )

    scopes: list[Any] = []
    if request_ids is not None:
        for result in bundle.results:
            if (
                result.request_id in request_ids
                and result.kind == "description_evidence"
                and (
                    selected_plan is None
                    or tuple(getattr(result.effective_scope, "plans", ()))
                    == (selected_plan,)
                )
            ):
                if (
                    result.effective_scope is not None
                    and result.effective_scope not in scopes
                ):
                    scopes.append(result.effective_scope)
    pair_plans = {
        pair.partition.get("plan")
        for pair in evidence.pairs
        if isinstance(pair.partition, Mapping)
    }
    if len(pair_plans) == 1:
        matching_scopes = [
            scope
            for scope in scopes
            if tuple(getattr(scope, "plans", ())) == (next(iter(pair_plans)),)
        ]
        if matching_scopes:
            scopes = matching_scopes
    effective_scope = scopes[0] if len(scopes) == 1 else None
    provenance = _similarity_provenance(evidence)
    if provenance is None:
        return GroundedClaim(
            claim_id="pending",
            operation="similarity",
            effective_scope=effective_scope,
            status="insufficient_evidence",
        )
    return GroundedClaim(
        claim_id="pending",
        operation="similarity",
        effective_scope=effective_scope,
        status=evidence.status,
        kind="deterministic_fact",
        value=evidence,
        evidence=evidence,
        provenance=provenance,
    )


def _logical_target_key(target: Any) -> tuple[str, str] | None:
    if not isinstance(target, Mapping):
        return None
    program = target.get("program")
    course_code = target.get("course_code")
    if (
        not isinstance(program, str)
        or not program.strip()
        or not isinstance(course_code, str)
        or not course_code.strip()
    ):
        return None
    return program, course_code


def _resolved_logical_targets(outcome: ResolutionOutcome) -> set[tuple[str, str]]:
    targets: set[tuple[str, str]] = set()
    for reference in outcome.course_references:
        for candidate in reference.candidates:
            key = _logical_target_key(candidate)
            if key is not None:
                targets.add(key)
    return targets


def _exact_no_prerequisite_existence_claim(
    query_spec: Any,
    resolution: ResolutionOutcome,
    bundle: EvidenceBundle,
) -> GroundedClaim | None:
    """Pair exact-course identity with a verified explicit-none prerequisite fact.

    ``valid_empty`` remains the correct status for the prerequisite fact itself.
    For an exact course lookup, a canonical identity claim also establishes that
    the current question has a complete answer; this must never run for a
    prerequisite collection or an unknown prerequisite state.
    """
    if (
        tuple(getattr(query_spec, "operations", ())) != ("prerequisite",)
        or not (
            tuple(getattr(query_spec, "course_codes", ()))
            or getattr(query_spec, "course_name", None) is not None
        )
        or resolution.action != "answer"
    ):
        return None
    prerequisite_results = _execution_results(bundle, "prerequisite_facts")
    if not prerequisite_results or any(
        result.planned_request.positive_prerequisite_collection
        or result.status != "valid_empty"
        or result.primitive_state != "explicit_none"
        for result in prerequisite_results
    ):
        return None
    for result in prerequisite_results:
        records = _payload_records(result)
        if not records or any(
            record.get("prerequisite_state") != "explicit_none"
            or not _provenance_from_records((record,))
            for record in records
        ):
            return None

    resolved_targets = _resolved_logical_targets(resolution)
    if len(resolved_targets) != 1:
        return None
    identity_result = _identity_result(resolution)
    identity_answer = compose_grounded_answer(identity_result=identity_result)
    if identity_answer.status != "answer" or len(identity_answer.claims) != 1:
        return None
    identity_claim = identity_answer.claims[0]
    identity_targets = {
        key
        for key in (
            _logical_target_key(identity)
            for identity in identity_claim.value
            if isinstance(identity, Mapping)
        )
        if key is not None
    }
    if identity_targets != resolved_targets or not identity_claim.provenance:
        return None
    return GroundedClaim(
        claim_id="pending",
        operation="existence",
        status="complete",
        kind="deterministic_fact",
        value=True,
        evidence=identity_claim.evidence,
        provenance=identity_claim.provenance,
    )


def _select_similarity_request_ids(
    plan: Any,
    outcome: ResolutionOutcome,
) -> tuple[str, str] | None:
    """Select exactly the planner-owned description requests for similarity."""
    requests = tuple(
        request
        for request in getattr(plan, "requests", ())
        if getattr(request, "kind", None) == "description_evidence"
        and isinstance(getattr(request, "request_id", None), str)
        and request.request_id.startswith("similarity_description_")
    )
    if len(requests) != 2:
        return None
    targets: list[tuple[str, str]] = []
    for request in requests:
        request_targets = getattr(request, "course_targets", ())
        if len(request_targets) != 1:
            return None
        key = _logical_target_key(request_targets[0])
        if key is None or key in targets:
            return None
        targets.append(key)
    scope_targets = tuple(
        key
        for key in (
            _logical_target_key(target)
            for target in getattr(getattr(plan, "scope", None), "course_targets", ())
        )
        if key is not None
    )
    resolved_targets = _resolved_logical_targets(outcome)
    if (
        len(scope_targets) != 2
        or len(set(scope_targets)) != 2
        or tuple(targets) != scope_targets
        or set(targets) != resolved_targets
    ):
        return None
    return requests[0].request_id, requests[1].request_id


def _compose_evidence_claims(
    query_spec: Any,
    bundle: EvidenceBundle,
    *,
    resolution: ResolutionOutcome | None = None,
    similarity_evidence: SimilarityEvidence | None = None,
    similarity_request_ids: tuple[str, str] | None = None,
    selected_plan: str | None = None,
) -> tuple[GroundedClaim, ...]:
    """Adapt an EvidenceBundle into ordered, partition-preserving typed claims.

    This is intentionally private until the 4I.4d runtime integration.  It
    performs no database/vector work and delegates all derived facts to the
    frozen aggregation/judgement contracts.
    """
    if not isinstance(bundle, EvidenceBundle) or query_spec is None:
        return ()
    operations = tuple(getattr(query_spec, "operations", ()))
    topic_collection_description = bool(
        getattr(query_spec, "topic", None) is not None
        and not getattr(query_spec, "course_codes", ())
        and getattr(query_spec, "course_name", None) is None
        and "describe" in operations
        and not ({"list", "count", "sum_credits", "existence"} & set(operations))
    )
    judgement = getattr(query_spec, "judgement", "none")
    if judgement in {"quantity", "workload", "preference"} and judgement not in operations:
        operations += (judgement,)
    relation_results = _relation_results(query_spec, bundle)
    credit_results = _execution_results(bundle, "credit_facts")
    greatest_credit_claims = _greatest_credit_claims(
        query_spec,
        bundle,
        credit_results,
    )
    coarse_credit_claims = _coarse_year_credit_claims(query_spec, credit_results)
    placement_results = _execution_results(bundle, "placement_facts")
    prerequisite_results = _execution_results(bundle, "prerequisite_facts")
    description_results = _execution_results(bundle, "description_evidence")
    comparison_results = tuple(
        result
        for result in bundle.results
        if isinstance(result.payload, ComparisonAggregation)
    )
    generated_placement_comparisons: tuple[ComparisonAggregation, ...] = ()
    generated_plan_comparison: PlanComparisonAggregation | None = None
    generated_comparison = None
    if not comparison_results:
        generated_placement_comparisons = _placement_comparison_payloads(
            query_spec,
            bundle,
        )
    if not comparison_results and not generated_placement_comparisons:
        generated_plan_comparison = _whole_plan_comparison_payload(
            query_spec,
            bundle,
        )
    if (
        not comparison_results
        and not generated_placement_comparisons
        and generated_plan_comparison is None
    ):
        generated_comparison = _earliest_comparison_payload(query_spec, bundle)
    course_cache: dict[int, CourseSetAggregation | None] = {}
    claims: list[GroundedClaim] = []
    preference_requires_prerequisite = (
        judgement == "preference"
        and "prerequisite" in operations
        and getattr(query_spec, "topic", None) is not None
    )
    positive_prerequisite_collection = any(
        result.kind == "prerequisite_facts"
        and result.planned_request.positive_prerequisite_collection
        for result in prerequisite_results
    )
    for operation in operations:
        if operation in {"list", "count", "existence"}:
            if operation == "list" and positive_prerequisite_collection:
                # The prerequisite-dependent claim below is the answer set.
                # The unfiltered course_set remains internal dependency
                # evidence and must not be exposed or retained as a list.
                continue
            for result in relation_results:
                claim = _claim_for_relation_operation(operation, result, course_cache)
                if claim is not None:
                    claims.append(claim)
        elif operation == "sum_credits":
            if greatest_credit_claims is not None:
                claims.extend(greatest_credit_claims)
            elif coarse_credit_claims is not None:
                claims.extend(coarse_credit_claims)
            else:
                claims.extend(
                    _claim_for_credit_operation(operation, result)
                    for result in credit_results
                )
        elif operation in {"placement", "earliest"}:
            claims.extend(
                _claim_for_placement_operation(operation, result)
                for result in placement_results
            )
        elif operation == "prerequisite":
            if preference_requires_prerequisite:
                continue
            course_results = _execution_results(bundle, "course_set")
            scope_prerequisites = tuple(
                result
                for result in prerequisite_results
                if _scope_prerequisite_dependency(result, course_results)
            )
            if scope_prerequisites:
                claims.extend(
                    _claim_for_scope_prerequisite(result, course_results)
                    for result in scope_prerequisites
                )
                claims.extend(
                    _claim_for_prerequisite_operation(operation, result)
                    for result in prerequisite_results
                    if result not in scope_prerequisites
                )
            else:
                claims.extend(
                    _claim_for_prerequisite_operation(operation, result)
                    for result in prerequisite_results
                )
        elif operation == "describe":
            if topic_collection_description:
                claims.extend(
                    claim
                    for result in _execution_results(bundle, "topic_matches")
                    if (
                        claim := _claim_for_relation_operation(
                            "list", result, course_cache
                        )
                    ) is not None
                )
            else:
                claims.extend(
                    _claim_for_description_operation(operation, result)
                    for result in description_results
                )
        elif operation in {"quantity", "workload", "preference"}:
            judgement_results = (
                relation_results
                if operation != "preference"
                else _execution_results(bundle, "topic_matches")
            )
            for result in judgement_results:
                claims.append(
                    _claim_for_judgement_operation(
                        operation,
                        result,
                        relation_results,
                        credit_results,
                        prerequisite_results,
                        preference_requires_prerequisite,
                    )
                )
        elif operation == "compare":
            if comparison_results:
                for result in comparison_results:
                    claims.extend(_comparison_claims(result))
            elif generated_placement_comparisons:
                for payload in generated_placement_comparisons:
                    claims.extend(
                        _comparison_claim_from_payload(
                            payload,
                            effective_scope=None,
                        )
                    )
            elif generated_plan_comparison is not None:
                claims.extend(
                    _comparison_claim_from_payload(
                        generated_plan_comparison,
                        effective_scope=None,
                    )
                )
            elif generated_comparison is not None:
                claims.extend(
                    _comparison_claim_from_payload(
                        generated_comparison,
                        effective_scope=None,
                    )
                )
            elif greatest_credit_claims is not None:
                # The supported greatest-credit operation is represented by
                # its winning sum_credits claims, not a synthetic pairwise
                # ComparisonAggregation.
                pass
            else:
                claims.append(_missing_comparison_claim())
        elif operation == "similarity":
            claims.append(
                _similarity_claim(
                    similarity_evidence,
                    bundle,
                    similarity_request_ids,
                    selected_plan=selected_plan,
                )
            )

    if resolution is not None:
        existence_claim = _exact_no_prerequisite_existence_claim(
            query_spec,
            resolution,
            bundle,
        )
        if existence_claim is not None:
            claims.insert(0, existence_claim)

    numbered: list[GroundedClaim] = []
    for index, claim in enumerate(claims, start=1):
        numbered.append(
            GroundedClaim(
                claim_id=f"claim_{index:03d}",
                operation=claim.operation,
                effective_scope=claim.effective_scope,
                status=claim.status,
                kind=claim.kind,
                value=claim.value,
                evidence=claim.evidence,
                provenance=claim.provenance,
            )
        )
    return tuple(numbered)


def _blocked_result(outcome: ResolutionOutcome) -> dict[str, Any]:
    return {
        "status": outcome.action,
        "action": outcome.action,
        "blocking_ambiguity": outcome.blocking_ambiguity,
        "context_conflicts": outcome.context_conflicts,
        "resolved_program": outcome.resolved_program,
        "resolved_plans": outcome.resolved_plans,
        "course_references": [
            {
                "reference_type": reference.reference_type,
                "reference": reference.reference,
                "candidates": [dict(candidate) for candidate in reference.candidates],
            }
            for reference in outcome.course_references
        ],
    }


def _identity_result(
    outcome: ResolutionOutcome,
    *,
    operation: str = "identity",
) -> dict[str, Any]:
    identities: list[dict[str, Any]] = []
    for reference in outcome.course_references:
        for candidate in reference.candidates:
            identities.append(
                {
                    "reference_type": reference.reference_type,
                    "reference": reference.reference,
                    **dict(candidate),
                }
            )
    return {
        "status": "answer",
        "action": "answer",
        "operation": operation,
        "resolved_program": outcome.resolved_program,
        "resolved_plans": outcome.resolved_plans,
        "identities": identities,
    }


def _has_unambiguous_recovery_program(
    program: Any,
    edition_keys: tuple[Any, ...],
    catalog_key: Any,
) -> bool:
    """Check that a recovery attempt would run under an authoritative program.

    NL-2 pre-guard rescue may run only when the program scope is already
    unambiguous: either the program resolves to a single edition, or an
    edition catalog is already selected. An ambiguous multi-edition program
    without a catalog stays on the deterministic clarify path with zero
    model calls. This predicate never invents scope; it only authorizes the
    interpreter to propose a literal title inside already-fixed scope.
    """
    if not isinstance(program, str) or not program.strip():
        return False
    if len(tuple(edition_keys)) == 1:
        return True
    return isinstance(catalog_key, str) and bool(catalog_key.strip())


def _attempt_structural_recovery(
    db_path: str | Path,
    base_spec: Any,
    base_resolution: ResolutionOutcome,
    base_completeness: StructuredParseCompleteness,
    question: str,
    context: QueryContext | None,
    resolution_context: QueryContext | None,
    intent_model_callable: Callable[[str], str],
) -> tuple[str, Any, ResolutionOutcome, StructuredParseCompleteness] | None:
    """Run one bounded query-structure recovery attempt.

    Pure routing/compilation logic shared by the existing post-guard path
    and the NL-2 pre-guard rescue seam. Returns None when the gate declines
    (caller keeps the existing deterministic flow), ("blocked", ...) when
    the compiled request hits a deterministic blocking outcome, ("failed",)
    for any other fail-closed outcome, and ("recovered", ...) with the
    compiled request otherwise. The interpreter proposes language only;
    facts still come from deterministic resolution afterward.
    """
    if not _should_use_query_structure_interpreter(
        base_spec,
        base_resolution,
        question,
        base_completeness,
        context,
    ):
        return None
    try:
        structure = interpret_question_intent(
            question,
            intent_model_callable,
            proposal_kind="query_structure",
        )
        if (
            structure.predicate is None
            and not getattr(base_spec, "course_codes", ())
            and getattr(base_spec, "course_name", None) is None
            and not tuple(getattr(base_spec, "plans", ()))
            and not tuple(getattr(base_spec, "years", ()))
            and not tuple(getattr(base_spec, "semesters", ()))
            and structure.course_name_span is None
        ):
            return ("failed", base_spec, base_resolution, base_completeness)
        compiled_spec = _compile_exact_course_language_recovery(
            base_spec,
            structure,
            context,
        )
        compiled_resolution = resolve_query_spec(
            compiled_spec,
            db_path,
            context=resolution_context,
        )
    except Exception:
        return ("failed", base_spec, base_resolution, base_completeness)
    if compiled_resolution.action in {
        "clarify_program",
        "context_conflict",
        "no_data",
        "unsupported",
    }:
        return ("blocked", base_spec, compiled_resolution, base_completeness)
    if compiled_resolution.action != "answer":
        return ("failed", base_spec, base_resolution, base_completeness)
    if tuple(compiled_spec.operations) != ("identity",):
        compiled_completeness = _classify_structured_parse_completeness(
            compiled_spec,
            compiled_resolution,
            context,
        )
        if compiled_completeness.classification != "complete":
            return ("failed", base_spec, base_resolution, base_completeness)
        return ("recovered", compiled_spec, compiled_resolution, compiled_completeness)
    return ("recovered", compiled_spec, compiled_resolution, base_completeness)


def ask(
    db_path: str | Path,
    question: str,
    structured_model_callable: Callable[[str], str] | None = None,
    top_k: int = 5,
    *,
    context: QueryContext | None = None,
    conversation_context: QueryContext | None = None,
    answer_model_callable: Callable[[str], str] | None = None,
    intent_model_callable: Callable[[str], str] | None = None,
    shadow_intent: bool = False,
    synthesize_answer: bool = False,
    semantic_topic: str | None = None,
) -> dict[str, Any]:
    """Run the typed evidence pipeline while retaining the legacy signature."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if semantic_topic is not None and (
        not isinstance(semantic_topic, str)
        or not semantic_topic.strip()
        or len(semantic_topic.strip()) > 80
    ):
        raise ValueError(
            "semantic_topic must be a non-empty string of at most 80 characters"
        )

    conversation_mode = conversation_context is not None
    parser_context = conversation_context if conversation_mode else context
    has_validated_context_scope = bool(
        isinstance(parser_context, QueryContext)
        and isinstance(parser_context.program, str)
        and parser_context.program.strip()
    )
    spec = parse_query_spec(
        question,
        has_validated_context_scope=has_validated_context_scope,
    )
    if conversation_mode:
        spec = _merge_conversation_context(spec, conversation_context)
    if spec.topic is None and semantic_topic is not None:
        # A retained conversational topic fills only a missing topic, exactly
        # as if the current turn had named it; explicit text always wins.
        # It is retrieval input only, never factual authority.
        spec = replace(spec, topic=semantic_topic.strip())
    active_context = conversation_context if conversation_mode else context
    program = getattr(spec, "program", None) or getattr(active_context, "program", None)
    catalog_key = getattr(active_context, "catalog_key", None)
    catalog_keys: tuple[str | None, ...] = ()
    edition_keys: tuple[str, ...] = ()
    if isinstance(program, str) and program.strip():
        try:
            catalog_keys = catalog_keys_for_program(db_path, program)
            edition_keys = edition_catalog_keys_for_program(db_path, program)
        except (FileNotFoundError, OSError, sqlite3.Error, TypeError, ValueError):
            catalog_keys = ()
            edition_keys = ()
    curriculum_scoped = bool(
        tuple(spec.operations)
        or spec.plans
        or spec.years
        or spec.semesters
        or spec.course_codes
        or spec.course_name
        or spec.topic
    )
    matching_catalog_key = next(
        (
            key for key in catalog_keys
            if isinstance(key, str)
            and isinstance(catalog_key, str)
            and key.casefold() == catalog_key.strip().casefold()
        ),
        None,
    )
    if catalog_key is not None and matching_catalog_key is None:
        return {
            "route": None,
            "result": {
                "status": "clarify_catalog",
                "action": "clarify_catalog",
                "program": program,
            },
        }
    if curriculum_scoped and edition_keys and catalog_key is None:
        return {
            "route": None,
            "result": {
                "status": "clarify_catalog",
                "action": "clarify_catalog",
                "program": program,
                "catalog_keys": list(edition_keys),
            },
        }
    if catalog_key is None and len(catalog_keys) == 1:
        catalog_key = catalog_keys[0]
    elif matching_catalog_key is not None:
        catalog_key = matching_catalog_key

    # A term/year selector without any requested operation is not yet a
    # bounded curriculum question. Do not turn it into a program prompt that
    # suggests clarification can make this credit maximum answerable.
    if (
        not spec.operations
        and (spec.years or spec.semesters)
        and not (spec.plans or spec.course_codes or spec.course_name or spec.category or spec.topic)
        and program is None
    ):
        return _intent_failure_result(question)

    policy_result = route_policy_question(
        db_path,
        question,
        catalog_key=(
            catalog_key
            if program and edition_catalog_keys_for_program(db_path, program)
            else None
        ),
        program_context=program if isinstance(program, str) else None,
    )
    if policy_result is not None:
        policy_query = parse_policy_question(question, program_context=program)
        if policy_query is None or policy_query.kind != "program_total_credits":
            return {"route": None, "result": policy_result}
        policy_context = {"program": program} if program else {}
        if catalog_key is not None:
            policy_context["catalog_key"] = catalog_key
        if len(spec.plans) == 1:
            policy_context["plan"] = spec.plans[0]
        return {
            "route": None,
            "result": policy_result,
            "next_context": policy_context or None,
        }

    operations = tuple(getattr(spec, "operations", ()))
    has_structural_target = bool(
        tuple(getattr(spec, "plans", ()))
        or tuple(getattr(spec, "years", ()))
        or tuple(getattr(spec, "semesters", ()))
        or tuple(getattr(spec, "course_codes", ()))
        or getattr(spec, "course_name", None)
        or getattr(spec, "category", None)
        or getattr(spec, "topic", None)
    )
    explicit_program_mentions = _explicit_program_mentions(question)
    if (
        len(explicit_program_mentions) <= 1
        and (explicit_program_mentions or spec.plans)
        and _is_whole_program_total(spec, program)
    ):
        # Canonical program/catalog-level total evidence (program
        # requirements) is the only authority for a whole-program sum.
        # The plan qualifier, if any, is preserved as conversational scope
        # but never invents a plan-specific total from placement rows.
        total_plans = tuple(getattr(spec, "plans", ()))
        total_answer = answer_policy_query(
            db_path,
            PolicyQuery(
                "program_total_credits",
                program=program,
                plan=total_plans[0] if len(total_plans) == 1 else None,
            ),
            catalog_key=catalog_key,
        )
        total_context: dict[str, Any] = {"program": program}
        if catalog_key is not None:
            total_context["catalog_key"] = catalog_key
        if len(total_plans) == 1:
            total_context["plan"] = total_plans[0]
        return {
            "route": None,
            "result": adapt_policy_answer(total_answer),
            "next_context": total_context,
        }

    resolution_context = (
        QueryContext(catalog_key=catalog_key)
        if conversation_mode and catalog_key is not None
        else None if conversation_mode else context
    )
    if getattr(spec, "judgement", None) == "unsupported":
        unsupported_resolution = resolve_query_spec(
            spec,
            db_path,
            context=resolution_context,
        )
        if unsupported_resolution.action == "unsupported":
            return {
                "route": None,
                "result": _blocked_result(unsupported_resolution),
            }

    structural_interpreted = False
    if (
        not shadow_intent
        and callable(intent_model_callable)
        and _has_unambiguous_recovery_program(program, edition_keys, catalog_key)
        and (
            (
                operations == ("sum_credits",)
                and not has_structural_target
                and len(explicit_program_mentions) <= 1
            )
            or (operations == ("existence",) and not has_structural_target)
            or (
                not has_answerable_target_or_scope(spec, active_context)
                and not (
                    operations == ("sum_credits",)
                    and len(explicit_program_mentions) > 1
                )
            )
        )
        and _DEICTIC_REFERENCE_RE.search(question) is None
    ):
        # NL-2 pre-guard rescue: a targetless exact-course-shaped request
        # would otherwise fail closed before the bounded query-structure
        # interpreter can recover the literal title. Attempt recovery with
        # the same helper as the existing post-guard path; any outcome
        # other than a compiled answerable request falls through to the
        # unchanged deterministic fail-closed guards below.
        base_resolution = resolve_query_spec(
            spec,
            db_path,
            context=resolution_context,
        )
        base_completeness = _classify_structured_parse_completeness(
            spec,
            base_resolution,
            context,
        )
        pre_guard_recovery = _attempt_structural_recovery(
            db_path,
            spec,
            base_resolution,
            base_completeness,
            question,
            context,
            resolution_context,
            intent_model_callable,
        )
        if pre_guard_recovery is not None and pre_guard_recovery[0] == "recovered":
            spec = pre_guard_recovery[1]
            resolution = pre_guard_recovery[2]
            completeness = pre_guard_recovery[3]
            structural_interpreted = True
            operations = tuple(getattr(spec, "operations", ()))
            has_structural_target = bool(
                tuple(getattr(spec, "plans", ()))
                or tuple(getattr(spec, "years", ()))
                or tuple(getattr(spec, "semesters", ()))
                or tuple(getattr(spec, "course_codes", ()))
                or getattr(spec, "course_name", None)
                or getattr(spec, "category", None)
                or getattr(spec, "topic", None)
            )

    if (
        operations == ("sum_credits",)
        and not has_structural_target
        and len(explicit_program_mentions) <= 1
    ):
        # A bare follow-up such as "กี่หน่วยอะ" with only program/catalog
        # context does not identify a semester, course, plan, or program-total
        # requirement. Never widen it to every term in the curriculum.
        # Explicit multi-program wording is excluded: that is a real
        # comparison shape handled by the existing structured path.
        return _intent_failure_result(question)
    if operations == ("existence",) and not has_structural_target:
        # Existence without an entity/filter otherwise degenerates into
        # "does this curriculum contain anything?" and can mask a missed
        # policy/topic parse.
        return _intent_failure_result(question)
    if (
        operations == ("placement",)
        and not (
            tuple(getattr(spec, "course_codes", ()))
            or getattr(spec, "course_name", None)
            or getattr(spec, "topic", None)
        )
        and re.search(r"(?:ตัวนี้|วิชานี้|อันนี้)", question, re.IGNORECASE)
    ):
        return _intent_failure_result(question)

    if (
        not has_answerable_target_or_scope(spec, active_context)
        and not (operations == ("sum_credits",) and len(explicit_program_mentions) > 1)
    ):
        # A targetless detail/reference request must fail closed here, before
        # resolution, retrieval, or any model call, instead of falling through
        # into broad program-wide retrieval with a confident dump.
        return _intent_failure_result(question)

    resolution = resolve_query_spec(spec, db_path, context=resolution_context)
    completeness = _classify_structured_parse_completeness(
        spec,
        resolution,
        context,
    )
    if not shadow_intent and callable(intent_model_callable):
        post_guard_recovery = _attempt_structural_recovery(
            db_path,
            spec,
            resolution,
            completeness,
            question,
            context,
            resolution_context,
            intent_model_callable,
        )
        if post_guard_recovery is not None:
            outcome, spec, resolution, completeness = post_guard_recovery
            if outcome == "blocked":
                return {"route": None, "result": _blocked_result(resolution)}
            if outcome != "recovered":
                return _intent_failure_result(question)
            structural_interpreted = True
    if resolution.action != "answer":
        if resolution.action == "clarify_program":
            consensus = _consensus_prerequisite_grounded_answer(
                db_path,
                spec,
                resolution,
                context,
            )
            if consensus is not None:
                return {
                    "route": None,
                    "result": render_grounded_answer(
                        consensus,
                        answer_model_callable=answer_model_callable,
                        question=question,
                    ),
                }
        return {"route": None, "result": _blocked_result(resolution)}

    if spec.operations == ("program_discovery",):
        grounded = compose_grounded_answer(
            identity_result=_identity_result(
                resolution,
                operation="program_discovery",
            ),
        )
        return {
            "route": None,
            "result": render_grounded_answer(
                grounded,
                answer_model_callable=answer_model_callable,
                question=question,
            ),
        }

    if spec.operations == ("identity",):
        grounded = compose_grounded_answer(
            identity_result=_identity_result(resolution),
        )
        response = {
            "route": None,
            "result": render_grounded_answer(grounded),
        }
        next_context = _next_conversation_context(
            spec, resolution, response["result"], catalog_key
        )
        if next_context is not None:
            response["next_context"] = next_context
        return response

    if not structural_interpreted:
        completeness = _classify_structured_parse_completeness(
            spec,
            resolution,
            context,
        )
    intent_shadow = (
        _run_placement_shadow(
            question,
            spec,
            completeness,
            resolution,
            context,
            intent_model_callable,
        )
        if shadow_intent
        else None
    )
    count_shadow = (
        _run_count_shadow(
            question,
            spec,
            completeness,
            resolution,
            context,
            intent_model_callable,
        )
        if shadow_intent
        else None
    )
    shadow_chain_busy = any(
        shadow is not None and shadow.eligible
        for shadow in (intent_shadow, count_shadow)
    )
    prerequisite_shadow = (
        run_exact_course_shadow(
            question,
            spec,
            completeness,
            resolution,
            context,
            intent_model_callable,
            family="prerequisite_query",
        )
        if shadow_intent and not shadow_chain_busy
        else None
    )
    description_shadow = (
        run_exact_course_shadow(
            question,
            spec,
            completeness,
            resolution,
            context,
            intent_model_callable,
            family="course_description",
        )
        if shadow_intent and not shadow_chain_busy
        and not (
            prerequisite_shadow is not None and prerequisite_shadow.eligible
        )
        else None
    )
    exact_shadow_eligible = any(
        shadow is not None and shadow.eligible
        for shadow in (prerequisite_shadow, description_shadow)
    )
    credit_shadow = (
        run_exact_course_shadow(
            question,
            spec,
            completeness,
            resolution,
            context,
            intent_model_callable,
            family="course_credit_query",
        )
        if shadow_intent and not shadow_chain_busy and not exact_shadow_eligible
        else None
    )
    exact_shadow_eligible = exact_shadow_eligible or (
        credit_shadow is not None and credit_shadow.eligible
    )
    existence_shadow = (
        run_exact_course_shadow(
            question,
            spec,
            completeness,
            resolution,
            context,
            intent_model_callable,
            family="existence_query",
        )
        if shadow_intent and not shadow_chain_busy and not exact_shadow_eligible
        else None
    )

    if (
        shadow_intent
        and completeness.classification == "not_eligible"
        and tuple(getattr(spec, "operations", ())) == ()
        and getattr(spec, "topic", None) is None
        and len(tuple(getattr(spec, "course_codes", ()))) == 1
    ):
        # In shadow mode the legacy interpreter branch is intentionally
        # disabled.  Keep unsupported exact-course wording fail-closed rather
        # than allowing an empty operation set into planning.
        return _intent_failure_result(question)

    shadow_executed = False
    if (
        shadow_intent
        and intent_shadow is not None
        and intent_shadow.eligible
        and intent_shadow.attempted
        and intent_shadow.status == "validated"
        and intent_shadow.comparison == "compatible_extension"
        and intent_shadow.compiled_spec is not None
    ):
        compiled_spec = intent_shadow.compiled_spec
        try:
            compiled_resolution = resolve_query_spec(
                compiled_spec,
                db_path,
                context=context,
            )
            compiled_completeness = _classify_structured_parse_completeness(
                compiled_spec,
                compiled_resolution,
                context,
            )
        except (FileNotFoundError, OSError, TypeError, ValueError, KeyError):
            compiled_resolution = None
            compiled_completeness = None

        authoritative_program = completeness.program or resolution.resolved_program
        same_authoritative_code = tuple(
            getattr(compiled_spec, "course_codes", ())
        ) == tuple(completeness.course_codes)
        safe_compiled_placement = bool(
            compiled_resolution is not None
            and compiled_resolution.action == "answer"
            and isinstance(authoritative_program, str)
            and authoritative_program.strip()
            and tuple(getattr(compiled_spec, "operations", ()))
            == ("placement",)
            and len(tuple(getattr(compiled_spec, "course_codes", ()))) == 1
            and same_authoritative_code
            and compiled_completeness is not None
            and compiled_completeness.classification == "complete"
            and not compiled_completeness.missing_filters
        )
        if safe_compiled_placement:
            spec = compiled_spec
            resolution = compiled_resolution
            completeness = compiled_completeness
            shadow_executed = True
            intent_shadow = replace(intent_shadow, executed=True)

    if (
        shadow_intent
        and not shadow_executed
        and count_shadow is not None
        and count_shadow.eligible
        and count_shadow.attempted
        and count_shadow.status == "validated"
        and count_shadow.comparison == "compatible_extension"
        and count_shadow.compiled_spec is not None
    ):
        compiled_spec = count_shadow.compiled_spec
        try:
            compiled_resolution = resolve_query_spec(
                compiled_spec,
                db_path,
                context=context,
            )
            compiled_completeness = _classify_structured_parse_completeness(
                compiled_spec,
                compiled_resolution,
                context,
            )
        except (FileNotFoundError, OSError, TypeError, ValueError, KeyError):
            compiled_resolution = None
            compiled_completeness = None

        authoritative_program = completeness.program or resolution.resolved_program
        safe_compiled_count = bool(
            compiled_resolution is not None
            and compiled_resolution.action == "answer"
            and isinstance(authoritative_program, str)
            and authoritative_program.strip()
            and tuple(getattr(compiled_spec, "operations", ())) == ("count",)
            and not getattr(compiled_spec, "course_codes", ())
            and getattr(compiled_spec, "course_name", None) is None
            and getattr(compiled_spec, "topic", None) is None
            and getattr(compiled_spec, "judgement", None) in {None, "none"}
            and _count_shadow_comparison(spec, compiled_spec, context)
            == "compatible_extension"
            and compiled_completeness is not None
            and compiled_completeness.classification == "complete"
            and not compiled_completeness.missing_filters
        )
        if safe_compiled_count:
            spec = compiled_spec
            resolution = compiled_resolution
            completeness = compiled_completeness
            shadow_executed = True
            count_shadow = replace(count_shadow, executed=True)

    if shadow_intent and not shadow_executed:
        for exact_shadow, family, operation in (
            (prerequisite_shadow, "prerequisite_query", "prerequisite"),
            (description_shadow, "course_description", "describe"),
            (credit_shadow, "course_credit_query", "sum_credits"),
            (existence_shadow, "existence_query", "existence"),
        ):
            if not (
                exact_shadow is not None
                and exact_shadow.eligible
                and exact_shadow.attempted
                and exact_shadow.status == "validated"
                and exact_shadow.comparison == "compatible_extension"
                and exact_shadow.compiled_spec is not None
            ):
                continue
            compiled_spec = exact_shadow.compiled_spec
            try:
                compiled_resolution = resolve_query_spec(
                    compiled_spec,
                    db_path,
                    context=context,
                )
                compiled_completeness = _classify_structured_parse_completeness(
                    compiled_spec,
                    compiled_resolution,
                    context,
                )
            except (FileNotFoundError, OSError, TypeError, ValueError, KeyError):
                continue

            authoritative_program = completeness.program or resolution.resolved_program
            exact_target_preserved = (
                tuple(getattr(compiled_spec, "course_codes", ()))
                == tuple(getattr(spec, "course_codes", ()))
                and getattr(compiled_spec, "course_name", None)
                == getattr(spec, "course_name", None)
            )
            safe_compiled_exact_course = bool(
                compiled_resolution.action == "answer"
                and isinstance(authoritative_program, str)
                and authoritative_program.strip()
                and tuple(getattr(compiled_spec, "operations", ())) == (operation,)
                and exact_target_preserved
                and getattr(compiled_spec, "topic", None) is None
                and getattr(compiled_spec, "judgement", None) in {None, "none"}
                and compiled_completeness.classification == "complete"
                and not compiled_completeness.missing_filters
            )
            if safe_compiled_exact_course:
                spec = compiled_spec
                resolution = compiled_resolution
                completeness = compiled_completeness
                shadow_executed = True
                if family == "prerequisite_query":
                    prerequisite_shadow = replace(exact_shadow, executed=True)
                elif family == "course_description":
                    description_shadow = replace(exact_shadow, executed=True)
                elif family == "course_credit_query":
                    credit_shadow = replace(exact_shadow, executed=True)
                else:
                    existence_shadow = replace(exact_shadow, executed=True)
                break

    def finish(result: dict[str, Any]) -> dict[str, Any]:
        diagnostics = {}
        if intent_shadow is not None:
            diagnostics["intent_shadow"] = intent_shadow
        if count_shadow is not None:
            diagnostics["count_shadow"] = count_shadow
        if prerequisite_shadow is not None:
            diagnostics["prerequisite_shadow"] = prerequisite_shadow
        if description_shadow is not None:
            diagnostics["description_shadow"] = description_shadow
        if credit_shadow is not None:
            diagnostics["credit_shadow"] = credit_shadow
        if existence_shadow is not None:
            diagnostics["existence_shadow"] = existence_shadow
        response = result if not diagnostics else {**result, **diagnostics}
        answer_result = response.get("result")
        next_context = _next_conversation_context(
            spec, resolution, answer_result, catalog_key
        )
        if next_context is not None:
            response = {**response, "next_context": next_context}
        return response

    filtered_credit_candidate = _is_filtered_sum_credits_fallback_candidate(
        spec,
        completeness,
    )
    if filtered_credit_candidate:
        filtered_scope = build_structural_scope(
            spec, resolution, catalog_key=catalog_key
        )
        if not callable(structured_model_callable):
            claim = _filtered_credit_status_claim(
                filtered_scope,
                "insufficient_evidence",
            )
        else:
            fallback_scope = _fallback_scope(
                completeness, resolution, catalog_key=catalog_key
            )
            if fallback_scope is None:
                claim = _filtered_credit_status_claim(
                    filtered_scope,
                    "insufficient_evidence",
                )
            else:
                fallback_result = run_structured_fallback(
                    db_path,
                    question,
                    fallback_scope,
                    structured_model_callable,
                )
                grounded_list = ground_course_list(
                    db_path,
                    fallback_result,
                    fallback_scope,
                )
                if grounded_list.status != "complete":
                    claim = _filtered_credit_status_claim(
                        filtered_scope,
                        grounded_list.status,
                    )
                elif not grounded_list.selected_targets:
                    claim = _filtered_credit_status_claim(
                        filtered_scope,
                        "insufficient_evidence",
                    )
                else:
                    filtered_plan = _filtered_credit_plan(
                        spec,
                        resolution,
                        grounded_list.selected_targets,
                        catalog_key=catalog_key,
                    )
                    credit_bundle = execute_evidence_plan(
                        db_path,
                        filtered_plan,
                    )
                    claims = _compose_evidence_claims(spec, credit_bundle)
                    if not claims or any(
                        claim.operation != "sum_credits" for claim in claims
                    ):
                        claim = _filtered_credit_status_claim(
                            filtered_scope,
                            "insufficient_evidence",
                        )
                    else:
                        grounded = compose_grounded_answer(
                            composed_claims=claims,
                        )
                        return finish({
                            "route": None,
                            "result": render_grounded_answer(
                                grounded,
                                answer_model_callable=None,
                                question=question,
                            ),
                        })
        grounded = compose_grounded_answer(composed_claims=(claim,))
        return finish({
            "route": None,
            "result": render_grounded_answer(
                grounded,
                answer_model_callable=None,
                question=question,
            ),
        })
    if (
        completeness.classification in {"partial", "unrecognized_structured"}
        and callable(structured_model_callable)
        and _is_course_list_fallback_candidate(spec, completeness)
    ):
        fallback_operation = _course_list_fallback_operation(spec)
        fallback_scope = _fallback_scope(
            completeness, resolution, catalog_key=catalog_key
        )
        if fallback_scope is not None and fallback_operation is not None:
            fallback_result = run_structured_fallback(
                db_path,
                question,
                fallback_scope,
                structured_model_callable,
            )
            grounded_list = ground_course_list(
                db_path,
                fallback_result,
                fallback_scope,
            )
            claim = _fallback_course_list_claim(
                grounded_list,
                fallback_scope,
                operation=fallback_operation,
            )
            grounded = compose_grounded_answer(composed_claims=(claim,))
            return finish({
                "route": None,
                "result": render_grounded_answer(
                    grounded,
                    answer_model_callable=None,
                    question=question,
                ),
            })

    if (
        completeness.classification in {"partial", "unrecognized_structured"}
        and callable(structured_model_callable)
        and _is_placement_fallback_candidate(spec, completeness)
    ):
        fallback_scope = _fallback_scope(
            completeness,
            resolution,
            placement_code_identity=True,
            catalog_key=catalog_key,
        )
        if fallback_scope is not None:
            fallback_result = run_structured_fallback(
                db_path,
                question,
                fallback_scope,
                structured_model_callable,
                selector_mode="placement",
            )
            grounded_placement = ground_placement(
                db_path,
                fallback_result,
                fallback_scope,
            )
            claim = _fallback_placement_claim(
                grounded_placement,
                fallback_scope,
            )
            grounded = compose_grounded_answer(composed_claims=(claim,))
            return finish({
                "route": None,
                "result": render_grounded_answer(
                    grounded,
                    answer_model_callable=None,
                    question=question,
                ),
            })

    if (
        completeness.classification in {"partial", "unrecognized_structured"}
        and callable(structured_model_callable)
        and _is_course_credit_fallback_candidate(spec, completeness, resolution)
    ):
        fallback_scope = _fallback_scope(
            completeness,
            resolution,
            placement_code_identity=True,
            catalog_key=catalog_key,
        )
        if fallback_scope is not None:
            fallback_result = run_structured_fallback(
                db_path,
                question,
                fallback_scope,
                structured_model_callable,
                selector_mode="course_credit",
            )
            grounded_credit = ground_course_credit(
                db_path,
                fallback_result,
                fallback_scope,
            )
            claim = _fallback_course_credit_claim(
                grounded_credit,
                fallback_scope,
            )
            grounded = compose_grounded_answer(composed_claims=(claim,))
            return finish({
                "route": None,
                "result": render_grounded_answer(
                    grounded,
                    answer_model_callable=None,
                    question=question,
                ),
            })

    intent_interpreted = shadow_executed or structural_interpreted
    if (
        not shadow_intent
        and not intent_interpreted
        and _should_use_course_list_interpreter(
            spec,
            completeness,
            resolution,
            context,
        )
    ):
        course_list_outcome = _run_course_list_interpreter(
            db_path,
            question,
            spec,
            context,
            intent_model_callable,
        )
        if course_list_outcome is None:
            return _intent_failure_result(question)
        spec, resolution, completeness = course_list_outcome
        intent_interpreted = True
    if not shadow_intent and not intent_interpreted and _should_use_intent_interpreter(
        spec,
        completeness,
        resolution,
        context,
    ):
        if not callable(intent_model_callable):
            return _intent_failure_result(question)

        try:
            interpretation = interpret_question_intent(
                question,
                intent_model_callable,
            )
        except Exception:
            # The interpreter is an optional proposal boundary.  A model or
            # validation failure must remain fail closed.
            return _intent_failure_result(question)
        try:
            compiled_spec = compile_intent_to_query_spec(
                spec,
                interpretation,
                **_intent_authoritative_scope(spec, context),
            )
        except (TypeError, ValueError):
            return _intent_failure_result(question)

        try:
            compiled_resolution = resolve_query_spec(
                compiled_spec,
                db_path,
                context=context,
            )
        except (FileNotFoundError, OSError, TypeError, ValueError, KeyError):
            return _intent_failure_result(question)

        if compiled_resolution.action != "answer":
            if compiled_resolution.action in {
                "clarify_program",
                "context_conflict",
                "no_data",
                "unsupported",
            }:
                return finish({"route": None, "result": _blocked_result(compiled_resolution)})
            return _intent_failure_result(question)

        compiled_completeness = _classify_structured_parse_completeness(
            compiled_spec,
            compiled_resolution,
            context,
        )
        if compiled_completeness.classification in {
            "partial",
            "unrecognized_structured",
        }:
            return _intent_failure_result(question)

        # Do not let an interpreted proposal re-enter any approved SQL seam.
        # It proceeds only through the existing deterministic evidence path.
        spec = compiled_spec
        resolution = compiled_resolution
        completeness = compiled_completeness
        intent_interpreted = True

    if completeness.classification in {"partial", "unrecognized_structured"}:
        grounded = compose_grounded_answer(
            resolution_status="insufficient_evidence",
        )
        return finish({
            "route": None,
            "result": render_grounded_answer(
                grounded,
                answer_model_callable=None,
                question=question,
            ),
        })

    try:
        plan = plan_evidence(spec, resolution, catalog_key=catalog_key)
        bundle = execute_evidence_plan(db_path, plan)
    except (FileNotFoundError, OSError, TypeError, ValueError, KeyError):
        grounded = compose_grounded_answer(
            resolution_status="insufficient_evidence",
        )
        return finish({
            "route": None,
            "result": render_grounded_answer(grounded),
        })

    similarity_evidence: SimilarityEvidence | None = None
    similarity_request_ids: tuple[str, str] | None = None
    selected_plan: str | None = None
    if "similarity" in spec.operations:
        similarity_request_ids = _select_similarity_request_ids(plan, resolution)
        if similarity_request_ids is not None:
            resolved_plans = tuple(resolution.resolved_plans)
            selected_plan = resolved_plans[0] if len(resolved_plans) == 1 else None
            try:
                similarity_evidence = execute_exact_similarity_from_bundle(
                    db_path,
                    bundle,
                    similarity_request_ids[0],
                    similarity_request_ids[1],
                    selected_plan=selected_plan,
                )
            except (FileNotFoundError, OSError, TypeError, ValueError, KeyError):
                similarity_evidence = None

    claims = _compose_evidence_claims(
        spec,
        bundle,
        resolution=resolution,
        similarity_evidence=similarity_evidence,
        similarity_request_ids=similarity_request_ids,
        selected_plan=selected_plan,
    )
    if "identity" in spec.operations:
        identity_claims = compose_grounded_answer(
            identity_result=_identity_result(resolution),
        ).claims
        claims = (*identity_claims, *claims)
    grounded = compose_grounded_answer(composed_claims=claims)
    preference_advisory = (
        intent_interpreted and getattr(spec, "judgement", None) == "preference"
    )
    return finish({
        "route": None,
        "result": render_grounded_answer(
            grounded,
            answer_model_callable=(
                answer_model_callable
                if preference_advisory
                else (None if intent_interpreted else answer_model_callable)
            ),
            question=question,
            preference_advisory=preference_advisory,
            synthesize_answer=(synthesize_answer and not intent_interpreted and not preference_advisory),
        ),
    })


__all__ = ["ask"]
