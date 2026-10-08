"""Execution bridges for semantic mode: deterministic, policy, and SQL.

All three bridges reuse the frozen execution/evidence machinery:

- deterministic: resolve_query_spec → plan_evidence → execute_evidence_plan
  → compose_grounded_answer (same COUNT/provenance contracts as legacy).
- policy: answer_policy_query for single-kind topics (honors merges both
  kinds); anything needing guessed thresholds fails closed.
- SQL: guarded ask_sql driven by a canonical-scope synthesized utterance
  (never the raw student wording); only status/provenance/rows are trusted,
  never the internal summary prose.

Bridges return VerifiedResult payloads; they never render user answers.
"""

from __future__ import annotations

import re
from dataclasses import replace
from types import MappingProxyType
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from rag.answer import _plan_display
from rag.evidence_executor import execute_evidence_plan
from rag.evidence_planner import plan_evidence
from rag.grounded_answer import compose_grounded_answer
from rag.policy.answer import PolicyQuery, answer_policy_query
from rag.policy.routing import adapt_policy_answer
# Reuse the frozen claim-adaptation boundary (the same private seam legacy
# qa.py uses to turn evidence bundles into typed claims). Importing the
# underscore helpers keeps one canonical implementation instead of a
# divergent duplicate; the semantic compiler feeds them a constructed
# QuerySpec built from resolved structure, never parsed language.
from rag.qa import _compose_evidence_claims as _legacy_compose_claims
from rag.qa import _identity_result as _legacy_identity_result
from rag.query_spec import QuerySpec
from rag.resolution import QueryContext, ResolutionOutcome, resolve_query_spec
from rag.semantic.compiler import (
    compile_resolved_intent_to_query_spec,
    synthesize_canonical_utterance,
)
from rag.semantic.errors import SemanticOperationalError, is_provider_error
from rag.semantic.planner import SemanticPlan
from rag.semantic.schema import (
    ResolvedIntent,
    ResolvedTarget,
    VerifiedAlternativeSelection,
    VerifiedNumericComparison,
    VerifiedNumericComparisonSide,
    VerifiedResult,
)
from rag.semantic.resolver import _connect_ro
from rag.structured.queries import scoped_course_set, _provenance_for

MAX_DISPLAYED_COURSES = 20

_SINGLE_KIND_POLICY_TOPICS = {
    "graduation": "graduation_requirements",
    "graduation_gpa": "graduation_gpa",
    "english_exit": "graduation_english_exit",
    "probation": "probation_entry",
    "leave": "leave_of_absence",
    "resignation": "resignation",
    "transfer": "credit_transfer",
    "assessment": "assessment_method",
    "grading": "gpa_calculation_method",
    "conduct": "student_conduct_rules",
}


def _claim_texts(claims: tuple[Any, ...]) -> tuple[str, ...]:
    """Render conservative human-readable lines from verified claims only."""
    facts: list[str] = []
    for claim in claims:
        try:
            operation = getattr(claim, "operation", "")
            value = getattr(claim, "value", None)
            line = _claim_line_with_evidence(claim)
        except Exception:
            continue
        if line:
            facts.append(line)
    return tuple(facts)


def _text(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _numbers(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, bool):
        return found
    if isinstance(value, (int, float)):
        return [str(value)]
    if isinstance(value, Mapping):
        for item in value.values():
            found.extend(_numbers(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.extend(_numbers(item))
    return found


def _codes(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, str):
        token = value.strip()
        if len(token) == 8 and token.isdigit():
            found.append(token)
        return found
    if isinstance(value, Mapping):
        for item in value.values():
            found.extend(_codes(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.extend(_codes(item))
    return found[:8]


def _collection_course_codes(value: Any) -> list[str]:
    """Collect distinct concrete collection identities in source order.

    Structured rows supply course_code; other row metadata is not identity.
    Raw code sequences and nested collection containers are also supported.
    Display and retention bounds belong to callers, not recursive traversal.
    """
    found: list[str] = []
    seen: set[str] = set()

    def collect(item: Any) -> None:
        if isinstance(item, str):
            code = item.strip()
            if len(code) == 8 and code.isascii() and code.isdigit() and code not in seen:
                seen.add(code)
                found.append(code)
        elif isinstance(item, Mapping):
            if "course_code" in item:
                collect(item["course_code"])
            else:
                for nested in item.values():
                    if isinstance(nested, (Mapping, list, tuple)):
                        collect(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                collect(nested)

    collect(value)
    return found


def _collection_course_identities(value: Any) -> list[tuple[str, str | None]]:
    """Pair each concrete code with a title from the same grounded row."""
    found: list[tuple[str, str | None]] = []
    positions: dict[str, int] = {}
    priorities: dict[str, int] = {}

    def collect(item: Any) -> None:
        if isinstance(item, Mapping) and "course_code" in item:
            code = item["course_code"]
            thai = _text(item.get("name_th"))
            english = _text(item.get("name_en"))
            title, priority = (thai, 2) if thai else (english, 1 if english else 0)
        elif isinstance(item, str):
            code, title, priority = item, None, 0
        else:
            if isinstance(item, Mapping):
                nested = (child for child in item.values() if isinstance(child, (Mapping, list, tuple)))
            else:
                nested = item if isinstance(item, (list, tuple)) else ()
            for child in nested:
                if isinstance(child, (Mapping, list, tuple, str)):
                    collect(child)
            return
        if not isinstance(code, str):
            return
        code = code.strip()
        display_reference = (
            len(code) == 8 and code[:5].isascii() and code[:5].isdigit()
            and code[5:] == "xxx"
        )
        numeric_identity = len(code) == 8 and code.isascii() and code.isdigit()
        if not numeric_identity and not display_reference:
            return
        if code not in positions:
            positions[code] = len(found)
            priorities[code] = priority
            found.append((code, title))
        elif priority > priorities[code]:
            # Better grounded title for the same identity; retain source order.
            found[positions[code]] = (code, title)
            priorities[code] = priority

    collect(value)
    return found


def _titles(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key in ("name_en", "name_th", "title", "course_name"):
            text = _text(value.get(key))
            if text is not None and text not in found:
                found.append(text)
                break
        for item in value.values():
            if len(found) >= 4:
                break
            if isinstance(item, (Mapping, list, tuple)):
                found.extend(t for t in _titles(item) if t not in found)
    elif isinstance(value, (list, tuple)):
        for item in value:
            if len(found) >= 4:
                break
            found.extend(t for t in _titles(item) if t not in found)
    return found[:4]


def _placement_terms(value: Any) -> list[tuple[int | None, int | None, str | None]]:
    """Collect distinct fixed and flexible (year, semester, plan) placements."""
    terms: list[tuple[int | None, int | None, str | None]] = []
    items = value if isinstance(value, (list, tuple)) else [value]
    for item in items:
        if not isinstance(item, Mapping):
            continue
        plan = item.get("plan_key")
        plan = plan.strip() if isinstance(plan, str) and plan.strip() else None

        choices = item.get("year_semester_choices")
        if isinstance(choices, (list, tuple)):
            for choice in choices:
                if not isinstance(choice, (list, tuple)) or len(choice) != 2:
                    continue
                year, semester = choice
                if (
                    isinstance(year, bool)
                    or not isinstance(year, int)
                    or isinstance(semester, bool)
                    or not isinstance(semester, int)
                ):
                    continue
                term = (year, semester, plan)
                if term not in terms:
                    terms.append(term)

        year = item.get("year_number")
        semester = item.get("semester_number")
        year = year if isinstance(year, int) and not isinstance(year, bool) else None
        semester = (
            semester if isinstance(semester, int) and not isinstance(semester, bool) else None
        )
        fixed_term = (year, semester, plan)
        if (
            (year is not None or semester is not None)
            and fixed_term not in terms
        ):
            terms.append(fixed_term)
    return terms


def _course_label(value: Any) -> str:
    codes = _codes(value)
    titles = _titles(value)
    parts = []
    if codes:
        parts.append(codes[0])
    parts.extend(title for title in titles[:2] if title not in parts)
    return " ".join(parts)


def _description_evidence_texts(value: Any) -> list[str]:
    """Return bounded, distinct text from retrieved course-description rows."""
    found: list[str] = []

    def collect(item: Any) -> None:
        if isinstance(item, Mapping):
            text = item.get("text")
            if (
                item.get("chunk_type") == "description"
                and item.get("entity_type") == "course"
                and isinstance(text, str)
                and text.strip()
            ):
                normalized = " ".join(text.split())[:600].rstrip()
                if normalized and normalized not in found:
                    found.append(normalized)
                return
            for nested in item.values():
                collect(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                collect(nested)

    collect(value)
    return found[:3]


def _claim_line(operation: str, value: Any) -> str | None:
    if operation in {"list", "topic_matches", "course_set"}:
        identities = _collection_course_identities(value)
        if identities:
            line = "รายวิชาที่พบ:\n" + "\n".join(
                f"- {code} — {title}" if title else f"- {code}"
                for code, title in identities[:MAX_DISPLAYED_COURSES]
            )
            if len(identities) > MAX_DISPLAYED_COURSES:
                line += (
                    f"\n(แสดง {MAX_DISPLAYED_COURSES} จาก "
                    f"{len(identities)} รายวิชา)"
                )
            return line
        titles = _titles(value)
        return (
            "รายวิชาที่พบ: "
            + ", ".join(titles[:MAX_DISPLAYED_COURSES])
            if titles else None
        )
    codes = _codes(value)
    titles = _titles(value)
    numbers = _numbers(value)
    subject = codes[0] if codes else (titles[0] if titles else "")
    if operation == "sum_credits" and numbers:
        label = _course_label(value) or subject
        return f"{label}: {numbers[0]} หน่วยกิต".strip(": ")
    if operation == "prerequisite":
        prereqs = ", ".join(codes[1:] + titles[1:]) if (codes or titles) else ""
        head = codes[0] if codes else (titles[0] if titles else "วิชา")
        return f"{head} มีวิชาบังคับก่อน: {prereqs}" if prereqs else f"{head} มีข้อมูลวิชาบังคับก่อน"
    if operation == "placement":
        terms = _placement_terms(value)
        label = _course_label(value)
        if not terms:
            return label or None
        lines: list[str] = []
        for year, semester, plan in terms:
            parts = []
            if year is not None:
                parts.append(f"ชั้นปีที่ {year}")
            if semester is not None:
                parts.append(f"ภาคการศึกษาที่ {semester}")
            if plan is not None:
                parts.append(f"แผน{_plan_display(plan)}")
            lines.append(
                "%s: %s" % (label, " ".join(parts)) if label else " ".join(parts)
            )
        return " | ".join(line for line in lines if line) or None
    if operation == "identity":
        return _course_label(value) or None
    if operation in {"describe", "description_evidence"}:
        label = _course_label(value)
        descriptions = _description_evidence_texts(value)
        if descriptions:
            return "\n".join(
                f"{label}: {text}" if label else text
                for text in descriptions
            )
        return label or None
    if operation == "existence":
        return "พบข้อมูลตามเงื่อนไข" if value else None
    if operation == "count":
        return f"จำนวน: {numbers[0]}" if numbers else None
    return None


def _claim_line_with_evidence(claim: Any) -> str | None:
    """Project a claim while retaining canonical raw credit notation, if present."""
    operation = str(getattr(claim, "operation", ""))
    value = getattr(claim, "value", None)
    line = _claim_line(operation, value)
    if operation != "sum_credits" or not line:
        return line
    evidence = getattr(claim, "evidence", None)
    components = getattr(evidence, "components", ())
    raw_values: list[str] = []
    if isinstance(components, (list, tuple)):
        for component in components:
            if not isinstance(component, Mapping):
                continue
            raw = component.get("credits_raw")
            if isinstance(raw, str) and raw.strip() and raw.strip() not in raw_values:
                raw_values.append(raw.strip())
    if len(raw_values) == 1:
        number = _numbers(value)
        if number:
            return (
                f"{_course_label(value)}: {raw_values[0]} "
                f"({number[0]} หน่วยกิต)"
            ).strip(": ")
    return line


def execute_deterministic(
    db_path: str | Path,
    spec: QuerySpec,
    resolution_context: QueryContext,
    question: str,
) -> VerifiedResult:
    """Run the frozen deterministic evidence pipeline for one compiled spec."""
    try:
        resolution = resolve_query_spec(spec, db_path, context=resolution_context)
    except Exception:
        return VerifiedResult(
            status="missing_data",
            missing_information=("ไม่สามารถตรวจสอบขอบเขตหลักสูตรได้",),
            failure_category="RESOLUTION_ERROR",
        )
    if resolution.action != "answer":
        return VerifiedResult(
            status="missing_data" if resolution.action == "no_data" else "missing_scope",
            missing_information=(f"resolution: {resolution.action}",),
            failure_category="EXPECTED_SAFE_FAILURE",
        )
    try:
        plan = plan_evidence(
            spec, resolution, catalog_key=resolution_context.catalog_key
        )
        bundle = execute_evidence_plan(db_path, plan)
        claims = _legacy_compose_claims(spec, bundle)
        if "identity" in tuple(getattr(spec, "operations", ())):
            identity_claims = compose_grounded_answer(
                identity_result=_legacy_identity_result(resolution)
            ).claims
            claims = (*identity_claims, *claims)
        grounded = compose_grounded_answer(composed_claims=claims)
    except Exception:
        return VerifiedResult(
            status="missing_data",
            missing_information=("ไม่สามารถประมวลผลหลักฐานได้",),
            failure_category="QUERY_ERROR",
        )
    if grounded.status != "answer":
        return VerifiedResult(
            status="missing_data",
            claims=tuple(grounded.claims),
            provenance=tuple(grounded.provenance),
            missing_information=(f"evidence: {grounded.status}",),
            failure_category="EXPECTED_SAFE_FAILURE",
        )
    facts = _claim_texts(tuple(grounded.claims))
    if not facts:
        return VerifiedResult(
            status="missing_data",
            claims=tuple(grounded.claims),
            provenance=tuple(grounded.provenance),
            missing_information=("ไม่มีข้อเท็จจริงที่ยืนยันได้",),
            failure_category="GROUNDING_ERROR",
        )
    retained = _retained_from_claims(
        tuple(grounded.claims),
        resolution_context.program,
        resolution_context.catalog_key,
    )
    return VerifiedResult(
        status="answer",
        summary_facts=facts,
        claims=tuple(grounded.claims),
        provenance=tuple(grounded.provenance),
        failure_category="NONE",
        result_courses=retained[0],
        result_scope_program=retained[1],
    )


def _complete_source_references(value: Any) -> bool:
    return isinstance(value, (tuple, list)) and bool(value) and all(
        isinstance(ref, Mapping)
        and isinstance(ref.get("source_filename"), str) and bool(ref["source_filename"].strip())
        and type(ref.get("source_page")) is int and ref["source_page"] > 0
        and isinstance(ref.get("document_category"), str) and bool(ref["document_category"].strip())
        for ref in value
    )


def _verify_alternative_selection(
    db_path: str | Path, resolved: ResolvedIntent,
) -> VerifiedAlternativeSelection:
    """Verify the complete group, never infer bounds from named alternatives."""
    scope = resolved.scope
    wanted = tuple(member.course_code for member in resolved.target.members)
    if not scope.program or not scope.catalog_key or not scope.plan or len(wanted) < 2:
        raise ValueError("alternative_selection needs complete canonical set and plan scope")
    payload = scoped_course_set(
        db_path, scope.program, (scope.plan,), catalog_key=scope.catalog_key,
        years=scope.years, semesters=scope.semesters,
        course_targets=tuple({"course_code": code} for code in wanted),
    )
    rows = payload.get("courses", ())
    if payload.get("status") != "ok" or not rows:
        raise ValueError("alternative group evidence missing")
    connection = _connect_ro(db_path)
    references = []
    signature = None
    try:
        catalog = connection.execute("SELECT catalog_id FROM catalogs WHERE catalog_key = ?", (scope.catalog_key,)).fetchone()
        if catalog is None:
            raise ValueError("alternative group catalog missing")
        for row in rows:
            group_id = row.get("alternative_group_id")
            members = row.get("alternative_courses", ())
            minimum, maximum = row.get("minimum_choices"), row.get("maximum_choices")
            if (
                type(group_id) is not int or group_id <= 0 or not row.get("is_alternative")
                or row.get("catalog_id") != catalog["catalog_id"]
                or row.get("program") != scope.program or row.get("plan_key") != scope.plan
                or not isinstance(members, (list, tuple)) or not members
                or type(minimum) is not int or type(maximum) is not int
                or not _complete_source_references(row.get("provenance"))
            ):
                raise ValueError("incomplete or out-of-scope alternative group evidence")
            codes = tuple(member.get("course_code") for member in members)
            if (
                len(codes) != len(set(codes)) or set(codes) != set(wanted)
                or not 0 <= minimum <= maximum <= len(codes)
                or any(member.get("catalog_id") != catalog["catalog_id"]
                       or not _complete_source_references(member.get("provenance")) for member in members)
            ):
                raise ValueError("alternative membership or choice bounds do not prove the exact set")
            current = (group_id, frozenset(codes), minimum, maximum)
            if signature is not None and signature != current:
                raise ValueError("conflicting alternative group evidence")
            signature = current
            # The merged row cannot prove that each provenance boundary exists.
            # Check placement, group and membership source links independently.
            links = [("plan_placement_provenance", "placement_id", row.get("placement_id")),
                     ("alternative_group_provenance", "alternative_group_id", group_id)]
            links.extend(("alternative_group_member_provenance", "alternative_group_member_id",
                          member.get("alternative_group_member_id")) for member in members)
            for table, column, identity in links:
                if type(identity) is not int:
                    raise ValueError("alternative evidence identity missing")
                source_refs = _provenance_for(connection, table, column, identity)
                if not _complete_source_references(source_refs):
                    raise ValueError("alternative evidence source link missing")
                for ref in source_refs:
                    if ref not in references:
                        references.append(ref)
            for ref in row["provenance"]:
                if ref not in references:
                    references.append(dict(ref))
    finally:
        connection.close()
    return VerifiedAlternativeSelection(
        program=scope.program, catalog_key=scope.catalog_key, plan=scope.plan,
        alternative_group_id=signature[0], member_course_codes=wanted,
        minimum_choices=signature[2], maximum_choices=signature[3],
        provenance=tuple(MappingProxyType(dict(ref)) for ref in references),
    )


def _member_fact(member: ResolvedTarget, claim: Any) -> str | None:
    """Project member-local evidence without choosing a nested first code."""
    label = " ".join(value for value in (member.course_code, member.course_name) if value)
    operation, value = claim.operation, claim.value
    if operation == "placement":
        terms = _placement_terms(value)
        if not terms:
            return None
        return label + ": " + " | ".join(
            f"ชั้นปีที่ {year} ภาคการศึกษาที่ {semester} แผน{_plan_display(plan)}"
            for year, semester, plan in terms
        )
    if operation == "prerequisite":
        records = value if isinstance(value, (list, tuple)) else ()
        if records and all(row.get("prerequisite_state") == "explicit_none" for row in records):
            return label + ": ไม่มีวิชาบังคับก่อน"
        codes = []
        for row in records:
            code = row.get("prerequisite_code")
            if row.get("alternative_group_id") is not None or not isinstance(code, str):
                return None  # Never flatten an alternative prerequisite into mandatory courses.
            if code not in codes:
                codes.append(code)
        return label + " มีวิชาบังคับก่อน: " + ", ".join(codes) if codes else None
    line = _claim_line_with_evidence(claim)
    return label + ": " + line if line else None


def execute_explicit_course_set(
    db_path: str | Path, resolved: ResolvedIntent, question: str,
) -> VerifiedResult:
    """Atomic bounded member-local execution over canonical resolved targets."""
    from rag.semantic.planner import plan_semantic_query, EXECUTION_DETERMINISTIC
    plan = plan_semantic_query(resolved)
    if plan.execution != EXECUTION_DETERMINISTIC or resolved.target.kind != "literal_set":
        return VerifiedResult(status="unsupported", missing_information=(plan.reason,),
                              failure_category="EXPECTED_SAFE_FAILURE")
    facts, claims, references, courses = [], [], [], []
    scope = resolved.scope
    try:
        spec = compile_resolved_intent_to_query_spec(resolved, question)
        if not spec.operations:
            raise ValueError("no executable member operations")
        context = QueryContext(program=scope.program, catalog_key=scope.catalog_key,
                               plan=scope.plan, years=scope.years, semesters=scope.semesters)
        for index, member in enumerate(resolved.target.members, 1):
            if (member.program, member.catalog_key) != (scope.program, scope.catalog_key):
                raise ValueError(f"member {index} has incompatible canonical scope")
            member_spec = replace(spec, course_codes=(member.course_code,))
            result = execute_deterministic(db_path, member_spec, context, question)
            operations = {claim.operation for claim in result.claims if claim.status in {"complete", "valid_empty"}}
            if (
                result.status != "answer" or not set(spec.operations).issubset(operations)
                or not result.provenance
                or any(claim.status not in {"complete", "valid_empty"} for claim in result.claims)
            ):
                raise ValueError(f"member {index} ({member.course_code}): incomplete requested evidence")
            for claim in result.claims:
                line = _member_fact(member, claim)
                if not line:
                    raise ValueError(f"member {index} ({member.course_code}): unsupported evidence projection")
                facts.append(line)
            claims.extend(result.claims)
            for ref in result.provenance:
                if ref not in references:
                    references.append(ref)
            courses.append({"course_code": member.course_code, "program": member.program,
                            "catalog_key": member.catalog_key})
        selections = ()
        if resolved.intent.relation == "alternative_selection" or "alternative_selection" in resolved.intent.requested_fields:
            selections = (_verify_alternative_selection(db_path, resolved),)
            for ref in selections[0].provenance:
                if ref not in references:
                    references.append(ref)
        verified = VerifiedResult(
            status="answer", summary_facts=tuple(facts), claims=tuple(claims),
            provenance=tuple(references), result_courses=tuple(courses), result_scope_program=scope.program,
            alternative_selections=selections, explicit_course_set=True,
        )
        # Do not accept more facts than the bounded renderer can retain.
        from rag.semantic.answerer import render_verified_course_set, MAX_ANSWER_LEN
        if len(render_verified_course_set(verified)) > MAX_ANSWER_LEN:
            return VerifiedResult(status="unsupported", missing_information=("explicit course-set answer exceeds presentation bound",),
                                  failure_category="EXPECTED_SAFE_FAILURE")
        return verified
    except Exception as error:
        return VerifiedResult(status="missing_data", missing_information=("course-set evidence: " + str(error),),
                              failure_category="EXPECTED_SAFE_FAILURE")


def _policy_kinds_for(topic: str | None) -> tuple[str, ...]:
    if topic == "honors":
        return ("honors_first", "honors_second")
    kind = _SINGLE_KIND_POLICY_TOPICS.get(topic or "")
    return (kind,) if kind is not None else ()


def _program_total_kinds(resolved: ResolvedIntent) -> tuple[str, ...] | None:
    """Return program-total kinds when the intent asks whole-program credits."""
    intent = resolved.intent
    if (
        intent.task == "lookup"
        and intent.subject == "program"
        and intent.relation == "credits"
    ):
        return ("program_total_credits",)
    return None


def execute_policy(
    db_path: str | Path,
    resolved: ResolvedIntent,
) -> VerifiedResult:
    """Answer single-kind policy topics from canonical policy evidence."""
    topic = resolved.intent.policy_topic
    if resolved.intent.observed_value is not None:
        return VerifiedResult(
            status="unsupported",
            missing_information=("ไม่สามารถรับรองสิทธิ์รายบุคคลจากค่าที่ให้มาได้",),
            failure_category="EXPECTED_SAFE_FAILURE",
        )
    kinds = _program_total_kinds(resolved) or _policy_kinds_for(topic)
    if not kinds:
        return VerifiedResult(
            status="unsupported",
            missing_information=("หัวข้อนโยบายนี้ยังไม่รองรับในโหมด semantic",),
            failure_category="EXPECTED_SAFE_FAILURE",
        )
    try:
        rendered: list[str] = []
        provenance: list[Any] = []
        program = resolved.scope.program
        for kind in kinds:
            answered = answer_policy_query(
                db_path,
                PolicyQuery(kind, program=program),
                catalog_key=resolved.scope.catalog_key,
            )
            adapted = adapt_policy_answer(answered)
            if adapted.status != "answer":
                return VerifiedResult(
                    status="missing_data",
                    missing_information=(f"policy: {adapted.status}",),
                    failure_category="DATA_ERROR",
                )
            if adapted.final_answer and adapted.final_answer not in rendered:
                rendered.append(adapted.final_answer)
            provenance.extend(adapted.provenance)
        facts = tuple(rendered) if rendered else ()
        if not facts:
            return VerifiedResult(
                status="missing_data",
                provenance=tuple(provenance),
                missing_information=("ไม่มีข้อเท็จนโยบายที่ยืนยันได้",),
                failure_category="DATA_ERROR",
            )
        return VerifiedResult(
            status="answer",
            summary_facts=facts,
            provenance=tuple(provenance),
            failure_category="NONE",
        )
    except Exception:
        return VerifiedResult(
            status="missing_data",
            missing_information=("ไม่สามารถประมวลผลข้อมูลนโยบายได้",),
            failure_category="QUERY_ERROR",
        )


def _legacy_grounding_adapter(
    db_path: str | Path,
    resolution_context: QueryContext,
) -> Callable[[str], dict[str, Any] | None]:
    """Ground a canonical utterance through the deterministic legacy pipeline.

    No model is ever attached (intent_model_callable stays None), so this
    is pure deterministic grounding of already-canonical scope wording —
    never language understanding of student phrasing.
    """
    import rag.qa as legacy_qa

    def ground(utterance: str) -> dict[str, Any] | None:
        try:
            outcome = legacy_qa.ask(db_path, utterance, context=resolution_context)
        except Exception:
            return None
        result = outcome.get("result") if isinstance(outcome, dict) else None
        if result is None:
            return None
        if isinstance(result, dict):
            status = result.get("status")
            final_answer = result.get("final_answer", "")
            raw_provenance = result.get("provenance", ())
            next_context = outcome.get("next_context")
        else:
            status = getattr(result, "status", None)
            final_answer = getattr(result, "final_answer", "")
            raw_provenance = getattr(result, "provenance", ())
            next_context = outcome.get("next_context") if isinstance(outcome, dict) else None
        provenance: list[Any] = []
        if isinstance(raw_provenance, (list, tuple)):
            for reference in raw_provenance:
                if isinstance(reference, Mapping):
                    provenance.append(dict(reference))
        if not isinstance(next_context, dict):
            next_context = None
        return {
            "status": status,
            "final_answer": final_answer if isinstance(final_answer, str) else "",
            "provenance": provenance,
            "next_context": next_context,
        }

    return ground


def _retained_courses_from_context(
    next_context: Any, resolved: ResolvedIntent
) -> tuple[tuple[dict[str, Any], ...], str | None]:
    """Extract bounded canonical retained identities from an ask_sql context."""
    if not isinstance(next_context, dict):
        return (), None
    raw_courses = next_context.get("result_courses")
    entries: list[dict[str, Any]] = []
    if isinstance(raw_courses, (list, tuple)):
        for item in raw_courses[:20]:
            if not isinstance(item, dict):
                continue
            code = item.get("course_code")
            if not isinstance(code, str) or not code.strip():
                continue
            entry: dict[str, Any] = {"course_code": code.strip()}
            for key in ("program", "catalog_key"):
                text = item.get(key)
                if isinstance(text, str) and text.strip():
                    entry[key] = text.strip()
            if "program" not in entry and resolved.scope.program is not None:
                entry["program"] = resolved.scope.program
            if "catalog_key" not in entry and resolved.scope.catalog_key is not None:
                entry["catalog_key"] = resolved.scope.catalog_key
            entries.append(entry)
    scope_program = next_context.get("result_scope_program")
    if not isinstance(scope_program, str) or not scope_program.strip():
        scope_program = resolved.scope.program
    return tuple(entries), scope_program


def _missing_answer_provider(*args: Any, **kwargs: Any) -> str:
    """Stand-in when no answer provider is injected (never a real provider)."""
    return ""


def execute_sql_bridge(
    db_path: str | Path,
    resolved: ResolvedIntent,
    plan: SemanticPlan,
    question: str,
    sql_callable: Callable[..., str],
    answer_callable: Callable[..., str],
    service_context: dict[str, Any],
    *,
    grounding_callable: Callable[[str], dict[str, Any] | None] | None = None,
) -> VerifiedResult:
    """Run compositional queries through guarded ask_sql with verified rows only.

    Acceptance rule (HSQL reuse, no second verifier): a semantic SQL-path
    answer is built ONLY from an ask_sql result with status == "answer"
    AND non-empty provenance — i.e. the existing grounded/verified output
    whose references already passed ask_sql's own provenance validation
    (source document + page per reference). The unverified summary shape
    (answer prose + rows without provenance) is fail-closed, never passed
    to the answerer. Row facts come from canonical row dicts; the summary
    prose is never treated as evidence.
    """
    _ = plan
    from backend.llm_sql_qa import ask_sql

    utterance = synthesize_canonical_utterance(resolved)
    if utterance is None:
        return VerifiedResult(
            status="unsupported",
            missing_information=("รูปแบบคำถามนี้ยังไม่รองรับในโหมด semantic",),
            failure_category="EXPECTED_SAFE_FAILURE",
        )
    resolution_context = QueryContext(
        program=resolved.scope.program,
        catalog_key=resolved.scope.catalog_key,
        plan=resolved.scope.plan,
        years=tuple(resolved.scope.years),
        semesters=tuple(resolved.scope.semesters),
    )
    if grounding_callable is None:
        grounding_callable = _legacy_grounding_adapter(db_path, resolution_context)
    try:
        result = ask_sql(
            db_path,
            utterance,
            resolved.scope.program,
            sql_callable,
            answer_callable,
            conversation_context=service_context or None,
            grounding_callable=grounding_callable,
        )
    except Exception as exc:
        if is_provider_error(exc):
            raise SemanticOperationalError("provider_unavailable") from None
        return VerifiedResult(
            status="missing_data",
            missing_information=("ไม่สามารถประมวลผลคำถามเชิงประกอบได้",),
            failure_category="QUERY_ERROR",
        )
    if not isinstance(result, dict) or result.get("status") != "answer":
        error = result.get("error") if isinstance(result, dict) else None
        code = error.get("code") if isinstance(error, dict) else None
        if code == "sql_model_failure" or (
            code == "answer_model_failure"
            and answer_callable is not _missing_answer_provider
        ):
            # The SQL/answer provider itself failed; this is operational,
            # never evidence insufficiency.
            raise SemanticOperationalError("provider_unavailable") from None
        return VerifiedResult(
            status="missing_data",
            missing_information=("ไม่มีผลลัพธ์ที่ยืนยันได้จากฐานข้อมูล",),
            failure_category="EXPECTED_SAFE_FAILURE",
        )
    raw_provenance = result.get("provenance")
    provenance: list[Any] = []
    if isinstance(raw_provenance, (list, tuple)):
        for reference in raw_provenance:
            if isinstance(reference, Mapping) and reference.get("source_filename"):
                provenance.append(dict(reference))
    rows = result.get("rows")
    if not isinstance(rows, list) or not rows or not provenance:
        # Non-empty factual rows without accepted canonical provenance are
        # fail-closed here: they must never become a grounded success.
        return VerifiedResult(
            status="missing_data",
            missing_information=("แถวข้อมูลขาดแหล่งอ้างอิงที่ยืนยันได้",),
            failure_category="DATA_ERROR",
        )
    facts = _claim_texts_from_rows(rows)
    if not facts:
        return VerifiedResult(
            status="missing_data",
            provenance=tuple(provenance),
            missing_information=("ไม่สามารถสรุปข้อเท็จจริงจากแถวข้อมูลได้",),
            failure_category="GROUNDING_ERROR",
        )
    retained, scope_program = _retained_courses_from_context(
        result.get("next_context"), resolved
    )
    if not retained:
        retained = tuple(
            {"course_code": code, **_scope_entry(resolved)}
            for code in _ordered_unique_codes(rows)
        )[:20]
        scope_program = resolved.scope.program
    return VerifiedResult(
        status="answer",
        summary_facts=facts,
        provenance=tuple(provenance),
        failure_category="NONE",
        result_courses=retained,
        result_scope_program=scope_program,
    )


def _ordered_unique_codes(rows: list[Any]) -> tuple[str, ...]:
    seen: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        for code in _codes(dict(row)):
            if code not in seen:
                seen.append(code)
    return tuple(seen)


def _scope_entry(resolved: ResolvedIntent) -> dict[str, Any]:
    entry: dict[str, Any] = {}
    if resolved.scope.program is not None:
        entry["program"] = resolved.scope.program
    if resolved.scope.catalog_key is not None:
        entry["catalog_key"] = resolved.scope.catalog_key
    return entry


def _connect_ro(db_path: str | Path) -> Any:
    import sqlite3
    from pathlib import Path as _Path

    uri = f"{_Path(db_path).resolve().as_uri()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _single_edition_catalog(db_path: str | Path, program: str) -> str | None:
    try:
        from rag.structured.queries import edition_catalog_keys_for_program

        editions = tuple(edition_catalog_keys_for_program(db_path, program))
    except Exception:
        return None
    if len(editions) != 1 or not isinstance(editions[0], str):
        return None
    return editions[0]


def _catalog_id(db_path: str | Path, catalog_key: str) -> int | None:
    try:
        connection = _connect_ro(db_path)
        try:
            rows = connection.execute(
                "SELECT catalog_id FROM catalogs WHERE lower(trim(catalog_key)) = ?",
                (catalog_key.strip().casefold(),),
            ).fetchall()
        finally:
            connection.close()
    except Exception:
        return None
    if len(rows) != 1:
        return None
    value = rows[0]["catalog_id"]
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _course_credit_value(db_path: str | Path, course_code: str) -> int | None:
    try:
        connection = _connect_ro(db_path)
        try:
            rows = connection.execute(
                "SELECT credits FROM courses WHERE course_code = ?", (course_code,)
            ).fetchall()
        finally:
            connection.close()
    except Exception:
        return None
    if len(rows) != 1:
        return None
    credits = rows[0]["credits"]
    if not isinstance(credits, str):
        return None
    match = re.match(r"\s*(\d+)", credits)
    return int(match.group(1)) if match else None


def _program_total_value(
    db_path: str | Path, program: str, catalog_key: str
) -> tuple[int | None, tuple[Any, ...]]:
    try:
        connection = _connect_ro(db_path)
        try:
            values = connection.execute(
                """SELECT pr.value FROM program_requirements pr
                   JOIN catalogs c ON c.catalog_id = pr.catalog_id
                   WHERE pr.program_code = ? AND lower(trim(c.catalog_key)) = ?
                     AND pr.requirement_type = 'total_program_credits'""",
                (program, catalog_key.strip().casefold()),
            ).fetchall()
            if len(values) != 1:
                return None, ()
            refs = connection.execute(
                """SELECT p.* FROM provenance p
                   JOIN program_requirement_provenance prp ON prp.provenance_id = p.provenance_id
                   JOIN program_requirements pr ON pr.requirement_id = prp.requirement_id
                   JOIN catalogs c ON c.catalog_id = pr.catalog_id
                   WHERE pr.program_code = ? AND lower(trim(c.catalog_key)) = ?
                     AND pr.requirement_type = 'total_program_credits'""",
                (program, catalog_key.strip().casefold()),
            ).fetchall()
        finally:
            connection.close()
    except Exception:
        return None, ()
    value = values[0]["value"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None, ()
    provenance = tuple(dict(row) for row in refs)
    if not provenance:
        return None, ()
    return int(value), provenance


def _side_label(side: Any) -> str:
    target = getattr(side, "target", None)
    if target is not None and getattr(target, "course_code", None):
        return str(target.course_code)
    scope = getattr(side, "scope", None)
    parts: list[str] = []
    if scope is not None:
        if getattr(scope, "program", None):
            parts.append(str(scope.program))
        if getattr(scope, "plan", None):
            parts.append(f"แผน{_plan_display(scope.plan)}")
        for year in tuple(getattr(scope, "years", ()) or ()):
            parts.append(f"ปี {year}")
        for semester in tuple(getattr(scope, "semesters", ()) or ()):
            parts.append(f"เทอม {semester}")
        if getattr(scope, "catalog_key", None):
            parts.append(str(scope.catalog_key))
    return " ".join(parts) or "ฝั่งเปรียบเทียบ"


def _side_credits(
    db_path: str | Path, side: Any
) -> tuple[int | None, tuple[Any, ...], str | None]:
    """Resolve one comparison side to verified credit int + provenance."""
    target = getattr(side, "target", None)
    scope = getattr(side, "scope", None)
    if target is not None and getattr(target, "course_code", None):
        code = str(target.course_code)
        if scope is None or not getattr(scope, "program", None):
            return None, (), "course comparison needs authoritative program scope"
        from rag.structured.queries import course_facts

        try:
            result = course_facts(db_path, code, program=scope.program)
        except Exception:
            return None, (), "course credit/evidence unavailable"
        courses = result.get("courses", ()) if isinstance(result, Mapping) else ()
        expected_catalog_id = (
            _catalog_id(db_path, scope.catalog_key)
            if scope.catalog_key is not None
            else None
        )
        if scope.catalog_key is not None and expected_catalog_id is None:
            return None, (), "course comparison catalog is invalid"
        matches = [
            item for item in courses
            if isinstance(item, Mapping)
            and (expected_catalog_id is None or item.get("catalog_id") == expected_catalog_id)
        ]
        if len(matches) != 1:
            return None, (), "course identity ambiguous in comparison scope"
        course = matches[0]
        credit_text = course.get("credits") or course.get("credit_units")
        if isinstance(credit_text, (int, float)) and not isinstance(credit_text, bool):
            value = int(credit_text)
        elif isinstance(credit_text, str):
            match = re.match(r"\s*(\d+)", credit_text)
            value = int(match.group(1)) if match else None
        else:
            value = None
        provenance = tuple(
            dict(reference)
            for reference in tuple(course.get("provenance", ()) or ())
            if isinstance(reference, Mapping)
        )
        if value is None or not provenance:
            return None, (), "course credit/evidence unavailable"
        return value, provenance, None
    if scope is None:
        return None, (), "operand without scope"
    program = getattr(scope, "program", None)
    catalog_key = getattr(scope, "catalog_key", None)
    plan = getattr(scope, "plan", None)
    years = tuple(getattr(scope, "years", ()) or ())
    semesters = tuple(getattr(scope, "semesters", ()) or ())
    if not program or (not years and not semesters):
        # Bare program totals use requirement evidence, handled by callers
        # through _program_total_value; anything else needs term scope.
        if program and not years and not semesters and plan is None:
            catalog = catalog_key or (
                _single_edition_catalog(db_path, program)
                if isinstance(program, str)
                else None
            )
            if catalog is None:
                return None, (), "program total needs an authoritative catalog"
            value, provenance = _program_total_value(db_path, program, catalog)
            if value is None:
                return None, (), "program total unavailable"
            return value, provenance, None
        return None, (), "numeric side needs plan-scoped term evidence"
    if catalog_key is None:
        catalog_key = (
            _single_edition_catalog(db_path, program)
            if isinstance(program, str)
            else None
        )
        if catalog_key is None:
            return None, (), "term aggregate needs an authoritative catalog"
    if plan is None:
        return None, (), "numeric comparison needs explicit plan scope"
    from rag.evidence_executor import execute_evidence_plan
    from rag.evidence_planner import plan_evidence
    from rag.query_spec import QuerySpec
    from rag.resolution import QueryContext, resolve_query_spec

    spec = QuerySpec(
        original_question="",
        normalized_question="",
        program=program,
        plans=(plan,),
        years=tuple(years),
        semesters=tuple(semesters),
        course_codes=(),
        course_name=None,
        category=None,
        topic=None,
        operations=("sum_credits",),
        group_by=(),
        judgement="none",
        credit_units=None,
        references_previous_result_set=False,
        result_ordinal=None,
    )
    context = QueryContext(program=program, catalog_key=catalog_key, plan=plan,
                           years=tuple(years), semesters=tuple(semesters))
    try:
        resolution = resolve_query_spec(spec, db_path, context=context)
        if resolution.action != "answer":
            return None, (), "term scope did not resolve"
        plan_evidence_result = plan_evidence(spec, resolution, catalog_key=catalog_key)
        bundle = execute_evidence_plan(db_path, plan_evidence_result)
        claims = _legacy_compose_claims(spec, bundle)
    except Exception:
        return None, (), "term aggregate failed"
    numbers: list[Any] = []
    provenance: list[Any] = []
    for claim in claims:
        try:
            value = getattr(claim, "value", None)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if value not in numbers:
                    numbers.append(value)
            for reference in tuple(getattr(claim, "provenance", ()) or ()):
                if isinstance(reference, Mapping) and reference not in provenance:
                    provenance.append(dict(reference))
        except Exception:
            continue
    if len(numbers) != 1 or not provenance:
        return None, (), "term aggregate ambiguous or unverified"
    number = numbers[0]
    if isinstance(number, bool) or not isinstance(number, (int, float)):
        return None, (), "term aggregate not numeric"
    return int(number), tuple(provenance), None


def _side_course_set(
    db_path: str | Path, side: Any
) -> tuple[tuple[tuple[str, int], ...] | None, str | None]:
    """Resolve one side to catalog-scoped canonical course identities.

    Course-code strings define cross-edition set equality only when the code
    is identical; differently coded rows are never equated. Each catalog's
    course_id is retained separately for provenance hydration.
    """
    scope = getattr(side, "scope", None)
    if scope is None:
        return None, "operand without scope"
    program = getattr(scope, "program", None)
    catalog_key = getattr(scope, "catalog_key", None)
    plan = getattr(scope, "plan", None)
    if catalog_key is None and program is None:
        return None, "set operand needs catalog or program scope"
    try:
        connection = _connect_ro(db_path)
        try:
            sql = """SELECT DISTINCT c.course_id, c.course_code FROM courses c
                     JOIN plan_placements pp ON pp.course_id = c.course_id
                     JOIN curriculum_plans cp ON cp.plan_id = pp.plan_id
                     JOIN catalogs c2 ON c2.catalog_id = cp.catalog_id
                     JOIN programs p ON p.program_id = cp.program_id
                     WHERE 1 = 1"""
            args: list[Any] = []
            if catalog_key is not None:
                sql += " AND lower(trim(c2.catalog_key)) = ?"
                args.append(str(catalog_key).strip().casefold())
            if program is not None:
                sql += " AND p.program_code = ?"
                args.append(program)
            if plan is not None:
                sql += " AND cp.plan_key = ?"
                args.append(plan)
            rows = connection.execute(sql, args).fetchall()
        finally:
            connection.close()
    except Exception:
        return None, "course set query failed"
    identities = tuple(
        (row["course_code"], row["course_id"])
        for row in rows
        if isinstance(row["course_code"], str)
        and row["course_code"].strip()
        and isinstance(row["course_id"], int)
        and not isinstance(row["course_id"], bool)
    )
    from rag.aggregation import is_masked_course_code

    if any(is_masked_course_code(code) for code, _ in identities):
        return None, "course set contains masked placeholder identities"
    if len(identities) > 200:
        return None, "course set exceeds bounded comparison size"
    return identities, None


def _hydrated_provenance(
    db_path: str | Path, course_ids: tuple[int, ...]
) -> tuple[Any, ...] | None:
    if not course_ids:
        return ()
    try:
        from backend.llm_sql_qa import hydrate_sql_row_provenance
    except Exception:
        return None
    identities = [{"course_id": course_id} for course_id in course_ids]
    try:
        hydration = hydrate_sql_row_provenance(db_path, identities)
    except Exception:
        return None
    if (
        getattr(hydration, "status", None) != "complete"
        or getattr(hydration, "covered_rows", 0) != len(identities)
        or not getattr(hydration, "provenance", None)
    ):
        return None
    return tuple(dict(ref) for ref in hydration.provenance)


def execute_comparison(db_path: str | Path, resolved: ResolvedIntent) -> VerifiedResult:
    """Execute verified numeric/set comparison with per-side provenance."""
    comparison = resolved.intent.comparison
    if comparison is None or comparison.operation not in {
        "greater", "less", "equal", "difference", "set_difference", "overlap",
    }:
        return VerifiedResult(
            status="unsupported",
            missing_information=("รูปแบบการเปรียบเทียบนี้ยังไม่รองรับ",),
            failure_category="EXPECTED_SAFE_FAILURE",
        )
    sides = tuple(resolved.comparison_sides)
    if len(sides) != 2 or any(getattr(side, "unresolved", True) for side in sides):
        return VerifiedResult(
            status="missing_data",
            missing_information=("ตัวเปรียบเทียบยังยืนยันไม่ได้",),
            failure_category="EXPECTED_SAFE_FAILURE",
        )
    left, right = sides
    if comparison.operation in {"greater", "less", "equal", "difference"}:
        if comparison.measure != "credits":
            return VerifiedResult(
                status="unsupported",
                missing_information=("มาตรการเปรียบเทียบนี้ยังไม่รองรับ",),
                failure_category="EXPECTED_SAFE_FAILURE",
            )
        left_value, left_prov, left_error = _side_credits(db_path, left)
        right_value, right_prov, right_error = _side_credits(db_path, right)
        if left_value is None or right_value is None:
            return VerifiedResult(
                status="missing_data",
                missing_information=(left_error or right_error or "หาค่าเปรียบเทียบไม่ได้",),
                failure_category="EXPECTED_SAFE_FAILURE",
            )
        left_label, right_label = _side_label(left), _side_label(right)
        if comparison.operation == "greater":
            verdict = f"{left_label} มากกว่า {right_label}: {left_value} ต่อ {right_value} หน่วยกิต" if left_value > right_value else f"{left_label} ไม่มากกว่า {right_label}: {left_value} ต่อ {right_value} หน่วยกิต"
            correct = left_value > right_value
        elif comparison.operation == "less":
            verdict = f"{left_label} น้อยกว่า {right_label}: {left_value} ต่อ {right_value} หน่วยกิต" if left_value < right_value else f"{left_label} ไม่น้อยกว่า {right_label}: {left_value} ต่อ {right_value} หน่วยกิต"
            correct = left_value < right_value
        elif comparison.operation == "equal":
            verdict = f"{left_label} เท่ากับ {right_label}: {left_value} หน่วยกิต" if left_value == right_value else f"{left_label} ไม่เท่ากับ {right_label}: {left_value} ต่อ {right_value} หน่วยกิต"
            correct = left_value == right_value
        else:
            verdict = f"{left_label} กับ {right_label} ต่างกัน {abs(left_value - right_value)} หน่วยกิต ({left_value} ต่อ {right_value})"
            correct = True
        _ = correct
        left_target = getattr(left, "target", None)
        right_target = getattr(right, "target", None)
        actual_relation = (
            "left_greater"
            if left_value > right_value
            else "right_greater"
            if left_value < right_value
            else "equal"
        )
        numeric_comparison = VerifiedNumericComparison(
            measure=comparison.measure,
            requested_operation=comparison.operation,
            actual_relation=actual_relation,
            left=VerifiedNumericComparisonSide(
                label=left_label,
                course_code=getattr(left_target, "course_code", None),
                course_name=getattr(left_target, "course_name", None),
                value=left_value,
            ),
            right=VerifiedNumericComparisonSide(
                label=right_label,
                course_code=getattr(right_target, "course_code", None),
                course_name=getattr(right_target, "course_name", None),
                value=right_value,
            ),
            absolute_difference=abs(left_value - right_value),
        )
        return VerifiedResult(
            status="answer",
            summary_facts=(
                f"{left_label}: {left_value} หน่วยกิต",
                f"{right_label}: {right_value} หน่วยกิต",
                verdict,
            ),
            provenance=tuple(left_prov) + tuple(right_prov),
            failure_category="NONE",
            numeric_comparison=numeric_comparison,
        )
    left_codes, left_error = _side_course_set(db_path, left)
    right_codes, right_error = _side_course_set(db_path, right)
    if left_codes is None or right_codes is None:
        return VerifiedResult(
            status="missing_data",
            missing_information=(left_error or right_error or "หาชุดวิชาเปรียบเทียบไม่ได้",),
            failure_category="EXPECTED_SAFE_FAILURE",
        )
    left_map = {code: course_id for code, course_id in left_codes}
    right_map = {code: course_id for code, course_id in right_codes}
    left_set, right_set = set(left_map), set(right_map)
    if comparison.operation == "set_difference":
        only = sorted(left_set - right_set)
        verdict = "วิชาที่อยู่ในขอบเขตซ้ายแต่ไม่อยู่ในขอบเขตขวา: %d วิชา%s" % (
            len(only), (": " + ", ".join(only[:20])) if only else "")
        result_codes = tuple(only)
    else:
        both = sorted(left_set & right_set)
        verdict = "วิชาที่อยู่ทั้งสองขอบเขต: %d วิชา%s" % (
            len(both), (": " + ", ".join(both[:20])) if both else "")
        result_codes = tuple(both)
    # Membership and absence in a set operation are grounded by the
    # canonical input sets, not only the output members. Hydrate every
    # bounded source identity from both sides so even a valid-empty
    # difference/overlap is auditable and has non-empty provenance.
    evidence_ids = tuple(left_map.values()) + tuple(right_map.values())
    provenance = _hydrated_provenance(db_path, evidence_ids)
    if provenance is None or not provenance:
        return VerifiedResult(
            status="missing_data",
            missing_information=("ชุดวิชาเปรียบเทียบขาดแหล่งอ้างอิงที่ยืนยันได้",),
            failure_category="DATA_ERROR",
        )
    return VerifiedResult(
        status="answer",
        summary_facts=(verdict,) if verdict else (),
        provenance=provenance,
        failure_category="NONE",
    )


def _retained_from_claims(
    claims: tuple[Any, ...],
    program: str | None,
    catalog_key: str | None,
) -> tuple[tuple[dict[str, Any], ...], str | None]:
    """Retain bounded canonical course identities from verified claims only.

    Mirrors the legacy retained-result contract: entries carry an exact
    course code plus already-validated scope, capped for bounded storage.
    Only verified claim values are read — never model text, never answer
    prose, never unresolved aliases.
    """
    seen: list[str] = []
    for claim in claims:
        try:
            if (
                getattr(claim, "operation", None) in {"list", "topic_matches", "course_set"}
                and getattr(claim, "status", None) == "complete"
            ):
                codes = _collection_course_codes(getattr(claim, "value", None))
            else:
                codes = _codes(getattr(claim, "value", None))
        except Exception:
            continue
        for code in codes:
            if code not in seen:
                seen.append(code)
    entries: list[dict[str, Any]] = []
    for code in seen[:20]:
        entry: dict[str, Any] = {"course_code": code}
        if program is not None:
            entry["program"] = program
        if catalog_key is not None:
            entry["catalog_key"] = catalog_key
        entries.append(entry)
    return tuple(entries), program


def _claim_texts_from_rows(rows: list[Any]) -> tuple[str, ...]:
    facts: list[str] = []
    for row in rows[:20]:
        if not isinstance(row, Mapping):
            continue
        codes = _codes(dict(row))
        titles = _titles(dict(row))
        numbers = _numbers(dict(row))
        label = codes[0] if codes else (titles[0] if titles else "")
        detail = " | ".join([label] + numbers[:3]).strip(" |")
        if detail:
            facts.append(detail)
    return tuple(facts)


__all__ = [
    "execute_comparison",
    "execute_deterministic",
    "execute_policy",
    "execute_sql_bridge",
]
