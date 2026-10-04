"""Deterministic single-turn routing for policy questions (H25-P1/H25-P3).

Pure boundary: parse the question with the existing bounded policy parser and,
for routed kinds, answer from the policy authority and adapt the result into
the public answer contract. No LLM, no QueryContext, no new semantics.
"""

from __future__ import annotations

from pathlib import Path

from rag.grounded_answer import GroundedAnswerResult
from rag.query_spec import parse_query_spec
from rag.structured.queries import edition_catalog_keys_for_program

from .answer import PolicyAnswer, answer_policy_question
from .query import parse_policy_question


POLICY_ROUTE_ALLOWLIST = frozenset(
    {
        "registration_regular_max",
        "registration_regular_min",
        "registration_exception_max",
        "registration_special_max",
        "probation_entry",
        "probation_cleared",
        "honors_first",
        "honors_second",
        "reentry_limit",
        "leave_of_absence",
        "resignation",
        "credit_transfer",
        "exam_dishonesty_penalty",
        "disciplinary_penalties",
        "sanction_appeal_deadline",
        "sanction_appeal_procedure",
        "graduation_requirements",
        "graduation_gpa",
        "graduation_english_exit",
        "graduation_no_debt",
        "student_status_termination_reasons",
        "student_status_termination_gpa",
        "gpa_calculation_method",
        "assessment_method",
        "student_conduct_rules",
        "serious_disciplinary_offenses",
    }
)

POLICY_GT_CATEGORY_COVERAGE = {
    "เกณฑ์การสำเร็จการศึกษา": frozenset(
        {
            "graduation_requirements",
            "graduation_gpa",
            "graduation_english_exit",
            "graduation_no_debt",
            "program_total_credits",
        }
    ),
    "เกณฑ์เกียรตินิยม": frozenset({"honors_first", "honors_second"}),
    "เกณฑ์พ้นสภาพนักศึกษา": frozenset(
        {"student_status_termination_reasons", "student_status_termination_gpa"}
    ),
    "เกณฑ์ภาคทัณฑ์ (probation)": frozenset(
        {"probation_entry", "probation_cleared"}
    ),
    "ระบบเกรด/การคิดคะแนน": frozenset({"gpa_calculation_method"}),
    "เกณฑ์การลงทะเบียน": frozenset(
        {
            "registration_regular_max",
            "registration_regular_min",
            "registration_exception_max",
            "registration_special_max",
            "registration_compare",
        }
    ),
    "การลาพักการศึกษา": frozenset({"leave_of_absence"}),
    "การลาออก": frozenset({"resignation"}),
    "การสอบ/วัดผล": frozenset({"assessment_method"}),
    "การทุจริตทางวิชาการ": frozenset({"exam_dishonesty_penalty"}),
    "ระเบียบความประพฤติ": frozenset(
        {"student_conduct_rules", "serious_disciplinary_offenses"}
    ),
    "บทลงโทษทางวินัย": frozenset({"disciplinary_penalties"}),
    "การอุทธรณ์": frozenset(
        {"sanction_appeal_deadline", "sanction_appeal_procedure"}
    ),
    "การกลับเข้าศึกษา": frozenset({"reentry_limit"}),
    "การเทียบโอนหน่วยกิต": frozenset({"credit_transfer"}),
}


_POLICY_STATUS_MAP = {
    "complete": "answer",
    "unsupported": "unsupported",
    "insufficient_evidence": "insufficient_evidence",
    "invalid_query": "unsupported",
}


def adapt_policy_answer(answer: PolicyAnswer) -> GroundedAnswerResult:
    """Map a policy answer into the public answer contract without new semantics."""
    if not isinstance(answer, PolicyAnswer):
        raise TypeError("answer must be a PolicyAnswer")
    return GroundedAnswerResult(
        status=_POLICY_STATUS_MAP.get(answer.status, "unsupported"),
        answer_mode="deterministic",
        final_answer=answer.rendered_answer,
        claims=(),
        provenance=answer.provenance,
    )


def route_policy_question(
    db_path: str | Path,
    question: str,
    *,
    catalog_key: str | None = None,
) -> GroundedAnswerResult | None:
    """Answer routed policy questions, or return None to keep existing behavior."""
    query = parse_policy_question(question)
    if query is None:
        return None
    if query.kind in POLICY_ROUTE_ALLOWLIST:
        return adapt_policy_answer(answer_policy_question(db_path, question))
    if query.kind == "program_total_credits":
        # H25-P3 R1: axis-free shapes ask the graduation requirement (policy);
        # any explicit curriculum axis keeps the existing scoped-sum path.
        spec = parse_query_spec(question)
        if _has_explicit_curriculum_axis(spec):
            return None
        if query.program and catalog_key is None and edition_catalog_keys_for_program(
            db_path, query.program
        ):
            return adapt_policy_answer(
                PolicyAnswer(status="insufficient_evidence", query_type=query.kind)
            )
        return adapt_policy_answer(
            answer_policy_question(db_path, question, catalog_key=catalog_key)
        )
    if query.kind == "registration_compare":
        # H25-P3 R2: a ("list",) parse would be a genuine dual (filtered list
        # vs registration verdict) — fail closed without touching the DB.
        # All other shapes never complete on the curriculum side (R1-partial).
        spec = parse_query_spec(question)
        if tuple(spec.operations) == ("list",):
            return adapt_policy_answer(
                PolicyAnswer(status="insufficient_evidence", query_type=query.kind)
            )
        return adapt_policy_answer(answer_policy_question(db_path, question))
    return None


def _has_explicit_curriculum_axis(spec: object) -> bool:
    """Return True when the curriculum parse carries any user-stated scope axis."""
    return bool(
        getattr(spec, "years", ())
        or getattr(spec, "semesters", ())
        or getattr(spec, "plans", ())
        or getattr(spec, "category", None)
        or getattr(spec, "course_codes", ())
        or getattr(spec, "course_name", None)
    )


__all__ = [
    "POLICY_GT_CATEGORY_COVERAGE",
    "POLICY_ROUTE_ALLOWLIST",
    "adapt_policy_answer",
    "route_policy_question",
]
