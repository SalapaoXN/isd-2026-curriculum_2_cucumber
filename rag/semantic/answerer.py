"""Evidence-bounded final answerer for semantic mode.

The answerer presents only verified facts in natural Thai. It must not add
curriculum facts, change numbers, or infer missing information. A
deterministic post-answer validation layer checks that required canonical
identifiers survived presentation and that no internal implementation
jargon leaked; on any violation the deterministic grounded renderer is
used instead. There is deliberately NO second LLM call to repair
presentation, and the fallback adds zero new factual authority.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping

from rag.answer import _plan_display
from rag.semantic.prompts import (
    ANSWERER_PROMPT_VERSION,
    build_semantic_answerer_prompt,
)
from rag.semantic.schema import (
    ResolvedIntent,
    VerifiedNumericComparison,
    VerifiedNumericComparisonSide,
    VerifiedResult,
)

MAX_ANSWER_LEN = 2000

_FORBIDDEN_JARGON = (
    "placement",
    "prerequisite",
    "sum_credits",
    "resolved intent",
    "resolved_intent",
    "query plan",
    "query_plan",
    "evidence row",
    "sql row",
    "semantic intent",
    "semantic_intent",
    "result courses",
    "result_courses",
    "provenance",
)

_COURSE_CODE_RE = re.compile(r"\d{8}")
_INTERNAL_PLAN_RE = re.compile(r"(?<![A-Za-z0-9_])(?:no_coop|coop)(?![A-Za-z0-9_])", re.IGNORECASE)

_TITLE_KEYS = ("name_en", "name_th", "title", "course_name")


def _scan_texts(verified: VerifiedResult) -> list[str]:
    texts: list[str] = []
    for fact in verified.summary_facts:
        if isinstance(fact, str) and fact:
            texts.append(fact)
    return texts


def _scan_claim_values(verified: VerifiedResult) -> list[object]:
    values: list[object] = []
    for claim in verified.claims:
        values.append(getattr(claim, "value", None))
    return values


def _collect_codes(verified: VerifiedResult) -> tuple[str, ...]:
    found: list[str] = []
    for text in _scan_texts(verified):
        for code in _COURSE_CODE_RE.findall(text):
            if code not in found:
                found.append(code)
    return tuple(found)


def _collect_titles(verified: VerifiedResult) -> tuple[str, ...]:
    found: list[str] = []
    for value in _scan_claim_values(verified):
        for title in _titles_from(value):
            if title not in found:
                found.append(title)
    return tuple(found)


def _titles_from(value: object) -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key in _TITLE_KEYS:
            text = value.get(key)
            if isinstance(text, str) and text.strip() and text.strip() not in found:
                found.append(text.strip())
                break
        for item in value.values():
            if isinstance(item, (Mapping, list, tuple)):
                found.extend(t for t in _titles_from(item) if t not in found)
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.extend(t for t in _titles_from(item) if t not in found)
    return found[:4]


def _displayed_list_pairs(verified: VerifiedResult) -> tuple[tuple[str, str | None], ...]:
    from rag.semantic.executor import _collection_course_identities

    pairs: list[tuple[str, str | None]] = []
    for claim in verified.claims:
        if (
            getattr(claim, "operation", None) in {"list", "topic_matches", "course_set"}
            and getattr(claim, "status", None) == "complete"
        ):
            pairs.extend(_collection_course_identities(getattr(claim, "value", None))[:10])
    return tuple(pairs)


def _required_identifiers(verified: VerifiedResult) -> tuple[str, ...]:
    """Canonical identifiers an answer must preserve verbatim.

    Every distinct course code in verified evidence is required: a grounded
    course-level answer must name its entities. Canonical titles are
    additionally required for single-course answers, where omitting the
    title leaves the answer ambiguous. Displayed collection identities require
    their selected grounded title, without requiring both language variants.
    """
    codes = _collect_codes(verified)
    pairs = _displayed_list_pairs(verified)
    if pairs:
        return codes + tuple(dict.fromkeys(title for _, title in pairs if title))
    if len(codes) != 1:
        return codes
    titles = _collect_titles(verified)
    return codes + tuple(title for title in titles[:2] if title not in codes)


def validate_answer_text(answer: str, verified: VerifiedResult) -> bool:
    """Return True when rendered text preserves identifiers without jargon."""
    if not isinstance(answer, str) or not answer.strip():
        return False
    lowered = answer.casefold()
    plan_labels = tuple(
        f"แผน{_plan_display(plan)}" for plan in ("coop", "no_coop")
        if any(f"แผน{_plan_display(plan)}" in fact for fact in _scan_texts(verified))
    )
    if plan_labels and (
        _INTERNAL_PLAN_RE.search(answer) or any(label not in answer for label in plan_labels)
    ):
        return False
    for token in _FORBIDDEN_JARGON:
        if token in lowered:
            return False
    for identifier in _required_identifiers(verified):
        if identifier not in answer:
            return False
    for code, title in _displayed_list_pairs(verified):
        if title and not any(
            code in line and title in line
            and set(_COURSE_CODE_RE.findall(line)) == {code}
            for line in answer.splitlines()
        ):
            # Do not accept names detached from or paired with other codes.
            return False
    return True


def _comparison_side_label(side: VerifiedNumericComparisonSide) -> str:
    if side.course_name and side.course_code:
        return f"{side.course_name} ({side.course_code})"
    if side.course_name:
        return side.course_name
    if side.course_code:
        return side.course_code
    return side.label


def _format_comparison_value(value: int | float) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _both_sides_have_course_identity(
    comparison: VerifiedNumericComparison,
) -> bool:
    return all(
        bool(side.course_code or side.course_name)
        for side in (comparison.left, comparison.right)
    )


def _verified_relation_sentence(
    comparison: VerifiedNumericComparison,
) -> str:
    left = comparison.left.course_name or comparison.left.label
    right = comparison.right.course_name or comparison.right.label
    if comparison.actual_relation == "equal":
        if comparison.measure == "credits":
            if _both_sides_have_course_identity(comparison):
                return "ทั้งสองวิชามีจำนวนหน่วยกิตเท่ากัน"
            return "ทั้งสองฝั่งมีจำนวนหน่วยกิตเท่ากัน"
        return "ทั้งสองฝั่งมีค่าเท่ากัน"
    if comparison.actual_relation == "left_greater":
        if comparison.measure == "credits":
            return f"{left} มีหน่วยกิตมากกว่า {right}"
        return f"{left} มีค่ามากกว่า {right}"
    if comparison.actual_relation == "right_greater":
        if comparison.measure == "credits":
            return f"{right} มีหน่วยกิตมากกว่า {left}"
        return f"{right} มีค่ามากกว่า {left}"
    # This schema version uses the actual relation enum above; fail closed
    # for any unknown value rather than inventing comparison language.
    return "ไม่สามารถสรุปความสัมพันธ์จากข้อมูลที่ตรวจสอบได้"


def _comparison_heading(comparison: VerifiedNumericComparison) -> str:
    if comparison.measure == "credits":
        return "เปรียบเทียบหน่วยกิต"
    if comparison.measure == "course_count":
        return "เปรียบเทียบจำนวนรายวิชา"
    if comparison.measure == "prerequisite_count":
        return "เปรียบเทียบจำนวนวิชาบังคับก่อน"
    return "ผลการเปรียบเทียบ"


def render_verified_comparison(
    verified: VerifiedResult,
    comparison: VerifiedNumericComparison | None = None,
) -> str:
    """Deterministic comparison fallback built only from typed verified facts."""
    comparison = comparison or verified.numeric_comparison
    if verified.status != "answer" or comparison is None:
        return render_verified_fallback("", verified)
    left_label = _comparison_side_label(comparison.left)
    right_label = _comparison_side_label(comparison.right)
    left_value = _format_comparison_value(comparison.left.value)
    right_value = _format_comparison_value(comparison.right.value)
    lines = [
        _comparison_heading(comparison),
        f"- {left_label}: {left_value} หน่วยกิต" if comparison.measure == "credits" else f"- {left_label}: {left_value}",
        f"- {right_label}: {right_value} หน่วยกิต" if comparison.measure == "credits" else f"- {right_label}: {right_value}",
    ]
    if comparison.requested_operation == "difference":
        difference = _format_comparison_value(comparison.absolute_difference)
        summary = f"มีผลต่าง {difference} หน่วยกิต" if comparison.measure == "credits" else f"มีผลต่าง {difference}"
    else:
        summary = _verified_relation_sentence(comparison)
    lines.extend(("", f"สรุป: {summary}"))
    return "\n".join(lines)[:MAX_ANSWER_LEN]


def _comparison_answer_valid(
    answer: str, comparison: VerifiedNumericComparison
) -> bool:
    """Conservatively validate identifiers, values, and the explicit verdict."""
    if "สรุป:" not in answer:
        return False
    sides = (comparison.left, comparison.right)
    for side in sides:
        for identifier in (side.course_code, side.course_name):
            if identifier and identifier.casefold() not in answer.casefold():
                return False
        formatted = _format_comparison_value(side.value)
        required_value = (
            f"{formatted} หน่วยกิต"
            if comparison.measure == "credits"
            else formatted
        )
        if required_value not in answer:
            return False
        if side.course_code and answer.count(side.course_code) != 1:
            return False
    summary = answer.split("สรุป:", 1)[1].strip()
    if comparison.actual_relation == "equal":
        expected_relation = _verified_relation_sentence(comparison)
        if expected_relation not in summary:
            return False
        if "มากกว่า" in summary or "น้อยกว่า" in summary:
            return False
    elif comparison.requested_operation == "difference":
        difference = _format_comparison_value(comparison.absolute_difference)
        if difference not in summary:
            return False
    else:
        expected_relation = _verified_relation_sentence(comparison)
        if expected_relation not in summary:
            return False
    return True


def render_verified_fallback(
    question: str, verified: VerifiedResult
) -> str:
    """Render verified facts deterministically without any provider call."""
    lines = [fact for fact in verified.summary_facts if fact][:10]
    if _displayed_list_pairs(verified):
        body = "\n".join(lines)
    else:
        body = "\n".join(f"- {line}" for line in lines) if lines else "- ไม่พบข้อเท็จจริงที่ยืนยันได้"
    missing = (
        "\nข้อมูลที่ยังขาด: " + "; ".join(verified.missing_information)
        if verified.missing_information
        else ""
    )
    _ = question
    return f"{body}{missing}"[:MAX_ANSWER_LEN]


def _comparison_operand_label(side: object) -> str | None:
    """Format only canonical resolved identity/scope for one comparison side."""
    target = getattr(side, "target", None)
    scope = getattr(side, "scope", None)
    if target is not None:
        name = getattr(target, "course_name", None)
        code = getattr(target, "course_code", None)
        if isinstance(name, str) and name.strip():
            if isinstance(code, str) and code.strip():
                return f"{name.strip()} ({code.strip()})"
            return name.strip()
        if isinstance(code, str) and code.strip():
            return code.strip()
    if scope is None:
        return None
    parts: list[str] = []
    for field, label in (("program", None), ("plan", "แผน")):
        value = getattr(scope, field, None)
        if isinstance(value, str) and value.strip():
            parts.append(f"{label}{_plan_display(value.strip())}" if label else value.strip())
    for year in tuple(getattr(scope, "years", ()) or ()):
        parts.append(f"ชั้นปีที่ {year}")
    for semester in tuple(getattr(scope, "semesters", ()) or ()):
        parts.append(f"ภาคการศึกษาที่ {semester}")
    catalog_key = getattr(scope, "catalog_key", None)
    if isinstance(catalog_key, str) and catalog_key.strip():
        parts.append(catalog_key.strip())
    return " ".join(parts) or None


def _looks_unsafe(text: str) -> bool:
    lowered = text.casefold()
    return "select " in lowered and "from " in lowered


def render_semantic_answer(
    question: str,
    verified: VerifiedResult,
    answer_callable: Callable[..., str] | None,
    *,
    numeric_comparison: VerifiedNumericComparison | None = None,
) -> tuple[str, str]:
    """Return (answer_text, answer_mode) for verified evidence.

    ``answer_mode`` is ``grounded_synthesis`` when the provider rendered the
    text and ``deterministic`` when the fallback renderer was used.
    """
    if verified.status != "answer" or not verified.summary_facts:
        return "", "deterministic"
    comparison = numeric_comparison or verified.numeric_comparison
    if answer_callable is None:
        if comparison is not None:
            return render_verified_comparison(verified, comparison), "deterministic"
        return render_verified_fallback(question, verified), "deterministic"
    try:
        prompt = build_semantic_answerer_prompt(
            question,
            verified.summary_facts,
            verified.missing_information,
            numeric_comparison=comparison,
        )
        text = answer_callable(prompt)
    except Exception:
        if comparison is not None:
            return render_verified_comparison(verified, comparison), "deterministic"
        return render_verified_fallback(question, verified), "deterministic"
    if (
        not isinstance(text, str)
        or not text.strip()
        or len(text) > MAX_ANSWER_LEN
        or _looks_unsafe(text)
        or not validate_answer_text(text, verified)
        or (comparison is not None and not _comparison_answer_valid(text, comparison))
    ):
        if comparison is not None:
            return render_verified_comparison(verified, comparison), "deterministic"
        return render_verified_fallback(question, verified), "deterministic"
    return text.strip(), "grounded_synthesis"


__all__ = [
    "ANSWERER_PROMPT_VERSION",
    "MAX_ANSWER_LEN",
    "render_semantic_answer",
    "render_verified_comparison",
    "render_verified_fallback",
    "validate_answer_text",
]
