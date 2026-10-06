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

from rag.semantic.prompts import (
    ANSWERER_PROMPT_VERSION,
    build_semantic_answerer_prompt,
)
from rag.semantic.schema import VerifiedResult

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


def _required_identifiers(verified: VerifiedResult) -> tuple[str, ...]:
    """Canonical identifiers an answer must preserve verbatim.

    Every distinct course code in verified evidence is required: a grounded
    course-level answer must name its entities. Canonical titles are
    additionally required for single-course answers, where omitting the
    title leaves the answer ambiguous. Multi-course listings keep codes
    only, so legitimate summaries are not forced to repeat every title.
    """
    codes = _collect_codes(verified)
    if len(codes) != 1:
        return codes
    titles = _collect_titles(verified)
    return codes + tuple(title for title in titles[:2] if title not in codes)


def validate_answer_text(answer: str, verified: VerifiedResult) -> bool:
    """Return True when rendered text preserves identifiers without jargon."""
    if not isinstance(answer, str) or not answer.strip():
        return False
    lowered = answer.casefold()
    for token in _FORBIDDEN_JARGON:
        if token in lowered:
            return False
    for identifier in _required_identifiers(verified):
        if identifier not in answer:
            return False
    return True


def render_verified_fallback(
    question: str, verified: VerifiedResult
) -> str:
    """Render verified facts deterministically without any provider call."""
    lines = [fact for fact in verified.summary_facts if fact][:10]
    body = "\n".join(f"- {line}" for line in lines) if lines else "- ไม่พบข้อเท็จจริงที่ยืนยันได้"
    missing = (
        "\nข้อมูลที่ยังขาด: " + "; ".join(verified.missing_information)
        if verified.missing_information
        else ""
    )
    _ = question
    return f"{body}{missing}"[:MAX_ANSWER_LEN]


def _looks_unsafe(text: str) -> bool:
    lowered = text.casefold()
    return "select " in lowered and "from " in lowered


def render_semantic_answer(
    question: str,
    verified: VerifiedResult,
    answer_callable: Callable[..., str] | None,
) -> tuple[str, str]:
    """Return (answer_text, answer_mode) for verified evidence.

    ``answer_mode`` is ``grounded_synthesis`` when the provider rendered the
    text and ``deterministic`` when the fallback renderer was used.
    """
    if verified.status != "answer" or not verified.summary_facts:
        return "", "deterministic"
    if answer_callable is None:
        return render_verified_fallback(question, verified), "deterministic"
    try:
        prompt = build_semantic_answerer_prompt(
            question, verified.summary_facts, verified.missing_information
        )
        text = answer_callable(prompt)
    except Exception:
        return render_verified_fallback(question, verified), "deterministic"
    if (
        not isinstance(text, str)
        or not text.strip()
        or len(text) > MAX_ANSWER_LEN
        or _looks_unsafe(text)
        or not validate_answer_text(text, verified)
    ):
        return render_verified_fallback(question, verified), "deterministic"
    return text.strip(), "grounded_synthesis"


__all__ = [
    "ANSWERER_PROMPT_VERSION",
    "MAX_ANSWER_LEN",
    "render_semantic_answer",
    "render_verified_fallback",
    "validate_answer_text",
]
