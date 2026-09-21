"""Small deterministic composition of curriculum load and policy limits."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Mapping
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag.structured.queries import get_semester_credits

from .answer import PolicyAnswer, answer_policy_question


COMBINED_STATUSES = ("complete", "unsupported", "insufficient_evidence")
_PROGRAM_RE = re.compile(r"\b(?P<program>AIT|BIT|DSBA|IT)\b", re.IGNORECASE)
_PLAN_RE = re.compile(
    r"(?:แผน\s*)?(?P<plan>ไม่\s*สหกิจ|สหกิจ|no[_ ]?coop|coop)\b",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"(?:ปี|Y)\s*(?P<year>[1-5])\b", re.IGNORECASE)
_SEMESTER_RE = re.compile(
    r"(?:เทอม|ภาคเรียน(?:ที่)?|ภาคการศึกษา(?:ที่)?)\s*(?P<semester>[1-2])\b",
    re.IGNORECASE,
)
_CURRENT_CREDITS_RE = re.compile(r"มี\s*(?P<credits>\d+)\s*หน่วยกิต")
_ADDED_CREDITS_RE = re.compile(
    r"(?:ลง\s*)?เพิ่ม(?:อีก)?\s*(?P<credits>\d+)\s*หน่วยกิต"
)
_STUDENT_SPECIFIC_RE = re.compile(
    r"(?:ฉัน|ผม|หนู|มีสิทธิ์|เข้าเงื่อนไข|ได้รับอนุมัติ|อนุมัติ)", re.IGNORECASE
)


@dataclass(frozen=True, slots=True)
class CombinedQuery:
    program: str | None
    plan_key: str | None
    year: int | None
    semester: int | None
    stated_current_credits: int | None
    added_credits: int | None


@dataclass(frozen=True, slots=True)
class CurriculumLoad:
    program: str
    plan_key: str
    plan_id: int
    year: int
    semester: int
    total_credits: int | float
    components: tuple[Mapping[str, Any], ...]
    provenance: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class CombinedAnswer:
    status: str
    query_type: str = "semester_load_plus_credits"
    program: str | None = None
    plan_key: str | None = None
    year: int | None = None
    semester: int | None = None
    curriculum_evidence: tuple[CurriculumLoad, ...] = ()
    policy_evidence: tuple[PolicyAnswer, ...] = ()
    current_credits: int | float | None = None
    added_credits: int | None = None
    resulting_total: int | float | None = None
    normal_maximum: int | float | None = None
    exception_maximum: int | float | None = None
    decision: str | None = None
    conditional_note: str | None = None
    curriculum_provenance: tuple[Mapping[str, Any], ...] = ()
    policy_provenance: tuple[Mapping[str, Any], ...] = ()
    rendered_answer: str = ""

    def __post_init__(self) -> None:
        if self.status not in COMBINED_STATUSES:
            raise ValueError(f"unsupported combined status: {self.status!r}")


def parse_combined_question(question: str) -> CombinedQuery | None:
    """Parse only the bounded semester-load-plus-credits question family."""

    if not isinstance(question, str) or not question.strip():
        return None
    text = re.sub(r"\s+", " ", question.strip())
    if "หน่วยกิต" not in text or not re.search(r"เพิ่ม", text):
        return None
    if _STUDENT_SPECIFIC_RE.search(text):
        return CombinedQuery(None, None, None, None, None, None)

    programs = {match.group("program").upper() for match in _PROGRAM_RE.finditer(text)}
    plans = []
    for match in _PLAN_RE.finditer(text):
        value = re.sub(r"\s+", "", match.group("plan").lower())
        plans.append("no_coop" if value in {"ไม่สหกิจ", "no_coop"} else "coop")
    years = [int(match.group("year")) for match in _YEAR_RE.finditer(text)]
    semesters = [int(match.group("semester")) for match in _SEMESTER_RE.finditer(text)]
    current = _CURRENT_CREDITS_RE.search(text)
    added = _ADDED_CREDITS_RE.search(text)
    return CombinedQuery(
        program=next(iter(programs)) if len(programs) == 1 else None,
        plan_key=plans[0] if len(set(plans)) == 1 else None,
        year=years[0] if len(set(years)) == 1 else None,
        semester=semesters[0] if len(set(semesters)) == 1 else None,
        stated_current_credits=int(current.group("credits")) if current else None,
        added_credits=int(added.group("credits")) if added else None,
    )


def _plan_keys(db_path: str | Path, program: str) -> tuple[str, ...]:
    connection = sqlite3.connect(f"file:{Path(db_path).resolve()}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT DISTINCT plan_key FROM curriculum_plans WHERE program_code = ? ORDER BY plan_key",
            (program,),
        )
        return tuple(str(row[0]) for row in rows)
    finally:
        connection.close()


def _provenance_from_components(
    components: tuple[Mapping[str, Any], ...],
) -> tuple[Mapping[str, Any], ...]:
    seen: set[tuple[tuple[str, str], ...]] = set()
    result: list[Mapping[str, Any]] = []
    for component in components:
        for reference in component.get("provenance", ()):
            if not isinstance(reference, Mapping):
                raise ValueError("curriculum provenance is malformed")
            key = tuple(sorted((str(k), repr(v)) for k, v in reference.items()))
            if key not in seen:
                seen.add(key)
                result.append(reference)
    if not result:
        raise ValueError("curriculum evidence lacks provenance")
    return tuple(result)


def _curriculum_loads(
    db_path: str | Path,
    query: CombinedQuery,
) -> tuple[CurriculumLoad, ...]:
    if not query.program or query.year is None or query.semester is None:
        raise ValueError("program, year, and semester are required")
    plan_keys = (query.plan_key,) if query.plan_key else _plan_keys(db_path, query.program)
    if not plan_keys:
        raise ValueError("program has no plans")
    loads: list[CurriculumLoad] = []
    for plan_key in plan_keys:
        payload = get_semester_credits(
            db_path,
            query.program,
            plan_key,
            query.year,
            query.semester,
        )
        if payload.get("status") != "ok" or not payload.get("plans"):
            raise ValueError("curriculum semester evidence is missing")
        for plan in payload["plans"]:
            components = tuple(plan.get("components", ()))
            provenance = _provenance_from_components(components)
            loads.append(
                CurriculumLoad(
                    program=query.program,
                    plan_key=str(plan["plan_key"]),
                    plan_id=int(plan["plan_id"]),
                    year=query.year,
                    semester=query.semester,
                    total_credits=plan["total_credits"],
                    components=components,
                    provenance=provenance,
                )
            )
    if not loads:
        raise ValueError("curriculum semester evidence is empty")
    totals = {load.total_credits for load in loads}
    if len(totals) != 1:
        raise ValueError("unconstrained plan loads disagree")
    return tuple(loads)


def answer_combined_question(
    db_path: str | Path,
    question: str,
) -> CombinedAnswer:
    """Compose canonical semester evidence with canonical registration policy."""

    query = parse_combined_question(question)
    if query is None:
        return CombinedAnswer(status="unsupported")
    if (
        not query.program
        or query.year is None
        or query.semester is None
        or query.added_credits is None
        or query.added_credits < 0
    ):
        return CombinedAnswer(status="insufficient_evidence")
    try:
        loads = _curriculum_loads(db_path, query)
        current = loads[0].total_credits
        if query.stated_current_credits is not None and query.stated_current_credits != current:
            return CombinedAnswer(
                status="insufficient_evidence",
                program=query.program,
                plan_key=query.plan_key,
                year=query.year,
                semester=query.semester,
                curriculum_evidence=loads,
                current_credits=current,
                added_credits=query.added_credits,
                curriculum_provenance=tuple(
                    reference for load in loads for reference in load.provenance
                ),
            )

        regular = answer_policy_question(db_path, "ปกติลงทะเบียนได้สูงสุดกี่หน่วยกิต")
        exception = answer_policy_question(db_path, "กรณีพิเศษลงได้สูงสุดเท่าไร")
        if regular.status != "complete" or exception.status != "complete":
            raise ValueError("registration policy evidence is incomplete")
        normal_max = regular.value
        exception_max = exception.value
        resulting = current + query.added_credits
        if resulting <= normal_max:
            decision = "within_normal_limit"
            note = None
            rendered = (
                f"หลักสูตร {query.program} ปี {query.year} ภาคเรียนที่ {query.semester} "
                f"มี {current} หน่วยกิต; เพิ่ม {query.added_credits} หน่วยกิตเป็น {resulting} "
                f"หน่วยกิต ซึ่งไม่เกินเพดานปกติ {normal_max} หน่วยกิต"
            )
        elif resulting <= exception_max:
            decision = "conditional_exception"
            note = "อาจทำได้เฉพาะเมื่อเข้าเงื่อนไขกรณีพิเศษที่มีหลักฐานรองรับ"
            rendered = (
                f"รวมเป็น {resulting} หน่วยกิต ซึ่งเกินเพดานปกติ {normal_max} หน่วยกิต "
                f"แต่อยู่ภายในเพดานกรณีพิเศษ {exception_max} หน่วยกิต; {note}"
            )
        else:
            decision = "exceeds_exception_maximum"
            note = None
            rendered = (
                f"รวมเป็น {resulting} หน่วยกิต ซึ่งเกินทั้งเพดานปกติ {normal_max} "
                f"และเพดานกรณีพิเศษ {exception_max} หน่วยกิต"
            )
        curriculum_provenance = tuple(
            reference for load in loads for reference in load.provenance
        )
        policy_provenance = tuple(
            reference
            for policy in (regular, exception)
            for reference in policy.provenance
        )
        return CombinedAnswer(
            status="complete",
            program=query.program,
            plan_key=query.plan_key,
            year=query.year,
            semester=query.semester,
            curriculum_evidence=loads,
            policy_evidence=(regular, exception),
            current_credits=current,
            added_credits=query.added_credits,
            resulting_total=resulting,
            normal_maximum=normal_max,
            exception_maximum=exception_max,
            decision=decision,
            conditional_note=note,
            curriculum_provenance=curriculum_provenance,
            policy_provenance=policy_provenance,
            rendered_answer=rendered,
        )
    except (OSError, sqlite3.Error, KeyError, TypeError, ValueError):
        return CombinedAnswer(
            status="insufficient_evidence",
            program=query.program,
            plan_key=query.plan_key,
            year=query.year,
            semester=query.semester,
        )


__all__ = [
    "CombinedAnswer",
    "CombinedQuery",
    "CurriculumLoad",
    "answer_combined_question",
    "parse_combined_question",
]
