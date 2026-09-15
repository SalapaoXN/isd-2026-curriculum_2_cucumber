"""Natural QA v1 QuerySpec representation and parser skeleton."""

from __future__ import annotations

from dataclasses import dataclass
import re
from types import MappingProxyType

from rag.normalization import normalize_thai_surface


_PROGRAM_ALIASES = (
    ("ait", "AIT"),
    ("bit", "BIT"),
    ("dsba", "DSBA"),
    ("gened", "GENED"),
    ("it", "IT"),
)
_PROGRAM_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])(ait|bit|dsba|gened|it)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)

_PLAN_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])(no_coop|coop|default|gened)(?![A-Za-z0-9_])"
    r"|ไม่สหกิจ|แผนปกติ|สหกิจ",
    re.IGNORECASE,
)
_PLAN_ALIASES = (
    ("no_coop", "no_coop"),
    ("coop", "coop"),
    ("default", "default"),
    ("gened", "gened"),
    ("ไม่สหกิจ", "no_coop"),
    ("แผนปกติ", "no_coop"),
    ("สหกิจ", "coop"),
)

_YEAR_PATTERN = re.compile(
    r"ปี\s*(?:ที่\s*)?([1-5])(?!\d)|\byear\s*([1-5])\b",
    re.IGNORECASE,
)
_SEMESTER_PATTERN = re.compile(
    r"เทอม\s*([1-2])(?!\d)"
    r"|ภาค(?:เรียน|การศึกษา)\s*([1-2])(?!\d)"
    r"|\b(?:semester|term)\s*([1-2])\b",
    re.IGNORECASE,
)
_COURSE_CODE_PATTERN = re.compile(r"(?<!\d)(\d{8})(?!\d)")
_COURSE_NAME_PATTERN = re.compile(
    r"(?<!\S)วิชา\s+(?P<name>[A-Za-z][A-Za-z0-9]*(?:[ \t]+[A-Za-z0-9]+)*)"
    r"\s+(?=(?:เรียนเรื่อง|เรียนเกี่ยวกับ|คืออะไร|เกี่ยวกับอะไร|มีอะไร|"
    r"รหัส(?:วิชา)?\s*อะไร))",
    re.IGNORECASE,
)
_CATEGORY_PATTERN = re.compile(r"วิชาเลือก")
_TOPIC_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])(programming|database|network|data|web|AI)"
    r"(?![A-Za-z0-9_])|เขียนโปรแกรม|คอมพิวเตอร์|คอม|เว็บ|ฐานข้อมูล",
    re.IGNORECASE,
)
_OPERATION_PATTERNS = (
    (
        "list",
        re.compile(
            r"เรียนอะไรบ้าง|มีอะไรบ้าง|มีวิชา(?:อะไร|ไหน)|มีวิชา.*?(?:อะไร|ไหน)(?:บ้าง)?|"
            r"ต้องเรียนอะไรบ้าง|วิชาอะไรบ้าง",
            re.IGNORECASE,
        ),
    ),
    (
        "describe",
        re.compile(
            r"เรียน(?:เกี่ยวกับ|เรื่อง)|ชื่ออะไร|เรียนอะไร(?!บ้าง)",
            re.IGNORECASE,
        ),
    ),
    (
        "count",
        re.compile(
            r"กี่\s*(?:วิชา|รายวิชา)|จำนวน(?:ของ)?วิชา|มีวิชา.*(?:เยอะ|มาก|น้อย)",
            re.IGNORECASE,
        ),
    ),
    ("sum_credits", re.compile(r"หน่วยกิต|เครดิต|\bcredits?\b", re.IGNORECASE)),
    (
        "existence",
        re.compile(r"(?:มี|อยู่|พบ).{0,40}(?:ไหม|มั้ย|หรือไม่)|\b(?:exists?|whether)\b", re.IGNORECASE),
    ),
    (
        "compare",
        re.compile(
            r"ต่างกัน|เปรียบเทียบ|\bcompare\b|(?:มาก|น้อย|เยอะ|เร็ว)(?:กว่า|สุด)",
            re.IGNORECASE,
        ),
    ),
    ("earliest", re.compile(r"เร็วกว่า|เร็วที่สุด|\b(?:earliest|sooner)\b", re.IGNORECASE)),
    (
        "placement",
        re.compile(
            r"เรียนปีไหน|เรียนเทอมไหน|อยู่ปีไหน|อยู่เทอมไหน|เรียนช่วงไหนของหลักสูตร|เปิดให้ลง|ลงช่วง|แผนไหน|เรียนก่อน|\bplacement\b",
            re.IGNORECASE,
        ),
    ),
    ("prerequisite", re.compile(r"ก่อนลง|ต้องผ่าน|เรียน.*มาก่อน|prerequisite", re.IGNORECASE)),
    ("similarity", re.compile(r"คล้าย|เหมือน|เนื้อหา.*กัน|\bsimilar(?:ity)?\b", re.IGNORECASE)),
)
_COURSE_DETAIL_PATTERN = re.compile(
    r"ลักษณะไหน|ด้าน(?:ไหน|ใด)(?:บ้าง)?|พูดถึง|อะไรบ้าง|อย่างไร|แบบไหน|"
    r"เนื้อหา.*?(?:ครอบคลุม|ช่วยจัดการ).*?เรื่องใด(?:บ้าง)?",
    re.IGNORECASE,
)
_IDENTITY_NAME_TO_CODE_PATTERN = re.compile(
    r"รหัส(?:วิชา)?\s*อะไร", re.IGNORECASE
)
_IDENTITY_CODE_TO_NAME_PATTERN = re.compile(
    r"(?:ชื่อวิชา\s*อะไร|ชื่อ\s*อะไร|คือวิชา\s*อะไร)",
    re.IGNORECASE,
)
_WORKLOAD_PATTERN = re.compile(r"หนัก(?:ไหม|มั้ย|หรือไม่)", re.IGNORECASE)
_QUANTITY_PATTERN = re.compile(r"เยอะ(?:ไหม|มั้ย|หรือไม่|ปะ)", re.IGNORECASE)
_PREFERENCE_PATTERN = re.compile(r"ชอบ|น่าสนใจ|แนะนำ|เหมาะ|\bprefer(?:ence)?\b", re.IGNORECASE)
_UNSUPPORTED_PATTERN = re.compile(
    r"(?<!อ)ยาก|ง่าย|เงินเดือน|รายได้|\b(?:difficulty|salary|hardest|easiest)\b",
    re.IGNORECASE,
)
_COMPARISON_BEFORE_PATTERN = re.compile(r"(?:ตัวไหน|อันไหน|วิชาไหน).{0,20}เรียนก่อน", re.IGNORECASE)

_OPERATION_ORDER = MappingProxyType(
    {
        "list": 0,
        "describe": 1,
        "count": 2,
        "sum_credits": 3,
        "existence": 4,
        "placement": 5,
        "prerequisite": 6,
        "similarity": 7,
        "earliest": 8,
        "compare": 9,
        "identity": 10,
    }
)

_GROUP_BY_PATTERNS = (
    ("plan", re.compile(r"แผนไหน|\bwhich\s+plan\b|\bplan\b.*\bcompare\b", re.IGNORECASE)),
    (
        "year",
        re.compile(r"ปีไหน(?=.*(?:กว่า|สุด|ต่าง|เปรียบ))|\bwhich\s+year\b|\byear\b.*\bcompare\b", re.IGNORECASE),
    ),
    (
        "semester",
        re.compile(r"เทอมไหน(?=.*(?:กว่า|สุด|ต่าง|เปรียบ))|แต่ละเทอม|\bwhich\s+semester\b|\bsemester\b.*\bcompare\b", re.IGNORECASE),
    ),
    ("course", re.compile(r"\bwhich\s+course\b|\bcourse\b.*\bcompare\b", re.IGNORECASE)),
)


def _ordered_unique(values):
    result = []
    seen = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)


def _extract_program(question: str) -> str | None:
    matches = sorted(
        (match.start(), match.group(1).casefold())
        for match in _PROGRAM_PATTERN.finditer(question)
    )
    if not matches:
        return None
    return next(canonical for alias, canonical in _PROGRAM_ALIASES if alias == matches[0][1])


def _extract_plans(question: str) -> tuple[str, ...]:
    matches = []
    for match in _PLAN_PATTERN.finditer(question):
        value = match.group(0).casefold()
        canonical = next(canonical for alias, canonical in _PLAN_ALIASES if alias == value)
        matches.append((match.start(), canonical))
    return _ordered_unique(value for _, value in sorted(matches))


def _extract_numbered_values(pattern: re.Pattern[str], question: str) -> tuple[int, ...]:
    matches = []
    for match in pattern.finditer(question):
        number = next(group for group in match.groups() if group is not None)
        matches.append((match.start(), int(number)))
    return _ordered_unique(value for _, value in sorted(matches))


def _extract_course_codes(question: str) -> tuple[str, ...]:
    return _ordered_unique(match.group(1) for match in _COURSE_CODE_PATTERN.finditer(question))


def _extract_course_name(question: str, course_codes: tuple[str, ...]) -> str | None:
    if course_codes:
        return None
    match = _COURSE_NAME_PATTERN.search(question)
    return match.group("name").strip() if match else None


def _extract_category(question: str) -> str | None:
    match = _CATEGORY_PATTERN.search(question)
    return match.group(0) if match else None


def _extract_topic(question: str, course_name: str | None) -> str | None:
    if course_name is not None:
        return None
    match = _TOPIC_PATTERN.search(question)
    if not match:
        return None
    if match.group(0) == "เขียนโปรแกรม":
        return "programming"
    if match.group(0) == "ฐานข้อมูล":
        return "database"
    return match.group(0)


def _extract_operations(
    question: str,
    judgement: str,
    has_scope: bool,
    *,
    course_codes: tuple[str, ...] = (),
    course_name: str | None = None,
) -> tuple[str, ...]:
    if not has_scope:
        return ()
    matches = []
    identity_request = bool(
        (course_name and _IDENTITY_NAME_TO_CODE_PATTERN.search(question))
        or (course_codes and _IDENTITY_CODE_TO_NAME_PATTERN.search(question))
    )
    course_targeted_detail = bool(course_codes or course_name) and not identity_request
    for operation, pattern in _OPERATION_PATTERNS:
        if operation == "describe" and identity_request:
            continue
        if operation == "existence" and judgement in {"quantity", "workload"}:
            continue
        for match in pattern.finditer(question):
            matches.append((match.start(), _OPERATION_ORDER[operation], operation))
    if course_targeted_detail:
        match = _COURSE_DETAIL_PATTERN.search(question)
        if match:
            matches.append((match.start(), _OPERATION_ORDER["describe"], "describe"))
    comparison_before = _COMPARISON_BEFORE_PATTERN.search(question)
    if comparison_before:
        matches.append((comparison_before.end(), _OPERATION_ORDER["compare"], "compare"))
    if judgement == "workload":
        match = _WORKLOAD_PATTERN.search(question)
        if match:
            matches.extend(
                (
                    (match.start(), -2, "count"),
                    (match.start(), -1, "sum_credits"),
                )
            )
    if identity_request:
        match = (
            _IDENTITY_NAME_TO_CODE_PATTERN.search(question)
            or _IDENTITY_CODE_TO_NAME_PATTERN.search(question)
        )
        matches.append((match.start() if match else 0, _OPERATION_ORDER["identity"], "identity"))
    return _ordered_unique(
        operation for _, _, operation in sorted(matches, key=lambda item: (item[0], item[1]))
    )


def _extract_group_by(question: str, plans: tuple[str, ...], years: tuple[int, ...],
                      course_codes: tuple[str, ...], has_scope: bool) -> tuple[str, ...]:
    if not has_scope:
        return ()
    matches = []
    if len(plans) > 1:
        first_plan = _PLAN_PATTERN.search(question)
        matches.append((first_plan.start() if first_plan else 0, "plan"))
    if len(years) > 1:
        first_year = _YEAR_PATTERN.search(question)
        matches.append((first_year.start() if first_year else 0, "year"))
    if len(course_codes) > 1:
        first_code = _COURSE_CODE_PATTERN.search(question)
        matches.append((first_code.start() if first_code else 0, "course"))
    for group_by, pattern in _GROUP_BY_PATTERNS:
        for match in pattern.finditer(question):
            matches.append((match.start(), group_by))
    return _ordered_unique(value for _, value in sorted(matches))


def _extract_judgement(question: str) -> str:
    if _UNSUPPORTED_PATTERN.search(question):
        return "unsupported"
    if _WORKLOAD_PATTERN.search(question):
        return "workload"
    if _PREFERENCE_PATTERN.search(question):
        return "preference"
    if _QUANTITY_PATTERN.search(question):
        return "quantity"
    return "none"


@dataclass(frozen=True, slots=True)
class QuerySpec:
    """Immutable normalized question representation for later QA stages."""

    original_question: str
    normalized_question: str
    program: str | None
    plans: tuple[str, ...]
    years: tuple[int, ...]
    semesters: tuple[int, ...]
    course_codes: tuple[str, ...]
    course_name: str | None
    category: str | None
    topic: str | None
    operations: tuple[str, ...]
    group_by: tuple[str, ...]
    judgement: str


def parse_query_spec(question: str) -> QuerySpec:
    """Parse deterministic surface entities into a QuerySpec."""
    if not isinstance(question, str):
        raise TypeError("question must be a string")

    normalized_question = normalize_thai_surface(question)
    program = _extract_program(normalized_question)
    plans = _extract_plans(normalized_question)
    years = _extract_numbered_values(_YEAR_PATTERN, normalized_question)
    semesters = _extract_numbered_values(_SEMESTER_PATTERN, normalized_question)
    course_codes = _extract_course_codes(normalized_question)
    course_name = _extract_course_name(normalized_question, course_codes)
    category = _extract_category(normalized_question)
    topic = _extract_topic(normalized_question, course_name)
    judgement = _extract_judgement(normalized_question)
    # A fully specified year/semester scope is sufficient to recognize an
    # operation even when the program will be supplied later through
    # QueryContext.  Keep one-axis, no-program questions on the existing
    # clarification path.
    has_scope = (
        program is not None
        or (bool(years) and bool(semesters))
        or bool(course_codes)
        or course_name is not None
        or topic is not None
    )

    return QuerySpec(
        original_question=question,
        normalized_question=normalized_question,
        program=program,
        plans=plans,
        years=years,
        semesters=semesters,
        course_codes=course_codes,
        course_name=course_name,
        category=category,
        topic=topic,
        operations=_extract_operations(
            normalized_question,
            judgement,
            has_scope,
            course_codes=course_codes,
            course_name=course_name,
        ),
        group_by=_extract_group_by(normalized_question, plans, years, course_codes, has_scope),
        judgement=judgement,
    )


__all__ = ["QuerySpec", "parse_query_spec"]
