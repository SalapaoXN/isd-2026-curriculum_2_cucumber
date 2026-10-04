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
    r"|ไม่\s*coop|ไม่สหกิจ|แผนปกติ|สหกิจ",
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
    r"ปี\s*(?:ที่\s*)?([1-5])(?!\d)|\byear\s*([1-5])\b|\bY\s*([1-4])\b",
    re.IGNORECASE,
)
_INVALID_YEAR_PATTERN = re.compile(
    r"\bY\s*(?:0|[5-9]|\d{2,})\b|"
    r"\byear\s*(?:0|[6-9]|\d{2,})\b|"
    r"ปี\s*(?:ที่\s*)?(?:0|[6-9]|\d{2,})(?!\d)",
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
    r"(?<!\S)วิชา\s+(?P<name>[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)?(?:[ \t]+[A-Za-z0-9]+(?:-[A-Za-z0-9]+)?)*)"
    r"\s+(?=(?:เรียนปีไหน|เรียนเทอมไหน|เรียนตอนไหน|เรียนเมื่อไหร่|เรียนเมื่อไร|"
    r"มีชื่อ(?:ภาษา)?(?:ไทย|อังกฤษ)|ชื่อ(?:ภาษา)?(?:ไทย|อังกฤษ)|"
    r"เรียนเรื่อง|เรียนเกี่ยวกับ|สอนเรื่อง|สอนเกี่ยวกับ|เนื้อหา|"
    r"คืออะไร|เกี่ยวกับอะไร|มีอะไร|(?:มี\s*)?รหัส(?:วิชา)?\s*อะไร))",
    re.IGNORECASE,
)
_BARE_COURSE_NAME_PATTERN = re.compile(
    r"^\s*(?P<name>[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)?(?:[ \t]+[A-Za-z0-9]+(?:-[A-Za-z0-9]+)?)*)"
    r"\s+(?=(?:เรียนปีไหน|เรียนเทอมไหน|เรียนตอนไหน|เรียนเมื่อไหร่|เรียนเมื่อไร|"
    r"มีชื่อ(?:ภาษา)?(?:ไทย|อังกฤษ)|ชื่อ(?:ภาษา)?(?:ไทย|อังกฤษ)|"
    r"เรียนเรื่อง|เรียนเกี่ยวกับ|เรียนไร|เรียนประมาณไหน|สอนเรื่อง|สอนเกี่ยวกับ|เนื้อหา|"
    r"คืออะไร|เกี่ยวกับอะไร|มีอะไร|(?:มี\s*)?รหัส(?:วิชา)?\s*อะไร|"
    r"มีวิชาบังคับก่อน(?:คือ)?อะไร(?:บ้าง)?|ต้อง(?:เรียน|ผ่าน).{0,30}มาก่อน|ต้องเรียนอะไรต่อ(?:ไหม|มั้ย|ปะ)?|"
    r"(?:อยู่|มีอยู่)\s*ในหลักสูตร(?:อะไร|ไหน)(?:บ้าง)?))",
    re.IGNORECASE,
)
_PROGRAM_DISCOVERY_PATTERN = re.compile(
    r"(?:อยู่|มีอยู่)\s*ในหลักสูตร\s*(?:อะไร|ไหน)(?:บ้าง)?",
    re.IGNORECASE,
)
_CATEGORY_PATTERN = re.compile(r"วิชาเลือก|(?<![A-Za-z0-9_])(?:electives?|gened)(?![A-Za-z0-9_])|ศึกษาทั่วไป|(?<![A-Za-z0-9_])gen\s+ed(?![A-Za-z0-9_])", re.IGNORECASE)
_TOPIC_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])(programming|database|network|data|web|AI)"
    r"(?![A-Za-z0-9_])|เขียนโปรแกรม|คอมพิวเตอร์|คอม|เว็บ|ฐานข้อมูล",
    re.IGNORECASE,
)
_GENERIC_TOPIC_COLLECTION_PATTERNS = (
    re.compile(
        r"(?:มี\s*)?(?:(?:ราย)?วิชา(?:ที่)?|ตัวไหน|วิชาไหน)\s*"
        r"(?:(?:เรียน|สอน)\s*)?เกี่ยวกับ\s*(?P<topic>[^?？\n]{1,80}?)"
        r"(?=\s*(?:มีอะไรบ้าง|อะไรบ้าง|ตัวไหนบ้าง|บ้าง|มั่ง|มีไหม|มีมั้ย|"
        r"ไหม|มั้ย|ปะ|ป่ะ|หรือไม่|[?？]|$))",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:มี\s*)?(?:course|subject|(?:ราย)?วิชา)?\s*แนว\s*"
        r"(?P<topic>[^?？\n]{1,80}?)"
        r"(?=\s*(?:มีอะไร|มีตัวไหน|อะไรบ้าง|ตัวไหนบ้าง|บ้าง|มั่ง|ไหม|มั้ย|ปะ|ป่ะ|[?？]|$))",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:อยากหา(?:เรียน)?|หา)\s*แนว\s*(?P<topic>[^?？\n]{1,80}?)"
        r"(?=\s*(?:มีตัวไหน|มีอะไร|ตัวไหน|บ้าง|มั่ง|[?？]|$))",
        re.IGNORECASE,
    ),
    re.compile(
        r"พวก\s*(?P<topic>[^?？\n]{1,80}?)\s*"
        r"(?:มีอะไร(?:เรียน)?บ้าง|มีตัวไหนบ้าง|มีอะไรมั่ง)",
        re.IGNORECASE,
    ),
    re.compile(
        r"มีอะไร\s*สาย\s*(?P<topic>[^?？\n]{1,80}?)"
        r"(?=\s*(?:บ้าง|มั่ง|ไหม|มั้ย|ปะ|ป่ะ|[?？]|$))",
        re.IGNORECASE,
    ),
    re.compile(
        r"ถ้าอยากเรียน\s*(?P<topic>[^?？\n]{1,80}?)\s*มีตัวไหนเกี่ยว",
        re.IGNORECASE,
    ),
    re.compile(
        r"หา\s*(?:course|subject|(?:ราย)?วิชา)\s*เกี่ยวกับ\s*"
        r"(?P<topic>[^?？\n]{1,80}?)"
        r"(?=\s*(?:ให้หน่อย|หน่อย|บ้าง|มั่ง|[?？]|$))",
        re.IGNORECASE,
    ),
)
_OPERATION_PATTERNS = (
    (
        "list",
        re.compile(
            r"เรียน(?:อะไร|ไร)บ้าง(?:อะ|นะ)?|มีอะไรบ้าง|มีอะไรมั่ง|มีวิชา(?:อะไร|ไหน)|มีวิชา.*?(?:อะไร|ไหน)(?:บ้าง)?|"
            r"ต้องเรียนอะไรบ้าง|วิชาอะไรบ้าง|"
            r"ขอ\s*รายวิชา(?:[^?\n]{0,60}(?:อะไร|ไหน|บ้าง))?|"
            r"มี(?:วิชา)?[^?\n]{0,40}ตัวไหนบ้าง|เรียนตัวไหนกันบ้าง|"
            r"เรียนอะไรกัน(?:บ้าง)?|"
            r"ลงเรียนวิชา\s*(?:gened|ศึกษาทั่วไป)\s*อะไรได้บ้าง|"
            r"(?:รายวิชา|วิชา)(?:ที่(?:สอน|เรียน))?\s*ในปี",
            re.IGNORECASE,
        ),
    ),
    (
        "describe",
        re.compile(
            r"เรียน(?:เกี่ยวกับ|เรื่อง)|สอน(?:เกี่ยวกับ|เรื่อง)|เนื้อหาเป็นอย่างไร|"
            r"ชื่ออะไร|เรียนอะไร(?!บ้าง|กัน)|เรียนไร(?!บ้าง|กัน)(?:อะ)?|เรียนประมาณไหน",
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
    (
        "sum_credits",
        re.compile(
            r"หน่วยกิต|เครดิต|\bcredits?\b|"
            r"กี่\s*หน่วย(?:กิต)?(?:อะ|นะ|วะ|ครับ|คะ)(?![ก-๙A-Za-z0-9_])|"
            r"หนัก\s*กี่\s*หน่วย(?:กิต)?(?![ก-๙A-Za-z0-9_])",
            re.IGNORECASE,
        ),
    ),
    (
        "existence",
        re.compile(r"(?:มี|อยู่|พบ).{0,40}(?:ไหม|มั้ย|ปะ|ป่ะ|หรือไม่)|\b(?:exists?|whether)\b", re.IGNORECASE),
    ),
    (
        "compare",
        re.compile(
            r"ต่างกัน|ช่วงเดียวกัน|เปรียบเทียบ|\bcompare\b|(?:มาก|น้อย|เยอะ|เร็ว)(?:กว่า|สุด)",
            re.IGNORECASE,
        ),
    ),
    ("earliest", re.compile(r"เร็วกว่า|เร็วที่สุด|เร็วสุด|ไวสุด|\b(?:earliest|sooner)\b", re.IGNORECASE)),
    (
        "placement",
        re.compile(
            r"เรียนปีไหน|เรียนเทอมไหน|อยู่ปีไหน|อยู่เทอมไหน|(?:เรียน|อยู่)\s*year\s*ไหน|"
            r"ปีใด|ภาคเรียนใด|เทอมอะไร|จัดไว้ปีไหน|ลงทะเบียนช่วงไหน|"
            r"(?:เรียน|อยู่)ช่วงไหนของหลักสูตร|เรียนช่วงเดียวกัน|เปิดให้ลง|ลงช่วง|"
            r"(?:ลง|เรียน|อยู่).{0,20}ตอนไหน|(?:ลง|เรียน|อยู่).{0,20}เมื่อไหร่|"
            r"(?:ลง|เรียน|อยู่).{0,20}เมื่อไร|ลง.{0,20}เทอมไหน|แผนไหน|เรียนก่อน|"
            r"\bplacement\b",
            re.IGNORECASE,
        ),
    ),
    (
        "prerequisite",
        re.compile(
            r"ก่อนลง|วิชาบังคับก่อน|ต้อง(?:เคย)?ผ่าน|เตรียมผ่านวิชา|"
            r"(?:เรียน|ผ่าน).*มาก่อน|ต้องเรียนอะไรต่อ(?:ไหม)?|prerequisite",
            re.IGNORECASE,
        ),
    ),
    ("similarity", re.compile(r"คล้าย|เหมือน|เนื้อหา.*กัน|\bsimilar(?:ity)?\b", re.IGNORECASE)),
)
_PREVIOUS_RESULT_SET_ANCHORS = (
    "วิชาเหล่านี้",
    "รายวิชาเหล่านี้",
    "พวกนี้",
    "ในพวกนี้",
    "รายการเหล่านี้",
    "จากรายการก่อนหน้า",
    "จากวิชาก่อนหน้า",
)
_COURSE_DETAIL_PATTERN = re.compile(
    r"ลักษณะไหน|ด้าน(?:ไหน|ใด)(?:บ้าง)?|พูดถึง|อะไรบ้าง|อย่างไร|แบบไหน|"
    r"เนื้อหา.*?(?:ครอบคลุม|ช่วยจัดการ).*?เรื่องใด(?:บ้าง)?",
    re.IGNORECASE,
)
_IDENTITY_NAME_TO_CODE_PATTERN = re.compile(
    r"รหัส(?:วิชา)?\s*อะไร", re.IGNORECASE
)
_IDENTITY_PREFIX_NAME_TO_CODE_PATTERN = re.compile(
    r"^\s*รหัส(?:ของวิชา|วิชาของ|วิชา)\s+", re.IGNORECASE
)
_IDENTITY_PREFIX_PROGRAM_QUALIFIER_PATTERN = re.compile(
    r"\s+ใน\s+(?:AIT|BIT|DSBA|GENED|IT)\s+คืออะไร\s*\??\s*$",
    re.IGNORECASE,
)
_IDENTITY_CODE_TO_NAME_PATTERN = re.compile(
    r"(?:ชื่อวิชา\s*อะไร|ชื่อ\s*อะไร|คือวิชา\s*อะไร|"
    r"ชื่อ(?:ภาษา)?อังกฤษ\s*(?:(?:ว่า|คือ)\s*)?อะไร|"
    r"ชื่อ(?:ภาษา)?ไทย\s*(?:(?:ว่า|คือ)\s*)?อะไร|"
    r"ชื่อ(?:ภาษา)?อังกฤษ\s*ด้วย\b|ชื่อ(?:ภาษา)?ไทย\s*ด้วย\b)",
    re.IGNORECASE,
)
_WORKLOAD_PATTERN = re.compile(r"หนัก(?:ไหม|มั้ย|หรือไม่)", re.IGNORECASE)
_QUANTITY_PATTERN = re.compile(r"เยอะ(?:ไหม|มั้ย|หรือไม่|ปะ)", re.IGNORECASE)
_PREFERENCE_PATTERN = re.compile(
    r"ชอบ|น่าสนใจ|แนะนำ|เหมาะ|อยาก\s*เน้น|\bprefer(?:ence)?\b",
    re.IGNORECASE,
)
_UNSUPPORTED_PATTERN = re.compile(
    r"(?<!อ)ยาก|ง่าย|เงินเดือน|รายได้|\b(?:difficulty|salary|hardest|easiest)\b",
    re.IGNORECASE,
)
_COMPARISON_BEFORE_PATTERN = re.compile(r"(?:ตัวไหน|อันไหน|วิชาไหน).{0,20}เรียนก่อน", re.IGNORECASE)
_COURSE_CONTENT_COMPARISON_PATTERN = re.compile(
    r"เนื้อหา|หัวข้อ|สาระ|เน้นเรื่องใด(?:บ้าง)?|\bcontent\b|\btopic\b",
    re.IGNORECASE,
)
_PREREQUISITE_OBJECT_PATTERN = re.compile(
    r"ก่อนลง\s*\d{8}\s*ต้อง(?:เคย)?ผ่านวิชาอะไร(?:บ้าง)?|"
    r"วิชาบังคับก่อน(?:ของ\s*\d{8})?|ต้องเรียนอะไรต่อ(?:ไหม)?",
    re.IGNORECASE,
)
_THAI_COURSE_CREDIT_NAME_PATTERN = re.compile(
    r"^\s*(?:วิชา\s*)?(?P<name>[\u0E00-\u0E7F][\u0E00-\u0E7F0-9 \t]*?)"
    r"\s+(?=(?:มี\s*)?(?:กี่\s*)?(?:หน่วยกิต|เครดิต))",
    re.IGNORECASE,
)
_PREREQUISITE_COLLECTION_PATTERN = re.compile(
    r"(?:วิชา|รายวิชา)\s*(?:ใด|ไหน)|"
    r"มี\s*(?:วิชา|รายวิชา)\s*อะไร|"
    r"(?:วิชา|รายวิชา).{0,40}บ้าง",
    re.IGNORECASE,
)
_PREREQUISITE_BURDEN_PREFERENCE_PATTERN = re.compile(
    r"(?:วิชาบังคับก่อน|prerequisite)[^?\n]{0,40}"
    r"(?:ไม่\s*(?:เยอะ|มาก)|น้อย|ไม่กี่)",
    re.IGNORECASE,
)

_OPERATION_ORDER = MappingProxyType(
    {
        "list": 0,
        "program_discovery": 0,
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
        if value == "gened" and re.search(
            r"วิชา\s*gened\b|gened\s+อะไร", question, re.IGNORECASE
        ):
            continue
        if re.fullmatch(r"ไม่\s*coop", value, re.IGNORECASE):
            canonical = "no_coop"
        else:
            canonical = next(
                canonical for alias, canonical in _PLAN_ALIASES if alias == value
            )
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


def _prefix_name_to_code_title_match(question: str) -> re.Match[str] | None:
    prefix = _IDENTITY_PREFIX_NAME_TO_CODE_PATTERN.match(question)
    if prefix is None:
        return None
    remainder = question[prefix.end() :]
    program_qualifier = _IDENTITY_PREFIX_PROGRAM_QUALIFIER_PATTERN.search(
        remainder
    )
    title_text = (
        remainder[: program_qualifier.start()] + " คืออะไร"
        if program_qualifier is not None
        else remainder
    )
    title = _BARE_COURSE_NAME_PATTERN.match(title_text)
    if title is None:
        return None
    suffix = title_text[title.end() :]
    if program_qualifier is None and not suffix.casefold().startswith("คืออะไร"):
        return None
    return title


def _extract_course_name(question: str, course_codes: tuple[str, ...]) -> str | None:
    if course_codes:
        return None
    match = _COURSE_NAME_PATTERN.search(question)
    if match is None:
        match = _BARE_COURSE_NAME_PATTERN.search(question)
    if match is None:
        match = _prefix_name_to_code_title_match(question)
    if match is None:
        thai_credit_match = _THAI_COURSE_CREDIT_NAME_PATTERN.match(question)
        if thai_credit_match is not None:
            thai_name = thai_credit_match.group("name").strip()
            # The bounded credit-question grammar can otherwise mistake a
            # bare quantifier such as "กี่หน่วยกิต" for a course title.
            thai_name_letters = re.sub(r"[\s\d]", "", thai_name)
            if len(thai_name_letters) >= 4 and thai_name not in {"มี", "กี่"}:
                return thai_name
    if not match:
        return None
    name = match.group("name").strip()
    if name.casefold() in {"ait", "bit", "dsba", "gened", "it"}:
        return None
    return name


def _is_prefix_name_to_code_request(
    question: str,
    course_name: str | None,
) -> bool:
    if course_name is None:
        return False
    title = _prefix_name_to_code_title_match(question)
    return bool(title and title.group("name").casefold() == course_name.casefold())


def _extract_category(question: str) -> str | None:
    match = _CATEGORY_PATTERN.search(question)
    if not match:
        return None
    matched = match.group(0).casefold()
    if matched == "วิชาเลือก" or matched in {"elective", "electives"}:
        return "วิชาเลือก"
    return "หมวดวิชาศึกษาทั่วไป"


_CREDIT_UNITS_PATTERN = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)\s*หน่วยกิต")
_CREDIT_TOTAL_PATTERN = re.compile(r"รวม|ทั้งหมด|กี่\s*หน่วยกิต")


def _extract_credit_units(question: str) -> int | None:
    """Capture an integral per-course credit predicate, if explicitly stated.

    Numeric parsing follows the existing parser convention (bare ``int``,
    consistent with year/semester extraction). Non-integral values stay
    unparsed so the existing partial path fails closed.
    """
    match = _CREDIT_UNITS_PATTERN.search(question)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _extract_topic(
    question: str,
    course_name: str | None,
    course_codes: tuple[str, ...] = (),
) -> str | None:
    if course_name is not None:
        return None

    # Collection wording owns the full literal span first so compound topics
    # such as "network security", "data center", and "big data" are not
    # truncated by the legacy single-token topic aliases below.
    if not course_codes:
        for pattern in _GENERIC_TOPIC_COLLECTION_PATTERNS:
            generic = pattern.search(question)
            if generic is None:
                continue
            topic = re.sub(r"\s+", " ", generic.group("topic")).strip(" \t,;:.-")
            if not topic or len(topic) > 80:
                continue

            # Do not treat an anaphoric phrase such as "วิชาแนวนี้ตั้งแต่ปีไหน"
            # as a new semantic topic. In those forms a real topic may already
            # be present earlier in the question and is handled by the legacy
            # bounded aliases below.
            if re.match(
                r"^(?:นี้|นั้น|โน้น|พวกนี้|พวกนั้น)(?:\s|ตั้งแต่|อยู่|ปี|เทอม|$)",
                topic,
            ):
                continue

            # Generic topic grammars deliberately capture a free text span.
            # Trim only known question/comparison/scope residue from its tail;
            # keep meaningful compounds such as "network security",
            # "data center", and "big data" intact.
            topic = re.sub(
                r"\s*(?:กี่\s*(?:วิชา|รายวิชา)|เยอะ(?:สุด)?|มาก(?:สุด)?|น้อย(?:สุด)?)\s*$",
                "",
                topic,
                flags=re.IGNORECASE,
            ).strip()
            topic = re.sub(
                r"\s*(?:มากกว่า|น้อยกว่า|ต่างจาก|เทียบกับ)\s*"
                r"(?:แผน)?(?:สหกิจ|ไม่สหกิจ|ปกติ|coop|no_coop).*$",
                "",
                topic,
                flags=re.IGNORECASE,
            ).strip()
            topic = re.sub(
                r"\s*(?:ใน|ของ)\s*(?:AIT|BIT|DSBA|GENED|IT)\s*$",
                "",
                topic,
                flags=re.IGNORECASE,
            ).strip()
            topic = re.sub(
                r"^(?:พวก\s*)?(?:ราย)?วิชา\s*",
                "",
                topic,
                flags=re.IGNORECASE,
            ).strip()
            if not topic or topic.casefold() in {
                "อะไร",
                "อะไรบ้าง",
                "เรื่องอะไร",
                "เกี่ยวกับอะไร",
                "ด้านไหน",
                "เรื่องไหน",
            }:
                continue
            if topic == "เขียนโปรแกรม":
                return "programming"
            if topic == "ฐานข้อมูล":
                return "database"
            return topic

    match = _TOPIC_PATTERN.search(question)
    if match:
        if match.group(0) == "เขียนโปรแกรม":
            return "programming"
        if match.group(0) == "ฐานข้อมูล":
            return "database"
        return match.group(0)
    return None


def _surface_operation_matches(
    question: str,
) -> tuple[tuple[int, int, str], ...]:
    matches: list[tuple[int, int, str]] = []
    for operation, pattern in _OPERATION_PATTERNS:
        matches.extend(
            (
                match.start(),
                _OPERATION_ORDER[operation],
                operation,
            )
            for match in pattern.finditer(question)
        )
    return tuple(matches)


def detect_surface_operations(question: str) -> tuple[str, ...]:
    """Return only operations explicitly surfaced in the current question."""
    if not isinstance(question, str):
        raise TypeError("question must be a string")
    normalized_question = normalize_thai_surface(question)
    return _ordered_unique(
        operation
        for _, _, operation in sorted(
            _surface_operation_matches(normalized_question),
            key=lambda item: (item[0], item[1]),
        )
    )


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
    if (
        (course_codes or course_name is not None)
        and _PROGRAM_DISCOVERY_PATTERN.search(question)
    ):
        return ("program_discovery",)
    matches = []
    identity_request = bool(
        (
            course_name
            and (
                _IDENTITY_NAME_TO_CODE_PATTERN.search(question)
                or _is_prefix_name_to_code_request(question, course_name)
                or _IDENTITY_CODE_TO_NAME_PATTERN.search(question)
            )
        )
        or (course_codes and _IDENTITY_CODE_TO_NAME_PATTERN.search(question))
    )
    explicit_describe_request = bool(
        _COURSE_DETAIL_PATTERN.search(question)
        or re.search(r"เรียน(?:เกี่ยวกับ|เรื่อง)?อะไร(?:อะ|บ้าง)?", question)
    )
    prerequisite_object_request = bool(_PREREQUISITE_OBJECT_PATTERN.search(question))
    prerequisite_collection_request = bool(
        _PREREQUISITE_COLLECTION_PATTERN.search(question)
        and any(
            operation == "prerequisite"
            for _, _, operation in _surface_operation_matches(question)
        )
    )
    prerequisite_burden_preference = (
        judgement == "preference"
        and bool(_PREREQUISITE_BURDEN_PREFERENCE_PATTERN.search(question))
    )
    course_targeted_detail = bool(course_codes or course_name) and not identity_request
    course_content_comparison = (
        len(course_codes) > 1
        and bool(_COURSE_CONTENT_COMPARISON_PATTERN.search(question))
    )
    for start, _, operation in _surface_operation_matches(question):
        if operation == "list" and prerequisite_collection_request:
            continue
        if operation in {"list", "describe"} and prerequisite_object_request:
            continue
        if operation == "count" and prerequisite_burden_preference:
            continue
        if operation == "describe" and identity_request and not explicit_describe_request:
            continue
        if operation == "existence" and judgement in {"quantity", "workload"}:
            continue
        resolved_operation = (
            "similarity"
            if operation == "compare" and course_content_comparison
            else operation
        )
        matches.append(
            (
                start,
                _OPERATION_ORDER[resolved_operation],
                resolved_operation,
            )
        )
    if course_content_comparison and not any(
        operation == "similarity" for _, _, operation in matches
    ):
        match = _COURSE_CONTENT_COMPARISON_PATTERN.search(question)
        matches.append(
            (
                match.start() if match else 0,
                _OPERATION_ORDER["similarity"],
                "similarity",
            )
        )
    if course_targeted_detail and not prerequisite_object_request:
        match = _COURSE_DETAIL_PATTERN.search(question)
        if match:
            matches.append((match.start(), _OPERATION_ORDER["describe"], "describe"))
    if (
        identity_request
        and explicit_describe_request
        and not any(operation == "describe" for _, _, operation in matches)
    ):
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


def _references_previous_result_set(question: str) -> bool:
    """Recognize only explicit, high-confidence anchors to a prior result set."""
    normalized = " ".join(question.casefold().split())
    compact = "".join(normalized.split())
    return any(
        "".join(anchor.casefold().split()) in compact
        for anchor in _PREVIOUS_RESULT_SET_ANCHORS
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
    credit_units: int | None = None
    references_previous_result_set: bool = False


def parse_query_spec(
    question: str,
    *,
    has_validated_context_scope: bool = False,
) -> QuerySpec:
    """Parse deterministic surface entities into a QuerySpec."""
    if not isinstance(question, str):
        raise TypeError("question must be a string")
    if not isinstance(has_validated_context_scope, bool):
        raise TypeError("has_validated_context_scope must be a boolean")

    normalized_question = normalize_thai_surface(question)
    program = _extract_program(normalized_question)
    plans = _extract_plans(normalized_question)
    years = _extract_numbered_values(_YEAR_PATTERN, normalized_question)
    semesters = _extract_numbered_values(_SEMESTER_PATTERN, normalized_question)
    course_codes = _extract_course_codes(normalized_question)
    course_name = _extract_course_name(normalized_question, course_codes)
    category = _extract_category(normalized_question)
    topic = _extract_topic(normalized_question, course_name, course_codes)
    judgement = _extract_judgement(normalized_question)
    credit_units = _extract_credit_units(normalized_question)
    if _INVALID_YEAR_PATTERN.search(normalized_question):
        judgement = "unsupported"
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
        or has_validated_context_scope
        or (
            bool(_PREREQUISITE_COLLECTION_PATTERN.search(normalized_question))
            and "prerequisite" in detect_surface_operations(normalized_question)
        )
    )

    operations = _extract_operations(
        normalized_question,
        judgement,
        has_scope,
        course_codes=course_codes,
        course_name=course_name,
    )
    if (
        topic is not None
        and not operations
        and any(
            pattern.search(normalized_question)
            for pattern in _GENERIC_TOPIC_COLLECTION_PATTERNS
        )
    ):
        # Every generic topic grammar above is explicitly collection-shaped.
        # If its colloquial tail did not hit a narrower surface operation,
        # keep the semantics bounded to a topic list instead of widening to
        # the whole curriculum or invoking an LLM.
        operations = ("list",)
    if (
        credit_units is not None
        and "sum_credits" in operations
        and not _CREDIT_TOTAL_PATTERN.search(normalized_question)
    ):
        # The credit token is consumed by the per-course predicate; the
        # wording-derived total was an over-trigger, not a requested total.
        operations = tuple(
            operation for operation in operations if operation != "sum_credits"
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
        operations=operations,
        group_by=_extract_group_by(normalized_question, plans, years, course_codes, has_scope),
        judgement=judgement,
        credit_units=credit_units,
        references_previous_result_set=_references_previous_result_set(
            normalized_question
        ),
    )


__all__ = ["QuerySpec", "detect_surface_operations", "parse_query_spec"]
