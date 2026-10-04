"""Bounded interpretation of the supported standalone policy questions."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag.query_spec import parse_query_spec


@dataclass(frozen=True, slots=True)
class PolicyQuery:
    kind: str
    program: str | None = None
    amount: int | None = None
    plan: str | None = None


_PROGRAM_RE = re.compile(r"\b(?P<program>AIT|BIT|DSBA|IT)\b", re.IGNORECASE)
_AMOUNT_RE = re.compile(r"(?:ลงทะเบียน|ลง|เรียน)\s*(?P<amount>\d+)\s*หน่วยกิต")
_GPA_RE = re.compile(r"GPA\s*(?:เท่าไร|เท่าไหร่|กี่คะแนน)", re.IGNORECASE)


def _normalized(question: str) -> str:
    return re.sub(r"\s+", " ", question.strip().lower())


def _program(question: str) -> str | None:
    matches = {match.group("program").upper() for match in _PROGRAM_RE.finditer(question)}
    return next(iter(matches)) if len(matches) == 1 else None


def _is_bare_regular_max(text: str, question: str) -> bool:
    """Return True for bare regular-semester maximum wording (H28-B).

    Matches only when a maximum synonym co-occurs with a registration
    verb, the shape carries no program-total verbs (those keep the
    program_total route), and the curriculum parse carries no explicit
    scope axis (axis shapes keep the curriculum/H27 path). Earlier
    branches (compare, exception, special, explicit-ปกติ, min) are
    untouched, so every previously routed shape keeps its outcome.
    """
    if not re.search(r"(?:สูงสุด|มากสุด|ไม่เกิน)", text):
        return False
    if not re.search(r"(?:ลงทะเบียน|ลง|เรียน)", text):
        return False
    if re.search(r"(?:ต้องเรียน|เรียนทั้งหมด|รวมทั้งหมด)", text):
        return False
    if re.search(r"(?:(?<!ไม่)เกิน|overload)", text):
        # Overload wording is ambiguous between the regular cap and the
        # exception cap: without an explicit กรณีพิเศษ marker it fails
        # closed instead of guessing the regular maximum. The negated
        # ไม่เกิน ("not exceeding") is a plain maximum synonym, not
        # overload wording, so the lookbehind exempts it.
        return False
    spec = parse_query_spec(question)
    return not bool(
        getattr(spec, "years", ())
        or getattr(spec, "semesters", ())
        or getattr(spec, "plans", ())
        or getattr(spec, "category", None)
        or getattr(spec, "course_codes", ())
        or getattr(spec, "course_name", None)
    )


def parse_policy_question(question: str) -> PolicyQuery | None:
    """Return a supported policy query, or ``None`` for unsupported/ambiguous text."""

    if not isinstance(question, str) or not question.strip():
        return None
    text = _normalized(question)
    program = _program(question)
    if re.search(r"(?:IT|AIT|BIT|DSBA)\s+.*(?:IT|AIT|BIT|DSBA)", question, re.IGNORECASE):
        return None

    general_text_policy = re.search(
        r"(?:ต้องทำอย่างไร|ต้องทำยังไง|ทำอย่างไร|ทำยังไง|มีขั้นตอนอะไร|"
        r"มีเงื่อนไขอะไร|เงื่อนไขเป็นอย่างไร|มีหลักเกณฑ์อะไรบ้าง|"
        r"หลักเกณฑ์(?:เป็นอย่างไร|มีอะไรบ้าง|อะไรบ้าง)|"
        r"ทำได้ไหม|ทำได้หรือไม่|ได้ไหม|ได้หรือไม่|คืออะไร|เป็นอย่างไร)\s*[?？]?$",
        text,
    )
    if general_text_policy:
        if "ลาพักการศึกษา" in text:
            return PolicyQuery("leave_of_absence")
        if "ลาออก" in text:
            return PolicyQuery("resignation")
        if (
            "เทียบโอนหน่วยกิต" in text
            or "โอนหน่วยกิต" in text
            or "โอนผลการเรียน" in text
        ):
            return PolicyQuery("credit_transfer")

    if (
        "ทุจริต" in text
        and "สอบ" in text
        and re.search(r"(?:มีโทษ|โทษ|ลงโทษ|เป็นอย่างไร|เป็นยังไง|เกิดอะไร)", text)
    ):
        return PolicyQuery("exam_dishonesty_penalty")

    if (
        ("โทษทางวินัย" in text or ("วินัย" in text and "โทษ" in text))
        and re.search(r"(?:มีอะไรบ้าง|อะไรบ้าง|มีอะไร|เป็นอย่างไร|เป็นยังไง)", text)
    ):
        return PolicyQuery("disciplinary_penalties")

    if "อุทธรณ์" in text and "ลงโทษ" in text:
        if re.search(r"(?:ภายในกี่วัน|กี่วัน|ภายในเท่าไร|ภายในเท่าไหร่)", text):
            return PolicyQuery("sanction_appeal_deadline")
        if re.search(
            r"(?:ต้องทำอย่างไร|ต้องทำยังไง|ทำอย่างไร|ทำยังไง|"
            r"มีขั้นตอนอะไร|ขั้นตอนเป็นอย่างไร|ขั้นตอนเป็นยังไง)",
            text,
        ):
            return PolicyQuery("sanction_appeal_procedure")

    amount_match = _AMOUNT_RE.search(text)
    if amount_match and re.search(r"(?:ได้ไหม|ได้หรือไม่|ได้หรือเปล่า)\s*[?？]?$", text):
        if "หน่วยกิต" not in text or text.count("หน่วยกิต") != 1:
            return None
        return PolicyQuery("registration_compare", amount=int(amount_match.group("amount")))

    if "หน่วยกิต" in text:
        if "กรณีพิเศษ" in text and re.search(r"(?:สูงสุด|เท่าไร|เท่าไหร่)", text):
            return PolicyQuery("registration_exception_max")
        if re.search(r"(?:ซัมเมอร์|ภาคฤดูร้อน|ภาคพิเศษ)", text) and re.search(
            r"(?:กี่|สูงสุด|เท่าไร|เท่าไหร่)", text
        ):
            return PolicyQuery("registration_special_max")
        if re.search(r"(?:ปกติ|ภาคปกติ)", text) and re.search(
            r"(?:สูงสุด|เท่าไร|เท่าไหร่)", text
        ):
            return PolicyQuery("registration_regular_max")
        if re.search(r"(?:ขั้นต่ำ|ต่ำสุด|อย่างน้อย)", text):
            return PolicyQuery("registration_regular_min")
        if _is_bare_regular_max(text, question):
            return PolicyQuery("registration_regular_max")
        if program and re.search(r"(?:ต้องเรียน|เรียนทั้งหมด|รวมทั้งหมด)", text):
            spec = parse_query_spec(question)
            if (
                getattr(spec, "years", ())
                or getattr(spec, "semesters", ())
                or getattr(spec, "category", None)
                or getattr(spec, "course_codes", ())
                or getattr(spec, "course_name", None)
            ):
                return None
            plans = tuple(getattr(spec, "plans", ()))
            if len(plans) > 1:
                return None
            return PolicyQuery(
                "program_total_credits",
                program=program,
                plan=plans[0] if len(plans) == 1 else None,
            )
        return None

    if "กรณีพิเศษ" in text and re.search(r"(?:สูงสุด|เท่าไร|เท่าไหร่)", text):
        return PolicyQuery("registration_exception_max")

    if (
        "พ้นสภาพ" in text
        and "นักศึกษา" in text
        and re.search(r"(?:กรณีอะไรบ้าง|มีกรณีอะไร|สาเหตุอะไรบ้าง|มีสาเหตุอะไร|เพราะอะไรบ้าง)", text)
    ):
        return PolicyQuery("student_status_termination_reasons")

    if (
        ("คิด gpa" in text or "คำนวณ gpa" in text or "การคิด gpa" in text)
        and re.search(r"(?:อย่างไร|ยังไง|แบบไหน|มีกี่ประเภท|กี่ประเภท)", text)
    ):
        return PolicyQuery("gpa_calculation_method")

    if (
        ("การวัดผล" in text or "วัดผลการศึกษา" in text)
        and re.search(r"(?:ทำได้อย่างไร|ทำได้ยังไง|มีวิธีอะไร|วิธีการ|ใคร.*อนุมัติ|ผู้อนุมัติ)", text)
    ):
        return PolicyQuery("assessment_method")

    if (
        ("ความผิดวินัย" in text or "ผิดวินัย" in text)
        and "ร้ายแรง" in text
        and re.search(r"(?:มีอะไรบ้าง|อะไรบ้าง|มีอะไร|คืออะไร)", text)
    ):
        return PolicyQuery("serious_disciplinary_offenses")

    if (
        (
            "ระเบียบความประพฤติ" in text
            or "ข้อปฏิบัติของนักศึกษา" in text
            or "นักศึกษาต้องปฏิบัติตัว" in text
            or "นักศึกษาต้องประพฤติตัว" in text
        )
        and re.search(r"(?:อย่างไร|ยังไง|มีอะไรบ้าง|อะไรบ้าง|คืออะไร)", text)
    ):
        return PolicyQuery("student_conduct_rules")

    graduation_wording = "สำเร็จการศึกษา" in text or "จบการศึกษา" in text
    if graduation_wording:
        if (
            ("english exit exam" in text or "สอบภาษาอังกฤษ" in text)
            and re.search(r"(?:ต้อง|จำเป็น|ผ่าน|สอบ|มีไหม|หรือไม่|ไหม)", text)
        ):
            return PolicyQuery("graduation_english_exit")
        if (
            ("หนี้สิน" in text or "ภาระผูกพัน" in text)
            and re.search(r"(?:ต้อง|มีได้ไหม|มีได้หรือไม่|ไม่มี|หรือไม่|ไหม)", text)
        ):
            return PolicyQuery("graduation_no_debt")
        if re.search(r"(?:เกณฑ์|เงื่อนไข|ต้องมีอะไร|ต้องทำอะไร|มีอะไรบ้าง)", text):
            return PolicyQuery("graduation_requirements")

    if "GPA" in text.upper() and _GPA_RE.search(question):
        if "พ้นสภาพ" in text and "นักศึกษา" in text:
            return PolicyQuery("student_status_termination_gpa")
        if "สำเร็จการศึกษา" in text or "จบการศึกษา" in text:
            return PolicyQuery("graduation_gpa")
        if "ติดโปร" in text or "ภาคทัณฑ์" in text:
            return PolicyQuery("probation_entry")
        if "พ้นโปร" in text or "พ้นภาคทัณฑ์" in text:
            return PolicyQuery("probation_cleared")
        if "เกียรตินิยม" in text:
            if "อันดับหนึ่ง" in text or "อันดับ 1" in text:
                return PolicyQuery("honors_first")
            if "อันดับสอง" in text or "อันดับ 2" in text:
                return PolicyQuery("honors_second")
        return None

    if "กลับเข้าศึกษา" in text and re.search(r"(?:กี่ปี|ภายในกี่ปี|เท่าไร|เท่าไหร่)", text):
        return PolicyQuery("reentry_limit")

    return None


__all__ = ["PolicyQuery", "parse_policy_question"]
