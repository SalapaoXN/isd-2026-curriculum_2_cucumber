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

# --- P0 typed fact binding (identity ↔ value ↔ unit ↔ relation) ---
#
# These helpers bind user-visible factual mentions to canonical verified
# facts. They deliberately parse only our own deterministic fact shapes
# (numbers beside credit units, placement markers, canonical relation
# cues) and closed cue lists for evidence domains that are absent from
# the VerifiedResult. They never attempt general Thai semantic
# understanding: anything that cannot be typed-checked fails closed to
# the deterministic renderer.

_CREDIT_MENTION_RE = re.compile(
    r"(\d[\d,]*\.?\d*)\s*(?:หน่วยกิต|credits?|หน่วย)", re.IGNORECASE
)
_CREDIT_NOTATION_RE = re.compile(
    r"(\d[\d,]*\.?\d*)\s*\(\s*\d+\s*-\s*\d+\s*-\s*\d+\s*\)"
)
_COUNT_MENTION_RE = re.compile(
    r"(\d[\d,]*)\s*(?:วิชา|รายวิชา|รายการ)"
)
_COUNT_TOTAL_RE = re.compile(r"จำนวน\s*:?\s*(\d[\d,]*)")
_DECIMAL_RE = re.compile(r"(\d+\.\d+)")
_YEAR_MARK_RE = re.compile(
    r"ชั้นปีที่\s*(\d+)|(?<![\wก-๙])ปี\s*(\d+)|year\s*(\d+)",
    re.IGNORECASE,
)
_SEMESTER_MARK_RE = re.compile(
    r"ภาคการศึกษาที่\s*(\d+)|(?<![\wก-๙])เทอม\s*(\d+)|semester\s*(\d+)",
    re.IGNORECASE,
)
_KNOWN_PLAN_WORDS = ("ไม่สหกิจ", "สหกิจ", "ไม่ระบุแผน", "default")
_NEGATION_RE = re.compile(r"ไม่ใช่|ไม่มี|ไม่|ยกเว้น")
_PLACEMENT_CUE_RE = re.compile(
    r"ชั้นปีที่|ภาคการศึกษาที่|เทอม|semester|แผนสหกิจ|แผนไม่สหกิจ"
)
_PREREQ_CUE_RE = re.compile(r"วิชาบังคับก่อน|บังคับก่อน")
_POLICY_CUE_RE = re.compile(
    r"ต้องสอบ|เกียรตินิยม|GPA|ภาษาอังกฤษ|ลงทะเบียน|ถอน|คุณสมบัติ"
    r"|จบการศึกษา|พ้นสภาพ|เกรดเฉลี่ย",
    re.IGNORECASE,
)
_PREREQ_CODE_KEYS = ("prerequisite_code", "prerequisite_course_id")

# Evidence domains granted by complete deterministic claim operations.
# Any operation outside this map (policy sentences, grounded summaries,
# unknown bridges) makes domain validation permissive instead of strict.
_CLAIM_OPERATION_DOMAINS = {
    "identity": frozenset({"identity"}),
    "sum_credits": frozenset({"credit"}),
    "placement": frozenset({"placement", "credit"}),
    "prerequisite": frozenset({"prerequisite"}),
    "list": frozenset({"list", "identity", "credit"}),
    "topic_matches": frozenset({"list", "identity", "credit"}),
    "course_set": frozenset({"list", "identity", "credit"}),
    "count": frozenset({"count"}),
    "compare": frozenset({"comparison"}),
}


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


def _parse_number(token: str) -> float | None:
    try:
        return float(token.replace(",", ""))
    except (TypeError, ValueError):
        return None


def _credit_mentions(text: str) -> list[float]:
    """Numbers directly bound to a credit unit or Thai credit notation."""
    found: list[float] = []
    for match in _CREDIT_MENTION_RE.finditer(text or ""):
        value = _parse_number(match.group(1))
        if value is not None:
            found.append(value)
    for match in _CREDIT_NOTATION_RE.finditer(text or ""):
        value = _parse_number(match.group(1))
        if value is not None and value not in found:
            found.append(value)
    return found


def _count_mentions(text: str) -> list[float]:
    """Bare course-count assertions (no credit unit attached)."""
    found: list[float] = []
    for pattern in (_COUNT_MENTION_RE, _COUNT_TOTAL_RE):
        for match in pattern.finditer(text or ""):
            value = _parse_number(match.group(1))
            if value is not None and value not in found:
                found.append(value)
    return found


def _decimal_mentions(text: str) -> list[float]:
    """Decimal measurements such as GPA thresholds."""
    found: list[float] = []
    for match in _DECIMAL_RE.finditer(text or ""):
        value = _parse_number(match.group(1))
        if value is not None and value not in found:
            found.append(value)
    return found


def _claim_count_values(value: object) -> list[float]:
    found: list[float] = []
    if isinstance(value, bool):
        return found
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, str):
        return _count_mentions(value)
    if isinstance(value, Mapping):
        for item in value.values():
            found.extend(_claim_count_values(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.extend(_claim_count_values(item))
    return found


def _verified_single_count(verified: VerifiedResult) -> float | None:
    """The one canonical course count, or None when not uniquely typed."""
    if verified.numeric_comparison is not None:
        return None
    if _displayed_list_pairs(verified):
        return None
    candidates: list[float] = []
    for claim in verified.claims:
        if (
            getattr(claim, "operation", None) == "count"
            and getattr(claim, "status", None) == "complete"
        ):
            candidates.extend(_claim_count_values(getattr(claim, "value", None)))
    for fact in _scan_texts(verified):
        candidates.extend(_count_mentions(fact))
    distinct = sorted(set(candidates))
    if len(distinct) == 1:
        return distinct[0]
    return None


def _verified_decimals(verified: VerifiedResult) -> set[float]:
    """Canonical decimal measurements (GPA thresholds and the like)."""
    found: set[float] = set()
    for fact in _scan_texts(verified):
        found.update(_decimal_mentions(fact))
    return found


def _marked_numbers(text: str, pattern: re.Pattern[str]) -> set[int]:
    """Study-year/semester assertions; ignores large non-study numbers."""
    found: set[int] = set()
    for match in pattern.finditer(text or ""):
        for group in match.groups():
            if not group:
                continue
            try:
                number = int(group)
            except ValueError:
                continue
            if 1 <= number <= 30:
                found.add(number)
    return found


def _answer_years(text: str) -> set[int]:
    return _marked_numbers(text, _YEAR_MARK_RE)


def _answer_semesters(text: str) -> set[int]:
    return _marked_numbers(text, _SEMESTER_MARK_RE)


def _answer_plan_labels(text: str) -> set[str]:
    return {
        word for word in _KNOWN_PLAN_WORDS if f"แผน{word}" in (text or "")
    }


def _segments(text: str) -> list[str]:
    parts = re.split(r"[\n.!?;…？！]+", text or "")
    return [part.strip() for part in parts if part.strip()]


def _claim_credit_values(value: object) -> list[float]:
    """Credit numbers from deterministic claim values (never bare ids)."""
    found: list[float] = []
    if isinstance(value, bool):
        return found
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, str):
        return _credit_mentions(value)
    if isinstance(value, Mapping):
        for item in value.values():
            found.extend(_claim_credit_values(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.extend(_claim_credit_values(item))
    return found


def _verified_single_credit(verified: VerifiedResult) -> float | None:
    """The one canonical credit value, or None when not uniquely typed."""
    if verified.numeric_comparison is not None:
        return None
    if _displayed_list_pairs(verified):
        return None
    candidates: list[float] = []
    for claim in verified.claims:
        if (
            getattr(claim, "operation", None) == "sum_credits"
            and getattr(claim, "status", None) == "complete"
        ):
            candidates.extend(_claim_credit_values(getattr(claim, "value", None)))
    for fact in _scan_texts(verified):
        candidates.extend(_credit_mentions(fact))
    distinct = sorted(set(candidates))
    if len(distinct) == 1:
        return distinct[0]
    return None


def _expected_placement(
    verified: VerifiedResult,
) -> tuple[set[int], set[int], set[str]]:
    """Canonical (years, semesters, plan labels) from verified evidence."""
    from rag.semantic.executor import _placement_terms

    years: set[int] = set()
    semesters: set[int] = set()
    plans: set[str] = set()
    for fact in _scan_texts(verified):
        years |= _marked_numbers(fact, _YEAR_MARK_RE)
        semesters |= _marked_numbers(fact, _SEMESTER_MARK_RE)
        plans |= {
            word for word in _KNOWN_PLAN_WORDS if f"แผน{word}" in fact
        }
    for claim in verified.claims:
        if (
            getattr(claim, "operation", None) == "placement"
            and getattr(claim, "status", None) == "complete"
        ):
            for year, semester, plan in _placement_terms(
                getattr(claim, "value", None)
            ):
                if isinstance(year, int) and 1 <= year <= 30:
                    years.add(year)
                if isinstance(semester, int) and 1 <= semester <= 30:
                    semesters.add(semester)
                if plan:
                    plans.add(_plan_display(plan))
    return years, semesters, plans


def _allowed_codes(verified: VerifiedResult) -> set[str] | None:
    """Every canonical code synthesis may repeat, or None when untyped."""
    from rag.semantic.executor import _codes as _executor_codes

    if not verified.claims and verified.numeric_comparison is None:
        return None
    allowed: set[str] = set()
    for fact in _scan_texts(verified):
        allowed.update(_COURSE_CODE_RE.findall(fact))
    comparison = verified.numeric_comparison
    if comparison is not None:
        for side in (comparison.left, comparison.right):
            code = side.course_code
            if isinstance(code, str) and _COURSE_CODE_RE.fullmatch(code.strip()):
                allowed.add(code.strip())
    for claim in verified.claims:
        try:
            allowed.update(_executor_codes(getattr(claim, "value", None)))
        except Exception:
            continue
    return allowed


def _prereq_codes(verified: VerifiedResult) -> set[str]:
    """Explicit canonical prerequisite codes from deterministic evidence."""
    found: set[str] = set()

    def collect(value: object) -> None:
        if isinstance(value, Mapping):
            for key in _PREREQ_CODE_KEYS:
                code = value.get(key)
                if isinstance(code, str) and _COURSE_CODE_RE.fullmatch(
                    code.strip()
                ):
                    found.add(code.strip())
            for item in value.values():
                if isinstance(item, (Mapping, list, tuple)):
                    collect(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect(item)

    for claim in verified.claims:
        if (
            getattr(claim, "operation", None) == "prerequisite"
            and getattr(claim, "status", None) == "complete"
        ):
            collect(getattr(claim, "value", None))
    return found


def _evidence_domains(verified: VerifiedResult) -> set[str] | None:
    """Evidence domains synthesis may draw on; None means untyped/permissive."""
    claims = verified.claims or ()
    if not claims:
        if verified.numeric_comparison is not None:
            return {"comparison", "placement", "identity", "credit"}
        return None
    domains: set[str] = set()
    for claim in claims:
        if getattr(claim, "status", None) != "complete":
            return None
        granted = _CLAIM_OPERATION_DOMAINS.get(
            getattr(claim, "operation", None), None
        )
        if granted is None:
            return None
        domains |= set(granted)
    if verified.numeric_comparison is not None:
        domains |= {"comparison", "placement", "identity", "credit"}
    return domains


def _strip_identifiers(text: str, verified: VerifiedResult) -> str:
    scrubbed = text or ""
    for identifier in _required_identifiers(verified):
        if identifier:
            scrubbed = scrubbed.replace(identifier, " ")
    allowed = _allowed_codes(verified) or set()
    for code in allowed:
        scrubbed = scrubbed.replace(code, " ")
    return scrubbed


def _list_credit_map(verified: VerifiedResult) -> dict[str, float]:
    """Displayed code → canonical credit for list rows carrying credits."""
    mapping: dict[str, float] = {}
    for claim in verified.claims:
        if (
            getattr(claim, "operation", None)
            not in {"list", "topic_matches", "course_set"}
            or getattr(claim, "status", None) != "complete"
        ):
            continue
        value = getattr(claim, "value", None)
        rows = value if isinstance(value, (list, tuple)) else [value]
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            code = row.get("course_code")
            if (
                not isinstance(code, str)
                or not _COURSE_CODE_RE.fullmatch(code.strip())
                or code.strip() in mapping
            ):
                continue
            raw = row.get("credits", row.get("credit"))
            parsed: float | None = None
            if isinstance(raw, bool):
                continue
            if isinstance(raw, (int, float)):
                parsed = float(raw)
            elif isinstance(raw, str):
                notations = _credit_mentions(raw)
                parsed = notations[0] if notations else _parse_number(raw.strip())
            if parsed is not None:
                mapping[code.strip()] = parsed
    return mapping


def validate_answer_details(
    answer: str, verified: VerifiedResult
) -> tuple[bool, str]:
    """Typed validation returning (accepted, machine reason for traces).

    The reason string is diagnostics-only and never shown to users.
    """
    if not isinstance(answer, str) or not answer.strip():
        return False, "empty_answer"
    lowered = answer.casefold()
    plan_labels = tuple(
        f"แผน{_plan_display(plan)}" for plan in ("coop", "no_coop")
        if any(f"แผน{_plan_display(plan)}" in fact for fact in _scan_texts(verified))
    )
    if plan_labels and (
        _INTERNAL_PLAN_RE.search(answer) or any(label not in answer for label in plan_labels)
    ):
        return False, "plan_label_mismatch"
    for token in _FORBIDDEN_JARGON:
        if token in lowered:
            return False, "unsupported_answer_fact"
    for identifier in _required_identifiers(verified):
        if identifier not in answer:
            return False, "identifier_missing"
    for code, title in _displayed_list_pairs(verified):
        if title and not any(
            code in line and title in line
            and set(_COURSE_CODE_RE.findall(line)) == {code}
            for line in answer.splitlines()
        ):
            # Do not accept names detached from or paired with other codes.
            return False, "list_pair_mismatch"
    allowed = _allowed_codes(verified)
    if allowed is not None:
        for code in set(_COURSE_CODE_RE.findall(answer)):
            if code not in allowed:
                return False, "unsupported_answer_fact"
    credit_value = _verified_single_credit(verified)
    if credit_value is not None:
        for segment in _segments(answer):
            mentions = _credit_mentions(segment)
            if any(value != credit_value for value in mentions):
                return False, "answer_value_mismatch"
            if _NEGATION_RE.search(segment) and credit_value in mentions:
                return False, "answer_relation_mismatch"
    count_value = _verified_single_count(verified)
    if count_value is not None:
        for segment in _segments(answer):
            mentions = _count_mentions(segment)
            if any(value != count_value for value in mentions):
                return False, "answer_value_mismatch"
            if _NEGATION_RE.search(segment) and count_value in mentions:
                return False, "answer_relation_mismatch"
    canonical_decimals = _verified_decimals(verified)
    if canonical_decimals:
        for value in _decimal_mentions(answer):
            if value not in canonical_decimals:
                return False, "answer_value_mismatch"
    expected_years, expected_semesters, _ = _expected_placement(verified)
    if expected_years and not set(_answer_years(answer)) <= expected_years:
        return False, "answer_value_mismatch"
    if expected_semesters and not set(_answer_semesters(answer)) <= expected_semesters:
        return False, "answer_value_mismatch"
    _, _, expected_plans = _expected_placement(verified)
    if expected_plans and not _answer_plan_labels(answer) <= expected_plans:
        return False, "answer_value_mismatch"
    prereqs = _prereq_codes(verified)
    if prereqs:
        for segment in _segments(answer):
            if _NEGATION_RE.search(segment) and _PREREQ_CUE_RE.search(segment):
                return False, "answer_relation_mismatch"
            if "มีวิชาบังคับก่อน" in segment:
                objects = _COURSE_CODE_RE.findall(
                    segment.split("มีวิชาบังคับก่อน", 1)[1]
                )
                if any(code not in prereqs for code in objects):
                    return False, "answer_relation_mismatch"
    credit_map = _list_credit_map(verified)
    if _displayed_list_pairs(verified):
        for line in answer.splitlines():
            codes = [
                code
                for code in set(_COURSE_CODE_RE.findall(line))
                if code in dict(_displayed_list_pairs(verified))
            ]
            if len(codes) != 1:
                continue
            mentions = _credit_mentions(line)
            if not mentions:
                continue
            expected = credit_map.get(codes[0])
            if expected is None:
                # Displayed values without canonical evidence are unsupported.
                return False, "unsupported_answer_fact"
            if any(value != expected for value in mentions):
                return False, "answer_value_mismatch"
    domains = _evidence_domains(verified)
    if domains is not None:
        scrubbed = _strip_identifiers(answer, verified)
        if _PLACEMENT_CUE_RE.search(scrubbed) and "placement" not in domains:
            return False, "unsupported_answer_fact"
        prereq_hit = _PREREQ_CUE_RE.search(scrubbed) or (
            "ต้องผ่าน" in scrubbed and "มาก่อน" in scrubbed
        )
        if prereq_hit and "prerequisite" not in domains:
            return False, "unsupported_answer_fact"
        if _POLICY_CUE_RE.search(scrubbed):
            return False, "unsupported_answer_fact"
    return True, "ok"


def validate_answer_text(answer: str, verified: VerifiedResult) -> bool:
    """Return True when rendered text preserves identifiers without jargon."""
    accepted, _ = validate_answer_details(answer, verified)
    return accepted


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


def _side_marker_pattern(marker: str) -> re.Pattern[str]:
    """Boundary-safe matcher for one comparison-side identity string."""
    if marker and all(char.isascii() and (char.isalnum() or char == "_") for char in marker):
        return re.compile(
            r"(?<![A-Za-z0-9_])" + re.escape(marker) + r"(?![A-Za-z0-9_])"
        )
    return re.compile(re.escape(marker or ""))


def _comparison_side_markers(
    side: VerifiedNumericComparisonSide,
) -> tuple[str, ...]:
    markers: list[str] = []
    for marker in (side.course_code, side.course_name, side.label):
        if isinstance(marker, str) and marker.strip() and marker not in markers:
            markers.append(marker)
    return tuple(markers)


def _resolve_comparison_side(
    comparison: VerifiedNumericComparison, fragment: str
) -> VerifiedNumericComparisonSide | None:
    """Map a summary fragment to one side, or None when ambiguous."""
    matched: list[VerifiedNumericComparisonSide] = []
    for side in (comparison.left, comparison.right):
        markers = _comparison_side_markers(side)
        hits = [
            marker
            for marker in markers
            if marker and _side_marker_pattern(marker).search(fragment)
        ]
        if hits:
            # Prefer the longest (most specific) identity that matched.
            hits.sort(key=len, reverse=True)
            if _side_marker_pattern(hits[0]).search(fragment):
                matched.append(side)
    if len(matched) != 1:
        return None
    return matched[0]


def _comparison_answer_details(
    answer: str, comparison: VerifiedNumericComparison
) -> tuple[bool, str]:
    """Typed side binding (identity ↔ value ↔ unit ↔ relation)."""
    if "สรุป:" not in answer:
        return False, "comparison_verdict_missing"
    body, summary = answer.split("สรุป:", 1)
    summary = summary.strip()
    body_lines = body.splitlines()
    sides = (comparison.left, comparison.right)
    for side in sides:
        for identifier in (side.course_code, side.course_name):
            if identifier and identifier.casefold() not in answer.casefold():
                return False, "identifier_missing"
        formatted = _format_comparison_value(side.value)
        required_value = (
            f"{formatted} หน่วยกิต"
            if comparison.measure == "credits"
            else formatted
        )
        if required_value not in answer:
            return False, "answer_value_mismatch"
        if side.course_code and answer.count(side.course_code) != 1:
            return False, "comparison_side_binding_mismatch"
    for side in sides:
        markers = _comparison_side_markers(side)
        primary = markers[0] if markers else ""
        if not primary:
            return False, "comparison_side_binding_mismatch"
        pattern = _side_marker_pattern(primary)
        bound = [line for line in body_lines if pattern.search(line)]
        if not bound:
            return False, "comparison_side_binding_mismatch"
        formatted = _format_comparison_value(side.value)
        required_value = (
            f"{formatted} หน่วยกิต"
            if comparison.measure == "credits"
            else formatted
        )
        if not any(required_value in line for line in bound):
            # The verified value must stay attached to its own side's line.
            return False, "comparison_side_binding_mismatch"
        if comparison.measure == "credits":
            for line in bound:
                for mention in _credit_mentions(line):
                    if mention != float(side.value):
                        return False, "comparison_side_binding_mismatch"
    if comparison.actual_relation == "equal":
        expected_relation = _verified_relation_sentence(comparison)
        if expected_relation not in summary:
            return False, "answer_relation_mismatch"
        if "มากกว่า" in summary or "น้อยกว่า" in summary:
            return False, "answer_relation_mismatch"
        return True, "ok"
    if "เท่ากัน" in summary:
        return False, "answer_relation_mismatch"
    if comparison.requested_operation == "difference":
        difference = _format_comparison_value(comparison.absolute_difference)
        if difference not in summary:
            return False, "answer_value_mismatch"
    else:
        expected_relation = _verified_relation_sentence(comparison)
        if expected_relation not in summary:
            return False, "answer_relation_mismatch"
    greater_side = (
        comparison.left
        if comparison.actual_relation == "left_greater"
        else comparison.right
    )
    lesser_side = (
        comparison.right
        if comparison.actual_relation == "left_greater"
        else comparison.left
    )
    if comparison.actual_relation in ("left_greater", "right_greater"):
        swapped = VerifiedNumericComparison(
            measure=comparison.measure,
            requested_operation=comparison.requested_operation,
            actual_relation=(
                "right_greater"
                if comparison.actual_relation == "left_greater"
                else "left_greater"
            ),
            left=comparison.left,
            right=comparison.right,
            absolute_difference=comparison.absolute_difference,
        )
        opposite = _verified_relation_sentence(swapped)
        if opposite and opposite != _verified_relation_sentence(comparison):
            if opposite in summary:
                return False, "answer_relation_mismatch"
    winner = re.search(
        r"(.+?)\s*มี(?:หน่วยกิต|ค่า)?มากกว่า\s*(.+)", summary
    )
    if winner:
        winner_side = _resolve_comparison_side(comparison, winner.group(1))
        loser_side = _resolve_comparison_side(comparison, winner.group(2))
        if winner_side is not None and loser_side is not None:
            if winner_side is not greater_side or loser_side is not lesser_side:
                return False, "answer_relation_mismatch"
    loser = re.search(
        r"(.+?)\s*มี(?:หน่วยกิต|ค่า)?น้อยกว่า\s*(.+)", summary
    )
    if loser:
        lesser_hit = _resolve_comparison_side(comparison, loser.group(1))
        greater_hit = _resolve_comparison_side(comparison, loser.group(2))
        if lesser_hit is not None and greater_hit is not None:
            if lesser_hit is not lesser_side or greater_hit is not greater_side:
                return False, "answer_relation_mismatch"
    return True, "ok"


def _comparison_answer_valid(
    answer: str, comparison: VerifiedNumericComparison
) -> bool:
    """Conservatively validate identifiers, values, and the explicit verdict."""
    accepted, _ = _comparison_answer_details(answer, comparison)
    return accepted


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
    "validate_answer_details",
    "validate_answer_text",
]
