import re
from collections.abc import Iterable
from typing import Any


THAI_DIGIT_TRANSLATION = str.maketrans(
    "๐๑๒๓๔๕๖๗๘๙",
    "0123456789",
)
DOCUMENT_PAGE_RE = re.compile(r"^(?:หน้า\s*[:\uFF1A-]?\s*)?([0-9]{1,4})$")
NON_PAGE_CONTEXT_RE = re.compile(
    r"^\s*(?:รวม|total|หน่วยกิต|credits|ปีที่|ชั้นปี|ภาคการศึกษา|"
    r"semester|year|term)\s*[:\uFF1A-]?\s*$",
    re.IGNORECASE,
)
CREDIT_LINE_RE = re.compile(
    r"^\s*(?:\d+\s*)?\(?\d+\s*[-–]\s*\d+\s*[-–]\s*\d+\)?\s*$"
)


def parse_document_page(value: Any) -> int | None:
    """Parse a printed page marker without treating source pages as evidence."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if 0 < value <= 9999 else None
    if not isinstance(value, str):
        return None

    normalized = value.lstrip("\ufeff").strip().translate(THAI_DIGIT_TRANSLATION)
    match = DOCUMENT_PAGE_RE.fullmatch(normalized)
    if not match:
        return None

    page = int(match.group(1))
    return page if page > 0 else None


def document_page_candidates_from_lines(lines: Iterable[Any] | None) -> set[int]:
    """Return unique printed-page candidates from the OCR page boundaries."""
    if lines is None:
        return set()

    non_empty_lines = []
    for line in lines:
        if line is None:
            continue
        text = str(line).strip()
        if text:
            non_empty_lines.append(text)

    if not non_empty_lines:
        return set()

    boundary_indexes = set(range(min(5, len(non_empty_lines))))
    boundary_indexes.update(range(max(0, len(non_empty_lines) - 5), len(non_empty_lines)))
    candidates = []

    for index in sorted(boundary_indexes):
        page = parse_document_page(non_empty_lines[index])
        if page is None:
            continue

        if index > 0:
            previous_line = non_empty_lines[index - 1]
            if NON_PAGE_CONTEXT_RE.fullmatch(previous_line):
                continue

        if index + 1 < len(non_empty_lines):
            next_line = non_empty_lines[index + 1]
            if CREDIT_LINE_RE.fullmatch(next_line):
                continue

        candidates.append(page)

    return set(candidates)


def document_page_from_lines(lines: Iterable[Any] | None) -> int | None:
    """Read a printed page marker from the OCR page boundaries."""
    normalized_candidates = document_page_candidates_from_lines(lines)
    if len(normalized_candidates) == 1:
        return next(iter(normalized_candidates))
    return None
