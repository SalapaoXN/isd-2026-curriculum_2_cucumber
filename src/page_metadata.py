import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
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
@dataclass(frozen=True)
class DocumentPageResolution:
    document_page: int | None
    reason: str


# These are the visually verified pages that cannot be resolved by the
# production OCR detector or the bounded GENED rule below.  The audit map is
# verification evidence only and is deliberately not loaded by production.
SOURCE_VERIFIED_DOCUMENT_PAGE_OVERRIDES = {
    ('DSBA', 'dsba_page_028.png', 28): 23,
    ('DSBA', 'dsba_page_030.png', 30): 25,
    ('DSBA', 'dsba_page_032.png', 32): 27,
    ('DSBA', 'dsba_page_035.png', 35): 30,
    ('DSBA', 'dsba_page_037.png', 37): 32,
    ('DSBA', 'dsba_page_039.png', 39): 34,
    ('DSBA', 'dsba_page_317.png', 317): 312,
    ('DSBA', 'dsba_page_318.png', 318): 313,
    ('DSBA', 'dsba_page_319.png', 319): 314,
    ('DSBA', 'dsba_page_320.png', 320): 315,
    ('DSBA', 'dsba_page_321.png', 321): 316,
    ('DSBA', 'dsba_page_322.png', 322): 317,
    ('DSBA', 'dsba_page_323.png', 323): 318,
    ('DSBA', 'dsba_page_324.png', 324): 319,
    ('DSBA', 'dsba_page_325.png', 325): 320,
    ('DSBA', 'dsba_page_326.png', 326): 321,
    ('DSBA', 'dsba_page_327.png', 327): 322,
    ('DSBA', 'dsba_page_328.png', 328): 323,
    ('DSBA', 'dsba_page_329.png', 329): 324,
    ('DSBA', 'dsba_page_330.png', 330): 325,
    ('DSBA', 'dsba_page_331.png', 331): 326,
    ('DSBA', 'dsba_page_332.png', 332): 327,
    ('DSBA', 'dsba_page_333.png', 333): 328,
    ('DSBA', 'dsba_page_334.png', 334): 329,
    ('DSBA', 'dsba_page_335.png', 335): 330,
    ('DSBA', 'dsba_page_336.png', 336): 331,
    ('DSBA', 'dsba_page_337.png', 337): 332,
    ('DSBA', 'dsba_page_338.png', 338): 333,
    ('DSBA', 'dsba_page_339.png', 339): 334,
    ('DSBA', 'dsba_page_340.png', 340): 335,
    ('DSBA', 'dsba_page_341.png', 341): 336,
    ('DSBA', 'dsba_page_342.png', 342): 337,
    ('DSBA', 'dsba_page_343.png', 343): 338,
    ('DSBA', 'dsba_page_344.png', 344): 339,
    ('IT', 'it_page_032.png', 32): 27,
    ('IT', 'it_page_034.png', 34): 29,
    ('IT', 'it_page_036.png', 36): 31,
    ('IT', 'it_page_038.png', 38): 33,
    ('IT', 'it_page_039.png', 39): 34,
    ('IT', 'it_page_041.png', 41): 36,
    ('IT', 'it_page_043.png', 43): 38,
    ('IT', 'it_page_328.png', 328): 323,
    ('IT', 'it_page_329.png', 329): 324,
    ('IT', 'it_page_330.png', 330): 325,
    ('IT', 'it_page_331.png', 331): 326,
    ('IT', 'it_page_332.png', 332): 327,
    ('IT', 'it_page_333.png', 333): 328,
    ('IT', 'it_page_334.png', 334): 329,
    ('IT', 'it_page_335.png', 335): 330,
    ('IT', 'it_page_336.png', 336): 331,
    ('IT', 'it_page_337.png', 337): 332,
    ('IT', 'it_page_338.png', 338): 333,
    ('IT', 'it_page_339.png', 339): 334,
    ('IT', 'it_page_340.png', 340): 335,
    ('IT', 'it_page_341.png', 341): 336,
    ('IT', 'it_page_342.png', 342): 337,
    ('IT', 'it_page_343.png', 343): 338,
    ('IT', 'it_page_344.png', 344): 339,
    ('IT', 'it_page_345.png', 345): 340,
    ('IT', 'it_page_346.png', 346): 341,
    ('IT', 'it_page_347.png', 347): 342,
    ('IT', 'it_page_348.png', 348): 343,
    ('IT', 'it_page_349.png', 349): 344,
    ('IT', 'it_page_350.png', 350): 345,
    ('IT', 'it_page_351.png', 351): 346,
    ('IT', 'it_page_352.png', 352): 347,
    ('IT', 'it_page_353.png', 353): 348,
    ('IT', 'it_page_354.png', 354): 349,
    ('IT', 'it_page_355.png', 355): 350,
    ('IT', 'it_page_356.png', 356): 351,
    ('IT', 'it_page_357.png', 357): 352,
    ('IT', 'it_page_358.png', 358): 353,
    ('IT', 'it_page_359.png', 359): 354,
    ('IT', 'it_page_360.png', 360): 355,
    ('IT', 'it_page_361.png', 361): 356,
    ('IT', 'it_page_362.png', 362): 357,
    ('IT', 'it_page_363.png', 363): 358,
    ('IT', 'it_page_364.png', 364): 359,
    ('IT', 'it_page_365.png', 365): 360,
    ('IT', 'it_page_366.png', 366): 361,
    ('IT', 'it_page_367.png', 367): 362,
    ('IT', 'it_page_368.png', 368): 363,
    ('IT', 'it_page_369.png', 369): 364,
    ('IT', 'it_page_370.png', 370): 365,
    ('IT', 'it_page_371.png', 371): 366,
}


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


def _source_verified_override(
    program: str | None,
    source_page: int | None,
    source_filename: str | Path | None,
) -> int | None:
    if (
        not isinstance(program, str)
        or not isinstance(source_page, int)
        or isinstance(source_page, bool)
        or source_filename is None
    ):
        return None
    filename = Path(source_filename).name
    return SOURCE_VERIFIED_DOCUMENT_PAGE_OVERRIDES.get(
        (program.strip().upper(), filename, source_page)
    )


def _bounded_document_page(program: str | None, source_page: int | None) -> int | None:
    if (
        isinstance(program, str)
        and isinstance(source_page, int)
        and not isinstance(source_page, bool)
        and program.strip().upper() == "GENED"
        and 44 <= source_page <= 117
    ):
        return source_page - 4
    return None


def resolve_document_page(
    program: str | None,
    source_page: int | None,
    source_filename: str | Path | None,
    text_lines: Iterable[Any] | None,
    explicit_document_page: Any = None,
    allow_bounded_rule: bool = True,
) -> DocumentPageResolution:
    """Resolve document-page metadata from trusted, deterministic evidence."""
    explicit = parse_document_page(explicit_document_page)
    if explicit is not None:
        return DocumentPageResolution(explicit, "explicit")

    detector_candidates = document_page_candidates_from_lines(text_lines)
    detector_page = next(iter(detector_candidates)) if len(detector_candidates) == 1 else None
    bounded_page = (
        _bounded_document_page(program, source_page) if allow_bounded_rule else None
    )
    override_page = _source_verified_override(program, source_page, source_filename)
    rule_page = bounded_page if bounded_page is not None else override_page
    rule_reason = "bounded_rule" if bounded_page is not None else "source_verified_override"

    if detector_page is not None:
        if rule_page is not None and detector_page != rule_page:
            return DocumentPageResolution(None, "conflict")
        return DocumentPageResolution(detector_page, "ocr_unique")

    if detector_candidates:
        if rule_page is not None and rule_page in detector_candidates:
            return DocumentPageResolution(rule_page, rule_reason)
        return DocumentPageResolution(None, "unresolved")

    if rule_page is not None:
        return DocumentPageResolution(rule_page, rule_reason)

    return DocumentPageResolution(None, "unresolved")
