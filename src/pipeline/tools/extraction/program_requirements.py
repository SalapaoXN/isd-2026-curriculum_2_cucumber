"""Extract program-level requirements from curriculum overview pages."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Mapping


PROGRAM_REQUIREMENT_SOURCES: Mapping[str, tuple[str, int, int, str]] = {
    "AIT": ("ait2566_page_005.png", 5, 1, "ait-2566"),
    "BIT": ("bit2565_page_006.png", 6, 1, "bit-2565"),
    "DSBA": ("dsba2565_page_006.png", 6, 1, "dsba-2565"),
    "IT": ("it2565_page_006.png", 6, 1, "it-2565"),
}

_TOTAL_CREDITS_HEADING_RE = re.compile(
    r"จำนวน\s*หน่วยกิต\s*ที่\s*เรียน\s*ตลอด\s*หลักสูตร"
)
_CREDIT_VALUE_RE = re.compile(
    r"(?P<value>[0-9๐-๙Oo]+)\s*หน่วยกิต"
)
_THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")


class ProgramRequirementExtractionError(ValueError):
    """Raised when a source page does not yield one safe credit value."""


def _normalize_lines(lines: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    for line in lines:
        if not isinstance(line, str):
            continue
        value = " ".join(line.split())
        if value:
            normalized.append(value)
    return normalized


def _normalize_credit_value(raw_value: str) -> int | None:
    value = raw_value.translate(_THAI_DIGITS).replace("O", "0").replace("o", "0")
    if not value.isdigit():
        return None
    normalized = int(value)
    return normalized if normalized > 0 else None


def _source_provenance(
    program: str,
    source_filename: str,
    source_page: int,
    document_page: int | None = None,
) -> list[dict[str, Any]]:
    entry: dict[str, Any] = {
        "program": program,
        "source_filename": source_filename,
        "source_page": source_page,
        "document_category": "program_requirement",
    }
    if document_page is not None:
        entry["document_page"] = document_page
    return [entry]


def extract_program_requirement_from_lines(
    program: str,
    lines: Iterable[str],
    *,
    source_filename: str,
    source_page: int,
    document_page: int | None = None,
) -> dict[str, Any]:
    """Extract one total-credit fact from already OCR'd source lines.

    The parser intentionally considers only values close to the exact overview
    heading. It does not use program-code prefixes or a fallback value.
    """
    normalized_program = str(program).strip().upper()
    if normalized_program not in PROGRAM_REQUIREMENT_SOURCES:
        raise ProgramRequirementExtractionError(
            f"Unsupported program requirement source: {program!r}"
        )

    normalized_lines = _normalize_lines(lines)
    catalog_key = PROGRAM_REQUIREMENT_SOURCES[normalized_program][3]
    anchor_indexes = [
        index
        for index, line in enumerate(normalized_lines)
        if _TOTAL_CREDITS_HEADING_RE.search(line)
    ]
    if not anchor_indexes:
        raise ProgramRequirementExtractionError(
            f"Total-credit heading not found for {normalized_program}"
        )

    candidates: list[tuple[int, str]] = []
    for anchor_index in anchor_indexes:
        window = " ".join(normalized_lines[anchor_index : anchor_index + 4])
        for match in _CREDIT_VALUE_RE.finditer(window):
            value = _normalize_credit_value(match.group("value"))
            if value is not None:
                candidate = (value, match.group("value"))
                if candidate not in candidates:
                    candidates.append(candidate)

    values = sorted({value for value, _ in candidates})
    if len(values) != 1:
        reason = "no bounded credit value" if not values else "multiple credit values"
        raise ProgramRequirementExtractionError(
            f"Could not safely extract {normalized_program} total credits: {reason}"
        )

    return {
        "program": normalized_program,
        "catalog_key": catalog_key,
        "requirement_type": "total_program_credits",
        "operator": "=",
        "value": values[0],
        "unit": "credits",
        "source_provenance": _source_provenance(
            normalized_program,
            source_filename,
            source_page,
            document_page,
        ),
    }


def extract_program_requirement_from_image(
    program: str,
    image_path: str | Path,
    *,
    ocr_engine: Any | None = None,
    gpu: bool = False,
) -> dict[str, Any]:
    """OCR and extract one program requirement from its source page."""
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Program requirement image not found: {path}")
    if ocr_engine is None:
        from src.pipeline.tools.ocr.engine import OCREngine

        ocr_engine = OCREngine(languages=["th", "en"], gpu=gpu)
    lines = ocr_engine.extract_text(path, detail=0)
    _, source_page, document_page, _ = PROGRAM_REQUIREMENT_SOURCES[
        str(program).strip().upper()
    ]
    return extract_program_requirement_from_lines(
        program,
        lines,
        source_filename=path.name,
        source_page=source_page,
        document_page=document_page,
    )


def extract_program_requirements(
    source_dir: str | Path,
    *,
    ocr_engine: Any | None = None,
    gpu: bool = False,
) -> list[dict[str, Any]]:
    """Extract all four program totals from the configured source pages."""
    root = Path(source_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Program requirement source directory not found: {root}")
    if ocr_engine is None:
        from src.pipeline.tools.ocr.engine import OCREngine

        ocr_engine = OCREngine(languages=["th", "en"], gpu=gpu)

    results: list[dict[str, Any]] = []
    for program, (filename, source_page, document_page, _) in PROGRAM_REQUIREMENT_SOURCES.items():
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(f"Program requirement image not found: {path}")
        lines = ocr_engine.extract_text(path, detail=0)
        results.append(
            extract_program_requirement_from_lines(
                program,
                lines,
                source_filename=filename,
                source_page=source_page,
                document_page=document_page,
            )
        )
    return results


# Plan-page extraction is intentionally separate from the overview-page API
# above. Callers supply the catalog identity and page provenance explicitly;
# this module does not infer an edition from a program name or OCR filename.
_PLAN_TOTAL_ANCHOR_RE = re.compile(
    r"รวม\s*ตลอด\s*หลักสูตร"
    r"|total\s+(?:credits?\s+(?:for\s+)?(?:the\s+)?(?:entire\s+)?(?:program|curriculum)"
    r"|(?:program|curriculum)\s+total\s+credits?)",
    re.IGNORECASE,
)
_GENERAL_ED_TOTAL_ANCHOR_RE = re.compile(
    r"รวม\s*(?:จำนวน\s*)?(?:หน่วยกิต\s*)?(?:หมวด\s*)?วิชา\s*ศึกษา\s*ทั่วไป"
    r"|total\s+credits?\s+(?:for|in)\s+(?:the\s+)?general\s+education"
    r"(?:\s+block)?|general\s+education\s+(?:block\s+)?total\s+credits?",
    re.IGNORECASE,
)
_PLAN_TOTAL_VALUE_RE = re.compile(
    r"^\s*[:：\-]?\s*(?:จำนวน\s*)?"
    r"(?P<value>[0-9๐-๙]+)\s*(?:หน่วยกิต|credits?)",
    re.IGNORECASE,
)
_THAI_TOTAL_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
_PLAN_TOTAL_LOOKAHEAD = 80


def extract_explicit_total_credits(
    text: str,
    *,
    requirement_type: str = "total_program_credits",
) -> int | None:
    """Return a credit total only when it follows an explicit total anchor.

    Whitespace and Unicode presentation artifacts are normalized before
    matching, and only a nearby number followed by a credit unit is accepted.
    An unanchored number is never treated as a program requirement.
    """
    if not isinstance(text, str):
        raise TypeError("OCR text must be a string")

    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\u200b", "").replace("\ufeff", "")
    normalized = " ".join(normalized.split())

    if requirement_type == "total_program_credits":
        anchors = _PLAN_TOTAL_ANCHOR_RE
    elif requirement_type == "general_education_total_credits":
        anchors = _GENERAL_ED_TOTAL_ANCHOR_RE
    else:
        raise ValueError(f"Unsupported requirement_type: {requirement_type!r}")

    values: set[int] = set()
    for anchor in anchors.finditer(normalized):
        bounded_text = normalized[
            anchor.end() : anchor.end() + _PLAN_TOTAL_LOOKAHEAD
        ]
        value_match = _PLAN_TOTAL_VALUE_RE.match(bounded_text)
        if value_match is None:
            continue
        digits = value_match.group("value").translate(_THAI_TOTAL_DIGITS)
        value = int(digits)
        if value > 0:
            values.add(value)

    if len(values) > 1:
        raise ProgramRequirementExtractionError(
            "Conflicting explicit total-credit values found on one source page"
        )
    return next(iter(values), None)


def extract_total_credit_requirement(
    text: str,
    *,
    requirement_type: str,
    source_provenance: Iterable[Mapping[str, Any]],
    catalog_key: str | None = None,
    program: str | None = None,
    scope: str | None = None,
) -> dict[str, Any] | None:
    """Build one normalized requirement from an explicitly scoped OCR page.

    Degree totals require caller-supplied ``catalog_key`` and ``program``.
    General Education totals use ``scope="GENED"`` and remain distinct from
    degree-program requirements. A page without an explicit total returns
    ``None``.
    """
    if requirement_type == "total_program_credits":
        if not isinstance(catalog_key, str) or not catalog_key.strip():
            raise ValueError("degree requirements need an explicit catalog_key")
        if not isinstance(program, str) or not program.strip():
            raise ValueError("degree requirements need an explicit program")
        identity = {
            "catalog_key": catalog_key.strip(),
            "program": program.strip().upper(),
        }
    elif requirement_type == "general_education_total_credits":
        if not isinstance(scope, str) or scope.strip().upper() != "GENED":
            raise ValueError("General Education requirements need scope='GENED'")
        identity = {"scope": "GENED"}
    else:
        raise ValueError(f"Unsupported requirement_type: {requirement_type!r}")

    provenance: list[dict[str, Any]] = []
    for item in source_provenance:
        if not isinstance(item, Mapping):
            raise TypeError("source_provenance entries must be mappings")
        provenance.append(dict(item))
    if not provenance:
        raise ValueError("source_provenance must contain at least one source")

    value = extract_explicit_total_credits(
        text,
        requirement_type=requirement_type,
    )
    if value is None:
        return None

    return {
        **identity,
        "requirement_type": requirement_type,
        "operator": "=",
        "value": value,
        "unit": "credits",
        "source_provenance": provenance,
    }


def merge_plan_requirements(
    requirements: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Merge plan-specific evidence for each canonical requirement identity.

    Equal totals retain provenance from each plan. Conflicting values or
    requirement semantics fail closed instead of selecting one plan.
    """
    merged: dict[tuple[str, ...], dict[str, Any]] = {}
    for source in requirements:
        if not isinstance(source, Mapping):
            raise TypeError("requirements must contain mappings")
        record = dict(source)
        requirement_type = record.get("requirement_type")
        if requirement_type == "total_program_credits":
            catalog_key = record.get("catalog_key")
            program = record.get("program")
            if not isinstance(catalog_key, str) or not catalog_key.strip():
                raise ValueError("degree requirements need an explicit catalog_key")
            if not isinstance(program, str) or not program.strip():
                raise ValueError("degree requirements need an explicit program")
            identity = (requirement_type, catalog_key.strip(), program.strip().upper())
        elif requirement_type == "general_education_total_credits":
            if str(record.get("scope", "")).strip().upper() != "GENED":
                raise ValueError("General Education requirements need scope='GENED'")
            identity = (requirement_type, "GENED")
        else:
            raise ValueError(f"Unsupported requirement_type: {requirement_type!r}")

        if not isinstance(record.get("source_provenance"), list) or not record[
            "source_provenance"
        ]:
            raise ValueError("requirements need non-empty source_provenance")
        if record.get("operator") != "=" or record.get("unit") != "credits":
            raise ValueError("requirements must use operator '=' and unit 'credits'")
        value = record.get("value")
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("requirement values must be positive integers")
        if any(not isinstance(item, Mapping) for item in record["source_provenance"]):
            raise TypeError("source_provenance entries must be mappings")

        current = merged.get(identity)
        if current is None:
            merged[identity] = record
            continue

        comparable = ("operator", "value", "unit")
        if any(current.get(key) != record.get(key) for key in comparable):
            raise ProgramRequirementExtractionError(
                "Conflicting plan-specific requirements for "
                f"{identity!r}: {current.get('value')!r} vs {record.get('value')!r}"
            )
        for source_provenance in record["source_provenance"]:
            if source_provenance not in current["source_provenance"]:
                current["source_provenance"].append(dict(source_provenance))

    return list(merged.values())
