"""Extract program-level requirements from curriculum overview pages."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Mapping


PROGRAM_REQUIREMENT_SOURCES: Mapping[str, tuple[str, int, int]] = {
    "AIT": ("ait_page_005.png", 5, 1),
    "BIT": ("bit_page_006.png", 6, 1),
    "DSBA": ("dsba_page_006.png", 6, 1),
    "IT": ("it_page_006.png", 6, 1),
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
    source_page, document_page = PROGRAM_REQUIREMENT_SOURCES[
        str(program).strip().upper()
    ][1:]
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
    for program, (filename, source_page, document_page) in PROGRAM_REQUIREMENT_SOURCES.items():
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
