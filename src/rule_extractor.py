"""Deterministic extraction of numbered academic rules from OCR text."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence


THAI_DIGIT_TRANSLATION = str.maketrans(
    "๐๑๒๓๔๕๖๗๘๙",
    "0123456789",
)
OCR_IDENTIFIER_TRANSLATION = str.maketrans(
    {
        "O": "0",
        "o": "0",
        "D": "0",
        "d": "0",
    }
)

_SUBRULE_IDENTIFIER = r"[0-9๐-๙]+\.[0-9๐-๙]+(?:\.[0-9๐-๙]+)*"
_EXPLICIT_IDENTIFIER = r"[0-9๐-๙OoDd]+(?:\.[0-9๐-๙OoDd]+)*"
_CHAPTER_RE = re.compile(
    rf"^\s*หมวด\s+[`'\"|:;,.]?\s*([0-9๐-๙OoDd]+)(?:\s+(.*?))?\s*$",
    re.IGNORECASE,
)
_CHAPTER_ONLY_RE = re.compile(r"^\s*หมวด\s*$", re.IGNORECASE)
_SPECIAL_HEADING_RE = re.compile(r"^\s*บทเฉพาะกาล(?:\s+(.*?))?\s*$")
_RULE_RE = re.compile(
    rf"^\s*ข้?อ\s+({_EXPLICIT_IDENTIFIER})(?:\s+(.*))?\s*$",
    re.IGNORECASE,
)
_RULE_PREFIX_RE = re.compile(r"^\s*ข้?อ\s*$", re.IGNORECASE)
_RULE_IDENTIFIER_LINE_RE = re.compile(
    rf"^\s*({_EXPLICIT_IDENTIFIER})(?:\s+(.*))?\s*$",
    re.IGNORECASE,
)
_SUBRULE_RE = re.compile(
    rf"^\s*(?:[-*•·]\s*)?({_SUBRULE_IDENTIFIER})(?:\s+(.*))?\s*$",
    re.IGNORECASE,
)
_REFERENCE_RE = re.compile(
    rf"ข้?อ\s*({_EXPLICIT_IDENTIFIER})",
    re.IGNORECASE,
)
_PAGE_NUMBER_RE = re.compile(r"^[0-9๐-๙]{1,3}$")
_SEPARATOR_RE = re.compile(r"^[.。…·•_\-–—\s]{3,}$")
_SIGNATURE_RE = re.compile(
    r"^(?:ประกาศ\s+ณ\s+วันที่|ลงชื่อ|ผู้รับสนองพระบรมราชโองการ)",
    re.IGNORECASE,
)
_PAGE_NAME_RE = re.compile(r"(?:page|หน้า)[_-]?(\d+)", re.IGNORECASE)


def normalize_identifier(identifier: str) -> str:
    """Normalize Thai digits and scoped OCR zero variants in an identifier."""
    return str(identifier).translate(THAI_DIGIT_TRANSLATION).translate(
        OCR_IDENTIFIER_TRANSLATION
    )


def _page_number_from_name(value: str | Path) -> Optional[int]:
    match = _PAGE_NAME_RE.search(Path(value).name)
    return int(match.group(1)) if match else None


def _coerce_page_number(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _json_text_lines(data: Any) -> Optional[list[str]]:
    if isinstance(data, list) and all(isinstance(item, str) for item in data):
        return list(data)
    if not isinstance(data, Mapping):
        return None

    for key in ("text_lines", "lines"):
        value = data.get(key)
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return list(value)

    for key in ("text", "ocr_text"):
        value = data.get(key)
        if isinstance(value, str):
            return value.splitlines()
    return None


@dataclass(frozen=True)
class RulePage:
    """OCR lines plus the source context needed for rule provenance."""

    lines: Sequence[str]
    source_filename: str
    source_page: Optional[int] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_file(cls, file_path: str | Path) -> "RulePage":
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Rule OCR file not found: {path}")

        metadata: Mapping[str, Any] = {}
        if path.suffix.casefold() == ".json":
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            lines = _json_text_lines(data)
            if lines is None:
                raise ValueError(f"Rule OCR JSON has no text lines: {path}")
            if isinstance(data, Mapping):
                metadata = dict(data)
        elif path.suffix.casefold() == ".txt":
            lines = path.read_text(encoding="utf-8").splitlines()
        else:
            raise ValueError(f"Unsupported Rule OCR file type: {path.suffix}")

        source_filename = metadata.get("source_filename")
        if not isinstance(source_filename, str) or not source_filename.strip():
            source_filename = path.name
        source_page = _coerce_page_number(metadata.get("source_page"))
        if source_page is None:
            source_page = _page_number_from_name(source_filename)
        if source_page is None:
            source_page = _page_number_from_name(path.name)

        return cls(
            lines=lines,
            source_filename=Path(source_filename).name,
            source_page=source_page,
            metadata=metadata,
        )

    def provenance(self) -> dict[str, Any]:
        entry = {
            "source_filename": self.source_filename,
            "source_page": self.source_page,
            "document_category": "rule",
        }
        program = self.metadata.get("program")
        if isinstance(program, str) and program.strip():
            entry["program"] = program
        return entry


def _is_ocr_json(file_path: Path) -> bool:
    try:
        with file_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    return _json_text_lines(data) is not None


def _file_page_sort_key(file_path: Path) -> tuple[int, int, str]:
    page = _page_number_from_name(file_path.name)
    if page is None:
        return (1, 0, file_path.name.casefold())
    return (0, page, file_path.name.casefold())


def discover_rule_ocr_files(input_path: str | Path) -> list[Path]:
    """Find Rule OCR files, preferring JSON over a same-stem TXT file."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Rule OCR input does not exist: {path}")

    if path.is_file():
        if path.suffix.casefold() not in {".txt", ".json"}:
            raise ValueError(f"Rule OCR input must be TXT or JSON: {path}")
        if path.suffix.casefold() == ".json" and not _is_ocr_json(path):
            raise ValueError(f"Rule OCR JSON has no text lines: {path}")
        return [path]

    candidates: dict[str, Path] = {}
    for candidate in path.iterdir():
        if not candidate.is_file():
            continue
        suffix = candidate.suffix.casefold()
        if suffix == ".txt":
            candidates.setdefault(candidate.stem.casefold(), candidate)
        elif suffix == ".json" and not candidate.name.endswith("_extracted.json"):
            if _is_ocr_json(candidate):
                candidates[candidate.stem.casefold()] = candidate

    if not candidates:
        raise FileNotFoundError(
            f"No Rule OCR TXT/JSON files found in: {path}"
        )

    return sorted(candidates.values(), key=_file_page_sort_key)


@dataclass
class _RuleBuilder:
    section_number: str
    category: Optional[str]
    text_lines: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    source_provenance: list[dict[str, Any]] = field(default_factory=list)

    @property
    def section_path(self) -> list[str]:
        parts = self.section_number.split(".")
        return [".".join(parts[:index]) for index in range(1, len(parts) + 1)]

    @property
    def parent_rule_id(self) -> Optional[str]:
        if "." not in self.section_number:
            return None
        return "rule:" + self.section_number.rsplit(".", 1)[0]

    def append(self, text: str, page: RulePage) -> None:
        text = text.strip()
        if not text:
            return
        self.text_lines.append(text)
        provenance = page.provenance()
        if provenance not in self.source_provenance:
            self.source_provenance.append(provenance)
        for reference in _REFERENCE_RE.findall(text):
            normalized = normalize_identifier(reference)
            if normalized not in self.references:
                self.references.append(normalized)

    def to_record(self) -> dict[str, Any]:
        return {
            "rule_id": "rule:" + self.section_number,
            "category": self.category,
            "section_path": self.section_path,
            "section_number": self.section_number,
            "parent_rule_id": self.parent_rule_id,
            "rule_text": "\n".join(self.text_lines).strip(),
            "references": list(self.references),
            "source_provenance": list(self.source_provenance),
        }


class RuleExtractor:
    """Extract numbered academic rules without using the course pipeline."""

    def __init__(self, source: str = "Academic Rules") -> None:
        self.source = source

    @staticmethod
    def _clean_line(value: Any) -> str:
        return str(value).strip() if value is not None else ""

    @staticmethod
    def _is_noise_line(line: str) -> bool:
        return bool(_PAGE_NUMBER_RE.fullmatch(line) or _SEPARATOR_RE.fullmatch(line))

    @staticmethod
    def _format_category(number: str, title: Optional[str] = None) -> str:
        category = f"หมวด {normalize_identifier(number)}"
        if title and title.strip():
            category += f" {title.strip()}"
        return category

    @staticmethod
    def _format_special_category(title: Optional[str] = None) -> str:
        category = "บทเฉพาะกาล"
        if title and title.strip():
            category += f" {title.strip()}"
        return category

    @staticmethod
    def _parse_rule_anchor(line: str) -> Optional[tuple[str, str]]:
        match = _RULE_RE.match(line)
        if match:
            return normalize_identifier(match.group(1)), (match.group(2) or "").strip()
        match = _SUBRULE_RE.match(line)
        if match:
            return normalize_identifier(match.group(1)), (match.group(2) or "").strip()
        return None

    @staticmethod
    def _parse_wrapped_rule_anchor(line: str) -> Optional[tuple[str, str]]:
        match = _RULE_IDENTIFIER_LINE_RE.match(line)
        if not match:
            return None
        return normalize_identifier(match.group(1)), (match.group(2) or "").strip()

    @staticmethod
    def _chapter_number(value: str) -> Optional[int]:
        normalized = normalize_identifier(value)
        return int(normalized) if normalized.isdigit() else None

    @staticmethod
    def _coerce_page(page: RulePage | Mapping[str, Any], index: int) -> RulePage:
        if isinstance(page, RulePage):
            return page
        if not isinstance(page, Mapping):
            raise TypeError("Rule pages must be RulePage instances or mappings")
        lines = page.get("lines", page.get("text_lines"))
        if not isinstance(lines, Sequence) or isinstance(lines, (str, bytes)):
            raise ValueError("Rule page mapping must contain a list of lines")
        source_filename = page.get("source_filename") or f"rule_page_{index + 1:03d}.txt"
        source_page = _coerce_page_number(page.get("source_page"))
        if source_page is None:
            source_page = _page_number_from_name(str(source_filename))
        return RulePage(
            lines=[str(line) for line in lines],
            source_filename=Path(str(source_filename)).name,
            source_page=source_page,
            metadata=page,
        )

    def extract_from_pages(
        self, pages: Iterable[RulePage | Mapping[str, Any]]
    ) -> dict[str, Any]:
        page_list = [self._coerce_page(page, index) for index, page in enumerate(pages)]
        page_list.sort(
            key=lambda page: (
                page.source_page is None,
                page.source_page if page.source_page is not None else 0,
                page.source_filename.casefold(),
            )
        )

        records: list[dict[str, Any]] = []
        current_category: Optional[str] = None
        pending_category_number: Optional[str] = None
        pending_rule_prefix: Optional[tuple[str, RulePage]] = None
        last_chapter_number: Optional[int] = None
        current_rule: Optional[_RuleBuilder] = None
        signature_started = False

        def close_rule() -> None:
            nonlocal current_rule
            if current_rule is not None:
                records.append(current_rule.to_record())
                current_rule = None

        for page in page_list:
            for raw_line in page.lines:
                line = self._clean_line(raw_line)
                if not line:
                    continue
                if signature_started:
                    continue
                if _SIGNATURE_RE.match(line):
                    signature_started = True
                    continue

                if pending_rule_prefix is not None:
                    wrapped_anchor = self._parse_wrapped_rule_anchor(line)
                    if wrapped_anchor:
                        if pending_category_number is not None:
                            current_category = self._format_category(
                                pending_category_number
                            )
                            pending_category_number = None
                        close_rule()
                        section_number, text = wrapped_anchor
                        current_rule = _RuleBuilder(
                            section_number=section_number,
                            category=current_category,
                        )
                        current_rule.append(text, page)
                        pending_rule_prefix = None
                        continue

                    if current_rule is not None:
                        prefix_line, prefix_page = pending_rule_prefix
                        current_rule.append(prefix_line, prefix_page)
                    pending_rule_prefix = None

                if self._is_noise_line(line):
                    if pending_category_number is not None:
                        current_category = self._format_category(
                            pending_category_number
                        )
                        pending_category_number = None
                    continue

                special_match = _SPECIAL_HEADING_RE.match(line)
                if special_match:
                    close_rule()
                    pending_category_number = None
                    current_category = self._format_special_category(
                        special_match.group(1)
                    )
                    continue

                chapter_match = _CHAPTER_RE.match(line)
                if chapter_match:
                    close_rule()
                    number = normalize_identifier(chapter_match.group(1))
                    title = chapter_match.group(2)
                    chapter_number = self._chapter_number(number)
                    if chapter_number is not None:
                        last_chapter_number = chapter_number
                    pending_category_number = number if not title else None
                    current_category = (
                        self._format_category(number, title)
                        if title
                        else None
                    )
                    continue

                if _CHAPTER_ONLY_RE.match(line):
                    close_rule()
                    # A numberless chapter heading advances the document's sequence.
                    inferred_number = (last_chapter_number or 0) + 1
                    last_chapter_number = inferred_number
                    pending_category_number = str(inferred_number)
                    current_category = None
                    continue

                if _RULE_PREFIX_RE.match(line):
                    pending_rule_prefix = (line, page)
                    continue

                anchor = self._parse_rule_anchor(line)
                if anchor:
                    if pending_category_number is not None:
                        current_category = self._format_category(
                            pending_category_number
                        )
                        pending_category_number = None
                    close_rule()
                    section_number, text = anchor
                    current_rule = _RuleBuilder(
                        section_number=section_number,
                        category=current_category,
                    )
                    current_rule.append(text, page)
                    continue

                if pending_category_number is not None:
                    current_category = self._format_category(
                        pending_category_number, line
                    )
                    pending_category_number = None
                    continue

                if current_rule is not None:
                    current_rule.append(line, page)

        close_rule()
        return {
            "source": self.source,
            "total_rules": len(records),
            "rules": records,
        }

    def extract_from_lines(
        self,
        lines: Sequence[str],
        source_filename: str = "rule_page_001.txt",
        source_page: Optional[int] = 1,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        page = RulePage(
            lines=list(lines),
            source_filename=Path(source_filename).name,
            source_page=source_page,
            metadata=metadata or {},
        )
        return self.extract_from_pages([page])

    def extract_from_files(self, file_paths: Sequence[str | Path]) -> dict[str, Any]:
        if not file_paths:
            raise FileNotFoundError("No Rule OCR TXT/JSON files were provided")
        pages = [RulePage.from_file(path) for path in file_paths]
        return self.extract_from_pages(pages)

    def extract(self, pages: Iterable[RulePage | Mapping[str, Any]]) -> dict[str, Any]:
        """Alias for callers that provide already-loaded OCR pages."""
        return self.extract_from_pages(pages)
