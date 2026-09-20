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
        "ด": "1",
        ":": ".",
    }
)

_NESTED_IDENTIFIER_COMPONENT = r"[0-9๐-๙OoDdด]+"
_NESTED_IDENTIFIER = (
    rf"{_NESTED_IDENTIFIER_COMPONENT}(?:[.:]{_NESTED_IDENTIFIER_COMPONENT})+"
)
_TRUNCATED_NESTED_IDENTIFIER = (
    rf"{_NESTED_IDENTIFIER_COMPONENT}(?:[.:]{_NESTED_IDENTIFIER_COMPONENT})*[.:]"
)
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
_NESTED_RULE_RE = re.compile(
    rf"^\s*(?:[-*•·]\s*)?({_NESTED_IDENTIFIER})(?:\s+(.*))?\s*$",
    re.IGNORECASE,
)
_TRUNCATED_NESTED_RULE_RE = re.compile(
    rf"^\s*(?:[-*•·]\s*)?({_TRUNCATED_NESTED_IDENTIFIER})(?:\s+(.*))?\s*$",
    re.IGNORECASE,
)
_BARE_IDENTIFIER_RE = re.compile(
    rf"^\s*({_NESTED_IDENTIFIER_COMPONENT})\s*$",
    re.IGNORECASE,
)
_REFERENCE_RE = re.compile(
    rf"ข้?อ\s*({_EXPLICIT_IDENTIFIER})",
    re.IGNORECASE,
)
_PAGE_NUMBER_RE = re.compile(r"^[0-9๐-๙]{1,3}$")
_SEPARATOR_RE = re.compile(r"^[.。…·•_\-–—\s]{3,}$")
_REFERENCE_PREFIX_RE = re.compile(
    r"(?:^|(?:ตาม|และ|หรือ|ถึง|ตามที่)\s*)ข้อ\s*$",
    re.IGNORECASE,
)
_SIGNATURE_RE = re.compile(
    r"^(?:ประกาศ\s+ณ\s+วันที่|ลงชื่อ|ผู้รับสนองพระบรมราชโองการ)",
    re.IGNORECASE,
)
_SIGNATURE_START_RE = re.compile(r"^\s*ประกาศ\s*$", re.IGNORECASE)
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
            self.add_reference(reference)

    def add_reference(self, reference: str) -> None:
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
    def _is_confident_page_number(
        line: str, page: RulePage, line_index: int, line_count: int
    ) -> bool:
        if not _PAGE_NUMBER_RE.fullmatch(line) or page.source_page is None:
            return False
        normalized = normalize_identifier(line)
        if not normalized.isdigit() or int(normalized) != page.source_page:
            return False
        return line_index == 0 or line_index >= max(0, line_count - 3)

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
    def _parse_explicit_rule_anchor(line: str) -> Optional[tuple[str, str]]:
        match = _RULE_RE.match(line)
        if match:
            return normalize_identifier(match.group(1)), (match.group(2) or "").strip()

        return None

    @staticmethod
    def _parse_nested_rule_candidate(line: str) -> Optional[tuple[str, str]]:
        match = _NESTED_RULE_RE.match(line)
        if match:
            return normalize_identifier(match.group(1)), (match.group(2) or "").strip()
        return None

    @classmethod
    def _parse_reference_candidate(cls, line: str) -> Optional[tuple[str, str]]:
        explicit = cls._parse_explicit_rule_anchor(line)
        if explicit:
            return explicit
        nested = cls._parse_nested_rule_candidate(line)
        if nested:
            return nested
        bare = cls._parse_bare_identifier(line)
        return (bare, "") if bare else None

    @staticmethod
    def _parse_truncated_rule_candidate(line: str) -> Optional[tuple[str, str]]:
        match = _TRUNCATED_NESTED_RULE_RE.match(line)
        if match:
            return normalize_identifier(match.group(1)), (match.group(2) or "").strip()
        return None

    @staticmethod
    def _parse_bare_identifier(line: str) -> Optional[str]:
        match = _BARE_IDENTIFIER_RE.match(line)
        return normalize_identifier(match.group(1)) if match else None

    @classmethod
    def _parse_rule_anchor(cls, line: str) -> Optional[tuple[str, str]]:
        explicit = cls._parse_explicit_rule_anchor(line)
        if explicit:
            return explicit
        return cls._parse_nested_rule_candidate(line)

    @staticmethod
    def _parse_wrapped_rule_anchor(line: str) -> Optional[tuple[str, str]]:
        match = _RULE_IDENTIFIER_LINE_RE.match(line)
        if not match:
            return None
        return normalize_identifier(match.group(1)), (match.group(2) or "").strip()

    @staticmethod
    def _identifier_parts(value: str) -> Optional[tuple[int, ...]]:
        normalized = normalize_identifier(value)
        parts = normalized.rstrip(".").split(".")
        if not parts or any(not part.isdigit() for part in parts):
            return None
        return tuple(int(part) for part in parts)

    @classmethod
    def _implicit_anchor_compatible(
        cls, candidate: str, current: Optional[str]
    ) -> bool:
        candidate_parts = cls._identifier_parts(candidate)
        current_parts = cls._identifier_parts(current) if current else None
        if not candidate_parts or not current_parts or len(candidate_parts) < 2:
            return False
        if candidate_parts[0] != current_parts[0]:
            return False

        candidate_parent = candidate_parts[:-1]
        current_parent = current_parts[:-1]
        return (
            candidate_parent == current_parts
            or candidate_parent == current_parent
            or current_parts[: len(candidate_parent)] == candidate_parent
            or candidate_parent[: len(current_parts)] == current_parts
        )

    @staticmethod
    def _line_ends_reference_prefix(line: str) -> bool:
        stripped = line.rstrip()
        return bool(_REFERENCE_PREFIX_RE.search(stripped)) or stripped.endswith(
            ("ข้อ", "ขอ")
        )

    @classmethod
    def _infer_truncated_identifier(
        cls,
        kind: str,
        pending_identifier: str,
        current_identifier: Optional[str],
        next_identifier: str,
    ) -> Optional[str]:
        current_parts = cls._identifier_parts(current_identifier or "")
        pending_parts = cls._identifier_parts(pending_identifier)
        next_parts = cls._identifier_parts(next_identifier)
        if not current_parts or not pending_parts or not next_parts:
            return None

        if kind == "partial":
            if (
                len(current_parts) == len(pending_parts) + 1
                and len(next_parts) == len(pending_parts) + 1
                and current_parts[:-1] == pending_parts
                and next_parts[:-1] == pending_parts
                and next_parts[-1] == current_parts[-1] + 2
            ):
                return ".".join(
                    str(part) for part in (*pending_parts, current_parts[-1] + 1)
                )

        if kind == "bare":
            if (
                pending_parts == current_parts
                and next_parts[:-1] == current_parts
                and next_parts[-1] == 2
            ):
                return ".".join(str(part) for part in (*current_parts, 1))

        return None

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
        pending_truncated_anchor: Optional[dict[str, Any]] = None
        reference_prefix_pending = False
        last_chapter_number: Optional[int] = None
        current_rule: Optional[_RuleBuilder] = None
        pending_signature_lines: Optional[list[tuple[str, RulePage]]] = None
        signature_started = False

        def close_rule() -> None:
            nonlocal current_rule
            if current_rule is not None:
                records.append(current_rule.to_record())
                current_rule = None

        def flush_truncated_anchor() -> None:
            nonlocal pending_truncated_anchor
            if pending_truncated_anchor is None:
                return
            if current_rule is not None:
                current_rule.append(
                    pending_truncated_anchor["raw_line"],
                    pending_truncated_anchor["page"],
                )
                for buffered_line, buffered_page in pending_truncated_anchor[
                    "buffer"
                ]:
                    current_rule.append(buffered_line, buffered_page)
            pending_truncated_anchor = None

        def recover_truncated_anchor(next_identifier: str) -> bool:
            nonlocal current_rule, pending_truncated_anchor
            if pending_truncated_anchor is None or current_rule is None:
                return False
            inferred_identifier = self._infer_truncated_identifier(
                pending_truncated_anchor["kind"],
                pending_truncated_anchor["identifier"],
                current_rule.section_number,
                next_identifier,
            )
            if inferred_identifier is None:
                return False

            close_rule()
            current_rule = _RuleBuilder(
                section_number=inferred_identifier,
                category=current_category,
            )
            anchor_text = pending_truncated_anchor["text"]
            if anchor_text:
                current_rule.append(anchor_text, pending_truncated_anchor["page"])
            for buffered_line, buffered_page in pending_truncated_anchor["buffer"]:
                current_rule.append(buffered_line, buffered_page)
            pending_truncated_anchor = None
            return True

        def flush_pending_signature() -> None:
            nonlocal pending_signature_lines
            if pending_signature_lines is None:
                return
            if current_rule is not None:
                for signature_line, signature_page in pending_signature_lines:
                    current_rule.append(signature_line, signature_page)
            pending_signature_lines = None

        for page in page_list:
            for line_index, raw_line in enumerate(page.lines):
                line = self._clean_line(raw_line)
                if not line:
                    continue
                if signature_started:
                    continue
                if _SIGNATURE_RE.match(line):
                    flush_truncated_anchor()
                    pending_signature_lines = None
                    signature_started = True
                    continue

                if pending_signature_lines is not None:
                    expected_line = "ณ" if len(pending_signature_lines) == 1 else "วันที่"
                    if line == expected_line:
                        pending_signature_lines.append((line, page))
                        if expected_line == "วันที่":
                            pending_signature_lines = None
                            signature_started = True
                        continue
                    flush_pending_signature()

                if _SIGNATURE_START_RE.match(line):
                    flush_truncated_anchor()
                    pending_signature_lines = [(line, page)]
                    continue

                if pending_truncated_anchor is not None:
                    nested_candidate = self._parse_nested_rule_candidate(line)
                    if nested_candidate and recover_truncated_anchor(
                        nested_candidate[0]
                    ):
                        pass
                    elif nested_candidate:
                        if self._implicit_anchor_compatible(
                            nested_candidate[0],
                            current_rule.section_number
                            if current_rule is not None
                            else None,
                        ):
                            flush_truncated_anchor()
                        else:
                            pending_truncated_anchor["buffer"].append((line, page))
                            continue
                    elif (
                        self._parse_explicit_rule_anchor(line)
                        or _RULE_PREFIX_RE.match(line)
                        or _SPECIAL_HEADING_RE.match(line)
                        or _CHAPTER_RE.match(line)
                        or _CHAPTER_ONLY_RE.match(line)
                    ):
                        flush_truncated_anchor()
                    else:
                        pending_truncated_anchor["buffer"].append((line, page))
                        continue

                if reference_prefix_pending and current_rule is not None:
                    if _RULE_PREFIX_RE.match(line):
                        current_rule.append(line, page)
                        reference_prefix_pending = True
                        continue
                    reference = None
                    if not self._parse_explicit_rule_anchor(line):
                        reference = self._parse_reference_candidate(line)
                    if reference:
                        current_rule.append(line, page)
                        current_rule.add_reference(reference[0])
                        reference_prefix_pending = (
                            self._line_ends_reference_prefix(line)
                            or not reference[1]
                        )
                        continue
                    reference_prefix_pending = False

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
                        reference_prefix_pending = False
                        continue

                    if current_rule is not None:
                        prefix_line, prefix_page = pending_rule_prefix
                        current_rule.append(prefix_line, prefix_page)
                    pending_rule_prefix = None

                special_match = _SPECIAL_HEADING_RE.match(line)
                if special_match:
                    close_rule()
                    pending_category_number = None
                    current_category = self._format_special_category(
                        special_match.group(1)
                    )
                    reference_prefix_pending = False
                    continue

                chapter_match = _CHAPTER_RE.match(line)
                if chapter_match:
                    close_rule()
                    number = normalize_identifier(chapter_match.group(1))
                    title = chapter_match.group(2)
                    chapter_parts = self._identifier_parts(number)
                    if chapter_parts and len(chapter_parts) == 1:
                        last_chapter_number = chapter_parts[0]
                    pending_category_number = number if not title else None
                    current_category = (
                        self._format_category(number, title)
                        if title
                        else None
                    )
                    reference_prefix_pending = False
                    continue

                if _CHAPTER_ONLY_RE.match(line):
                    close_rule()
                    # A numberless chapter heading advances the document's sequence.
                    inferred_number = (last_chapter_number or 0) + 1
                    last_chapter_number = inferred_number
                    pending_category_number = str(inferred_number)
                    current_category = None
                    reference_prefix_pending = False
                    continue

                if _RULE_PREFIX_RE.match(line):
                    pending_rule_prefix = (line, page)
                    continue

                explicit_anchor = self._parse_explicit_rule_anchor(line)
                if explicit_anchor:
                    if pending_category_number is not None:
                        current_category = self._format_category(
                            pending_category_number
                        )
                        pending_category_number = None
                    close_rule()
                    section_number, text = explicit_anchor
                    current_rule = _RuleBuilder(
                        section_number=section_number,
                        category=current_category,
                    )
                    current_rule.append(text, page)
                    reference_prefix_pending = False
                    continue

                nested_anchor = self._parse_nested_rule_candidate(line)
                if nested_anchor and self._implicit_anchor_compatible(
                    nested_anchor[0],
                    current_rule.section_number if current_rule is not None else None,
                ):
                    if pending_category_number is not None:
                        current_category = self._format_category(
                            pending_category_number
                        )
                        pending_category_number = None
                    close_rule()
                    section_number, text = nested_anchor
                    current_rule = _RuleBuilder(
                        section_number=section_number,
                        category=current_category,
                    )
                    current_rule.append(text, page)
                    reference_prefix_pending = False
                    continue

                truncated_anchor = self._parse_truncated_rule_candidate(line)
                if truncated_anchor and current_rule is not None:
                    pending_parts = self._identifier_parts(truncated_anchor[0])
                    current_parts = self._identifier_parts(current_rule.section_number)
                    if (
                        pending_parts
                        and current_parts
                        and len(current_parts) == len(pending_parts) + 1
                        and current_parts[:-1] == pending_parts
                    ):
                        pending_truncated_anchor = {
                            "kind": "partial",
                            "identifier": truncated_anchor[0],
                            "raw_line": line,
                            "text": truncated_anchor[1],
                            "page": page,
                            "buffer": [],
                        }
                        continue

                bare_identifier = self._parse_bare_identifier(line)
                if (
                    bare_identifier
                    and current_rule is not None
                    and bare_identifier == current_rule.section_number
                    and len(self._identifier_parts(bare_identifier) or ()) == 1
                ):
                    pending_truncated_anchor = {
                        "kind": "bare",
                        "identifier": bare_identifier,
                        "raw_line": line,
                        "text": "",
                        "page": page,
                        "buffer": [],
                    }
                    continue

                if _SEPARATOR_RE.fullmatch(line):
                    if pending_category_number is not None:
                        current_category = self._format_category(
                            pending_category_number
                        )
                        pending_category_number = None
                    continue

                if _PAGE_NUMBER_RE.fullmatch(line):
                    if self._is_confident_page_number(
                        line, page, line_index, len(page.lines)
                    ):
                        if pending_category_number is not None:
                            current_category = self._format_category(
                                pending_category_number
                            )
                            pending_category_number = None
                    elif current_rule is not None:
                        current_rule.append(line, page)
                    elif pending_category_number is not None:
                        current_category = self._format_category(
                            pending_category_number, line
                        )
                        pending_category_number = None
                    continue

                if pending_category_number is not None:
                    current_category = self._format_category(
                        pending_category_number, line
                    )
                    pending_category_number = None
                    continue

                if current_rule is not None:
                    current_rule.append(line, page)
                    reference_prefix_pending = self._line_ends_reference_prefix(line)

        flush_pending_signature()
        flush_truncated_anchor()
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
