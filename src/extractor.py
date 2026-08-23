"""
Block-Based curriculum extractor.

Academic-curriculum OCR text (the "study plan" tables) is parsed through a
clean 3-step pipeline instead of one monolithic loop:

    Step 1  split_into_blocks  : group raw OCR lines into course blocks,
                                 tracking the surrounding year / semester /
                                 category context for each block.
    Step 2  parse_single_block : turn one block into a structured course dict,
                                 cleaning common OCR noise along the way.
    Step 3  post_process       : high-level combinations / clean-ups applied
                                 to the full extracted course list.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from .pre_clean import pre_clean_with_regex


SOURCE_PROVENANCE_KEY = "source_provenance"
SOURCE_PROVENANCE_FIELDS = (
    "program",
    "source_filename",
    "source_page",
    "document_category",
)
PAGE_RE = re.compile(r"page_(\d+)", re.IGNORECASE)


def merge_source_provenance(*records) -> List[Dict]:
    """Combine source entries in order without duplicating an occurrence."""
    merged: List[Dict] = []
    seen = set()

    for record in records:
        if isinstance(record, dict):
            entries = record.get(SOURCE_PROVENANCE_KEY, [])
        elif isinstance(record, list):
            entries = record
        else:
            continue

        if isinstance(entries, dict):
            entries = [entries]
        if not isinstance(entries, list):
            continue

        for entry in entries:
            if not isinstance(entry, dict):
                continue
            normalized = {
                field: entry.get(field) for field in SOURCE_PROVENANCE_FIELDS
            }
            identity = tuple(normalized[field] for field in SOURCE_PROVENANCE_FIELDS)
            if identity in seen:
                continue
            seen.add(identity)
            merged.append(normalized)

    return merged


def _page_number(value) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _page_number_from_name(value: str | None) -> Optional[int]:
    if not value:
        return None
    match = PAGE_RE.search(value)
    return int(match.group(1)) if match else None


def clean_ocr_en_text(text: str) -> str:
    """Fix common OCR typos in English text."""
    if not text:
        return text
    text = re.sub(r"\bEDUCATIOM\b", "EDUCATION", text, flags=re.IGNORECASE)
    text = re.sub(r"\bfoundatlon\b", "foundation", text, flags=re.IGNORECASE)
    text = re.sub(r"^l\s+", "1 ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+l$", " 1", text, flags=re.IGNORECASE)
    text = re.sub(r"^\bL\b$", "1", text, flags=re.IGNORECASE)
    text = re.sub(r"\bL\b", "1", text, flags=re.IGNORECASE)
    return text.strip()


def normalize_course_code(code: str) -> str:
    """
    Normalize a course code.

    Codes containing letters (OCR placeholders like '06026XX' or 'XWX') are
    padded with trailing 'x' so that every code is exactly 8 characters.
    """
    code = code.strip()
    code_lower = code.lower()

    if re.search(r"[a-z]", code_lower):
        m = re.match(r"^(\d+)", code)
        if m:
            prefix = m.group(1)
            return prefix + ("x" * (8 - len(prefix)))
        return "xxxxxxxx"

    return code


def _is_valid_numeric_course_code(code: str) -> bool:
    """Accept fully numeric course codes only when they contain eight digits."""
    return not code.isdigit() or len(code) == 8


CODE_ONLY_LINE_REGEX = re.compile(r"^[0-9xX\)\.\|_]{4,12}$", re.IGNORECASE)
def try_clean_code_line(line: str) -> Union[str, None]:
    """
    Check if a line is 'likely' a course code that OCR misread with junk chars.
    E.g. '06026X)X', '9064XX)X' -> return cleaned code (junk chars removed).
    Return None if not a course code.
    """
    stripped = line.strip()
    if not stripped or not CODE_ONLY_LINE_REGEX.match(stripped):
        return None

    # Remove junk chars that are not digits/x (e.g. ) . | _)
    cleaned = re.sub(r"[^\dxX]", "", stripped, flags=re.IGNORECASE)

    # Must keep a reasonable length (real course codes are 5-9 chars before normalize)
    # and must contain at least one digit (avoid treating a lone 'X' or 'L' as a code)
    if (
        5 <= len(cleaned) <= 9
        and re.search(r"\d", cleaned)
        and _is_valid_numeric_course_code(cleaned)
    ):
        return cleaned
    return None


@dataclass
class CourseBlock:
    """A single course group collected from the OCR lines of a study-plan table."""
    code: str = ""                                       # normalized course code
    lines: List[str] = field(default_factory=list)       # content lines (after the code)
    year: int = 1
    semester: int = 1
    category: str = "หมวดวิชาเฉพาะ"
    type: str = "บังคับ"


class CurriculumExtractor:

    # ------------------------------------------------------------------ #
    #  OCR patterns & regexes used by the study-plan pipeline             #
    # ------------------------------------------------------------------ #
    HAS_THAI_RE = re.compile(r"[\u0e00-\u0e7f]")
    HAS_ENG_RE = re.compile(r"[a-zA-Z]")

    # A standard course-code line: 8 digits, a padded placeholder (e.g. 06026XX),
    # or a run of X/O/W placeholder letters.
    COURSE_CODE_RE = re.compile(
        r"(?:^|\s)(\b[0-9]{8}\b|\b[0-9xX]{5,9}\b|\b\d{5}[a-zA-Z]{3}\b|^[xX]+$|^[xXoOwW]{3,8}$)(?:\s|$)"
    )
    # Elective placeholder codes that OCR mangled into letters, e.g. XNWWX / XOWX / XWX.
    # These are normalized to the generic elective code "xxxxxxxx".
    JUNK_PLACEHOLDER_RE = re.compile(r"^[xXwWoOnN]{3,6}$", re.IGNORECASE)

    # Credits: "3 (3-0-6)" or an alternative "3 (3-0-6) หรือ 3 (2-2-5)".
    # Placeholder credits use X wildcards, e.g. "3 (X-X-X)", "3 (X-X X)",
    # "3 (X X X)" (separators are OCR-noisy dashes/spaces).
    CREDIT_GROUP_RE = r"(?:\d+\s*)?\([0-9xX]+[ -][0-9xX]+[ -][0-9xX]+\)"
    CREDITS_RE = re.compile(
        rf"{CREDIT_GROUP_RE}(?:\s*(?:หรือ|or|/)\s*{CREDIT_GROUP_RE})?",
        re.IGNORECASE,
    )
    SINGLE_CREDIT_RE = re.compile(CREDIT_GROUP_RE)

    CATEGORY_HEADER_RE = re.compile(r"^\s*(?:\d+\.\s*)?(?:หมวดวิชา|กลุ่มวิชา)", re.IGNORECASE)
    IT_SECTION_HEADER_RE = re.compile(
        r"^\s*(?:\d+\.\s*)?(?:หมวด|กลุ่ม)\s*วิชา(?!ที่กำหนด)",
        re.IGNORECASE,
    )
    GROUP_LABEL_RE = re.compile(
        r"^\s*(?:กลุ่ม|หมวด)\s*วิชา.*กำหนด\s*โดย\s*คณะ\s*\*?\s*$",
        re.IGNORECASE,
    )
    OR_KEYWORD_RE = re.compile(r"^\s*(?:หรือ|หรอ|or|/)\s*$", re.IGNORECASE)

    # Year / semester headers, with or without the number on the same line.
    YEAR_HEADER_RE = re.compile(r"(?:ชั้น)?[ปขชบ]ี\s*ที่?")                    # "ปีที่"
    SEM_HEADER_RE = re.compile(r"(?:ภาค|เทอม)\s*(?:การศึกษา|เรียน)?\s*ที่?")   # "ภาคการศึกษาที่"
    YEAR_VALUE_RE = re.compile(r"(?:ชั้น)?[ปขชบ]ี\s*ที่?\s*(\d+)")
    SEM_VALUE_RE = re.compile(r"(?:ภาค|เทอม)\s*(?:การศึกษา|เรียน)?\s*ที่?\s*(\d+)")
    # Whole-line match for a pure meta header: "ปีที่ 4", "ภาคการศึกษาที่ 1",
    # "ปีที่", or "ปีที่ 2 ภาคการศึกษาที่" (number spilled onto the next line).
    META_LINE_RE = re.compile(
        r"^\s*"
        r"(?:(?:ชั้น)?[ปขชบ]ี\s*ที่?\s*\d*|(?:ภาค|เทอม)\s*(?:การศึกษา|เรียน)?\s*ที่?\s*\d*)"
        r"(?:\s+(?:(?:ชั้น)?[ปขชบ]ี\s*ที่?\s*\d*|(?:ภาค|เทอม)\s*(?:การศึกษา|เรียน)?\s*ที่?\s*\d*))*"
        r"\s*$"
    )

    PREREQ_KEYWORD_RE = re.compile(
        r"(?:วิชาบังคับก่อน|บังคับก่อน|ความรู้พื้นฐาน|prerequisite|pre-requisite|PRERE\s*[A-Z]*|PRERECUISITE)",
        re.IGNORECASE,
    )
    # Course-description paragraph openers (Thai & English). A general
    # description (e.g. "วิชานี้จะศึกษา...", "ศึกษาเกี่ยวกับ...", "Study of...")
    # only ever appears AFTER a block's metadata (code, Thai name, credits,
    # English name, prerequisite). Hitting one ends the block.
    DESCRIPTION_START_RE = re.compile(
        r"(?:วิชานี้|วิชานี|จะศึกษา|ศึกษาเกี่ยวกับ|ศึกษาถึง|เน้นการ|มุ่งเน้น|โดยเน้น|"
        r"THIS COURSE|COURSE WILL|COURSE DESCRIPTION|STUDY OF)",
        re.IGNORECASE,
    )
    DESCRIPTION_CODE_LINE_RE = re.compile(r"^\s*\d{8}\s*$")
    DESCRIPTION_PAGE_HEADER_RE = re.compile(
        r"^\s*(?:\d{1,4}|มคอ\.?\s*\d*|รายละเอียดหลักสูตร|"
        r"คำอธิบายรายวิชา(?:เฉพาะ)?|หน่วยกิต)\s*$",
        re.IGNORECASE,
    )
    DESCRIPTION_SECTION_BOUNDARY_RE = re.compile(
        r"^\s*(?:หมวด|กลุ่ม)\s*\d*\s*[| ]*วิชา|"
        r"^\s*(?:คำอธิบายรายวิชา(?:เฉพาะ)?|รายละเอียดหลักสูตร)\s*$",
        re.IGNORECASE,
    )
    DESCRIPTION_FOOTER_RE = re.compile(
        r"(?:วท\.?\s*\.?บ\.?|^\s*วท\.?\s*$|^\s*\.?บ\.?\s*\(|"
        r"คณะเทคโนโลยีสารสนเทศ|สาขาวิชา)",
        re.IGNORECASE,
    )
    NOTE_RE = re.compile(
        r"^\s*[-*]|(?:ประเมิน|เกณฑ์|ผลการเรียน|ผ่าน\s*\(S\)|\(S\)|\(U\)|ให้นักศึกษา)",
        re.IGNORECASE,
    )

    def __init__(
        self,
        program: str = "DSBA",
        plan: Optional[str] = "coop",
        source: str = "GT_Template-2.xlsx / Academic Plan GT — DSBA coop",
        coop_pairs: Optional[List[Tuple[str, str, str]]] = None,
    ):
        self.program = program
        self.plan = plan
        # Co-op / alternative course pairs to merge.  Each tuple is
        # (code_a, code_b, merged_credits).  Universal default keeps the
        # known pairs; callers may override for their own curriculum.
        self.coop_pairs = coop_pairs if coop_pairs is not None else [
            ("06026259", "06026260", "6(0-35-0)"),
            ("06046443", "06046444", "6(0-45-0)"),
        ]
        if program == "DSBA" and plan == "coop":
            self.source = "GT_Template-2.xlsx / Academic Plan GT — DSBA coop"
        elif program == "DSBA" and plan == "no_coop":
            self.source = "GT_Template-2.xlsx / Academic Plan GT — DSBA N0 coop"
        elif program == "BIT" and plan == "coop":
            self.source = "GT_Template-2.xlsx / Academic Plan GT — BIT coop"
        elif program == "BIT" and plan == "no_coop":
            self.source = "GT_Template-2.xlsx / Academic Plan GT — BIT no coop"
        elif program == "IT" and plan == "coop":
            self.source = "GT_Template-2.xlsx / Academic Plan GT — IT coop"
        elif program == "IT" and plan == "no_coop":
            self.source = "GT_Template-2.xlsx / Academic Plan GT — IT no coop"
        elif program == "AIT":
            self.source = "GT_Template-2.xlsx / Academic Plan GT — AIT"
        elif program == "GENED":
            self.source = "GT_Template-2.xlsx / General Education"
        else:
            self.source = source

    def _source_context(
        self,
        input_path: Path | None = None,
        metadata: Optional[dict] = None,
        document_category: str = "unknown",
    ) -> Dict:
        metadata = metadata if isinstance(metadata, dict) else {}

        source_filename = metadata.get("source_filename")
        if not isinstance(source_filename, str) or not source_filename.strip():
            source_filename = input_path.name if input_path is not None else None
        else:
            source_filename = Path(source_filename).name

        source_page = _page_number(metadata.get("source_page"))
        if source_page is None:
            source_page = _page_number_from_name(source_filename)
        if source_page is None and input_path is not None:
            source_page = _page_number_from_name(input_path.name)

        program = metadata.get("program")
        if not isinstance(program, str) or not program.strip():
            program = self.program if isinstance(self.program, str) and self.program.strip() else None

        return {
            "program": program,
            "source_filename": source_filename,
            "source_page": source_page,
            "document_category": document_category
            if document_category in {"plan", "description"}
            else "unknown",
        }

    @staticmethod
    def _attach_source_provenance(course: Dict, source_context: Dict) -> Dict:
        result = dict(course)
        result[SOURCE_PROVENANCE_KEY] = [dict(source_context)]
        return result

    # ------------------------------------------------------------------ #
    #  Step 1: split raw OCR lines into course blocks                     #
    # ------------------------------------------------------------------ #
    def split_into_blocks(self, lines: List[str]) -> List[CourseBlock]:
        """
        Group raw OCR lines into individual course blocks.

        Walks the lines once, keeping track of the surrounding year, semester,
        category and course type. A block starts at a course-code line and is
        closed whenever a new term, category, or noise line appears.
        """
        # Context carried between blocks (fresh per call).
        self._ctx_year = 1
        self._ctx_semester = 1
        self._ctx_category = "หมวดวิชาเฉพาะ"
        self._ctx_type = "บังคับ"

        blocks: List[CourseBlock] = []
        current: Optional[CourseBlock] = None

        idx = 0
        n = len(lines)
        while idx < n:
            line = lines[idx].strip()
            if not line:
                idx += 1
                continue

            # 1) Year / semester header -> a new term starts, close any open block.
            if self.META_LINE_RE.match(line):
                current = None
                idx += self._apply_meta_context(line, lines, idx)
                continue

            # 2) Course-code line -> begin a new block.
            code, remainder = self._extract_code(line)
            if code is not None:
                # Drop lone OCR junk tokens that trail a fully numeric code
                # (e.g. a stray 'X' after "06026200").
                if (
                    line.upper() in {"X", "^", "D9", "L"}
                    and blocks
                    and len(blocks[-1].code) == 8
                    and blocks[-1].code.isdigit()
                ):
                    idx += 1
                    continue

                current = CourseBlock(
                    code=code,
                    year=self._ctx_year,
                    semester=self._ctx_semester,
                    category=self._ctx_category,
                    type=self._ctx_type,
                )
                # The name / credits may share the code's line (rare) -> keep the tail.
                if remainder:
                    current.lines.append(remainder)
                blocks.append(current)
                idx += 1
                continue

            # 3) Table / total / page-number noise -> close any open block.
            if self._is_noise_line(line):
                current = None
                idx += 1
                continue

            # 4) Category header -> update context, close any open block.
            is_category_header = self.CATEGORY_HEADER_RE.search(line) is not None
            is_it_section_header = (
                self.program == "IT" and self.IT_SECTION_HEADER_RE.search(line) is not None
            )
            if (is_category_header or is_it_section_header) and not self.CREDITS_RE.search(line):
                self._ctx_category = line
                self._ctx_type = "เลือก" if "เลือก" in line else "บังคับ"
                current = None
                idx += 1
                continue

            # IT plan OCR sometimes emits a section-heading continuation as a
            # Thai-only line after a complete course row. It is not part of the
            # preceding title; discard it until the next course code.
            if (
                self.program == "IT"
                and current is not None
                and self._has_completed_course_metadata(current)
                and self.HAS_THAI_RE.search(line)
                and not self.HAS_ENG_RE.search(line)
                and not self.PREREQ_KEYWORD_RE.search(line)
                and not self.GROUP_LABEL_RE.match(line)
            ):
                current = None
                idx += 1
                continue

            # 5) Anything else belongs to the current block (or is ignored).
            if current is not None:
                current.lines.append(line)
            idx += 1

        return blocks

    def _apply_meta_context(self, line: str, lines: List[str], idx: int) -> int:
        """Apply a year/semester header to the context; return lines consumed."""
        yv = self.YEAR_VALUE_RE.search(line)
        sv = self.SEM_VALUE_RE.search(line)
        if yv:
            self._ctx_year = int(yv.group(1))
        if sv:
            self._ctx_semester = int(sv.group(1))

        # The number may have spilled onto the next line (e.g. "ปีที่" / "1").
        needs_year = yv is None and self.YEAR_HEADER_RE.search(line) is not None
        needs_sem = sv is None and self.SEM_HEADER_RE.search(line) is not None
        if needs_year or needs_sem:
            j = idx + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and re.fullmatch(r"\d{1,2}", lines[j].strip()):
                value = int(lines[j].strip())
                if needs_year:
                    self._ctx_year = value
                elif needs_sem:
                    self._ctx_semester = value
                return j - idx + 1
        return 1

    def _is_noise_line(self, line: str) -> bool:
        """Table headers, totals, page numbers and note lines are never course content."""
        if line.startswith("รวม"):
            return True
        if line.startswith("มคอ."):
            return True
        if (
            line.startswith("หน่วยกิต")
            or line.startswith("รหัสวิชา")
            or line.startswith("ชื่อวิชา")
        ):
            return True
        if line.startswith("(บรรยาย"):
            return True
        # Page numbers / credit totals (course-number suffixes are single digits).
        if re.fullmatch(r"\d{2,}", line):
            return True
        if self.NOTE_RE.search(line):
            return True
        return False

    def _has_completed_course_metadata(self, block: CourseBlock) -> bool:
        """Return whether a block already contains credits followed by English text."""
        credit_seen = False
        for line in block.lines:
            if self.SINGLE_CREDIT_RE.search(line):
                credit_seen = True
                continue
            if credit_seen and self.HAS_ENG_RE.search(line):
                return True
        return False

    def _extract_code(self, line: str) -> Tuple[Optional[str], str]:
        """
        Return (normalized_code, remainder_after_code) if `line` holds a course
        code, otherwise (None, "").
        """
        # OCR-mangled elective placeholder code (XNWWX / XOWX / XWX).
        if self.JUNK_PLACEHOLDER_RE.match(line):
            return "xxxxxxxx", ""

        # A clean code line (possibly containing junk chars OCR added).
        cleaned = try_clean_code_line(line)
        if cleaned:
            return normalize_course_code(cleaned), ""

        # A code embedded in a line (the tail, if any, is course content).
        m = self.COURSE_CODE_RE.search(line)
        if m:
            candidate = m.group(1)
            if not _is_valid_numeric_course_code(candidate):
                return None, ""
            return normalize_course_code(candidate), line[m.end(1):].strip()

        return None, ""

    # ------------------------------------------------------------------ #
    #  Step 2: turn one block into a structured course                    #
    # ------------------------------------------------------------------ #
    def parse_single_block(self, block: CourseBlock) -> Dict:
        """
        Process a single course block (3-6 OCR lines) into a structured course.

        Extracts code, Thai/English name, credits, type and prerequisite while
        handling common OCR noise:
          - alternative credit rows joined with "หรือ"/"or",
          - course-number suffixes on their own line ("CALCULUS" + "1"),
          - OCR junk codes like "XNWWX" (already normalized by Step 1),
          - stray "|" pipes and trailing hyphens from table borders.
        """
        code = block.code
        name_th = ""
        name_en = ""
        credits = ""
        prerequisite = "ไม่มี"

        # A few table rows place a group label before the credit and the real
        # Thai course title after it.  Only discard a label when that structure
        # is explicit; otherwise preserve the normal line-based behavior.
        group_label_index = None
        group_label = None
        first_credit_index = None
        for line_index, line in enumerate(block.lines):
            if self.SINGLE_CREDIT_RE.search(line) or self.OR_KEYWORD_RE.search(line):
                first_credit_index = line_index
                break
            if group_label_index is None and self.GROUP_LABEL_RE.match(line):
                group_label_index = line_index
                group_label = line.strip()

        drop_group_label = False
        if group_label_index is not None and first_credit_index is not None:
            for line in block.lines[first_credit_index + 1 :]:
                if self.HAS_ENG_RE.search(line):
                    break
                if self.SINGLE_CREDIT_RE.search(line) or self.OR_KEYWORD_RE.search(line):
                    continue
                if self.HAS_THAI_RE.search(line):
                    drop_group_label = True
                    break

        note = None
        if drop_group_label and group_label:
            note = re.sub(r"^\s*(กลุ่ม|หมวด)\s+วิชา", r"\1วิชา", group_label)
            note = re.sub(r"\s*\*\s*$", "", note).strip()

        for line_index, line in enumerate(block.lines):
            if drop_group_label and line_index == group_label_index:
                continue

            # Prerequisite (rare in the plan tables, kept for robustness).
            if self.PREREQ_KEYWORD_RE.search(line):
                p_val = line.split(":", 1)[1].strip() if ":" in line else line
                prerequisite = p_val.upper() if p_val else "ไม่มี"
                continue

            # Truncation rule: a line that opens a course-description paragraph
            # (e.g. "วิชานี้จะศึกษา...", "ศึกษาเกี่ยวกับ...", "THIS COURSE...",
            # "STUDY OF ...") marks the end of this block's metadata. STOP
            # reading further lines so all trailing description text is
            # discarded and never bleeds into the name / credit / prereq fields.
            if self.DESCRIPTION_START_RE.search(line):
                break

            # Lone OCR junk tokens that slip past a fully numeric code.
            if (
                len(code) == 8
                and code.isdigit()
                and line.upper() in {"X", "^", "D9", "L"}
            ):
                continue

            # A lone number / letter is a course-number suffix (e.g. "CALCULUS" + "1").
            if line.upper() in {"L", "1", "2", "3", "4", "I", "II"}:
                num = "1" if line.upper() in {"L", "I"} else ("2" if line.upper() == "II" else line)
                if not name_en:
                    name_th = f"{name_th} {num}".strip()
                else:
                    name_en = f"{name_en} {clean_ocr_en_text(line).upper()}".strip()
                continue

            # Credits and the "หรือ" keyword that joins alternative credit rows.
            if self.SINGLE_CREDIT_RE.search(line) or self.OR_KEYWORD_RE.search(line):
                credit_piece = "หรือ" if "หรอ" in line else line
                credits = f"{credits} {credit_piece}".strip() if credits else credit_piece
                continue

            # Thai / English course names can wrap across several lines.
            if self.HAS_THAI_RE.search(line):
                name_th = f"{name_th} {line}".strip()
            elif self.HAS_ENG_RE.search(line):
                name_en = f"{name_en} {line}".strip()

        # ---- clean up OCR noise ---------------------------------------------- #
        # Remove the "กลุ่ม วิชาที่กำหนดโดยคณะ*" label (supports spaces + asterisk).
        name_th = re.sub(r"กลุ่ม\s*วิชาที่กำหนดโดยคณะ\*", "", name_th).strip()
        # Remove "|" pipes (OCR table borders) and leading/trailing dashes/colons.
        name_th = name_th.replace("|", "").strip()
        name_th = re.sub(r"^\s*[-:]\s*", "", name_th).strip()
        name_th = re.sub(r"\s*[-–—]\s*$", "", name_th).strip()
        name_en = name_en.replace("|", "").strip()
        name_en = re.sub(r"^\s*[-:]\s*", "", name_en).strip()
        name_en = re.sub(r"\s*[-–—]\s*$", "", name_en).strip()
        name_en = clean_ocr_en_text(name_en).upper()

        # Normalize credits: "3 (3-0-6)" -> "3(3-0-6)".
        credits_clean = re.sub(r"\s*\(\s*", "(", credits)
        credits_clean = re.sub(r"\s*\)\s*", ")", credits_clean)
        credits_clean = re.sub(r"\)+", ")", credits_clean)
        credits_clean = re.sub(
            r"\s*(?:หรือ|or|/)\s*$", "", credits_clean, flags=re.IGNORECASE
        ).strip()
        if credits_clean.startswith("(0-35"):
            credits_clean = f"6{credits_clean}"

        # Normalize placeholder credits to GT convention: "3(X X X)" -> "3(x-x-x)".
        credits_clean = re.sub(
            r"\(([0-9xX])\s*[-\s]\s*([0-9xX])\s*[-\s]\s*([0-9xX])\)",
            lambda m: "({}-{}-{})".format(*m.group(1, 2, 3)).lower(),
            credits_clean,
        )

        final_credits = credits_clean if credits_clean else "3(3-0-6)"
        if final_credits == "3(3-0-6)" and ("สหกิจ" in name_th or "COOP" in name_en):
            final_credits = "6(0-35-0)"
        
        if code.startswith("90") :
            category = "หมวดวิชาศึกษาทั่วไป"
        elif code.startswith("xx") :
            category = "หมวดวิชาเสรี"
        else :
            category = "หมวดวิชาเฉพาะ"

        return {
            "code": code,
            "name_th": name_th if name_th else "ไม่ระบุ",
            "name_en": name_en if name_en else "N/A",
            "credits": final_credits,
            "year": 0 if self.plan == "gened" else block.year,
            "semester": 0 if self.plan == "gened" else block.semester,
            "category": category,
            "type": "เลือก" if self.plan == "gened" else block.type,
            "prerequisite": None if self.plan == "gened" else prerequisite,
            "flexible_year_semester": None,
            "note": note,
        }

    # ------------------------------------------------------------------ #
    #  Step 3: post-process the whole course list                         #
    # ------------------------------------------------------------------ #
    def post_process(self, courses: List[Dict]) -> List[Dict]:
        """
        High-level combinations applied to the extracted course list.

        Combines co-op (cooperative education) alternative course pairs into a
        single entry occupying the co-op slot.  Pairs are configured via
        `coop_pairs` (universal default covers the DSBA / AIT curricula).
        """
        combined: List[Dict] = []
        idx = 0
        while idx < len(courses):
            current = courses[idx]

            # Merge any configured co-op alternative pair (code_a + code_b)
            # into one "code_a หรือ code_b" entry.
            merged_credits = None
            for code_a, code_b, credits in self.coop_pairs:
                if (
                    idx + 1 < len(courses)
                    and current["code"] == code_a
                    and courses[idx + 1]["code"] == code_b
                ):
                    merged_credits = credits
                    break
            if merged_credits is not None:
                nxt = courses[idx + 1]
                merged = {
                    **current,
                    "code": f"{current['code']} หรือ {nxt['code']}",
                    "name_th": f"{current['name_th']}\n{nxt['name_th']}",
                    "name_en": f"{current['name_en']}\n{nxt['name_en']}",
                    "credits": merged_credits,
                }
                merged[SOURCE_PROVENANCE_KEY] = merge_source_provenance(current, nxt)
                combined.append(merged)
                idx += 2
                continue

            combined.append(current)
            idx += 1

        return combined

    # ------------------------------------------------------------------ #
    #  Public entry points                                                #
    # ------------------------------------------------------------------ #
    def extract_from_lines(
        self, lines: List[str], source_context: Optional[Dict] = None
    ) -> Dict:
        """Run the full block-based pipeline over the study-plan OCR lines."""
        print(" [DEBUG] running: extract_from_lines (study plan table)")
        source_context = source_context or self._source_context(
            document_category="unknown"
        )
        blocks = self.split_into_blocks(lines)
        courses = [
            self._attach_source_provenance(self.parse_single_block(block), source_context)
            for block in blocks
        ]
        courses = self.post_process(courses)

        return {
            "source": self.source,
            "description": f"Ground Truth รายวิชาหลักสูตร {self.program} (แผน {self.plan})",
            "program": self.program,
            "plan": self.plan,
            "courses": courses,
        }

    @classmethod
    def _is_description_code_anchor(cls, line: str) -> bool:
        return cls.DESCRIPTION_CODE_LINE_RE.fullmatch(line.strip()) is not None

    @classmethod
    def _is_description_page_header(cls, line: str) -> bool:
        return cls.DESCRIPTION_PAGE_HEADER_RE.match(line.strip()) is not None

    @classmethod
    def _is_description_structural_boundary(cls, line: str) -> bool:
        stripped = line.strip()
        return (
            cls._is_description_code_anchor(stripped)
            or cls._is_description_page_header(stripped)
            or cls.DESCRIPTION_SECTION_BOUNDARY_RE.search(stripped) is not None
            or cls.DESCRIPTION_FOOTER_RE.search(stripped) is not None
            or re.fullmatch(r"\d{1,4}", stripped) is not None
        )

    @staticmethod
    def _join_description_lines(lines: List[str]) -> str:
        return "\n".join(line.strip() for line in lines if line.strip()).strip()

    def _split_description_lines(
        self, lines: List[str], english_started: bool = False
    ) -> Tuple[List[str], List[str]]:
        """Split source lines without correcting their OCR wording."""
        thai_lines: List[str] = []
        english_lines: List[str] = []

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue

            if english_started:
                english_lines.append(line)
                continue

            has_thai = self.HAS_THAI_RE.search(line) is not None
            has_english = self.HAS_ENG_RE.search(line) is not None
            if has_thai:
                thai_lines.append(line)
            elif has_english:
                english_started = True
                english_lines.append(line)
            elif thai_lines:
                thai_lines.append(line)

        return thai_lines, english_lines

    def _append_description_lines(
        self, course: Dict, lines: List[str], source_context: Optional[Dict] = None
    ) -> None:
        """Append source lines to description fields, removing page-edge overlap."""
        thai_lines, english_lines = self._split_description_lines(
            lines, english_started=bool(course.get("desc_en"))
        )

        def append_field(field: str, field_lines: List[str]) -> bool:
            addition = self._join_description_lines(field_lines)
            if not addition:
                return False

            existing = self._join_description_lines(
                str(course.get(field, "")).splitlines()
            )
            if not existing:
                course[field] = addition
                return True

            existing_lines = existing.splitlines()
            addition_lines = addition.splitlines()
            overlap = 0
            max_overlap = min(len(existing_lines), len(addition_lines))
            for size in range(max_overlap, 0, -1):
                if existing_lines[-size:] == addition_lines[:size]:
                    overlap = size
                    break

            remaining = addition_lines[overlap:]
            if remaining:
                course[field] = "\n".join(existing_lines + remaining)
                return True
            return False

        changed = append_field("desc_th", thai_lines)
        changed = append_field("desc_en", english_lines) or changed
        if (changed or any(line.strip() for line in lines)) and source_context:
            course[SOURCE_PROVENANCE_KEY] = merge_source_provenance(
                course, [source_context]
            )

    def _extract_description_page(
        self, lines: List[str], source_context: Dict
    ) -> Tuple[List[Dict], List[str]]:
        """Extract one description page and retain leading continuation lines."""
        courses = []
        leading_lines: List[str] = []
        leading_blocked = False
        saw_course = False
        total = len(lines)

        code_regex = re.compile(r"\b\d{8}\b")
        credit_regex = re.compile(r"\d+\s*[({]\d+-\d+-\d+[)}]")
        any_prereq_key_regex = re.compile(
            r"(?:วิชาบังคับก่อน|บังคับก่อน|ความรู้พื้นฐาน|PRERE\s*[A-Z]*|"
            r"PRERECUISITE|PRERECUSITE|PREREQUISITE)",
            re.IGNORECASE,
        )
        prereq_eng_key_regex = re.compile(
            r"(?:PRERE\s*[A-Z]*|PRERECUISITE|PRERECUSITE|PREREQUISITE|"
            r"PRLRLCUISIIT|PRERLOUSIIE|FRFRROUISIIT)",
            re.IGNORECASE,
        )
        has_thai_regex = re.compile(r"[\u0e00-\u0e7f]")

        i = 0
        while i < total:
            line = lines[i].strip()
            code_match = code_regex.search(line)

            if not code_match or any_prereq_key_regex.search(line):
                if not saw_course and not leading_blocked and line:
                    if self._is_description_page_header(line):
                        i += 1
                        continue
                    if self.DESCRIPTION_SECTION_BOUNDARY_RE.search(line):
                        leading_blocked = True
                    elif self.DESCRIPTION_FOOTER_RE.search(line):
                        leading_blocked = True
                    elif not re.fullmatch(r"\d{1,4}", line):
                        leading_lines.append(line)
                i += 1
                continue

            code = code_match.group(0)
            saw_course = True
            name_th = ""
            name_en = ""
            credits = "3(3-0-6)"
            credits_seen = False

            th_words = []
            line_after_code = line[code_match.end() :].strip()
            if line_after_code and not line_after_code.isdigit():
                th_words.append(line_after_code)

            j = i + 1
            en_words: List[str] = []

            # Read the Thai course name, credits, and English course name.
            while j < total:
                curr = lines[j].strip()
                if not curr:
                    j += 1
                    continue

                if (
                    self._is_description_code_anchor(curr)
                ):
                    break

                if any_prereq_key_regex.search(curr):
                    break

                c_match = credit_regex.search(curr)
                if c_match:
                    credits = c_match.group(0).strip()
                    credits_seen = True
                    before_c = curr[: c_match.start()].strip()
                    if before_c and not before_c.isdigit():
                        th_words.append(before_c)
                    j += 1
                    continue

                if re.search(r"[a-zA-Z]", curr) and not has_thai_regex.search(curr):
                    clean_en = clean_ocr_en_text(curr).upper()
                    if clean_en and clean_en not in ["L", "NONE"]:
                        en_words.append(clean_en)
                elif en_words and curr in ["1", "2", "3", "L", "l"]:
                    en_words.append(curr)
                elif has_thai_regex.search(curr) and not credits_seen:
                    if not (curr.isdigit() and len(curr) <= 2):
                        th_words.append(curr)
                elif th_words and curr in ["1", "2", "3"]:
                    th_words.append(curr)

                j += 1

            if th_words:
                cleaned_th_words = []
                for word in th_words:
                    trailing_num = re.match(r"^(.*?)\s+(\d+)$", word)
                    if trailing_num:
                        cleaned_th_words.append(
                            trailing_num.group(1).replace(" ", "")
                            + " "
                            + trailing_num.group(2)
                        )
                    else:
                        cleaned_th_words.append(word.replace(" ", ""))
                name_th = "".join(cleaned_th_words).strip()

            if en_words:
                name_en = re.sub(r"\s+", " ", " ".join(en_words)).strip()

            # Read the prerequisite value without consuming the next course.
            prereq_tokens = []
            prev_line = ""
            while j < total:
                curr = lines[j].strip()
                if not curr:
                    j += 1
                    continue

                stop_code_match = code_regex.search(curr)
                if (
                    stop_code_match
                    and stop_code_match.group(0) != code
                    and not prereq_eng_key_regex.search(curr)
                    and not any_prereq_key_regex.search(prev_line)
                ):
                    break

                if prereq_eng_key_regex.search(curr):
                    remainder = re.sub(prereq_eng_key_regex, "", curr).strip(": ").strip()
                    if remainder:
                        if has_thai_regex.search(remainder):
                            break
                        cleaned_rem = clean_ocr_en_text(remainder).upper()
                        if cleaned_rem:
                            prereq_tokens.append(cleaned_rem)

                    j += 1
                    while j < total:
                        sub_line = lines[j].strip()
                        if not sub_line:
                            j += 1
                            continue

                        if has_thai_regex.search(sub_line):
                            prereq_code_match = code_regex.search(sub_line)
                            if prereq_code_match and prereq_code_match.group(0) != code:
                                prereq_tokens.append(prereq_code_match.group(0))
                                j += 1
                            break

                        sub_upper = clean_ocr_en_text(sub_line).upper()
                        if sub_upper:
                            prereq_tokens.append(sub_upper)
                        j += 1
                    break

                prev_line = curr
                j += 1

            prerequisite = "ไม่มี"
            if prereq_tokens:
                clean_prereq = " ".join(prereq_tokens).strip()
                if clean_prereq not in ["NONE", "ไม่มี", ""]:
                    codes_found = re.findall(r"\b\d{8}\b", clean_prereq)
                    prerequisite = ", ".join(dict.fromkeys(codes_found)) if codes_found else clean_prereq

            desc_lines: List[str] = []
            while j < total:
                curr = lines[j].strip()
                if not curr:
                    j += 1
                    continue
                if self._is_description_structural_boundary(curr):
                    break
                desc_lines.append(curr)
                j += 1

            is_gened = code.startswith("90")
            course = {
                "code": code,
                "name_th": name_th if name_th else "ไม่ระบุ",
                "name_en": name_en if name_en else "N/A",
                "credits": credits.replace(" ", "").replace("{", "(").replace("}", ")"),
                "year": 0,
                "semester": 0,
                "category": "หมวดวิชาศึกษาทั่วไป" if is_gened else "หมวดวิชาเฉพาะ",
                "type": "เลือก",
                "prerequisite": None if is_gened else prerequisite,
                "flexible_year_semester": (
                    None
                    if is_gened
                    else "3/1, 3/2, 4/1"
                    if self.program == "DSBA"
                    else "4/1"
                    if self.program == "IT" and self.plan == "coop"
                    else "3/1, 3/2, 4/1"
                    if self.program == "IT"
                    else "3/1, 3/2"
                    if self.program == "AIT"
                    else "4/2"
                ),
                "note": "เฉพาะโครงการเข้าร่วมสหกิจ" if "สหกิจศึกษา" in name_th else None,
            }
            self._append_description_lines(course, desc_lines)
            courses.append(self._attach_source_provenance(course, source_context))
            i = j

        return courses, leading_lines

    def extract_descriptions(
        self, lines: List[str], source_context: Optional[Dict] = None
    ) -> Dict:
        print(" [DEBUG] running: extract_descriptions (course descriptions)")
        source_context = source_context or self._source_context(
            document_category="unknown"
        )
        courses, _ = self._extract_description_page(lines, source_context)
        return {
            "source": self.source,
            "description": f"Ground Truth รายวิชาหลักสูตร {self.program} (แผน {self.plan})",
            "program": self.program,
            "plan": self.plan,
            "courses": courses,
        }

    def extract_descriptions_from_pages(
        self, pages: List[Tuple[List[str], Dict]]
    ) -> Dict:
        """Extract ordered description pages and join page-edge continuations."""

        def page_sort_key(item):
            index, (_, context) = item
            page = _page_number(context.get("source_page")) if isinstance(context, dict) else None
            if page is None and isinstance(context, dict):
                page = _page_number_from_name(context.get("source_filename"))
            return (0, page) if page is not None else (1, index)

        ordered_pages = sorted(
            enumerate(pages),
            key=page_sort_key,
        )

        courses: List[Dict] = []
        for _, page in ordered_pages:
            lines, source_context = page
            page_courses, leading_lines = self._extract_description_page(
                lines, source_context
            )
            if leading_lines and courses:
                self._append_description_lines(
                    courses[-1], leading_lines, source_context
                )
            courses.extend(page_courses)

        return {
            "source": self.source,
            "description": f"Ground Truth รายวิชาหลักสูตร {self.program} (แผน {self.plan})",
            "program": self.program,
            "plan": self.plan,
            "courses": courses,
        }

    def process_file(self, file_path: Union[str, Path]) -> Dict:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        lines = []
        metadata = {}
        if file_path.suffix == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                lines = data.get("text_lines", [])
                metadata = data if isinstance(data, dict) else {}
        else:
            lines = file_path.read_text(encoding="utf-8").splitlines()

        lines = [line.upper() for line in lines]
        content_upper = "\n".join(lines)

        # Universal pre-clean (deterministic regex, no LLM): repair OCR noise
        # so extraction only trusts the two universal anchors (8-digit codes
        # and X(X-X-X) credits).  Works for ANY university's OCR output.
        cleaned = pre_clean_with_regex(content_upper)
        lines = [ln for ln in cleaned.split("\n") if ln.strip()]
        content_upper = "\n".join(lines)

        #  Fix point 1: detect the "study plan" structure decisively (contains "ปีที่/ชั้นปีที่" or has a course code + credits table header)
        is_plan_page = bool(re.search(r"(?:ปีที่|ชั้นปีที่)\s*\d+", content_upper)) or \
                       (bool(re.search(r"รหัสวิชา", content_upper)) and bool(re.search(r"หน่วยกิต", content_upper)))

        if is_plan_page:
            return self.extract_from_lines(
                lines,
                self._source_context(file_path, metadata, "plan"),
            )

        # If not a study plan, check whether it is a course description page
        is_description_page = bool(
            re.search(r"(?:คำอธิบายรายวิชา|COURSE\s*DESCRIPTION|PREREQUISITE|PRERE)", content_upper)
        )

        if is_description_page:
            return self.extract_descriptions(
                lines,
                self._source_context(file_path, metadata, "description"),
            )

        return self.extract_from_lines(
            lines,
            self._source_context(file_path, metadata, "unknown"),
        )
