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
    if 5 <= len(cleaned) <= 9 and re.search(r"\d", cleaned):
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
    NOTE_RE = re.compile(
        r"^\s*[-*]|(?:ประเมิน|เกณฑ์|ผลการเรียน|ผ่าน\s*\(S\)|\(S\)|\(U\)|ให้นักศึกษา)",
        re.IGNORECASE,
    )

    def __init__(
        self,
        program: str = "DSBA",
        plan: str = "coop",
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
        else:
            self.source = source

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
            if self.CATEGORY_HEADER_RE.search(line) and not self.CREDITS_RE.search(line):
                self._ctx_category = line
                self._ctx_type = "เลือก" if "เลือก" in line else "บังคับ"
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
            return normalize_course_code(m.group(1)), line[m.end(1):].strip()

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

        for line in block.lines:
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
            "note": None,
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
                combined.append(merged)
                idx += 2
                continue

            combined.append(current)
            idx += 1

        return combined

    # ------------------------------------------------------------------ #
    #  Public entry points                                                #
    # ------------------------------------------------------------------ #
    def extract_from_lines(self, lines: List[str]) -> Dict:
        """Run the full block-based pipeline over the study-plan OCR lines."""
        print(" [DEBUG] running: extract_from_lines (study plan table)")
        blocks = self.split_into_blocks(lines)
        courses = [self.parse_single_block(block) for block in blocks]
        courses = self.post_process(courses)

        return {
            "source": self.source,
            "description": f"Ground Truth รายวิชาหลักสูตร {self.program} (แผน {self.plan})",
            "program": self.program,
            "plan": self.plan,
            "courses": courses,
        }

    def extract_descriptions(self, lines: List[str]) -> Dict:
        print(" [DEBUG] running: extract_descriptions (course descriptions)")
        courses = []
        seen_codes = set()
        i = 0
        total = len(lines)

        code_regex = re.compile(r"\b\d{8}\b")
        credit_regex = re.compile(r"\d+\s*[({]\d+-\d+-\d+[)}]")
        
        # Combine mandatory keywords in both Thai and English to stop reading the course name
        any_prereq_key_regex = re.compile(
            r"(?:วิชาบังคับก่อน|บังคับก่อน|ความรู้พื้นฐาน|PRERE\s*[A-Z]*|PRERECUISITE|PRERECUSITE|PREREQUISITE)",
            re.IGNORECASE,
        )
        
        # English keywords for starting to collect the prerequisite value
        prereq_eng_key_regex = re.compile(
            r"(?:PRERE\s*[A-Z]*|PRERECUISITE|PRERECUSITE|PREREQUISITE|PRLRLCUISIIT|PRERLOUSIIE|FRFRROUISIIT)",
            re.IGNORECASE,
        )
        has_thai_regex = re.compile(r"[\u0e00-\u0e7f]")

        while i < total:
            line = lines[i].strip()
            code_match = code_regex.search(line)

            if code_match and not any_prereq_key_regex.search(line):
                code = code_match.group(0)

                if code in seen_codes:
                    i += 1
                    continue

                name_th = ""
                name_en = ""
                credits = "3(3-0-6)"
                credits_seen = False
                prerequisite = "ไม่มี"

                th_words = []
                line_after_code = line[code_match.end() :].strip()
                if line_after_code and not line_after_code.isdigit():
                    th_words.append(line_after_code)

                j = i + 1

                # 1. Read the Thai course name, credits, and English name
                name_en = ""
                en_words = [] # use a List to hold English chunks across multiple lines
                
                if name_en: 
                    en_words.append(name_en)

                while j < total:
                    curr = lines[j].strip()
                    if not curr:
                        j += 1
                        continue

                    # Block boundary: a line holding the next 8-digit course code ends
                    # this block's metadata (a missing prereq keyword must not swallow
                    # the following course's lines as this course's name).
                    boundary_code_match = code_regex.search(curr)
                    if (
                        boundary_code_match
                        and boundary_code_match.group(0) != code
                        and boundary_code_match.group(0) not in seen_codes
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

                    # Collect multi-line English names
                    if re.search(r"[a-zA-Z]", curr) and not has_thai_regex.search(curr):
                        clean_en = clean_ocr_en_text(curr).upper()
                        if clean_en and clean_en not in ["L", "NONE"]:
                            en_words.append(clean_en)
                    # Case: lone numbers that fell onto their own line
                    elif en_words and curr in ["1", "2", "3", "L", "l"]:
                        en_words.append(clean_ocr_en_text(curr).upper())

                    # Collect multi-line Thai names.  The Thai course name always
                    # precedes the credits line, so once a credit line has been
                    # read the Thai name is complete: any later Thai line is the
                    # course-description body, not the name.
                    elif has_thai_regex.search(curr) and not credits_seen:
                        if not (curr.isdigit() and len(curr) <= 2):
                            th_words.append(curr)
                    elif th_words and curr in ["1", "2", "3"]:
                        th_words.append(curr)

                    j += 1

                # Assemble Thai name without spaces
                if th_words:
                    # Remove spaces within each item, then join them together
                    cleaned_th_words = [w.replace(" ", "") for w in th_words]
                    name_th = "".join(cleaned_th_words).strip()

                # Assemble English name with single-space separators
                if en_words:
                    name_en = " ".join(en_words).strip()
                    # Remove any spaces that exceed 1 space
                    name_en = re.sub(r'\s+', ' ', name_en)

                # Read the PREREQUISITE part (skip all Thai until English PREREQUISITE is found)
                prereq_tokens = []
                prev_line = ""

                while j < total:
                    curr = lines[j].strip()
                    if not curr:
                        j += 1
                        continue

                    # A course code directly after a prerequisite keyword (Thai/English) is
                    # the prerequisite itself, not the start of a new course (e.g. page 321).
                    stop_code_match = code_regex.search(curr)
                    if (
                        stop_code_match
                        and stop_code_match.group(0) != code
                        and stop_code_match.group(0) not in seen_codes
                        and not prereq_eng_key_regex.search(curr)
                        and not any_prereq_key_regex.search(prev_line)
                    ):
                        break

                    if prereq_eng_key_regex.search(curr):
                        remainder = re.sub(prereq_eng_key_regex, "", curr).strip(": ").strip()
                        
                        if remainder:
                            if has_thai_regex.search(remainder):
                                break
                            else:
                                #  Fix item 2: clean text via clean_ocr_en_text (change L to 1 if at word end)
                                cleaned_rem = clean_ocr_en_text(remainder).upper()
                                if cleaned_rem:
                                    prereq_tokens.append(cleaned_rem)

                        j += 1

                        while j < total:
                            sub_line = lines[j].strip()
                            if not sub_line:
                                j += 1
                                continue

                            #  When Thai is found (course description line) = stop collecting
                            #  Prerequisite immediately. A prerequisite course code may share
                            #  that line (e.g. "06066001 ความน่าจะเจ็") — the code is the
                            #  prerequisite, NOT the start of a new course block.
                            if has_thai_regex.search(sub_line):
                                prereq_code_match = code_regex.search(sub_line)
                                if prereq_code_match and prereq_code_match.group(0) != code:
                                    prereq_tokens.append(prereq_code_match.group(0))
                                j += 1
                                break

                            sub_upper = clean_ocr_en_text(sub_line).upper()
                            if len(sub_upper) > 0:
                                prereq_tokens.append(sub_upper)

                            j += 1

                        break

                    prev_line = curr
                    j += 1

                # Summarize prerequisite value
                if prereq_tokens:
                    clean_prereq = " ".join(prereq_tokens).strip()
                    if clean_prereq in ["NONE", "ไม่มี", ""]:
                        prerequisite = "ไม่มี"
                    else:
                        # GT stores prerequisites as plain course codes, so reduce
                        # any captured text to just the 8-digit codes (e.g.
                        # "06046401 CALC01US 2, 06046402 LINEAR" -> "06046401, 06046402").
                        codes_found = re.findall(r"\b\d{8}\b", clean_prereq)
                        if codes_found:
                            prerequisite = ", ".join(dict.fromkeys(codes_found))
                        else:
                            prerequisite = clean_prereq
                else:
                    prerequisite = "ไม่มี"

                # 3. Skip course description content lines to find the next course code
                while j < total:
                    curr = lines[j].strip()
                    m_next = code_regex.search(curr)

                    if m_next and not any_prereq_key_regex.search(curr):
                        next_code = m_next.group(0)
                        if next_code != code and next_code not in seen_codes:
                            break

                    if curr.startswith("วท.บ."):
                        break

                    j += 1

                seen_codes.add(code)
                credits = credits.replace(" ", "").replace("{", "(").replace("}", ")")
                
                if code.startswith("90"):  # institutional GenEd code
                    courses.append(
                        {
                            "code": code,
                            "name_th": name_th if name_th else "ไม่ระบุ",
                            "name_en": name_en if name_en else "N/A",
                            "credits": credits,
                            "category": "หมวดวิชาศึกษาทั่วไป",
                            "year": 0,
                            "semester": 0,
                            "type": "เลือก",
                            "prerequisite": None,
                            "flexible_year_semester": None,
                            "note": None,
                        }
                    )
                else:  # specific / faculty course codes (06xxxxx)
                    if self.program == "DSBA" :
                        flex_year = "3/1, 3/2, 4/1"
                    elif self.program == "IT" and self.plan == "coop":
                        flex_year = "4/1"
                    elif self.program == "IT" :
                        flex_year = "3/1, 3/2, 4/1"
                    elif self.program == "AIT" :
                        flex_year = "3/1, 3/2"
                    elif self.program == "BIT" :
                        flex_year = "4/2"
                    
                    if code.startswith("90") :
                        category = "หมวดวิชาศึกษาทั่วไป"
                    elif code.startswith("xx") :
                        category = "หมวดวิชาเสรี"
                    else :
                        category = "หมวดวิชาเฉพาะ"
                    
                    courses.append(
                        {
                            "code": code,
                            "name_th": name_th if name_th else "ไม่ระบุ",
                            "name_en": name_en if name_en else "N/A",
                            "credits": credits,
                            "year": 0,
                            "semester": 0,
                            "category": category,
                            "type": "เลือก",
                            "prerequisite": prerequisite,
                            "flexible_year_semester": flex_year,
                            "note": "เฉพาะโครงการเข้าร่วมสหกิจ" if "สหกิจศึกษา" in name_th else None,
                        }
                    )
                i = j
                continue
            i += 1


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
        if file_path.suffix == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                lines = data.get("text_lines", [])
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
            return self.extract_from_lines(lines)

        # If not a study plan, check whether it is a course description page
        is_description_page = bool(
            re.search(r"(?:คำอธิบายรายวิชา|COURSE\s*DESCRIPTION|PREREQUISITE|PRERE)", content_upper)
        )

        if is_description_page:
            return self.extract_descriptions(lines)

        return self.extract_from_lines(lines)
