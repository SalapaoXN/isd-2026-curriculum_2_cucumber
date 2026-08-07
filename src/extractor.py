import json
from pathlib import Path
import re
from typing import Dict, List, Union


class CurriculumExtractor:

    def __init__(
        self,
        program: str = "DSBA",
        plan: str = "coop",
        source: str = "GT_Template-2.xlsx / Academic Plan GT — DSBA coop",
    ):
        self.program = program
        self.plan = plan
        self.source = source

    def extract_from_lines(self, lines: List[str]) -> Dict:
        """Parse raw OCR text lines into structured curriculum JSON format."""
        courses = []

        current_year = 1
        current_semester = 1
        current_category = "หมวดวิชาเฉพาะ"
        current_type = "บังคับ"

        # Regex Patterns
        course_code_regex = re.compile(r"\b(\d{8})\b")
        credits_regex = re.compile(r"(\d\s*\(\s*\d\s*-\s*\d\s*-\s*\d\s*\))")
        year_sem_regex = re.compile(
            r"(?:ชั้นปีที่|ปีที่|ปี)\s*(\d+).*(?:ภาคการศึกษาที่|ภาคเรียนที่|ภาค)\s*(\d+)",
            re.IGNORECASE,
        )
        prereq_keyword_regex = re.compile(
            r"(?:วิชาบังคับก่อน|พื้นฐาน|prerequisite|pre-requisite)",
            re.IGNORECASE,
        )

        idx = 0
        total = len(lines)

        while idx < total:
            line = lines[idx].strip()
            if not line:
                idx += 1
                continue

            # 1. Update Year & Semester Context
            ys_match = year_sem_regex.search(line)
            if ys_match:
                try:
                    current_year = int(ys_match.group(1))
                    current_semester = int(ys_match.group(2))
                except ValueError:
                    current_year = ys_match.group(1)
                    current_semester = ys_match.group(2)
                idx += 1
                continue

            # 2. Update Category / Course Type Context
            if "หมวดวิชา" in line or "หมวด" in line:
                current_category = line
                if "เลือก" in line:
                    current_type = "เลือก"
                elif "บังคับ" in line:
                    current_type = "บังคับ"
                idx += 1
                continue
            elif line in ["วิชาบังคับ", "บังคับ"]:
                current_type = "บังคับ"
                idx += 1
                continue
            elif line in ["วิชาเลือก", "เลือก"]:
                current_type = "เลือก"
                idx += 1
                continue

            # 3. Handle standalone prerequisite lines attached to previous course
            if prereq_keyword_regex.search(line) and courses:
                p_text = line
                if ":" in line:
                    p_text = line.split(":", 1)[1].strip()
                courses[-1]["prerequisite"] = p_text if p_text else "ไม่มี"
                idx += 1
                continue

            # 4. Extract Course Code (8 digits)
            code_match = course_code_regex.search(line)
            if code_match and not prereq_keyword_regex.search(line):
                code = code_match.group(1)

                # Text after code on the same line
                line_after_code = line[
                    line.find(code) + len(code) :
                ].strip()

                name_th = line_after_code
                name_en = ""
                credits = ""
                prerequisite = "ไม่มี"
                flexible_year_semester = None
                note = None

                # Look ahead in subsequent lines for details
                j = idx + 1
                while j < total:
                    next_line = lines[j].strip()
                    if not next_line:
                        j += 1
                        continue

                    # Stop if next course code, year/sem header, or new category appears
                    if (
                        course_code_regex.search(next_line)
                        and not prereq_keyword_regex.search(next_line)
                    ) or year_sem_regex.search(next_line) or "หมวด" in next_line:
                        break

                    # Check prerequisite line
                    if prereq_keyword_regex.search(next_line):
                        p_val = next_line
                        if ":" in next_line:
                            p_val = next_line.split(":", 1)[1].strip()
                        prerequisite = p_val if p_val else "ไม่มี"
                        j += 1
                        continue

                    # Check credit pattern
                    cred_m = credits_regex.search(next_line)
                    if cred_m:
                        credits = cred_m.group(1).replace(" ", "")
                    elif re.search(r"[a-zA-Z]", next_line):
                        if not name_en:
                            name_en = next_line
                        else:
                            name_en += " " + next_line
                    elif not name_th:
                        name_th = next_line

                    j += 1

                # Clean strings
                name_th = re.sub(r"^\s*[-:]\s*", "", name_th).strip()
                name_en = re.sub(r"^\s*[-:]\s*", "", name_en).strip()

                courses.append({
                    "code": code,
                    "name_th": name_th if name_th else "ไม่ระบุ",
                    "name_en": name_en if name_en else "N/A",
                    "credits": credits if credits else "3(3-0-6)",
                    "year": current_year,
                    "semester": current_semester,
                    "category": current_category,
                    "type": current_type,
                    "prerequisite": prerequisite,
                    "flexible_year_semester": flexible_year_semester,
                    "note": note,
                })

                idx = j
                continue

            idx += 1

        return {
            "source": self.source,
            "description": f"Ground Truth รายวิชาหลักสูตร {self.program} (แผน {self.plan})",
            "program": self.program,
            "plan": self.plan,
            "courses": courses,
        }

    def process_file(self, file_path: Union[str, Path]) -> Dict:
        """Process a .txt or .json OCR output file."""
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

        return self.extract_from_lines(lines)