import json
from pathlib import Path
import re
from typing import Dict, List, Union


def clean_ocr_en_text(text: str) -> str:
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
    code = code.strip()
    code_lower = code.lower()

    if re.search(r"[a-z]", code_lower):
        m = re.match(r"^(\d+)", code)
        if m:
            prefix = m.group(1)
            return prefix + ("x" * (8 - len(prefix)))
        return "xxxxxxxx"

    return code


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
        print("🔥 [DEBUG] กำลังรัน: extract_from_lines (ตารางเรียน)")
        courses = []

        current_year = 1
        current_semester = 1
        current_category = "หมวดวิชาเฉพาะ"
        current_type = "บังคับ"

        course_code_regex = re.compile(
            r"(?:^|\s)(\b[0-9]{8}\b|\b[0-9xX]{5,9}\b|\b\d{5}[a-zA-Z]{3}\b|^[xX]+$|^[xX][wW]$)(?:\s|$)", re.IGNORECASE
        )
        credits_regex = re.compile(
            r"(?:\d+\s*)?\(\d+-\d+-\d+\)(?:\s*(?:หรือ|or|/)\s*(?:\d+\s*)?\(\d+-\d+-\d+\))?",
            re.IGNORECASE,
        )
        single_credit_regex = re.compile(r"(?:\d+\s*)?\(\d+-\d+-\d+\)")
        category_header_regex = re.compile(
            r"^\s*(?:\d+\.\s*)?(?:หมวดวิชา|กลุ่มวิชา)", re.IGNORECASE
        )
        or_keyword_regex = re.compile(r"^\s*(?:หรือ|หรอ|or|/)\s*$", re.IGNORECASE)
        year_regex = re.compile(r"(?:ชั้น|[ปขชบ])ี\s*ที่?\s*(\d+)", re.IGNORECASE)
        sem_regex = re.compile(
            r"(?:ภาค|เทอม)\s*(?:การศึกษา|เรียน)?\s*ที่?\s*(\d+)", re.IGNORECASE
        )
        prereq_keyword_regex = re.compile(
            r"(?:วิชาบังคับก่อน|บังคับก่อน|ความรู้พื้นฐาน|prerequisite|pre-requisite|PRERE\s*[A-Z]*|PRERECUISITE)",
            re.IGNORECASE,
        )
        note_regex = re.compile(
            r"^\s*[-*]|(?:ประเมิน|เกณฑ์|ผลการเรียน|ผ่าน\s*\(S\)|\(S\)|\(U\)|ให้นักศึกษา)",
            re.IGNORECASE,
        )

        has_thai_regex = re.compile(r"[\u0e00-\u0e7f]")
        has_eng_regex = re.compile(r"[a-zA-Z]")

        idx = 0
        total = len(lines)

        while idx < total:
            line = lines[idx].strip()
            if not line or line == "รวม":
                idx += 1
                continue

            if note_regex.search(line) and not course_code_regex.search(line):
                idx += 1
                continue

            y_match = year_regex.search(line)
            s_match = sem_regex.search(line)

            if y_match or s_match:
                if y_match:
                    current_year = int(y_match.group(1))
                if s_match:
                    current_semester = int(s_match.group(1))

            if (
                (y_match or s_match)
                and not course_code_regex.search(line)
                and not credits_regex.search(line)
            ):
                idx += 1
                continue

            is_category_header = bool(category_header_regex.search(line)) and not (
                course_code_regex.search(line) or credits_regex.search(line)
            )

            if is_category_header:
                current_category = line
                current_type = "เลือก" if "เลือก" in line else "บังคับ"
                idx += 1
                continue

            if prereq_keyword_regex.search(line) and courses:
                p_text = line.split(":", 1)[1].strip() if ":" in line else line
                courses[-1]["prerequisite"] = p_text.upper() if p_text else "ไม่มี"
                idx += 1
                continue

            code_match = course_code_regex.search(line)
            if code_match and not prereq_keyword_regex.search(line):
                raw_code = code_match.group(1)
                code = normalize_course_code(raw_code)

                idx_code = line.upper().find(raw_code.upper())
                if idx_code != -1:
                    line_after_code = line[idx_code + len(raw_code):].strip()
                else:
                    line_after_code = line.replace(raw_code, "").strip()

                name_th = ""
                name_en = ""
                credits = ""
                prerequisite = "ไม่มี"

                same_line_cred = credits_regex.search(line_after_code)
                if same_line_cred:
                    credits = same_line_cred.group(0).strip()
                    line_after_code = line_after_code.replace(credits, "").strip()

                if line_after_code:
                    if has_thai_regex.search(line_after_code):
                        name_th = line_after_code
                    elif has_eng_regex.search(line_after_code):
                        name_en = line_after_code

                j = idx + 1
                while j < total:
                    next_line = lines[j].strip()
                    if not next_line:
                        j += 1
                        continue

                    # 🟢 [แก้ไขจุดนี้]: เช็กว่าเป็นรหัสวิชาใหม่จริงหรือไม่
                    is_next_code = bool(course_code_regex.search(next_line)) and not prereq_keyword_regex.search(next_line)

                    # 💡 ทริกสำคัญ: ถ้าวิชาปัจจุบันมีรหัสตัวเลข 8 หลักแล้ว (เช่น 06026200)
                    # แต่ next_line ดันเป็นตัว "X" ตัวเดียวขยะๆ หลุดเข้ามา -> ข้ามมันไป ห้ามสั่ง break!
                    if is_next_code and next_line.upper() in ["X", "^", "D9", "L"]:
                        if len(code) == 8 and code.isdigit():
                            j += 1
                            continue # ข้ามขยะ OCR ตัวนี้ไป แล้วอ่านบรรทัดถัดไปต่อ

                    is_next_category = bool(
                        category_header_regex.search(next_line)
                    ) and not (
                        course_code_regex.search(next_line)
                        or credits_regex.search(next_line)
                    )

                    # สั่ง break ตัดจบรายการเดิมเมื่อเจอวิชาใหม่จริงๆ เท่านั้น
                    if (
                        is_next_code
                        or (
                            year_regex.search(next_line)
                            and not credits_regex.search(next_line)
                        )
                        or (
                            sem_regex.search(next_line)
                            and not credits_regex.search(next_line)
                        )
                        or is_next_category
                        or next_line == "รวม"
                    ):
                        break

                    if prereq_keyword_regex.search(next_line):
                        p_val = (
                            next_line.split(":", 1)[1].strip()
                            if ":" in next_line
                            else next_line
                        )
                        prerequisite = p_val.upper() if p_val else "ไม่มี"
                        j += 1
                        continue
                    
                    if next_line.upper() in ["L", "1", "2", "3", "4", "I", "II"]:
                        if not name_en:
                            num = "1" if next_line.upper() in ["L", "I"] else ("2" if next_line.upper() == "II" else next_line)
                            name_th = f"{name_th} {num}".strip()
                        else:
                            name_en = f"{name_en} {clean_ocr_en_text(next_line).upper()}".strip()
                        j += 1
                        continue

                    if single_credit_regex.search(next_line) or or_keyword_regex.search(
                        next_line
                    ):
                        clean_credit_text = "หรือ" if "หรอ" in next_line else next_line

                        if not credits:
                            credits = clean_credit_text
                        else:
                            credits += f" {clean_credit_text}"
                        j += 1
                        continue

                    if has_thai_regex.search(next_line):
                        name_th = f"{name_th} {next_line}".strip()
                    elif has_eng_regex.search(next_line):
                        name_en = f"{name_en} {next_line}".strip()

                    j += 1

                # ลบคำว่า "กลุ่ม วิชาที่กำหนดโดยคณะ*" (รองรับกรณีเว้นวรรคและมี/ไม่มีดอกจัน)
                name_th = re.sub(r"กลุ่ม\s*วิชาที่กำหนดโดยคณะ\*", "", name_th).strip()
                
                # ลบสัญลักษณ์ | (Pipe) ที่เกิดจาก OCR สแกนขอบตารางเพี้ยน
                name_th = name_th.replace("|", "").strip()

                # ลบขีด หรือ โคลอน ที่อยู่หน้าสุด
                name_th = re.sub(r"^\s*[-:]\s*", "", name_th).strip()
                name_en = re.sub(r"^\s*[-:]\s*", "", name_en).strip()
                name_en = clean_ocr_en_text(name_en).upper()

                credits_clean = re.sub(r"\s*\(\s*", "(", credits)
                credits_clean = re.sub(r"\s*\)\s*", ")", credits_clean)
                credits_clean = re.sub(r"\)+", ")", credits_clean)
                credits_clean = re.sub(r"\s*(?:หรือ|or|/)\s*$", "", credits_clean, flags=re.IGNORECASE).strip()

                if credits_clean.startswith("(0-35"):
                    credits_clean = f"6{credits_clean}"

                final_credits = credits_clean if credits_clean else "3(3-0-6)"
                if final_credits == "3(3-0-6)" and ("สหกิจ" in name_th or "COOP" in name_en):
                    final_credits = "6(0-35-0)"

                courses.append(
                    {
                        "code": code,
                        "name_th": name_th if name_th else "ไม่ระบุ",
                        "name_en": name_en if name_en else "N/A",
                        "credits": credits_clean if credits_clean else "3(3-0-6)",
                        "year": current_year,
                        "semester": current_semester,
                        "category": current_category,
                        "type": current_type,
                        "prerequisite": prerequisite,
                        "flexible_year_semester": None,
                        "note": None,
                    }
                )

                idx = j
                continue

            idx += 1
        
        # ==========================================
        # ลอจิกรวมวิชาทางเลือก 06026259/06026260
        # ==========================================
        combined_courses = []
        idx_c = 0
        while idx_c < len(courses):
            current_course = courses[idx_c]
            
            # เช็คว่ามีวิชาถัดไป และเป็นคู่ 06026259 กับ 06026260 หรือไม่
            if idx_c + 1 < len(courses):
                next_course = courses[idx_c + 1]
                
                if current_course["code"] == "06026259" and next_course["code"] == "06026260":
                    # 1. รวมรหัส
                    current_course["code"] = f"{current_course['code']} หรือ {next_course['code']}"
                    
                    # 2. รวมชื่อไทย คั่น \n
                    current_course["name_th"] = f"{current_course['name_th']}\n{next_course['name_th']}"
                    
                    # 3. รวมชื่ออังกฤษ คั่น \n
                    current_course["name_en"] = f"{current_course['name_en']}\n{next_course['name_en']}"
                    
                    # 4. บังคับหน่วยกิตเป็น 6(0-35-0)
                    current_course["credits"] = "6(0-35-0)"
                    
                    combined_courses.append(current_course)
                    idx_c += 2  # ข้ามวิชาถัดไปเพราะโดนจับรวมแล้ว
                    continue
                    
            combined_courses.append(current_course)
            idx_c += 1
            
        courses = combined_courses
        # ==========================================

        return {
            "source": self.source,
            "description": f"Ground Truth รายวิชาหลักสูตร {self.program} (แผน {self.plan})",
            "program": self.program,
            "plan": self.plan,
            "courses": courses,
        }

    def extract_descriptions(self, lines: List[str]) -> Dict:
        print("⚡ [DEBUG] กำลังรัน: extract_descriptions (คำอธิบายรายวิชา)")
        courses = []
        seen_codes = set()
        i = 0
        total = len(lines)

        code_regex = re.compile(r"\b\d{7,8}\b")
        credit_regex = re.compile(r"\d+\s*[({]\d+-\d+-\d+[)}]")
        
        # รวม Keyword บังคับก่อนทั้งภาษาไทยและอังกฤษ เพื่อใช้หยุดการอ่านชื่อวิชา (name_th)
        any_prereq_key_regex = re.compile(
            r"(?:วิชาบังคับก่อน|บังคับก่อน|ความรู้พื้นฐาน|PRERE\s*[A-Z]*|PRERECUISITE|PRERECUSITE|PREREQUISITE)",
            re.IGNORECASE,
        )
        
        # Keyword ภาษาอังกฤษ สำหรับเริ่มเก็บค่า Prerequisite
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
                prerequisite = "ไม่มี"

                th_words = []
                line_after_code = line[code_match.end() :].strip()
                if line_after_code and not line_after_code.isdigit():
                    th_words.append(line_after_code)

                j = i + 1

                # 1. อ่านชื่อวิชาภาษาไทย, หน่วยกิต, และชื่อภาษาอังกฤษ
                name_en = ""
                en_words = [] # ใช้ List เก็บก้อนภาษาอังกฤษเพื่อรองรับหลายบรรทัด
                
                if name_en: 
                    en_words.append(name_en)

                while j < total:
                    curr = lines[j].strip()
                    if not curr:
                        j += 1
                        continue

                    if any_prereq_key_regex.search(curr):
                        break

                    c_match = credit_regex.search(curr)
                    if c_match:
                        credits = c_match.group(0).strip()
                        before_c = curr[: c_match.start()].strip()
                        if before_c and not before_c.isdigit():
                            th_words.append(before_c)
                        j += 1
                        continue

                    # เก็บชื่อภาษาอังกฤษแบบหลายบรรทัด
                    if re.search(r"[a-zA-Z]", curr) and not has_thai_regex.search(curr):
                        clean_en = clean_ocr_en_text(curr).upper()
                        if clean_en and clean_en not in ["L", "NONE"]:
                            en_words.append(clean_en)
                    # กรณีเป็นตัวเลขโดดๆ ที่ตกบรรทัด
                    elif en_words and curr in ["1", "2", "3", "L", "l"]:
                        en_words.append(clean_ocr_en_text(curr).upper())

                    # เก็บชื่อภาษาไทยแบบหลายบรรทัด
                    elif has_thai_regex.search(curr):
                        if not (curr.isdigit() and len(curr) <= 2):
                            th_words.append(curr)
                    elif th_words and curr in ["1", "2", "3"]:
                        th_words.append(curr)

                    j += 1

                # ประกอบร่างชื่อภาษาไทยแบบไม่เว้นวรรค
                if th_words:
                    # ตัดช่องว่างในตัวรายการทิ้ง แล้วนำมาต่อกัน
                    cleaned_th_words = [w.replace(" ", "") for w in th_words]
                    name_th = "".join(cleaned_th_words).strip()
                    name_th = re.sub(r"\bแคลคูลส\b", "แคลคูลัส", name_th)

                # ประกอบร่างชื่อภาษาอังกฤษเว้นวรรค 1 เคาะ
                if en_words:
                    name_en = " ".join(en_words).strip()
                    # กำจัดช่องว่างที่อาจซ้อนกันเกิน 1 เคาะ
                    name_en = re.sub(r'\s+', ' ', name_en)

                # อ่านส่วน PREREQUISITE (ข้ามภาษาไทยทั้งหมด จนกว่าจะเจอ PREREQUISITE อิ้ง)
                prereq_tokens = []

                while j < total:
                    curr = lines[j].strip()
                    if not curr:
                        j += 1
                        continue

                    if prereq_eng_key_regex.search(curr):
                        remainder = re.sub(prereq_eng_key_regex, "", curr).strip(": ").strip()
                        
                        if remainder:
                            if has_thai_regex.search(remainder):
                                break
                            else:
                                # 🟢 แก้ไขข้อ 2: คลีนข้อความผ่าน clean_ocr_en_text (เปลี่ยน L ให้เป็นเลข 1 ถ้าอยู่ท้ายคำ)
                                cleaned_rem = clean_ocr_en_text(remainder).upper()
                                if cleaned_rem:
                                    prereq_tokens.append(cleaned_rem)

                        j += 1

                        while j < total:
                            sub_line = lines[j].strip()
                            if not sub_line:
                                j += 1
                                continue

                            # 🛑 เจอภาษาไทยเมื่อไหร่ (บรรทัดคำอธิบายรายวิชา) = หยุดเก็บ Prerequisite ทันที!
                            if has_thai_regex.search(sub_line):
                                break

                            sub_upper = clean_ocr_en_text(sub_line).upper()
                            if len(sub_upper) > 0:
                                prereq_tokens.append(sub_upper)

                            j += 1

                        break

                    j += 1

                # สรุปค่า Prerequisite
                if prereq_tokens:
                    clean_prereq = " ".join(prereq_tokens).strip()
                    if clean_prereq in ["NONE", "ไม่มี", ""]:
                        prerequisite = "ไม่มี"
                    else:
                        prerequisite = clean_prereq
                else:
                    prerequisite = "ไม่มี"

                # 3. ข้ามบรรทัดเนื้อหาคำอธิบายวิชา เพื่อไปหารหัสวิชาถัดไป
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
                
                if code.startswith("90"):  # รหัส GenEd สถาบัน
                    courses.append(
                        {
                            "code": code,
                            "name_th": name_th if name_th else "ไม่ระบุ",
                            "name_en": name_en if name_en else "N/A",
                            "credits": credits,
                            "category": "หมวดวิชาศึกษาทั่วไป",
                            "type": "เลือก",
                            "prerequisite": None,
                            "flexible_year_semester": None,
                            "note": None,
                        }
                    )
                else:  # รหัสวิชาเฉพาะ / วิชาคณะ (06xxxxx)
                    courses.append(
                        {
                            "code": code,
                            "name_th": name_th if name_th else "ไม่ระบุ",
                            "name_en": name_en if name_en else "N/A",
                            "credits": credits,
                            "year": 0,
                            "semester": 0,
                            "category": "หมวดวิชาเฉพาะ",
                            "type": "เลือก",
                            "prerequisite": prerequisite,
                            "flexible_year_semester": "3/1, 3/2, 4/1",
                            "note": None,
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

        # 🟢 จุดที่แก้ไข 1: ดักจับโครงสร้าง "ตารางเรียน" ให้เด็ดขาด (มีคำว่า ปีที่/ภาคการศึกษาที่ หรือ มีรหัสวิชา+หน่วยกิตเป็นหัวตาราง)
        is_plan_page = bool(re.search(r"(?:ปีที่|ชั้นปีที่)\s*\d+", content_upper)) or \
                       (bool(re.search(r"รหัสวิชา", content_upper)) and bool(re.search(r"หน่วยกิต", content_upper)))

        if is_plan_page:
            return self.extract_from_lines(lines)

        # ถ้าไม่ใช่ตารางเรียน ค่อยมาเช็คว่าเป็นหน้าคำอธิบายรายวิชาหรือไม่
        is_description_page = bool(
            re.search(r"(?:คำอธิบายรายวิชา|COURSE\s*DESCRIPTION|PREREQUISITE|PRERE)", content_upper)
        )

        if is_description_page:
            return self.extract_descriptions(lines)

        return self.extract_from_lines(lines)