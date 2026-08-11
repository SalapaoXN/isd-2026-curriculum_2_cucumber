# llm_clean_txt.py
import json
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

MODEL = "qwen2.5:3b"


def clean_ocr_text(raw_ocr_text: str) -> str:
    """ส่งข้อความดิบให้ LLM คลีนคำผิด แล้วส่งข้อความที่สะอาดกลับมา"""
    prompt = f"""Role: Computer Science Syllabus OCR Cleaner.
    Task: Fix Thai/English OCR typos, keep original lines.

    Rules:
    1. Fix English/Tech terms based on IT context.
    2. Fix Thai typos & credits format to x(x-x-x).
    3. Do NOT change course codes (numbers). Do NOT merge/split lines.
    4. COURSE CODE WILDCARDS: 
    - If course name is "วิชาเลือกเสรี...", enforce course code above it to be "XXXXXXXX".
    - If course code has partial 'X' (e.g. 06026XXX, 9064XXXX), preserve the 'X's.

    5. FIX OCR NUMBER CONFUSION:
    - Trailing letters 'L', 'l', '|', 'I' at the end of course names or on their own line after a course name are usually the number '1' or '2' (e.g., "CALCULUS L" -> "CALCULUS 1"). Fix them to Arabic numbers.

    6. BILINGUAL NUMBER ALIGNMENT:
    - Ensure course sequence numbers (1, 2, 3, etc.) are synchronized between Thai and English names.
    - If one language has a suffix number (e.g., "CALCULUS 1") but the other is missing it (e.g., "แคลคูลัส"), ADD the corresponding number to the missing language so both match (e.g., "แคลคูลัส 1").

    7. Return ONLY cleaned text. No chat, no markdown.

    Text:
    {raw_ocr_text}"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )

    cleaned_text = response.choices[0].message.content.strip()
    return cleaned_text.replace("```txt", "").replace("```", "").strip()


def _extract_json_response(content: str) -> Any:
    """Parse JSON out of an LLM reply, tolerating markdown fences."""
    text = content.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def clean_extracted_courses(courses: List[Dict]) -> List[Dict]:
    """
    Post-extraction OCR fixer.

    Takes the extractor's structured course list and lets the LLM fix typos
    in the text fields (code, name_th, name_en, credits, prerequisite) while
    preserving every other key (year, semester, category, type, ...).

    Works on the compact structured output instead of the whole raw page,
    so it is fast and precise. On any LLM/parse failure the original list is
    returned unchanged.
    """
    if not courses:
        return courses

    cleanable = [
        {
            "index": idx,
            "code": c.get("code", ""),
            "name_th": c.get("name_th", ""),
            "name_en": c.get("name_en", ""),
            "credits": c.get("credits", ""),
            "prerequisite": c.get("prerequisite", ""),
        }
        for idx, c in enumerate(courses)
    ]

    prompt = f"""Role: Thai University Computer Science Curriculum Cleaner.
Task: Fix OCR typos in the extracted course JSON below.

Rules:
1. COURSE CODE: must be exactly 8 digits starting with "06" (e.g. "06066100").
   - If a leading digit was dropped by OCR (e.g. "6066100" -> "06066100"), restore it.
   - If a code keeps 'X' wildcards (e.g. "06026XXX"), preserve the 'X's and do not invent digits.
   - Do NOT change codes that are already valid 8-digit codes.
2. NAME_TH: fix Thai OCR typos using Thai curriculum context (e.g. "เพือ" -> "เพื่อ", "แคลคูลส" -> "แคลคูลัส", "ไม่ต่อเนือง" -> "ไม่ต่อเนื่อง").
3. NAME_EN: fix English / IT-term OCR typos (e.g. "LIEAR ALGEBRA" -> "LINEAR ALGEBRA", "COMPUTER PROGRAMMNG" -> "COMPUTER PROGRAMMING", "ITRODUCTION" -> "INTRODUCTION", "CYBERSECURITV" -> "CYBERSECURITY", "PROBABILITV" -> "PROBABILITY").
4. CREDITS: must be format X(X-X-X) (e.g. "3(3-0-6)"). Normalize separators if OCR mangled them.
5. PREREQUISITE: fix typos in course codes and English terms. Keep "ไม่มี" / "NONE" as-is.
6. Do NOT reorder, add, or remove entries. Keep the same number of items and same "index".
7. Return ONLY valid JSON: an array of objects with keys "index", "code", "name_th", "name_en", "credits", "prerequisite". No chat, no markdown.

Input:
{json.dumps(cleanable, ensure_ascii=False, indent=2)}"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        content = response.choices[0].message.content.strip()
        fixed_list = _extract_json_response(content)
        if not isinstance(fixed_list, list):
            return courses

        fixed_by_index = {item.get("index"): item for item in fixed_list if isinstance(item, dict)}

        result = []
        for idx, course in enumerate(courses):
            fixed = fixed_by_index.get(idx)
            if not fixed:
                result.append(course)
                continue
            merged = dict(course)
            for key in ("code", "name_th", "name_en", "credits", "prerequisite"):
                val = fixed.get(key)
                if isinstance(val, str) and val.strip() and key in course:
                    merged[key] = val.strip()
            result.append(merged)
        return result
    except Exception as exc:  # noqa: BLE001 - never let LLM failures break the pipeline
        print(f" [LLM clean] post-extraction cleanup failed, keeping original: {exc}")
        return courses
