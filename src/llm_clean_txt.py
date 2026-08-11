# llm_cleaner.py
import os
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

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
        model="qwen2.5:3b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )

    cleaned_text = response.choices[0].message.content.strip()
    return cleaned_text.replace("```txt", "").replace("```", "").strip()