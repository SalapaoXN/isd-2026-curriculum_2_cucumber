'''Tranfrom json to text that easy to read'''
from json import loads
import re

'''Use json_to_text() to make JSON -> String'''

def is_valid(value):
    """เช็คว่าค่านี้ใช้ได้จริงหรือไม่ (กรองพวก null, ค่าว่าง, OCR พลาด)"""
    if value is None:
        return False
    #if value in (0 , '0'):
    #    return False
    if isinstance(value, str):
        # OCR อาจพลาดออกมาเป็น string ว่างๆ หรือคำที่แปลว่า "ไม่มีค่า"
        cleaned = value.strip().lower()
        if cleaned in {"", "null", "none", "-", "n/a", "nan"}:
            return False
    return True

def check_0(value):
    '''ตัว OCR มีปัญหาเล็กน้อยเรื่องจัวเลข'''
    if value in ('0' , 0):
        return False
    return True

PATTERN_KEEP_CHARS = re.compile(r'[^a-zA-Z0-9\u0E00-\u0E7F\s]')
PATTERN_SPACES = re.compile(r'\s+')
def clean_thai_text_fast(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace('\u200b', '')
    # 2. เรียกใช้ .sub() จาก Object ที่เราคอมไพล์ไว้แล้ว
    text = PATTERN_KEEP_CHARS.sub('', text)
    text = PATTERN_SPACES.sub(' ', text)
    return text.strip()



def json_to_text(item, code_to_name=None):
    parts = []
    # field บังคับ (ควรมีอยู่แล้วแทบทุกกรณี แต่เช็คไว้กันเหนียว)
    if is_valid(item.get("name_th")) and is_valid(item.get("name_en")):
        parts.append(f"วิชา {item['name_th']} ({item['name_en']})")
    elif is_valid(item.get("name_th")):
        parts.append(f"วิชา {item['name_th']}")

    if is_valid(item.get("code")):
        parts.append(f"รหัสวิชา {item['code']}")

    if is_valid(item.get("credits")):
        parts.append(f"หน่วยกิต {item['credits']}")

    year = item.get("year")
    if is_valid(year) and is_valid(item.get("semester")) and check_0(year):
        parts.append(f"เปิดสอนปีที่ {item['year']} ภาคเรียนที่ {item['semester']}")
    elif is_valid(year) and check_0(year):
        parts.append(f"เปิดสอนปีที่ {item['year']}")

    if is_valid(item.get("category")) and is_valid(item.get("type")):
        parts.append(f"เป็น{item['category']} ประเภทวิชา{item['type']}")
    elif is_valid(item.get("category")):
        parts.append(f"เป็น{item['category']}")

    if is_valid(item.get("prerequisite")):
        prereq_name = None
        if code_to_name:
            prereq_name = code_to_name.get(item["prerequisite"])
        parts.append(f"วิชาที่ต้องเรียนก่อน {prereq_name or item['prerequisite']}")

    if is_valid(item.get("flexible_year_semester")):
        parts.append(f"สามารถเรียนปี/เทอมที่ {item['flexible_year_semester']}")

    if is_valid(item.get("note")):
        parts.append(f"หมายเหตุ: {item['note']}")
    return " ".join(parts)

def json_to_desc(item):
    parts = []
    text_thai = item.get('desc_th')
    text_eng = item.get('desc_en')
    if text_thai is not None:
        text_thai = clean_thai_text_fast(text_thai)
        if is_valid(text_thai):
            parts.append(f"คำอธิบายรายวิชา: {text_thai}")
    elif text_eng is not None and is_valid(text_eng):
        parts.append(f"คำอธิบายรายวิชา: {clean_thai_text_fast(text_eng)}")
    return " ".join(parts)