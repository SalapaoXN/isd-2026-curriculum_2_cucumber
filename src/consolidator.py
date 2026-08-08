import json
import argparse
from pathlib import Path
from typing import Dict, List

class CurriculumConsolidator:
    """
    คลาสสำหรับจัดการรวมข้อมูล (Merge) ระหว่างโครงสร้างตารางเรียน และ คำอธิบายรายวิชา
    """
    
    @staticmethod
    def _is_valid_value(value) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            invalid_words = {"", "ไม่มี", "NONE", "NULL", "N/A", "ไม่ระบุ"}
            if value.strip().upper() in invalid_words:
                return False
        return True

    @staticmethod
    def consolidate(plan_data: Dict, desc_data: Dict, fields_to_update: List[str] = None) -> Dict:
        if fields_to_update is None:
            fields_to_update = ["prerequisite", "name_th", "name_en"]
            
        desc_lookup = {c["code"]: c for c in desc_data.get("courses", [])}
        merged_courses = []
        seen_codes = set()

        for course in plan_data.get("courses", []):
            code = course["code"]
            course_copy = dict(course)
            
            if code in desc_lookup:
                desc_course = desc_lookup[code]
                for field in fields_to_update:
                    if field in desc_course:
                        new_value = desc_course[field]
                        if CurriculumConsolidator._is_valid_value(new_value):
                            course_copy[field] = new_value

            merged_courses.append(course_copy)
            seen_codes.add(code)

        for course in desc_data.get("courses", []):
            code = course["code"]
            if code not in seen_codes:
                merged_courses.append(course)
                seen_codes.add(code)

        return {
            "source": plan_data.get("source"),
            "description": str(plan_data.get("description", "")) + " (Consolidated Phase)",
            "program": plan_data.get("program"),
            "plan": plan_data.get("plan"),
            "courses": merged_courses
        }

    @staticmethod
    def merge_json_files(plan_filepath: str, desc_filepath: str, output_filepath: str, fields_to_update: List[str] = None):
        with open(Path(plan_filepath), 'r', encoding='utf-8') as f:
            plan_data = json.load(f)
            
        with open(Path(desc_filepath), 'r', encoding='utf-8') as f:
            desc_data = json.load(f)
            
        merged_data = CurriculumConsolidator.consolidate(
            plan_data=plan_data, 
            desc_data=desc_data, 
            fields_to_update=fields_to_update
        )
        
        # สร้างโฟลเดอร์ปลายทางอัตโนมัติหากยังไม่มี
        output_path = Path(output_filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=4)
        
        print(f"✅ บันทึกไฟล์ที่รวมข้อมูลสำเร็จแล้วที่: {output_filepath}")
        print(f"🔄 ฟิลด์ที่ดึงไปอัปเดต: {fields_to_update if fields_to_update else ['prerequisite', 'name_th', 'name_en']}")
        print(f"📊 จำนวนรายวิชาทั้งหมดหลังรวม: {len(merged_data['courses'])} วิชา")

# ==========================================
# ส่วนรองรับการเรียกใช้งานผ่าน Command Line
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="เครื่องมือสำหรับรวมข้อมูล JSON จากตารางเรียนและคำอธิบายรายวิชาเข้าด้วยกัน")
    
    # กำหนด Flag ที่ต้องการรับจากผู้ใช้
    parser.add_argument('-b', '--base', type=str, required=True, help="เส้นทางไฟล์ JSON ของตารางเรียน (Base)")
    parser.add_argument('-d', '--desc', type=str, required=True, help="เส้นทางไฟล์ JSON ของคำอธิบายรายวิชา (Data Source)")
    parser.add_argument('-o', '--output', type=str, required=True, help="เส้นทางและชื่อไฟล์ JSON ปลายทาง (Output)")
    parser.add_argument('-f', '--fields', type=str, nargs='+', default=["prerequisite", "name_th", "name_en"],
                        help="ระบุฟิลด์ที่ต้องการเขียนทับ (เว้นวรรค) เช่น prerequisite name_th credits (ค่าเริ่มต้น: prerequisite name_th name_en)")

    args = parser.parse_args()

    print("🚀 [CONSOLIDATOR] เริ่มต้นการรวมไฟล์...")
    CurriculumConsolidator.merge_json_files(
        plan_filepath=args.base,
        desc_filepath=args.desc,
        output_filepath=args.output,
        fields_to_update=args.fields
    )