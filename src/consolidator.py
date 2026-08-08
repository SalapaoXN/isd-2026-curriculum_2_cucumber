import argparse
import json
from pathlib import Path
from typing import Dict, Union


class CurriculumConsolidator:
    def __init__(self, plan_data: Dict, description_data: Dict):
        """
        :param plan_data: JSON Data จากหน้าตารางเรียน (ที่มี list ของ courses)
        :param description_data: JSON Data จากหน้าคำอธิบายรายวิชา (ที่มี list ของ descriptions หรือ courses)
        """
        self.plan_data = plan_data
        self.description_data = description_data

    def consolidate(self) -> Dict:
        # รองรับทั้งคีย์ "descriptions" หรือ "courses" ในไฟล์คำอธิบาย
        descriptions = self.description_data.get("descriptions") or self.description_data.get("courses", [])
        
        # 1. ทำ Index Lookup จากไฟล์คำอธิบายรายวิชา
        desc_lookup = {}
        for desc in descriptions:
            code = desc.get("code")
            if code:
                desc_lookup[code] = desc

        consolidated_courses = []
        processed_codes = set()  # เก็บ code ที่จัดการไปแล้ว เพื่อป้องกันวิชาซ้ำ

        # 2. วนลูปวิชาจากแผนการเรียน (Plan Data) เป็นหลัก
        for course in self.plan_data.get("courses", []):
            course_code = course.get("code", "")
            
            # คัดลอก Properties ทั้งหมดของวิชาเดิมจาก plan เพื่อไม่ให้ฟิลด์ใดๆ หล่นหาย
            merged_course = course.copy()

            # กรณีที่ 2.1: รหัสวิชาเดี่ยวๆ (ไม่มีคำว่า "หรือ")
            if "หรือ" not in course_code and course_code in desc_lookup:
                target_desc = desc_lookup[course_code]
                
                # [แก้ใหม่]: ไม่อัปเดตทับทั้งหมด เพื่อรักษา flexible_year_semester และข้อมูลเดิมไว้
                # อัปเดตเฉพาะ prerequisite (และ type/desc กรณีที่มี)
                if "prerequisite" in target_desc:
                    merged_course["prerequisite"] = target_desc["prerequisite"]
                    
                if "desc_th" in target_desc:
                    merged_course["desc_th"] = target_desc["desc_th"]
                    
                if "desc_en" in target_desc:
                    merged_course["desc_en"] = target_desc["desc_en"]

                processed_codes.add(course_code)

            # กรณีที่ 2.2: รหัสวิชาแบบวิชาเลือกคู่ เช่น "06026259 หรือ 06026260"
            elif "หรือ" in course_code:
                sub_codes = [c.strip() for c in course_code.split("หรือ")]
                th_list = []
                en_list = []
                
                for sub_code in sub_codes:
                    if sub_code in desc_lookup:
                        target_desc = desc_lookup[sub_code]
                        if target_desc.get("desc_th"):
                            th_list.append(target_desc.get("desc_th"))
                        if target_desc.get("desc_en"):
                            en_list.append(target_desc.get("desc_en"))
                        processed_codes.add(sub_code)

                if th_list:
                    merged_course["desc_th"] = "\n".join(th_list)
                if en_list:
                    merged_course["desc_en"] = "\n".join(en_list)
                processed_codes.add(course_code)

            else:
                if course_code:
                    processed_codes.add(course_code)

            # -----------------------------------------------------
            # เซ็ต default ป้องกัน schema พัง กรณี field นี้ไม่มีตั้งแต่แรก
            # ถ้ามีอยู่แล้ว ค่าจะไม่ถูกเปลี่ยน
            merged_course.setdefault("flexible_year_semester", None)
            # -----------------------------------------------------

            consolidated_courses.append(merged_course)

        # 3. ดึงวิชาที่มีเฉพาะในคำอธิบายรายวิชา (วิชาเลือก/วิชาใหม่) เพิ่มต่อท้าย
        for code, desc_item in desc_lookup.items():
            if code not in processed_codes:
                new_elective_course = desc_item.copy()
                
                # กำหนดค่า default สำหรับวิชาเลือกถ้าไม่มีระบุไว้
                new_elective_course.setdefault("year", 0)
                new_elective_course.setdefault("semester", 0)
                new_elective_course.setdefault("category", "หมวดวิชาเฉพาะ")
                new_elective_course.setdefault("type", "เลือก")
                new_elective_course.setdefault("prerequisite", "ไม่มี")
                
                new_elective_course.setdefault("flexible_year_semester", None)
                
                consolidated_courses.append(new_elective_course)
                processed_codes.add(code)

        # 4. ประกอบข้อมูล Output
        return {
            "source": self.plan_data.get("source", "Merged Academic Plan & Course Descriptions"),
            "description": self.plan_data.get("description", "Ground Truth รายวิชาหลักสูตร"),
            "program": self.plan_data.get("program", ""),
            "plan": self.plan_data.get("plan", ""),
            "total_courses": len(consolidated_courses),
            "courses": consolidated_courses,
        }


def save_json(data: Dict, output_path: Union[str, Path]):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f" บันทึกไฟล์สำเร็จ: {output_path} (รวมทั้งหมด {data.get('total_courses', 0)} รายวิชา)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consolidate Plan and Description JSON files by page ranges.")
    
    # 1. รับช่วงเลขหน้าของตารางเรียน (Plan Pages)
    parser.add_argument(
        "-p", "--plan-pages", 
        required=True, 
        help="ช่วงหน้าของตารางเรียน เช่น 030-036 หรือ 023-029"
    )
    
    # 2. รับช่วงเลขหน้าของคำอธิบายรายวิชา (Desc Pages)
    parser.add_argument(
        "-d", "--desc-pages", 
        required=True, 
        help="ช่วงหน้าของคำอธิบายรายวิชา เช่น 314-341"
    )
    
    # 3. ตั้งชื่อไฟล์ Output
    parser.add_argument(
        "-o", "--output", 
        default=None, 
        help="ชื่อไฟล์ Output หรือ Path ที่ต้องการเซฟ"
    )

    # 4. ระบุไฟล์ GENED (เช่น consolidated_page_151-224.json) เพื่อซิงค์หมวดหมู่วิชา GENED
    #    ถ้าไม่ระบุ จะค้นหาไฟล์ consolidated_page_*.json ที่เหลือใน consolidated_outputs ให้อัตโนมัติ
    parser.add_argument(
        "--gened",
        default=None,
        help="GENED consolidated JSON (เช่น consolidated_page_151-224.json) สำหรับซิงค์หมวดหมู่เข้าสู่ไฟล์ Output",
    )

    args = parser.parse_args()

    #  กำหนด Base Folder
    base_dir = Path("consolidated_outputs")
    
    plan_file = base_dir / f"consolidated_page_{args.plan_pages}.json"
    desc_file = base_dir / f"consolidated_page_{args.desc_pages}.json"

    if not plan_file.exists():
        plan_file = base_dir / f"page_{args.plan_pages}.json"
    if not desc_file.exists():
        desc_file = base_dir / f"page_{args.desc_pages}.json"

    if not plan_file.exists():
        raise FileNotFoundError(f" ไม่พบไฟล์ตารางเรียน: {plan_file}")
    if not desc_file.exists():
        raise FileNotFoundError(f" ไม่พบไฟล์คำอธิบายรายวิชา: {desc_file}")

    print(f" กำลังโหลดไฟล์ตารางเรียน: {plan_file.name}")
    print(f" กำลังโหลดไฟล์คำอธิบาย: {desc_file.name}")

    with open(plan_file, "r", encoding="utf-8") as f:
        plan_data = json.load(f)

    with open(desc_file, "r", encoding="utf-8") as f:
        desc_data = json.load(f)

    # รวมร่าง
    consolidator = CurriculumConsolidator(plan_data, desc_data)
    final_result = consolidator.consolidate()

    #  กำหนด Output Path อัตโนมัติถ้าไม่ได้ระบุ -o
    if args.output:
        output_path = Path(args.output)
        if not output_path.suffix:
            output_path = output_path / f"consolidated_page_{args.plan_pages}.json"
    else:
        output_path = base_dir / f"final_consolidated_page_{args.plan_pages}.json"

    save_json(final_result, output_path)

    # 5. ซิงค์หมวดหมู่วิชา GENED เข้าสู่ไฟล์ Output (รันอัตโนมัติ)
    #    ต้องรันหลังจาก consolidate เรียบร้อย และก่อนนำไฟล์ไป evaluate
    #    - ถ้าระบุ --gened ใช้ไฟล์นั้นโดยตรง
    #    - ถ้าไม่ระบุ จะค้นหาไฟล์ consolidated_page_*.json ที่เหลือใน base_dir เป็นแหล่งหมวดหมู่
    if args.gened:
        gened_sources = [Path(args.gened)]
        if not gened_sources[0].exists():
            gened_sources[0] = base_dir / args.gened
        if not gened_sources[0].exists():
            raise FileNotFoundError(f" ไม่พบไฟล์ GENED: {gened_sources[0]}")
    else:
        gened_sources = [
            f
            for f in base_dir.glob("consolidated_page_*.json")
            if f not in (plan_file, desc_file, output_path)
        ]

    if gened_sources:
        from transfer import sync_categories

        names = ", ".join(f.name for f in gened_sources)
        print(f" กำลังซิงค์หมวดหมู่ GENED จาก: {names}")
        sync_categories(gened_sources, output_path, output_path)
    else:
        print(" ไม่พบไฟล์ GENED สำหรับซิงค์หมวดหมู่ (ข้ามขั้นตอน)")