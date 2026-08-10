import json

def merge_duplicate_courses(courses):
    merged_dict = {}
    
    for course in courses:
        code = course.get('code')
        # เช็กกรณีเผื่อไฟล์ json บางตัวแปลง page เป็น list ไว้แล้ว
        page_val = course.get('page')
        page = page_val[0] if isinstance(page_val, list) else page_val
        
        if code not in merged_dict:
            new_course = course.copy()
            new_course['page'] = [page]
            merged_dict[code] = new_course
            
        else:
            existing_course = merged_dict[code]
            lowest_existing_page = min(existing_course['page'])
            
            # กรณีที่ 1: เจอหน้าที่เลขน้อยกว่า (ได้เป็นร่างต้นแบบใหม่)
            if page < lowest_existing_page:
                for key in course:
                    if key != 'page':
                        val_new = course.get(key)
                        val_old = existing_course.get(key)
                        
                        # เอาค่าจากหน้าใหม่มาใช้ ยกเว้นว่าหน้าใหม่เป็น null/ว่างเปล่า แล้วของเดิมมีข้อมูล
                        if (val_new is not None and val_new != ""):
                            existing_course[key] = val_new
                        else:
                            existing_course[key] = val_old
                            
            # กรณีที่ 2: เจอหน้าที่เลขมากกว่าหรือเท่ากับ (ร่างต้นแบบยังเป็นตัวเดิม)
            else:
                for key in course:
                    if key != 'page':
                        val_new = course.get(key)
                        val_old = existing_course.get(key)
                        
                        # ถ้าของเดิมเป็น null/ว่างเปล่า แต่หน้าที่เจอใหม่มีข้อมูล ให้ดึงมาเติมเต็ม
                        if (val_old is None or val_old == "") and (val_new is not None and val_new != ""):
                            existing_course[key] = val_new
                            
            # จัดการนำเลขหน้ามาต่อท้ายใน List
            if page not in existing_course['page']:
                existing_course['page'].append(page)
                existing_course['page'].sort()
                
    return list(merged_dict.values())

def process_json_file(input_filename, output_filename):
    try:
        # 1. อ่านข้อมูลจากไฟล์ JSON เดิม
        with open(input_filename, 'r', encoding='utf-8') as f:
            raw_courses = json.load(f)
            
        print(f"อ่านข้อมูลสำเร็จ: พบทั้งหมด {len(raw_courses)} รายการ")
        
        # 2. นำข้อมูลไปทำการ Merge
        merged_courses = merge_duplicate_courses(raw_courses)
        
        # 3. บันทึกข้อมูลที่ Merge แล้วลงไฟล์ JSON ใหม่
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(merged_courses, f, ensure_ascii=False, indent=2)
            
        print(f"Merge ข้อมูลสำเร็จ! เหลือ {len(merged_courses)} วิชา")
        print(f"บันทึกไฟล์ใหม่ไว้ที่: {output_filename}")
        
    except FileNotFoundError:
        print(f"ไม่พบไฟล์ชื่อ '{input_filename}' กรุณาตรวจสอบชื่อไฟล์และที่อยู่ให้ถูกต้อง")
    except Exception as e:
        print(f"เกิดข้อผิดพลาด: {str(e)}")

# ==========================================
# ส่วนเรียกใช้งาน (กำหนดชื่อไฟล์เข้า และ ไฟล์ออก)
# ==========================================
if __name__ == "__main__":
    # ชื่อไฟล์ JSON ต้นฉบับที่คุณมีอยู่
    INPUT_JSON = 'courses.json'
    
    # ชื่อไฟล์ JSON ใหม่ที่จะสร้างขึ้นมา
    OUTPUT_JSON = 'merged_courses.json'
    
    process_json_file(INPUT_JSON, OUTPUT_JSON)