import re
import json

def extract_courses_to_json(input_file, output_file):
    # กำหนด Regex Patterns พื้นฐาน
    page_pattern = re.compile(r'=\s*Page:\s*(?:page)?(\d+)\.png\s*=')
    year_sem_pattern = re.compile(r'ปีที่\s*([1-4])\s*ภาคการศึกษาที่\s*([1-2])')
    course_start_pattern = re.compile(r'^((?:060|906)\d{5})(?:\s+(.+))?$')
    prereq_pattern = re.compile(r'(?:ราย)?วิชาบังคับก่อน\s*:\s*(.+)$')
    credits_pattern = re.compile(r'(\d\s*\(\d+\s*-\s*\d+\s*-\s*\d+\))')
    
    # Regex สำหรับดึงข้อมูลจากตาราง HTML
    tr_pattern = re.compile(r'<tr[^>]*>(.*?)</tr>', re.IGNORECASE | re.DOTALL)
    td_pattern = re.compile(r'<td[^>]*>(.*?)</td>', re.IGNORECASE | re.DOTALL)

    current_page = None
    current_year = None
    current_semester = None
    
    courses = []
    current_course = None
    expecting_name = 0 
    
    # ตัวแปรสำหรับอ่านตาราง HTML
    in_table = False
    table_buffer = ""

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # 1. ตรวจสอบการขึ้นหน้าใหม่
        m_page = page_pattern.search(line)
        if m_page:
            if current_course:
                courses.append(current_course)
                current_course = None
                expecting_name = 0
            
            current_page = int(m_page.group(1))
            
            # === รีเซ็ตปีและเทอมเป็น None เสมอเมื่อเริ่มหน้าใหม่ ===
            current_year = None
            current_semester = None
            continue
            
        # 2. ตรวจสอบปีและเทอม (หากมีประกาศในหน้านี้ จะอัปเดตข้อมูล)
        m_ys = year_sem_pattern.search(line)
        if m_ys:
            current_year = m_ys.group(1)
            current_semester = m_ys.group(2)
            continue
            
        # 3. จัดการข้อมูลรูปแบบตาราง HTML
        if "<table" in line.lower() or in_table:
            in_table = True
            table_buffer += line + " " # เก็บข้อมูลตารางไว้จนกว่าจะเจอ </table>
            
            if "</table>" in line.lower():
                in_table = False
                
                # บันทึกวิชาธรรมดาที่อาจจะค้างอยู่ก่อนเข้าตาราง
                if current_course:
                    courses.append(current_course)
                    current_course = None
                    expecting_name = 0
                    
                # เริ่มสกัดข้อมูลจากแต่ละแถวในตาราง
                rows = tr_pattern.findall(table_buffer)
                for row in rows:
                    cols = td_pattern.findall(row)
                    if not cols or len(cols) < 3:
                        continue # ข้ามแถวที่คอลัมน์ไม่ครบ (เช่น ข้ามแถวเปล่า)
                    
                    code_col = re.sub(r'<[^>]+>', '', cols[0]).strip()
                    
                    # ตรวจสอบว่าคอลัมน์แรกเป็นรหัสวิชาหรือไม่ (ข้ามพวกคำว่า "รวม" หรือ "รหัสวิชา" ที่เป็น Header)
                    if re.match(r'^(?:060|906)\d{5}$', code_col):
                        name_col = cols[1]
                        credit_col = re.sub(r'<[^>]+>', '', cols[2]).strip()
                        
                        # แยกชื่อภาษาไทยและอังกฤษ โดยอาศัยแท็ก <br/> ที่กั้นกลาง
                        names = re.split(r'<br\s*/?>', name_col, flags=re.IGNORECASE)
                        name_th = re.sub(r'<[^>]+>', '', names[0]).strip()
                        name_en = ""
                        
                        if len(names) > 1:
                            name_en = re.sub(r'<[^>]+>', '', names[1]).strip()
                        else:
                            # กรณีไม่มี <br/> ให้แยกด้วยการหาตัวอักษรภาษาอังกฤษที่ปนมา
                            match = re.search(r'([A-Za-z]+.*)', name_th)
                            if match:
                                name_en = match.group(1).strip()
                                name_th = name_th.replace(name_en, '').strip()
                        
                        courses.append({
                            "code": code_col,
                            "name_th": name_th,
                            "name_en": name_en,
                            "credits": credit_col.replace(" ", ""), # เอาระยะห่างออก
                            "year": current_year,
                            "semester": current_semester,
                            "category": None,
                            "type": None,
                            "prerequisite": "ไม่มี", 
                            "flexible_year_semester": None,
                            "note": None,
                            "page": current_page
                        })
                
                table_buffer = "" # ล้างบัฟเฟอร์ตารางเพื่อรอรับตารางถัดไป
            continue # ถ้ากำลังจัดการตารางอยู่ ให้ข้ามลอจิกบรรทัดธรรมดาไปเลย
            
        # 4. จัดการข้อมูล Text ธรรมดา (สำหรับวิชาที่ไม่ได้อยู่ในตาราง)
        clean_line = re.sub(r'<[^>]+>', '', line)
        clean_line = clean_line.replace('**', '').strip()
        
        m_course = course_start_pattern.search(clean_line)
        if m_course:
            if current_course:
                courses.append(current_course)
                
            code = m_course.group(1)
            raw_name = m_course.group(2)
            raw_name_th = raw_name.strip() if raw_name else ""
            course_credit = ""
            
            if raw_name_th:
                m_inline_credit = credits_pattern.search(raw_name_th)
                if m_inline_credit:
                    course_credit = m_inline_credit.group(1).replace(" ", "")
                    raw_name_th = raw_name_th.replace(m_inline_credit.group(0), "").strip()
                    
            current_course = {
                "code": code,
                "name_th": raw_name_th,
                "name_en": "",
                "credits": course_credit,
                "year": current_year,
                "semester": current_semester,
                "category": None, 
                "type": None,     
                "prerequisite": "ไม่มี", 
                "flexible_year_semester": None,
                "note": None,
                "page": current_page
            }
            
            if not raw_name_th:
                expecting_name = 1
            else:
                expecting_name = 2
            continue
            
        if current_course:
            m_prereq = prereq_pattern.search(clean_line)
            if m_prereq:
                current_course["prerequisite"] = m_prereq.group(1).strip()
                continue
                
            m_credit = credits_pattern.search(clean_line)
            if m_credit and not current_course["credits"]:
                current_course["credits"] = m_credit.group(1).replace(" ", "")
                clean_text = clean_line.replace(m_credit.group(0), "").strip()
                if not clean_text:
                    continue
                clean_line = clean_text 

            if expecting_name == 1 and not clean_line.startswith("PREREQUISITE"):
                if re.match(r'^[A-Za-z0-9\s\W]+$', clean_line):
                    current_course["name_en"] = clean_line
                else:
                    current_course["name_th"] = clean_line
                expecting_name = 2
                continue
                
            elif expecting_name == 2 and not clean_line.startswith("PREREQUISITE"):
                if not current_course["name_en"]:
                     current_course["name_en"] = clean_line
                expecting_name = 0
                continue

    if current_course:
        courses.append(current_course)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(courses, f, ensure_ascii=False, indent=2)
        
    print(f"แปลงข้อมูลสำเร็จ! ได้ทั้งหมด {len(courses)} วิชา บันทึกลงในไฟล์ {output_file}")

# เรียกใช้งาน
extract_courses_to_json('output.txt', 'courses.json')