import json

def sync_categories_by_file(gened_filepath, dsba_filepath, output_filepath):
    # 1. โหลดข้อมูลจากไฟล์ gened (ไฟล์ต้นทาง)
    with open(gened_filepath, 'r', encoding='utf-8') as f:
        gened_data = json.load(f)

    # 2. โหลดข้อมูลจากไฟล์ dsba (ไฟล์ปลายทาง)
    with open(dsba_filepath, 'r', encoding='utf-8') as f:
        dsba_data = json.load(f)

    # 3. สร้าง Mapping รหัสวิชา -> หมวดหมู่ จากไฟล์ gened ทั้งหมด
    gened_map = {}
    for course in gened_data.get('courses', []):
        code = course.get('code')
        if code:
            gened_map[code] = course.get('category', '')

    # 4. อัปเดต category ในไฟล์ dsba ทันทีที่รหัสวิชา (code) ตรงกัน
    update_count = 0
    for course in dsba_data.get('courses', []):
        code = course.get('code')
        
        # ถ้ารหัสวิชานี้มีอยู่ในไฟล์ gened ให้ทำการแทนที่ category
        if code in gened_map:
            course['category'] = gened_map[code]
            update_count += 1

    # 5. บันทึกข้อมูลที่อัปเดตแล้วลงไฟล์ใหม่
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(dsba_data, f, ensure_ascii=False, indent=4)

    print(f"ดำเนินการเสร็จสิ้น! ทำการแมพและอัปเดตหมวดหมู่ไปทั้งหมด {update_count} รายวิชา")

# --- วิธีการเรียกใช้งาน ---
# ใส่แค่ชื่อไฟล์ 3 ตัว: 1. ไฟล์ต้นทาง 2. ไฟล์ที่จะแก้ 3. ชื่อไฟล์ใหม่ที่ได้
# นำเครื่องหมาย # ออกเพื่อรันคำสั่ง
sync_categories_by_file('consolidated_outputs/consolidated_page_151-224.json', 'consolidated_outputs/dsba_coop_full.json', 'dsba_coop_full_updated.json')