# isd-2026-curriculum_2_cucumber
We do OCR curriculum and some LLM with model name CUCUMBER

Project : P2 LLM ถาม-ตอบหลักสูตร

Member:
1. 67070049 Nattachai Kaewchum >> Discord: GoodDee
2. 67070063 Thanachin Chukiatchai >> Discord: วันลพ มีงบมาก
3. 67070103 Pongsakorn Panyacom >> Discord: เบบี๋คือดวงใจ


## Run guide:

### OCR
```bash
# ocr each .jpg/.png file
python cli.py inputs/dsba/curriculum_page_016.jpg

# ocr entire folder
python cli.py inputs/dsba/

# -o outputs/..... (set specific output folder)
python cli.py inputs/dsba/ -o outputs/dsba_raw_ocr
```

### JSON Extraction
```bash
# extract each .txt file
python extract.py outputs/curriculum_page_016_ocr.txt

# extract entire folder
python extract.py outputs/
python extract.py outputs/dsba_raw_ocr

# full Customize
python extract.py outputs/curriculum_page_016_ocr.txt \
  --program DSBA \
  --plan coop \
  --output-dir outputs/extracted \
  --source "GT_Template-2.xlsx / Academic Plan GT — DSBA coop"
```

### Evaluate
DSBA no coop -> 31 - 36

```bash
python evaluate.py consolidated_outputs/consolidated_page_030-036.json --gt ground_truth/DSBA/DSBA_academic_plan_coop.json
```

### รัน Pipeline อัตโนมัติ (OCR ➔ Extract)
ใช้สำหรับรันประมวลผลรูปภาพเอกสารตามเลขหน้าที่กำหนด และสกัดออกมาเป็น JSON รายวิชาทันที

```bash
# รันแบบกำหนดช่วงหน้า (เช่น หน้า 32 ถึง 36)
python -m src.run_pipeline -p 32-36 inputs/dsba

# รันเฉพาะบางหน้า
python -m src.run_pipeline -p 16,18,20 inputs/dsba

# รันหน้าเดียว
python -m src.run_pipeline -p 16 inputs/dsba

# ตัวเลือกเพิ่มเติม (เปลี่ยนโฟลเดอร์ หรือ ปรับแผนการเรียน)
python -m src.run_pipeline -p 32-36 -i inputs/it -o outputs --plan no_coop
python -m src.run_pipeline -p 32-36 -i inputs/dsba -o outputs --plan no_coop
python -m src.run_pipeline -p 32-36 -i inputs/ait -o outputs --plan no_coop
python -m src.run_pipeline -p 32-36 -i inputs/bit -o outputs --plan no_coop
```

### Merge JSON
```bash
python merge_json.py -p 30-36

# รวมทุกไฟล์ที่มีในโฟลเดอร์ (ไม่ใส่ -p)
python merge_json.py
```