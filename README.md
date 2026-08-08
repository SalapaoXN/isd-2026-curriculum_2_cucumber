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

```bash
python evaluate.py consolidated_outputs/dsba_coop_full.json --gt ground_truth/DSBA/DSBA_academic_plan_coop.json
python evaluate.py consolidated_outputs/dsba_nocoop_full.json --gt ground_truth/DSBA/DSBA_academic_plan_no_coop.json
```

### รัน Pipeline อัตโนมัติ (OCR ➔ Extract)
ใช้สำหรับรันประมวลผลรูปภาพเอกสารตามเลขหน้าที่กำหนด และสกัดออกมาเป็น JSON รายวิชาทันที

```bash
# รันแบบกำหนดช่วงหน้า (เช่น หน้า 32 ถึง 36)
python -m src.run_pipeline -p 32-36 -i inputs/dsba

# ตัวเลือกเพิ่มเติม (เปลี่ยนโฟลเดอร์ หรือ ปรับแผนการเรียน)
python -m src.run_pipeline -p 23-29 -i inputs/dsba --plan no_coop
python -m src.run_pipeline -p 30-36 -i inputs/dsba --plan coop
python -m src.run_pipeline -p 151-224 -i inputs/dsba --plan gened
python -m src.run_pipeline -p 314-341 -i inputs/dsba --plan other
```

### Merge JSON
รวมไฟล์จาก output เป็น consolidate
```bash
python merge_json.py -p 30-36
```

### Consolidated
รวมไฟล์ consolidate เป็น full coop, nocoop
```bash
python src/consolidator.py -p 023-029 -d 314-341 -o consolidated_outputs/dsba_nocoop_full.json
python src/consolidator.py -p 030-036 -d 314-341 -o consolidated_outputs/dsba_coop_full.json
```