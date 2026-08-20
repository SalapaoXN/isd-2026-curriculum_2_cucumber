# isd-2026-curriculum_2_cucumber
We do OCR curriculum and some LLM with model name CUCUMBER

Project : P2 LLM ถาม-ตอบหลักสูตร

Member:
1. 67070049 Nattachai Kaewchum >> Discord: GoodDee
2. 67070063 Thanachin Chukiatchai >> Discord: วันลพ มีงบมาก
3. 67070103 Pongsakorn Panyacom >> Discord: เบบี๋คือดวงใจ


## Run guide:

### Step by step (full pipeline)
Run the whole pipeline in order: OCR → Extract → Merge (table + description) → Evaluate.

```bash
# 1. OCR a folder of images into outputs/ (per-page .txt + .json)
python cli.py inputs/dsba/ -o outputs

# 2. Extract structured courses from each OCR file
python extract.py outputs/ -o outputs

# 3. Merge plan table + course descriptions into a full file (saved in consolidated_outputs/)
python merge_consecutive.py --prefix dsba --plan coop -d 317-344

# 4. Evaluate against ground truth
python evaluate.py consolidated_outputs/merged_dsba_coop_full.json --gt ground_truth/DSBA/DSBA_academic_plan_coop.json
```

For the no-coop plan, replace the plan flag and ground truth accordingly:
```bash
python merge_consecutive.py --prefix dsba --plan no_coop -d 317-344
python evaluate.py consolidated_outputs/merged_dsba_no_coop_full.json --gt ground_truth/DSBA/DSBA_academic_plan_no_coop.json
```

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
# extract an explicitly supplied TXT file
python extract.py outputs/curriculum_page_016_ocr.txt

# explicitly supplied JSON files are also supported
python extract.py outputs/curriculum_page_016_ocr.json

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

When a folder contains matching `.txt` and `.json` files with the same stem, that OCR page is processed once and JSON is preferred. TXT-only and JSON-only files still work.

### Evaluate

```bash
python evaluate.py consolidated_outputs/merged_dsba_coop_full.json --gt ground_truth/DSBA/DSBA_academic_plan_coop.json
python evaluate.py consolidated_outputs/merged_dsba_no_coop_full.json --gt ground_truth/DSBA/DSBA_academic_plan_no_coop.json
```

### รัน Pipeline อัตโนมัติ (OCR -> Extract)
ใช้สำหรับรันประมวลผลรูปภาพเอกสารตามเลขหน้าที่กำหนด และสกัดออกมาเป็น JSON รายวิชาทันที

`--input-dir` is a filesystem path. The default is `inputs/dsba`; use paths such as `inputs/dsba` or `inputs/it` when running from the repository root.

โปรแกรมที่มีแผนการเรียน 2 แบบ (coop / no_coop) รันแบบนี้:
```bash
# DSBA
python -m src.run_pipeline -p 26-32 -i inputs/dsba --plan no_coop --program DSBA
python -m src.run_pipeline -p 33-39 -i inputs/dsba --plan coop --program DSBA
python -m src.run_pipeline -p 317-344 -i inputs/dsba --program DSBA

# IT
python -m src.run_pipeline -p 26-32 -i inputs/it --plan no_coop --program IT
python -m src.run_pipeline -p 33-39 -i inputs/it --plan coop --program IT

# BIT
python -m src.run_pipeline -p 26-32 -i inputs/bit --plan no_coop --program IT
python -m src.run_pipeline -p 33-39 -i inputs/bit --plan coop --program IT
```

โปรแกรมที่ไม่มีแผนการเรียน เขียนได้บรรทัดเดียว:
```bash
# AIT
python -m src.run_pipeline -p 23-26 -i inputs/ait --program AIT
python -m src.run_pipeline -p 287-302 -i inputs/ait --program AIT

# GENED
python -m src.run_pipeline -p 16-30 -i inputs/gened --plan gened --program DSBA

# RULE
python -m src.run_pipeline -p 1-13 -i inputs/rule --plan rule --program DSBA
```

### Merge (รวม table + description)
รวมไฟล์จาก output และควบรายวิชาจากตารางแผนการเรียนกับคำอธิบายรายวิชาเข้าเป็นไฟล์เดียว

โปรแกรมที่มีแผนการเรียน 2 แบบ (coop / no_coop):
```bash
python merge_consecutive.py --prefix dsba --plan coop -d 317-344
python merge_consecutive.py --prefix dsba --plan no_coop -d 317-344
python merge_consecutive.py --prefix it --plan coop -d 328-371
python merge_consecutive.py --prefix it --plan no_coop -d 328-371
python merge_consecutive.py --prefix bit --plan coop -d 328-371
python merge_consecutive.py --prefix bit --plan no_coop -d 328-371
```

โปรแกรมที่ไม่มีแผนการเรียน เขียนได้บรรทัดเดียว:
```bash
python merge_consecutive.py --prefix ait -d 287-302
python merge_consecutive.py --prefix gened
python merge_consecutive.py --prefix rule
```

ผลลัพธ์: ไฟล์ page-range ใช้รูปแบบ `merged_<group>_<plan>_page_<start>-<end>.json` และไฟล์เต็มใช้รูปแบบ `merged_<group>_<plan>_full.json` เช่น `merged_dsba_coop_full.json` จะถูกบันทึกใน `consolidated_outputs/`

การรวมหลักสูตรจะรักษารายการที่มีการลงซ้ำอย่างถูกต้องตามตำแหน่งและลำดับต้นฉบับ ไม่ลบรายการเพียงเพราะใช้รหัสวิชาเดียวกัน และจะรักษารายการที่ไม่มีรหัส, มีรหัสเป็น null หรือมีรหัสที่ใช้ค้นหาไม่ได้ไว้ด้วย รหัสวิชาที่ใช้ได้ยังสามารถรับข้อมูล metadata เพิ่มเติมได้

ไฟล์สรุปจาก `extract.py` ใช้รูปแบบ `consolidated_curriculum_<group>_<program>_<plan>.json` กลุ่มข้อมูลใช้ตัวระบุที่กำหนดแน่นอนและปลอดภัยต่อชื่อไฟล์ โดย path ปกติอย่าง `inputs/dsba` ยังคงใช้ชื่อสั้น `dsba`; path ที่มีอักขระต้องแปลง, Unicode, nested หรืออยู่นอก `inputs/` อาจมี digest ต่อท้ายเพื่อป้องกันชื่อซ้ำ
