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
python evaluate.py consolidated_outputs/merged_coop_full.json --gt ground_truth/DSBA/DSBA_academic_plan_coop.json
```

For the no-coop plan, replace the plan flag and ground truth accordingly:
```bash
python merge_consecutive.py --prefix dsba --plan no_coop -d 317-344
python evaluate.py consolidated_outputs/merged_no_coop_full.json --gt ground_truth/DSBA/DSBA_academic_plan_no_coop.json
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
python evaluate.py consolidated_outputs/merged_coop_full.json --gt ground_truth/DSBA/DSBA_academic_plan_coop.json
python evaluate.py consolidated_outputs/merged_no_coop_full.json --gt ground_truth/DSBA/DSBA_academic_plan_no_coop.json
```

### รัน Pipeline อัตโนมัติ (OCR -> Extract)
ใช้สำหรับรันประมวลผลรูปภาพเอกสารตามเลขหน้าที่กำหนด และสกัดออกมาเป็น JSON รายวิชาทันที

โปรแกรมที่มีแผนการเรียน 2 แบบ (coop / no_coop) รันแบบนี้:
```bash
# DSBA
python -m src.run_pipeline -p 26-32 -i dsba --plan no_coop --program DSBA
python -m src.run_pipeline -p 33-39 -i dsba --plan coop --program DSBA
python -m src.run_pipeline -p 317-344 -i dsba --program DSBA

# IT
python -m src.run_pipeline -p 26-32 -i it --plan no_coop --program IT
python -m src.run_pipeline -p 33-39 -i it --plan coop --program IT

# BIT
python -m src.run_pipeline -p 26-32 -i bit --plan no_coop --program IT
python -m src.run_pipeline -p 33-39 -i bit --plan coop --program IT
```

โปรแกรมที่ไม่มีแผนการเรียน เขียนได้บรรทัดเดียว:
```bash
# AIT
python -m src.run_pipeline -p 23-26 -i ait --program AIT
python -m src.run_pipeline -p 287-302 -i ait --program AIT

# GENED
python -m src.run_pipeline -p 16-30 -i gened --plan gened --program DSBA

# RULE
python -m src.run_pipeline -p 1-13 -i rule --plan rule --program DSBA
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

ผลลัพธ์: ไฟล์ `merged_<plan>_full.json` (เช่น `merged_coop_full.json`) จะถูกบันทึกใน `consolidated_outputs/`