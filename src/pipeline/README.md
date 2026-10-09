# src/pipeline — Data Pipeline

ส่วนนี้แปลงภาพหลักสูตรเป็น canonical curriculum data สำหรับสร้าง runtime database

## คำสั่งหลัก

ผู้ใช้เลือกเพียง **dataset** เดียว ระบบจะรู้เองว่าเป็นหลักสูตรอะไร ปีไหน แผนสหกิจ/ไม่สหกิจอยู่หน้าใด และคำอธิบายรายวิชาอยู่หน้าใด

```powershell
python -m src.pipeline.run --dataset it2560
```

ตัวอย่าง dataset ที่รองรับ:

```text
ait2566
bit2565
bit2560
dsba2565
dsba2560
gened2564
gened2557
it2565
it2560
```

ตัวอย่าง:

```powershell
python -m src.pipeline.run --dataset it2565
python -m src.pipeline.run --dataset it2560
python -m src.pipeline.run --dataset dsba2565
python -m src.pipeline.run --dataset dsba2560
python -m src.pipeline.run --dataset bit2565
python -m src.pipeline.run --dataset ait2566
```

ไม่ต้องระบุ `--plan`, `--pages`, `--catalog-key`, `--academic-year` หรือ `--output-dir` สำหรับการใช้งานปกติ ค่าเหล่านี้ถูกกำหนดแบบ deterministic ใน `src/pipeline/datasets.py`

ถ้าต้องการดูว่าจะรันอะไรโดยไม่ประมวลผลจริง:

```powershell
python -m src.pipeline.run --dataset it2560 --dry-run
```

ถ้าเครื่องไม่มี GPU:

```powershell
python -m src.pipeline.run --dataset it2560 --no-gpu
```

ถ้าต้องการสร้าง runtime DB ต่อหลังจบ pipeline:

```powershell
python -m src.pipeline.run --dataset it2560 --with-index
```

## Flow

```text
Source images
→ EasyOCR (TH + EN)
→ deterministic pre-clean
→ Extract structured curriculum fields
→ Merge study plan + course descriptions
→ Gemini proofreading เฉพาะชื่อวิชา
→ Final reviewed JSON
→ Evaluation (ถ้ามี Ground Truth ที่ตรงกัน)
→ Build Index (เมื่อใช้ --with-index)
```

## Dataset configuration

หน้า plan/description และ edition metadata อยู่ที่:

```text
src/pipeline/datasets.py
```

ตัวอย่าง IT 2560:

```text
dataset        = it2560
program        = IT
academic year  = 2560
no_coop        = pages 27-33
coop           = pages 34-40
description    = pages 222-269
catalog key    = it-2560
```

ดังนั้นคำสั่งเดียว:

```powershell
python -m src.pipeline.run --dataset it2560
```

จะ OCR เฉพาะหน้าที่ dataset นี้ต้องใช้ แล้วแยก plan ตอน Extract/Preparation เอง

## Inputs

รูปแบบที่แนะนำคือใช้ชื่อ dataset พร้อมปี:

```text
data/input/it2565/
data/input/it2560/
data/input/bit2565/
data/input/bit2560/
data/input/dsba2565/
data/input/dsba2560/
data/input/ait2566/
data/input/gened2564/
data/input/gened2557/
```

สำหรับ source ปัจจุบันที่ยังใช้ชื่อ folder เดิม (`it`, `bit`, `dsba`, `ait`, `gened`) ระบบยังรองรับเป็น fallback เพื่อไม่ให้ dataset เดิมพัง แต่ generated artifacts ใหม่จะใช้ชื่อแบบมีปี เช่น `it2565_page_032_ocr.json`

ชื่อ source image ต้องลงท้ายด้วยเลขหน้า 3 หลัก เช่น:

```text
it_page_032.png
it2560_page_027.png
```

ชื่อ source จริงจะถูกเก็บใน provenance แม้ generated output จะใช้ dataset key แบบมีปี

## Outputs

output root ถูกกำหนดตายตัวที่:

```text
data/output/
```

### OCR

```text
data/output/ocr/<dataset>/
```

เช่น:

```text
data/output/ocr/it2560/it2560_page_027_ocr.txt
data/output/ocr/it2560/it2560_page_027_ocr.json
```

JSON OCR เก็บทั้งข้อความและ provenance เช่น source filename, source page, document page, confidence/bounding box และขนาดภาพ

### Extracted

ถ้าใช้ `--keep-intermediates` จะเก็บไฟล์ per-page ที่:

```text
data/output/extracted/<dataset>/
```

### Consolidated ก่อน LLM

ไฟล์ที่รวม Study Plan + Course Description แล้ว แต่ยังไม่ผ่าน Gemini:

```text
data/output/consolidated/<dataset>/curriculum_<plan>.json
```

เช่น:

```text
data/output/consolidated/it2560/curriculum_no_coop.json
data/output/consolidated/it2560/curriculum_coop.json
```

### Final หลัง LLM correction

ไฟล์ canonical ที่ใช้ต่อกับระบบ:

```text
data/output/final/<dataset>/curriculum_<plan>.json
data/output/final/<dataset>/corrections_<plan>.json
data/output/final/<dataset>/manifest.json
```

เช่น:

```text
data/output/final/it2560/curriculum_no_coop.json
data/output/final/it2560/corrections_no_coop.json
data/output/final/it2560/curriculum_coop.json
data/output/final/it2560/corrections_coop.json
data/output/final/it2560/manifest.json
```

`curriculum_*.json` คือข้อมูลหลัง correction ส่วน `corrections_*.json` คือ audit log ว่าเปลี่ยนข้อความใดจากอะไรเป็นอะไร

## Gemini correction แก้อะไรบ้าง

Gemini ใช้ `gemini-3.5-flash-lite`, `temperature=0` และทำงานเป็น batch

แก้อัตโนมัติเฉพาะ:

```text
name_th
name_en
```

ไม่อนุญาตให้ Gemini แก้ factual fields เช่น:

```text
course code
credits
year
semester
prerequisite
desc_th
desc_en
```

`desc_th` และ `desc_en` ยังคงมาจาก deterministic extraction/merge เพราะเป็นข้อความ factual ยาว การให้ LLM rewrite โดยอัตโนมัติมีความเสี่ยงเปลี่ยนความหมายมากกว่าการ proofread ชื่อวิชา

ก่อนรับผล Gemini ระบบตรวจจำนวน record, unit index, field identity, empty replacement และ terminal numeric suffix และยังมี source-verified deterministic corrections สำหรับกรณีที่ตรวจต้นฉบับแล้ว

## Resume / Debug

ปกติให้เริ่มจาก OCR ด้วยคำสั่ง dataset เดียว

ถ้ามี intermediate ที่สร้างไว้แล้ว:

```powershell
python -m src.pipeline.run --dataset it2560 --from extracted
python -m src.pipeline.run --dataset it2560 --from consolidated
python -m src.pipeline.run --dataset it2560 --from corrected
```

เก็บ extracted intermediates สำหรับ debug:

```powershell
python -m src.pipeline.run --dataset it2560 --keep-intermediates
```

## Runtime DB

สร้างอย่างเดียวจาก final artifacts ที่มีอยู่แล้ว:

```powershell
python -m rag.build_index
```

หรือให้ pipeline สร้างต่อท้าย:

```powershell
python -m src.pipeline.run --dataset it2560 --with-index
```

runtime database:

```text
cucumber_outputs/runtime/curriculum.db
```

เมื่อมีทั้ง final layout ใหม่และ flat `*_corrected.json` แบบเก่า ตัว build index จะเลือก structured artifact ใหม่ก่อน และยังอ่าน legacy artifacts ของ dataset ที่ยังไม่ได้ regenerate เพื่อให้ migration ทำได้ทีละ dataset

## Rules pipeline

ข้อกำหนดสถาบันยังใช้ entry point แยก:

```powershell
python -m src.pipeline.run_rules
```

รายละเอียดดูที่ `docs/academic_rules.md`
