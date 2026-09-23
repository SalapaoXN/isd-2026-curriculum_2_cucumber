# src/pipeline — สายการผลิตข้อมูลหลักสูตร

## ภาพรวม
ส่วนนี้ทำหน้าที่แปลงภาพหน้าหนังสือหลักสูตรให้กลายเป็น JSON
มาตรฐานที่ผ่านการตรวจแก้แล้ว และสร้างฐานข้อมูล SQLite สำหรับ
ระบบถาม-ตอบต่อไป ข้อมูลกฎระเบียบของสถาบัน (ภาพ `rule/`) ใช้ทาง
เข้าอีกสายหนึ่งคือ `run_rules.py`

## ข้อมูลนำเข้า
- ภาพหน้าหนังสือ: `data/input/<program>/<program>_page_NNN.png`
  (`ait`, `bit`, `dsba`, `gened`, `it`) และ `data/input/rule/`
  (`rule_page_001.png` … `rule_page_013.png` พร้อมไฟล์ต้นทางของ
  program requirement) สำหรับสาย rules
- ไฟล์ตั้งค่า: `config/pipeline.yaml`, `config/programs.yaml`
  ส่วนความลับต่าง ๆ เก็บในไฟล์ `.env` ที่ root

## กระบวนการทำงาน
คำสั่ง `python -m src.pipeline.run` ทำงานตามลำดับดังนี้

```text
ocr → extract → merge → correct → (evaluate) → (build_index)
```

1. **ocr** (`tools/ocr/`) — อ่านตัวอักษรจากภาพด้วย EasyOCR
   พร้อมบันทึก metadata ของแต่ละหน้า
2. **extract** (`tools/extraction/engine.py` รวมทั้ง `rules.py` /
   `program_requirements.py` สำหรับสาย rules) — สกัดรายวิชา,
   แผนการจัดวาง, หน่วยกิต และวิชาบังคับก่อน
3. **merge** (`tools/merge/consolidator.py`, `policy.py`) — รวม
   ข้อมูลให้เป็นระเบียนเดียวต่อ program/plan
4. **correct** (`tools/correction/`) — ใช้ LLM ช่วยแก้คำที่ OCR
   อ่านผิด พร้อมบันทึก log การแก้ไขทุกจุด
5. **evaluate** (`tools/evaluation/`) — ตรวจคุณภาพเป็นชั้น ๆ และ
  ออกรายงาน (ข้ามได้ด้วย `--skip-eval`)
6. **build_index** — ทำเฉพาะเมื่อใส่ `--with-index` โดยโหลด
   `data/output/final/*_corrected.json` ลง `curriculum.db`

## ผลลัพธ์
- `data/output/final/*_corrected.json` (คู่กับ
  `*_corrections.json`) — ชุดข้อมูลหลัก (corpus) ที่เป็นมาตรฐาน
  และเป็น input เพียงอย่างเดียวที่ `rag.build_index` ใช้
- `cucumber_outputs/runtime/curriculum.db` — เมื่อสั่งสร้าง index
- ไฟล์ระหว่างทางจะอยู่แค่ใน temp แล้วลบทิ้ง ยกเว้นใส่
  `--keep-intermediates` ซึ่งจะเก็บ `extracted/` + `consolidated/`
  ไว้ใต้ `--output-dir`

## องค์ประกอบหลัก
- `run.py` — CLI และลำดับ stage ของสายหลักสูตร
- `run_rules.py` — CLI ของสายกฎระเบียบ/ข้อกำหนดของหลักสูตร
- `config.py`, `models.py` — การแยก program/plan และโครงระเบียน
- `tools/preparation/`, `tools/ocr/`, `tools/extraction/`,
  `tools/merge/`, `tools/correction/`, `tools/evaluation/`,
  `tools/indexing/` — ดูแลทีละ stage ส่วน `utils/` เป็น helper
  กลาง

## วิธีใช้งาน
```powershell
python -m src.pipeline.run --program it --dry-run
python -m src.pipeline.run --program it --with-index
python -m src.pipeline.run --program it --pages 32-38
python -m src.pipeline.run --program it --no-gpu
python -m src.pipeline.run --program it --from corrected --with-index
python -m src.pipeline.run --only-index
```
`--program` เลือกได้: `ait`, `bit`, `dsba`, `gened`, `it`
(ค่าเริ่มต้นคือ `it`) ส่วน `--from` เลือกได้: `ocr`,
`extracted`, `consolidated`, `corrected`

## ข้อจำกัด
- การรันเต็มรูปแบบต้องใช้ dependency ของ OCR/LLM และต้องมี
  API credentials สำหรับขั้น correction ส่วน `--dry-run` แสดง
  แค่ config/stage โดยไม่รันจริง
- งาน QA ไม่จำเป็นต้องรัน pipeline ใหม่ เพราะ artifact ใต้
  `data/output/final/` กับ runtime DB ที่มีอยู่เป็นปัจจุบันแล้ว
