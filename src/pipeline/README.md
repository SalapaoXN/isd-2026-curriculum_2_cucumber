# src/pipeline — Data Pipeline

ส่วนนี้แปลงภาพหลักสูตรให้เป็น canonical data ที่ใช้สร้าง runtime database

## Flow

```text
OCR
→ Extract
→ Merge
→ Correct
→ Evaluate
→ Build Index
```

entry point:

```powershell
python -m src.pipeline.run
```

ตัวอย่าง:

```powershell
python -m src.pipeline.run --program it --dry-run
python -m src.pipeline.run --program it --with-index
python -m src.pipeline.run --program it --pages 32-38
python -m src.pipeline.run --program it --no-gpu
python -m src.pipeline.run --program it --from corrected --with-index
```

program ที่รองรับ: `ait`, `bit`, `dsba`, `gened`, `it`

## Inputs

ภาพต้นฉบับ:

```text
data/input/<program>/<program>_page_NNN.png
```

ไฟล์ภาพขนาดใหญ่ไม่ได้เก็บครบใน repo ถ้าจะรัน OCR ใหม่ต้องเตรียม source images ก่อน

## Canonical outputs

ผลหลักสูตรปัจจุบันอยู่ที่:

```text
data/output/final/*_corrected.json
data/output/final/*_corrections.json
```

- `*_corrected.json` = canonical curriculum corpus
- `*_corrections.json` = audit log ของ correction

สาย rules/policy สร้าง supplemental authority:

```text
data/output/final/institution_policy.json
data/output/final/program_requirements.json
```

## Runtime DB

สร้างด้วย:

```powershell
python -m rag.build_index
```

default build จะโหลด:

1. canonical `*_corrected.json`
2. `institution_policy.json`
3. `program_requirements.json`

แล้วสร้าง:

```text
cucumber_outputs/runtime/curriculum.db
```

QA อ่าน DB นี้แบบ runtime authority และไม่ควรแก้ DB ด้วยมือ

## Intermediate data

โดย default intermediate data ใช้สำหรับการประมวลผล/debug และไม่ใช่ factual authority ของ QA

เมื่อต้องการเก็บ:

```powershell
python -m src.pipeline.run --program it --keep-intermediates
```

จะเก็บ extracted/consolidated output เพิ่มตาม configuration

## Evaluation

pipeline สามารถประเมิน canonical data เทียบ Ground Truth ได้ รายงานอยู่ใต้:

```text
reports/evaluation/
```

runtime ปัจจุบันมีหลาย curriculum editions แต่ Ground Truth ยังไม่ได้มีแยกทุก edition ดังนั้น final report ต้องประเมินเฉพาะ prediction/GT pairs ที่ตรงกันอย่างชัดเจน ไม่ใช้ no-argument discovery เพื่อเอา historical editions มาปนกับ metric

คำสั่งและ scope ที่ใช้สร้าง report ปัจจุบันดูที่ `reports/README.md`

`ground_truth/` ใช้สำหรับ evaluation/test เท่านั้น ไม่ถูกใช้เป็น production factual source

## Rules pipeline

ข้อกำหนดสถาบันใช้ entry point แยก:

```powershell
python -m src.pipeline.run_rules
```

รายละเอียดอยู่ที่ `docs/academic_rules.md`
