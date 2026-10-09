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

program หลักที่รองรับ: `ait`, `bit`, `dsba`, `gened`, `it`

> คำสั่ง `src.pipeline.run --program ...` ใช้ชื่อ program หลัก 5 ชื่อนี้ ส่วนข้อมูลหลักสูตรฉบับเก่า เช่น `it2560`, `bit2560`, `dsba2560` และ `gened2557` ให้รัน OCR ด้วยคำสั่ง edition-specific ที่อธิบายในหัวข้อถัดไป

## Inputs

ภาพต้นฉบับอยู่ใต้ `data/input/` โดยแยกตามหลักสูตรและฉบับ เช่น:

```text
data/input/
├─ ait/
├─ bit/          # BIT ฉบับปัจจุบัน (2565)
├─ bit2560/      # BIT 2560
├─ dsba/         # DSBA ฉบับปัจจุบัน (2565)
├─ dsba2560/     # DSBA 2560
├─ gened/
├─ gened2557/    # GENED 2557
├─ it/           # IT ฉบับปัจจุบัน (2565)
├─ it2560/       # IT 2560
└─ rule/         # เอกสารกฎ/ข้อกำหนดของสถาบัน
```

รูปแบบชื่อภาพโดยทั่วไป:

```text
data/input/<dataset>/<dataset>_page_NNN.png
```

ไฟล์ภาพขนาดใหญ่ไม่ได้เก็บครบใน repo ถ้าจะรัน OCR ใหม่ต้องเตรียม source images ก่อน

## การรัน OCR

### OCR ฉบับปัจจุบัน

สำหรับ dataset หลัก `ait`, `bit`, `dsba`, `gened`, `it` ใช้ standalone OCR CLI ได้โดยตรง:

```powershell
python -m src.pipeline.tools.ocr.cli --prefix it
python -m src.pipeline.tools.ocr.cli --prefix bit
python -m src.pipeline.tools.ocr.cli --prefix dsba
python -m src.pipeline.tools.ocr.cli --prefix gened
python -m src.pipeline.tools.ocr.cli --prefix ait
```

ถ้าต้องการรันเฉพาะบางหน้า:

```powershell
python -m src.pipeline.tools.ocr.cli --prefix it --pages 32-38
python -m src.pipeline.tools.ocr.cli --prefix it --pages 32
```

ถ้าต้องการบังคับใช้ CPU:

```powershell
python -m src.pipeline.tools.ocr.cli --prefix it --no-gpu
```

ผล OCR จะถูกเก็บใต้:

```text
data/output/ocr/<program>/
```

เช่น `data/output/ocr/it/`

### OCR หลักสูตรฉบับเก่า 2560 / 2557

ฉบับเก่าแยก source folder และ `dataset-key` เพื่อไม่ให้ผล OCR ไปชนกับฉบับปัจจุบัน

> **สำคัญ:** ในขั้น OCR ค่า `--plan` **ไม่ได้ใช้กรองว่าจะ OCR เฉพาะแผน `coop` หรือ `no_coop`** แต่ใช้เป็นค่าประกอบ/validation ของคำสั่งเท่านั้น สำหรับหลักสูตรที่มีหลายแผนจึงต้องระบุค่าที่ถูกต้องสักหนึ่งค่า เช่น `--plan no_coop` ส่วนหน้าที่ OCR จริงถูกกำหนดด้วย `--pages` เท่านั้น ถ้าไม่ใส่ `--pages` ระบบจะค้นและ OCR ทุกภาพที่พบใน `--input-dir` การแยกข้อมูลเป็น `coop` และ `no_coop` จะเกิดในขั้น Extract/Preparation ตาม configuration ของแต่ละฉบับ

ช่วงหน้าที่ pipeline ใช้สำหรับฉบับเก่า:

| Dataset | แผน `no_coop` | แผน `coop` | Course description / shared pages |
| --- | --- | --- | --- |
| `it2560` | 27-33 | 34-40 | 222-269 |
| `bit2560` | 23-26 | 27-30 | 170-192 |
| `dsba2560` | 25-29 | 30-34 | 175-207 |
| `gened2557` | - | - | 11-18,47-92 (`gened`) |

ถ้าต้องการสร้าง OCR ที่จำเป็นต่อทั้งสองแผนของแต่ละฉบับ แนะนำระบุช่วงหน้าที่ต้องใช้โดยตรงดังนี้

#### IT 2560 — ทั้งสองแผน + description

```powershell
python -m src.pipeline.tools.ocr.pipeline_runner `
  --input-dir data/input/it2560 `
  --program IT `
  --plan no_coop `
  --dataset-key it2560 `
  --pages 27-40,222-269
```

เฉพาะหน้าเดียวสำหรับทดสอบหรือ demo:

```powershell
python -m src.pipeline.tools.ocr.pipeline_runner `
  --input-dir data/input/it2560 `
  --program IT `
  --plan no_coop `
  --dataset-key it2560 `
  --pages 27
```

#### BIT 2560 — ทั้งสองแผน + description

```powershell
python -m src.pipeline.tools.ocr.pipeline_runner `
  --input-dir data/input/bit2560 `
  --program BIT `
  --plan no_coop `
  --dataset-key bit2560 `
  --pages 23-30,170-192
```

#### DSBA 2560 — ทั้งสองแผน + description

```powershell
python -m src.pipeline.tools.ocr.pipeline_runner `
  --input-dir data/input/dsba2560 `
  --program DSBA `
  --plan no_coop `
  --dataset-key dsba2560 `
  --pages 25-34,175-207
```

ตัวอย่างถ้าต้องการ OCR เฉพาะหน้าของแผน `no_coop` ของ DSBA 2560 และ description ที่ใช้ร่วมกัน:

```powershell
python -m src.pipeline.tools.ocr.pipeline_runner `
  --input-dir data/input/dsba2560 `
  --program DSBA `
  --plan no_coop `
  --dataset-key dsba2560 `
  --pages 25-29,175-207
```

ตัวอย่างถ้าต้องการ OCR เฉพาะหน้าของแผน `coop` ของ DSBA 2560 และ description ที่ใช้ร่วมกัน:

```powershell
python -m src.pipeline.tools.ocr.pipeline_runner `
  --input-dir data/input/dsba2560 `
  --program DSBA `
  --plan coop `
  --dataset-key dsba2560 `
  --pages 30-34,175-207
```

> ในสองตัวอย่างด้านบน สิ่งที่ทำให้ OCR ต่างกันจริงคือค่า `--pages` ไม่ใช่ค่า `--plan`

#### GENED 2557

```powershell
python -m src.pipeline.tools.ocr.pipeline_runner `
  --input-dir data/input/gened2557 `
  --program GENED `
  --plan gened `
  --dataset-key gened2557 `
  --pages 11-18,47-92
```

ผลจะถูกแยกตามฉบับ:

```text
data/output/ocr/it2560/
data/output/ocr/bit2560/
data/output/ocr/dsba2560/
data/output/ocr/gened2557/
```

`--dataset-key` ใช้ระบุฉบับของ source dataset และป้องกันไม่ให้ output ของฉบับเก่าเขียนรวมกับฉบับปัจจุบัน

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
