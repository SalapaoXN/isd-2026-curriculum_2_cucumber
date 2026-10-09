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

### GPU / CUDA สำหรับ EasyOCR

ถ้ามี NVIDIA GPU แนะนำให้ตรวจว่า **PyTorch ใน virtual environment รองรับ CUDA จริง** ก่อนรัน OCR หลาย dataset เพราะการติดตั้ง `easyocr` อย่างเดียวอาจทำให้ได้ PyTorch แบบ CPU-only ซึ่ง EasyOCR จะมองไม่เห็น GPU แม้ Windows จะเห็นการ์ดจอปกติ

ตรวจว่า Windows/NVIDIA driver เห็น GPU:

```powershell
nvidia-smi
```

จากนั้นตรวจใน Python environment ที่กำลังใช้รันโปรเจกต์:

```powershell
python -c "import torch; print('torch=', torch.__version__); print('cuda available=', torch.cuda.is_available()); print('cuda version=', torch.version.cuda); print('device=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU ONLY')"
```

ค่าที่ต้องการสำหรับการใช้ GPU คือ:

```text
cuda available= True
device= <ชื่อ NVIDIA GPU>
```

ถ้า `nvidia-smi` ใช้งานได้ แต่ `torch.cuda.is_available()` เป็น `False` หรือ `torch.version.cuda` เป็น `None` แปลว่า PyTorch ใน environment นั้นไม่ใช่ CUDA-enabled build ให้ติดตั้ง PyTorch รุ่นที่รองรับ CUDA ตามคำแนะนำจากเว็บไซต์ PyTorch สำหรับเครื่องนั้นก่อนรัน OCR จริง แล้วตรวจคำสั่งด้านบนซ้ำอีกครั้ง

> `requirements.txt` ของโปรเจกต์ไม่ได้บังคับ CUDA build ของ PyTorch เนื่องจากแต่ละเครื่องอาจใช้ driver/CUDA compatibility ต่างกัน จึงต้องตรวจ GPU environment แยกต่างหาก

ถ้าต้องการบังคับรันด้วย CPU:

```powershell
python -m src.pipeline.run --dataset it2560 --no-gpu
```

ถ้าต้องการสร้าง runtime DB ต่อหลังจบ pipeline:

```powershell
python -m src.pipeline.run --dataset it2560 --with-index
```

## Runbook ที่ใช้จริง: OCR ใหม่ทุกหลักสูตร → Final

ส่วนนี้รวมคำสั่งที่ใช้จริงเมื่อต้องการสร้าง curriculum artifacts ใหม่ทั้งชุด โดย **ยังไม่รวม rules pipeline**

### 0. เข้า virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

ติดตั้ง dependency หลักของโปรเจกต์ถ้ายังไม่ได้ติดตั้ง:

```powershell
python -m pip install -r requirements.txt
```

> สำหรับ NVIDIA GPU ต้องตรวจ CUDA-enabled PyTorch แยกตามหัวข้อ GPU / CUDA ด้านบน เพราะ `requirements.txt` ไม่ได้บังคับ CUDA build

### 1. ตรวจ dataset configuration ก่อนรันจริง

ตัวอย่าง dry-run:

```powershell
python -m src.pipeline.run --dataset it2565 --dry-run
python -m src.pipeline.run --dataset it2560 --dry-run
python -m src.pipeline.run --dataset dsba2560 --dry-run
```

### 2. OCR dataset เดียว

ใช้ standalone OCR เมื่อยังไม่ต้องการให้ pipeline ไปถึง Gemini correction:

```powershell
python -m src.pipeline.tools.ocr.cli --dataset ait2566
```

เปลี่ยน `ait2566` เป็น dataset key อื่นได้ตามต้องการ

ถ้าจะบังคับ CPU:

```powershell
python -m src.pipeline.tools.ocr.cli --dataset ait2566 --no-gpu
```

### 3. OCR curriculum ทุก dataset โดยยังไม่แตะ rules

```powershell
$datasets = @(
    "ait2566",
    "bit2565",
    "bit2560",
    "dsba2565",
    "dsba2560",
    "gened2564",
    "gened2557",
    "it2565",
    "it2560"
)

foreach ($d in $datasets) {
    Write-Host "`n===== OCR $d ====="
    python -m src.pipeline.tools.ocr.cli --dataset $d

    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: $d"
        break
    }
}
```

ขั้นนี้ทำเฉพาะ OCR และไม่เรียก Gemini

### 4. Extract + Merge จาก OCR ที่มีอยู่

```powershell
python -m src.pipeline.tools.preparation.tool
```

คำสั่งนี้ค้นหา supported OCR datasets ภายใต้ `data/output/ocr/` แล้ว Extract + Merge แยก dataset/plan ตาม `src/pipeline/datasets.py` โดยไม่เอาหลักสูตรทั้งหมดไปรวมเป็นไฟล์เดียว และยังไม่เรียก Gemini

ถ้ารัน preparation ซ้ำแล้วเจอ `edition artifact already exists` ให้เก็บ consolidated รอบเก่าเป็น backup ก่อน:

```powershell
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
Rename-Item data\output\consolidated "consolidated_backup_$stamp"
python -m src.pipeline.tools.preparation.tool
```

OCR เดิมไม่ต้องรันใหม่

### 5. ทดลอง Gemini correction หนึ่ง dataset ก่อน

ตัวอย่าง AIT 2566:

```powershell
python -m src.pipeline.run --dataset ait2566 --from consolidated --skip-eval
```

ตัวอย่าง BIT 2565:

```powershell
python -m src.pipeline.run --dataset bit2565 --from consolidated --skip-eval
```

`--from consolidated` หมายถึงไม่ OCR, ไม่ Extract และไม่ Merge ซ้ำ แต่เริ่มจาก consolidated ที่มีอยู่ → Gemini correction → publish `*_final.json` และ `*_corrections.json`

### 6. Correction ทุก curriculum dataset

เมื่อลอง dataset เดียวผ่านแล้ว สามารถรันต่อทุก dataset โดยเว้นช่วงระหว่าง dataset เพื่อลดความเสี่ยง API rate limit:

```powershell
$datasets = @(
    "ait2566",
    "bit2565",
    "bit2560",
    "dsba2565",
    "dsba2560",
    "gened2564",
    "gened2557",
    "it2565",
    "it2560"
)

foreach ($d in $datasets) {
    Write-Host "`n===== CORRECT $d ====="

    python -m src.pipeline.run --dataset $d --from consolidated --skip-eval

    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: $d"
        break
    }

    Write-Host "Waiting 30 seconds..."
    Start-Sleep -Seconds 30
}
```

ถ้าบาง dataset ทำเสร็จแล้ว ให้ลบชื่อ dataset เหล่านั้นออกจาก `$datasets` แล้วรันเฉพาะตัวที่เหลือ ไม่จำเป็นต้อง correction ซ้ำทั้งหมด

Gemini corrector ปัจจุบันแบ่งข้อความเป็น batch ละ 50 unique units และ deduplicate ชื่อซ้ำระหว่าง plan files ของ dataset เดียวกันก่อนส่ง API ถ้าเจอ `429`, `RESOURCE_EXHAUSTED`, `quota` หรือ `Gemini batch ... failed` ให้หยุดชุดนั้นก่อน ไม่ควรยิง loop ซ้ำทันที

### 7. ตรวจ Final artifacts

หลัง correction สำเร็จ ไฟล์หลักอยู่ที่:

```text
data/output/final/
data/output/corrections/
```

ตัวอย่าง:

```text
ait2566_final.json
bit2565_coop_final.json
bit2565_no_coop_final.json
it2565_coop_final.json
it2565_no_coop_final.json
```

### 8. Rules gate ก่อน Build Index

ถ้า runtime ต้องใช้ข้อกำหนดสถาบันและ program requirements ด้วย ให้ทำ rules pipeline ให้พร้อมก่อน build index:

```powershell
python -m src.pipeline.run_rules
```

rules source ใช้ dataset identity `rule2564` สำหรับหน้ากฎ 1-13 แม้จะเก็บรวมอยู่ใน category folder `data/input/rule/` โดย output OCR จะอยู่ที่ `data/output/ocr/rule2564/` แยกจาก curriculum OCR ชัดเจน

### 9. Evaluation → canonicalize → preflight → Build runtime index

เมื่อมี reviewed finals ใน data/output/final/*_final.json ให้รันตามลำดับนี้:

    .\.venv\Scripts\python.exe -m src.pipeline.tools.evaluation.evaluate --reports-dir reports/evaluation_precanonical
    .\.venv\Scripts\python.exe -m src.pipeline.tools.canonicalize_runtime
    .\.venv\Scripts\python.exe -m src.pipeline.tools.preflight_runtime
    .\.venv\Scripts\python.exe -m rag.build_index

Evaluation ใช้ raw reviewed finals ก่อน canonicalization และเขียนผลแยกใน reports/evaluation_precanonical/. ประเมินเฉพาะ 8 scopes ที่มี accepted GT ตรง edition: AIT 2566; BIT/DSBA/IT 2565 ทั้ง coop และ no_coop; GENED 2564. BIT/DSBA/IT 2560 และ GENED 2557 จะถูกข้ามเพราะไม่มี accepted Ground Truth ของ edition เดียวกัน.

หลัง evaluation เสร็จ canonicalizer ใช้ accepted teacher GT เป็น authority เฉพาะ fields ที่มีใน GT: code, name_th, name_en, credits, year, semester, category, type, prerequisite, flexible_year_semester และ note. Descriptions และ provenance มาจาก reviewed finals. สำหรับ legacy 2560 ใช้ data/corrections/legacy_2560_source_verified_corrections.json เท่านั้น; ไม่ใช้ current GT. preflight_runtime แสดง unresolved shared-course conflicts ทั้งหมด และ build จะหยุดถ้ายังมี conflict.

rag.build_index ใช้ curriculum artifacts จาก data/output/canonical/*_final.json เท่านั้น. institution_policy.json และ program_requirements.json ยังคงโหลดจาก data/output/final/ เป็น supplemental sources. เมื่อ raw finals พร้อมแล้วไม่ต้องรัน OCR หรือ Gemini ซ้ำ. คำสั่ง src.pipeline.run --dataset it2565 --from final --with-index ทำ evaluation → canonicalization → preflight → build ต่อเนื่อง; ห้ามใช้ --skip-eval กับ --with-index. --only-index ใช้ canonical artifacts ที่มีอยู่แล้ว.

หากต้องสร้าง raw finals ใหม่ ให้รัน Source/OCR → Extract/Merge → Gemini correction ก่อน แล้วจึงใช้ลำดับสี่คำสั่งด้านบน.

## Flow

```text
Source images
→ EasyOCR (TH + EN)
→ deterministic pre-clean
→ Extract structured curriculum fields
→ Merge study plan + course descriptions
→ Gemini proofreading เฉพาะชื่อวิชา
→ Final reviewed JSON
→ Evaluation against accepted teacher Ground Truth
→ Deterministic post-evaluation canonicalization
→ Shared-course preflight
→ Rules / program requirements gate
→ Build Index
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

ชื่อ folder และชื่อไฟล์ source ต้องใช้ **dataset key พร้อมปีเดียวกัน** ระบบไม่ fallback ไปหา folder แบบไม่มีปีสำหรับ curriculum dataset แล้ว

ชื่อ source image ต้องเป็น `<dataset>_page_<NNN>.<ext>` เช่น:

```text
it2565_page_032.png
it2560_page_027.png
bit2565_page_026.png
dsba2565_page_317.png
```

ชื่อ source จริงจะถูกเก็บใน provenance ดังนั้นการใส่ปีไว้ในชื่อไฟล์ทำให้ source identity แยกฉบับหลักสูตรได้ชัดเจน

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
data/output/consolidated/<dataset>_<plan>_consolidated.json
```

เช่น:

```text
data/output/consolidated/it2560_no_coop_consolidated.json
data/output/consolidated/it2560_coop_consolidated.json
data/output/consolidated/ait2566_consolidated.json
```

### Correction logs

audit log จากขั้น correction แยกออกจาก final data โดยตรง:

```text
data/output/corrections/it2560_no_coop_corrections.json
data/output/corrections/it2560_coop_corrections.json
```

### Final หลัง LLM correction

`final/` เก็บเฉพาะ canonical curriculum data หลังตรวจแก้แล้ว:

```text
data/output/final/it2560_no_coop_final.json
data/output/final/it2560_coop_final.json
data/output/final/ait2566_final.json
data/output/final/gened2557_final.json
```

ไฟล์ `*_final.json` ใช้สร้าง runtime database ส่วน `*_corrections.json` เป็น audit log และไม่ถูกใช้เป็น curriculum source

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
python -m src.pipeline.run --dataset it2560 --from final
```

เก็บ extracted intermediates สำหรับ debug:

```powershell
python -m src.pipeline.run --dataset it2560 --keep-intermediates
```

## Runtime DB

python -m rag.build_index สร้าง runtime database จาก data/output/canonical/*_final.json หลังผ่าน preflight; ไม่อ่าน raw curriculum finals เป็น default curriculum source. ให้ทำ evaluation → canonicalize → preflight ก่อน build ตามคำสั่งในหัวข้อ 9. institution_policy.json และ program_requirements.json ยังคงเป็น supplemental sources จาก data/output/final/.

runtime database:

    cucumber_outputs/runtime/curriculum.db

## Rules pipeline

ข้อกำหนดสถาบันใช้ entry point แยกจาก curriculum pipeline:

```powershell
python -m src.pipeline.run_rules
```

ตรวจ manifest โดยยังไม่ OCR จริง:

```powershell
python -m src.pipeline.run_rules --dry-run
```

ถ้าเครื่องไม่มี CUDA/GPU:

```powershell
python -m src.pipeline.run_rules --no-gpu
```

source images อยู่รวมกันที่ `data/input/rule/` แต่ใช้ชื่อ source แบบ edition-aware ดังนี้:

```text
data/input/rule/rule2564_page_001.png
...
data/input/rule/rule2564_page_013.png

data/input/rule/ait2566_page_005.png
data/input/rule/bit2565_page_006.png
data/input/rule/dsba2565_page_006.png
data/input/rule/it2565_page_006.png
```

Rule OCR ถูก publish แยกที่:

```text
data/output/ocr/rule2564/rule2564_page_001_ocr.json
...
data/output/ocr/rule2564/rule2564_page_013_ocr.json
```

ผล canonical สำหรับ runtime คือ:

```text
data/output/final/institution_policy.json
data/output/final/program_requirements.json
```

ถ้ามี canonical intermediate จาก rule OCR อยู่แล้วและต้องการ map policy ใหม่โดยไม่ OCR ซ้ำ:

```powershell
python -m src.pipeline.run_rules --skip-ocr
```

รายละเอียดดูที่ `docs/academic_rules.md`
