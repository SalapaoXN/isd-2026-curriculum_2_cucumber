# CUCUMBER

**P2 LLM ถาม-ตอบหลักสูตร**

CUCUMBER คือระบบแปลงเอกสารหลักสูตรเป็นข้อมูลที่มีโครงสร้าง แล้วนำไปสร้างระบบถาม-ตอบภาษาไทยที่ตรวจย้อนกลับถึงแหล่งข้อมูลได้

Member:
1. 67070049 Nattachai Kaewchum — Discord: GoodDee
2. 67070063 Thanachin Chukiatchai — Discord: วันลพ มีงบมาก
3. 67070103 Pongsakorn Panyacom — Discord: เบบี๋คือดวงใจ

---

## 1. ระบบทำอะไร

ภาพรวมการทำงาน:

```text
เอกสารหลักสูตร
→ OCR
→ Extraction / Merge
→ LLM Correction
→ Canonical JSON
→ SQLite + Semantic Index
→ Natural-language QA
→ Grounded Answer + Provenance
```

แนวคิดหลักคือ **ข้อเท็จจริงมาจากข้อมูลหลักสูตรและฐานข้อมูล ไม่ใช่ให้ LLM เดาเอง**

ระบบรองรับข้อมูล:

- AIT
- BIT — `coop`, `no_coop`
- DSBA — `coop`, `no_coop`
- GENED — `gened`
- IT — `coop`, `no_coop`

แต่ละ program/plan ถูกแยก identity ออกจากกันตลอด pipeline เพื่อลดการปนข้อมูลข้ามหลักสูตรหรือข้ามแผน

---

## 2. ใช้งานเร็วที่สุด

### 2.1 สร้าง environment

ต้องใช้ Python **3.10 ขึ้นไป**

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 2.2 ตั้งค่า Gemini

สร้างไฟล์ `.env` ที่ root ของ repository:

```dotenv
GEMINI_API_KEY=your_key_here
```

`scripts/ask.py` สร้าง Gemini provider ตอนเริ่มโปรแกรม จึงต้องมี `GEMINI_API_KEY`

ปัจจุบัน provider ใช้โมเดล:

```text
gemini-3.5-flash-lite
```

ห้าม commit `.env` หรือ API key

### 2.3 ถามคำถาม

> หลัง restructure ไม่มี `ask.py` ที่ root แล้ว

ถามหนึ่งข้อ:

```powershell
python scripts/ask.py "IT ปี 2 เทอม 1 เรียนวิชาอะไรบ้าง"
```

โหมดถามต่อเนื่อง:

```powershell
python scripts/ask.py
```

ออกด้วย:

```text
exit
quit
```

ตัวอย่างคำถาม:

```text
IT ปี 3 เทอม 1 มีวิชาอะไรบ้าง
06016414 กี่หน่วยกิต
06016414 เรียนปีไหนเทอมไหน
06016414 ต้องผ่านวิชาอะไรมาก่อน
มีวิชาเกี่ยวกับ data หรือ database อะไรบ้าง
แผนสหกิจกับไม่สหกิจของ IT ปี 4 ต่างกันยังไง
IT ปี 3 อยากเน้น data มีวิชาไหนที่วิชาบังคับก่อนไม่เยอะบ้าง
```

ผลลัพธ์ปกติ:

```text
ถาม: ...
ตอบ: ...
แหล่งข้อมูล:
เล่มหลักสูตร: ...
หน้า: ...
```

ถ้าหลักฐานไม่พอ ระบบจะขอให้ระบุ program/plan เพิ่ม หรือคืนสถานะไม่พบข้อมูลแทนการเดา

---

## 3. Runtime database

QA ใช้ฐานข้อมูล:

```text
cucumber_outputs/runtime/curriculum.db
```

repository ปัจจุบันมี runtime database อยู่แล้ว จึงถามระบบได้ทันทีหลังติดตั้ง dependency และตั้งค่า `.env`

ถ้าแก้ข้อมูลใน `data/output/final/` หรือไม่มี database ให้สร้างใหม่ด้วย:

```powershell
python -m rag.build_index
```

RAG จะอ่านไฟล์:

```text
data/output/final/*_corrected.json
```

แล้วสร้าง relational data + semantic chunks ไว้ใน database เดียวกัน

---

## 4. โครงสร้าง repository

```text
config/
  pipeline.yaml
  programs.yaml

data/
  input/
  output/
    final/

ground_truth/
rag/
reports/
scripts/
  ask.py
src/
  pipeline/
    run.py
    config.py
    models.py
    tools/
      ocr/
      extraction/
      correction/
      merge/
      preparation/
      evaluation/
      indexing/
    utils/
submission/
tests/
  pipeline/
  rag/
  tools/
  reference/
cucumber_outputs/
  runtime/
    curriculum.db
```

หน้าที่สำคัญ:

- `data/input/` — ภาพหน้าหลักสูตรสำหรับ OCR
- `data/output/final/` — **canonical final corpus** ที่ RAG ใช้จริง
- `src/pipeline/` — pipeline ตั้งแต่ OCR ถึง index
- `rag/` — query parsing, intent, retrieval, aggregation, grounding และ answer
- `scripts/ask.py` — CLI สำหรับผู้ใช้
- `ground_truth/` — reference สำหรับ evaluation ไม่ใช่ production factual authority
- `tests/reference/ocr/` — OCR fixtures สำหรับ regression test
- `submission/` — ชุดส่งงานเดิม แยกจาก runtime ปัจจุบัน
- `cucumber_outputs/runtime/curriculum.db` — database ที่ QA ใช้

### แผนที่ subsystem (เอกสารละเอียดแยกตามส่วน)

- `src/pipeline/README.md` — OCR → canonical JSON → database
- `data/README.md` — ชั้นข้อมูลและความหมายของแต่ละ layer
- `rag/README.md` — QA pipeline, query families, evaluation
- `lab10_fastapi/README.md` — API, UI, provider/key behavior
- `scripts/README.md` — CLI และเครื่องมือ developer

---

## 5. Pipeline เต็ม

entry point หลักหลัง restructure คือ:

```powershell
python -m src.pipeline.run
```

ลำดับ stage:

```text
ocr
→ extract
→ merge
→ correct
→ evaluate
→ build_index (เปิดเพิ่มด้วย --with-index)
```

### 5.1 เช็กก่อนรันจริง

```powershell
python -m src.pipeline.run --program it --dry-run
```

`--dry-run` แสดง config/stage แต่ไม่รัน OCR หรือ LLM

### 5.2 รันตั้งแต่ OCR จนถึง database

```powershell
python -m src.pipeline.run --program it --with-index
```

โดยปกติ evaluation เปิดอยู่ แต่ build index ปิดอยู่ จึงต้องใส่ `--with-index` ถ้าต้องการต่อถึง `curriculum.db`

### 5.3 เลือกหน้า OCR

```powershell
python -m src.pipeline.run --program it --pages 32-38
```

หลายช่วง:

```powershell
python -m src.pipeline.run --program it --pages 32-38,328-371
```

### 5.4 ใช้ CPU

```powershell
python -m src.pipeline.run --program it --no-gpu
```

### 5.5 เริ่มต่อจาก stage เดิม

```powershell
python -m src.pipeline.run --program it --from extracted
python -m src.pipeline.run --program it --from consolidated
python -m src.pipeline.run --program it --from corrected --with-index
```

ค่า `--from` ที่รองรับ:

```text
ocr
extracted
consolidated
corrected
```

### 5.6 เก็บ intermediate files สำหรับ debug

ปกติ extracted/consolidated working data บางส่วนใช้ temp directory แล้วลบทิ้งอัตโนมัติ

ถ้าต้องการเก็บไว้:

```powershell
python -m src.pipeline.run --program it --keep-intermediates
```

### 5.7 สร้าง index อย่างเดียว

จาก pipeline:

```powershell
python -m src.pipeline.run --program it --only-index
```

หรือสร้าง database จาก canonical final corpus ทั้งหมด:

```powershell
python -m rag.build_index
```

---

## 6. Data flow และ artifact

### Input

ใส่ภาพตาม program เช่น:

```text
data/input/it/it_page_032.png
data/input/it/it_page_033.png
...
```

program key ที่รองรับ:

```text
ait
bit
dsba
gened
it
```

### OCR

ผล OCR อยู่ใต้:

```text
data/output/ocr/<program>/
```

### Extraction / Merge

เมื่อ debug ด้วย `--keep-intermediates` จะเห็น:

```text
data/output/extracted/
data/output/consolidated/
```

### Corrected final data

ข้อมูลปลายทางอยู่ที่:

```text
data/output/final/
```

ไฟล์หลักมีสองแบบ:

```text
*_corrected.json
*_corrections.json
```

- `*_corrected.json` — ข้อมูลที่ใช้สร้าง runtime database
- `*_corrections.json` — log การแก้ข้อความ

`data/output/final/` คือ source หลักของ RAG ใน layout ปัจจุบัน ส่วน path เก่า `outputs/llm/` มีไว้เป็น legacy fallback ในบางโมดูลเท่านั้น

---

## 7. QA ทำงานอย่างไร

ระบบไม่ได้ส่งคำถามให้ LLM แล้วเชื่อคำตอบทันที

flow โดยย่อ:

```text
Question
→ deterministic parse / scope resolution
→ optional intent interpretation
→ evidence planning
→ structured / semantic retrieval
→ deterministic aggregation
→ grounded claims
→ answer rendering
→ provenance
```

### Structured

ใช้กับข้อเท็จจริง เช่น:

- รายชื่อวิชา
- จำนวนวิชา
- หน่วยกิต
- ปี / เทอม
- prerequisite
- plan comparison

### Semantic

ใช้กับคำถามด้านความหมาย เช่น:

```text
มีวิชาเกี่ยวกับ machine learning ไหม
มีวิชาแนว database อะไรบ้าง
```

embedding model ปัจจุบัน:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

โมเดลจะถูกโหลดเมื่อ semantic embedding ถูกใช้งานครั้งแรก

### Natural / advisory query

ระบบมี intent layer สำหรับคำถามที่เขียนไม่ตรง template เช่น:

```text
IT ปี 3 อยากไปสาย data แต่ไม่อยากเจอ prereq เยอะ มีตัวไหนบ้าง
```

LLM ที่ intent layer ให้ได้เพียง **proposal ของความหมายคำถาม** ไม่ใช่ข้อเท็จจริงของหลักสูตร จากนั้น deterministic scope/evidence layer จะตรวจอีกครั้งก่อนใช้

### SQL fallback

กรณี structured wording แปลกกว่ากฎปกติ ระบบมี guarded SQL fallback แต่ SQL ใช้เพียงเลือก candidate เช่น `course_id` หรือ `placement_id`

COUNT / SUM / existence / credit totals และ factual fields จะคำนวณหรือ hydrate จาก canonical data แบบ deterministic หลังจากนั้น

---

## 8. หลักการความถูกต้อง

ระบบใช้แนวทาง:

### Deterministic-first

ข้อเท็จจริงหลักสูตรมาจาก SQLite/canonical data ก่อน

### Scope isolation

ไม่ขยาย program/plan/year/semester เอง และไม่เดา program จาก prefix ของ course code

### Grounding

คำตอบสร้างจาก evidence ที่หาได้จริง

### Provenance

พยายามแนบ program และ source page กลับไปกับคำตอบ

### Fail closed

ถ้าหลักฐานไม่พอหรือ scope ขัดแย้ง ระบบคืน `insufficient_evidence`, `valid_empty`, clarification หรือข้อความไม่พบข้อมูล แทนการแต่งข้อเท็จจริง

---

## 9. Testing

รัน test ทั้งหมด:

```powershell
python -m unittest discover -s tests -t .
```

OCR:

```powershell
python -m unittest tests.pipeline.test_ocr -v
```

OCR reference / description:

```powershell
python -m unittest tests.tools.test_descriptions tests.tools.test_course_code_validation -v
```

RAG ตัวอย่าง:

```powershell
python -m unittest tests.rag.test_rag_qa tests.rag.test_rag_answer -v
```

รายละเอียด architecture และสถานะ validation ปัจจุบันดูที่:

```text
docs/review.md
```

---

## 10. สิ่งที่ควรจำ

1. ใช้ `python scripts/ask.py` ไม่ใช่ `python ask.py`
2. RAG source ปัจจุบันคือ `data/output/final/`
3. Runtime DB คือ `cucumber_outputs/runtime/curriculum.db`
4. `ground_truth/` ใช้ตรวจผล ไม่ใช่ production authority
5. `submission/` เป็น artifact แยกจาก runtime ปัจจุบัน
6. ถ้าแก้ canonical JSON ให้รัน `python -m rag.build_index` ใหม่ก่อนทดสอบ QA

---

## 11. RAG Final: ขอบเขตที่รองรับ & demo

### คำถามที่รองรับ
- รายชื่อวิชา / จำนวนวิชา / มีวิชานี้ไหม / วิชานี้เรียนตอนไหน (ระบุ program เสมอ)
- หน่วยกิตรายวิชา; กรองรายชื่อด้วย `N หน่วยกิต` (list-only)
- ผลรวมหน่วยกิตตาม scope; ผลรวมหมวด (`วิชาเลือก` / `หมวดวิชาศึกษาทั่วไป`) เฉพาะปี+เทอมที่ระบุชัด
- วิชาบังคับก่อน (รายวิชา + ติดตามผลแบบ result-set); รายละเอียดวิชา; ความคล้าย 2 วิชา; เปรียบเทียบแผน
- คำถามนอก template บางรูป (op-free scope, preference) ใช้ bounded interpretation (สูงสุด 1 call)
- นโยบายแบบ single-turn: เพดาน/ขั้นต่ำลงทะเบียน, กรณีพิเศษ, ซัมเมอร์, โปร, เกียรตินิยม, กลับเข้าศึกษา, รวมหลักสูตร,  verdict ลงทะเบียน

### Multi-turn
ส่ง `next_context` จาก response กลับมาเป็น `conversation_context` ในครั้งถัดไป (โครงสร้างล้วน ไม่มีแคชคำตอบ) เทิร์นปัจจุบันที่ระบุชัดชนะ context; anaphora ที่กู้ไม่ได้ fail safe

### Authority
ข้อเท็จจริงหลักสูตรมาจาก SQLite deterministic; นโยบายมาจาก policy authority แยกกัน; LLM เป็นได้แค่ interpreter/presenter; `ground_truth/` ใช้ประเมินเท่านั้น ทุกคำตอบมี provenance; หลักฐานไม่พอ → `insufficient_evidence` / `valid_empty` / `no_data` / `clarify_program` / `unsupported` แทนการเดา

### ทดสอบแบบ offline (ไม่ต้องมี API key)
```powershell
python -m unittest tests.rag.test_final_core_eval -v
```
Core Set 35 ข้อ (`tests/rag/fixtures/final_core_eval_v1.json`) รันผ่าน public `ask()` ทั้งหมด

### Demo
CLI ต้องมี `GEMINI_API_KEY`:
```powershell
python scripts/ask.py "IT ปี 1 เทอม 1 มีวิชาอะไรบ้าง"
```
API: แอป FastAPI ใต้ `lab10_fastapi/` (`POST /api/ask` รับ `question` + `conversation_context` ต่อได้)

### ข้อจำกัดที่รู้แล้ว / งานในอนาคต
- ผลรวมหมวดระดับ program/year-only/semester-only ไม่มี semantics (fail closed)
- นโยบายแบบ multi-turn, earliest-year, English/word-form credit filter: ไม่อยู่ใน scope
- CLI แสดง label แหล่งนโยบายเป็น `เล่มหลักสูตร: RULE` (contract ถูก, label รอ product decision)
