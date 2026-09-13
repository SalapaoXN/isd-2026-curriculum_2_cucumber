# isd-2026-curriculum_2_cucumber

We do OCR curriculum and some LLM with model name **CUCUMBER**

Project: **P2 LLM ถาม-ตอบหลักสูตร**

Member:
1. 67070049 Nattachai Kaewchum >> Discord: GoodDee
2. 67070063 Thanachin Chukiatchai >> Discord: วันลพ มีงบมาก
3. 67070103 Pongsakorn Panyacom >> Discord: เบบี๋คือดวงใจ

## CUCUMBER

CUCUMBER คือระบบที่นำเอกสารหลักสูตรมาแปลงเป็นข้อมูลที่มีโครงสร้าง และใช้ข้อมูลนั้นตอบคำถามเกี่ยวกับหลักสูตรเป็นภาษาไทย

เป้าหมายหลักคือให้คำตอบอ้างอิงกลับไปยังข้อมูลต้นทางได้ เพื่อให้ตรวจสอบได้ว่าคำตอบมาจากหน้าใดหรือไฟล์ใดของหลักสูตร

---

## Architecture

ระบบแบ่งการทำงานออกเป็นหลายช่วง โดยแต่ละช่วงจะสร้างไฟล์ผลลัพธ์เก็บไว้

ข้อดีคือ ถ้าทำขั้นตอนหนึ่งเสร็จแล้ว สามารถเริ่มทำต่อจากไฟล์ที่มีอยู่ได้ โดยไม่ต้องรัน OCR ใหม่ทุกครั้ง

```text
Part 1 — OCR
inputs/
  -> python ocr.py --prefix <program>
  -> outputs/ocr/

Part 2 — Data preparation
outputs/ocr/
  -> python prepare_data.py
  -> outputs/extracted/
  -> outputs/consolidated/
  -> python llm_spell_corrector.py
  -> outputs/llm/
  -> python evaluate.py
  -> reports/evaluation/

Part 3 — RAG / QA
outputs/llm/*_corrected.json
  -> python -m rag.build_index
  -> cucumber_outputs/runtime/curriculum.db
  -> python ask.py
```

สรุปง่าย ๆ คือ

```text
เอกสารภาพ
→ OCR อ่านข้อความ
→ จัดข้อมูลหลักสูตร
→ แก้ข้อความบางส่วน
→ ตรวจผล
→ สร้างฐานข้อมูล
→ ถามคำถามกับระบบ
```

---

## Installation

ให้รันคำสั่งจาก root ของ repository

สร้าง virtual environment และติดตั้ง dependency:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt -r requirements-rag.txt
```

สำหรับ macOS/Linux ใช้:

```bash
source .venv/bin/activate
```

ถ้าจะใช้ขั้นตอนที่เรียก Gemini ให้สร้างไฟล์ `.env` ในเครื่อง:

```dotenv
GEMINI_API_KEY=...
HF_TOKEN=...
```

ความหมายของแต่ละตัว:

- `GEMINI_API_KEY` ใช้ในขั้นตอนแก้ข้อความด้วย LLM และการสร้างคำตอบใน `ask.py`
- `HF_TOKEN` ไม่จำเป็นเสมอไป และใช้เฉพาะบางกรณีที่เกี่ยวข้องกับโมเดลจาก Hugging Face

ห้าม commit ไฟล์ `.env` หรือเผยแพร่ API key

บางโมเดลที่ใช้กับ OCR หรือ RAG อาจถูกดาวน์โหลดอัตโนมัติในครั้งแรกที่ใช้งาน

---

## Part 1: OCR

OCR เป็นขั้นตอนที่ใช้เวลาและทรัพยากรค่อนข้างมาก จึงแยกออกมาเป็นขั้นตอนของตัวเอง

ระบบจะอ่านภาพจาก:

```text
inputs/<program>/
```

แล้วบันทึกผล OCR ลงที่:

```text
outputs/ocr/<program>/
```

คำสั่งปกติ:

```powershell
python ocr.py --prefix it
```

ถ้าต้องการเลือกเฉพาะบางหน้า:

```powershell
python ocr.py --prefix it --pages 32-38
```

ตัวอย่าง:

```text
--prefix it
```

หมายถึง

```text
inputs/it/
→ outputs/ocr/it/
```

prefix ที่รองรับ:

- `ait`
- `bit`
- `dsba`
- `gened`
- `it`

ถ้าไม่ใส่ `--pages` ระบบจะ OCR ทุกภาพที่มีใน program นั้น

ถ้าต้องการใช้ CPU แทน GPU:

```powershell
python ocr.py --prefix it --no-gpu
```

การเลือก program และ plan สำหรับการเตรียมข้อมูลจะทำใน `prepare_data.py` ไม่ได้ทำในคำสั่ง OCR ปกติ

คำสั่งเดิม:

```powershell
python -m src.run_pipeline ...
```

ยังสามารถใช้ได้สำหรับ replay หรือ debug โดยเฉพาะ

---

## Part 2: Data Preparation

หลังจากมีผล OCR แล้ว ให้รันตามลำดับนี้:

```powershell
python prepare_data.py
python llm_spell_corrector.py
python evaluate.py
```

### `prepare_data.py`

ไฟล์นี้มีหน้าที่นำผล OCR มาจัดเป็นข้อมูลหลักสูตรที่มีโครงสร้าง

ระบบจะค้นหาโฟลเดอร์ OCR ที่รองรับและมีข้อมูลอยู่จริง แล้วทำขั้นตอน extraction และ merge ตาม program, plan และช่วงหน้าที่กำหนดไว้

ถ้า program ใดไม่มีข้อมูล ระบบจะข้าม program นั้น

ถ้ามีโฟลเดอร์ที่ชื่อไม่ตรงกับ program ที่รองรับ ระบบจะแจ้งให้ทราบ

ขั้นตอนนี้ไม่ทำ:

- OCR
- LLM correction
- evaluation
- RAG

ผลลัพธ์จะถูกเก็บไว้หลัก ๆ ที่:

```text
outputs/extracted/
outputs/consolidated/
```

### `llm_spell_corrector.py`

ขั้นตอนนี้ใช้ LLM ช่วยแก้ข้อความบางส่วนหลังจาก extraction แล้ว

ระบบจะค้นหาไฟล์รูปแบบ:

```text
outputs/consolidated/**/full/merged_*_full.json
```

จากนั้นสร้างไฟล์ที่แก้แล้วไว้ที่:

```text
outputs/llm/
```

ไฟล์สำคัญที่ได้ เช่น:

```text
*_corrected.json
*_corrections.json
```

`*_corrected.json` คือข้อมูลที่แก้แล้ว

`*_corrections.json` คือบันทึกว่ามีการแก้อะไรบ้าง

### `evaluate.py`

ใช้ตรวจคุณภาพของข้อมูลหลังการแก้

ระบบจะอ่าน:

```text
outputs/llm/*_corrected.json
```

แล้วเปรียบเทียบกับ ground truth ที่กำหนดไว้สำหรับ program/plan นั้น

รายงานจะถูกเก็บที่:

```text
reports/evaluation/
```

ตัวอย่าง metric ที่มี:

- CER
- WER
- Course-record Coverage Precision
- Course-record Coverage Recall
- Course-record Coverage F1

ความหมายแบบง่าย:

- CER/WER ใช้วัดความผิดพลาดของข้อความ
- Coverage ใช้วัดว่าระบบเก็บ record รายวิชาได้ครบแค่ไหน

Coverage ไม่ได้ใช้วัดว่าคำสะกดถูกหรือผิด

---

## Part 3: RAG / QA

ขั้นตอนนี้ใช้ข้อมูลที่ผ่านการแก้แล้วมาสร้างฐานข้อมูลสำหรับระบบถาม-ตอบ

สร้างหรือสร้างใหม่ runtime database:

```powershell
python -m rag.build_index
```

ข้อมูลต้นทางของ RAG คือ:

```text
outputs/llm/*_corrected.json
```

ฐานข้อมูลที่สร้างขึ้นจะอยู่ที่:

```text
cucumber_outputs/runtime/curriculum.db
```

`outputs/consolidated/` ไม่ใช่ข้อมูลที่ RAG อ่านโดยตรงในขั้นตอนนี้

### ถามคำถามหนึ่งข้อ

```powershell
python ask.py "IT ปี 2 เทอม 1 เรียนวิชาอะไรบ้าง"
```

### เปิดโหมดถามต่อเนื่อง

```powershell
python ask.py
```

ออกจากโหมดถามต่อเนื่องได้ด้วย:

```text
exit
quit
```

หรือ EOF

ระบบจะเลือกวิธีค้นหาให้อัตโนมัติระหว่าง:

- `structured`
- `semantic`
- `hybrid`

ผู้ใช้ไม่ต้องเลือกเอง

ความหมายแบบง่าย:

- **structured** ใช้กับข้อมูลที่เป็นช่องชัดเจน เช่น รหัสวิชา ปี เทอม หน่วยกิต หรือ prerequisite
- **semantic** ใช้ค้นหาจากความหมายของเนื้อหารายวิชา เช่น ถามว่าเรียนเกี่ยวกับอะไร
- **hybrid** ใช้ทั้งสองแบบร่วมกัน

ก่อนใช้ `ask.py` ต้องมี runtime database อยู่ก่อน

ถ้ายังไม่มี ให้รัน:

```powershell
python -m rag.build_index
```

`ask.py` จะไม่สร้างฐานข้อมูลให้เองโดยอัตโนมัติ

รูปแบบ output ปกติ:

```text
ถาม: <question>
ตอบ: <final answer>
แหล่งข้อมูล: <existing provenance/evidence>
```

ถ้าหลักสูตรไม่มีข้อมูลที่ถาม ระบบจะตอบข้อความนี้:

```text
ไม่พบข้อมูลนี้ในเล่มหลักสูตร
```

---

## Artifact Boundaries / Repository Structure

โฟลเดอร์สำคัญของโปรเจกต์:

```text
inputs/                         source images
outputs/ocr/                    persistent OCR artifacts
outputs/extracted/              extraction artifacts
outputs/consolidated/           merged curriculum artifacts
outputs/llm/                    corrected downstream corpus and logs
reports/evaluation/             evaluation reports
ground_truth/                   accepted evaluation references
cucumber_outputs/runtime/       generated RAG database
src/                            OCR implementation
rag/                            indexing, routing, retrieval, and QA
tests/                          focused and regression tests
submission/                     separate frozen submission package
```

อธิบายแบบง่าย:

- `inputs/` — ภาพเอกสารต้นฉบับ
- `outputs/ocr/` — ข้อความที่อ่านจาก OCR
- `outputs/extracted/` — ข้อมูลที่แยกออกจากผล OCR
- `outputs/consolidated/` — ข้อมูลที่รวมและจัดให้อยู่ในรูปเดียวกัน
- `outputs/llm/` — ข้อมูลหลังผ่านการแก้ข้อความ
- `reports/evaluation/` — ผลการประเมินคุณภาพข้อมูล
- `ground_truth/` — ข้อมูลอ้างอิงที่ใช้ตรวจผล
- `cucumber_outputs/runtime/` — ฐานข้อมูลที่ใช้ตอนถาม-ตอบ
- `src/` — โค้ด OCR
- `rag/` — โค้ด RAG และระบบถาม-ตอบ
- `tests/` — ชุดทดสอบ
- `submission/` — ชุดไฟล์สำหรับส่งงาน

ไฟล์ใน `outputs/`, `reports/` และ runtime database เป็นไฟล์ที่สามารถสร้างใหม่ได้จาก pipeline

`outputs/llm/` เป็นข้อมูลปลายทางที่ใช้สำหรับสร้าง RAG และสามารถเก็บไว้เพื่อให้คนอื่นสร้างฐานข้อมูลหรือทดสอบระบบได้โดยไม่ต้องรัน OCR และ LLM correction ใหม่

`submission/` เป็นชุดไฟล์สำหรับส่งงานโดยเฉพาะ และไม่ใช่ input ปกติของ runtime pipeline

---

## Optional Debugging / Replay

ถ้าต้องการ debug เป็นบางขั้นตอน สามารถรันคำสั่งแยกได้

ตัวอย่าง extraction:

```powershell
python extract.py outputs/ocr/it --output-dir outputs/extracted --program IT --plan no_coop
```

ตัวอย่าง merge:

```powershell
python merge_consecutive.py --prefix it --plan no_coop -p 32-38,328-371 -d 328-371
```

คำสั่งเหล่านี้มีไว้สำหรับ debug หรือ replay บางขั้นตอน และไม่จำเป็นสำหรับ workflow ปกติ

โมดูล:

```text
rag.hybrid_demo
```

ยังมีไว้สำหรับ development/demo

ส่วน interface ปกติสำหรับผู้ใช้คือ:

```text
ask.py
```

---

## Evaluation

ระบบ evaluation ใช้ตรวจคุณภาพข้อมูลที่ได้จาก OCR และขั้นตอน extraction/correction

metric หลักที่ใช้ เช่น:

- CER
- WER
- Coverage Precision
- Coverage Recall
- Coverage F1

CER/WER เน้นวัดความผิดพลาดของข้อความ

Coverage เน้นวัดว่าระบบเก็บข้อมูลรายวิชาครบหรือไม่

ค่าผลลัพธ์ปัจจุบันจะอยู่ใน:

```text
reports/evaluation/
```

README จะไม่เขียนตัวเลขผลลัพธ์ตายตัว เพราะค่าอาจเปลี่ยนเมื่อมีการรันข้อมูลใหม่

---

## Testing

ตัวอย่างการรัน test เฉพาะส่วน:

```powershell
python -m unittest tests.test_ask
python -m unittest tests.test_prepare_data
python -m unittest tests.test_llm_spell_corrector
python -m unittest tests.test_rag_qa tests.test_rag_hybrid_demo
python -m unittest tests.test_evaluate tests.test_evaluate_gold_questions
```

ถ้าต้องการรัน test ทั้งหมด:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

---

## Supported Programs / Plans

ขอบเขตข้อมูลที่ระบบ preparation รองรับในปัจจุบัน:

- AIT
- BIT: `coop`, `no_coop`
- DSBA: `coop`, `no_coop`
- GENED
- IT: `coop`, `no_coop`

ความหมายของ plan:

- `coop` = แผนสหกิจ
- `no_coop` = แผนไม่สหกิจ
