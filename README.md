# CUCUMBER

**P2 LLM ถาม-ตอบหลักสูตร**

CUCUMBER เป็นระบบแปลงข้อมูลหลักสูตรให้เป็นฐานข้อมูลที่ตรวจสอบย้อนกลับได้ แล้วใช้ตอบคำถามภาษาไทยแบบ **grounded** โดยให้ข้อเท็จจริงมาจาก canonical data / SQLite ไม่ใช่จากความจำของ LLM

สมาชิก:
1. 67070049 Nattachai Kaewchum — Discord: GoodDee
2. 67070063 Thanachin Chukiatchai — Discord: วันลพ มีงบมาก
3. 67070103 Pongsakorn Panyacom — Discord: เบบี๋คือดวงใจ

---

## 1. ภาพรวมระบบ

```text
เอกสารหลักสูตร
→ OCR / Extraction / Merge
→ LLM-assisted correction
→ Canonical JSON
→ SQLite + Semantic Index
→ Query understanding
→ Evidence retrieval
→ Deterministic aggregation
→ Grounded answer + Provenance
```

ระบบรองรับข้อมูลหลักสูตร AIT, BIT, DSBA, GENED และ IT โดยแยก **ฉบับหลักสูตร** ด้วย `catalog_key` และแยกแผน เช่น `coop` / `no_coop` เมื่อมี

หลักการสำคัญ:

- `catalog_key` = ตัวตนของฉบับหลักสูตร
- `program` = หลักสูตร เช่น IT, DSBA
- `plan` = แผนการเรียนภายในฉบับ
- ปี/เทอม/หน่วยกิต/prerequisite ต้องมาจากหลักฐานที่ตรวจสอบได้
- LLM ใช้ช่วยตีความภาษาและงานที่ถูกจำกัดขอบเขต แต่ไม่ใช่ factual authority
- ถ้าหลักฐานไม่พอ ระบบจะ clarify / fail closed แทนการเดา
- backend เป็น stateless; การสนทนาหลายเทิร์นใช้ `next_context` ที่ฝั่ง client ส่งกลับมา โดยเก็บเฉพาะ scope/ตัวอ้างอิงที่มีขอบเขต ไม่ใช้ประวัติแชตทั้งหมดเป็น factual authority

หน้าเว็บหลัก:

- `/chat` — ถามคำถามแบบมี scope และ conversation context
- `/curriculum` — ค้นและดูข้อมูลรายวิชาตาม program / edition / plan / year / semester

---

## 2. Source of truth

ลำดับข้อมูล production:

```text
data/output/final/*_corrected.json
        ↓
python -m rag.build_index
        ↓
cucumber_outputs/runtime/curriculum.db
        ↓
QA / API / Web UI
```

นอกจาก curriculum corpus แล้ว การสร้าง DB แบบ default จะโหลด supplemental authority สองไฟล์อย่างชัดเจน:

- `data/output/final/institution_policy.json`
- `data/output/final/program_requirements.json`

บทบาทของไฟล์สำคัญ:

- `data/output/final/*_corrected.json` — canonical curriculum corpus
- `institution_policy.json` — ข้อกำหนด/นโยบายสถาบันที่ผ่านการจัดโครงสร้าง
- `program_requirements.json` — ข้อกำหนดหน่วยกิตรวมแบบผูกกับ `catalog_key`; คำถาม “หลักสูตรนี้รวมกี่หน่วยกิต” ใช้ authority นี้แทนการบวก placement รายเทอม
- `cucumber_outputs/runtime/curriculum.db` — runtime DB ที่สร้างจากข้อมูลข้างต้น
- `ground_truth/` — ใช้ evaluation/test เท่านั้น ไม่ใช่ production authority
- `ground_truth/GT_FIXED.md` — audit log ว่า Ground Truth เคยแก้อะไร จากค่าใด เป็นค่าใด และเพราะอะไร
- `submission/` — **frozen historical submission** ที่เคยส่งแล้ว ไม่ใช่ runtime ปัจจุบัน

ถ้าแก้ canonical data ให้สร้าง runtime DB ใหม่:

```powershell
python -m rag.build_index
```

---

## 3. Run the web app

ต้องใช้ Python 3.10+ และ Node.js

> สำหรับเครื่องที่เพิ่ง clone repo ใหม่ ต้องทำ **ทุกขั้นตามลำดับ** ด้านล่าง โดยเฉพาะ `npm run build` ก่อนเปิดเว็บผ่าน FastAPI ที่พอร์ต `8000`

### 3.1 Python environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r backend/requirements.txt
```

### 3.2 Gemini

สร้าง `.env` ที่ root:

```dotenv
GEMINI_API_KEY=your_key_here
```

ห้าม commit `.env` หรือ API key

provider ปัจจุบันใช้:

```text
gemini-3.5-flash-lite
```

### 3.3 Build frontend — ห้ามข้ามเมื่อรันผ่าน FastAPI

รันจาก root repo:

```powershell
cd frontend
npm install
npm run build
cd ..
```

หลัง build สำเร็จควรมีไฟล์ประมาณนี้:

```text
frontend/
└─ dist/
   ├─ index.html
   └─ assets/
```

FastAPI จะ serve React bundle จาก `frontend/dist/` หากยังไม่มี `dist` ระบบจะ fallback ไปที่ `frontend/index.html` ซึ่งเป็น Vite development entry และ browser จะร้องขอ `/src/main.jsx`; FastAPI production-style server ไม่ได้ serve path นี้ จึงจะเห็น `GET /src/main.jsx 404 Not Found`

### 3.4 Start backend

หลังจาก build frontend แล้ว ให้กลับมาที่ root repo และรัน:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

เปิด:

```text
http://127.0.0.1:8000/chat
http://127.0.0.1:8000/curriculum
http://127.0.0.1:8000/api/health
```

ถ้า log มี:

```text
GET /src/main.jsx 404 Not Found
```

ให้หยุด server แล้วรัน `npm install` และ `npm run build` ในโฟลเดอร์ `frontend` ก่อน จากนั้นจึง start backend ใหม่

`GET /favicon.ico 404 Not Found` ไม่กระทบการทำงานของเว็บ

### 3.5 Frontend development mode

ถ้าต้องการใช้ Vite dev server แทน built frontend:

1. รัน backend ที่พอร์ต `8000`
2. เข้า `frontend` แล้วรัน `npm run dev`
3. เปิดเว็บที่ `http://127.0.0.1:5173`

ทั้งโหมด built frontend และ Vite development ใช้ backend ที่พอร์ต `8000` เหมือนกัน โดย Vite config ปัจจุบัน proxy `/api` ไปที่ `http://127.0.0.1:8000`

---

## 4. API Contract

### `POST /api/ask`

Request:

```json
{
  "question": "DSBA ปี 1 เทอม 1 มีวิชาอะไรบ้าง",
  "conversation_context": {
    "program": "DSBA",
    "catalog_key": "dsba-2565"
  }
}
```

- `question`: string, 2–500 ตัวอักษร
- `conversation_context`: object หรือ `null`
- เทิร์นถัดไปควรส่ง `next_context` จาก response กลับมาเป็น `conversation_context`
- context เป็น state แบบ bounded และ client-held: เก็บ scope ที่ตรวจสอบแล้ว เช่น program/catalog/plan/year/semester รวมถึงตัวอ้างอิงรายวิชา/ผลลัพธ์/คำตอบก่อนหน้าที่จำเป็นต่อ follow-up เท่านั้น ไม่ใช่การเก็บ transcript ทั้งหมดหรือ cached factual answer

Response หลัก:

```json
{
  "question": "...",
  "answer": "...",
  "status": "answer",
  "action": null,
  "route": "llm_sql",
  "provenance": [],
  "next_context": {}
}
```

response อาจมี field เพิ่ม เช่น `comparison`, `plan_results`, `hard_task_type`

กรณีข้อมูลไม่พอหรือ scope ไม่ชัด ระบบจะคืนสถานะ เช่น `insufficient_evidence` หรือ `clarification_required` แทนการสร้างข้อเท็จจริงเอง

### Curriculum API

```text
GET /api/programs
GET /api/curriculum?program=&catalog_key=&plan=&year=&semester=&search=&limit=&offset=
GET /api/courses/{course_code}?program=&catalog_key=
GET /api/health
```

FastAPI error response ใช้ field `detail`; validation error อาจเป็น structured list

Frontend ฝั่ง Chat มีสถานะ UX หลักครบ: Idle, Loading, Success และ Error

---

## 5. CLI

ถามหนึ่งข้อ:

```powershell
python scripts/ask.py "IT ปี 2 เทอม 1 เรียนวิชาอะไรบ้าง"
```

โหมดต่อเนื่อง:

```powershell
python scripts/ask.py
```

CLI loop นี้ไม่ได้เก็บ conversation context ระหว่างคำถามแบบ web chat

---

## 6. Pipeline

entry point:

```powershell
python -m src.pipeline.run
```

flow:

```text
ocr → extract → merge → correct → evaluate → build_index
```

ตัวอย่าง:

```powershell
python -m src.pipeline.run --program it --dry-run
python -m src.pipeline.run --program it --with-index
python -m src.pipeline.run --program it --pages 32-38
```

รายละเอียด: `src/pipeline/README.md`

---

## 7. Testing และ Evaluation

รัน test suite:

```powershell
python -m unittest discover -s tests -t .
```

evaluation ล่าสุดถูก regenerate เมื่อ 2026-10-04 จาก **8 GT-backed current scopes** ที่มี Ground Truth ตรงกัน

ผล record coverage:

```text
GT records:         839
Prediction records: 839
Matched:            839
Precision/Recall/F1: 100%
```

100% ตรงนี้หมายถึง **record coverage** ไม่ได้หมายความว่าทุก field ตรง 100%; รายละเอียด CER/WER และ error rows อยู่ที่ `reports/README.md` และ `reports/evaluation/`

ไฟล์ corrected ของ historical editions ที่ยังไม่มี edition-specific Ground Truth จะไม่ถูกนำมาปนกับ metric ชุดนี้ เพื่อหลีกเลี่ยงการเทียบคนละฉบับหลักสูตร

Runtime benchmark ล่าสุดอยู่ที่ `reports/runtime_benchmark.md` โดย snapshot ปัจจุบันชี้ว่า local parsing/SQLite ใช้เวลาเพียงระดับ sub-ms ถึงไม่กี่ ms ขณะที่ latency หลักมาจาก external model call

---

## 8. ข้อจำกัดที่ตั้งใจ fail closed

- ระบบไม่มีข้อมูลการเปิดสอนจริงของรายวิชาในภาคเรียนอนาคต จึงไม่ยืนยันว่า “ถอน/ตกแล้วจะเปิดให้ลงใหม่เทอมไหน”
- หาก policy ระบุว่าต้องผ่าน English Exit แต่ canonical evidence ไม่มีคะแนนผ่าน ระบบจะไม่สร้างคะแนนขึ้นมาเอง
- คำถามสิทธิ์ส่วนบุคคล เช่น “GPA เท่านี้ลงสหกิจได้ไหม” จะไม่ถูกฟันธงถ้าฐานข้อมูลยังไม่มีเกณฑ์เฉพาะของคณะ/หลักสูตรเพียงพอ
- คำถามที่ต้องใช้หลักฐานนอก canonical curriculum / policy data จะไม่ถูกเดาคำตอบ
- program ที่มีหลายฉบับต้องรักษา `catalog_key` ไม่รวมข้อมูลข้าม edition
- ordinal/follow-up ที่อ้างอิงเป้าหมายไม่ได้อย่างปลอดภัยจะ fail closed แทนการย้อนกลับไปใช้ referent เก่า
- semantic result ยังขึ้นกับคุณภาพ course description และหลักฐานที่มีจริง

---

## 9. เอกสารที่ควรอ่าน

เอกสาร active ถูกลดให้เหลือเฉพาะส่วนที่มีหน้าที่ชัดเจน:

- `README.md` — ภาพรวม, setup, API contract
- `rag/README.md` — QA/RAG architecture ปัจจุบัน
- `src/pipeline/README.md` — OCR → canonical data → DB
- `docs/academic_rules.md` — policy/rules subsystem
- `docs/wireframes/` — Week 11 low-fidelity wireframes และ user flow
- `reports/README.md` — วิธี evaluation และ report artifacts
- `reports/runtime_benchmark.md` — latency benchmark
- `ground_truth/GT_FIXED.md` — audit log ของการแก้ Ground Truth
- `submission/submission.md` — frozen historical submission; เก็บเพื่ออ้างอิงเท่านั้น

เอกสาร phase/baseline/history รุ่นเก่าถูกนำออกจาก active tree เพื่อไม่ให้ปนกับสถานะปัจจุบัน

## Week 11 Deliverables

- API Contract: README.md → Section 4
- Chat Wireframe: docs/wireframes/chat-wireframe.svg
- Curriculum Wireframe: docs/wireframes/curriculum-wireframe.svg
- User Flow: docs/wireframes/user-flow.svg
- Implemented UI: frontend/