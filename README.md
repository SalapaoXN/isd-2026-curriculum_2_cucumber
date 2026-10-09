# CUCUMBER

**P2 ระบบถาม-ตอบข้อมูลหลักสูตรด้วย LLM**

CUCUMBER เป็นระบบถาม-ตอบข้อมูลหลักสูตรภาษาไทย โดยให้ **LLM ช่วยตีความว่าผู้ใช้ต้องการถามอะไร** แต่ข้อเท็จจริง เช่น รหัสวิชา หน่วยกิต ปี/เทอม แผนการเรียน และวิชาบังคับก่อน จะต้องมาจาก **ฐานข้อมูล SQLite และหลักฐานที่ระบบตรวจสอบแล้ว** เท่านั้น

ถ้าระบบระบุวิชา ขอบเขตหลักสูตร หรือหลักฐานได้ไม่ชัดเจน ระบบจะ **หยุดและขอข้อมูลเพิ่มหรือไม่ตอบ** แทนการคาดเดา (fail closed)

สมาชิก:
1. 67070049 Nattachai Kaewchum — Discord: GoodDee
2. 67070063 Thanachin Chukiatchai — Discord: วันลพ มีงบมาก
3. 67070103 Pongsakorn Panyacom — Discord: เบบี๋คือดวงใจ

## สถาปัตยกรรมของระบบปัจจุบัน

```text
เอกสารหลักสูตร / กฎของสถาบัน
→ OCR / ดึงข้อมูล / ตรวจแก้
→ ข้อมูล JSON มาตรฐาน
→ ฐานข้อมูล SQLite ที่ใช้ตอนรันระบบ
→ LLM ตีความความหมายของคำถามเป็น SemanticIntent
→ ตรวจความถูกต้องและรวมบริบทของบทสนทนา
→ ระบุหลักสูตร / ฉบับ / แผน / รายวิชาแบบ deterministic
→ วางแผนค้นหลักฐานและใช้ SQL แบบจำกัดขอบเขตเมื่อจำเป็น
→ VerifiedResult + แหล่งอ้างอิง (provenance)
→ สร้างคำตอบให้ผู้ใช้
```

หลักการแบ่งหน้าที่ของระบบ:

```text
การตีความภาษา          → LLM
การระบุตัวตน/ขอบเขต    → ตัว resolver แบบ deterministic
ข้อเท็จจริง              → SQLite มาตรฐาน
การตรวจสอบความน่าเชื่อถือ → หลักฐาน + provenance
การเรียบเรียงคำตอบ       → LLM แบบจำกัดขอบเขต หรือ renderer แบบ deterministic
```

Semantic QA ปัจจุบันใช้:

- สัญญารูปแบบ intent: `semantic-intent/v4`
- prompt สำหรับตีความคำถาม: `semantic-interpreter/v15`
- โหมดรันระบบ: `CUCUMBER_QA_MODE=semantic`

ระบบรองรับหลักสูตร AIT, BIT, DSBA, GENED และ IT โดยใช้ `catalog_key` แยก **ฉบับหลักสูตร** และใช้ `plan` แยก **แผนการเรียน** เช่น `coop` และ `no_coop`

## ความสามารถที่รองรับ

Semantic QA ปัจจุบันรองรับงานหลักดังนี้

- ค้นหารหัส ชื่อ คำอธิบาย และหน่วยกิตของรายวิชา
- ตรวจว่ารายวิชาเรียนปีไหน เทอมไหน และอยู่ในแผนใด
- ตรวจวิชาบังคับก่อน และกรณีที่ไม่มีวิชาบังคับก่อน
- แสดงรายวิชา นับจำนวนวิชา และรวมหน่วยกิตตามขอบเขตที่กำหนด
- ตอบหน่วยกิตรวมทั้งหลักสูตรจาก `program_requirements` ซึ่งเป็นแหล่งข้อมูลที่กำหนดไว้โดยตรง
- ค้นหารายวิชาที่เกี่ยวข้องกับหัวข้อจากคำอธิบายรายวิชา
- รองรับคำถามที่ระบุหลายวิชา โดยตรวจแต่ละวิชาแยกกัน
- ตรวจกลุ่มวิชาเลือก เช่น “ต้องเลือกกี่วิชาจากกลุ่มนี้”
- ตอบคำถามที่รวมข้อมูลรายวิชากับหน่วยกิตรวมของเทอมเดียวกัน
- เปรียบเทียบช่วงเวลาที่เรียนได้ระหว่างแผนต่าง ๆ และเก็บช่วงปี/เทอมที่เป็นไปได้ทั้งหมด
- เปรียบเทียบแผนที่มีอยู่ เมื่อผู้ใช้ถามว่า “ควรเลือกแผนไหน” โดยไม่ได้ระบุชื่อแผน
- เรียงลำดับรายวิชาตามปี/เทอมจากข้อมูลจริง โดยไม่สร้างความสัมพันธ์วิชาบังคับก่อนขึ้นมาเอง
- ตอบคำถามเกี่ยวกับกฎหรือข้อกำหนดของสถาบันที่มีข้อมูลรองรับ
- รองรับบริบทการสนทนาหลายเทิร์นผ่าน `next_context` โดยเก็บเฉพาะข้อมูลโครงสร้างที่จำเป็น

ถ้ารูปแบบคำถามยังไม่รองรับหรือหลักฐานไม่ครบ ระบบจะคืนสถานะขอข้อมูลเพิ่ม (`clarification_required`), หลักฐานไม่พอ (`insufficient_evidence`) หรือยังไม่รองรับ (`unsupported`) แทนการเดาคำตอบ

## แหล่งข้อมูลหลักของระบบ

```text
data/output/final/*_final.json
        +
data/output/final/institution_policy.json
        +
data/output/final/program_requirements.json
        ↓
python -m rag.build_index
        ↓
cucumber_outputs/runtime/curriculum.db
        ↓
Semantic QA / API / Web UI
```

บทบาทของไฟล์สำคัญ:

- `*_final.json` — ข้อมูลหลักสูตรที่ผ่านการตรวจแก้และใช้เป็นข้อมูลมาตรฐาน
- `institution_policy.json` — ข้อมูลกฎและข้อกำหนดของสถาบันในรูปแบบที่ระบบใช้งานได้
- `program_requirements.json` — ข้อมูลข้อกำหนดและหน่วยกิตรวมของแต่ละฉบับหลักสูตร
- `cucumber_outputs/runtime/curriculum.db` — ฐานข้อมูลข้อเท็จจริงที่ระบบใช้ตอนรัน
- `ground_truth/` — ใช้สำหรับทดสอบและประเมินผลเท่านั้น ไม่ใช่แหล่งข้อเท็จจริงของระบบ production
- `ground_truth/GT_FIXED.md` — บันทึกประวัติการแก้ Ground Truth
- `submission/` — เอกสาร submission รุ่นเก่าที่เก็บไว้เป็นหลักฐาน ไม่ใช่สถานะปัจจุบันของระบบ

ถ้าแก้ข้อมูลมาตรฐานของหลักสูตร ต้องสร้าง runtime DB ใหม่ด้วย:

```powershell
python -m rag.build_index
```

## วิธีเปิดเว็บ

ต้องใช้ Python 3.10+ และ Node.js

### 1. เตรียม Python

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r backend/requirements.txt
```

สร้างไฟล์ `.env` ที่ root ของโปรเจกต์:

```dotenv
GEMINI_API_KEY=your_key_here
```

ห้าม commit `.env` หรือ API key

### 2. Build frontend

```powershell
cd frontend
npm install
npm run build
cd ..
```

### 3. เปิด backend ในโหมด semantic

```powershell
$env:CUCUMBER_QA_MODE = "semantic"
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

หรือใช้สคริปต์:

```powershell
.\scripts\run_semantic.ps1
```

เปิดหน้าเว็บ:

```text
http://127.0.0.1:8000/chat
http://127.0.0.1:8000/curriculum
http://127.0.0.1:8000/api/health
```

ก่อนเดโมควรตรวจ `GET /api/health` ให้ได้:

```text
status = ok
database_ready = true
qa_mode = semantic
```

> `rag/semantic/modes.py` ยังตั้งค่าเริ่มต้นเป็น `legacy` เพื่อความปลอดภัย ดังนั้นถ้าต้องการรัน Semantic QA ต้องกำหนด `CUCUMBER_QA_MODE=semantic` ให้ชัดเจน

### โหมดพัฒนา frontend

```powershell
# backend: port 8000
cd frontend
npm run dev
```

เปิด `http://127.0.0.1:5173`

## รูปแบบ API

### `POST /api/ask`

ตัวอย่าง request:

```json
{
  "question": "DSBA ปี 1 เทอม 1 มีวิชาอะไรบ้าง",
  "conversation_context": {
    "program": "DSBA",
    "catalog_key": "dsba-2565"
  }
}
```

ตัวอย่าง response จาก semantic path:

```json
{
  "question": "...",
  "answer": "...",
  "status": "answer",
  "action": null,
  "route": "semantic",
  "provenance": [],
  "next_context": {},
  "comparison": null
}
```

ในคำถามถัดไปควรส่ง `next_context` จาก response ก่อนหน้ากลับมาเป็น `conversation_context`

ระบบเก็บเฉพาะข้อมูลโครงสร้างที่จำเป็น เช่น หลักสูตร (`program`), ฉบับหลักสูตร (`catalog_key`), แผน (`plan`), ปี, เทอม, วิชาที่กำลังอ้างถึง และรายการวิชาจากผลลัพธ์ก่อนหน้า โดยไม่ใช้ข้อความคำตอบเก่าเป็นข้อเท็จจริงของเทิร์นใหม่

สถานะที่อาจพบเมื่อระบบไม่สามารถตอบได้ทันที:

- `clarification_required` — ต้องการข้อมูลจากผู้ใช้เพิ่ม
- `insufficient_evidence` — หลักฐานไม่เพียงพอ
- `unsupported` — รูปแบบคำถามยังไม่รองรับ

### Curriculum API

```text
GET /api/programs
GET /api/curriculum?program=&catalog_key=&plan=&year=&semester=&search=&limit=&offset=
GET /api/courses/{course_code}?program=&catalog_key=
GET /api/health
```

## การใช้งานผ่านคำสั่ง CLI

ถามหนึ่งคำถาม:

```powershell
python scripts/ask.py "IT ปี 2 เทอม 1 เรียนวิชาอะไรบ้าง"
```

หรือเปิดโหมดถามต่อเนื่อง:

```powershell
python scripts/ask.py
```

Web chat รองรับบริบทการสนทนาได้มากกว่า CLI loop

## กระบวนการเตรียมข้อมูล

```text
OCR → ดึงข้อมูล → รวมข้อมูล → ตรวจแก้ → ประเมินผล → สร้างดัชนีและฐานข้อมูล
```

ตัวอย่างคำสั่ง:

```powershell
python -m src.pipeline.run --dataset it2565 --dry-run
python -m src.pipeline.run --dataset it2565 --with-index
```

รายละเอียดเพิ่มเติม: `src/pipeline/README.md`

## การทดสอบ

รันชุดทดสอบ Python ทั้งหมด:

```powershell
python -m unittest discover -s tests -t .
```

สถานะล่าสุดก่อนปิดการพัฒนา G5-C:

- ชุดทดสอบแบบไม่เรียก provider ภายนอก รันทั้งหมด **2,804 tests**
- **0 failures**, **0 errors**, **3 skipped**
- ชุดทดสอบ placement sequence ผ่าน **35/35**
- ชุด semantic regression ที่เกี่ยวข้องหลังแก้ลำดับ `result_courses` ผ่าน **140/140**

ผลประเมินคุณภาพข้อมูลของชุดที่มี Ground Truth อยู่ใน `reports/README.md`

ตัวเลข 100% ที่รายงานในส่วนนั้นหมายถึง **ความครบถ้วนของจำนวน record** ไม่ได้หมายความว่าทุกข้อความหรือทุก field ถูกต้องระดับตัวอักษร 100%

ไฟล์ใน `eval/results/*.md` และรายงาน hardening ที่มีวันที่ เป็น **ผลการทดสอบย้อนหลังของแต่ละช่วงเวลา** ไม่ควรใช้แทนสถานะ implementation ปัจจุบันโดยตรง

## ข้อจำกัดที่ระบบตั้งใจไม่เดาคำตอบ

- ไม่ยืนยันว่ารายวิชาจะเปิดสอนจริงในอนาคต เพราะ runtime ไม่มีข้อมูลการเปิดสอนจริงของภาคเรียนอนาคต
- ไม่สร้างคะแนนผ่าน English Exit ขึ้นเอง ถ้าแหล่งข้อมูลไม่มีระบุ
- ไม่ตัดสินสิทธิ์ส่วนบุคคล เช่น “GPA เท่านี้ลงสหกิจได้ไหม” ถ้ายังไม่มีเกณฑ์ทางการครบ
- ไม่รวมข้อมูลข้าม `catalog_key` เพื่อบังคับให้คำถามตอบได้
- ถ้าผู้ใช้ระบุหลายวิชา ระบบต้องระบุและตรวจได้ครบทุกวิชา มิฉะนั้นจะหยุดแทนการตัดบางวิชาออก
- การเรียงลำดับวิชาตามปี/เทอมไม่ถือว่าเป็นหลักฐานว่าวิชาก่อนหน้าคือวิชาบังคับก่อน
- ถ้าหลายวิชาอยู่เทอมเดียวกันหรือมีช่วงปี/เทอมที่ซ้อนกันจนเรียงลำดับเดียวไม่ได้ ระบบจะไม่สร้างลำดับขึ้นเอง
- ถ้าคำถามรวมเงื่อนไขหลายแบบที่ระบบยังไม่รองรับ ระบบจะไม่ตัดเงื่อนไขที่ยากทิ้งแล้วตอบเฉพาะส่วนที่ง่ายกว่า
- ถ้าคำถามต่อเนื่องอ้างถึง “วิชาที่ 2” หรือผลลัพธ์ก่อนหน้า แต่ระบบระบุไม่ได้อย่างปลอดภัย จะไม่ย้อนกลับไปเลือกวิชาเก่าแบบเงียบ ๆ
- ข้อความจาก LLM ไม่ถือเป็นข้อเท็จจริงของระบบโดยตรง

## เอกสารที่เกี่ยวข้อง

เริ่มจาก `docs/README.md` ซึ่งแบ่งเอกสารปัจจุบันออกจากรายงานย้อนหลังไว้แล้ว

เอกสารหลัก:

- `README.md` — ภาพรวมระบบ วิธีติดตั้ง วิธีรัน API และความสามารถปัจจุบัน
- `docs/semantic-qa-vnext.md` — สถาปัตยกรรมของ Semantic QA
- `rag/README.md` — โครงสร้างการทำงานของ QA/RAG
- `src/pipeline/README.md` — กระบวนการ OCR จนถึงฐานข้อมูล
- `docs/academic_rules.md` — ระบบกฎและข้อกำหนดของสถาบัน
- `reports/README.md` — วิธีอ่านผลประเมินคุณภาพข้อมูลและรายงานต่าง ๆ
- `ground_truth/GT_FIXED.md` — ประวัติการแก้ Ground Truth

ไฟล์จาก Week 11:

- `docs/wireframes/chat-wireframe.svg`
- `docs/wireframes/curriculum-wireframe.svg`
- `docs/wireframes/user-flow.svg`
- `frontend/`

`submission/` เป็น submission รุ่นเก่าที่เก็บไว้เป็นประวัติ และไม่ได้ถูกแก้ในการปรับเอกสารรอบนี้
