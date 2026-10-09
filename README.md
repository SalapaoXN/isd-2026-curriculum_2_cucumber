# CUCUMBER

**P2 LLM ถาม-ตอบหลักสูตร**

CUCUMBER เป็นระบบถาม-ตอบข้อมูลหลักสูตรภาษาไทยที่ให้ **LLM ทำหน้าที่ตีความภาษา** แต่ให้ **canonical SQLite + deterministic evidence เป็น factual authority** คำตอบเชิงข้อเท็จจริงจึงต้องมีหลักฐานรองรับและ fail closed เมื่อ identity, scope หรือ evidence ไม่พอ

สมาชิก:
1. 67070049 Nattachai Kaewchum — Discord: GoodDee
2. 67070063 Thanachin Chukiatchai — Discord: วันลพ มีงบมาก
3. 67070103 Pongsakorn Panyacom — Discord: เบบี๋คือดวงใจ

## Current architecture

```text
Curriculum / policy sources
→ OCR / extraction / correction
→ canonical JSON
→ runtime SQLite
→ LLM SemanticIntent (language only)
→ deterministic validation + context merge
→ deterministic program/catalog/plan/course resolution
→ deterministic planner / evidence / guarded SQL where needed
→ VerifiedResult + provenance
→ bounded answer presentation
```

Authority rule:

```text
LANGUAGE      → LLM
IDENTITY/SCOPE→ deterministic resolver
FACTS         → canonical SQLite
TRUST         → evidence + provenance
PRESENTATION  → bounded LLM or deterministic renderer
```

Semantic production path ปัจจุบันใช้:

- semantic intent contract: `semantic-intent/v4`
- interpreter prompt: `semantic-interpreter/v15`
- runtime mode: `CUCUMBER_QA_MODE=semantic`

ระบบรองรับ AIT, BIT, DSBA, GENED และ IT โดยแยก curriculum edition ด้วย `catalog_key` และแยก study plan เช่น `coop` / `no_coop`

## Supported QA surface

ปัจจุบัน semantic path รองรับ capability หลักดังนี้

- exact course identity / name / description / credits
- placement ตามปี/เทอม/แผน
- direct prerequisites และ explicit no-prerequisite
- scoped course lists, counts และ credit totals
- whole-program total credits จาก authoritative `program_requirements`
- semantic topic discovery จาก course descriptions
- explicit multi-course sets โดย resolve สมาชิกแต่ละตัวแยกกัน
- alternative-group selection เช่น “เลือกกี่วิชาจากกลุ่มนี้”
- mixed-scope planning: facts ของรายวิชา + total credits ของเทอมเดียวกัน
- placement comparison ข้ามแผน รวม flexible placement sets
- available-plan comparison เมื่อผู้ใช้ถามว่า “ควรเลือกแผนไหน” โดยไม่ระบุชื่อแผน
- chronological placement sequence ของ explicit course set โดยเรียงจาก canonical year/semester และไม่สร้าง prerequisite chain จากลำดับเวลา
- policy / institutional rules ที่มี canonical authority รองรับ
- bounded conversation context ผ่าน `next_context`

เมื่อ semantic shape ยังไม่รองรับหรือ evidence ไม่ครบ ระบบจะคืน clarification / `insufficient_evidence` / `unsupported` แทนการเดา

## Source of truth

```text
data/output/final/*_corrected.json
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

บทบาทสำคัญ:

- `*_corrected.json` — canonical curriculum corpus
- `institution_policy.json` — normalized institution-policy facts
- `program_requirements.json` — edition-scoped whole-program credit requirements
- `cucumber_outputs/runtime/curriculum.db` — runtime factual authority
- `ground_truth/` — evaluation/test only; ไม่ใช่ production authority
- `ground_truth/GT_FIXED.md` — audit log ของ Ground Truth corrections
- `submission/` — frozen historical submission; **ไม่ใช่เอกสารสถานะ runtime ปัจจุบัน**

หาก canonical data เปลี่ยน ให้ rebuild runtime DB:

```powershell
python -m rag.build_index
```

## Run the web app

ต้องใช้ Python 3.10+ และ Node.js

### Python

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r backend/requirements.txt
```

สร้าง `.env` ที่ root:

```dotenv
GEMINI_API_KEY=your_key_here
```

ห้าม commit `.env` หรือ API key

### Build frontend

```powershell
cd frontend
npm install
npm run build
cd ..
```

### Start semantic backend

```powershell
$env:CUCUMBER_QA_MODE = "semantic"
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

หรือ:

```powershell
.\scripts\run_semantic.ps1
```

เปิด:

```text
http://127.0.0.1:8000/chat
http://127.0.0.1:8000/curriculum
http://127.0.0.1:8000/api/health
```

ก่อน demo ตรวจ health:

```text
status = ok
database_ready = true
qa_mode = semantic
```

> `rag/semantic/modes.py` ยังตั้ง default เป็น `legacy` เพื่อความปลอดภัย ดังนั้นการรัน production semantic ต้องตั้ง environment variable ให้ชัดเจน

### Frontend development mode

```powershell
# backend: port 8000
cd frontend
npm run dev
```

เปิด `http://127.0.0.1:5173`

## API contract

### `POST /api/ask`

```json
{
  "question": "DSBA ปี 1 เทอม 1 มีวิชาอะไรบ้าง",
  "conversation_context": {
    "program": "DSBA",
    "catalog_key": "dsba-2565"
  }
}
```

Response semantic path หลัก:

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

เทิร์นถัดไปควรส่ง `next_context` กลับมาเป็น `conversation_context` ระบบเก็บเฉพาะ bounded structural references เช่น program/catalog/plan/year/semester, focus course และ result identities ไม่ใช้ transcript หรือ cached answer prose เป็น factual authority

สถานะ non-answer ที่พบได้ เช่น:

- `clarification_required`
- `insufficient_evidence`
- `unsupported`

### Curriculum API

```text
GET /api/programs
GET /api/curriculum?program=&catalog_key=&plan=&year=&semester=&search=&limit=&offset=
GET /api/courses/{course_code}?program=&catalog_key=
GET /api/health
```

## CLI

```powershell
python scripts/ask.py "IT ปี 2 เทอม 1 เรียนวิชาอะไรบ้าง"
```

หรือ interactive loop:

```powershell
python scripts/ask.py
```

Web chat มี conversation context มากกว่า CLI loop

## Data pipeline

```text
OCR → Extract → Merge → Correct → Evaluate → Build Index
```

```powershell
python -m src.pipeline.run --program it --dry-run
python -m src.pipeline.run --program it --with-index
```

รายละเอียด: `src/pipeline/README.md`

## Testing

Full Python suite:

```powershell
python -m unittest discover -s tests -t .
```

ล่าสุดในรอบ G5-C provider-isolated full discovery รัน **2,804 tests, 0 failures, 0 errors, 3 skipped** ก่อน closeout; focused sequence และ semantic regressions ผ่านหลังแก้ retained-result ordering ด้วย

ผล data-quality report ของ canonical GT-backed scopes อยู่ใน `reports/README.md` ตัวเลข 100% ที่รายงานที่นั่นหมายถึง **record coverage** ไม่ใช่ทุก field มี character accuracy 100%

`eval/results/*.md` และ dated hardening reports เป็น historical snapshots ของแต่ละรอบ ไม่ควรถูกใช้แทนสถานะ implementation ปัจจุบัน

## Important fail-closed boundaries

- ไม่ยืนยัน future course offering เพราะ runtime ไม่มี authority ว่าวิชาจะเปิดจริงในอนาคต
- ไม่สร้าง English Exit threshold ที่ source ไม่มี
- ไม่ฟันธง personal eligibility ถ้าไม่มีเกณฑ์ authoritative ครบ
- ไม่รวมข้อมูลข้าม `catalog_key` เพื่อทำให้คำตอบสำเร็จ
- explicit multi-course request ต้อง resolve สมาชิกครบทั้งหมด มิฉะนั้น fail closed
- placement sequence ไม่สร้าง dependency จาก chronological order
- ambiguous multi-placement sequence / same-term tie ที่ให้ลำดับเดียวไม่ได้จะ fail closed
- topic discovery + unsupported combined constraints จะไม่ถูกลดเหลือคำถามที่ง่ายกว่า
- ordinal/follow-up ที่ resolve referent อย่างปลอดภัยไม่ได้จะไม่ย้อนใช้ referent เก่าแบบเงียบ ๆ
- LLM output ไม่เป็น factual authority ไม่ว่ากรณีใด

## Documentation map

เริ่มจาก `docs/README.md` ซึ่งแยก active documentation ออกจาก historical snapshots ไว้แล้ว

Active docs หลัก:

- `README.md` — overview / setup / API / current capability
- `docs/semantic-qa-vnext.md` — Semantic QA architecture ปัจจุบัน
- `rag/README.md` — implementation map ของ QA/RAG
- `src/pipeline/README.md` — data pipeline
- `docs/academic_rules.md` — policy subsystem
- `reports/README.md` — evaluation/data-quality reports
- `ground_truth/GT_FIXED.md` — GT correction audit log

Week 11 artifacts:

- `docs/wireframes/chat-wireframe.svg`
- `docs/wireframes/curriculum-wireframe.svg`
- `docs/wireframes/user-flow.svg`
- `frontend/`

`submission/` ถูกเก็บแบบ frozen และไม่ได้ rewrite ในการจัดเอกสารรอบนี้
