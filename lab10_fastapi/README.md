# Lab 10 — CUCUMBER Curriculum API

แอป FastAPI ส่วนหน้าสำหรับเส้นทาง Natural QA ของ CUCUMBER
ไม่ใช่แอป text-to-SQL แบบ standalone

## ระบบใช้อะไรตอบคำถาม

คำถามถูกจัดการด้วย deterministic QuerySpec, resolution,
หลักฐาน SQLite ทางการ และ pipeline คำตอบแบบมีหลักฐานรองรับ
คำตอบคง provenance ทางการไว้ กฎระเบียบวิชาการกับหลักฐาน
policy ใช้ backend ชุดเดียวกันที่ผ่านการ grounded แล้ว
การตีความ placement/count แบบมีขอบเขตอาจใช้ Gemini provider
เสริมเฉพาะข้อเสนอ intent ที่ผ่านการตรวจแล้วเท่านั้น
โมเดลไม่มีสิทธิ์ให้ข้อเท็จจริงหลักสูตร

คำถาม deterministic ที่ backend ตอบได้ไม่ต้องใช้
`GEMINI_API_KEY` provider จะถูกสร้างแบบ lazy เฉพาะเมื่อ
request นั้นต้องใช้โมเดลช่วย ดังนั้นการตั้งค่าที่ขาดไปจึง
ไม่กระทบ import, startup, health check หรือคำตอบ
deterministic ที่รองรับ

## วิธีใช้งาน

จาก root ของ repository ติดตั้ง dependency ของแอปก่อน

```powershell
python -m pip install -r lab10_fastapi/curriculum_app/requirements.txt
python -m uvicorn lab10_fastapi.curriculum_app.main:app --reload --host 127.0.0.1 --port 8000
```

แอปใช้ runtime database ทางการที่ backend CUCUMBER เลือกอยู่
ตรวจ `/api/health` ก่อนถามคำถาม

- UI: <http://127.0.0.1:8000/>
- Health: <http://127.0.0.1:8000/api/health>
- OpenAPI: <http://127.0.0.1:8000/docs>

ถ้าจะเปิดการตีความแบบมีขอบเขต ให้ตั้ง `GEMINI_API_KEY` ใน
ไฟล์ `.env` ที่ root เป็นทางเลือกสำหรับ path deterministic
และห้าม commit เด็ดขาด

## API

`POST /api/ask` รับ

```json
{"question": "IT ปี 2 เทอม 1 มีวิชาอะไรบ้าง"}
```

response เปิดเฉพาะ contract ของ QA ปัจจุบัน

```json
{
  "question": "...",
  "answer": "...",
  "status": "answer",
  "action": null,
  "route": null,
  "provenance": [{"program": "IT", "source_page": 12}]
}
```

ระบบไม่แต่ง SQL, แถว database, model trace หรือ field อื่น
ที่ผล QA แบบ grounded ไม่ได้ให้มา คำถามที่รองรับไม่ได้หรือ
ยังกำกวมจะคงสถานะ fail-closed ไว้ ถ้า provider เสริมที่ต้อง
ใช้ไม่พร้อม endpoint จะคืน `503` แบบควบคุมไว้

## Multi-turn

ส่ง `next_context` จาก response ก่อนหน้ากลับมาเป็น
`conversation_context` เพื่อคุยต่อเนื่อง (มีแค่โครงสร้าง
scope ทุกครั้งถามหลักฐานทางการใหม่) ส่วน static UI ที่แถม
มาใช้ได้เทิร์นเดียว ไม่ได้ส่ง context ต่อ อยากสาธิต
follow-up ให้ใช้ `/docs` หรือยิง `POST /api/ask` ตรง ๆ

## Tests

รัน API test แบบโฟกัสจาก repository root

```powershell
python -m unittest tests.test_lab10_curriculum_app -v
```

test QA หลักยังคงเป็น authority ของ semantics หลักสูตร

```powershell
python -m unittest tests.rag.test_rag_qa
```
