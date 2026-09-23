# rag/ — ระบบถาม-ตอบหลักสูตรแบบมีหลักฐานอ้างอิง

## ภาพรวม
ส่วนนี้รับคำถามภาษาไทย แล้วตอบโดยอ้างอิงหลักฐานจากฐานข้อมูล
SQLite ทางการ พร้อมบอกแหล่งที่มาของข้อมูลทุกครั้ง ทางเข้า
หลักคือ `rag.qa.ask()` (มี `rag.hybrid_demo.answer_question_once()`
หุ้มไว้อีกชั้นหนึ่ง)

## ข้อมูลนำเข้า
รับข้อความคำถามหนึ่งข้อ กับ `QueryContext`
(`rag/resolution.py`) ที่บรรจุได้เฉพาะโครงสร้างอ้างอิง
(program/plan/ปี/เทอม/หมวด/วิชา/operation) เท่านั้น —
ห้ามเก็บคำตอบ — พร้อม model callable ที่ส่งเข้ามาจากภายนอก
(ถ้ามี)

## กระบวนการทำงาน
```text
คำถาม → ตีความ → คำขอแบบมีโครงสร้าง → หลักฐานทางการ →
ผลลัพธ์แบบมีหลักฐานรองรับ → คำตอบสำเร็จรูป
```
1. **แยกสาย policy ก่อน** (`policy/routing.py`) — คำถาม
   วงเงินลงทะเบียน/โปร/เกียรตินิยม/กลับเข้าศึกษา/รวมหลักสูตร
   ตามรายการที่รองรับ จะตอบจากตาราง policy โดยตรงโดยไม่
   ใช้โมเดลเลย
2. **แยกวิเคราะห์แบบ deterministic** (`query_spec.py`) ได้เป็น
   `QuerySpec` แล้ว **resolve** (`resolution.py`) เพื่อตรึง
   ตัวตนของวิชาและขอบเขตให้ชัดเจน
3. **ตีความแบบมีขอบเขต** (`intent_interpreter.py`,
   `intent_compiler.py`) — โมเดลถูกเรียกได้มากสุดหนึ่งครั้ง
   และเฉพาะรูปคำถามที่เข้าเกณฑ์ (เช่น อยากได้รายชื่อแต่ไม่ได้
   ระบุ intent, หรือคำถามเชิงความชอบ) ข้อเสนอของโมเดลทุกชิ้น
   ต้องผ่านการตรวจกับขอบเขตที่แน่นอนก่อน ไม่ผ่านคือทิ้ง
4. **วางแผนและดึงหลักฐาน** (`evidence_planner.py`,
   `evidence_executor.py`, `structured/queries.py`,
   `retrieval/retrieve.py`) — สร้างกราฟคำขอนิรนามแล้วรันแบบ
   deterministic ทีละ plan ส่วน SQL fallback
   (`structured/fallback.py`) ให้โมเดลใช้เลือกได้แค่ id ของ
   candidate ข้อเท็จจริงทุก field ต้อง hydrate จากข้อมูล
   ทางการใหม่เสมอ
5. **สรุปผลแบบมีหลักฐาน** (`aggregation.py`,
   `judgement.py`, `grounded_answer.py`) — รวมผลด้วยฟังก์ชัน
   deterministic ล้วน ได้เป็น claim ที่มีชนิดชัดเจน
6. **แสดงผล** (`answer.py`) — เรียงข้อความเป็นภาษาไทย
   การขัดเกลาถ้อยคำเป็นเพียงเรื่องสำนวนโดยคงหลักฐานเดิม
   และจะถูกปิดหลังเทิร์นที่ใช้การตีความ (ยกเว้นคำแนะนำเชิง
   ความชอบที่รองรับไว้โดยเฉพาะ)

## ผลลัพธ์
`GroundedAnswerResult` ประกอบด้วยสถานะ (`answer`,
`insufficient_evidence`, `valid_empty`, `no_data`,
`clarify_program`, `context_conflict`, `unsupported`)
ข้อความตอบสุดท้าย claim ที่มีชนิด และแหล่งอ้างอิงของข้อมูล
(program + เลขหน้า) การสนทนาหลายเทิร์นใช้วิธีส่ง
`next_context` จาก response กลับมาเป็น
`conversation_context` ของครั้งถัดไป

## กลุ่มคำถามที่รองรับ
รายชื่อวิชา / จำนวนวิชา / มีวิชานี้ไหม / วิชานี้เรียนตอนไหน
หน่วยกิตรายวิชา กรองรายชื่อด้วยเงื่อนไข `N หน่วยกิต`
(เฉพาะ list) ผลรวมหน่วยกิตตาม scope และผลรวมรายหมวด
เฉพาะปี+เทอมที่ระบุชัด วิชาบังคับก่อนทั้งแบบรายวิชาและ
แบบติดตามผลข้ามเทิร์น รหัสวิชา/หลักสูตรของวิชา/คำอธิบาย
รายวิชา ความคล้ายสองวิชา การเปรียบเทียบช่วงเรียนและ
เปรียบเทียบแผน ความชอบที่ระบุชัด และคำถามนโยบายแบบ
เทิร์นเดียวตามรายการที่รองรับ รูปที่รองรับแต่หลักฐานไม่พอ
จะได้คำตอบแบบ fail-closed ที่ควบคุมไว้ ไม่ใช่การเดา

## องค์ประกอบหลัก
`qa.py` (pipeline) · `query_spec.py` · `resolution.py` ·
`policy/` (routing/answer/repository) · `evidence_planner.py` ·
`evidence_executor.py` · `structured/` · `retrieval/` ·
`aggregation.py` · `judgement.py` · `grounded_answer.py` ·
`answer.py` · `intent_interpreter.py` · `intent_compiler.py` ·
`intent_gate.py` (shadow สำหรับประเมินผลเท่านั้น) ·
`hybrid_demo.py` · `providers/gemini.py` · `build_index.py`

## วิธีใช้งาน
```powershell
python -m unittest tests.rag.test_final_core_eval -v
python scripts/ask.py "IT ปี 1 เทอม 1 มีวิชาอะไรบ้าง"
```
ชุดประเมินผลไม่ต้องใช้ API key (model ทุกจุดเป็น stub)
ส่วน CLI จะสร้าง Gemini provider ตั้งแต่เริ่มโปรแกรม จึงต้อง
มี `GEMINI_API_KEY`

## การประเมินผล
Core Evaluation 35/35 PASS
(`tests/rag/fixtures/final_core_eval_v1.json`) Real-user
Robustness 50/50 PASS (ชุดเสริมด้านสำนวนภาษา) fixture ใต้
`tests/rag/fixtures/` กับ `ground_truth/` ใช้ประเมินผล
เท่านั้น ไม่ใช่ข้อเท็จจริงของระบบ production

## ข้อจำกัด
- ข้อเท็จจริงทางการคือหลักฐาน SQLite/deterministic ส่วนผล
  จาก LLM ไม่ใช่ข้อเท็จจริงไม่ว่ากรณีใด
- ห้ามแก้ runtime DB จากเส้นทาง QA
- ตั้งใจไม่รองรับ: ผลรวมรายหมวดนอก scope ปี+เทอมชัด
  นโยบายแบบหลายเทิร์น earliest-year สำนวน credit ภาษาอังกฤษ/
  เขียนเป็นคำ SQL ตามใจ และ memory ข้าม session
