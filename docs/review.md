# CUCUMBER — System Review

เอกสารนี้สรุป architecture และสถานะของระบบบน branch `project_restart` หลัง restructure โดยเน้นสิ่งที่ระบบทำจริงและขอบเขตที่ต้องรักษา

---

## 1. เป้าหมายของระบบ

CUCUMBER เป็นระบบถาม-ตอบหลักสูตรที่แยกหน้าที่ออกเป็นชั้นชัดเจน:

```text
Document
→ OCR
→ Structured Curriculum Data
→ Corrected Canonical Corpus
→ SQLite + Semantic Index
→ Query Understanding
→ Evidence Retrieval
→ Deterministic Aggregation
→ Grounded Answer
→ Provenance
```

เป้าหมายไม่ใช่ให้ LLM อ่านเอกสารแล้วตอบเอง แต่ให้ LLM ช่วยเฉพาะส่วนที่เหมาะกับภาษา ส่วนข้อเท็จจริงต้องมาจากหลักฐานที่ระบบควบคุมได้

ข้อมูลปัจจุบันครอบคลุม 8 partition:

```text
AIT/default
BIT/coop
BIT/no_coop
DSBA/coop
DSBA/no_coop
GENED/gened
IT/coop
IT/no_coop
```

---

## 2. Source of truth

ลำดับ authority ของข้อมูล production:

```text
เอกสารหลักสูตร
→ OCR / extracted evidence
→ data/output/final/*_corrected.json
→ cucumber_outputs/runtime/curriculum.db
→ QA evidence
```

ขอบเขตสำคัญ:

- `data/output/final/` คือ canonical final corpus ของ layout ปัจจุบัน
- `cucumber_outputs/runtime/curriculum.db` เป็น generated runtime artifact
- `ground_truth/` ใช้สำหรับ evaluation/test ไม่ใช่แหล่งเติมข้อเท็จจริงใน production
- `tests/reference/ocr/` ใช้ regression test OCR/reference behavior
- `submission/` เป็น historical submission package และไม่ใช่ runtime source

ระบบต้องไม่ใช้ test fixture หรือ ground truth เพื่อแต่ง factual answer

---

## 3. Data pipeline

entry point หลัก:

```powershell
python -m src.pipeline.run --program <program>
```

stage:

```text
OCR
→ Extract
→ Merge
→ Correct
→ Evaluate
→ Build Index (optional)
```

### 3.1 OCR

source image:

```text
data/input/<program>/
```

OCR output:

```text
data/output/ocr/<program>/
```

OCR เก็บ text พร้อม metadata/provenance ที่ใช้ต่อใน extraction และ QA

### 3.2 Extraction / Merge

ทำหน้าที่แปลง OCR เป็น record เช่น:

- course code
- ชื่อไทย / อังกฤษ
- credits
- category / requirement type
- year / semester
- prerequisite
- description
- program / plan
- source provenance

การ merge ต้องรักษา identity ของ program, plan, year, semester และ course ไม่ให้ปนข้าม partition

intermediate data จะใช้ temporary directory โดย default ถ้าต้องการ inspect ให้ใช้:

```powershell
python -m src.pipeline.run --program it --keep-intermediates
```

### 3.3 Correction

LLM correction ทำงานหลัง consolidate แล้ว และเขียน final data ไปที่:

```text
data/output/final/
```

ผลหลัก:

```text
*_corrected.json
*_corrections.json
```

### 3.4 Evaluation

pipeline เปิด evaluation โดย default และสามารถปิดด้วย:

```text
--skip-eval
```

reports อยู่ใต้ `reports/`

### 3.5 Index

สร้าง runtime DB ด้วย:

```powershell
python -m rag.build_index
```

หรือให้ end-to-end pipeline ต่อถึง index:

```powershell
python -m src.pipeline.run --program it --with-index
```

---

## 4. Runtime data model

runtime ใช้ฐานข้อมูลเดียว:

```text
cucumber_outputs/runtime/curriculum.db
```

ภายในมี relational curriculum data และ semantic chunks

validated snapshot ล่าสุด:

- programs / plan partitions: 8
- courses: 816
- plan placements: 841
- prerequisites: 57
- semantic chunks: 1667

identity ที่ต้องรักษา:

```text
program
plan
year
semester
course
placement
source provenance
```

course หนึ่งตัวสามารถมีหลาย placement ได้ จึงห้ามถือว่า course identity เท่ากับตำแหน่งในแผนเรียน

---

## 5. QA architecture

user-facing entry point:

```powershell
python scripts/ask.py
```

หรือ:

```powershell
python scripts/ask.py "<question>"
```

flow หลัก:

```text
Question
  ↓
Deterministic Query Parse
  ↓
Scope Resolution
  ↓
Intent Interpretation (เมื่อจำเป็น)
  ↓
Evidence Plan
  ↓
Evidence Executor
  ├─ Structured / SQLite
  ├─ Constrained Semantic Retrieval
  └─ Guarded SQL Fallback
  ↓
Deterministic Aggregation / Judgement Evidence
  ↓
Grounded Claims
  ↓
Natural-language Answer
  ↓
Critical-fact validation / deterministic fallback
  ↓
Provenance
```

---

## 6. Deterministic parse และ scope resolution

ระบบพยายามแยกสิ่งต่อไปนี้จากคำถาม:

- program
- plan
- year
- semester
- course code / course name
- operation
- topic
- comparison / judgement intent

หลักสำคัญคือ scope ต้องมาจากข้อมูลที่พิสูจน์ได้

ระบบไม่ควร:

- เดา program จาก prefix ของ course code
- เลือก plan ให้เองเมื่อมีหลายแผนและคำถามต้องแยกแผน
- ขยาย year/semester นอกคำถามโดยไม่มี evidence contract รองรับ
- ใช้ candidate unanimity เป็นเหตุผลสร้าง program fact

ถ้า scope ยังไม่ชัด ระบบต้อง clarify หรือ fail closed

---

## 7. Intent Interpreter

สำหรับ wording ที่ deterministic parser เข้าใจไม่ครบ ระบบมี intent layer ช่วยตีความภาษาธรรมชาติ

intent ที่รองรับครอบคลุมเช่น:

- topic course search
- course description
- placement
- prerequisite
- similarity
- course / plan comparison
- workload evidence
- preference / recommendation evidence

ข้อจำกัดของ intent model:

- output เป็น **proposal** เท่านั้น
- ไม่ใช่ factual curriculum evidence
- ห้ามสร้าง SQL
- ห้ามสร้าง database ID
- ห้ามสร้าง credits / placements / prerequisites เป็นข้อเท็จจริง
- ห้ามสร้าง final answer หรือ recommendation conclusion
- authoritative scope ต้องผ่าน deterministic validation ก่อน execution

ดังนั้น LLM ช่วยเข้าใจภาษา แต่ไม่ได้มีสิทธิ์เปลี่ยน facts ของหลักสูตร

---

## 8. Evidence planning และ retrieval

### 8.1 Structured evidence

เหมาะกับ:

- list / count
- existence
- credits / sum credits
- placement / earliest placement
- prerequisite
- plan comparison

ข้อเท็จจริงอ่านจาก canonical SQLite records

### 8.2 Semantic evidence

ใช้เมื่อคำถามเกี่ยวกับความหมายหรือเนื้อหารายวิชา เช่น:

```text
มีวิชาเกี่ยวกับ data engineering ไหม
วิชาไหนเรียนคล้าย machine learning
```

embedding model:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

semantic search ต้องจำกัด structural identity ก่อน ไม่ควรค้นทั้งฐานแล้วค่อยพยายามแก้ scope ภายหลัง

### 8.3 Guarded SQL fallback

SQL fallback ใช้เฉพาะเมื่อ structured intent เหมาะสมแต่ wording ไม่เข้า deterministic rule เดิม

contract สำคัญ:

- read-only
- table/view allowlist
- deterministic scope เป็น authoritative constraint
- SQL เลือก candidate identity เท่านั้น
- course query ควรคืน `course_id`
- placement query ควรคืน `placement_id`
- ห้ามให้ SQL ตัดสิน COUNT / SUM / AVG / MIN / MAX เป็น final fact

หลัง SQL เลือก candidate แล้ว ระบบ hydrate canonical record และคำนวณ aggregate แบบ deterministic

---

## 9. Aggregation และ judgement

ค่าที่คำนวณได้ เช่น:

- จำนวนวิชา
- existence
- total credits
- earliest placement
- plan comparison
- workload evidence
- preference evidence
- direct prerequisite burden

ต้องคำนวณจาก evidence ที่ canonical แล้ว

ตัวอย่างคำถาม:

```text
IT ปี 3 อยากเน้น data มีวิชาไหนที่วิชาบังคับก่อนไม่เยอะบ้าง
```

ระบบใช้ topic evidence + direct prerequisite evidence แล้วให้ answer layer อธิบายเชิงคุณภาพจากข้อมูลที่มี ไม่กำหนด threshold ว่า "น้อย" ด้วยการเดา และไม่ใช้ transitive prerequisite depth แทน direct burden

---

## 10. Grounded answer

retrieval result ไม่ถูกส่งตรงให้ผู้ใช้ทันที แต่ถูกแปลงเป็น grounded claims ก่อน

สถานะหลัก:

```text
complete
valid_empty
insufficient_evidence
```

หลักการ:

- `complete` — evidence ครบพอสำหรับ claim
- `valid_empty` — query ถูกต้องแต่ไม่มี record ตามเงื่อนไข
- `insufficient_evidence` — หลักฐานไม่พอสำหรับข้อสรุป

empty ไม่ได้แปลว่า "ทั้งหมด" และห้ามใช้ missing evidence เป็นข้อสรุปเชิงลบ

answer model มีหน้าที่เรียบเรียง grounded facts ให้เป็นภาษาไทยอ่านง่าย เมื่อผลจาก LLM ไม่ผ่าน critical-fact validation ระบบสามารถใช้ deterministic answer แทน

---

## 11. LLM ใช้ตรงไหน

ระบบเป็น **deterministic-first, LLM-assisted**

LLM มีสาม seam หลัก:

1. Intent interpretation — ช่วยแปล wording แปลกให้เป็น intent proposal
2. Structured fallback — ช่วยเขียน bounded candidate-selector SQL
3. Answer rendering — ช่วยเรียบเรียง grounded evidence

ส่วนที่ LLM ไม่ควรเป็น authority:

- จำนวนวิชา
- ผลรวมหน่วยกิต
- course identity
- program / plan identity
- year / semester
- prerequisite facts
- provenance

ภายใน QA engine คำถามง่ายบางประเภทสามารถใช้ deterministic path โดยไม่จำเป็นต้องเรียก model แต่ CLI `scripts/ask.py` ปัจจุบันยัง initialize Gemini provider ตอนเริ่ม จึงต้องมี `GEMINI_API_KEY`

---

## 12. Provenance และ fail-closed behavior

คำตอบควรตรวจย้อนกลับถึงหลักสูตรและหน้า source ได้

CLI แสดงรูปแบบ:

```text
ตอบ: ...
แหล่งข้อมูล:
เล่มหลักสูตร: IT
หน้า: ...
```

กรณีที่ไม่ควรตอบแบบเดา:

- program ไม่ชัด
- plan ไม่ชัดและมีผลต่อคำตอบ
- scope ขัดแย้ง
- candidate ไม่ครบ
- evidence หาย
- query อยู่นอก domain

ระบบจะคืน clarification, `valid_empty`, `insufficient_evidence` หรือ unsupported state ตามกรณี

---

## 13. ความสามารถที่ระบบรองรับปัจจุบัน

### Factual

```text
06016414 กี่หน่วยกิต
IT ปี 2 เทอม 1 มีวิชาอะไรบ้าง
06016414 เรียนช่วงไหน
06016414 ต้องเรียนอะไรมาก่อน
```

### Semantic

```text
IT มีวิชาเกี่ยวกับ database อะไรบ้าง
มีวิชาไหนเนื้อหาคล้าย data mining
```

### Comparison

```text
06016414 กับ 06016465 ต่างกันตรงไหน
แผน IT coop กับ no_coop ปี 4 ต่างกันยังไง
```

### Advisory / preference evidence

```text
ถ้าอยากเน้น data มีวิชาไหนที่ prereq ไม่เยอะบ้าง
ปีไหนเทอมไหน workload สูงกว่า
```

คำถาม advisory ยังต้อง grounded กับ evidence จริง ระบบไม่ได้มี authority ให้ LLM สร้าง ranking หรือ curriculum fact เอง

---

## 14. Validation snapshot

checkpoint หลัง restructure และ OCR reference restoration:

```text
Full unittest suite: 1302
PASS:                1285
Known FAIL:             17
ERROR:                   0

V6 focused gate:      304/304 PASS
OCR gate:              15/15 PASS
OCR reference gate:    19/19 PASS
New refactor regressions: 0
```

17 failures ที่เหลือเป็น known pre-existing/frozen expectations ไม่ใช่ regression จาก restructure:

- query-spec frozen fixtures: `nq_016`, `nq_020`, `nq_025`, `nq_028`, `nq_029`
- answer-rendering expectations
- context-program-topic expectation
- course-placement integration expectations
- exact-course-candidate expectations

จึงไม่ควรตีความว่า test suite ปัจจุบัน clean 100% แต่ checkpoint restructure ไม่ได้เพิ่ม regression ใหม่

Post-fix validation (final semester/elective work, code commit `2f7a0f2`):
focused executor + QA + answer validation passes 273/273 with no new focused
regression observed. Live smoke confirmed a plain exact-term query returns only
the curriculum table placements, while an explicit elective query retains
flexible elective candidates. The full suite was not rerun after these fixes,
so the snapshot above (17 known failures, not all green) still stands.

---

## 15. ข้อจำกัดปัจจุบัน

1. ระบบยังเป็น terminal prototype ไม่มี web UI
2. ความสามารถถูกจำกัดด้วยข้อมูลที่อยู่ใน canonical corpus
3. semantic retrieval ขึ้นกับคุณภาพ course description และ similarity behavior
4. CLI ต้องใช้ Gemini API key
5. interactive CLI เป็น loop ของคำถามแต่ละข้อ ไม่ใช่ conversation-memory layer เต็มรูปแบบ
6. คำถามที่กำกวมมากยังอาจต้องระบุ program/plan/course เพิ่ม
7. LLM wording อาจเปลี่ยนระหว่าง run แต่ critical facts ต้องยัง grounded
8. cold semantic path อาจช้าครั้งแรกเพราะต้องโหลด embedding model

---

## 16. วิธีใช้ที่แนะนำ

ถ้าต้องการแค่ถามระบบ:

```powershell
python scripts/ask.py
```

ถ้าแก้ canonical final JSON:

```powershell
python -m rag.build_index
python scripts/ask.py
```

ถ้าจะทดสอบ pipeline ก่อนรันจริง:

```powershell
python -m src.pipeline.run --program it --dry-run
```

ถ้าจะสร้างข้อมูลใหม่ตั้งแต่ OCR จนถึง index:

```powershell
python -m src.pipeline.run --program it --with-index
```

ถ้าจะรัน regression suite:

```powershell
python -m unittest discover -s tests -t .
```

---

## สรุป

CUCUMBER เป็น curriculum QA pipeline ที่ใช้ LLM เป็นผู้ช่วยด้านภาษา ไม่ใช่แหล่งข้อเท็จจริง

แกนของระบบคือ:

```text
canonical curriculum data
+ deterministic scope
+ bounded retrieval
+ explicit evidence
+ grounded claims
+ provenance
+ fail-closed behavior
```

เมื่อแก้หรือเพิ่มความสามารถใหม่ ควรรักษา invariant เหล่านี้ก่อนเพิ่มความฉลาดของ model เสมอ
