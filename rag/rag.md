# CUCUMBER RAG — สรุปพัฒนาการตั้งแต่เริ่มจนถึงปัจจุบัน

> เอกสารนี้เขียนสำหรับคนที่ **ยังไม่เคยทำ RAG มาก่อน** และต้องการเข้าใจว่าโปรเจกต์ CUCUMBER RAG ทำอะไรไปแล้ว ทำไมต้องทำแบบนั้น และตอนนี้ระบบเดินมาถึงไหน  
> แบ่งเป็น 2 ช่วงตามพัฒนาการจริงของโปรเจกต์:
>
> 1. **ช่วงที่ 1 — ตั้งแต่เริ่มทำ RAG, สร้าง embedding ใน `curriculum.db`, จนทำ submission เสร็จและส่ง**
> 2. **ช่วงที่ 2 — หลัง submission จนถึง architecture / implementation รุ่นใหม่ในปัจจุบัน**

---

# 0. ก่อนอื่น: RAG คืออะไรในโปรเจกต์นี้?

RAG ย่อมาจาก **Retrieval-Augmented Generation**

ความคิดง่าย ๆ คือ:

```text
ผู้ใช้ถามคำถาม
        ↓
ระบบหาข้อมูลที่เกี่ยวข้องจากฐานข้อมูล/เอกสารก่อน
        ↓
เอาข้อมูลที่หาได้มาเป็นหลักฐาน
        ↓
ค่อยสร้างคำตอบ
```

จุดสำคัญคือ **LLM ไม่ควรตอบจากความจำของโมเดลอย่างเดียว** เพราะคำถามของเราเป็นข้อมูลเฉพาะหลักสูตร เช่น

- IT ปี 2 เทอม 1 เรียนอะไร
- วิชา 06016414 เรียนเกี่ยวกับอะไร
- ต้องเรียนวิชาไหนก่อน 06016414
- ปี 3 มีวิชาเกี่ยวกับ AI กี่วิชา
- แผนสหกิจกับไม่สหกิจต่างกันอย่างไร

ข้อมูลเหล่านี้ต้องอ้างอิง **curriculum จริงของโปรเจกต์**

ดังนั้น CUCUMBER RAG ของเราไม่ได้เป็นแค่ “Vector Search + LLM” แต่เป็นระบบที่ผสม:

```text
Relational / Structured QA
        +
Semantic Retrieval
        +
Deterministic Logic
        +
Grounded Answer
```

## คำศัพท์หลักที่ควรรู้

### Structured / Relational

ข้อมูลที่มีโครงสร้างชัด เช่น

```text
program = IT
plan = coop
year = 2
semester = 1
course_code = 06016414
credit = 3
```

ข้อมูลแบบนี้ **SQLite ตอบได้ตรงกว่า LLM**

ตัวอย่าง:

> IT ปี 2 เทอม 1 มีกี่หน่วยกิต

ไม่จำเป็นต้องใช้ semantic search  
ควร query DB แล้วบวกเลขแบบ deterministic

---

### Semantic Retrieval

ใช้เมื่อคำถามถาม “ความหมายของเนื้อหา” เช่น

> มีวิชาเกี่ยวกับ AI ไหม  
> วิชาไหนเกี่ยวกับ database  
> 06016414 เรียนเรื่องอะไร

เราแปลงข้อความเป็น **embedding vector** แล้วดูว่าข้อความไหนมีความหมายใกล้กับ topic ที่ถาม

---

### Embedding

Embedding คือการแปลงข้อความเป็นชุดตัวเลข

ตัวอย่างเชิงแนวคิด:

```text
"machine learning"
→ [0.18, -0.42, 0.07, ...]
```

ข้อความที่ความหมายใกล้กันจะมี vector ใกล้กัน

โปรเจกต์ใช้:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
dimension = 384
```

ดังนั้นข้อความหนึ่งชิ้นถูกแทนด้วยตัวเลข 384 ค่า

---

### Chunk

เราไม่ได้ embed ฐานข้อมูลทั้งก้อนเป็น vector เดียว แต่แบ่งข้อมูลออกเป็นชิ้น ๆ หรือ **chunks**

เช่น description ของแต่ละวิชาเป็นหนึ่ง chunk

```text
course-06016414-description
course-06016419-description
...
```

แล้วแต่ละ chunk มี:

- course identity
- text
- chunk type
- provenance
- embedding vector

---

### Provenance

Provenance คือข้อมูลว่า “หลักฐานนี้มาจากไหน”

เช่น:

```text
program = IT
course_code = 06016414
source = curriculum
chunk_id = course-xxx-description
```

มีไว้เพื่อให้คำตอบ **trace กลับไปยังข้อมูลต้นทางได้**

---

# ช่วงที่ 1 — จาก curriculum.db / embeddings จนทำ Submission เสร็จ

---

# 1. เริ่มจากการเตรียมข้อมูลหลักสูตร

ก่อน RAG จะตอบอะไรได้ เราต้องมีข้อมูลที่เชื่อถือได้ก่อน

pipeline โดยรวมในช่วง submission คือ:

```text
OCR / extracted data
        ↓
canonical consolidated JSON
        ↓
corrected canonical data
        ↓
submission/curriculum.json
        ↓
submission/curriculum.db
        ↓
relational tables
+ semantic_chunks
+ vectors
        ↓
QA / RAG
```

หลักสำคัญคือ:

> **Source of Truth อยู่ที่ canonical corrected curriculum data**

ไม่ใช่ vector DB

เหตุผลคือถ้าข้อความต้นทางผิด แล้วเราไปแก้เฉพาะ vector database จะกลายเป็นว่าระบบมีข้อมูลสองเวอร์ชัน

ตัวอย่างปัญหาที่เคยเจอคือ `06016414`

เดิมมี OCR error เช่น

```text
NOSOL
USON)
```

แม้ชื่อวิชาบางส่วนถูกแก้เป็น NOSQL แล้ว แต่ description บางส่วนยังเป็น OCR เก่า

สุดท้ายจึงแก้ให้ corrected canonical data propagate ลงมาถึง:

```text
curriculum.json
→ curriculum.db
→ semantic description chunk
→ vector
```

นี่สำคัญมาก เพราะ RAG ที่ดีเริ่มจาก **data quality**

---

# 2. โครงสร้าง `curriculum.db`

ฐานข้อมูลไม่ได้มีเฉพาะ vector

มันมีทั้งข้อมูล relational และ semantic index อยู่ด้วยกัน

แนวคิดคือ:

```text
curriculum.db
│
├── programs
├── curriculum_plans
├── courses
├── plan_placements
├── prerequisite / relationship data
│
├── semantic_chunks
└── vector
```

ช่วง submission รุ่น IT มีข้อมูลประมาณ:

```text
Program:
- IT

Plans:
- coop
- no_coop

Courses:
- 208

Placements:
- 212

Semantic chunks:
- 424

Vectors:
- 424

Embedding dimension:
- 384
```

เหตุผลที่ `courses` กับ `placements` ไม่เท่ากัน เพราะวิชาเดียวอาจปรากฏในหลาย plan / placement ได้

---

# 3. การสร้าง Semantic Chunks และ Embeddings

เมื่อมี course description แล้ว เราสร้าง semantic chunks

ตัวอย่างแนวคิด:

```text
Course 06016414
    ↓
description chunk
    ↓
embed text
    ↓
384-dimensional vector
    ↓
เก็บใน sqlite-vec
```

Vector table เดิมคือ sqlite-vec `vec0`

แนวคิดตอน search:

```text
คำถาม
→ embed
→ nearest neighbor search
→ ได้ chunks ที่ vector ใกล้ที่สุด
```

นี่คือ RAG semantic รุ่นแรกของเรา

---

# 4. RAG รุ่นแรก: Structured / Semantic / Hybrid

ระบบเดิมแบ่งคำถามประมาณ 3 route:

```text
structured
semantic
hybrid
```

## Structured

สำหรับคำถามที่ SQL / relational data ตอบได้

เช่น:

```text
IT ปี 2 เทอม 1 เรียนอะไร
06016414 เรียนปีไหน
06016414 ต้องเรียนอะไรก่อน
```

มี deterministic query บางส่วน เช่น:

- semester courses
- semester credits
- course placement
- prerequisite

และบางคำถามยัง fallback ไปหา NL-to-SQL

---

## Semantic

สำหรับคำถามเกี่ยวกับเนื้อหา

เช่น:

```text
06016414 เรียนเรื่องอะไร
มีวิชาเกี่ยวกับ AI ไหม
```

ใช้ embeddings + vector search

---

## Hybrid

กรณีต้องใช้ทั้งสองด้าน

ตัวอย่าง:

> IT ปี 2 มีวิชาเกี่ยวกับ database อะไรบ้าง

ต้องใช้:

```text
Structured:
หาเฉพาะวิชา IT ปี 2

Semantic:
ดูว่าวิชาใดเกี่ยวกับ database
```

แนวคิด hybrid ถูกต้อง แต่ implementation เดิมยังไม่ constrain semantic search ได้ดีพอ

---

# 5. ปัญหาสำคัญของ RAG รุ่นแรก

เมื่อทดสอบจริง เราเริ่มเห็นปัญหาหลายประเภท

## 5.1 Global semantic search มากเกินไป

เดิม retrieval ทำประมาณ:

```text
embed question
→ search vector ทั้งฐาน
→ top-k
→ ค่อย filter
```

ปัญหาคือถ้าผู้ใช้ถาม:

> IT ปี 2 มีวิชาเกี่ยวกับ database ไหม

vector search อาจหยิบ course จาก program / year / plan อื่นมาก่อน

แม้เราจะ filter ทีหลัง แต่ course ที่ควรอยู่ใน candidate set อาจไม่ติด global top-k ตั้งแต่แรก

นี่เป็นปัญหาหลักที่ภายหลังเรา redesign

---

## 5.2 Exact course query ยังใช้ semantic search มากเกินไป

ถ้าถาม:

> 06016414 เรียนเรื่องอะไร

จริง ๆ เรารู้ exact course code อยู่แล้ว

ไม่ควร:

```text
embed ทั้งคำถาม
→ global vector search
```

ควร:

```text
resolve 06016414
→ ไปหยิบ description ของ 06016414 โดยตรง
```

---

## 5.3 Routing ไม่ได้แปลว่าคำตอบถูก

ช่วงแรกเราสนใจว่า query ถูก route ไป structured / semantic / hybrid หรือไม่

แต่ภายหลังพบว่า:

> route เป็นแค่ implementation detail

เช่น route อาจเป็น semantic แต่สุดท้ายหลักฐานถูกต้องก็ยังตอบถูกได้

ดังนั้นภายหลังเราจึงลดความสำคัญของ route ให้เป็น **diagnostic**

---

## 5.4 LLM ทำงานที่ deterministic code ควรทำ

เช่น:

- count
- sum credits
- earliest semester
- compare plan
- prerequisite relation

งานเหล่านี้ควรใช้ Python/SQLite

ไม่ควรหวังให้ LLM อ่านข้อความแล้วบวกเลขเอง

---

# 6. การพัฒนา Structured QA ก่อน Submission

เราเริ่มเพิ่ม deterministic operations ทีละส่วน

ตัวอย่างสำคัญ:

## Placement normalization

placement บางตัวเป็น fixed:

```text
year = 3
semester = 2
```

บางตัว flexible:

```text
3/1, 3/2, 4/1
```

จึง normalize เป็น:

```python
[(3, 1), (3, 2), (4, 1)]
```

แล้วหา earliest ได้ด้วย:

```python
min(year_semester_choices)
```

เพราะ tuple `(year, semester)` เรียงได้ deterministic

---

## Semester credits

สร้าง operation:

```python
get_semester_credits(program, plan_key, year, semester)
```

ทำด้วย parameterized SQLite

ต้องระวัง alternative groups เช่นกลุ่มที่ “เลือกอย่างน้อย 1 วิชา”

จึงไม่สามารถบวก credit ทุก member ตรง ๆ ได้ เพราะจะ double-count

หลักการคือ:

```text
ordinary course
→ นับตามปกติ

alternative group
→ นับตาม minimum_choices
```

และทุก component ต้องมี provenance

---

# 7. Gold Questions และ Evaluator

สำหรับ submission มี Gold Questions 30 ข้อ

โครงสร้าง:

```text
Easy   15
Medium 10
Hard    5
Total  30
```

มี safe in-domain `no_data` cases อย่างน้อย 2 ข้อ

fallback ที่ต้องตรงคือ:

```text
ไม่พบข้อมูลนี้ในเล่มหลักสูตร
```

สิ่งที่ต้องระวังคือ:

> ห้ามแก้ Gold เพื่อทำให้คะแนนดีขึ้น

Gold เปลี่ยนได้เฉพาะเมื่อ canonical source data เปลี่ยนจริง เช่น OCR correction ที่พิสูจน์แล้ว

---

# 8. Submission Package

สุดท้าย submission ต้องมี artifact หลัก:

```text
submission/
├── schema/curriculum.schema.json
├── schema/schema.sql
├── curriculum.json
├── curriculum.conversion.json
├── curriculum.db
├── verify.json
├── gold_questions.json
└── eval_result.json
```

แนวคิดคือ submission ไม่ได้ส่งแค่ code แต่ต้องส่ง:

```text
data
+ schema
+ DB
+ verification
+ gold questions
+ evaluation result
```

---

# 9. ผลลัพธ์และสิ่งที่เรียนรู้จาก Submission

ก่อนส่ง เรามี RAG ที่ใช้งานได้แล้วในระดับหนึ่ง แต่ยังมีข้อจำกัด

รุ่น submission สามารถ:

- query relational facts
- retrieve semantic descriptions
- answer exact no-data cases
- handle plan placement บางส่วน
- attach provenance
- run evaluation pipeline

ใน project history มี stable submission candidate ที่ทำ 30-question execution ได้ครบ และ strict correctness อยู่ประมาณ 23/30 (~76.7%) หลังการแก้ evaluator/data ช่วงท้าย โดย provenance/evidence coverage อยู่ในระดับสูง

แต่ประเด็นสำคัญกว่าคะแนนคือ failure patterns ที่พบ:

```text
1. semantic retrieval ยัง global เกินไป
2. exact entity resolution ยังไม่แข็งแรง
3. ambiguity เช่น program ไม่ถูกจัดการก่อน retrieval
4. composition ของหลาย operation ยังไม่ deterministic พอ
5. topic query บางแบบได้ false positive
6. route-based architecture ทำให้ logic กระจาย
```

ดังนั้นหลัง submission เราไม่ได้ “เพิ่ม prompt ให้ LLM ฉลาดขึ้น”

แต่ตัดสินใจ **redesign RAG architecture**

---

# ช่วงที่ 2 — หลัง Submission จนถึงปัจจุบัน

---

# 10. เป้าหมายหลัง Submission

หลังส่ง เราถามคำถามใหม่ว่า:

> ถ้าจะทำ RAG นี้ให้ “เป็นระบบจริง” และอธิบายได้ในงานวิชาการ เราควรออกแบบใหม่อย่างไร?

เราเลือกแนวทาง:

```text
Deterministic Query Understanding
        +
Relational Constraint Planning
        +
Constrained Vector Retrieval
        +
Deterministic Aggregation
        +
Grounded LLM Synthesis เฉพาะเมื่อจำเป็น
```

แทนที่จะใช้:

```text
question
→ router
→ SQL หรือ vector
→ LLM
```

---

# 11. Natural QA v1 — สร้างชุดคำถามธรรมชาติ 40 ข้อ

เรา freeze คำถาม 40 ข้อที่เหมือนนักศึกษาถามจริง

ตัวอย่าง:

```text
IT ปี 2 เทอม 1 เรียนอะไรบ้าง
ปีสองเทอมสองของ IT มีวิชาอะไร
IT เทอมไหนหน่วยกิตเยอะสุด
06016414 เรียนเกี่ยวกับอะไร
IT มีวิชาเกี่ยวกับ AI อะไรบ้าง
ปีสองของ IT เรียนคอมหนักมั้ย
06016414 กับ 06016419 เนื้อหาคล้ายกันไหม
วิชาไหนยากที่สุดใน IT
06019999 เรียนอะไร
```

จุดประสงค์คือให้ 40 ข้อนี้เป็น **behavioral contract**

ไม่ใช่เขียน code hack รายข้อ

แต่ใช้มันตรวจว่า architecture รองรับภาษาธรรมชาติจริงไหม

---

# 12. Baseline Audit — หาว่าระบบเก่าพังตรงไหน

แทนที่จะรีบแก้ code เรา classify failure stage ของทั้ง 40 ข้อก่อน

ผลโดยรวม:

```text
semantic retrieval          11
ambiguity                    8
structured query/filter      7
normalization                5
composition                  3
unsupported guard            2
aggregation/comparison       1
supported                    3
```

สิ่งสำคัญที่ค้นพบ:

> ไม่มีเคสไหนที่ “routing” เป็น earliest root failure

ดังนั้นการแก้ router อย่างเดียวไม่ช่วยระบบมาก

นี่นำไปสู่ architecture ใหม่

---

# 13. Architecture รุ่นใหม่

pipeline ใหม่:

```text
User Question
     ↓
[1] Thai Surface Normalization
     ↓
[2] Query Understanding / QuerySpec
     ↓
[3] Exact Resolution + Guards
     ↓
[4] Evidence Planner
     ↓
[5] Relational Evidence
     +
    Constrained Semantic Retrieval
     ↓
[6] Deterministic Aggregation / Composition
     ↓
[7] Grounded Judgement
     ↓
[8] Final Grounded Answer + Provenance
```

หลักคิดคือ:

> แต่ละ stage มีหน้าที่เดียว

เพื่อ debug ง่ายและ test แยกได้

---

# 14. Phase 4A — Thai Surface Normalization

สร้าง:

```text
rag/normalization.py
```

ตัวอย่าง:

```text
ปีสอง       → ปี 2
ปีสาม       → ปี 3
ตอนปีสาม    → ปี 3
เทอมสอง     → เทอม 2
เทอมปลาย    → เทอม 2
```

แต่ห้าม semantic rewrite เช่น:

```text
คอม  ≠ programming
data ≠ database
หนัก ≠ ยาก
```

เหตุผลคือ normalization มีหน้าที่แก้ **surface form**

ไม่ใช่ตีความความหมาย

---

# 15. Phase 4B — QuerySpec

สร้าง immutable `QuerySpec`

แนวคิดคือแปลงภาษาคนให้กลายเป็น representation กลาง

ตัวอย่าง:

```text
"IT ปี 2 กับปี 3 ปีไหนมีวิชา programming เยอะกว่า"
```

กลายเป็นประมาณ:

```python
QuerySpec(
    program="IT",
    plans=(),
    years=(2, 3),
    semesters=(),
    topic="programming",
    operations=("count", "compare"),
    group_by=("year",),
    judgement="none",
)
```

Fields หลัก:

```text
original_question
normalized_question
program
plans
years
semesters
course_codes
course_name
category
topic
operations
group_by
judgement
```

QuerySpec **ยังไม่แตะ DB**

มันแค่ตอบว่า:

> “ผู้ใช้พูดอะไรออกมาบ้าง?”

ไม่ใช่:

> “ข้อมูลนั้นมีอยู่จริงหรือไม่?”

สุดท้าย parser ผ่าน frozen 40 cases = **40/40**

---

# 16. Phase 4C — Exact Resolution + Guards

หลังเข้าใจคำถามแล้ว เราค่อยถาม DB ว่า exact entity มีจริงหรือไม่

ลำดับ guard ถูก freeze เป็น:

```text
1. unsupported
2. exact entity resolution
3. no_data
4. ambiguity
5. missing program
6. answerable
```

เหตุผลที่ order สำคัญ เช่น:

## ตัวอย่าง 1

```text
06019999 เรียนอะไร
```

ถ้า code ไม่มีจริง:

```text
no_data
```

ไม่ควรถามกลับว่า “หมายถึง program ไหน?”

เพราะ course ไม่มีตั้งแต่แรก

---

## ตัวอย่าง 2

```text
วิชา NOSQL เรียนอะไร
```

ถ้า NOSQL มีทั้ง IT และ DSBA:

```text
clarify_program
```

---

## ตัวอย่าง 3

```text
เรียนวิชาไหนแล้วเงินเดือนสูงสุด
```

เป็น unsupported judgement

ต้องหยุดก่อน DB/retrieval

---

## Exact candidate lookup

เพิ่ม:

```python
exact_course_candidates(...)
```

รองรับ:

- exact 8-digit code
- exact / deterministic lexical course name
- optional program scope
- collapse duplicate plan identities

ไม่มี:

- fuzzy matching
- embedding
- program inference จาก code prefix

---

# 17. Phase 4D — Evidence Planner

นี่เป็นส่วนที่สำคัญมากของ architecture ใหม่

แทนที่จะถามว่า:

> “query นี้เป็น structured หรือ semantic?”

เราเปลี่ยนเป็น:

> “คำถามนี้ต้องใช้หลักฐานอะไรบ้าง?”

Evidence Planner มี 6 primitives:

```text
1. course_set
2. placement_facts
3. credit_facts
4. prerequisite_facts
5. description_evidence
6. topic_matches
```

ตัวอย่าง:

## คำถาม

```text
IT ปี 2 มีวิชาเกี่ยวกับ database อะไรบ้าง
```

Planner สร้าง dependency:

```text
IT / applicable plans / year 2
        ↓
course_set
        ↓
topic_matches("database")
        ↓
list
```

---

## คำถาม workload

```text
ปีสองของ IT เรียนคอมหนักมั้ย
```

ต้องใช้ target เดียวกัน:

```text
course_set
   ↓
topic_matches("คอม")
   ├── count
   └── credit_facts
```

ห้าม:

```text
count topic courses
แต่ sum credits ของทุกวิชาในปี
```

เพราะ target relation คนละชุด

---

# 18. Structural Partitions

Planner ต้องรักษา scope เช่น:

```text
program
plan
year
semester
course
```

ตัวอย่าง:

```text
IT ปี 2 กับปี 3 ปีไหนมี programming เยอะกว่า
```

ถ้า IT มี coop/no_coop:

```text
coop / year 2
coop / year 3
no_coop / year 2
no_coop / year 3
```

ต้องแยกไว้ก่อน

ห้าม merge plan แล้วค่อย count เพราะอาจนับวิชาซ้ำหรือเสียความแตกต่างระหว่างแผน

---

# 19. Phase 4E — Constrained Semantic Retrieval

นี่คือการแก้ปัญหา vector retrieval รุ่นเก่าโดยตรง

มี 2 path

---

## Path A — Exact Course

ถ้ารู้ course identity แล้ว:

```text
resolved course_id
→ fetch description chunk โดยตรง
```

เพิ่ม API:

```python
fetch_course_description_evidence(...)
```

ไม่มี:

```text
embedding
ANN
global search
```

ถ้า description ไม่มี:

```text
empty semantic evidence
```

ห้ามเอา metadata มาแทน

---

## Path B — Topic Search

ตัวอย่าง:

```text
IT ปี 2 มีวิชาเกี่ยวกับ database
```

flow ใหม่:

```text
course_set จาก structural scope
        ↓
map เฉพาะ course candidates
        ↓
เลือก description chunks เท่านั้น
        ↓
embed topic "database" 1 ครั้ง
        ↓
score เฉพาะ candidate vectors ทั้งหมด
        ↓
threshold
        ↓
topic_matches
```

ต่างจากระบบเก่าอย่างชัดเจน:

### เก่า

```text
global vector search
→ top-k
→ filter IT/year2
```

### ใหม่

```text
filter IT/year2 ก่อน
→ vector score เฉพาะ candidates
```

นี่เรียกว่า **constrained retrieval**

---

# 20. ทำไมไม่ใช้ ANN top-k สำหรับ Topic Membership?

ถ้า candidate set มี 30 วิชา แต่ global ANN คืนแค่ top 10

วิชาที่เกี่ยวจริงอาจอยู่อันดับ 11 และหายไป

ดังนั้นใน architecture ใหม่:

> `top-k` ห้ามเป็นตัวตัดสินว่า course “เกี่ยวหรือไม่เกี่ยว”

เรา score **ทุก candidate description**

แล้วใช้ threshold

---

# 21. Candidate-only Cosine Scoring

sqlite-vec API เดิมรองรับ global KNN เป็นหลัก

เราเลยเพิ่ม:

```python
score_candidate_vectors(
    db_path,
    query_embedding,
    candidate_chunk_ids,
)
```

flow:

```text
candidate IDs
→ SELECT vectors เฉพาะ IDs เหล่านั้น
→ cosine similarity
→ cosine distance
→ sort
```

สูตร:

```text
cosine_distance = 1 - cosine_similarity
```

ดังนั้น:

```text
distance ต่ำ = ใกล้ / เกี่ยวกว่า
distance สูง = ไกล / ไม่เกี่ยว
```

ไม่ใช้ global ANN ใน path นี้

---

# 22. Partition + Provenance Preservation

วิชาเดียวกันอาจปรากฏหลาย partition

เช่น:

```text
course X
→ coop / year2
→ no_coop / year2
```

semantic score สามารถคำนวณครั้งเดียว

แต่ผลต้องยังรักษา:

```text
coop partition
no_coop partition
```

ห้าม merge

จึงมีขั้น:

```python
map_course_candidates_to_description_evidence(...)
score_candidate_vectors(...)
enrich_candidate_description_scores(...)
```

---

# 23. Empty / Missing Evidence States

ระบบใหม่พยายามไม่ใช้ `[]` แบบกำกวม

เพราะ empty มีหลายความหมาย

เราแยก:

```text
empty_structural_candidates
description_missing
vector_missing_or_invalid
scored
```

ภายหลังจะมี:

```text
no threshold matches
```

อีกสถานะหนึ่ง

ความแตกต่างสำคัญมาก

ตัวอย่าง:

```text
ไม่มี course ใน scope
```

ไม่เหมือน:

```text
มี course แต่ OCR ไม่มี description
```

และไม่เหมือน:

```text
มี description แต่ semantic score ไม่ผ่าน threshold
```

---

# 24. High-level Constrained Topic Retrieval

ตอนนี้มี:

```python
retrieve_constrained_topic_evidence(
    db_path,
    topic,
    candidate_courses,
)
```

มันทำ:

```text
candidate courses
→ map descriptions
→ short-circuit empty/missing
→ embed topic once
→ unique candidate scoring
→ reuse score across partitions
→ preserve provenance
```

ยังไม่มี threshold ใน 4E เพราะ threshold แยกเป็น Phase 4F

---

# 25. Phase 4F — Semantic Threshold Calibration (สถานะปัจจุบัน)

ตอนนี้เรากำลังทำตรงนี้

คำถามคือ:

> distance เท่าไรถึงจะถือว่า “เกี่ยว”?

เราไม่ควรเดา:

```text
threshold = 0.5
```

เฉย ๆ

จึงสร้าง calibration แบบมี dev set

---

# 26. Calibration Logic

สร้าง:

```text
rag/retrieval/threshold.py
```

มี:

```python
calibrate_threshold(samples)
is_topic_match(distance, threshold)
```

กฎ:

```text
distance <= threshold
→ relevant
```

ทดลอง threshold ที่ observed distance boundaries

เลือก:

```text
1. F1 สูงสุด
2. ถ้าเสมอ → precision สูงกว่า
3. ถ้ายังเสมอ → threshold ต่ำกว่า
```

เหตุผลที่ชอบ threshold ต่ำเมื่อเสมอ:

> ลด false positives

ซึ่งเหมาะกับ curriculum QA เพราะตอบวิชามั่วออกมา 1 ตัวสามารถทำลายความน่าเชื่อถือได้ง่าย

---

# 27. Human-reviewed Dev Set

สร้าง:

```text
tests/fixtures/semantic_threshold_dev_v1.json
```

เริ่มต้น:

```text
24 pairs
8 topics
3 pairs/topic
```

topics เช่น:

- computer networks
- database/SQL
- AI/ML
- web development
- cybersecurity
- programming/software
- data structures/algorithms
- statistics/probability

แต่พบว่า negative บางตัวง่ายเกินไป เช่น:

```text
music
tourism
public speaking
```

จึงกำลังเพิ่ม **hard negatives / borderline pairs**

เช่น:

```text
database vs data mining
AI vs analytics
programming vs automation
network vs cloud/security
web vs API/business systems
```

นี่คือสถานะปัจจุบัน

---

# 28. ทำไมต้อง Human Label?

เพราะถ้า AI สร้าง label เอง:

```text
AI เลือก relevant/not relevant
→ แล้วเราใช้ label นั้น calibrate embedding
```

จะกลายเป็นวงจรที่ไม่อิสระ

เราจึงให้คนดู:

```text
topic
+
course description
```

แล้วตัดสิน:

```text
T = เนื้อหาสอน topic อย่างมีสาระสำคัญ
F = ไม่เกี่ยว หรือแค่แตะเล็กน้อย
```

ที่สำคัญ:

> ตอน label ไม่ควรดู distance ก่อนตัดสิน

เพราะ distance คือสิ่งที่เรากำลัง evaluate

---

# 29. ปัญหาที่เราเห็นจาก Embedding Dev Set

ตอน review จริงพบ pattern ที่น่าสนใจ

embedding บางครั้งดึง course ผิดเพราะมีคำกว้าง ๆ เช่น:

```text
data
analysis
information
system
management
```

ตัวอย่างเชิง failure:

```text
data mining
→ ดูคล้าย database / AI / analytics

business analysis
→ ดูคล้าย technical analysis

information systems
→ ดูคล้าย information security
```

ดังนั้น semantic similarity ไม่ได้แปลว่า:

> “วิชานี้ตรงกับ intent ของผู้ใช้จริง”

เสมอไป

นี่คือเหตุผลที่ต้องมี:

```text
structural constraint
+
description-only scoring
+
fixed threshold
```

---

# 30. สิ่งที่ยังเหลือหลัง Phase 4F

architecture วางไว้ดังนี้:

```text
4A normalization             ✅
4B QuerySpec                 ✅
4C resolution + guards       ✅
4D Evidence Planner          ✅
4E constrained retrieval     ✅
4F threshold calibration     ← กำลังทำ

4G deterministic aggregation / comparison
4H grounded judgement / similarity
4I pipeline integration + provenance
Phase 5 final 40-case E2E
```

---

# 31. Phase 4G จะทำอะไร?

หลัง retrieval ได้ evidence แล้ว ต้องคำนวณผลแบบ deterministic

เช่น:

```text
count
sum credits
existence
earliest
compare years
compare plans
```

ตัวอย่าง:

```text
IT ปี 2 กับปี 3 ปีไหนมี programming เยอะกว่า
```

flow:

```text
topic_matches(year2)
→ count

topic_matches(year3)
→ count

compare counts
```

LLM ไม่ควรนับเอง

---

# 32. Phase 4H จะทำอะไร?

เป็น judgement ที่ต้องใช้ grounded facts

เช่น:

## Quantity

> ปี 2 มีวิชาเกี่ยวกับคอมเยอะไหม

ดูจำนวนจริงก่อน

---

## Workload

> ปี 3 เทอมปลายเรียนหนักไหม

เราไม่มีข้อมูล “ความยากจริง”

จึงใช้ proxy:

```text
จำนวนวิชา
+
หน่วยกิต
```

แล้วต้องบอกผู้ใช้ว่าคือ proxy

ไม่ควรฟันธงว่า:

```text
ยาก
ง่าย
```

---

## Similarity

> 06016414 กับ 06016419 เนื้อหาคล้ายกันไหม

ต้อง:

```text
fetch description ของ course A
fetch description ของ course B
→ compare จากสอง evidence นี้เท่านั้น
```

---

# 33. Phase 4I จะทำอะไร?

เอาของทั้งหมดมาต่อ pipeline จริง:

```text
question
→ normalize
→ QuerySpec
→ resolution/guards
→ evidence plan
→ relational/constrained semantic evidence
→ aggregation
→ judgement
→ answer
```

และยังคง:

```text
router = diagnostic only
```

ไม่ใช่ตัวกำหนด correctness

---

# 34. Phase 5

สุดท้ายจึงรัน frozen 40-question end-to-end

ตอนนี้เรายังไม่ใช้ 40 ข้อนั้นปรับ threshold

เพราะจะกลายเป็น:

```text
train/tune on test set
```

ซึ่งทำให้ evaluation ไม่น่าเชื่อถือ

---

# 35. ความแตกต่างระหว่าง RAG รุ่น Submission กับรุ่นปัจจุบัน

## Submission-era

```text
Question
→ Route
   ├ structured
   ├ semantic
   └ hybrid
→ execute
→ answer
```

ข้อดี:

- ทำงานได้เร็ว
- submission ได้
- architecture ไม่ซับซ้อน

ข้อเสีย:

- route มี logic กระจาย
- semantic search global
- ambiguity จัดการไม่ครบ
- multi-operation composition ยาก
- exact entity query ยัง retrieval กว้าง
- LLM/SQL fallback รับภาระมากไป

---

## Current architecture

```text
Question
→ Normalize
→ QuerySpec
→ Resolve / Guard
→ Evidence Planner
→ Constrained Evidence
→ Deterministic Derivation
→ Grounded Judgement
→ Answer
```

ข้อดี:

- แต่ละ stage อธิบายได้
- test แยกได้
- deterministic มากขึ้น
- semantic search ไม่มั่วข้าม scope
- plan/year/semester preserved
- provenance ไม่หาย
- เหมาะกับ student budget เพราะไม่ต้องใช้ LLM ทุกขั้น

---

# 36. ถ้าเพื่อนถามว่า “RAG ของมึงทำงานยังไง?”

ตอบสั้น ๆ แบบนี้ได้:

> ระบบเราไม่ได้เอาคำถามไป vector search ทั้งฐานตรง ๆ  
> ก่อนอื่นมัน parse คำถามเป็น QuerySpec เช่น program, year, semester, course, topic และ operation  
> จากนั้น resolve exact entity และเช็ก ambiguity/no-data ก่อน  
> แล้ว Evidence Planner จะกำหนดว่าต้องใช้ข้อมูลอะไร  
> ถ้าเป็นข้อมูลโครงสร้างก็ใช้ SQLite  
> ถ้าเป็น topic semantic จะสร้าง candidate courses จาก relational scope ก่อน แล้วค่อย embedding search เฉพาะ description ของ candidate เหล่านั้น  
> สุดท้าย count/credits/comparison ทำด้วย deterministic Python แล้วค่อยสร้างคำตอบจาก evidence

---

# 37. ถ้าเพื่อนถามว่า “ทำไมไม่ใช้ Vector DB อย่างเดียว?”

ตอบ:

> เพราะข้อมูลหลักสูตรมีโครงสร้างเยอะมาก เช่น ปี เทอม แผน หน่วยกิต prerequisite ถ้าใช้ vector search อย่างเดียวจะไม่แม่นในการ filter/count/compare  
> เราเลยใช้ relational DB สำหรับ facts และ vector retrieval เฉพาะเรื่องความหมายของเนื้อหาวิชา

---

# 38. ถ้าเพื่อนถามว่า “Hybrid RAG ของมึงคืออะไร?”

ตอบ:

> Hybrid ของเราไม่ใช่ยิง SQL หนึ่งทีและ vector search อีกทีแล้วเอาผลมารวมเฉย ๆ  
> เราใช้ structured constraints สร้าง candidate set ก่อน แล้ว semantic search ทำเฉพาะใน candidate set นั้น

ตัวอย่าง:

```text
IT ปี 2 + database
        ↓
SQLite หา IT ปี 2
        ↓
ได้ candidate courses
        ↓
semantic score description ของ candidates เท่านั้น
```

---

# 39. ถ้าเพื่อนถามว่า “LLM ใช้ตรงไหน?”

แนวทางปัจจุบันคือ:

```text
ไม่ใช้ LLM สำหรับ:
- count
- sum
- filter
- exact relation
- prerequisite
- earliest
- no-data decision

ใช้ LLM เฉพาะเมื่อจำเป็นสำหรับ:
- semantic interpretation บางประเภท
- similarity/judgement ที่มี evidence แล้ว
- final natural-language synthesis
```

เป้าหมายคือ:

```text
Simple
Deterministic
Grounded
Traceable
Affordable
```

---

# 40. ถ้าเพื่อนถามว่า “ตอนนี้เสร็จกี่เปอร์เซ็นต์?”

ใน architecture ใหม่:

```text
4A ✅ Normalization
4B ✅ Query Understanding
4C ✅ Resolution + Guards
4D ✅ Evidence Planner
4E ✅ Constrained Retrieval
4F 🟡 Threshold Calibration
4G ⬜ Aggregation / Comparison
4H ⬜ Judgement / Similarity
4I ⬜ Integration
Phase 5 ⬜ Final E2E Evaluation
```

ดังนั้นตอนนี้:

> ส่วนที่ “เข้าใจคำถาม → กำหนด scope → หา semantic evidence แบบ constrained” ทำไปค่อนข้างครบแล้ว  
> สิ่งที่เหลือคือ freeze semantic threshold, ทำ aggregation/judgement และต่อ pipeline end-to-end

---

# 41. Mental Model ที่ควรจำ

ถ้าจำทุก implementation ไม่ไหว ให้จำแค่นี้:

```text
1. Understand
2. Resolve
3. Plan
4. Retrieve
5. Compute
6. Answer
```

หรือภาษาไทย:

```text
เข้าใจว่าเขาถามอะไร
→ เช็กว่า entity ไหนจริง
→ วางแผนว่าต้องใช้หลักฐานอะไร
→ ดึงหลักฐานเฉพาะที่เกี่ยว
→ คำนวณ deterministic
→ ตอบจากหลักฐาน
```

นี่คือแกนหลักของ CUCUMBER RAG รุ่นปัจจุบัน

---

# 42. สรุปสองช่วงแบบสั้นที่สุด

## ช่วงที่ 1 — Submission

เราเริ่มจาก:

```text
curriculum.db
+ semantic chunks
+ embeddings
```

สร้าง RAG แบบ:

```text
structured / semantic / hybrid
```

เพิ่ม deterministic structured QA, placement, credits, provenance, evaluator และ Gold 30 ข้อ จนสร้าง submission package และส่งได้

สิ่งที่ได้จากช่วงนี้คือ:

> ระบบใช้งานได้จริง และทำให้เราเห็น failure จริง

---

## ช่วงที่ 2 — หลัง Submission

เราไม่ patch ทีละ query แต่ audit 40 Natural QA แล้ว redesign architecture

กลายเป็น:

```text
Normalization
→ QuerySpec
→ Exact Resolution + Guards
→ Evidence Planner
→ Constrained Retrieval
→ Threshold
→ Aggregation
→ Judgement
→ Integration
```

เป้าหมายไม่ใช่ทำ RAG ที่ซับซ้อนที่สุด

แต่ทำ RAG ที่:

```text
อธิบายได้
ทดสอบได้
ไม่เดา
ไม่ดึงข้อมูลมั่ว
ใช้ LLM เท่าที่จำเป็น
และเหมาะกับ scope โปรเจกต์นักศึกษา
```

---

# Current Status

ณ จุดที่เขียนเอกสารนี้:

```text
Branch: heart

Completed:
- Natural QA v1 contract
- baseline audit
- architecture 3A–3F
- 4A normalization
- 4B QuerySpec (40/40)
- 4C exact resolution + guards
- 4D Evidence Planner
- 4E constrained semantic retrieval
- 4F.1 threshold calibration logic

In progress:
- 4F.2 human-reviewed threshold development set
- adding hard-negative / borderline pairs

Next:
- freeze one semantic threshold
- 4G deterministic aggregation/comparison
- 4H grounded judgement/similarity
- 4I pipeline integration
- Phase 5 final 40-question E2E
```

---

# One-line Summary

> **CUCUMBER RAG คือระบบ QA หลักสูตรที่ใช้ SQLite จัดการ facts, ใช้ constrained semantic retrieval จัดการความหมายของเนื้อหาวิชา, ใช้ deterministic logic สำหรับการคำนวณ/เปรียบเทียบ และให้ LLM ทำเฉพาะงานภาษาหรือ judgement ที่มี evidence รองรับแล้ว**
