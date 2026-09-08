# CUCUMBER: Technical Review of the Curriculum Data Pipeline

> ขอบเขตการตรวจสอบ: โค้ด, tests, Ground Truth, consolidated artifacts และ `reports/evaluation/` ที่มีอยู่ใน workspace ณ เวลาตรวจสอบเท่านั้น ไม่มีการ rerun OCR, extraction, merge, evaluation หรือ tests เพื่อเขียนเอกสารนี้
>
> สถานะ BIT ในเอกสารนี้อ้างอิงข้อมูลอัปเดตของโครงการ: พบ source คือ `IT_inter2565.pdf` สำหรับ Business Information Technology (International Program), revised curriculum 2565 แล้ว แต่ยัง **ไม่ได้ integrate เข้า pipeline** ดังนั้นสถานะที่ถูกต้องคือ **source found, integration/OCR/consolidation/evaluation pending**

## วิธีอ่านเอกสาร

- **ข้อเท็จจริง** คือพฤติกรรมที่ตรวจพบจาก source code, schema, Ground Truth หรือ artifact ที่มีอยู่
- **ข้อเสนอแนะ** คือสิ่งที่ควรทำต่อ ไม่ใช่สิ่งที่ระบบทำอยู่แล้ว
- คำว่า **RAG blocker** ใช้เฉพาะสิ่งที่ขวางการสร้าง corpus/retrieval/Q&A ที่เชื่อถือได้จริง ไม่ใช้กับทุก evaluation improvement
- `source_provenance` หมายถึง metadata ระดับ record ที่ได้จาก source/page ไม่ใช่ root field ชื่อ `source` ใน JSON เสมอไป

---

## 1. Project Overview

### CUCUMBER แก้ปัญหาอะไร

CUCUMBER เป็นโครงการเตรียมข้อมูลหลักสูตรเพื่อรองรับการถาม-ตอบหลักสูตรในอนาคต ปัญหาต้นทางคือข้อมูลหลักสูตรและกฎระเบียบอยู่ในเอกสารสแกน/OCR ที่เป็นข้อความไม่เป็นโครงสร้าง ระบบจึงทำให้ข้อมูลเหล่านั้นกลายเป็น record ที่ค้นหาและอ้างอิงกลับไปยังหน้าเอกสารได้

สิ่งที่ระบบทำจริงในปัจจุบันคือ:

```text
Curriculum images
  -> EasyOCR Thai + English
  -> deterministic course/description extraction
  -> provenance-preserving consolidation
  -> Ground Truth comparison and error reports

Academic Rules images
  -> OCR TXT/JSON
  -> deterministic RuleExtractor
  -> deterministic RulesPolicyMapper
```

ระบบยัง **ไม่มี** vector store, embedding, retriever, chatbot, API, database หรือ UI สำหรับ RAG/Q&A ตามที่ระบุไว้ใน `README.md` และไม่ควรนำเสนอว่าเป็น chatbot ที่เสร็จแล้ว

### Architecture ปัจจุบัน

| Layer | Component จริง | หน้าที่ |
|---|---|---|
| OCR | `src/ocr_engine.py::OCREngine` | เรียก EasyOCR ด้วยภาษา `['th', 'en']`; `detail=0` ส่งคืนบรรทัดข้อความที่ถูก uppercase |
| Course runner | `src/run_pipeline.py` | resolve program/plan, รัน OCR, pre-clean, บันทึก OCR metadata, extract และ optional English enrichment |
| Course parser | `src/extractor.py::CurriculumExtractor` | แยก table/course block, course description, `desc_th`, `desc_en` และ provenance |
| Consolidation | `merge_consecutive.py::CurriculumConsolidator` | รวมหลายหน้าและจับคู่ plan record กับ description record แบบ conservative |
| Evaluation | `evaluate.py` | align Course GT กับ prediction, คำนวณ coverage, CER/WER และ flat reports |
| Rules parser | `src/rule_extractor.py::RuleExtractor` | แยก rule hierarchy, reference และ provenance จาก OCR text |
| Rules policy | `src/rules_policy_mapper.py::RulesPolicyMapper` | map rule IDs ที่กำหนดไว้เป็น 12 policy categories แบบ deterministic |
| Inactive experiment | `src/llm_clean_txt.py` | local Ollama cleaner ที่ไม่ถูกเรียกจาก production flow |

`src/pipeline_config.py` เป็นจุดที่ทำให้ program/plan semantics สม่ำเสมอ: รองรับ `DSBA`, `IT`, `AIT`, `GENED`, `BIT`; DSBA/IT/BIT ต้องระบุ `coop` หรือ `no_coop`; AIT ต้องไม่มี plan; GENED ต้องระบุ `gened` โดยชัดเจน ระบบจงใจไม่เดา variant ของสหกิจศึกษา

### ระยะการพัฒนา

**ข้อเท็จจริง:** โครงการอยู่ในระยะ **late prototype / data-preparation checkpoint** ที่มี OCR, deterministic extraction, merge, provenance, course evaluation และ Rules components แล้ว มี tests ระดับ unit/regression หลายส่วน แต่ artifact และ source data ยังไม่ถูก freeze เป็น corpus ที่ reproducible สำหรับ RAG

**ยังไม่เสร็จ:** retrieval, chunking contract, embedding/index, retrieval benchmark, cited answer generation, Rules semantic evaluation แบบ end-to-end และ BIT integration

### Data layer รองรับ Curriculum Q&A / RAG อย่างไร

Course record ปัจจุบันมี field หลักที่เหมาะกับ retrieval:

```json
{
  "code": "06026200",
  "name_th": "...",
  "name_en": "...",
  "credits": "3(3-0-6)",
  "year": 1,
  "semester": 1,
  "category": "...",
  "type": "...",
  "prerequisite": "...",
  "desc_th": "...",
  "desc_en": "...",
  "source_provenance": [
    {
      "program": "DSBA",
      "source_filename": "dsba_page_026.png",
      "source_page": 26,
      "document_category": "plan"
    }
  ]
}
```

Rule record มี `rule_id`, `section_number`, `section_path`, `parent_rule_id`, `rule_text`, `references` และ provenance จึงมีฐานสำหรับ retrieval เชิงกฎระเบียบด้วย

**ข้อเท็จจริงที่สำคัญ:** data model พร้อมเริ่มออกแบบ corpus ได้ แต่ corpus ที่จะใช้จริงยังต้อง freeze, ตรวจ currency ของ artifacts, แก้ความสามารถในการ reproduce merge บางกรณี และกำหนด citation contract ก่อน

---

## 2. Course Pipeline

### Real flow

```text
Images
  -> OCREngine / EasyOCR ['th', 'en']
  -> uppercase text + deterministic pre_clean_with_regex
  -> *_ocr.txt and *_ocr.json with source metadata
  -> CurriculumExtractor.process_file()
       -> detect plan versus description page
       -> plan block extraction OR description extraction
  -> optional English-only second pass for name_en
  -> per-page *_ocr_extracted.json
  -> merge_consecutive.py
       -> ordered description re-extraction where raw OCR exists
       -> conservative plan/description consolidation
       -> merged_*_full.json
  -> evaluate.py against Course GT
```

`src.run_pipeline.py:104-206` เป็น canonical automated runner แต่ **ไม่ทำ consolidation เอง**. `cli.py` เป็น OCR-only CLI ทั่วไป และ `extract.py` รับ OCR TXT/JSON เพื่อ extract แบบแยกต่างหาก. การรวม plan กับ description ต้องใช้ `merge_consecutive.py` ต่อ

Provenance เป็น cross-cutting concern ไม่ได้เกิดหลัง merge เท่านั้น: `save_ocr_results()` บันทึก original filename/page/program ก่อน, extractor attach provenance ในแต่ละ record, แล้ว merge ทำ union ของ provenance จาก plan และ description

### OCR และ pre-clean

`src/ocr_engine.py` ใช้ EasyOCR local ไม่ส่ง document ไป external service โดย default. `src/run_pipeline.py:130-143` ใช้ GPU ถ้าไม่ได้ระบุ `--no-gpu`; README กำหนด CPU (`--no-gpu`) เป็น reproducible baseline

`src/pre_clean.py::pre_clean_with_regex()` เป็น deterministic regex cleanup ก่อน parser และ parser เรียกซ้ำเพื่อรองรับ input OCR ที่มาจาก CLI อื่น กฎหลักคือ:

| กฎ | เหตุผล |
|---|---|
| แปลง wildcard OCR เช่น `XWXN` เป็น `XXXXXXXX` | รักษา placeholder semantics แทนการเดารหัสจริง |
| เติม leading `0` ให้ standalone 7-digit token | แก้รูปแบบ OCR ที่ทำ `06026100` หายเลขศูนย์ตัวแรก |
| ตัด table border และ watermark-like mixed token | ป้องกัน noise เข้า name/description |
| ตัด Thai connector ที่ท้ายบรรทัด English แบบผสม script | ลด line contamination |

**ข้อควรระวัง:** กฎเหล่านี้เป็น structure/script based และไม่มีชื่อรายวิชาเฉพาะ แต่ regex เติม `0` และตัด mixed script ยังเป็น heuristic จึงต้องอยู่ภายใต้ regression tests ไม่ใช่ถือว่า correct กับทุกเอกสารโดยอัตโนมัติ

### Plan extraction: state + structural rules

`CurriculumExtractor.split_into_blocks()` ใน `src/extractor.py:335-442` เดิน OCR lines แบบ state machine และ track context สี่ค่า:

```text
year, semester, category header, course type
```

เมื่อเจอ year/semester header จะ update context; เมื่อเจอ code จะเริ่ม `CourseBlock`; เมื่อเจอ total/table/page/note noise จะปิด block; เมื่อเจอ category header จะ update category/type. จากนั้น `parse_single_block()` แยก Thai name, credits, English name, prerequisite และ note

นี่คือ parser เชิงโครงสร้าง ไม่ใช่ LLM extraction และไม่มี lookup รายวิชาทั้งหมดเพื่อแทนผล OCR

### Course-code validation และ placeholder

| พฤติกรรม | หลักฐาน/ผลลัพธ์ |
|---|---|
| Numeric code ปกติ | `_is_valid_numeric_course_code()` ใน `src/extractor.py:118-120` ยอมรับ numeric code เฉพาะ 8 digits |
| Embedded/standalone code | `COURSE_CODE_RE` และ `try_clean_code_line()` รองรับ OCR junk รอบ code แต่ยัง enforce numeric 8 digits |
| Truncated numeric code | เช่น `0604640` ถูก reject โดย current validation แทนการเดาเป็น `06046407` |
| Placeholder | code ที่มี digits + `X` ถูก normalize/pad ให้ยาว 8 ตัว หรือ OCR junk placeholder กลายเป็น `xxxxxxxx` |
| Description anchor | `_is_description_code_anchor()` ต้องเป็น exact 8-digit line; placeholder ไม่ใช่ anchor สำหรับ association |

การไม่เดารหัสที่ขาดเป็น design ที่ถูกต้องสำหรับ data ที่จะใช้เป็น evidence: false course identity อันตรายกว่า missing record เพราะจะทำ citation และ policy ผิดรายวิชาได้

### Thai/English names, credits และ prerequisite

| Field | Logic ปัจจุบัน | ข้อจำกัด |
|---|---|---|
| `name_th` | สะสม lines ที่มี Thai script | fallback เป็น `ไม่ระบุ`; OCR typo ยังอยู่ได้ |
| `name_en` | สะสม lines ที่มี English script, uppercase และแก้ typo เพียงไม่กี่ regex ใน `clean_ocr_en_text()` | ไม่ใช่ spell correction เต็มรูปแบบ |
| `credits` | `CREDITS_RE` หา `X(X-X-X)` รวม alternative credit row ได้ | ถ้าไม่พบ จะ default เป็น `3(3-0-6)` ซึ่งควรถือเป็น heuristic ไม่ใช่ source-confirmed value |
| `prerequisite` ใน plan | default `ไม่มี`; หา keyword เช่น `PREREQUISITE` | แผนบางหน้าไม่มี prerequisite detail |
| `prerequisite` ใน description | parse marker Thai/English, รวบรวม exact code หากพบ | OCR damage ใน marker/value ยังทำให้ missing หรือ contaminated ได้ |

`--english-second-pass` เป็น optional path ใน `src/english_name_enricher.py`. มันใช้ English-only OCR แบบ `detail=1`, ต้องมี exact 8-digit code occurrence, credit anchor, title band เดียว และ confidence อย่างน้อย `0.5`; ถ้า unsafe จะเก็บ canonical `name_en` เดิมและบันทึก `english_second_pass` เฉพาะตอน helper ถูกเรียก. มันเปลี่ยนแค่ `name_en`, ไม่แตะ Thai name, code, credits หรือ prerequisite

### Year, semester, category, type, program และ plan

| Field/semantic | พฤติกรรมจริง |
|---|---|
| `year`, `semester` | state ถูก update จาก Thai headers รวมกรณีเลขหลุดไปบรรทัดถัดไป; default เริ่ม `1/1` |
| GENED catalog | `year=0`, `semester=0`, `type=เลือก`, `prerequisite=null` เพราะไม่ใช่ placement ใน curriculum plan เดียว |
| Description record | `year=0`, `semester=0`; `flexible_year_semester` ถูกตั้งตาม branch ของ program/plan |
| `category` ที่ output | `parse_single_block()` คำนวณใหม่จาก prefix: `90...` = General Education, `xx...` = free elective, อื่น = specific curriculum (`src/extractor.py:655-660`) |
| `type` | header ที่มี `เลือก` ให้ elective มิฉะนั้น default compulsory; GENED ถูก force เป็น elective |
| Program/plan | resolve centrally ใน `src/pipeline_config.py`; ห้าม infer coop/no_coop |

**ข้อเท็จจริง:** `CourseBlock.category` track header ระหว่าง parse แต่ category ที่ส่งออกถูกเขียนทับด้วย prefix mapping. นี่มีผลกับหลักสูตรที่ใช้ prefix General Education อื่น เช่น BIT GT มี `9664...` จึงต้อง validate taxonomy mapping ก่อน BIT integration ไม่เช่นนั้น current logic จะจัดเป็น `หมวดวิชาเฉพาะ`

### Description extraction, cross-page continuation และ `desc_th` / `desc_en`

`process_file()` ใน `src/extractor.py:1143-1210` ใช้ marker เพื่อตัดสินว่า page เป็น plan หรือ description. GenEd catalog มี exception ที่มอง code+credits list เป็น plan แม้ไม่มี year/table header

Description parser:

```text
exact 8-digit course anchor
  -> Thai title / credits / English title
  -> prerequisite marker
  -> collect body until structural boundary
  -> split body into desc_th and desc_en
```

Boundary ที่ใช้หยุด body ได้แก่ course code ใหม่, page/header/footer, section header และ numeric page noise. `_append_description_lines()` ใน `src/extractor.py:824-864` ลบ duplicate page-edge overlap โดยเปรียบเทียบท้าย description เดิมกับต้นของ page ถัดไป

`extract_descriptions_from_pages()` sort pages ตาม `source_page`, เก็บ leading continuation ของหน้าถัดไปไปต่อท้าย course ก่อนหน้า จึงรองรับ body ข้ามหน้าโดยไม่ทำให้ provenance ของหน้าที่ต่อหายไป

### Repeated codes, ambiguity และ `unresolved_descriptions`

`merge_consecutive.py::dedupe_courses()` ตั้งใจคืน list เดิม ไม่ลบ record ซ้ำ เพราะ course code ไม่ใช่ unique placement key เสมอไป. การ merge ใช้ rule ต่อไปนี้ใน `CurriculumConsolidator.consolidate()`:

```text
merge description into a plan record only when
plan occurrence count(code) = 1 and description occurrence count(code) = 1
```

เมื่อ code ซ้ำหรือ description ซ้ำ ระบบไม่จับคู่ตามลำดับ occurrence. Description ที่ ambiguous ถูกเก็บเป็น `unresolved_descriptions` แทนการเดา. นี่เป็น design ที่เหมาะกับ RAG: การคืนคำตอบว่า evidence ยังไม่สามารถ associate อย่างมั่นใจ ดีกว่าการ attach description ผิดรายวิชา

สำหรับ code composite เช่น `06026259 หรือ 06026260`, merge จะลองจับ description ของแต่ละ sub-code แบบ unique แล้วรวม `desc_th`/`desc_en` และ provenance ให้ record เดียว

### Provenance ระหว่าง extraction/merge

`source_provenance` มี identity สี่ค่า:

```text
program + source_filename + source_page + document_category
```

`merge_source_provenance()` ใน `src/extractor.py:35-65` ทำ union แบบรักษาลำดับและลบ duplicate identity. Plan+description merge จึงอธิบายได้ว่า field ใดมาจากหน้าใด แม้ current schema ยังไม่มี field-level provenance

### General rules หรือ per-course hardcoding?

**สรุป:** แกนหลักเป็น general structural rules/regex/state ไม่ใช่ hardcoding รายวิชาทีละรหัส แต่มี exception ที่ต้องรับรู้

| ลักษณะ | หลักฐาน |
|---|---|
| General | course-block state machine, eight-digit anchor, credit pattern, Thai/English script split, plan/description boundary, cross-page overlap removal |
| Program-specific structural branch | IT ตัด Thai-only section heading หลัง complete row; GENED ตัด numeric section heading และมี description footer/boundary เฉพาะ |
| Exact-code configuration | default `coop_pairs` มี DSBA `06026259/06026260` และ AIT `06046443/06046444` ใน `src/extractor.py:270-273` |
| Program-specific schedule | `flexible_year_semester` ของ description มี DSBA/IT/AIT/BIT fallback branch ใน `src/extractor.py:1069-1081` |
| Misleading fixed metadata | root `source` และ `description` ของ extractor มี label `GT_Template-2.xlsx / ... Ground Truth` ที่ hardcode ตาม program/plan |

ดังนั้นไม่ควรเรียกว่า parser “universal for any university” แบบไม่มีเงื่อนไข แม้ `pre_clean.py` จะพยายามเป็น universal. คำอธิบายที่แม่นยำคือ: **แกน parser เป็น deterministic structural parser ที่มี domain-specific program exceptions และ known alternative-code rules**

---

## 3. Dataset Behavior

### ภาพรวม

Artifacts ภายใต้ `inputs/`, `outputs/` (รวม `outputs/consolidated/`) และ `prototype_outputs/` ถูก ignore ใน `.gitignore`. สิ่งที่เห็นใน workspace ช่วยตรวจพฤติกรรมได้ แต่ไม่ใช่ portable tracked artifact จาก fresh clone. Course GT กลับถูกเก็บใน `ground_truth/`

| Dataset | Data/status ที่มี | Plan semantics | RAG readiness ปัจจุบัน |
|---|---|---|---|
| DSBA | Course GT coop/no_coop, local OCR/consolidated artifacts | ต้อง explicit `coop`/`no_coop` | schema/provenance พร้อมเริ่ม freeze แต่ full artifact ที่เก่าไม่มี description body |
| IT | Course GT coop/no_coop, local OCR/consolidated artifacts | ต้อง explicit `coop`/`no_coop` | เช่นเดียวกับ DSBA; มี repeated/alternative-code edge cases |
| AIT | Course GT และ local artifacts | `plan=null`; `no_plan` เป็นเพียง filename label | ใช้ course data ได้หลัง reconcile stale artifact/GT note issue |
| GENED | catalog plan, description pages, GT และ merged full artifact | explicit `program=GENED`, `plan=gened` | พร้อมที่สุดในเชิง course-description corpus แต่ unresolved codes และ OCR quality ยังต้อง preserve |
| BIT | พบ `IT_inter2565.pdf`; GT/config มีแล้ว | ต้อง explicit `coop`/`no_coop` เมื่อ integrate | source found, integration/OCR/consolidation/evaluation pending |

### DSBA

**ข้อมูลที่มี:** `ground_truth/DSBA/DSBA_academic_plan_coop.json` และ `DSBA_academic_plan_no_coop.json`; local plan/description OCR และ full consolidated artifacts มีอยู่ใน workspace

**Plan semantics:** `src/pipeline_config.py::resolve_plan()` บังคับให้ DSBA ระบุ `coop` หรือ `no_coop`; ไม่อนุมานจากชื่อ directory/page range

**Edge cases สำคัญ:**

- configured composite pair `06026259 หรือ 06026260` ถูก merge ใน extraction เมื่ออยู่ติดกัน
- รายงานปัจจุบันมี DSBA coop `89/90` matched และ no_coop `90/91` matched; ทั้งสองกรณีมี missing `06016401`
- archived full artifacts ที่ตรวจพบไม่มี `desc_th`/`desc_en` แม้ current parser รองรับ description body แล้ว จึงเป็น version skew ระหว่าง logic กับ generated artifact
- description page artifacts ที่ใช้อยู่ถูก tag เป็น `plan=coop`; ถ้านำ command documented สำหรับ no_coop มาใช้, `merge_consecutive.py` จะ group ด้วย plan ที่อยู่ใน artifact ทำให้ table no_coop และ description coop ไม่รวมเป็น full file เดียวอย่างที่คาด

**ข้อจำกัด/RAG:** DSBA มีโครงสร้างเพียงพอสำหรับ corpus หลัง artifact reconciliation. ปัญหา no_coop metadata และ stale full output เป็น blocker ของ **trusted DSBA no_coop corpus** ไม่ใช่ข้อห้ามในการทดลอง retrieval จาก record ที่ตรวจ provenance แล้ว

### IT

**ข้อมูลที่มี:** GT สำหรับ coop/no_coop และ local OCR/consolidated artifacts

**Plan semantics:** เหมือน DSBA, explicit `coop` หรือ `no_coop` เท่านั้น. README ระบุ page allocation ของ IT เป็น user-supplied ไม่ได้ infer อัตโนมัติ

**Edge cases สำคัญ:**

- code `06016418` ซ้ำ ทำให้ association ของ description ถูกปล่อย unresolved อย่างตั้งใจ
- IT GT ใช้ composite alternative `06016481 หรือ 06016482` แต่ parser ปัจจุบันปล่อยเป็นสอง record เพราะ pair นี้ไม่อยู่ใน `coop_pairs`. นี่อธิบาย pattern หนึ่ง missing composite + two extra records ใน report
- GT มี instructional/flexible-year note ที่บันทึกใน field `code` เป็น pseudo-course record ซึ่งทำให้ total/coverage ไม่ใช่ pure course count ทั้งหมด
- current stored IT no_coop description artifacts มี plan metadata `coop` เช่นเดียวกับ DSBA จึง reproduce no_coop full merge ด้วย command ปัจจุบันไม่ได้
- Thai name quality ใน current report อ่อนกว่า dataset อื่น: IT coop CER `0.1146`, character accuracy `88.54%`; IT no_coop CER `0.0971`, character accuracy `90.29%`

**ข้อจำกัด/RAG:** IT มี raw material สำหรับ RAG preparation แต่ต้องไม่ใช้ code เป็น unique document ID. ต้องเก็บ occurrence/provenance และทำ unresolved handling ต่อไป

### AIT

**ข้อมูลที่มี:** `ground_truth/AIT/AIT_academic_plan.json`, local OCR และ consolidated artifact

**Plan semantics:** AIT ไม่มี study-plan variant. `resolve_plan()` จะ reject `--plan`; JSON ใช้ `plan=null`, ส่วน `no_plan` ใช้แค่ safe filename label

**Edge cases สำคัญ:**

- configured pair `06046443 หรือ 06046444` มีอยู่ใน current extractor
- OCR artifact เก่าเคยมี truncated `0604640`; current strict validation ปฏิเสธแล้วแทนการเดา code
- GT มี instructional note pseudo-record เช่นเดียวกับ IT ทำให้ stored count `58` ไม่เท่ากับจำนวนรายวิชาธรรมดาอย่างเดียว
- AIT description `flexible_year_semester` ถูก hardcode เป็น `3/1, 3/2` ใน description branch ไม่ได้ derive จาก source table ทุก record

**ข้อจำกัด/RAG:** สามารถเตรียม course corpus ได้ แต่ก่อน freeze ต้องแยก pseudo-record/administrative note ออกจาก course retrieval unit และตรวจว่า full artifact สอดคล้องกับ current strict validation

### GenEd

**ข้อมูลและ semantics ที่ต้องอธิบายให้ถูกต้อง:**

```text
pages 016-030  = catalog/plan source
pages 044-117  = course-description source
```

ทั้งสองชุดต้องถูกเก็บไว้ เพราะตอบคำถามคนละชนิด:

| Source set | ให้ข้อมูลอะไร |
|---|---|
| Catalog/plan pages 016-030 | code, names, credits, catalog membership และ metadata ของรายวิชา |
| Description pages 044-117 | prerequisite และ narrative body ที่กลายเป็น `desc_th`/`desc_en` |

`process_file()` มี GenEd catalog detection โดยใช้ presence ของ exact code และ credit พร้อม absence ของ description marker (`src/extractor.py:1177-1196`). เมื่อ merge, plan record เป็น primary record และ description เติมเฉพาะเมื่อ one catalog occurrence จับคู่กับ one description occurrence อย่างชัดเจน

**ข้อเท็จจริง:** TODO ระบุ GenEd catalog มี `269` records, `266` distinct codes, และได้ `263` conservative unique associations. Codes `90644004`, `90644005`, `90644006` ซ้ำ จึงยังอยู่ใน `unresolved_descriptions`; ระบบไม่เดาว่า description occurrence ที่หนึ่งควรผูกกับ catalog occurrence ที่หนึ่ง

`merged_gened_gened_full.json` มี `desc_th`, `desc_en` และ provenance ทั้ง plan/description จึงเป็น dataset ที่พร้อมที่สุดสำหรับการทดลอง course-description retrieval. อย่างไรก็ดี OCR wording ใน body ยังคงมีอยู่และไม่ควรถูก silent-correct ก่อนรักษาหลักฐานต้นทาง

### BIT

**สถานะที่ถูกต้อง:** พบ source `IT_inter2565.pdf` แล้ว สำหรับ Business Information Technology (International Program), revised curriculum 2565 แต่ยังไม่มี integration เข้า repository pipeline

| หัวข้อ | สถานะ |
|---|---|
| Source document | found |
| Page allocation / plan semantics ที่ยืนยันจาก source | pending |
| OCR artifacts | pending |
| Extraction/consolidation | pending |
| Evaluation report | pending |
| GT/config | มี BIT coop/no_coop GT และ program configuration อยู่แล้ว |

**ข้อควรระวังด้าน parser:** ก่อน ingest BIT ต้อง validate category mapping. Current code กำหนด General Education จาก prefix `90...` เท่านั้น แต่ BIT GT มี General Education codes กลุ่ม `9664...`; หากไม่แก้/กำหนด mapping อย่างมีหลักฐาน record เหล่านี้จะออกเป็น `หมวดวิชาเฉพาะ`

**RAG readiness:** BIT ไม่ใช่ “source unavailable”; แต่ยังไม่ควร claim BIT support จนกว่าจะมี source-page inventory, OCR, extraction, consolidated corpus และ evaluation ของ BIT เอง

---

## 4. Academic Rules

### Real rules flow

```text
Rules images
  -> EasyOCR through generic OCR path
  -> Rule OCR TXT/JSON
  -> extract_rules.py
  -> RuleExtractor
  -> extracted Rules JSON
  -> map_rules_policy.py
  -> RulesPolicyMapper policy JSON
```

จุดสำคัญคือ `extract_rules.py` **ไม่รับ PNG/PDF โดยตรง**. มันรับ OCR TXT/JSON แล้วเรียก `RuleExtractor`. `src.run_pipeline.py` เป็น course runner และไม่ได้เรียก RuleExtractor. ดังนั้น Rules flow แยกจาก Course flow โดยเจตนา

คำว่า **RuleCollection** ใน flow นี้เป็น logical collection ไม่ใช่ class ที่มีชื่อนี้ใน repository: `RuleExtractor.extract_from_pages()` คืน envelope `{"source", "total_rules", "rules"}` (`src/rule_extractor.py:788-792`) และ `RulesPolicyMapper.map_data()` อ่าน list จาก key `rules` (`src/rules_policy_mapper.py:768-802`). Envelope นี้คือ collection boundary ระหว่าง structural extraction กับ deterministic policy mapping

มี rule source images 13 หน้าใน `inputs/rule`, แต่ไม่พบ generated Rules envelope หรือ policy-map artifact ที่คงอยู่ใน workspace ณ เวลาตรวจสอบ. จึงสามารถอธิบาย implementation และ unit coverage ได้ แต่ไม่ควรอ้างผล run ล่าสุดของ Rules เป็น fact จาก workspace นี้

### Rule ID, hierarchy และ parent relationship

`RuleExtractor` ใน `src/rule_extractor.py` สร้าง record เช่น:

```json
{
  "rule_id": "rule:37.6.1",
  "category": "หมวด 10 ...",
  "section_path": ["37", "37.6", "37.6.1"],
  "section_number": "37.6.1",
  "parent_rule_id": "rule:37.6",
  "rule_text": "...",
  "references": ["37.6", "20"],
  "source_provenance": []
}
```

| Feature | พฤติกรรมจริง |
|---|---|
| Identifier normalization | Thai digits เป็น Arabic digits; `O/o/D/d` เป็น `0`, Thai `ด` เป็น `1`, `:` เป็น `.` เฉพาะ identifier |
| `rule_id` | สร้างแบบ syntactic: `rule:` + `section_number` |
| `section_path` | แตก prefix ของ dotted identifier; `37.6.1` เป็น `37`, `37.6`, `37.6.1` |
| `parent_rule_id` | prefix ก่อนจุดสุดท้าย; parent ไม่จำเป็นต้องปรากฏเป็น record |
| Explicit anchor | รองรับ `ข้อ` และ OCR variant `ขอ`, รวมทั้ง `ข้อ` ในบรรทัดหนึ่งและ identifier ในบรรทัดถัดไป |
| Nested anchor | ยอมรับ dotted identifier เฉพาะเมื่อ compatible กับ active hierarchy |
| Truncated anchor | infer ได้เฉพาะ narrow sequence ที่พิสูจน์ได้จาก current/next identifier; ถ้าไม่แน่ใจเก็บเป็น text |

นี่เป็น hierarchy extraction แบบ deterministic. ไม่มี semantic ontology ที่ตัดสินว่ากฎ “มีผลใช้บังคับ” หรือ “ใช้กับหลักสูตรใด”

### Nested/cross-page rules, references และ numeric protection

| เรื่อง | วิธีที่ระบบทำ | ความหมาย |
|---|---|---|
| Cross-page continuation | pages ถูก sort ตาม `source_page`; active rule เปิดข้ามหน้าได้จนเจอ anchor ใหม่ | rule body และ provenance หลายหน้าถูกเก็บร่วมกัน |
| References | จับ explicit `ข้อ <identifier>` และ split reference เช่น `ตามข้อ` ตามด้วย identifier | output เป็น section strings, ไม่ resolve เป็น graph link ที่รับรองว่าปลายทางมีอยู่ |
| Tables | ตารางถูกคงเป็น `rule_text` ไม่แปลงทุกเลขในตารางเป็น rule | ลด false rule IDs |
| Numeric false positives | unprefixed dotted number ต้อง compatible กับ current hierarchy | ป้องกัน value เช่น `4.00` ถูกมองเป็น child rule ในหลายกรณี |
| Page number noise | ตัดเฉพาะเลขที่เท่ากับ known page และอยู่ต้นหรือท้าย page ที่มั่นใจ | เลือก preserve uncertain numeric text มากกว่าลบทิ้ง |
| Signature/footer | signature marker หยุดการอ่านส่วนท้าย | อาจไม่เหมาะหาก source มี annex หลัง signature |

ข้อจำกัดที่ยังมีคือ contextual protection ไม่ใช่ classifier สมบูรณ์: numeric line ที่มี hierarchy form สอดคล้องกับ active rule อาจถูกอ่านเป็น nested anchor ได้ และ bare nested rule ที่ต้นหน้าโดยไม่มี parent active อาจไม่ถูก recover

### Rule provenance

`RulePage.provenance()` ส่ง:

```json
{
  "source_filename": "rule_page_005.png",
  "source_page": 5,
  "document_category": "rule",
  "program": "... optional ..."
}
```

provenance ถูก add เมื่อ nonblank text ถูก append ให้ rule และ dedupe ต่อ rule. ข้อดีคือ rule ที่ข้ามหน้ามี citation page-level. ข้อจำกัดคือไม่มี document ID, legal version/effective date, image hash, OCR coordinates, line span หรือ field-level citation

### RulesPolicyMapper และ deterministic policy mapping

`RulesPolicyMapper` ไม่ใช้ LLM, ไม่ infer policy category จาก rule text ทั่วไป, และไม่ใช้ `RuleExtractor.category` เพื่อ map policy. มัน index exact `section_number` แล้วใช้ fixed mapping ใน `CATEGORY_RULES` ตามด้วย regex/value extractor ที่จำกัด context

| Policy category ที่รองรับ | ตัวอย่าง required Rules |
|---|---|
| เกณฑ์ภาคทัณฑ์ | 22, 33.11 |
| การกลับเข้าศึกษา | 36 |
| เกณฑ์พ้นสภาพนักศึกษา | 33, 33.1-33.12, 34, 35 |
| ระบบเกรด/การคิดคะแนน | 19.3, 21, 21.1-21.2.3 |
| เกณฑ์เกียรตินิยม | 27, 27.1-27.2.3 |
| การลาพักการศึกษา | 31, 31.1-31.4 |
| การลาออก | 32 |
| การสอบ/วัดผล | 19, 19.1, 19.2, 19.4-19.7, 20, 23, 24 |
| การทุจริตทางวิชาการ | 20, 33.8, 37.6.5, 41, 45.8-45.10 |
| ระเบียบความประพฤติ | 37, 37.1-37.6.10 |
| บทลงโทษทางวินัย | 38-42 และ child rules ที่เกี่ยวข้อง |
| การอุทธรณ์ | 43, 48-51.2 |

ผลลัพธ์มี `category`, `present`, `values`, `summary` และ `evidence` ที่คง rule IDs, missing IDs, source provenance, full supporting text และ snippets

### `present=true` และ `null` หมายความว่าอะไร

| ค่า | ความหมายจริง |
|---|---|
| `true` | required rule ทุกข้อใน fixed mapping มีอยู่, `rule_text` ไม่ว่าง และ `source_provenance` เป็น nonempty list |
| `null` | required rule อย่างน้อยหนึ่งข้อ missing, text ว่าง หรือ provenance ว่าง |
| `false` | mapper ไม่ส่งออกค่า `false` |

ดังนั้น `present=true` **ไม่** หมายถึง “ยืนยันว่ากฎมีผลใช้บังคับกับทุก program”, “เอกสารเป็น version ที่ถูกต้อง”, หรือ “policy program-specific ได้รับการตรวจโดยผู้เชี่ยวชาญ”. หมายถึงหลักฐานโครงสร้างครบตาม fixed expected rule IDs ของ mapper เท่านั้น

### Safe numeric extraction

แนวทางของ mapper คือ “output structured number เฉพาะเมื่อ context และ token ปลอดภัยพอ”:

| Area | Safeguard |
|---|---|
| Numeric normalization | `normalize_structured_number()` รับ complete token ที่มี Arabic/Thai digit จริง; pure OCR `OO` ถูก reject |
| Probation/status/re-entry | ต้อง match phrase context เช่น GPA, ต่ำกว่า/ไม่ต่ำกว่า, ปี/ภาคการศึกษา |
| Grade points | รับ row format ที่เฉพาะเจาะจงและยอมรับแค่ `0`, `0.00-4.00` ใน half-point scale |
| Honors | GPA ถูก allowlist ตาม rule ID (`3.75`, `3.25`); damaged/ambiguous symbols เก็บเป็น evidence แต่ไม่ output เป็น value |
| Disciplinary count | ตรวจ expected child-rule structure; ตัวเลขที่ไม่สอดคล้องถูกปฏิเสธ |
| Appeal deadlines | ต้องมี specific rule context และ `ภายใน <number> วัน/วันทำการ` |

นี่เป็น conservative extraction ไม่ใช่ legal validation สมบูรณ์. OCR digit ที่ผิดแต่ยัง match contextual regex อาจผ่านได้ จึงต้องรักษา `raw_value`, `source_snippet`, `source_rule_id` และ citation ไว้ทุกครั้ง

### Institutional Rules กับ curriculum/program policy ต้องแยกกัน

| Dataset/flow | ความหมาย | ใช้แทนกันได้หรือไม่ |
|---|---|---|
| Institutional Rules source + RuleExtractor | numbered institutional rules, hierarchy, text, references, rule provenance | ไม่ใช่ program curriculum policy โดยตรง |
| `ground_truth/rules_ground_truth.json` | official semantic policy GT, 16 categories x 4 programs จาก MCO.2 | ไม่ใช่ structural RuleExtractor GT |
| `ground_truth/rules_extraction_eval.json` | internal 14-record regression subset สำหรับ structure/text/reference/provenance | metadata ระบุ `official_semantic_gt=false` และ `runtime_policy_truth=false` |

Rules mapper รองรับ 12 categories แต่ official semantic GT มี 16. Categories ที่ยังไม่ถูก mapper output คือ:

- เกณฑ์การสำเร็จการศึกษา
- เกณฑ์การลงทะเบียน
- การเทียบโอนหน่วยกิต
- ระเบียบอื่น ๆ

สี่หมวดนี้มีลักษณะ source/program-dependent มากกว่า fixed institutional rule numbering. Mapper ปัจจุบันไม่มี `--program`, `--plan`, document-version validation หรือ logic reconcile institutional Rules กับ MCO.2 ของแต่ละหลักสูตร

---

## 5. Ground Truth and Evaluation

### Ground Truth มีสามชนิดที่ต้องไม่ปนกัน

| ชนิด | Path/schema | วัตถุประสงค์ |
|---|---|---|
| Course GT | `ground_truth/DSBA/`, `ground_truth/IT/`, `ground_truth/AIT/`, `ground_truth/BIT/`, `ground_truth/general_education_ground_truth.json`; top-level `courses` | เทียบ course extraction |
| Official Rules semantic GT | `ground_truth/rules_ground_truth.json`; top-level `programs` | เทียบ policy categories/value/summary ของ 4 programs ในอนาคต |
| Internal RuleExtractor regression subset | `ground_truth/rules_extraction_eval.json`; top-level `records`, 14 records | regression ของ RuleExtractor structure, hierarchy, references, provenance |

Course evaluator `evaluate.py` อ่านเฉพาะ top-level `courses`; จึงไม่สามารถ evaluate Rules semantic GT ได้ในปัจจุบัน. Rules internal subset ก็ไม่ควรเรียกว่า instructor semantic GT

### Course alignment

`evaluate_json_structure()` ใน `evaluate.py:235-579` align record แบบ one-to-one ดังนี้:

```text
1. Exact match by normalized non-empty code, consume prediction once
2. For remaining GT records, choose unmatched prediction with best
   character-Levenshtein code similarity strictly > 0.85
3. A code-aligned pair becomes a matched record
```

ข้อเท็จจริงที่ต้องสื่อให้ถูก:

- alignment ใช้ code เป็นหลัก ไม่ใช้ name, credits, prerequisite, year, semester, category, plan หรือ provenance เพื่อเลือกคู่
- duplicate/placeholder code ถูก consume ตาม input order; ไม่มี global optimal matching
- fuzzy match ไม่ถูกระบุแยกใน current flat report
- matched record ยังเป็น TP แม้ title/credits/category/placement จะผิด

### TP, FN, FP และเหตุผลที่ไม่มี TN

| Metric | นิยามในระบบ |
|---|---|
| TP | GT record และ prediction record ที่ align กันด้วย code |
| FN | GT record ที่ไม่มี prediction aligned |
| FP | prediction record ที่ไม่มี GT aligned |
| TN | ไม่คำนวณ |

ไม่มี meaningful TN เพราะงานนี้เป็น open-ended extraction coverage: ไม่มี universe ที่นิยามว่า “ทุกข้อความที่ไม่ใช่รายวิชา” เป็น negative sample. การ fabricate TN จะทำให้ accuracy สูงปลอมและไม่ช่วยตัดสินคุณภาพ extractor

```text
Precision = TP / predicted records
Recall    = TP / GT records
F1        = 2 * Precision * Recall / (Precision + Recall)
```

metrics เหล่านี้คือ **coverage ของ record alignment ที่อิง code**, ไม่ใช่ OCR text accuracy หรือ semantic correctness

### CER, WER และ Thai tokenization

| Metric | Implementation | วิธีอ่าน |
|---|---|---|
| CER | Levenshtein character distance / normalized GT length | macro average เฉพาะ fields/records ที่ matched และ GT field มีอยู่ |
| Legacy WER | whitespace split ทุกภาษา | อยู่ใน legacy sections ของ `evaluation.json`; ไม่เหมาะกับ Thai มากนัก |
| Report Thai WER | `PyThaiNLP.word_tokenize(..., engine="newmm")` สำหรับ `name_th`, `desc_th` | อยู่ใน `field_metrics.csv` และ error rows |
| Report English WER | whitespace tokenization สำหรับ `name_en`, `desc_en`, `prerequisite` | เหมาะกับ whitespace-separated English |
| Code/credits/year/semester WER | not applicable | report เว้นว่าง |

ถ้า GT value ว่าง, current `calculate_cer()`/`calculate_field_wer()` คืน `0.0`. จึงห้ามอ่าน “0 CER” ของ field ที่ GT ว่างว่า prediction ถูก เช่น non-empty GENED prerequisite เทียบกับ blank GT ยังอาจแสดง 0 error ใน text metric แต่ record/field อาจไม่ถูกต้องเชิงเนื้อหา

### Field Level, Category Level และ true Page Level

| Level | ความหมายจริง | ข้อจำกัด |
|---|---|---|
| Legacy `field_level` | CER/WER เฉลี่ยของ matched fields | ไม่รวม FN/FP ใน text quality |
| Rubric `field_level` | text quality + field presence coverage + missing field count | presence คือ key อยู่ใน JSON ไม่ใช่ field ถูกต้อง |
| Legacy `page_level` | aggregate ของทุก field comparison ใน matched pairs | **ไม่ใช่ per-page metric** แม้ชื่อจะเป็น page level |
| Legacy `category_level` | แบ่ง plan/description จาก GT year/semester | ไม่ใช่ curriculum category correctness |
| Rubric `category_level` | group coverage ตาม GT `category` | prediction ที่ category ผิดแต่ code match จะยังอยู่ใต้ GT category |
| Rubric `page_level` | true source/page grouping, ตรวจ misplaced match | ใช้ได้ต่อเมื่อทุก GT record มี authoritative `source_provenance` |

Course GT ปัจจุบันไม่มี authoritative per-record source/page provenance จึงทำให้ rubric true Page Level ของทั้งหก stored results เป็น `unavailable`. `code_page_mapping.csv` ไม่ควรถูกใช้แทน GT authoritative page provenance ตาม TODO และ README

### Current `reports/evaluation/` และการตีความ

| File | เนื้อหา | ใช้อย่างไร |
|---|---|---|
| `evaluation.json` | legacy + rubric result ของหลาย cases ใต้ `results` | ดู semantics, field/category/page status อย่างละเอียด |
| `evaluation_summary.csv` | GT/pred/TP/FN/FP/Precision/Recall/F1 | ดู record coverage ก่อนอ่านข้อความ CER/WER |
| `field_metrics.csv` | per-field CER/WER และ accuracy percentage | ดู OCR text quality เฉพาะ matched samples; Thai WER ใช้ NewMM |
| `evaluation_errors.csv` | rows ที่ matched-but-different, missing หรือ extra | trace concrete error patterns และอธิบาย FN/FP |

ผลปัจจุบันเป็น **stored snapshot** ไม่ใช่ผลที่ rerun เพื่อ review นี้:

| Program/plan | GT | Pred | TP | FN | FP | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AIT | 58 | 57 | 56 | 2 | 1 | 0.9825 | 0.9655 | 0.9739 |
| DSBA coop | 90 | 89 | 89 | 1 | 0 | 1.0000 | 0.9889 | 0.9944 |
| DSBA no_coop | 91 | 90 | 90 | 1 | 0 | 1.0000 | 0.9890 | 0.9945 |
| GENED gened | 266 | 269 | 266 | 0 | 3 | 0.9888 | 1.0000 | 0.9944 |
| IT coop | 107 | 107 | 105 | 2 | 2 | 0.9813 | 0.9813 | 0.9813 |
| IT no_coop | 109 | 109 | 107 | 2 | 2 | 0.9817 | 0.9817 | 0.9817 |

การนำเสนอต่ออาจารย์ควรพูดว่า “coverage F1 ระหว่าง 0.9739-0.9945 สำหรับ six archived cases” ไม่ควรสรุปว่า “OCR accuracy 97-99%” เพราะ field text quality โดยเฉพาะ Thai/English names แตกต่างกันมาก เช่น IT coop Thai-name character accuracy `88.54%`, และ English-name WER ของหลาย datasets อยู่ประมาณ `0.20-0.28`

**จุดตีความ error ที่มีหลักฐาน:**

- DSBA coop/no_coop missing `06016401`
- GENED มี extra `90644004`, `90644005`, `90644006` ซึ่งสอดคล้องกับ repeated/unresolved behavior
- IT composite GT `06016481 หรือ 06016482` เทียบกับ predictions แยกสอง record เป็นหนึ่ง FN และสอง FP
- AIT มี FP legacy/truncated `0604640` ใน stored report; current strict parser ตั้งใจ reject code นี้แล้ว

---

## 6. Provenance

### สิ่งที่ระบบเก็บอยู่แล้ว

Course provenance ถูกสร้างจาก OCR metadata (`src/file_handler.py:27-39`) แล้วส่งผ่าน `_source_context()` ของ extractor:

```json
{
  "program": "DSBA",
  "source_filename": "dsba_page_026.png",
  "source_page": 26,
  "document_category": "plan"
}
```

`document_category` สำหรับ course คือ `plan`, `description` หรือ `unknown`; Rules ใช้ `rule`. การ merge ทำ union แบบ stable order และ dedupe identity จึงไม่ทำให้ separate plan/description pages หาย

### เหตุผลที่ provenance สำคัญต่อ RAG citations

Provenance ทำให้ answer ในอนาคตสามารถส่ง citation ที่มีความหมาย เช่น:

```text
รายวิชา 06026200, DSBA, หน้า 26 (plan)
คำอธิบายรายวิชา 06026200, DSBA, หน้า 317-318 (description)
ข้อ 37.6.1, Academic Rules, หน้า 9 (rule)
```

มันยังช่วยตอบคำถามเชิง audit:

- record นี้เกิดจาก source page ไหน
- body text มาจาก description page หรือ plan table
- course ที่มี multiple provenance รวมข้อมูลข้ามหน้าอย่างไร
- answer ต้อง cite page ใดบ้างหากใช้ name จาก plan และ body จาก description

### ข้อจำกัดปัจจุบัน

| Issue | ผลกระทบ |
|---|---|
| root `source`/`description` ของ course extraction มีคำว่า `GT_Template`/`Ground Truth` แบบ hardcoded | อาจทำให้ OCR prediction ถูกเข้าใจผิดว่าเป็น Ground Truth |
| provenance ไม่มี source document ID/version/effective date/hash/OCR runtime version/run command | citation ใช้ filename/page ได้ แต่ยัง audit artifact lineage ไม่ครบ |
| generic `cli.py` ไม่ส่ง original image metadata ให้ `save_ocr_results()` | extraction ต้อง fallback จาก OCR filename/page pattern |
| provenance ระดับ record ไม่ใช่ field-level | ยังตอบไม่ได้ตรง ๆ ว่า credit หรือ sentence เฉพาะมาจากหน้าใดใน multi-page record |
| Course GT ไม่มี authoritative provenance | true page-level evaluation ยังไม่ได้ |

**ข้อเสนอแนะ:** ใน retrieval freeze ให้ใช้ `source_provenance` เป็น runtime citation metadata แต่ห้ามใช้ root label `source` ที่ hardcode เป็น authority. เพิ่ม document identity/version/hash ใน manifest รอบ artifact แทนการแก้ไข historical OCR text แบบเงียบ ๆ

---

## 7. Engineering Decisions and Tradeoffs

| Decision | เหตุผล/ข้อดี | Tradeoff ที่ต้องยอมรับ |
|---|---|---|
| EasyOCR local processing | privacy/control, repeatable environment, ไม่พึ่ง OpenAI/Ollama production path | model install/cache, OCR quality และ GPU setup ยังเป็น operational concern |
| GPU optional, CPU baseline | GPU เร็วขึ้น; `--no-gpu` ให้ baseline ที่ระบุชัดใน README | default runner พยายามใช้ GPU; English second pass ถูก skip บน CPU/CUDA unavailable |
| Deterministic extraction แทน production LLM | inspectable, unit-testable, predictable, ไม่ hallucinate course code | ต้อง maintain regex/state exceptions และไม่แก้ OCR language ได้กว้างเท่า LLM |
| English second pass เป็น opt-in | จำกัด impact ให้เฉพาะ `name_en`, ใช้ exact code/credit/confidence guards | ยัง GPU-only, benchmark unseen production evidence ยังไม่พร้อม |
| Conservative ambiguity handling | repeated code ไม่ถูก pair ตามลำดับ, เก็บ `unresolved_descriptions` เป็น evidence | coverage/body enrichment ต่ำกว่าการเดา และผู้ใช้ downstream ต้อง handle unresolved records |
| Avoid destructive deduplication | course occurrence/placement/provenance ไม่หาย | consumer ต้องใช้ compound identity ไม่ใช่ `code` อย่างเดียว |
| Separation of course, Rules, evaluation, RAG, LLM | ลด coupling: Rules ไม่ถูกบังคับให้มี schema เหมือน course; GT ไม่กลายเป็น runtime data | ต้องมี integration contract เพิ่มใน phase RAG |

`src/llm_clean_txt.py` เป็นตัวอย่างว่าทีมเข้าใจประโยชน์ของ LLM cleanup แต่เลือกไม่ใช้เป็น canonical path. มันเป็น local Ollama implementation ที่ inactive และไม่มี post-response immutable-code/schema guard เพียงพอหากเปิดใช้ จึงควรคง inactive จนมี requirement และ validation gate ที่ชัดเจน

---

## 8. Strengths and Weaknesses

### Strengths ที่มีหลักฐานจาก implementation

| Strength | หลักฐาน |
|---|---|
| Explicit program/plan semantics | `src/pipeline_config.py::resolve_plan()` ไม่เดา coop/no_coop และ reject combination ที่ผิด |
| Deterministic, explainable course parser | block state machine, regex anchors และ program branches อยู่ใน `src/extractor.py` อ่าน/ทดสอบได้ |
| Strict numeric code defense | current code reject truncated numeric code แทนการ fabricate course identity |
| Conservative repeated-code merge | one-plan/one-description rule, `unresolved_descriptions`, no destructive dedupe ใน `merge_consecutive.py` |
| Body preservation for RAG | `desc_th`, `desc_en`, cross-page continuation และ overlap removal อยู่ใน parser/merge flow |
| Citation foundation | course/rule provenance ถูก carry และ union ผ่าน merge |
| Evaluation มี coverage พร้อม text metrics | TP/FN/FP/Precision/Recall/F1 แยกจาก CER/WER, มี errors CSV ที่ trace ได้ |
| Thai-aware report WER | flat reports ใช้ PyThaiNLP NewMM แทน whitespace-only WER สำหรับ Thai |
| Rules safeguards | hierarchy compatibility, explicit references, table/numeric protections, evidence snippets และ safe numeric constraints |

### นิยามระดับความสำคัญ

- **CRITICAL**: ขวางการ claim capability หลักหรือการสร้าง trusted corpus ใน scope นั้น
- **IMPORTANT**: ทำให้ data quality, reproducibility หรือ claim เกินจริงเสี่ยง แต่ไม่จำเป็นต้องหยุด prototype ทุกอย่าง
- **LATER**: ควรทำเมื่อมี downstream need/benchmark; ไม่ควรสร้างงานเพียงเพื่อ completeness

### CRITICAL

| Finding | Evidence | ผลกระทบ/ข้อเสนอแนะ |
|---|---|---|
| Non-GenEd description corpus ยังไม่ freeze/reproducible อย่างเชื่อถือได้ | stored DSBA/IT/AIT full artifacts ไม่มี `desc_th`/`desc_en` ขณะที่ code ปัจจุบันรองรับ; DSBA/IT no_coop descriptions ถูก group เป็น coop จาก metadata ปัจจุบัน | **Blocker สำหรับประกาศว่า non-GenEd full corpus พร้อมเป็น canonical RAG source**. Reconcile artifact version และ plan metadata จาก stored OCR โดยไม่ rerun OCR ก่อน |
| ไม่มี retrieval/citation-serving implementation | README ระบุ RAG/retrieval/chatbot/API/database/UI ยังไม่ complete; ไม่มี vector/retrieval code/dependency | **Blocker สำหรับ claim ว่ามี RAG/Q&A system**. ไม่ใช่ blocker ของการเตรียม data corpus |
| BIT ยังไม่ integrated end-to-end | source found แต่ไม่มี BIT OCR/extraction/consolidation/evaluation artifact | **Blocker เฉพาะการ claim BIT support**. ไม่ block RAG preparation สำหรับ DSBA/IT/AIT/GENED |
| Rules runtime corpus ยังไม่ถูก generate/freeze ใน workspace | มี source images และ code แต่ไม่พบ generated extracted Rules/policy output | **Blocker เฉพาะการตอบ RAG ที่ต้องอ้าง Rules/policy output**; ไม่ block course-only retrieval prototype |

### IMPORTANT

| Finding | Evidence | ผลกระทบ/ข้อเสนอแนะ |
|---|---|---|
| Generated inputs/outputs ถูก ignore และ tests บางส่วนพึ่ง local artifacts | `.gitignore` ignore `inputs/*`, `outputs/*` (รวม `outputs/consolidated/`); fresh clone อาจ reproduce suite/data ไม่ได้ | ทำ source/artifact manifest และ fixture strategy ขนาดเล็กหรือ external versioned storage |
| OCR prediction ถูก label เป็น GT ที่ root | `CurriculumExtractor.__init__()` hardcode `GT_Template`/`Ground Truth` labels | แยก `prediction_artifact` จาก GT metadata ใน future schema/manifest; อย่า cite root label เป็น authority |
| Prefix category mapping ไม่ครอบคลุม BIT | `src/extractor.py:655-660` รู้ `90...` และ `xx...`; BIT GT มี `9664...` | validate/parameterize category mapping ก่อน BIT ingestion |
| Evaluator coverage ไม่ใช่ semantic/placement correctness | code-first alignment, fuzzy threshold, category grouped from GT, true page unavailable | รายงาน metrics พร้อม caveat; เพิ่ม placement/category/description evaluation เมื่อมี authoritative GT ไม่ต้อง block retrieval prototype |
| Course GT มี pseudo-course notes และไม่มี description/page GT | AIT/IT note stored in `code`; all Course GT lacks authoritative provenance/description body | clean evaluation contract หรือ annotate supplemental GT ก่อน claim detailed accuracy/page metric |
| Rules mapper ผูกกับ fixed numbering และครอบคลุม 12/16 official categories | `CATEGORY_RULES`; mapper ไม่มี program/version awareness | ใช้เป็น evidence extraction จาก institutional Rules เท่านั้น; ไม่ใช้แทน program policy truth |
| OCR wording ยังมี material errors โดยเฉพาะ IT/body text | `field_metrics.csv`, `evaluation_errors.csv` และ existing artifacts | prioritize errors ที่เปลี่ยน identity, prerequisite, credits หรือ legal meaning; ไม่ไล่แก้ cosmetic typo ทุกคำ |
| `credits` fallback มีค่าที่เดา | `parse_single_block()` default `3(3-0-6)` | ใน RAG UI/citation ควรแยก field confidence/source-backed status ก่อนใช้ตอบ factual question |

### LATER

| Finding | เหตุผลที่เป็น LATER |
|---|---|
| English second-pass unseen validation | useful ก่อนเปิดเป็น default แต่ canonical path ทำงานได้และ second pass ปัจจุบัน opt-in |
| Revival ของ LLM cleaner | ไม่มี downstream requirement; ปัจจุบันเสี่ยงแก้ immutable facts หากเปิดโดยไม่มี guard |
| Deep structured extraction จาก course description | TODO ระบุให้ทำเมื่อพิสูจน์ว่าช่วย retrieval/Q&A มากกว่า raw body + metadata |
| Corpus-level/page-level evaluation ที่ละเอียดขึ้น | สำคัญต่อ quality claim แต่ไม่ใช่ prerequisite ของ proof-of-concept retrieval หาก citation provenance ของ prediction corpus ถูก preserve |

---

## 9. RAG Readiness

### ข้อมูลที่พร้อมใช้เป็น retrieval foundation

| Data | Ready now | ข้อจำกัดก่อน runtime use |
|---|---|---|
| Course plan records | code, bilingual names, credits, placement, category/type/prerequisite, course provenance | OCR error และ archived artifact currency ต้องตรวจ |
| Course description bodies | parser รองรับ `desc_th`/`desc_en`, cross-page provenance; GenEd full artifact มี body | non-GenEd full artifact ต้อง regenerate/freeze จาก stored OCR; duplicate codes ต้อง not auto-associate |
| Rules records | schema/RuleExtractor รองรับ hierarchy/text/reference/provenance | ไม่มี frozen generated output ใน workspace |
| Policy records | mapper output schema/evidence design พร้อม | only 12 categories, fixed institution numbering, `present` ไม่ใช่ policy truth |

### Retrieval text กับ metadata ควรแยกอย่างไร

| Retrieval text ที่ควร embed/search | Metadata/filter/citation ที่ควรเก็บ |
|---|---|
| course code + Thai/English names | program, plan, record occurrence identity |
| `desc_th`, `desc_en` | credits, year, semester, category, type |
| prerequisite และ note ที่ source-backed | `source_provenance` ทุก entry |
| `rule_text` พร้อม section number/path | rule_id, parent_rule_id, references, document category |
| policy `summary` และ evidence snippets เมื่อ corpus Rules ถูก freeze | source rule IDs, policy category, `present`, source provenance |

course code ไม่ควรเป็น primary key เดี่ยว เพราะมี repeated codes, placeholder codes, alternative/composite codes และ records ที่ unresolved. Runtime document ID ควรเป็น stable artifact/record ID ที่รวม program, plan, source occurrence และ code อย่างน้อย

### สิ่งที่ต้องไม่ index เป็น runtime truth

- Course GT, official Rules semantic GT และ internal regression subset ต้องเป็น evaluation/training-reference only ไม่ใช่ knowledge corpus ที่ตอบผู้ใช้โดยตรง
- `reports/evaluation/` เป็น diagnostics ไม่ใช่ evidence source ของคำตอบ
- root labels ที่มีคำว่า `Ground Truth` ใน OCR prediction artifact ต้องไม่ถูก index เป็น authority statement
- `unresolved_descriptions` ต้องไม่ถูก merge เข้า course record แบบเงียบ ๆ; ถ้าจะ index ให้เป็น separate diagnostic candidate ที่ไม่ตอบเป็น fact
- placeholder code, `N/A`, `ไม่ระบุ`, default credits และ pseudo-course instructional notes ต้องถูก filter/flag ตาม purpose
- mapper `present=true` และ numeric `values` ต้องไม่ถูกยกเป็น legal/program truth โดยไม่มี rule evidence และ version context

### Actual blockers กับ work ที่ทำคู่ขนานได้

| Category | รายการ |
|---|---|
| Actual blockers ก่อน trusted RAG corpus | freeze/reconcile source-derived artifacts; resolve no_coop merge reproducibility; define canonical document/record/citation identity; implement retrieval and citation-serving flow |
| Conditional blockers | integrate BIT หาก scope อ้างว่ารองรับ BIT; generate/freeze Rules corpus หาก Q&A จะตอบ Academic Rules |
| Quality work ที่ทำคู่ขนานได้ | authoritative GT page annotations, description-body GT, richer category/placement evaluation, OCR wording improvement, English second-pass validation |

**ข้อสรุป:** โครงการพร้อมเริ่ม **controlled RAG preparation** สำหรับ data ที่ freeze แล้ว แต่ยังไม่พร้อมเรียกตัวเองว่า “production-ready RAG” หรือ “all-program cited Q&A”

---

## 10. Minimal Ordered Roadmap

งานต่อไปนี้ตั้งใจให้เล็กและเรียงตาม dependency. ไม่รวมการเปิด LLM cleaner หรือ deep description extraction เพราะยังไม่มี evidence ว่าจำเป็นต่อ downstream use case

| ลำดับ | Objective | Why it matters | Blocking? | Recommended model/mode |
|---:|---|---|---|---|
| 1 | Freeze/reconcile course artifacts จาก existing OCR: ระบุ source inventory, artifact version, reconcile stale non-GenEd description outputs และ DSBA/IT no_coop metadata | ทำให้ corpus ที่จะใช้เป็น authoritative prediction artifact และ reproduce ได้; ทำโดยไม่ rerun OCR | Blocking for trusted non-GenEd corpus | Luna + Build; Terra only for independent final audit of discrepancies |
| 2 | Define canonical corpus contract: record ID, retrieval text, metadata, provenance/citation schema, treatment of placeholders/composites/unresolved records | ป้องกัน code collision และทำให้ retrieval/citation implementation ตรงกัน | Blocking for durable RAG corpus | Luna + Build |
| 3 | Integrate BIT source `IT_inter2565.pdf`: identify authoritative relevant pages, explicit coop/no_coop allocation, OCR/extract/merge/evaluate, validate category mapping | เปลี่ยน BIT จาก source found เป็น supported dataset อย่างมีหลักฐาน | Blocking only for BIT support | Luna + Build; Terra only if page/plan semantics remain ambiguous after source review |
| 4 | Generate/freeze Rules extraction and policy artifact from the approved Rules source, add document-version identity, compare the needed scope with official semantic GT | ทำให้ Rules retrieval/citation เป็น reproducible corpus แทน code-only capability | Blocking only for Rules-enabled Q&A | Luna + Build; Terra audit for policy/source-boundary claims |
| 5 | Build a narrow retrieval baseline with known query set and citation checks | วัดว่า chunks retrieve source/page ที่ถูกต้องก่อนเพิ่ม LLM answer generation | Blocking for claiming RAG behavior | Luna + Build |
| 6 | Add cited answer layer that answers only from retrieved evidence and qualifies unsupported questions | เปลี่ยน retrieval artifact เป็น Q&A ที่ตรวจย้อนกลับได้ | Blocking for Q&A deliverable | Luna + Build; Terra only for adversarial final review |
| 7 | Improve evaluation only where it changes a decision: page provenance GT, description quality, category/placement correctness, high-impact OCR errors | ทำ quality claims แข็งแรงขึ้นโดยไม่ขยายงานแบบ cosmetic | Non-blocking for prototype retrieval; important before broad accuracy claims | Luna + Build |

ข้อห้ามของ checkpoint นี้ยังเหมือนเดิม: อย่า rerun OCR เพียงเพื่อให้มี output ใหม่. เริ่มจาก source inventory และ deterministic re-extraction/merge จาก stored OCR เฉพาะเมื่อได้รับอนุมัติและมี artifact manifest

---

## 11. Presentation Guide

### How I Would Explain This Project to an Instructor

> โครงการ CUCUMBER แก้ปัญหาการนำข้อมูลหลักสูตรและกฎระเบียบที่อยู่ในเอกสารสแกนมาใช้ตอบคำถามได้อย่างตรวจสอบย้อนกลับได้ แทนที่จะให้ LLM อ่าน PDF แล้วตอบทันที เราเริ่มจาก EasyOCR แบบ local แล้วใช้ deterministic parser เพื่อแยก code, ชื่อไทย/อังกฤษ, หน่วยกิต, แผนการเรียน, prerequisite และคำอธิบายรายวิชาเป็น structured records
>
> ส่วนสำคัญของ design คือเราแยก OCR, extraction, consolidation, evaluation และ future RAG ออกจากกัน เพราะแต่ละปัญหามี failure mode ต่างกัน เราไม่เดารหัสที่ OCR ขาด ไม่ deduplicate รหัสซ้ำแบบทำลายข้อมูล และไม่จับคู่ description ตามลำดับเมื่อมี ambiguity; เราเก็บ `unresolved_descriptions` ไว้เพื่อไม่สร้าง fact ผิด
>
> ทุก record มี source filename, page และ document category. เมื่อรวม plan กับ description provenance จะถูกเก็บทั้งสองฝั่ง จึงวางฐานสำหรับ citation ในอนาคตได้ การประเมินปัจจุบันวัด coverage ด้วย TP/FN/FP/Precision/Recall/F1 และวัด text quality ด้วย CER/WER โดย Thai WER ใช้ PyThaiNLP NewMM แต่เราระบุชัดว่า code-matching F1 ไม่ใช่ OCR accuracy และ page-level จริงยังทำไม่ได้เพราะ GT ไม่มี authoritative page metadata
>
> ตอนนี้ระบบเป็น data-preparation pipeline ที่ใกล้พร้อมสำหรับ RAG preparation ไม่ใช่ chatbot ที่เสร็จแล้ว ขั้นถัดไปคือ freeze corpus ที่ provenance ครบ, integrate BIT, สร้าง retrieval benchmark และให้ answer layer cite หลักฐานที่ retrieve ได้

### Likely instructor questions

| คำถาม | คำตอบที่กระชับและถูกต้อง |
|---|---|
| ทำไมไม่ใช้ LLM extract ทุกอย่าง? | curriculum facts เช่น code, credits และ prerequisite ต้อง audit ได้. Deterministic parser ให้ repeatability, testability และไม่ hallucinate identity; LLM จะถูกวางไว้หลัง retrieval ไม่ใช่แทน source extraction |
| OCR ผิดแล้วข้อมูลเชื่อถือได้อย่างไร? | เราไม่อ้างว่า OCR ถูกทั้งหมด. เราวัด coverage และ text error แยกกัน, เก็บ error rows, preserve source/page provenance และไม่เดา code ที่ไม่ครบ |
| F1 ประมาณ 0.99 แปลว่าระบบแม่น 99% ไหม? | ไม่ใช่. F1 นี้เป็น code-aligned record coverage. ชื่อ Thai/English, credits, category และ placement ยังมี metrics/limitations แยกต่างหาก |
| ทำไมไม่มี True Negative? | ไม่มี universe ของทุก non-course text ที่เป็น negative class. งานนี้เป็น extraction coverage; fabricate TN จะทำให้ metric หลอกตา |
| ทำไม repeated code ไม่จับคู่ตามลำดับ? | เพราะ occurrence order ไม่ใช่หลักฐานว่า description เป็นของ plan row เดียวกัน. การเดาจะสร้าง citation ผิด; เราเก็บ unresolved candidate ไว้ให้ตรวจต่อ |
| GenEd ทำไมต้องเก็บสองช่วงหน้า? | หน้า 016-030 คือ catalog/plan metadata, หน้า 044-117 คือ body description. ต้อง merge สอง source เพื่อได้ record ที่ตอบทั้ง “มีวิชาอะไร” และ “วิชาเรียนอะไร” |
| `present=true` ใน Rules mapper แปลว่ากฎใช้ได้จริงหรือไม่? | ไม่. แปลเพียง fixed expected rule IDs มี text/provenance ครบใน input. ยังต้องตรวจ source version และ program-specific policy separately |
| BIT รองรับแล้วหรือไม่? | พบ source แล้ว แต่ยัง integration/OCR/consolidation/evaluation pending. จึงยังไม่ claim end-to-end BIT support |
| Page-level evaluation ทำไมยัง unavailable? | prediction มี provenance แต่ Course GT ยังไม่มี authoritative per-record page provenance. เราไม่ใช้ helper mapping มาอ้างเป็น GT |
| RAG จะป้องกัน hallucination อย่างไร? | ทำ retrieval จาก frozen source-derived records, require source/page citation, และให้ระบบ qualify/reject คำถามที่ไม่มี evidence แทนการสร้างคำตอบจาก GT หรือ model memory |

---

## 12. Final Assessment

### Current maturity

CUCUMBER เป็น **strong curriculum extraction and preparation prototype**: มี deterministic pipeline, explicit plan semantics, conservative merge, provenance foundation, course evaluation และ Rules extraction/mapping components ที่แยก concern ถูกต้อง

มันยังไม่ใช่ deployed RAG/Q&A system และไม่ควรสรุปความพร้อมจาก coverage F1 เพียงตัวเดียว

### สิ่งที่ควร freeze

- source inventory และ source-document identity ของแต่ละ program
- current canonical extraction schema และ provenance semantics
- separate Ground Truth datasets พร้อมระบุว่าเป็น evaluation only
- raw OCR artifacts ที่เป็น input ของ deterministic replay พร้อม version/command metadata
- corpus artifact ที่ผ่าน reconciliation แล้วเท่านั้น

**ไม่ควร freeze เป็น runtime corpus ทันที:** stale non-GenEd `*_full.json` ที่ไม่มี body descriptions, root labels ที่บอกว่าเป็น Ground Truth, unresolved descriptions ที่ยังไม่ได้ associate อย่างมั่นใจ และ policy value ที่ไม่มี Rules source/version evidence

### สิ่งที่ยัง incomplete

- canonical frozen/reproducible full course corpus สำหรับทุก dataset
- BIT integration, OCR, consolidation และ evaluation
- Rules generated corpus/policy artifact และ semantic evaluation against official program GT
- retrieval/index/chunking/citation service/Q&A
- authoritative page and description-quality evaluation

### Ready to proceed toward RAG?

**พร้อมเริ่ม RAG preparation แบบควบคุมได้** สำหรับ DSBA, IT, AIT และ GENED หลัง reconcile/freeze artifacts. GenEd มีพื้นฐาน description corpus ที่ชัดที่สุด

**ยังไม่พร้อมสำหรับ production RAG, all-program RAG หรือ cited Q&A claim** จนกว่าจะมี canonical corpus, retrieval/citation implementation, BIT integration ตาม scope และ Rules artifact หากต้องตอบเกี่ยวกับกฎระเบียบ

### Top 3 next actions

1. Reconcile และ freeze provenance-preserving course artifacts จาก existing OCR โดยแก้/บันทึก no_coop merge metadata และ stale description outputs
2. Integrate `IT_inter2565.pdf` เพื่อสร้าง BIT pipeline evidence แบบครบ source-to-evaluation พร้อม validate category mapping
3. กำหนด corpus/chunk/citation contract แล้วสร้าง retrieval baseline ที่วัดการคืน source/page ที่ถูกต้องก่อนทำ LLM Q&A

---

## Evidence Map

| Area | Primary files |
|---|---|
| Program/plan semantics | `src/pipeline_config.py` |
| OCR and automated flow | `src/ocr_engine.py`, `src/run_pipeline.py`, `src/file_handler.py`, `cli.py` |
| Course extraction | `src/pre_clean.py`, `src/extractor.py`, `src/english_name_enricher.py`, `extract.py` |
| Merge and ambiguity | `merge_consecutive.py` |
| Rules | `src/rule_extractor.py`, `extract_rules.py`, `src/rules_policy_mapper.py`, `map_rules_policy.py` |
| Course evaluation | `evaluate.py`, `reports/evaluation/evaluation.json`, `evaluation_summary.csv`, `field_metrics.csv`, `evaluation_errors.csv` |
| Ground Truth distinction | `ground_truth/`, `ground_truth/rules_ground_truth.json`, `ground_truth/rules_extraction_eval.json` |
| Roadmap/history | `README.md`, `TODO_PIPELINE.md` |
