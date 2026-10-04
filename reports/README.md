# Reports

โฟลเดอร์นี้เก็บผลการประเมินคุณภาพข้อมูลและ runtime benchmark ของ CUCUMBER

## 1. Current curriculum evaluation

รายงานปัจจุบันถูก regenerate เมื่อ **2026-10-04** จาก canonical corrected files ที่มี Ground Truth ตรงกันโดยตรง

ประเมินทั้งหมด 8 scopes:

- AIT
- BIT / coop
- BIT / no_coop
- DSBA / coop
- DSBA / no_coop
- GENED / gened
- IT / coop
- IT / no_coop

ผล record coverage:

| Metric | Result |
| --- | ---: |
| Ground Truth records | 839 |
| Prediction records | 839 |
| Matched records | 839 |
| Missing | 0 |
| Extra | 0 |
| Precision | 100% |
| Recall | 100% |
| F1 | 100% |

**หมายเหตุ:** 100% ด้านบนคือ record coverage เท่านั้น ไม่ได้หมายความว่าทุกข้อความ/field ตรง 100%

Weighted field quality across 839 matched records:

| Field | Character Accuracy | Word Accuracy |
| --- | ---: | ---: |
| code | 100.00% | N/A |
| name_th | 99.29% | 98.35% |
| name_en | 99.36% | 95.58% |
| credits | 99.59% | N/A |
| prerequisite | 99.98% | 99.90% |

รายละเอียด error จริงอยู่ใน `evaluation_errors.csv`

### Rebuild integrity fix

การ rebuild runtime DB ถูกทดสอบกับ corrected files ปัจจุบันทั้งชุดแล้ว โดยแก้:

- เติม `catalog_key = bit-2565` ให้ BIT 2565 ทั้ง `coop` และ `no_coop`
- ทำให้ shared-course facts ของ BIT 2565 สอดคล้องกันก่อน merge ข้ามแผน
- ให้ blank credit ถูกตีความเป็น missing value แทน raw empty string เพื่อไม่สร้าง false conflict; ถ้าอีกแผนของ course เดียวกันมีค่า canonical ที่ยืนยันได้ loader จึงใช้ shared course fact นั้นได้
- เพิ่ม regression test ที่โหลด corrected files จริงทั้งหมดร่วมกับ `program_requirements.json`

default rebuild path ผ่าน end-to-end test ด้วย deterministic test embeddings และ program requirement ของ `bit-2565` resolve ได้เพียงหนึ่ง catalog ตาม contract

### Verified data fixes

หลังตรวจ error ที่มีผลต่อข้อมูลจริง แก้เฉพาะรายการที่มีหลักฐานรองรับชัดเจน:

- IT `06066302`: ชื่อไทย → `การเขียนโปรแกรมเว็บพื้นฐาน`
- IT `06016465`: ชื่อไทย → `การออกแบบศูนย์ข้อมูล`
- GENED `90642045`: `BE MV BEV.` → `BE MY BEV.`

focused correction tests, fail-closed credit tests, canonical assertions และ evaluation regression checks ผ่านทั้งหมด ส่วน BIT `06036135` credits ยังเว้นว่างโดยตั้งใจ เพราะยังไม่มี source-verified production evidence เพียงพอให้เติมค่าจาก Ground Truth


## 2. Evaluation scope boundary

`data/output/final/` ปัจจุบันมี corrected files หลาย curriculum editions รวมถึง historical editions เช่น 2560/2557

แต่ Ground Truth ปัจจุบันมีเพียงหนึ่ง accepted GT ต่อ program/plan scope และยัง **ไม่มี edition-specific GT สำหรับ historical editions ทุกชุด**

ดังนั้น report ปัจจุบันตั้งใจใช้เฉพาะ 8 prediction/GT pairs ที่ตรงกับ GT ที่มีอยู่ และ **ไม่เอา historical edition files มาเทียบกับ GT ของอีก edition**

นี่ทำให้ metric ชุดนี้เป็น:

> คุณภาพของ current GT-backed evaluation set

ไม่ใช่:

> coverage/accuracy ของทุก catalog edition ใน runtime DB

ถ้าจะวัด historical editions เพิ่ม ต้องสร้างและตรวจ edition-specific Ground Truth ก่อน

## 3. Reproduce current report

ใช้ explicit pairs เพื่อป้องกัน cross-edition evaluation:

```powershell
python -m src.pipeline.tools.evaluation.evaluate `
  --pair data/output/final/merged_ait_no_plan_full_corrected.json ground_truth/AIT/AIT_academic_plan.json `
  --pair data/output/final/merged_bit_coop_full_corrected.json ground_truth/BIT/BIT_academic_plan_coop.json `
  --pair data/output/final/merged_bit_no_coop_full_corrected.json ground_truth/BIT/BIT_academic_plan_no_coop.json `
  --pair data/output/final/merged_dsba_coop_full_corrected.json ground_truth/DSBA/DSBA_academic_plan_coop.json `
  --pair data/output/final/merged_dsba_no_coop_full_corrected.json ground_truth/DSBA/DSBA_academic_plan_no_coop.json `
  --pair data/output/final/merged_gened_gened_edition-gened-2564_full_corrected.json ground_truth/general_education_ground_truth.json `
  --pair data/output/final/merged_it_coop_full_corrected.json ground_truth/IT/IT_academic_plan_coop.json `
  --pair data/output/final/merged_it_no_coop_full_corrected.json ground_truth/IT/IT_academic_plan_no_coop.json
```

ผลลัพธ์เขียนลง:

```text
reports/evaluation/
├── evaluation_summary.csv
├── field_metrics.csv
├── evaluation_errors.csv
└── evaluation.json
```

- `evaluation_summary.csv` — record coverage
- `field_metrics.csv` — CER/WER และ accuracy ต่อ field
- `evaluation_errors.csv` — รายการค่าที่ไม่ตรง
- `evaluation.json` — รายละเอียดเต็ม

## 4. Historical reference

```text
reports/evaluation_reference/
```

เป็น snapshot เก่าสำหรับเปรียบเทียบเท่านั้น ไม่ใช่ current result และไม่ควรถูก evaluator ปกติเขียนทับ

## 5. Runtime benchmark

```text
reports/runtime_benchmark.md
```

วัด runtime QA latency แยก local deterministic work ออกจาก external model latency

รันด้วย:

```powershell
.\.venv\Scripts\python.exe scripts/benchmark_runtime.py --runs 5
```

snapshot ล่าสุดแสดงว่า parsing/SQLite เร็วมากเมื่อเทียบกับ API path ที่ต้องเรียก external model

## 6. Authority

reports และ Ground Truth เป็น **evaluation artifacts** ไม่ใช่ production factual authority

production flow คือ:

```text
canonical curriculum / policy data
→ runtime SQLite
→ grounded QA
```

ประวัติการแก้ Ground Truth ดูที่ `ground_truth/GT_FIXED.md`
