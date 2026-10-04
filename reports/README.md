# Reports

โฟลเดอร์นี้เก็บผลการประเมินคุณภาพข้อมูลและ benchmark ของ CUCUMBER

## 1. Curriculum evaluation

Source:

```text
data/output/final/*_corrected.json
        +
ground_truth/
        ↓
src.pipeline.tools.evaluation.evaluate
        ↓
reports/evaluation/
```

รันใหม่ด้วย:

```powershell
python -m src.pipeline.tools.evaluation.evaluate
```

ผลที่ได้:

```text
reports/evaluation/
├── evaluation_summary.csv
├── field_metrics.csv
├── evaluation_errors.csv
└── evaluation.json
```

### `evaluation_summary.csv`

ใช้ดู record coverage ต่อ program/plan:

- `gt_total`
- `pred_total`
- `tp`, `fn`, `fp`
- precision / recall / F1

### `field_metrics.csv`

ใช้วัดคุณภาพระดับ field เช่นชื่อวิชาและ prerequisite:

- CER
- Character Accuracy
- WER
- Word Accuracy

### `evaluation_errors.csv`

ใช้ดู record/field ที่ไม่ตรง Ground Truth เพื่อ debug

### `evaluation.json`

ผลแบบเต็มสำหรับตรวจรายละเอียดหรือวิเคราะห์ต่อด้วย script

## 2. Current-report rule

**ก่อนใช้ตัวเลขใน `reports/evaluation/` เป็นผลสุดท้ายของโปรเจกต์ ต้อง rerun evaluator จาก canonical data ปัจจุบันอีกครั้ง**

เหตุผลคือ runtime/canonical data มีการพัฒนาต่อจาก evaluation snapshot เดิม และไม่ควรอ้างตัวเลขเก่าว่าเป็นผลปัจจุบันโดยไม่ regenerate

หลัง rerun ให้ตรวจอย่างน้อย:

1. ทุก scope ที่ตั้งใจประเมินมีอยู่ครบ
2. coverage metrics
3. CER/WER
4. error rows
5. ไม่มีการใช้ Ground Truth เป็น production factual authority

## 3. Evaluation reference

```text
reports/evaluation_reference/
```

เป็น historical/reference snapshot สำหรับเทียบพฤติกรรม ไม่ใช่ current evaluation และไม่ควรถูกเขียนทับโดย evaluator ปกติ

## 4. Runtime benchmark

```text
reports/runtime_benchmark.md
```

เป็น benchmark ของ runtime QA latency โดยแยก local deterministic work ออกจาก external model latency

script:

```powershell
.\.venv\Scripts\python.exe scripts/benchmark_runtime.py --runs 5
```

snapshot ที่บันทึกไว้ชี้ว่า parsing/SQLite เร็วมากเมื่อเทียบกับ API path ที่ต้องเรียก external model

## 5. Authority

reports เป็น **ผลการวัด** ไม่ใช่ source of truth

production authority ยังคงเป็น:

```text
canonical curriculum/policy data
→ runtime SQLite
→ grounded QA
```

Ground Truth ใช้สำหรับ evaluation/test เท่านั้น
