# Reports — evaluation และหลักฐานการทดสอบ

ไฟล์ใน reports/ มีทั้งผล evaluation ที่สร้างซ้ำได้และรายงาน snapshot. อ่าน ../README.md, ../src/pipeline/README.md และ ../rag/README.md สำหรับ architecture ปัจจุบัน; อย่าใช้ตัวเลขจาก historical snapshots แทนผลที่สร้างจาก raw finals ปัจจุบัน.

## Evaluation ปัจจุบัน

Evaluation วัดผลของ reviewed prediction ใน data/output/final/*_final.json เทียบกับ accepted teacher Ground Truth **ก่อน** canonicalization. ผลอยู่ที่ reports/evaluation_precanonical/ และไม่ถูกเขียนทับด้วย GT runtime override. การมี GT ในขั้น canonicalization ภายหลังไม่ใช่ evaluation leakage เพราะ prediction และรายงานถูกสร้างก่อน override.

ประเมินเฉพาะ 8 scopes ที่มี accepted Ground Truth ของ edition เดียวกัน:

- AIT 2566
- BIT 2565 / coop และ no_coop
- DSBA 2565 / coop และ no_coop
- GENED 2564
- IT 2565 / coop และ no_coop

ไม่ประเมิน BIT 2560, DSBA 2560, IT 2560 หรือ GENED 2557 เพราะไม่มี accepted Ground Truth ของ edition เดียวกัน. Legacy runtime ใช้ source-verified corrections ตาม artifact ที่กำหนด โดยไม่ใช้ current GT.

รายงานใน reports/evaluation_precanonical/:

- evaluation_summary.csv — coverage และจำนวน record
- field_metrics.csv — metrics แยก field
- evaluation_errors.csv — รายการค่าที่ไม่ตรง
- evaluation.json — รายละเอียดผลทั้งหมด

## Rebuild จาก reviewed finals

เมื่อมี raw reviewed finals แล้ว สามารถประเมินและ rebuild runtime ได้โดยไม่รัน OCR หรือ Gemini ซ้ำ. จาก repository root ใช้คำสั่งตามลำดับนี้:

    .\.venv\Scripts\python.exe -m src.pipeline.tools.evaluation.evaluate --reports-dir reports/evaluation_precanonical
    .\.venv\Scripts\python.exe -m src.pipeline.tools.canonicalize_runtime
    .\.venv\Scripts\python.exe -m src.pipeline.tools.preflight_runtime
    .\.venv\Scripts\python.exe -m rag.build_index

Flow เมื่อ raw finals ต้องสร้างใหม่คือ Source/OCR → Extract/Merge → Gemini correction → reviewed finals → evaluation → canonicalization → preflight → build. Canonical curriculum artifacts อยู่ใน data/output/canonical/*_final.json. rag.build_index ใช้ไฟล์เหล่านี้เป็น curriculum source; institution_policy.json และ program_requirements.json ยังคงอ่านจาก data/output/final/ เป็น supplemental sources.

Teacher GT ใช้เป็น runtime authority หลัง evaluation เฉพาะ structured academic-plan fields ที่มีใน accepted GT: code, name_th, name_en, credits, year, semester, category, type, prerequisite, flexible_year_semester และ note. Descriptions และ source provenance ยังคงมาจาก reviewed finals. Rules GT/evaluation files ไม่ใช่ runtime truth; runtime rules มาจาก source-verified rules pipeline/corrections.

## Historical reports และ benchmark

reports/evaluation/ เป็น report path เดิมและไม่ใช่ output ของคำสั่งปัจจุบัน; evaluator ปัจจุบันเขียนไปที่ reports/evaluation_precanonical/. เก็บ historical metrics ตามสภาพเดิม ไม่ควรเปลี่ยนย้อนหลังเพื่อให้ตรงกับ code ปัจจุบัน.

- final_automated_hardening.md — snapshot วันที่ 2026-10-08; findings ในนั้นไม่ใช่ TODO ปัจจุบันโดยอัตโนมัติ
- runtime_benchmark.md — latency snapshot ณ วันที่ benchmark

รัน benchmark ใหม่ด้วย:

    .\.venv\Scripts\python.exe scripts/benchmark_runtime.py --runs 5

ประวัติการแก้ Ground Truth ดูที่ ../ground_truth/GT_FIXED.md.
