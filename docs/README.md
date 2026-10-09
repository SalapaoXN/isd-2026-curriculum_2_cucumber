# CUCUMBER Documentation

เอกสารใน repo แบ่งเป็น 3 กลุ่มเพื่อไม่ให้สถานะปัจจุบันปนกับรายงานย้อนหลัง

## Active documentation

อ่านชุดนี้เมื่อต้องการเข้าใจระบบปัจจุบัน

- `../README.md` — ภาพรวมระบบ, setup, run, API และข้อจำกัด
- `semantic-qa-vnext.md` — architecture ของ Semantic QA production path
- `../rag/README.md` — implementation map ของ QA/RAG และ authority boundaries
- `academic_rules.md` — policy / institution rules subsystem
- `../src/pipeline/README.md` — Source/OCR → reviewed finals → evaluation → canonical runtime → DB
- `../reports/README.md` — evaluation/data-quality artifacts และวิธีอ่าน report

## Evaluation and audit artifacts

เอกสารกลุ่มนี้เป็นหลักฐานการทดสอบหรือ snapshot ณ เวลาที่รัน ไม่ควรถูกตีความว่าเป็นสถานะล่าสุดโดยอัตโนมัติ

- `../eval/results/*.md` — semantic evaluation snapshots จากรอบก่อนหน้า
- `../reports/final_automated_hardening.md` — automated hardening snapshot วันที่ 2026-10-08
- `../reports/runtime_benchmark.md` — latency snapshot จากวันที่รัน benchmark
- `../ground_truth/GT_FIXED.md` — audit log ของการแก้ Ground Truth

หากตัวเลขใน historical report ขัดกับ code/test ปัจจุบัน ให้ใช้ repository state และ active documentation เป็นตัวอธิบาย implementation ปัจจุบัน ส่วน historical report ให้เก็บไว้เป็นหลักฐานของรอบนั้น

## Frozen submission

`../submission/` เป็น historical submission ที่เคยส่งแล้ว **ห้าม rewrite เพื่อให้ตรงกับ runtime ปัจจุบัน** เว้นแต่มีคำสั่งชัดเจนให้สร้าง submission รุ่นใหม่

## Local agent guidance

- `../AGENTS.md`
- `../.agent-docs/`

เอกสารเหล่านี้ใช้กำกับ coding agent ใน workspace นี้และอาจถูก exclude จาก Git ตาม local policy

## Non-project skill documentation

`../.agents/skills/` เป็น skill/reference material ของ agent tooling ไม่ใช่เอกสารสถานะของ CUCUMBER จึงไม่ควรแก้เพียงเพราะ project architecture เปลี่ยน
