# isd-2026-curriculum_2_cucumber
We do OCR curriculum and some LLM with model name CUCUMBER

Project : P2 LLM ถาม-ตอบหลักสูตร

Member:
1. 67070049 Nattachai Kaewchum >> Discord: GoodDee
2. 67070063 Thanachin Chukiatchai >> Discord: วันลพ มีงบมาก
3. 67070103 Pongsakorn Panyacom >> Discord: เบบี๋คือดวงใจ


```bash
python -m ocr_system.cli ocr data/input/sample.pdf --engine ensemble

python -m ocr_system.cli evaluate data/ground_truth/DSBA/DSBA_academic_plan_coop.json outputs/img27_ocr_extracted.json --mode ocr-extracted

python -m ocr_system.cli evaluate data/ground_truth/DSBA/DSBA_academic_plan_coop.json outputs/img27_ocr_extracted.json
```
