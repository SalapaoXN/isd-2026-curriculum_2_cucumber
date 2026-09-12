# isd-2026-curriculum_2_cucumber
We do OCR curriculum and some LLM with model name CUCUMBER

Project : P2 LLM ถาม-ตอบหลักสูตร

Member:
1. 67070049 Nattachai Kaewchum >> Discord: GoodDee
2. 67070063 Thanachin Chukiatchai >> Discord: วันลพ มีงบมาก
3. 67070103 Pongsakorn Panyacom >> Discord: เบบี๋คือดวงใจ

## Quick Commands

### Install Requirements

```bash
python -m pip install -r requirements.txt -r requirements-rag.txt
```

### OCR and Extraction

```bash
python -m src.run_pipeline -p 26-32 -i inputs/dsba --program DSBA --plan no_coop
python extract.py outputs/ocr/dsba --output-dir outputs/extracted --prefix dsba -p 26-32 --program DSBA --plan no_coop
```

### Merge / Consolidation

```bash
python merge_consecutive.py --prefix dsba --plan no_coop -p 26-32,317-344 -d 317-344
```

### Tests

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

### Build Unified Curriculum Database

```bash
python -m rag.build_index
```

### Run Hybrid Demo

```bash
python -m rag.hybrid_demo "มีวิชาไหนเกี่ยวกับฐานข้อมูลบ้าง"
```

## RAG

```text
consolidated JSON
  -> cucumber_outputs/runtime/curriculum.db
     - relational curriculum tables and SQL views
     - persisted retrieval chunks and provenance metadata
     - sqlite-vec 384-d embeddings
  -> retrieval selects SQL, vector similarity, or both
  -> grounded final answer
```

The unified database is persistent at `cucumber_outputs/runtime/curriculum.db`. Building it loads all canonical consolidated files and embeds curriculum chunks once; later queries reuse the database and embed only the user question when vector matching is used. A source JSON fingerprint or embedding-model change triggers a rebuild. Stored chunks retain program, plan, provenance, and `source_page` metadata when available.

## Environment

Create a local `.env` file:

```dotenv
GEMINI_API_KEY=...
HF_TOKEN=...
```

`.env` is local configuration and must not be committed. `HF_TOKEN` is optional; anonymous Hugging Face access remains available.


## Overview

CUCUMBER extracts structured curriculum data from Thai/English curriculum images and prepares the result for evaluation and later LLM/RAG use.

```text
Image
  -> EasyOCR (Thai + English)
  -> structured course extraction
  -> optional English name enrichment
  -> merge/consolidation
  -> evaluation
```

The canonical OCR/extraction output is authoritative by default. The optional English second pass may improve only `name_en` when a safe candidate is accepted.

## Setup

Use Python `3.10–3.13`; Python `3.11` is the preferred baseline.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt -r requirements-rag.txt
```

### macOS / Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-rag.txt
```

EasyOCR uses Thai and English (`['th', 'en']`). Missing models may be downloaded on first use and reused from the local EasyOCR cache.

Use `--no-gpu` for the reproducible CPU baseline. GPU execution is optional and follows the existing EasyOCR/PyTorch environment.

## Input and Output

Production page images use:

```text
<input-group>_page_<NNN>.<ext>
```

Example:

```text
inputs/dsba/dsba_page_026.jpg
```

Supported image extensions: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`.

Typical per-page outputs:

```text
outputs/ocr/dsba/dsba_page_026_ocr.txt
outputs/ocr/dsba/dsba_page_026_ocr.json
outputs/extracted/dsba/dsba_page_026_ocr_extracted.json
```

Consolidated curriculum files are written under `outputs/consolidated/`.

## OCR

Run EasyOCR directly with `cli.py`:

```bash
# Single image
python cli.py inputs/dsba/dsba_page_026.jpg

# Directory batch
python cli.py inputs/dsba/ -o outputs/ocr

# CPU mode
python cli.py inputs/dsba/ -o outputs/ocr --no-gpu
```

For the automated page runner:

```bash
python -m src.run_pipeline -p 26-32 -i inputs/dsba --program DSBA --plan no_coop
```

`src.run_pipeline` supports `-p/--pages`, `-i/--input-dir`, `-o/--output-dir`, `--program`, `--plan`, `--no-gpu`, and the opt-in `--english-second-pass`.

## Extraction

`extract.py` reads OCR TXT/JSON files and writes `*_ocr_extracted.json`. When matching TXT and JSON files share a stem, JSON is preferred and processed once.

### DSBA

```bash
# No co-op plan
python extract.py outputs/ocr/dsba --output-dir outputs/extracted --prefix dsba -p 26-32 --program DSBA --plan no_coop

# Co-op plan
python extract.py outputs/ocr/dsba --output-dir outputs/extracted --prefix dsba -p 33-39 --program DSBA --plan coop

# Course descriptions
python extract.py outputs/ocr/dsba --output-dir outputs/extracted --prefix dsba -p 317-344 --program DSBA --plan coop
```

### IT

```bash
# No co-op plan
python extract.py outputs/ocr/it --output-dir outputs/extracted --prefix it -p 32-38 --program IT --plan no_coop

# Co-op plan
python extract.py outputs/ocr/it --output-dir outputs/extracted --prefix it -p 39-45 --program IT --plan coop

# Course descriptions
python extract.py outputs/ocr/it --output-dir outputs/extracted --prefix it -p 328-371 --program IT --plan coop
```

### AIT

AIT has no `coop` / `no_coop` plan variant.

```bash
python extract.py outputs/ocr/ait --output-dir outputs/extracted --prefix ait --program AIT
```

### GENED

```bash
python extract.py outputs/ocr/gened --output-dir outputs/extracted --prefix gened -p 16-30,44-117 --program GENED --plan gened
```

### BIT

```bash
# No co-op plan
python extract.py outputs/ocr/bit --output-dir outputs/extracted --prefix bit -p 26-30 --program BIT --plan no_coop

# Co-op plan
python extract.py outputs/ocr/bit --output-dir outputs/extracted --prefix bit -p 31-35 --program BIT --plan coop

# Course descriptions
python extract.py outputs/ocr/bit --output-dir outputs/extracted --prefix bit -p 238-257 --program BIT --plan coop
```

## Merge / Consolidation

`merge_consecutive.py` combines extracted plan and description pages into consolidated curriculum JSON. `-p/--pages` selects source pages and `-d/--desc-pages` identifies description pages.

Repeated course placements are preserved. A single unambiguous description may enrich repeated placements of the same course code; ambiguous multiple descriptions are not guessed by occurrence order.

### DSBA

```bash
python merge_consecutive.py --prefix dsba --plan no_coop -p 26-32,317-344 -d 317-344
python merge_consecutive.py --prefix dsba --plan coop -p 33-39,317-344 -d 317-344
```

### IT

```bash
python merge_consecutive.py --prefix it --plan no_coop -p 32-38,328-371 -d 328-371
python merge_consecutive.py --prefix it --plan coop -p 39-45,328-371 -d 328-371
```

### AIT

```bash
python merge_consecutive.py --prefix ait -d 287-302
```

### GENED

```bash
python merge_consecutive.py --prefix gened --plan gened -p 16-30,44-117 -d 44-117
```

### BIT

```bash
python merge_consecutive.py --prefix bit --plan no_coop -p 26-30,238-257 -d 238-257
python merge_consecutive.py --prefix bit --plan coop -p 31-35,238-257 -d 238-257
```

## Evaluation

The evaluator reports text quality with CER/WER and extraction coverage with matched, missing, extra, Precision, Recall, and F1.

Equivalent repeated prediction placements may be collapsed only in the canonical evaluation view when the GT contains one logical course and the repeated records agree on evaluated course fields. The original consolidated artifact remains unchanged.

### DSBA

```bash
python evaluate.py outputs/consolidated/dsba/coop/full/merged_dsba_coop_full.json --gt ground_truth/DSBA/DSBA_academic_plan_coop.json
python evaluate.py outputs/consolidated/dsba/no_coop/full/merged_dsba_no_coop_full.json --gt ground_truth/DSBA/DSBA_academic_plan_no_coop.json
```

### IT

```bash
python evaluate.py outputs/consolidated/it/coop/full/merged_it_coop_full.json --gt ground_truth/IT/IT_academic_plan_coop.json
python evaluate.py outputs/consolidated/it/no_coop/full/merged_it_no_coop_full.json --gt ground_truth/IT/IT_academic_plan_no_coop.json
```

### AIT

```bash
python evaluate.py outputs/consolidated/ait/full/merged_ait_no_plan_full.json --gt ground_truth/AIT/AIT_academic_plan.json
```

### GENED

```bash
python evaluate.py outputs/consolidated/gened/full/merged_gened_gened_full.json --gt ground_truth/general_education_ground_truth.json
```

### BIT

Add BIT evaluation commands after the accepted BIT ground-truth paths are finalized.

### Batch Evaluation

```bash
python evaluate.py `
  --pair outputs/consolidated/dsba/coop/full/merged_dsba_coop_full.json ground_truth/DSBA/DSBA_academic_plan_coop.json `
  --pair outputs/consolidated/dsba/no_coop/full/merged_dsba_no_coop_full.json ground_truth/DSBA/DSBA_academic_plan_no_coop.json `
  --pair outputs/consolidated/it/coop/full/merged_it_coop_full.json ground_truth/IT/IT_academic_plan_coop.json `
  --pair outputs/consolidated/it/no_coop/full/merged_it_no_coop_full.json ground_truth/IT/IT_academic_plan_no_coop.json `
  --pair outputs/consolidated/ait/full/merged_ait_no_plan_full.json ground_truth/AIT/AIT_academic_plan.json `
  --pair outputs/consolidated/gened/full/merged_gened_gened_full.json ground_truth/general_education_ground_truth.json `
  --pair outputs/consolidated/bit/coop/full/merged_bit_coop_full.json ground_truth/BIT/BIT_academic_plan_coop.json `
  --pair outputs/consolidated/bit/no_coop/full/merged_bit_no_coop_full.json ground_truth/BIT/BIT_academic_plan_no_coop.json
```

Evaluation reports are written to `reports/evaluation/`:

- `evaluation.json`
- `evaluation_summary.csv`
- `field_metrics.csv`
- `evaluation_errors.csv`

Page-level evaluation is reported only when GT contains authoritative source/page provenance. It is not inferred from project-created mappings.

## Optional English Name Enrichment

Enable the auxiliary English-only OCR pass with:

```bash
python -m src.run_pipeline -p 26 -i inputs/dsba --program DSBA --plan coop --english-second-pass
```

The pass is opt-in. It may update only `name_en`; Thai names, credits, prerequisites, categories, and other canonical fields are not replaced. Unsafe or ambiguous candidates fall back to canonical OCR.

## Source Provenance

Extracted records include `source_provenance`, for example:

```json
{
  "program": "DSBA",
  "source_filename": "dsba_page_026.png",
  "source_page": 26,
  "document_category": "plan"
}
```

Provenance is derived from source/input context, not ground truth, and is preserved through merge/consolidation.

## LLM Spell Corrector

```bash
python llm_spell_corrector.py ./outputs/consolidated/dsba/coop/full/merged_dsba_coop_full.json
```

## Testing

Run the regression suite from the repository root:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

Useful focused suites:

```bash
python -m unittest discover -s tests -p "test_gened_cleanup.py" -v
python -m unittest discover -s tests -p "test_cli_semantics.py" -v
python -m unittest discover -s tests -p "test_evaluate.py" -v
python -m unittest discover -s tests -p "test_evaluation_reports.py" -v
```

## Notes

- JSON OCR input is preferred when matching TXT/JSON files share a stem.
- DSBA, IT, and BIT require explicit `coop` or `no_coop` plans.
- GENED uses `--plan gened`.
- AIT has no plan variant and must omit `--plan`.
- Description pages are shared between `coop` and `no_coop` variants where applicable.
- Missing or ambiguous data is preserved conservatively rather than guessed.
